import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
import random
import logging


from model.CGCN import CGCN

from torch.amp import autocast, GradScaler
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

import optuna
from optuna.trial import TrialState
from optuna.samplers import TPESampler

from dataset import FDIADataset
from dataset_generators.functions import preparePyGDataset, selectDevice

# TODO: delete at the final stage
import sys

# -- Define the params and hyperparams for tuning --
config = {
              "dataset_root": "./dataset",
    
              "num_epochs": 100,
              "batch_size": 4, # 256 with gradient accumulation
              "lr": 1e-3, 
              "weight_decay": 1e-5,
              "n_lrWarmup_steps": 500, # 1 batch = 1 step
              
              "total_num_of_samples": 36000,
              "Ad_start": 0,
              "Ad_end": 9000,
              "As_start": 9000,
              "As_end": 18000,
              "Ad_train": 6000,
              "As_train": 6000,
              "Ad_val": 1500,
              "As_val": 1500,
              "Ad_test": 1500,
              "As_test": 1500,
              "norm_train": 12000,
              "norm_val": 3000,
              "norm_test": 3000,
              
              "num_nodes": 2848,
              "u": 32, # hidden channels
              "Ks": 5,
              "dropout": 0.2,
              
              "transformer_layers": 6,
              "transformer_heads": 8
          }

# -- Prepare the dataset --
train_loader, valid_loader, _, in_feats = preparePyGDataset(config, FDIADataset)

# - Select device -
device = selectDevice()

# - Define the criterion --
#criterion = nn.CrossEntropyLoss() # maps logits [N,2] + labels [N] → scalar
criterion = nn.BCEWithLogitsLoss()

# - Scaler and Autocast -
# (seems to have negative impact on the training; hence, turned off)
#use_cuda = (device.type == 'cuda')
#scaler = GradScaler(enabled=use_cuda)
use_cuda = False
scaler = GradScaler(enabled=False)


# -- Custom train and validation --
def train_epoch(model, optimizer, epoch, graph_loss_importance):
    # Turn on the training mode to include features like GraphNorm normalization
    model.train()
    
    # Declare default value for loss and # of batches(steps) for grad accumulation
    total_loss = 0
    accum_steps = 64 # 64 mini batches filled with 4 samples = 256 samples per optimizer.step()
    
    # Reset accumulated grads
    optimizer.zero_grad()
    
    for minibatch_id, minibatch in enumerate(train_loader):
        minibatch = minibatch.to(device)
        
        with autocast(device_type=device.type, enabled=use_cuda):
            # Get target for the batch
            target_nodes = minibatch.y
            target_graph = minibatch.y_graph
            
            # Get model's raw output (logits)
            logits_nodes, logits_graph = model(minibatch.x, 
                           minibatch.edge_index, 
                           weights=minibatch.edge_attr, 
                           batch=minibatch.batch)
            
            # Compute loss and save it
            loss = criterion(logits_nodes.view(-1), target_nodes)
            loss += (criterion(logits_graph, target_graph)*graph_loss_importance)
            total_loss += (loss.item()/accum_steps)

        # Scale down the loss so that the grads are averaged over accum_steps
        scaler.scale(loss / accum_steps).backward()
        
        # Gradient Accumulation to simulate 2^8 samples per batch while maintaining low RAM usage
        if (minibatch_id+1) % accum_steps == 0:
            print(f"Epoch#{epoch}, Batch#{(minibatch_id // accum_steps):04d} | Current loss: {loss:.4f}")
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
    
    # Extra gradient step to make sure there are no leftovers
    print(f"Epoch#{epoch}, Last batch")
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
        
    return total_loss / len(train_loader)


def validate(model, valid_loader):
    # Turn on the evaluation mode to exclude features like dropout regularizatiion
    model.eval()
    
    # Keep track of loss for early stopping
    total_loss = 0
    
    # Declare arrays where the predictions and corresponding targets(labels) will be stored
    all_preds = []
    all_targets = []

    # Make sure grads of the model won't be affected
    with torch.no_grad():
        for batch in valid_loader:
            batch = batch.to(device)

            # Get model's raw output(logits)
            _, logits_graph = model(
                batch.x,
                batch.edge_index,
                batch.edge_attr,
                batch.batch
            )
            
            # Get target for the batch
            target_graph = batch.y_graph
            
            # Compute loss
            loss = criterion(logits_graph, target_graph)
            total_loss += loss.item()
            
            # Apply activation function on the logits to get probability
            prob = torch.sigmoid(logits_graph)
            
            # Convert probability into a classification
            pred = 1 if prob > 0.5 else 0
             
            # Append current outputs
            all_preds.append(pred)
            all_targets.append(target_graph.item())
        

        # Concatenate across batches
        all_preds   = np.asarray(all_preds)
        all_targets = np.asarray(all_targets)
    
        # Compute FP, TN
        FP = np.logical_and(all_preds == 1, all_targets == 0).sum()
        TN = np.logical_and(all_preds == 0, all_targets == 0).sum()
    
        # Metrics
        precision = precision_score(all_targets, all_preds, zero_division=0)
        recall    = recall_score(all_targets, all_preds, zero_division=0)
        accuracy  = (all_preds == all_targets).mean() * 100
        f1        = f1_score(all_targets, all_preds, zero_division=0)
        FA        = FP / (FP + TN) if (FP + TN) > 0 else 0.0
        
        return precision, recall, f1, accuracy, FA, total_loss/len(valid_loader)


# -- Optuna functions --
def define_model(trial, dropout):
    model = CGCN(
        in_channels=in_feats,
        u=config["u"],
        Ks=config["Ks"],
        dropout=dropout,
        num_nodes = config["num_nodes"],
        trial = trial
    )
    
    return model

def objective(trial):
    print(f"---- Current trial: {trial.number} ----")
    
    # Generate dropout
    dropout= trial.suggest_float("dropout", 0.1, 0.5)
    
    # Generate the model.
    model = define_model(trial, dropout).to(device)
    
    # Generate the lr
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    
    # Generate graph_loss importance
    graph_loss_importance = trial.suggest_float("graph_loss_importance", 0.1, 2.0)

    # Generate the optimizer.
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=config["weight_decay"])

    # Training of the model.
    for epoch in range(config["num_epochs"]):
        
        train_loss = train_epoch(model, optimizer, epoch, graph_loss_importance)
        prec, rec, f1, accuracy, FA, avg_loss = validate(model, valid_loader)
        
        print(f"\nEpoch {epoch}: Average Train Loss={train_loss:.4f},")
        print(f"Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f},")
        print(f"FA={FA:.4f}, Accuracy={accuracy:.4f}, Validation Loss: {avg_loss:.4f}")
        print("----------------------------------------------------\n\n")

        trial.report(accuracy, epoch)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return accuracy


# # # # # # # # # # # # # # # # # #
# ---- Hyperparameter tuning ---- #
# # # # # # # # # # # # # # # # # #

sampler = TPESampler(seed=123)


optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
study_name = "study2_06_10_25"  # Unique identifier of the study.
storage_name = "sqlite:///{}.db".format(study_name)
study = optuna.create_study(
                            direction="maximize", 
                            sampler=sampler,
                            pruner=optuna.pruners.PatientPruner(wrapped_pruner=None, patience=5, min_delta=0.0),
                            study_name=study_name,
                            storage=storage_name
                            )

study.optimize(objective, n_trials=100, timeout=None)

pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

print("Study statistics: ")
print("  Number of finished trials: ", len(study.trials))
print("  Number of pruned trials: ", len(pruned_trials))
print("  Number of complete trials: ", len(complete_trials))

print("Best trial:")
trial = study.best_trial

print("  Value: ", trial.value)

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))