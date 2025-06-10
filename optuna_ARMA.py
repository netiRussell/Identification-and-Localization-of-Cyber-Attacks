import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
import random
import logging


from model.ARMA import GNNArma

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
    
              "num_epochs": 256,
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
              
              "in_channels": 2,
              "hidden_channels": 32,
              "num_stacks": 4,  # K
              "num_layers": 8,
              "dropout": 0.3, 
              "num_nodes": 2848,
              
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
def train_epoch(model, optimizer, epoch, graph_loss_importance, train_loader):
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
            # Get targets for the batch
            target_nodes = minibatch.y
            target_graph = minibatch.y_graph
            
            # Get model's raw output (logits)
            logits_nodes, logits_graph = model(minibatch.x, 
                           minibatch.edge_index, 
                           weights=minibatch.edge_attr,
                           batch=minibatch.batch)
            
            # Compute loss and save it
            loss = criterion(logits_nodes.view(-1), target_nodes)
            loss += (criterion(logits_graph, target_graph.squeeze(0))*graph_loss_importance)
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
    
    # Declare holders for metrics
    FA = 0
    F1 = 0 
    recall = 0
    accuracy = 0
    
    # Make sure grads of the model won't be affected
    with torch.no_grad():
        for batch in valid_loader:
            batch = batch.to(device)

            # Get model's raw output(logits)
            logits_nodes, logits_graph = model(
                batch.x,
                batch.edge_index,
                batch.edge_attr,
                batch.batch
            )
            
            # Change dimensions of the graph output to work with batch=1 case
            logits_graph = logits_graph.squeeze(0)
            logits_nodes = logits_nodes.squeeze(0)
            
            # Get targets for the batch
            target_nodes = batch.y
            target_graph = batch.y_graph.squeeze(0)
            
            # Compute loss
            loss = criterion(logits_nodes, target_nodes)
            loss += criterion(logits_graph, target_graph)
            total_loss += loss.item()
            
            # Apply activation function on the logits to get node-level probability
            # and convert it into a classification
            pred_nodes = (torch.sigmoid(logits_nodes) > 0.5).float()  
            
            # Get graph-level classification:
            #graph_pred = (logits_graph > 0).long()
            
            # Turn tensors into NumPy arrays
            pred_nodes = pred_nodes.cpu().numpy()
            target_nodes = target_nodes.cpu().numpy()
            
            # Append current outputs
            all_preds.append(pred_nodes)
            all_targets.append(target_nodes)
            
            # Calculate current metrics
            if( target_graph == 0 ):
                # - No attack case -
                # If prediction is 100% correct (all nodes = 0 since no attack)
                if(np.sum(pred_nodes) == 0):
                    FA += 0
                    F1 += 1
                    recall += 1
                    accuracy += 1
                    
                # If at least 1 node is a mismatch (has a value of 1)    
                else:
                   FA += 1
            else:
                # - Attack took place case -
                # Compute FP, TN, and FA
                FP = np.logical_and(pred_nodes == 1, target_nodes == 0).sum()
                TN = np.logical_and(pred_nodes == 0, target_nodes == 0).sum()
                FA += FP / (FP + TN)
                
                # Compute Recall and F1
                F1 += f1_score(target_nodes, pred_nodes, zero_division=0)
                recall += recall_score(target_nodes, pred_nodes, zero_division=0) # DR
                
                # Compute a strict match
                if( np.array_equal(pred_nodes, target_nodes) ):
                    accuracy += 1
                
        

        # Concatenate across batches
        all_preds   = np.asarray(all_preds)
        all_targets = np.asarray(all_targets)
    
        # Metrics
        precision = precision_score(all_targets, all_preds, zero_division=0, average="micro")
        accuracy  = (all_preds == all_targets).mean() * 100
        
        # Number of elements in metrices (used to get mean)
        n_elem_metrices = len(valid_loader)
        return precision, recall/n_elem_metrices, F1/n_elem_metrices, accuracy/n_elem_metrices, FA/n_elem_metrices, total_loss/n_elem_metrices


# -- Optuna functions --
def define_model(trial, dropout, hidden_channels, num_stacks, num_layers):
    model = GNNArma(
        in_channels=config["in_channels"], 
        hidden_channels=hidden_channels,
        num_stacks=num_stacks,
        num_layers=num_layers, 
        dropout=dropout,
        num_nodes=config["num_nodes"]
    )
    
    return model

def objective(trial):
    print(f"---- Current trial: {trial.number} ----")
    
    # Generate dropout
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    
    # Generate hidden features
    hidden_channels = trial.suggest_categorical("hidden_channels", [32, 64, 128])
    
    # Generate number of stacks (K)
    num_stacks = trial.suggest_categorical("num_stacks", [3, 4, 5, 6])
    
    # Generate number of layers (T)
    num_layers = trial.suggest_categorical("num_layers", [4, 5, 6, 7, 8])
    
    # Generate the model
    model = define_model(trial, dropout, hidden_channels, num_stacks, num_layers).to(device)
    
    # Generate the lr
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    
    # Generate graph_loss importance
    graph_loss_importance = trial.suggest_float("graph_loss_importance", 0.1, 2.0)

    # Generate the optimizer.
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=config["weight_decay"])

    # Training of the model.
    for epoch in range(config["num_epochs"]):
        
        train_loss = train_epoch(model, optimizer, epoch, graph_loss_importance, train_loader)
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
study_name = "studyARMA1_06_10_25"  # Unique identifier of the study.
storage_name = "sqlite:///{}.db".format(study_name)
study = optuna.create_study(
                            direction="maximize", 
                            sampler=sampler,
                            pruner=optuna.pruners.PatientPruner(wrapped_pruner=None, patience=10, min_delta=0.0),
                            study_name=study_name,
                            storage=storage_name
                            )

study.optimize(objective, n_trials=150, timeout=None)

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