import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
import random


from ..model.CGCN import CGCN

from torch.amp import autocast, GradScaler
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

import optuna
from optuna.trial import TrialState

from ..dataset import FDIADataset

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
              
              "num_nodes": 2848,
              "u": 32, # hidden channels
              "Ks": 5,
              "dropout": 0.2,
              
              "transformer_layers": 6,
              "transformer_heads": 8
          }

# --- Prepare the dataset ---

# Definition of the lists containing indices of Ad ans As samples
Ad_indices = list(range(config["Ad_start"], config["Ad_end"]))
As_indices = list(range(config["As_start"], config["As_end"]))

# Definition of the list containing indices of not attacked(normal) samples
normal_indices = list(range(int(config["total_num_of_samples"]/2), config["total_num_of_samples"]))

# 4/6 1/6 1/6 split
train_len = int(4/6 * config["total_num_of_samples"])
val_len   = int(1/6 * config["total_num_of_samples"]) + train_len
#test_len   = int(1/6 * config["total_num_of_samples"]) + train_len + val_len

# Get training indices: 6k of Ad + 6k of As + 12k of norm = 4/6 of 36k samples or 24k samples total
train_indices = Ad_indices[:config["Ad_train"]] + As_indices[:config["As_train"]] + normal_indices[:config["norm_train"]]

# Get validation indices: 1.5k of Ad + 1.5k of As + 3k of norm = 1/6 of 36k samples or 6k samples total
val_indices = (Ad_indices[config["Ad_train"]:config["Ad_train"]+config["Ad_val"]] + 
               As_indices[config["As_train"]:config["As_train"]+config["As_val"]] + 
               normal_indices[config["norm_train"]:config["norm_train"]+config["norm_val"]])

# Mix attacks and normal indices within the subsets
random.shuffle(train_indices)
random.shuffle(val_indices)

# Get datasets
train_dataset = FDIADataset(train_indices, config["dataset_root"])
val_dataset = FDIADataset(val_indices, config["dataset_root"])

# Define loaders for each data subset to sequentially load batches
train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
valid_loader   = DataLoader(val_dataset,   batch_size=1, shuffle=False)

# Get input features (Size of X in the first dimension)
# Supposed to be 4 (P, Q, V, angle) or 2 (P, Q)
in_feats = train_dataset[0].x.size(1)

# -- Select device --
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True
print(f"Device selected: {device}")


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
def train_epoch(model, optimizer, epoch):
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
            target = minibatch.y_graph
            
            # Get model's raw output (logits)
            logits = model(minibatch.x, 
                           minibatch.edge_index, 
                           weights=minibatch.edge_attr, 
                           batch=minibatch.batch)
            
            # Compute loss and save it 
            loss   = criterion(logits, target)
            total_loss += loss.item()

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


def validate(model, val_loader):
    # Turn on the evaluation mode to exclude features like dropout regularizatiion
    model.eval()
    
    # Keep track of loss for early stopping
    total_loss = 0
    
    # Declare arrays where the predictions and corresponding targets(labels) will be stored
    all_preds = []
    all_targets = []

    # Make sure grads of the model won't be affected
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)

            # Get model's raw output(logits)
            logits = model(
                batch.x,
                batch.edge_index,
                batch.edge_attr,
                batch.batch
            )
            
            # Get target for the batch
            target = batch.y_graph
            
            # Compute loss
            loss = criterion(logits, target)
            total_loss += loss.item()
            
            # Apply activation function on the logits to get probability
            prob = torch.sigmoid(logits)
            
            # Convert probability into a classification
            pred = 1 if prob > 0.5 else 0
             
            # Append current outputs
            all_preds.append(pred)
            all_targets.append(target.item())
        

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
        
        return precision, recall, f1, accuracy, FA, total_loss/len(val_loader)

# -- Optuna functions --
def define_model(trial):
    model = CGCN(
        in_channels=in_feats,
        u=config["u"],
        Ks=config["Ks"],
        dropout=config["dropout"],
        num_nodes = config["num_nodes"],
        trial = trial
    )
    
    return model

def objective(trial):
    # Generate the model.
    model = define_model(trial).to(device)
    
    # Generate the lr
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)

    # Generate the optimizer.
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=config["weight_decay"])

    # Training of the model.
    for epoch in range(config["num_epochs"]):
        
        train_loss = train_epoch(model, optimizer, epoch)
        prec, rec, f1, accuracy, FA, avg_loss = validate(model, valid_loader)
        
        print(f"\nEpoch {epoch}: Average Train Loss={train_loss:.4f},")
        print(f"Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f},")
        print(f"FA={FA:.4f}, Accuracy={accuracy}, Validation Loss: {avg_loss:.4f}")
        print("----------------------------------------------------\n\n")

        trial.report(accuracy, epoch)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return accuracy


# # # # # # # # # # # # # # # # # #
# ---- Hyperparameter tuning ---- #
# # # # # # # # # # # # # # # # # #

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=300, timeout=600)

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