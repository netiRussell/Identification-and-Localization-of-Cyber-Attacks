import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
import random
import logging


from model.ARMA import GNNArma

from torch.amp import autocast, GradScaler
import numpy as np
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR

import optuna
from optuna.trial import TrialState
from optuna.samplers import TPESampler
from datetime import datetime

from dataset import FDIADataset
from dataset_generators.functions import preparePyGDataset, selectDevice

# TODO: delete at the final stage
import sys

# -- Define the params and hyperparams for tuning --
config = {
              "dataset_root": "./final_dataset",
    
              "num_epochs": 80,
              "batch_size": 4, # 256 with gradient accumulation
              "lr": 1e-3, 
              "weight_decay": 0,
              "n_lrWarmup_steps": 500, # 1 batch = 1 step
              
              "in_channels": 2,
              "hidden_channels": 32,
              "num_stacks": 4,  # K
              "num_layers": 8,
              "dropout": 0.3, 
              "num_nodes": 300,
          }

# -- Prepare the dataset --
train_loader, valid_loader, _, in_feats = preparePyGDataset(config, FDIADataset)

# - Select device -
device = selectDevice()

# - Scaler and Autocast -
# (seems to have negative impact on the training; hence, turned off)
#use_cuda = (device.type == 'cuda')
#scaler = GradScaler(enabled=use_cuda)
use_cuda = False
scaler = GradScaler(enabled=False)


def lr_lambda(current_step: int):
    if current_step < config["n_lrWarmup_steps"]:
        return float(current_step) / config["n_lrWarmup_steps"]
    return 1.0  # keep base_lr after warm-up

# -- Custom train and validation --
def train_epoch(model, optimizer, epoch, train_loader, criterion, scheduler):
    # TO BE DELETED
    gradSpike = False
    
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
            loss += (criterion(logits_graph, target_graph.squeeze(0)))
            total_loss += (loss.item()/accum_steps)

        # Scale down the loss so that the grads are averaged over accum_steps
        scaler.scale(loss / accum_steps).backward()
        
        # Gradient Accumulation to simulate 2^8 samples per batch while maintaining low RAM usage
        if (minibatch_id+1) % accum_steps == 0:
            # Clip gradients so their total L2 norm ≤ 1
            total_norm = clip_grad_norm_(model.parameters(), max_norm=10.0)
            if(total_norm > 10):
                gradSpike = True
            print(f"Grad norm before clipping: {total_norm:.4f}. Presence of spike: {gradSpike}")
            
            # Applying grads
            print(f"Epoch#{epoch}, Batch#{(minibatch_id // accum_steps):04d} | Current loss: {loss:.4f}\n")
            scaler.step(optimizer)
            scheduler.step()
            scaler.update()
            optimizer.zero_grad() 
    
    # Clip gradients so their total L2 norm ≤ 1
    total_norm = clip_grad_norm_(model.parameters(), max_norm=1.0)
    print(f"Grad norm before clipping: {total_norm:.4f}")
    
    # Extra gradient step to make sure there are no leftovers
    print(f"Epoch#{epoch}, Last batch\n")
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
        
    return total_loss / len(train_loader)


def validate(model, valid_loader, criterion):
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
    temp_F1 = 0
    recall = 0
    precision = 0
    localization_accuracy = 0
    detection_accuracy = 0
    
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
            pred_nodes = (torch.sigmoid(logits_nodes) >= 0.5).float()  
            
            # Get graph-level classification:
            graph_pred = (logits_graph >= 0.5).long()
            
            # Turn tensors into NumPy arrays
            pred_nodes = pred_nodes.cpu().numpy()
            target_nodes = target_nodes.cpu().numpy()
            
            # Append current outputs
            all_preds.append(pred_nodes)
            all_targets.append(target_nodes)
            
            # Calculate current metrics
            # Graph-wise
            if( target_graph == graph_pred):
                detection_accuracy += 1
            
            # Node-wise
            if( target_graph == 0 ):
                # - No attack case -
                
                # If prediction is 100% correct (all nodes = 0 since no attack)
                if(np.sum(pred_nodes) == 0):
                    temp_F1 = 1
                    recall += 1
                    localization_accuracy += 1
                    
                # If at least 1 node is a mismatch (has a value of 1)    
                else:
                    temp_F1 = 0
                    FA += 1
            else:
                # - Attack took place case -
                # Compute predictions
                FP = np.logical_and(pred_nodes == 1, target_nodes == 0).sum()
                FN = np.logical_and(pred_nodes == 0, target_nodes == 1).sum()
                TP = np.logical_and(pred_nodes == 1, target_nodes == 1).sum()
                TN = np.logical_and(pred_nodes == 0, target_nodes == 0).sum()
                
                # Compute FA
                FA += FP / (FP + TN)
                
                # Compute Recall(DR)
                temp_recall = TP / (TP+FN)
                recall += temp_recall
                
                # Compute Precision
                prec_denom = (TP+FP)
                if( prec_denom == 0 ):
                    temp_prec = 0
                else:
                    temp_prec = TP / prec_denom
                precision += temp_prec
                
                # Compute F1
                if(temp_prec == 0 or temp_recall == 0):
                    temp_F1 = 0
                else:
                    temp_F1 = 2*((temp_prec*temp_recall)/(temp_prec+temp_recall))
                
                # Compute a strict match
                if( np.array_equal(pred_nodes, target_nodes) ):
                    localization_accuracy += 1
            
            # Sample percentage having F1 >= 95%
            if(temp_F1 >= 0.95):
                 F1 += 1
                
        
        # Number of elements in metrices (is used to get mean)
        n_elem_metrices = len(valid_loader)
        return precision/n_elem_metrices, recall/n_elem_metrices, F1/n_elem_metrices, localization_accuracy/n_elem_metrices, detection_accuracy/n_elem_metrices, FA/n_elem_metrices, total_loss/n_elem_metrices


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
    
    # Initialization of variables for early pruning
    best_loss = 99.99
    patience_counter = 0

    
    """
    # Generate dropout
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    
    # Generate hidden features
    hidden_channels = trial.suggest_categorical("hidden_channels", [32, 64, 128])
    
    # Generate number of stacks (K)
    num_stacks = trial.suggest_categorical("num_stacks", [3, 4, 5])
    
    # Generate number of layers (T)
    num_layers = trial.suggest_categorical("num_layers", [4, 5, 6, 7])
    
    # Generate the lr
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    """
    
    # Generate dropout
    dropout = 0.4
    
    # Generate hidden features
    hidden_channels = 32
    
    # Generate number of stacks (K)
    num_stacks = 3
    
    # Generate number of layers (T)
    num_layers = 5
    
    # Generate the model
    model = define_model(trial, dropout, hidden_channels, num_stacks, num_layers).to(device)
    
    # Generate the lr
    lr = 1e-3
    
    # - Define the criterion --
    pos_weight = trial.suggest_float("pos_weight", 1, 20)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight).to(device))

    # Define the optimizer.
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=config["weight_decay"])
    
    # Define scheduler
    scheduler1 = LambdaLR(optimizer, lr_lambda)
    scheduler2 = CosineAnnealingLR(optimizer, T_max=90*config["num_epochs"])
    scheduler = SequentialLR(optimizer, schedulers=[scheduler1, scheduler2], milestones=[config["n_lrWarmup_steps"]])

    # Training&Validating of the model.
    for epoch in range(config["num_epochs"]):
        
        train_loss = train_epoch(model, optimizer, epoch, train_loader, criterion, scheduler)
        prec, rec, f1, loc_acc, det_acc, FA, avg_loss = validate(model, valid_loader, criterion)
        
        print(f"Current trial: {trial.number}")
        print(f"\nEpoch {epoch}: Average Train Loss={train_loss:.4f}")
        print(f"Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}")
        print(f"FA={FA:.4f}, Validation Loss: {avg_loss:.4f}")
        print(f"Localization accuracy={loc_acc:.4f}, Detection accuracy:{det_acc:.4f}")
        print("\n🔧 Trial Hyperparameters:")
        print(f"  • Pos. weight:                {pos_weight:.3f}")
        print(f"  • Dropout:                {dropout:.3f}")
        print(f"  • Hidden channels:        {hidden_channels}")
        print(f"  • Num stacks (K):         {num_stacks}")
        print(f"  • Num layers (T):         {num_layers}")
        print(f"  • Learning rate (lr):     {lr:.2e}")
        print(f"  • Weight decay:           {config['weight_decay']}")
        print("----------------------------------------------------\n\n")

        trial.report(f1, epoch)
        
        # Early pruning
        if best_loss - avg_loss >= 1e-4:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 16:
                print(f"Early pruning: no Validation loss improvement for 16 epochs. Best loss: {best_loss:.4f}")
                raise optuna.exceptions.TrialPruned()

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
            

    return f1


# # # # # # # # # # # # # # # # # #
# ---- Hyperparameter tuning ---- #
# # # # # # # # # # # # # # # # # #

sampler = TPESampler(seed=123)


optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
study_name = "studyARMA_posweight5"+datetime.now().strftime("%m_%d_%y")  # Unique identifier of the study.
storage_name = "sqlite:///saved_Optuna/{}.db".format(study_name)
study = optuna.create_study(
                            direction="maximize", 
                            sampler=sampler,
                            pruner=optuna.pruners.PatientPruner(wrapped_pruner=None, patience=5, min_delta=0.0),
                            study_name=study_name,
                            storage=storage_name
                            )

study.optimize(objective, n_trials=50, timeout=None)

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