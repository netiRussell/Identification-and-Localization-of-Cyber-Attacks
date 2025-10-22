"""
TO MAKE OPTUNA MORE EFFICIENT:
    EXCLUDE NUMPY->TENSOR->NUMPY TRANSFORMATIONS, STICK TO TENSOR
    
    GO OVER VALIDATIO AND TRAINING TO OMIT THINGS LIKE:
        CALCULATING THE PRESENSE OF SPIKE AND THE GRADS IN THE TRAINING LOOP INSTEAD OF JUST CLIPPING AND THAT IS IT
"""



import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
import random
import logging

from torch.amp import autocast, GradScaler
import numpy as np
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR

import optuna
from optuna.trial import TrialState
from optuna.samplers import TPESampler, CmaEsSampler
from datetime import datetime

from dataset import FDIADataset
from dataset_generators.functions import preparePyGDataset, selectDevice
from dataset_generators.functions import define_modelCGCN, define_modelARMA, define_modelTransformer, define_modelCNN, define_modelMLP, define_modelLSTM

# TODO: delete at the final stage
import sys

# hyperparams at every function, configs

# -- Define the params and hyperparams for training --
config = {
              "dataset_root": "./final_dataset",
              "num_train_samples": 23040,
              "accumulated_batch_size": 256,
              "in_channels": 2,
              
              
              "num_nodes": 300,
              "num_epochs": 256,
              "batch_size": 4, # 256 with gradient accumulation
              "n_lrWarmup_steps": 500, # 1 batch = 1 step
              
              "model_selected": 1,
                  #0 = CGCN
                  #1 = ARMA
                  #2 = ARMA-Transformer
                  #3 = CNN
                  #4 = MLP
                  #5 = LSTM
              
              "cgcn_specific": {
                  "lr": 1e-3, 
                  "weight_decay": 1e-2,
                  "n_lrWarmup_steps": 500, # 1 batch = 1 step
                  
                  "num_CGCN_layers": 3,
                  "u": 96, # hidden channels
                  "Ks": 4,
                  "dropout": 0.3,
                  "pos_weight": 3,
                  },
              
              "arma_specific": {
                  "lr": 1e-3, 
                  "weight_decay": 1e-2,
                  "n_lrWarmup_steps": 500, # 1 batch = 1 step
                  
                  "num_ARMA_layers": 3, # TODO: TRY CHANGING THESE FOUR
                  "hidden_channels": 64, # TODO: TRY CHANGING THESE FOUR
                  "num_stacks": 3,  # K
                  "num_layers": 5, # TODO: TRY CHANGING THESE FOUR
                  "dropout": 0.2,
                  "pos_weight": 1.5,
                  },
              
              "transformer_specific": {
                  # TODO: make the dim_feedforward parameter adjustable from here
                  "lr": 1e-3, 
                  "weight_decay": 1e-2,
                  "n_lrWarmup_steps": 500, # 1 batch = 1 step
                  
                  "num_ARMA_layers": 3,
                  "hidden_channels": 64, # channels for ARMA
                  "num_stacks": 3,  # K
                  "num_layers": 5,
                  "d_model_channels": 256, # channels of a transformer-encoder
                  "dim_feedforward": 1024, # Feed Forward dimensions
                  "transformer_heads": 4,
                  "transformer_layers": 2,
                  "leakyRelu_rate": 0.1,
                  "dropout": 0.2, 
                  "pos_weight": 9,
                  },
              
              "cnn_specific": {
                  "lr": 1e-3, 
                  "weight_decay": 1e-2,
                  "n_lrWarmup_steps": 500, # 1 batch = 1 step
                  
                  "num_CNN_layers": 3,
                  "Ks": 3,
                  "units": 64,
                  "padding": "same",
                  "dropout": 0.2,
                  "pos_weight": 2,
                  },
              
              "mlp_specific": {
                  "lr": 1e-3, 
                  "weight_decay": 1e-2,
                  "n_lrWarmup_steps": 500, # 1 batch = 1 step
                  
                  "num_MLP_layers": 4,
                  "units": 128,
                  "dropout": 0.2,
                  "pos_weight": 4.5,
                  },
              
              "lstm_specific": {
                  "lr": 1e-3, 
                  "weight_decay": 1e-2,
                  "n_lrWarmup_steps": 500, # 1 batch = 1 step
                  
                  "num_LSTM_layers": 3,
                  "units": 128,
                  "dropout": 0.2,
                  "pos_weight": 4.5,
                  },
              
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

n_lrWarmup_steps = 0
if( config["model_selected"] == 0 ):
    print("CGCN selected for the training")
    
    n_lrWarmup_steps = config["cgcn_specific"]["n_lrWarmup_steps"]
    config["study_name"] = "CGCN"
    
elif( config["model_selected"] == 1 ):
    print("ARMA selected for the training")
    
    n_lrWarmup_steps = config["arma_specific"]["n_lrWarmup_steps"]
    config["study_name"] = "ARMA"
    
elif( config["model_selected"] == 2 ):
    print("ARMA-Transformer selected for the training")
    
    n_lrWarmup_steps = config["transformer_specific"]["n_lrWarmup_steps"]
    config["study_name"] = "ARMA-Transformer"
    
elif( config["model_selected"] == 3 ):
    print("CNN selected for the training")
    
    n_lrWarmup_steps = config["cnn_specific"]["n_lrWarmup_steps"]
    config["study_name"] = "CNN"

elif( config["model_selected"] == 4 ):
    print("MLP selected for the training")
    
    n_lrWarmup_steps = config["mlp_specific"]["n_lrWarmup_steps"]
    config["study_name"] = "MLP"
    
elif( config["model_selected"] == 5 ):
    print("LSTM selected for the training")
    
    n_lrWarmup_steps = config["lstm_specific"]["n_lrWarmup_steps"]
    config["study_name"] = "LSTM"
    
else:
    raise IndexError("Incorrect model ID selected")

# Warm-up function
def lr_lambda(current_step: int):
    if current_step < n_lrWarmup_steps:
        return float(current_step) / n_lrWarmup_steps
    return 1.0  # keep base_lr after warm-up


# -- Custom train and validation --
def train_epoch(model, optimizer, epoch, train_loader, criterion, scheduler):
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
            #target_graph = minibatch.y_graph
            
            # Get model's raw output (logits)
            logits_nodes, logits_graph = model(minibatch.x, 
                           minibatch.edge_index, 
                           weights=minibatch.edge_attr,
                           batch=minibatch.batch)
            
            # Compute loss and save it
            loss = criterion(logits_nodes.view(-1), target_nodes)
            #loss += (criterion(logits_graph, target_graph.squeeze(0)))
            total_loss += (loss.item()/accum_steps)

        # Scale down the loss so that the grads are averaged over accum_steps
        scaler.scale(loss / accum_steps).backward()
        
        # Gradient Accumulation to simulate 2^8 samples per batch while maintaining low RAM usage
        if (minibatch_id + 1) % accum_steps == 0 or (minibatch_id + 1) == len(train_loader):
            # Clip gradients so their total L2 norm ≤ 1
            clip_grad_norm_(model.parameters(), max_norm=10.0)

            # Applying grads
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
        
    return total_loss / len(train_loader)

def validate(model, valid_loader, criterion):
    # Turn on the evaluation mode to exclude features like dropout regularizatiion
    model.eval()
    
    # Keep track of loss for early stopping
    total_loss = 0
    
    # Declare holders for metrics
    det_FA = 0
    det_recall = 0
    det_F1 = 0
    
    F1_95_SW = 0
    F1_5_SW = 0
    
    F1_95_NW = 0
    F1_5_NW = 0
    
    localization_accuracy = 0
    detection_accuracy = 0
    
    # Number of elements in metrices (total # of samples)
    n_elem_metrices = len(valid_loader)
    
    # Holder for NW evaluation
    # [bus_id][0][sample] = predictions
    # [bus_id][1][sample] = targets
    nw_metrics_holder = np.zeros((config["num_nodes"], 2, n_elem_metrices), dtype="float")
    
    # Holder for graph DR, FA, F1 computation
    # [0][sample] = predictions
    # [1][sample] = targets
    det_metrics_holder = np.zeros( (2, n_elem_metrices), dtype="int" )
    
    # Make sure grads of the model won't be affected
    with torch.no_grad():
        for current_sample_id, sample in enumerate(valid_loader):
            sample = sample.to(device)
            
            """
            Sticking to tensors:
                - Make sure the output of model(logits_nodes, logits_graph) is a tensor
                - Delete any redundant tensor transformation since everything is already in tensors
                - NumPy...
            """

            # Get model's raw output(logits)
            logits_nodes, logits_graph = model(
                sample.x,
                sample.edge_index,
                weights = sample.edge_attr,
                batch = sample.batch
            )
            
            # Change dimensions of the graph output to work with batch=1 case
            logits_graph = logits_graph.squeeze(0)
            logits_nodes = logits_nodes.squeeze(0)
            
            # Get node targets for the batch of size 1
            target_nodes = sample.y
            
            # Compute loss
            loss = criterion(logits_nodes, target_nodes)
            #loss += criterion(logits_graph, target_graph)
            total_loss += loss.item()
            
            # Apply activation function on the logits to get node-level probability
            # and convert it into a classification
            pred_nodes = (torch.sigmoid(logits_nodes) >= 0.5).float()  
            
            # Get graph-level classification and the correspoding target
            graph_pred   = int((torch.sigmoid(logits_graph) >= 0.5).item())
            target_graph = int(sample.y_graph.squeeze(0).item())
            
            # Save the information for the future detection evaluation
            det_metrics_holder[0, current_sample_id] = graph_pred
            det_metrics_holder[1, current_sample_id] = target_graph
            
            # Turn tensors into NumPy arrays
            pred_nodes = pred_nodes.cpu().numpy()
            target_nodes = target_nodes.cpu().numpy()
            
            # Save the information for the future NW evaluation
            nw_metrics_holder[:, 0, current_sample_id ] = pred_nodes
            nw_metrics_holder[:, 1, current_sample_id ] = target_nodes          
            
            # Calculate current metrics 
            # Graph-level
            if( target_graph == graph_pred):
                detection_accuracy += 1
            
            # Node-level (SW evaluation)
            if( target_graph == 0 ):
                # - No attack case -
                
                # If prediction is 100% correct (all nodes = 0 since no attack)
                if(np.sum(pred_nodes) == 0):
                    temp_F1_SW = 1.0
                    localization_accuracy += 1
                    
                # If at least 1 node is a mismatch (has a value of 1)    
                else:
                    temp_F1_SW = 0.0
            else:
                # - Attack took place case -
                # Compute predictions
                FP = np.logical_and(pred_nodes == 1, target_nodes == 0).sum()
                FN = np.logical_and(pred_nodes == 0, target_nodes == 1).sum()
                TP = np.logical_and(pred_nodes == 1, target_nodes == 1).sum()
                
                # Compute Recall(DR)
                if((TP+FN) == 0):
                    recall_SW = 0
                else:
                    recall_SW = TP / (TP+FN)
                
                # Compute Precision
                prec_denom = (TP+FP)
                if( prec_denom == 0 ):
                    precision_SW = 0
                else:
                    precision_SW = TP / prec_denom
                
                # Compute F1 of the current sample
                if(precision_SW == 0 or recall_SW == 0):
                    temp_F1_SW = 0
                else:
                    temp_F1_SW = 2*((precision_SW*recall_SW)/(precision_SW+recall_SW))
                
                
                # Compute a strict match
                if( np.array_equal(pred_nodes, target_nodes) ):
                    #print("+1 Strict match")
                    localization_accuracy += 1
            
            # Increment the amount of samples >= 95%
            if(temp_F1_SW >= 0.95):
                F1_95_SW += 1
            # Increment the amount of samples <= 5%
            elif(temp_F1_SW <= 0.05):
                F1_5_SW += 1
            
            # TO BE DELETED:
            #print(f"Attack: {target_graph}/{graph_pred}, Current F1: {temp_F1_SW:.2f}, Num of buses with f1 > 95%: {F1_95_SW}")
            #print(f"Number of ones exptected: {target_nodes.sum()}, predicted: {pred_nodes.sum()}\n")         
        
        
        # Node-Wise (NW) evaluation:
        for bus_id in range( config["num_nodes"] ):      
            # Compute predictions
            FP = np.logical_and(nw_metrics_holder[bus_id][0] == 1, nw_metrics_holder[bus_id][1] == 0).sum()
            FN = np.logical_and(nw_metrics_holder[bus_id][0] == 0, nw_metrics_holder[bus_id][1] == 1).sum()
            TP = np.logical_and(nw_metrics_holder[bus_id][0] == 1, nw_metrics_holder[bus_id][1] == 1).sum()
            
            # Check if there was no attack on this bus 
            # and the model predicted everything correctly
            if( TP == 0 and FN == 0  ):
                if( FP == 0 ):
                    F1_95_NW += 1 # +1 perfect case
                else:
                    F1_5_NW += 1 # +1 worst case
                    
            # Otherwise, this bus has been attacked at some point
            else:
                # Compute Recall(DR)
                if((TP+FN) == 0):
                    recall_NW = 0
                else:
                    recall_NW = TP / (TP+FN)
                
                # Compute Precision
                prec_denom = (TP+FP)
                if( prec_denom == 0 ):
                    precision_NW = 0
                else:
                    precision_NW = TP / prec_denom
                
                # Compute F1 of the current sample
                if(precision_NW == 0 or recall_NW == 0):
                    temp_F1_NW = 0
                else:
                    temp_F1_NW = 2*((precision_NW*recall_NW)/(precision_NW+recall_NW))
                
                
                # Increment the amount of samples >= 95%
                if(temp_F1_NW >= 0.95):
                    F1_95_NW += 1
                # Increment the amount of samples <= 5%
                elif(temp_F1_NW <= 0.05):
                    F1_5_NW += 1
            
            
        # Graph detection evaluation
        # Compute predictions
        FP = np.logical_and(det_metrics_holder[0] == 1, det_metrics_holder[1] == 0).sum()
        FN = np.logical_and(det_metrics_holder[0] == 0, det_metrics_holder[1] == 1).sum()
        TP = np.logical_and(det_metrics_holder[0] == 1, det_metrics_holder[1] == 1).sum()
        TN = np.logical_and(det_metrics_holder[0] == 0, det_metrics_holder[1] == 0).sum()
        
        # Compute FA
        if( (FP + TN) == 0 ):
            det_FA = 0
        else:
            det_FA = FP / (FP + TN)
        
        # Compute Recall(DR)
        if( (TP+FN) == 0 ):
            det_recall = 0
        else:
            det_recall = TP / (TP+FN)
        
        # Compute Precision
        prec_denom = (TP+FP)
        if( prec_denom == 0 ):
            det_precision = 0
        else:
            det_precision = TP / prec_denom
        
        # Compute F1 of the current sample
        if(det_precision == 0 or det_recall == 0):
            det_F1 = 0
        else:
            det_F1 = 2*((det_precision*det_recall)/(det_precision+det_recall))
        
        return  det_recall, det_FA, det_F1, F1_95_SW/n_elem_metrices, F1_5_SW/n_elem_metrices, F1_95_NW/config["num_nodes"], F1_5_NW/config["num_nodes"], localization_accuracy/n_elem_metrices, detection_accuracy/n_elem_metrices,  total_loss/n_elem_metrices

# -- Optuna functions --
def objective(trial):
    model = None
    optimizer = None
    scheduler = None
    criterion = None
    
    try:
        print(f"---- Current trial: {trial.number} ----")
        
        # Initialization of variables for early pruning
        best_loss = 99.99
        patience_counter = 0
        
        # Model-specific values:
        weight_decay = 0
        if( config["model_selected"] == 0 ):
            weight_decay = config["cgcn_specific"]["weight_decay"]
        elif( config["model_selected"] == 1 ):
            weight_decay = config["arma_specific"]["weight_decay"]
        elif( config["model_selected"] == 2 ):
            weight_decay = config["transformer_specific"]["weight_decay"]
        elif( config["model_selected"] == 3 ):
            weight_decay = config["cnn_specific"]["weight_decay"]
        elif( config["model_selected"] == 4 ):
            weight_decay = config["mlp_specific"]["weight_decay"]
        elif( config["model_selected"] == 5 ):
            weight_decay = config["lstm_specific"]["weight_decay"]
        else:
            raise IndexError("Incorrect model ID selected")
        
        hyperparams = {}
        
        # - Define the hyperparameters --
        
        # Generate dropout
        hyperparams["dropout"] = trial.suggest_float("dropout", 0.1, 0.5)
        
        # Generate the lr
        hyperparams["lr"] = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
        
        # Generate the model and model-specific hyperparameters
        model = None
        if( config["model_selected"] == 0 ):
            print("CGCN has been selected")
            
            # Generate positive weight
            hyperparams["pos_weight"] = trial.suggest_float("pos_weight", 1, 4)
            
            # Generate hidden features
            hyperparams["num_CGCN_layers"] = trial.suggest_categorical("num_CGCN_layers", [3, 4])
            
            # Generate number of layers
            hyperparams["u"] = trial.suggest_categorical("u", [16, 32, 64, 96, 128])
            
            # Generate number of stacks
            hyperparams["Ks"] = trial.suggest_categorical("Ks", [3, 5, 7])
            
            # Model
            model = define_modelCGCN(hyperparams, config).to(device)
            
        elif( config["model_selected"] == 1 ):
            print("ARMA has been selected")
            
            # Generate positive weight
            hyperparams["pos_weight"] = trial.suggest_float("pos_weight", 1, 4)
            
            # Generate hidden features
            hyperparams["hidden_channels"] = trial.suggest_categorical("hidden_channels", [32, 64, 96, 128])
            
            # Generate number of stacks
            hyperparams["num_stacks"] = trial.suggest_categorical("num_stacks", [3, 4, 5])
            
            # Generate number of layers
            hyperparams["num_layers"] = trial.suggest_categorical("num_layers", [4, 5, 6])
            
            # Generate number of ARMA layers
            hyperparams["num_ARMA_layers"] = trial.suggest_categorical("num_ARMA_layers", [3, 4])
            
            # Model
            model = define_modelARMA(hyperparams, config).to(device)
            
        elif( config["model_selected"] == 2 ):
            print("Transformer has been selected")
            
            # Generate positive weight
            hyperparams["pos_weight"] = trial.suggest_float("pos_weight", 1, 12)
            
            # Model
            model = define_modelTransformer(hyperparams, config).to(device)
            
        elif( config["model_selected"] == 3 ):
            print("CNN has been selected")
            
            # Generate positive weight
            hyperparams["pos_weight"] = trial.suggest_float("pos_weight", 1, 10)
            
            # Generate number of CNN layers
            hyperparams["num_CNN_layers"] = trial.suggest_categorical("num_CNN_layers", [3, 4])
            
            # Generate Ks
            hyperparams["Ks"] = trial.suggest_categorical("Ks", [3, 4])
            
            # Generate Units
            hyperparams["units"] = trial.suggest_categorical("units", [32, 64, 128])
            
            # Padding is set to "same"
            hyperparams["padding"] = "same"
            
            # Model
            model = define_modelCNN(hyperparams, config).to(device)
            
        elif( config["model_selected"] == 4 ):
             print("MLP has been selected")
             
             # Generate positive weight
             hyperparams["pos_weight"] = trial.suggest_float("pos_weight", 1, 10)
             
             # Generate number of MLP layers
             hyperparams["num_MLP_layers"] = trial.suggest_categorical("num_MLP_layers", [3, 4])
             
             # Generate Units
             hyperparams["units"] = trial.suggest_categorical("units", [32, 64, 128])
             
             # Model
             model = define_modelMLP(hyperparams, config).to(device)  
        
        elif( config["model_selected"] == 5 ):
            print("LSTM has been selected")
            
            # Generate positive weight
            hyperparams["pos_weight"] = trial.suggest_float("pos_weight", 1, 10)
            
            # Generate number of CNN layers
            hyperparams["num_LSTM_layers"] = trial.suggest_categorical("num_LSTM_layers", [3, 4])
            
            # Generate Units
            hyperparams["units"] = trial.suggest_categorical("units", [32, 64, 128])
            
            # Model
            model = define_modelLSTM(hyperparams, config).to(device)
            
        else:
            raise IndexError("Incorrect model ID selected")
            
        # Define the criterion  
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(hyperparams["pos_weight"]).to(device))
    
        # Define the optimizer.
        optimizer = optim.AdamW(model.parameters(), lr=hyperparams["lr"], weight_decay=weight_decay)
        
        # Define scheduler
        scheduler1 = LambdaLR(optimizer, lr_lambda)
        num_batches = config["num_train_samples"] / config["accumulated_batch_size"]
        scheduler2 = CosineAnnealingLR(optimizer, T_max=num_batches*config["num_epochs"]-n_lrWarmup_steps)
        scheduler = SequentialLR(optimizer, schedulers=[scheduler1, scheduler2], milestones=[n_lrWarmup_steps])
        
        # Useful data holders
        loc_accuracies_arr = []
        det_accuracies_arr = []
        f1_arr = []
        rec_arr = []
        fa_arr = []
        
        # Training&Validating of the model.
        for epoch in range(config["num_epochs"]):
            print(f"\n\nCurrent trial: {trial.number}")  
            print(f"Epoch {epoch}")
            
            train_loss = train_epoch(model, optimizer, epoch, train_loader, criterion, scheduler)
            
            if epoch % 2 != 0:
                # Conduct a validation for a single epoch
                det_rec, det_FA, det_f1, f1_95_sw, f1_5_sw, f1_95_nw, f1_5_nw, loc_acc, det_acc, avg_loss = validate(model, valid_loader, criterion)
                
                # Append all the metrics from the validation
                loc_accuracies_arr.append(loc_acc)
                det_accuracies_arr.append(det_acc)
                f1_arr.append(f1_95_sw)
                rec_arr.append(det_rec)
                fa_arr.append(det_FA)
                
                # Check how much time have passed since the beginning of the model training
                # and print out the information of the epoch
                print(f"Avg Train Loss={train_loss:.4f}, Avg Validation Loss: {avg_loss:.4f},")
                print(f"Detection DR: {det_rec:.4f}, Detection FA: {det_FA:.4f}, Detection F1: {det_f1:.4f}")
                print(f"SW F1>=95: {f1_95_sw:.4f}, SW F1<=5: {f1_5_sw:.4f}")
                print(f"NW F1>=95: {f1_95_nw:.4f}, NW F1<=5: {f1_5_nw:.4f}")
                print(f"Localization accuracy={loc_acc:.4f}, Detection accuracy:{det_acc:.4f}")
                
                best_loc = max(loc_accuracies_arr)
                best_det = max(det_accuracies_arr)
                best_f1 = max(f1_arr)
                print(f"Best localization={best_loc*100:.2f}% at epoch#{loc_accuracies_arr.index(best_loc)}, Best detection={best_det*100:.2f} at epoch#{det_accuracies_arr.index(best_det)}")
                print(f"Best f1: {best_f1:.3f} at epoch#{f1_arr.index(best_f1)}")
                print("----------------------------------------------------")
                print("🔧 Trial Hyperparameters:")
                for key, value in hyperparams.items():
                    print(f"\t• {key}: {value:.6f}")
                print("----------------------------------------------------\n\n")
        
                trial.report(f1_95_nw, epoch)
                
                # Handle pruning based on the intermediate value.
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
            
                # Early pruning that avoid overfitting
                if best_loss - avg_loss >= 1e-4:
                    best_loss = avg_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= 8:
                        print(f"Early pruning: no Validation loss improvement for 16 epochs. Best loss: {best_loss:.4f}")
                        raise optuna.exceptions.TrialPruned()
    
        return f1_95_nw
    
    # Free GPU memory between each trial
    finally:
        del model, optimizer, scheduler, criterion
        torch.cuda.empty_cache()


# # # # # # # # # # # # # # # # # #
# ---- Hyperparameter tuning ---- #
# # # # # # # # # # # # # # # # # #
resume = True

if( resume ):
    # Hyper-tuning values:
    study_name = "study12ARMA_09_24_25"
    
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    storage_name = "sqlite:///saved_Optuna/{}.db".format(study_name)
    
else:
    # Hyper-tuning specific values:
    study_name = "study"+config["study_name"]+"_"+datetime.now().strftime("%m_%d_%y")
    
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    storage_name = "sqlite:///saved_Optuna/{}.db".format(study_name)
    
# Study definition
study = optuna.create_study(
                            direction="maximize", 
                            sampler=TPESampler(multivariate=True, group=True, n_startup_trials=20),
                            pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5),
                            study_name=study_name,
                            storage=storage_name,
                            load_if_exists=True
                            )

# Study with aggressive sampling
study.optimize(objective, n_trials=200, timeout=None)
    
""" In case you want to mix samplers in a single tuning run
# Studying with fine sampling
study.sampler = TPESampler(multivariate=True);
study.optimize(objective, n_trials=100, timeout=None)
"""



#  # # # # # # # # # # # # # #
# ---- Printing results ---- #
#  # # # # # # # # # # # # # #

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