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
from dataset_generators.functions import train_epoch, validate

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
              
              "model_selected": 5,
                  #0 = CGCN
                  #1 = ARMAConv
                  #2 = ARMA-Transformer
                  #3 = CNN
                  #4 = MLP
                  #5 = LSTM
                  
              "optuna_study_name": "studyLSTM_10_27_25",
                  # study12ARMA_09_24_25
                  # studyCGCN_09_25_25
                  # studyMLP_10_02_25
                  # studyLSTM_10_27_25
              
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
                  "dim_feedforward": 512, # Feed Forward dimensions
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
use_amp = (device.type == 'cuda')  # or make it a config flag
scaler = GradScaler(enabled=use_amp)

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
            
        # - Define the criterion  - 
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(hyperparams["pos_weight"]).to(device))
    
        # - Define the optimizer -
        optimizer = optim.AdamW(model.parameters(), lr=hyperparams["lr"], weight_decay=weight_decay)
        
        # - Define scheduler -
        # Computing steps per epoch
        steps_per_epoch = (len(train_loader) + (config["accumulated_batch_size"] - 1)) // config["accumulated_batch_size"]

        # Linear scheduler for a warmup
        scheduler1 = LambdaLR(optimizer, lr_lambda)

        # Cosine drop in learning rate all the way to nearly 0 at the last epoch
        scheduler2 = CosineAnnealingLR(
            optimizer, T_max=steps_per_epoch * config["num_epochs"] - n_lrWarmup_steps
        )

        # Combination of the two schedulers
        scheduler = SequentialLR(
            optimizer, schedulers=[scheduler1, scheduler2], milestones=[n_lrWarmup_steps]
        )
        
        # - Training&Validating of the model -
        for epoch in range(config["num_epochs"]):
            print(f"\n\nCurrent trial: {trial.number}")  
            print(f"Epoch {epoch}")
            
            train_loss = train_epoch(model, optimizer, epoch, train_loader, criterion, 
                                     scheduler, config, device, autocast, use_amp, scaler)
            
            if epoch % 2 != 0:
                # Conduct a validation for a single epoch
                det_rec, det_FA, det_f1, f1_95_sw, f1_5_sw, f1_95_nw, f1_5_nw, loc_acc, det_acc, avg_loss = validate(model, valid_loader, criterion, config, device)
                
                
                # Check how much time have passed since the beginning of the model training
                # and print out the information of the epoch
                print(f"Avg Train Loss={train_loss:.4f}, Avg Validation Loss: {avg_loss:.4f},")
                print(f"Detection DR: {det_rec:.4f}, Detection FA: {det_FA:.4f}, Detection F1: {det_f1:.4f}")
                print(f"SW F1>=95: {f1_95_sw:.4f}, SW F1<=5: {f1_5_sw:.4f}")
                print(f"NW F1>=95: {f1_95_nw:.4f}, NW F1<=5: {f1_5_nw:.4f}")
                print(f"Localization accuracy={loc_acc:.4f}, Detection accuracy:{det_acc:.4f}")
                

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
    # Study name:
    study_name = config["optuna_study_name"]
else:
    # Study name:
    study_name = "study"+config["study_name"]+"_"+datetime.now().strftime("%m_%d_%y")

# Enabling print out
optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))

# Enabling the reusability of the study
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