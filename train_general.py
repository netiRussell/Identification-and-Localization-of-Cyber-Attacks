import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
import random

from torch.amp import autocast, GradScaler
import gc
import time
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from torch_geometric.utils import to_dense_batch
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR


from dataset import FDIADataset
from dataset_generators.functions import save_checkpoint, preparePyGDataset, selectDevice
from dataset_generators.functions import define_modelCGCN, define_modelARMA, define_modelTransformer, define_modelCNN, define_modelMLP, define_modelLSTM
from dataset_generators.functions import train_epoch, train_epoch_old, validate

# TODO: delete at the final stage
import sys
import matplotlib.pyplot as plt


# -- Clear cache from the previous training --
torch.cuda.empty_cache()
gc.collect()

# -- Enable reproducibility --
torch.backends.cudnn.deterministic = True
random.seed(123)
torch.manual_seed(123)
torch.cuda.manual_seed(123)
np.random.seed(123)

'''
Input: edge_indices, weights, node features
Output: 2848 boolean values where False means no attack on the bus and Trues means the bus has been attacked
'''
        
    
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
              
              "model_selected": 4,
                  #0 = CGCN
                  #1 = ARMAConv
                  #2 = ARMA-Transformer (ACEOT)
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
                  
                  "num_ARMA_layers": 3, 
                  "hidden_channels": 64,
                  "num_stacks": 3,
                  "num_layers": 5,
                  "dropout": 0.2,
                  "pos_weight": 1.5,
                  },
              
              "transformer_specific": {
                  "lr": 1e-3, 
                  "weight_decay": 1e-2,
                  "n_lrWarmup_steps": 500, # 1 batch = 1 step
                  
                  "num_ARMA_layers": 3,
                  "hidden_channels": 32, # channels for ARMA
                  "num_stacks": 3,  # K
                  "num_layers": 5,
                  "d_model_channels": 256, # channels of a transformer-encoder
                  "dim_feedforward": 512,
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
                  
                  "num_MLP_layers": 3,
                  "units": 128,
                  "dropout": 0.3,
                  "pos_weight": 1,
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

# -- Instantiate model, loss, optimizer, scheduler --
# - Select device -
device = selectDevice()

# - Scaler and Autocast -
use_amp = (device.type == 'cuda')  # or make it a config flag
scaler = GradScaler(enabled=use_amp)

# - Define the model and model-specific hyperparameters -
model = None
lr = 1e-3
weight_decay = 0
pos_weight = 0
n_lrWarmup_steps = 0
if( config["model_selected"] == 0 ):
    print("CGCN selected for the training")
    
    model = define_modelCGCN(None, config).to(device)
    lr = config["cgcn_specific"]["lr"]
    weight_decay = config["cgcn_specific"]["weight_decay"]
    pos_weight = config["cgcn_specific"]["pos_weight"]
    n_lrWarmup_steps = config["cgcn_specific"]["n_lrWarmup_steps"]
    
elif( config["model_selected"] == 1 ):
    print("ARMA selected for the training")
    
    model = define_modelARMA(None, config).to(device)
    lr = config["arma_specific"]["lr"]
    weight_decay = config["arma_specific"]["weight_decay"]
    pos_weight = config["arma_specific"]["pos_weight"]
    n_lrWarmup_steps = config["arma_specific"]["n_lrWarmup_steps"]
    
elif( config["model_selected"] == 2 ):
    print("ARMA-Transformer selected for the training")
    
    model = define_modelTransformer(None, config).to(device)
    lr = config["transformer_specific"]["lr"]
    weight_decay = config["transformer_specific"]["weight_decay"]
    pos_weight = config["transformer_specific"]["pos_weight"]
    n_lrWarmup_steps = config["transformer_specific"]["n_lrWarmup_steps"]

elif( config["model_selected"] == 3 ):
    print("CNN selected for the training")
    
    model = define_modelCNN(None, config).to(device)
    lr = config["cnn_specific"]["lr"]
    weight_decay = config["cnn_specific"]["weight_decay"]
    pos_weight = config["cnn_specific"]["pos_weight"]
    n_lrWarmup_steps = config["cnn_specific"]["n_lrWarmup_steps"]

elif( config["model_selected"] == 4 ):
    print("MLP selected for the training")
    
    model = define_modelMLP(None, config).to(device)
    lr = config["mlp_specific"]["lr"]
    weight_decay = config["mlp_specific"]["weight_decay"]
    pos_weight = config["mlp_specific"]["pos_weight"]
    n_lrWarmup_steps = config["mlp_specific"]["n_lrWarmup_steps"]
    
elif( config["model_selected"] == 5 ):
    print("LSTM selected for the training")
    
    model = define_modelLSTM(None, config).to(device)
    lr = config["lstm_specific"]["lr"]
    weight_decay = config["lstm_specific"]["weight_decay"]
    pos_weight = config["lstm_specific"]["pos_weight"]
    n_lrWarmup_steps = config["lstm_specific"]["n_lrWarmup_steps"]
    
else:
    raise IndexError("Incorrect model ID selected")
    
#model = torch.compile(model, backend="aot_eager")
model = model.to(device)

# - Define the criterion, optimizer, and scheduler -
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight).to(device))

optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

def lr_lambda(current_step: int):
    if current_step < n_lrWarmup_steps:
        return float(current_step) / n_lrWarmup_steps
    return 1.0  # keep base_lr after warm-up


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


#scheduler = CosineAnnealingLR(optimizer, T_max=90*6) # every 6 epochs (90 batches=steps per epoch)
"""
lr_arr = []
for t in range(90*config["num_epochs"]):
    lr_arr.append(scheduler.get_last_lr())
    scheduler.step()

flat_lr_arr = [x for sub in lr_arr for x in sub]

plt.plot(range(len(lr_arr)), flat_lr_arr )
plt.show()
sys.exit()
"""


# # # # # # # # # # # # # # # # # # #
# ---- Training and validation ---- #
# # # # # # # # # # # # # # # # # # #

# Declaration of global variables
best_loss = 99.99
patience_counter = 0
start = time.time()
loc_accuracies_arr = []
det_accuracies_arr = []
f1_arr = []
rec_arr = [] # DR
fa_arr = []
prec = 0

checkpoint_name = ""
if( config["model_selected"] == 0 ):
    checkpoint_name = "CGCN"
elif( config["model_selected"] == 1 ):
    checkpoint_name = "ARMA" 
elif( config["model_selected"] == 2 ):
    checkpoint_name = "ARMATrans"
elif( config["model_selected"] == 3 ):
    checkpoint_name = "CNN"
elif( config["model_selected"] == 4 ):
    checkpoint_name = "MLP"
elif( config["model_selected"] == 5 ):
    checkpoint_name = "LSTM"
else:
    raise IndexError("Incorrect model ID selected")

for epoch in range(1, config["num_epochs"] + 1):
    # Conduct a training for a single epoch
    train_loss = train_epoch(model, optimizer, epoch, train_loader, criterion, scheduler, config, device, autocast, use_amp, scaler)
    
    # Conduct a validation for a single epoch
    det_rec, det_FA, det_f1, f1_95_sw, f1_5_sw, f1_95_nw, f1_5_nw, loc_acc, det_acc, avg_loss = validate(model, valid_loader, criterion, config, device)
    
    # Append all the metrics from the validation
    loc_accuracies_arr.append(loc_acc)
    det_accuracies_arr.append(det_acc)
    f1_arr.append(f1_95_sw)
    rec_arr.append(det_rec)
    fa_arr.append(det_FA)
    
    # Check how much time have passed since the beginning of the model training
    # and print out the information of the epoch
    elapsed = time.time() - start
    print(f"\nEpoch {epoch}: Time={elapsed:.1f}s,")
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
    print("----------------------------------------------------\n\n")

    # Save checkpoint every 5 epochs
    if(epoch % 5 == 0):
        save_checkpoint({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': config,
            'currentEpoch': epoch,
            'trainingTime': time.time() - start,
            'loc_accuracies': np.array(loc_accuracies_arr),
            'det_accuracies': np.array(det_accuracies_arr),
            'prec': prec,
            'rec': rec_arr,
            'f1': f1_arr,
            'fa': fa_arr
        }, checkpoint_name)
        
        print(f"Checkpoint saved at epoch#{epoch}")
        
    # Early stopping if loss doesn't decrease at least by 10^-4 through 16 epochs
    
    #if epoch >= 100:
    if best_loss - avg_loss >= 1e-4:
        best_loss = avg_loss
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= 16:
            print(f"Early stopping: no Validation loss improvement for 16 epochs. Best loss: {best_loss:.4f}")
            break

# Save checkpoint last time
save_checkpoint({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'config': config,
    'currentEpoch': config["num_epochs"],
    'trainingTime': time.time() - start,
    'loc_accuracies': np.array(loc_accuracies_arr),
    'det_accuracies': np.array(det_accuracies_arr),
    'prec': prec,
    'rec': rec_arr,
    'f1': f1_arr,
    'fa': fa_arr
}, checkpoint_name)

