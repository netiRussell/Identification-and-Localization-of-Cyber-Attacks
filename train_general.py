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
from torch.nn.utils import clip_grad_norm_
from torch_geometric.utils import to_dense_batch
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR


from dataset import FDIADataset
from dataset_generators.functions import save_checkpoint, preparePyGDataset, selectDevice
from dataset_generators.functions import define_modelCGCN, define_modelARMA, define_modelTransformer, define_modelCNN, define_modelMLP, define_modelLSTM

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
Questions:
...

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
              
              "model_selected": 0,
                  #0 = CGCN
                  #1 = ARMA
                  #2 = ARMA-Transformer
                  #3 = CNN
                  #4 = MLP
                  #5 = LSTM
              
              "cgcn_specific": {
                  "lr": 1e-4, 
                  "weight_decay": 0,
                  "n_lrWarmup_steps": 500, # 1 batch = 1 step
                  
                  "num_CGCN_layers": 3,
                  "u": 64, # hidden channels
                  "Ks": 4,
                  "dropout": 0.5,
                  "pos_weight": 4.5,
                  },
              
              "arma_specific": {
                  "lr": 1e-3, 
                  "weight_decay": 1e-2,
                  "n_lrWarmup_steps": 500, # 1 batch = 1 step
                  
                  "num_ARMA_layers": 3,
                  "hidden_channels": 32,
                  "num_stacks": 3,  # K
                  "num_layers": 5,
                  "dropout": 0.4,
                  "pos_weight": 6,
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
                  "transformer_heads": 4,
                  "transformer_layers": 2,
                  "leakyRelu_rate": 0.1,
                  "dropout": 0.2, 
                  "pos_weight": 6,
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

# -- Instantiate model, loss, optimizer, scheduler --
# - Select device -
device = selectDevice()

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


# Linear scheduler for a warmup
scheduler1 = LambdaLR(optimizer, lr_lambda)

# Compute T_max for  CosineAnnealingLR
steps_per_epoch = int(np.ceil(config["num_train_samples"] / config["accumulated_batch_size"]))
total_steps = steps_per_epoch * config["num_epochs"]
T_max = max(1, total_steps - n_lrWarmup_steps)

# Cosine drop in learning rate all the way to nearly 0 at the last epoch
scheduler2 = CosineAnnealingLR(optimizer, T_max=T_max)

# Combination of the two schedulers
scheduler = SequentialLR(optimizer, schedulers=[scheduler1, scheduler2], milestones=[n_lrWarmup_steps])


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

# Scaler and Autocast 
# (seems to have negative impact on the training; hence, turned off)
#use_cuda = (device.type == 'cuda')
#scaler = GradScaler(enabled=use_cuda)
use_cuda = False
scaler = GradScaler(enabled=False)


# -- Training & evaluation functions --
def train_epoch(epoch):
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
            #target_graph = minibatch.y_graph
            
            # Get model's raw output (logits)
            logits_nodes, logits_graph = model(minibatch.x, 
                           minibatch.edge_index, 
                           weights=minibatch.edge_attr,
                           batch=minibatch.batch)
            
            # Compute loss and save it
            loss = criterion(logits_nodes.view(-1), target_nodes)
            #loss += criterion(logits_graph, target_graph.squeeze(0))
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
    
    # Extra gradient step to make sure there are no leftovers
    if (minibatch_id + 1) % accum_steps != 0:
        # Clip gradients so their total L2 norm ≤ 1
        total_norm = clip_grad_norm_(model.parameters(), max_norm=1.0)
        print(f"Grad norm before clipping: {total_norm:.4f}")
        
        # Applying grads
        print(f"Epoch#{epoch}, Last batch\n")
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        optimizer.zero_grad()
        
    return total_loss / len(train_loader)


def validate(valid_loader):
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
    det_metrics_holder = np.zeros( (2, n_elem_metrices), dtype="np.uint8" )
    
    # Make sure grads of the model won't be affected
    with torch.no_grad():
        for current_sample_id, sample in enumerate(valid_loader):
            sample = sample.to(device)

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
                    print("+1 Strict match")
                    localization_accuracy += 1
            
            # Increment the amount of samples >= 95%
            if(temp_F1_SW >= 0.95):
                F1_95_SW += 1
            # Increment the amount of samples <= 5%
            elif(temp_F1_SW <= 0.05):
                F1_5_SW += 1
            
            # TO BE DELETED:
            print(f"Attack: {target_graph}/{graph_pred}, Current F1: {temp_F1_SW:.2f}, Num of buses with f1 > 95%: {F1_95_SW}")
            print(f"Number of ones exptected: {target_nodes.sum()}, predicted: {pred_nodes.sum()}\n")         
        
        
        # Node-Wise (NW) evaluation:
        for bus_id in range( config["num_nodes"] ):      
            # Compute predictions
            FP = np.logical_and(nw_metrics_holder[bus_id][0] == 1, nw_metrics_holder[bus_id][1] == 0).sum()
            FN = np.logical_and(nw_metrics_holder[bus_id][0] == 0, nw_metrics_holder[bus_id][1] == 1).sum()
            TP = np.logical_and(nw_metrics_holder[bus_id][0] == 1, nw_metrics_holder[bus_id][1] == 1).sum()
            
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
    train_loss = train_epoch(epoch)
    
    # Conduct a validation for a single epoch
    det_rec, det_FA, det_f1, f1_95_sw, f1_5_sw, f1_95_nw, f1_5_nw, loc_acc, det_acc, avg_loss = validate(valid_loader)
    
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

