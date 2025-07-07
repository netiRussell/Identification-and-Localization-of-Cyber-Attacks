import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
import random

from model.ARMA_Transformer import GNNArmaTransformer

from torch.amp import autocast, GradScaler
import gc
import time
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from torch.nn.utils import clip_grad_norm_
from torch_geometric.utils import to_dense_batch
from torch.optim.lr_scheduler import LambdaLR


from dataset import FDIADataset
from dataset_generators.functions import save_checkpoint, preparePyGDataset, selectDevice

# TODO: delete at the final stage
import sys
# TODO: delete at the final stage
torch.autograd.set_detect_anomaly(True)


# -- Clear cache from the previous training --
torch.cuda.empty_cache()
gc.collect()

# -- Enable reproducibility --
torch.backends.cudnn.deterministic = True
random.seed(123)
torch.manual_seed(123)
torch.cuda.manual_seed(123)
np.random.seed(123)
        
    
# -- Define the params and hyperparams for training --
config = {
              "dataset_root": "./final_dataset",
    
              "num_epochs": 256,
              "batch_size": 4, # 256 with gradient accumulation
              "lr": 1e-3,
              "weight_decay": 1e-2,
              "n_lrWarmup_steps": 600, # 1 batch = 1 step
              
              "in_channels": 2,
              "hidden_channels": 32, # channels for ARMA
              "num_stacks": 3,  # K
              "num_layers": 5,
              "out_channels": 256, # channels of a transformer-encoder
              "transformer_heads": 4,
              "transformer_layers": 2,
              "leakyRelu_rate": 0.1,
              "dropout": 0.2, 
              "num_nodes": 300
          }




# -- Prepare the dataset --
train_loader, valid_loader, _, in_feats = preparePyGDataset(config, FDIADataset)

# -- Instantiate model, loss, optimizer, scheduler --
# - Select device -
device = selectDevice()

# Define the model
model = GNNArmaTransformer(
    in_channels=config["in_channels"],
    hidden_channels=config["hidden_channels"],
    out_channels=config["out_channels"],
    num_stacks=config["num_stacks"], 
    num_layers=config["num_layers"],
    transformer_heads=config["transformer_heads"],
    transformer_layers=config["transformer_layers"],
    dropout=config["dropout"],
    num_nodes=config["num_nodes"],
    leakyRelu_rate=config["leakyRelu_rate"]
)
#model = torch.compile(model, backend="aot_eager")
model = model.to(device)

# - Define the criterion, optimizer, and scheduler -
#criterion = nn.CrossEntropyLoss() # maps logits [N,2] + labels [N] → scalar
criterion = nn.BCEWithLogitsLoss()

optimizer = optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])

def lr_lambda(current_step: int):
    if current_step < config["n_lrWarmup_steps"]:
        return float(current_step) / config["n_lrWarmup_steps"]
    return 1.0  # keep base_lr after warm-up

scheduler = LambdaLR(optimizer, lr_lambda)


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
            
            # TODO: TO BE DELETED, USED FOR DEBUG ONLY
            h, mask = to_dense_batch(minibatch.x, minibatch.batch) # h: [B, N_max, F], mask: [B, N_max]
            actual = mask.sum(dim=1)                       # actual nodes per graph
            if (actual != config["num_nodes"]).any():
                print("Varying sizes:", actual)
                raise ValueError("Graphs have different dimensions")
            
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
            loss += criterion(logits_graph, target_graph.squeeze(0))
            total_loss += (loss.item()/accum_steps)

        # Scale down the loss so that the grads are averaged over accum_steps
        scaler.scale(loss / accum_steps).backward()
        
        # Gradient Accumulation to simulate 2^8 samples per batch while maintaining low RAM usage
        if (minibatch_id+1) % accum_steps == 0:
            # Clip gradients so their total L2 norm ≤ 1
            total_norm = clip_grad_norm_(model.parameters(), max_norm=1.0)
            if(total_norm > 1):
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


def validate(valid_loader):
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
            graph_pred = (logits_graph > 0).long()
            
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
                # Compute FP, TN, and FA
                FP = np.logical_and(pred_nodes == 1, target_nodes == 0).sum()
                TN = np.logical_and(pred_nodes == 0, target_nodes == 0).sum()
                FA += FP / (FP + TN)
                
                # Compute Recall and F1
                temp_F1 = f1_score(target_nodes, pred_nodes, zero_division=0)
                recall += recall_score(target_nodes, pred_nodes, zero_division=0) # DR
                
                # Compute a strict match
                if( np.array_equal(pred_nodes, target_nodes) ):
                    localization_accuracy += 1
            
            # Sample percentage having F1 >= 95%
            if(temp_F1 >= 0.95):
                 F1 += 1
                
        

        # Concatenate across batches
        all_preds   = np.asarray(all_preds)
        all_targets = np.asarray(all_targets)
    
        # Metrics
        precision = precision_score(all_targets, all_preds, zero_division=0, average="micro")
        
        # Number of elements in metrices (is used to get mean)
        n_elem_metrices = len(valid_loader)
        return precision, recall/n_elem_metrices, F1/n_elem_metrices, localization_accuracy/n_elem_metrices, detection_accuracy/n_elem_metrices, FA/n_elem_metrices, total_loss/n_elem_metrices


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

for epoch in range(1, config["num_epochs"] + 1):
    # Conduct a training for a single epoch
    train_loss = train_epoch(epoch)
    
    # Conduct a validation for a single epoch
    prec, rec, f1, loc_acc, det_acc, FA, avg_loss = validate(valid_loader)
    # Append all the metrics from the validation
    loc_accuracies_arr.append(loc_acc)
    det_accuracies_arr.append(det_acc)
    f1_arr.append(f1)
    rec_arr.append(rec)
    fa_arr.append(FA)
    
    # Check how much time have passed since the beginning of the model training
    # and print out the information of the epoch
    elapsed = time.time() - start
    print(f"\nEpoch {epoch}: Average Train Loss={train_loss:.4f}, Time={elapsed:.1f}s")
    print(f"Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}")
    print(f"FA={FA:.4f}, Validation Loss: {avg_loss:.4f}")
    print(f"Localization accuracy={loc_acc:.4f}, Detection accuracy:{det_acc:.4f}")
    print("----------------------------------------------------\n\n")

    # Save checkpoint
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
    })

    # Early stopping if loss doesn't decrease at least by 10^-4 through 16 epochs
    if best_loss - avg_loss >= 1e-4:
        best_loss = avg_loss
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= 16:
            print(f"Early stopping: no Validation loss improvement for 16 epochs. Best loss: {best_loss:.4f}")
            break

