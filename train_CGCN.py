import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
import random

from model.CGCN import CGCN

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
    1) Should we only attack loads, no slack/PV bus:
...

Input: edge_indices, weights, node features
Output: 2848 boolean values where False means no attack on the bus and Trues means the bus has been attacked
'''
        
    
# -- Define the params and hyperparams for training --
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
              "u": 64, # hidden channels
              "Ks": 6,
              "dropout": 0.4,
              
              "transformer_layers": 6,
              "transformer_heads": 8
          }




# --- Prepare the dataset ---
train_loader, valid_loader, _, in_feats = preparePyGDataset(config, FDIADataset)

# -- Instantiate model, loss, optimizer, scheduler --
# - Select device -
device = selectDevice()

# - Define the model -
model = CGCN(
    in_channels=in_feats,
    u=config["u"],
    Ks=config["Ks"],
    dropout=config["dropout"],
    num_nodes = config["num_nodes"]
)
#model = torch.compile(model, backend="aot_eager")
model = model.to(device)

# - Define the criterion, optimizer, and scheduler -
#criterion = nn.CrossEntropyLoss() # maps logits [N,2] + labels [N] → scalar
criterion = nn.BCEWithLogitsLoss()

optimizer = optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])

def lr_lambda(current_step: int):
    if current_step < config["n_lrWarmup_steps"]:
        return float(current_step) / float(500)
    return 1.0  # keep base_lr after warm-up

scheduler = LambdaLR(optimizer, lr_lambda)

# - Scaler and Autocast -
# (seems to have negative impact on the training; hence, turned off)
#use_cuda = (device.type == 'cuda')
#scaler = GradScaler(enabled=use_cuda)
use_cuda = False
scaler = GradScaler(enabled=False)



# -- Training & evaluation functions --
def train_epoch(epoch):
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
            loss += (criterion(logits_graph, target_graph)/2)
            total_loss += (loss.item()/accum_steps)

        # Scale down the loss so that the grads are averaged over accum_steps
        scaler.scale(loss / accum_steps).backward()
        
        # Gradient Accumulation to simulate 2^8 samples per batch while maintaining low RAM usage
        if (minibatch_id+1) % accum_steps == 0:
            print(f"Epoch#{epoch}, Batch#{(minibatch_id // accum_steps):04d} | Current loss: {loss:.4f}")
            scaler.step(optimizer)
            scheduler.step()
            scaler.update()
            optimizer.zero_grad()
    
    # Extra gradient step to make sure there are no leftovers
    print(f"Epoch#{epoch}, Last batch")
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


# # # # # # # # # # # # # # # # # # #
# ---- Training and validation ---- #
# # # # # # # # # # # # # # # # # # #

# Declaration of global variables
best_loss = 99.99
patience_counter = 0
start = time.time()
accuracies_arr = []
f1_arr = []
rec_arr = []
fa_arr = []

for epoch in range(1, config["num_epochs"] + 1):
    # Conduct a training for a single epoch
    train_loss = train_epoch(epoch)
    
    # Conduct a validation for a single epoch
    prec, rec, f1, accuracy, FA, avg_loss = validate(valid_loader)
    # Append all the metrics from the validation
    accuracies_arr.append(accuracy)
    f1_arr.append(f1)
    rec_arr.append(rec)
    fa_arr.append(FA)
    
    # Check how much time have passed since the beginning of the model training
    # and print out the information of the epoch
    elapsed = time.time() - start
    print(f"\nEpoch {epoch}: Average Train Loss={train_loss:.4f}, Time={elapsed:.1f}s")
    print(f"Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}")
    print(f"FA={FA:.4f}, Accuracy={accuracy:.4f}, Validation Loss: {avg_loss:.4f}")
    print("----------------------------------------------------\n\n")

    # Save checkpoint
    save_checkpoint({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
        'currentEpoch': epoch,
        'trainingTime': time.time() - start,
        'accuracies': np.array(accuracies_arr),
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

