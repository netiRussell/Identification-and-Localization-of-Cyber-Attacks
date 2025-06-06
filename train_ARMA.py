import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
import random

from model.ARMA import GNNArma

from torch.utils.data import random_split
from torch.amp import autocast, GradScaler
import gc
import time
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from torch.nn.utils import clip_grad_norm_
from torch_geometric.utils import to_dense_batch


from dataset import FDIADataset
from dataset_generators.functions import save_checkpoint

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
              "num_stacks": 3, 
              "num_layers": 5,
              "dropout": 0.1, 
              
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
val_loader   = DataLoader(val_dataset,   batch_size=1, shuffle=False)

# Get input features (Size of X in the first dimension)
# Supposed to be 4 (P, Q, V, angle) or 2 (P, Q)
in_feats = train_dataset[0].x.size(1)



# -- Instantiate model, loss, optimizer, scheduler --
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True
print(f"Device selected: {device}")

# Define the model
model = GNNArma(
    in_channels=config["in_channels"], 
    hidden_channels=config["hidden_channels"],
    num_stacks=config["num_stacks"],
    num_layers=config["num_layers"], 
    dropout=config["dropout"]  
)
#model = torch.compile(model, backend="aot_eager")
model = model.to(device)

# Define the criterion, optimizer, and scheduler
#criterion = nn.CrossEntropyLoss() # maps logits [N,2] + labels [N] → scalar
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
#use_cuda = (device.type == 'cuda')
#scaler = GradScaler(enabled=use_cuda)

# Scaler and Autocast 
# (seems to have negative impact on the training; hence, turned off)
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
            # Get targets for the batch
            target_nodes = minibatch.y
            target_graph = minibatch.y_graph
            
            # Get model's raw output (logits)
            logits_nodes, logits_graph = model(minibatch.x, 
                           minibatch.edge_index, 
                           weights=minibatch.edge_attr, 
                           batch=minibatch.batch)
            
            # Compute loss and save it 
            loss = criterion(logits_nodes, target_nodes)
            loss += criterion(logits_graph, target_graph)
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


def validate(val_loader):
    # Turn on the evaluation mode to exclude features like dropout regularizatiion
    model.eval()
    
    # Keep track of loss for early stopping
    total_loss = 0
    
    # Declare arrays where the predictions and corresponding targets(labels) will be stored
    all_preds = []
    all_targets = []
    
    # Declare arrays for metrics
    all_recalls = []
    all_FAs = []
    all_F1s = []

    # Make sure grads of the model won't be affected
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)

            # Get model's raw output(logits)
            logits_nodes, logits_graph = model(
                batch.x,
                batch.edge_index,
                batch.edge_attr,
                batch.batch
            )
            
            # Get targets for the batch
            target_nodes = batch.y
            target_graph = batch.y_graph
            
            # Compute loss
            loss = criterion(logits_nodes, target_nodes)
            loss += criterion(logits_graph, target_graph)
            total_loss += loss.item()
            
            # Apply activation function on the logits to get node-level probability
            # and convert it into a classification
            pred_nodes = 1 if torch.sigmoid(logits_nodes) > 0.5 else 0    
            
            # Get graph-level classification:
            graph_pred = (logits_graph > 0).long()
            
            # Turn tensors into NumPy arrays
            pred_nodes = pred_nodes.numpy()
            target_nodes = target_nodes.numpy()
            
            # Append current outputs
            print(pred_nodes, target_nodes)
            sys.exit()
            all_preds.append(pred_nodes)
            all_targets.append(target_nodes)
            
            # Calculate current metrics
            FA, F1, recall = 0
            if( target_graph == 0 ):
                # - No attack case -
                # If prediction is 100% correct (all nodes = 0 since no attack)
                if(np.sum(pred_nodes) == 0):
                    FA = 0
                    F1 = 1
                    recall = 1
                # If at least 1 node is a mismatch (has a value of 1)    
                else:
                   FA = 1
                   F1 = 0
                   recall = 0
            else:
                # - Attack took place case -
                # Compute FP, TN, and FA
                FP = np.logical_and(pred_nodes == 1, target_nodes == 0).sum()
                TN = np.logical_and(pred_nodes == 0, target_nodes == 0).sum()
                FA = FP / (FP + TN) if (FP + TN) > 0 else 0.0
                
                # Compute Recall and F1
                F1 = f1_score(target_nodes, pred_nodes, zero_division=0)
                recall = recall_score(target_nodes, pred_nodes, zero_division=0) # DR
                
            # Save metrics
            all_FAs(FA)
            all_F1s.append(F1)
            all_recalls.append(recall)
                
        

        # Concatenate across batches
        all_preds   = np.asarray(all_preds)
        all_targets = np.asarray(all_targets)
    
        # Metrics
        precision = precision_score(all_targets, all_preds, zero_division=0)
        accuracy  = (all_preds == all_targets).mean() * 100
        
        return precision, recall, f1, accuracy, FA, total_loss/len(val_loader)


# # # # # # # # # # # # # # # # # # #
# ---- Training and validation ---- #
# # # # # # # # # # # # # # # # # # #

# Declaration of global variables
best_loss = 99.99
patience_counter = 0
start = time.time()
accuracies_arr = []
f1_arr = []
rec_arr = [] # DR
fa_arr = []

for epoch in range(1, config["num_epochs"] + 1):
    # Conduct a training for a single epoch
    train_loss = train_epoch(epoch)
    
    # Conduct a validation for a single epoch
    prec, rec, f1, accuracy, FA, avg_loss = validate(val_loader)
    # Append all the metrics from the validation
    accuracies_arr.append(accuracy)
    f1_arr.append(f1)
    rec_arr.append(rec)
    fa_arr.append(FA)
    
    # Update sheduler to decrease learning rate in case f1 doesn't improve
    scheduler.step(f1)
    
    # Check how much time have passed since the beginning of the model training
    # and print out the information of the epoch
    elapsed = time.time() - start
    print(f"\nEpoch {epoch}: Average Train Loss={train_loss:.4f}, Time={elapsed:.1f}s")
    print(f"Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}")
    print(f"FA={FA:.4f}, Accuracy={accuracy}, Validation Loss: {avg_loss:.4f}")
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

