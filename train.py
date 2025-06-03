import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
import random

from model.CGCN import CGCN
from model.ARMA_Transformer import GNNArmaTransformer

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
              
              "hidden_channels": 128,
              "out_channels": 256,
              "num_stacks": 4, 
              "num_layers": 5,
              
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

# Get test indices: 1.5k of Ad + 1.5k of As + 3k of norm = 1/6 of 36k samples or 6k samples total
test_indices = (Ad_indices[config["Ad_train"]+config["Ad_val"]:] + 
               As_indices[config["As_train"]+config["As_val"]:] + 
               normal_indices[config["norm_train"]+config["norm_val"]:])

# Print the arrays
print("Train indices:", train_indices)
print("Validation indices:", val_indices)
print("Test indices:", test_indices)

# Print their sizes
print("Train size:", len(train_indices))
print("Validation size:", len(val_indices))
print("Test size:", len(test_indices))
sys.exit()

# Randomize the indices to mix attacked and normal samples
random.seed(123) # for reproducibility
random.shuffle(train_indices)
random.shuffle(normal_indices)

train_dataset = FDIADataset(train_indices, config["dataset_root"])
val_dataset = FDIADataset(train_indices, config["dataset_root"])


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
model = CGCN(
    in_channels=in_feats,
    u=32,
    Ks=5    
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
            # Get target for a batch
            y_dense, _ = to_dense_batch(minibatch.y, minibatch.batch)
            target = (y_dense.sum(dim=1) > 0).float()
            
            # Get model's raw output (logits)
            logits = model(minibatch.x, minibatch.edge_index, weights=minibatch.edge_attr, batch=minibatch.batch)    # [total_nodes, 2]
            
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


def validate(val_loader):
    # Turn on the evaluation mode to exclude features like dropout regularizatiion
    model.eval()
    
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
            )                           # shape: [batch_size]
            
            # Apply activation function on the logits to get probability
            prob = torch.sigmoid(logits)
            
            # Convert probability into a classification
            graph_pred = 1 if prob > 0.5 else 0
            
            # Get target
            graph_target = torch.max(batch.y).item()
            
            # Append current outputs
            all_preds.append(graph_pred)
            all_targets.append(graph_target)
            
            """
            # TODO: to be finished for ARMA training
            # Get metrics
            if(graph_target == 0):
                # No attack case written explicitly (to avoid div by 0)
                if(graph_pred == 0):
                    # Prediction is correct
                    all_f1.append(1)
                    all_fa.append(0)
                    all_dr.append(1)
                else:
                    # Prediction isn't correct
                    all_f1.append(0)
                    all_fa.append(1)
                    all_dr.append(0)      
            else:
            """
        

        # Concatenate across batches
        all_preds   = np.asarray(all_preds)
        all_targets = np.asarray(all_targets)
    
        # Compute TP, FN, FP, TN
        FP = np.logical_and(all_preds == 1, all_targets == 0).sum()
        TN = np.logical_and(all_preds == 0, all_targets == 0).sum()
    
        # Metrics
        precision = precision_score(all_targets, all_preds, zero_division=0)
        recall    = recall_score(all_targets, all_preds, zero_division=0)
        accuracy  = (all_preds == all_targets).mean() * 100
        f1        = f1_score(all_targets, all_preds, zero_division=0)
        FA        = FP / (FP + TN) if (FP + TN) > 0 else 0.0
        
        return precision, recall, f1, accuracy, FA


# # # # # # # # # # # # # # # # # # #
# ---- Training and validation ---- #
# # # # # # # # # # # # # # # # # # #

# Declaration of global variables
best_f1 = 0.0
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
    prec, rec, f1, accuracy, FA = validate(val_loader)
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
    print(f"FA={FA:.4f}, Accuracy={accuracy}")
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

    # Early stopping if f1 doesn't improve 16 epochs in a row
    if f1 > best_f1:
        best_f1 = f1
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= 16:
            print("Early stopping: no Accuracy improvement for 16 epochs.")
            break

