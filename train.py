import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader

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
              
              "hidden_channels": 128,
              "out_channels": 256,
              "num_stacks": 4, 
              "num_layers": 5,
              
              "transformer_layers": 6,
              "transformer_heads": 8
          }




# --- Prepare the dataset ---
dataset = FDIADataset(config["dataset_root"])

# 4/6 1/6 1/6 split
train_len = int(4/6 * len(dataset))
val_len   = int(1/6 * len(dataset))
test_len   = int(1/6 * len(dataset))

train_dataset, val_dataset, _ = random_split(
    dataset,
    [train_len, val_len, test_len],
    generator=torch.Generator().manual_seed(123)  # for reproducibility
)

train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=config["batch_size"])

in_feats = dataset[0].x.size(1)


# -- Instantiate model, loss, optimizer, scheduler --
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True
print(f"Device selected: {device}")

model = CGCN(
    in_channels=in_feats,
    u=32,
    Ks=5    
)
#model = torch.compile(model, backend="aot_eager")
model = model.to(device)

#criterion = nn.CrossEntropyLoss() # maps logits [N,2] + labels [N] → scalar
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
#use_cuda = (device.type == 'cuda')
#scaler = GradScaler(enabled=use_cuda)
use_cuda = False
scaler = GradScaler(enabled=False)



# -- Training & evaluation functions --
def train_epoch(epoch):
    model.train()
    total_loss = 0
    accum_steps = 64 # 64 mini batches filled with 4 samples = 256 samples per optimizer.step()
    optimizer.zero_grad()
    
    grad_norms = []
    
    for minibatch_id, minibatch in enumerate(train_loader):
        minibatch = minibatch.to(device)
        
        with autocast(device_type=device.type, enabled=use_cuda): 
            y_dense, mask = to_dense_batch(minibatch.y, minibatch.batch)
            target = (y_dense.sum(dim=1) > 0).float()
            
            logits = model(minibatch.x, minibatch.edge_index, weights=minibatch.edge_attr, batch=minibatch.batch)    # [total_nodes, 2]
            
            loss   = criterion(logits, target)
            total_loss += loss.item()

        # Scale down the loss so grads are averaged over accum_steps
        scaler.scale(loss / accum_steps).backward()
        
        if (minibatch_id+1) % accum_steps == 0:
            print(f"Epoch#{epoch}, Batch#{(minibatch_id // accum_steps):04d} | Current loss: {loss:.4f}")
            
            # TODO: check if clipping helps. Find the best threshold value:
            # Unscale so p.grad is the true gradient
            scaler.unscale_(optimizer)
            
            # Find average gradient
            total_grad = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    # L2 norm of this param’s gradient
                    param_norm = p.grad.data.norm(2)
                    total_grad += param_norm.item() ** 2
            total_grad = total_grad ** 0.5
            grad_norms.append(total_grad)
            
            # Clip gradients to avoid overflow (NaN, inf)
            clip_grad_norm_(model.parameters(), 1.0)
            
            
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
    
    # Extra gradient step to make sure there are no leftovers
    print(f"Epoch#{epoch}, Last batch")
    
    norms = np.array(grad_norms)
    print(f"Gradient 85th percentile: {np.percentile(norms, 85)}")
    print(f"Gradient 90th percentile: {np.percentile(norms, 90)}")
    # ! max_norm = percentile*1.1

    clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
        
    return total_loss / len(train_loader)


def validate(val_loader):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)

            # 1) model forward → one logit per graph
            logits = model(
                batch.x,
                batch.edge_index,
                batch.edge_attr,
                batch.batch
            )                           # shape: [batch_size]

            # 2) collapse node labels → [batch_size] float tensor
            y_dense, mask      = to_dense_batch(batch.y.float(), batch.batch)
            graph_targets      = (y_dense.sum(dim=1) > 0).long().cpu()

            # 3) logits → probs → binary preds
            probs      = torch.sigmoid(logits)
            graph_preds = (probs > 0.5).long().cpu()

            all_preds.append(graph_preds)
            all_targets.append(graph_targets)

    # 4) concatenate across batches
    all_preds   = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()

    # 5) compute TP, FN, FP, TN
    TP = np.logical_and(all_preds == 1, all_targets == 1).sum()
    FN = np.logical_and(all_preds == 0, all_targets == 1).sum()
    FP = np.logical_and(all_preds == 1, all_targets == 0).sum()
    TN = np.logical_and(all_preds == 0, all_targets == 0).sum()

    # 6) metrics
    precision = precision_score(all_targets, all_preds, zero_division=0)
    recall    = recall_score(all_targets, all_preds, zero_division=0)
    f1        = f1_score(all_targets, all_preds, zero_division=0)
    accuracy  = (all_preds == all_targets).mean() * 100
    DR        = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    FA        = FP / (FP + TN) if (FP + TN) > 0 else 0.0
    
    return precision, recall, f1, accuracy, DR, FA


# # # # # # # # # # # # # # # # # # #
# ---- Training and validation ---- #
# # # # # # # # # # # # # # # # # # #
best_f1 = 0.0
patience_counter = 0
start = time.time()
accuracies_arr = []
f1_arr = []
dr_arr = []
fa_arr = []

for epoch in range(1, config["num_epochs"] + 1):
    train_loss = train_epoch(epoch)
    
    prec, rec, f1, accuracy, DR, FA = validate(val_loader)
    accuracies_arr.append(accuracy)
    f1_arr.append(f1)
    dr_arr.append(DR)
    fa_arr.append(FA)
    
    scheduler.step(f1)
    elapsed = time.time() - start
    print(f"Epoch {epoch}: Average Train Loss={train_loss:.4f}, Time={elapsed:.1f}s")
    print(f"Accuracy={accuracy}")
    print(f"Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}")
    print(f"DR={DR:.4f}, FA={FA:.4f}")

    # save checkpoint
    save_checkpoint({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
        'currentEpoch': epoch,
        'trainingTime': time.time() - start,
        'accuracies': np.array(accuracies_arr),
        'prec': prec,
        'rec': rec,
        'f1': f1_arr,
        'dr': dr_arr,
        'fa': fa_arr
    })

    # early stopping on F1
    if f1 > best_f1:
        best_f1 = f1
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= 16:
            print("Early stopping: no F1 improvement for 10 epochs.")
            break

