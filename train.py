import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from model.ARMA_Transformer import GNNArmaTransformer
from torch.utils.data import random_split
from torch.amp import autocast, GradScaler
import gc
import time
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from torch.nn.utils import clip_grad_norm_


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

# -- Define focal loss criterion --
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        prob = torch.softmax(logits, dim=1)
        p_t = prob.gather(1, targets.unsqueeze(1)).squeeze(1)
        loss = -self.alpha * (1 - p_t)**self.gamma * torch.log(p_t + 1e-9)
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
        
    
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

model = GNNArmaTransformer(
    in_channels=in_feats,
    hidden_channels=config["hidden_channels"],
    out_channels=config["out_channels"],
    num_stacks=config["num_stacks"], 
    num_layers=config["num_layers"],
    transformer_heads=config["transformer_heads"],
    transformer_layers=config["transformer_layers"]
)
model = torch.compile(model, backend="aot_eager")
model = model.to(device)

criterion = FocalLoss(alpha=0.25, gamma=2.0)
optimizer = optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
use_cuda = (device.type == 'cuda')
scaler = GradScaler(enabled=use_cuda)



# -- Training & evaluation functions --
def train_epoch(epoch):
    model.train()
    total_loss = 0
    accum_steps = 64 # 64 mini batches filled with 4 samples = 256 samples per optimizer.step()
    
    for minibatch_id, minibatch in enumerate(train_loader):
        minibatch = minibatch.to(device)
        
        with autocast(device_type=device.type, enabled=use_cuda):   
            logits = model(minibatch.x, minibatch.edge_index, weights=minibatch.edge_attr, batch=minibatch.batch)    # [total_nodes, 2]
            loss   = criterion(logits, minibatch.y)
            total_loss += loss.item()

        # scale down the loss so grads are averaged over accum_steps
        scaler.scale(loss / accum_steps).backward()
        
        if (minibatch_id+1) % accum_steps == 0:
            print(f"Epoch#{epoch}, Batch#{(minibatch_id // accum_steps):04d} | Current loss: {loss:.4f}")
            # TODO: check if clipping helps. Find the best threshold value:
            """
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), 1.0)
            """
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
    
    # Extra gradient step to make sure there are no leftovers
    print(f"Epoch#{epoch}, Last batch")
    """
    scaler.unscale_(optimizer)
    clip_grad_norm_(model.parameters(), 1.0)
    """
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
        
    return total_loss / len(train_loader)


def evaluate(val_loader):
    with torch.no_grad():
        model.eval()
        all_preds, all_targets = [], []
        strict_correct = 0
        
        for batch in val_loader:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, weights=batch.edge_attr, batch=batch.batch)
            
            preds = logits.argmax(dim=1).cpu().numpy()
            targets = batch.y.cpu().numpy()
            current_correct = (preds == batch.y).sum().item()
            
            if(current_correct == len(batch.y)):
                strict_correct +=1
            
            all_preds.append(preds)
            all_targets.append(targets)
             
            
        
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        precision = precision_score(all_targets, all_preds, zero_division=0)
        recall = recall_score(all_targets, all_preds, zero_division=0)
        f1 = f1_score(all_targets, all_preds, zero_division=0)
        
        return precision, recall, f1, ((strict_correct / len(val_loader))*100)


# # # # # # # # # # # # # # # # # # #
# ---- Training and validation ---- #
# # # # # # # # # # # # # # # # # # #
best_f1 = 0.0
patience_counter = 0
start = time.time()
accuracies = np.array()

for epoch in range(1, config["num_epochs"] + 1):
    train_loss = train_epoch(epoch)
    prec, rec, f1, accuracy = evaluate(val_loader)
    accuracies.append(accuracy)
    scheduler.step(f1)
    elapsed = time.time() - start
    print(f"Epoch {epoch}: Average Train Loss={train_loss:.4f}, Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}, Time={elapsed:.1f}s")

    # save checkpoint
    save_checkpoint({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
        'currentEpoch': epoch,
        'trainingTime': time.time() - start,
        'accuracies': np.array(accuracies),
        'prec': prec,
        'rec': rec,
        'f1': f1
    })

    # early stopping on F1
    if f1 > best_f1:
        best_f1 = f1
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= 10:
            print("Early stopping: no F1 improvement for 10 epochs.")
            break

