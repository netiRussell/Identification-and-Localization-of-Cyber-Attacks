import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from model.ARMA_Transformer import GNNArmaTransformer
from torch.utils.data import random_split
from torch.amp import autocast, GradScaler
import gc
import time


from dataset import FDIADataset
from dataset_generators.functions import save_checkpoint

# TODO: delete at the final stage
import sys


# Clear cache from the previous training:
torch.cuda.empty_cache()
gc.collect()

'''
Questions:
1) Is it okay to randomly apply either As or Ad attack to get resulting 36k samples.

Input: edge_indices, weights, node features
Output: 2848 boolean values where False means no attack on the bus and Trues means the bus has been attacked
'''


config = {
              "dataset_root": "./dataset",
    
              "num_epochs": 256,
              "batch_size": 4, # 256 with gradient accumulation
              "lr": 1e-3, 
              "weight_decay": 1e-5,
              
              "hidden_channels": 128,
              "out_channels": 512,
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

criterion = nn.CrossEntropyLoss()            # maps logits [N,2] + labels [N] → scalar
optimizer = optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
scheduler = optim.lr_scheduler.StepLR(
    optimizer, step_size=20, gamma=0.5
)
use_cuda = (device.type == 'cuda')
scaler = GradScaler(enabled=use_cuda)



# -- Training & evaluation functions --
def train_epoch(epoch):
    model.train()
    total_loss = 0
    accum_steps = 64 # 64 mini batches filled with 4 samples = 256 samples per optimizer.step()
    optimizer.zero_grad()
    
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
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
    return total_loss / len(train_loader)


@torch.no_grad() # TODO is it in the right place?
def evaluate(eval_loader, batch_size):
    model.eval()
    correct = 0
    strict_correct = 0
    total   = len(eval_loader) * batch_size * 2848
    
    for batch in eval_loader:
        batch = batch.to(device)
        logits = model(batch.x, batch.edge_index, weights=batch.edge_attr, batch=batch.batch)
        preds  = logits.argmax(dim=1)
        current_correct = (preds == batch.y).sum().item()
        correct += current_correct
        
        if(current_correct == len(batch.y)):
            strict_correct +=1
    
    print(f"Current validation score: {((correct / total) * 100):.2f}%")
    print(f"Entire mini-batch correctness: {((strict_correct / len(eval_loader))*100):.2f}%")
    return correct / total


# # # # # # # # # # # # # # # # # # #
# ---- Training and validation ---- #
# # # # # # # # # # # # # # # # # # #
start = time.time()
losses = []
accuracies = []

for epoch in range(1, config["num_epochs"]+1):
    loss = train_epoch(epoch)
    losses.append(loss)      
    
    val_acc  = evaluate(val_loader, config['batch_size'])
    accuracies.append(val_acc)
    
    scheduler.step()
    print(f"Average train loss: {loss:.4f}")
    
    # -- Save progress of training --
    save_checkpoint({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
                'currentEpoch': epoch,
                'trainingTime': time.time() - start,
                'losses': losses,
                'accuracies': accuracies,
                })
    print('The model has been successfully saved\n')
    
    
    if( len(losses) > 16 ):
        current_difference = losses[-17] - losses[-1]
        if(current_difference < 1e-4):
            print(f'Early stop of the training to prevent overfitting. losses[-17]: {losses[-17]}, losses[-1]: {losses[-1]}')
            break

