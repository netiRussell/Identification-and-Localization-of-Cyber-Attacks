import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from model.ARMA_Transformer import GNNArmaTransformer
from torch.utils.data import random_split
import gc

from dataset import FDIADataset
from dataset_generators.functions import save_checkpoint

# TODO: delete at the final stage
import sys

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

train_dataset, val_dataset, test_dataset = random_split(
    dataset,
    [train_len, val_len, test_len],
    generator=torch.Generator().manual_seed(123)  # for reproducibility
)

train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=config["batch_size"])

in_feats = dataset[0].x.size(1)


# -- Instantiate model, loss, optimizer, scheduler --
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device selected: {device}")

model = GNNArmaTransformer(
    in_channels=in_feats,
    hidden_channels=config["hidden_channels"],
    out_channels=config["out_channels"],
    num_stacks=config["num_stacks"], 
    num_layers=config["num_layers"],
    transformer_heads=config["transformer_heads"],
    transformer_layers=config["transformer_layers"]
).to(device)

criterion = nn.CrossEntropyLoss()            # maps logits [N,2] + labels [N] → scalar
optimizer = optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
scheduler = optim.lr_scheduler.StepLR(
    optimizer, step_size=20, gamma=0.5
)



# -- Training & evaluation functions --
def train_epoch(epoch):
    model.train()
    total_loss = 0
    accum_steps = 64 # 64 mini batches filled with 4 samples = 256 samples per optimizer.step()
    optimizer.zero_grad()
    
    for minibatch_id, minibatch in enumerate(train_loader):
        minibatch = minibatch.to(device)
        logits = model(minibatch.x, minibatch.edge_index, weights=minibatch.edge_attr, batch=minibatch.batch)    # [total_nodes, 2]
        loss   = criterion(logits, minibatch.y)
        total_loss += loss.item()

        # scale down the loss so grads are averaged over accum_steps
        (loss / accum_steps).backward()
        print(f"Epoch#{epoch} Mini-Batch#{minibatch_id} | Current loss: {loss:.4f}")
        
        if minibatch_id % accum_steps == 0:
            print("One batch is finished")
            optimizer.step()
            optimizer.zero_grad()
        
    return total_loss / len(train_loader)

@torch.no_grad()
def evaluate(eval_loader):
    model.eval()
    correct = 0
    total   = 0
    for batch in eval_loader:
        batch = batch.to(device)
        logits = model(batch.x, batch.edge_index, weights=batch.edge_attr, batch=batch.batch)
        preds  = logits.argmax(dim=1)
        correct += (preds == batch.y).sum().item()
        total   += batch.num_nodes
    
    print(f"Current validation score: {correct / total}")
    return correct / total


# # # # # # # # # # # # # # # # # # #
# ---- Training and validation ---- #
# # # # # # # # # # # # # # # # # # #
losses = []
accuracies = []

for epoch in range(1, config["num_epochs"]+1):
    loss = train_epoch(epoch)
    losses.append(loss)
            
    
    val_acc  = evaluate(val_loader)
    accuracies.append(val_acc)
    
    scheduler.step()
    print(f"Epoch {epoch:02d} | Train Loss: {loss:.4f} | Val Acc: {val_acc:.4f}")
    
    # -- Save progress of training --
    save_checkpoint({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
                'currentEpoch': epoch,
                'losses': losses,
                'accuracies': accuracies,
                })
    print('The model has been successfully saved')
    
    # Clear cache to keep RAM usage low
    torch.cuda.empty_cache()
    gc.collect()
    
    if( len(losses) > 16 ):
        current_difference = losses[-17] - losses[-1]
        if(current_difference < 1e-4):
            print(f'Early stop of the training to prevent overfitting. losses[-17]: {losses[-17]}, losses[-1]: {losses[-1]}')
            break


# # # # # # # # # # #
# ---- Testing ---- #
# # # # # # # # # # #

# TODO: to be implemented in a separate file with the same
# random split manual_seed = 123
#
# attacked_ids = predict_attacked_buses(model, data, threshold=0.5)
