import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from model.ARMA_Transformer import GNNArmaTransformer

'''
Questions:
1) Is it okay to have more than 36k samples?
2) Is it okay to have 1-15 buses attacked instead of 5-15?

'''

# --- 1) Prepare your data (PyG dataset) ---
# Assume you have a PyG `InMemoryDataset` where each Data.y is a long tensor
# of size [num_nodes] with values 0 (safe) or 1 (attacked).

dataset = ...                # your full dataset
train_dataset, val_dataset = dataset[:len(dataset)//2], dataset[len(dataset)//2:]

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=4)

# --- 2) Instantiate model, loss, optimizer, scheduler ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = GNNArmaTransformer(
    in_channels=dataset.num_node_features,
    hidden_channels=128,
    gnn_out_channels=256,
    num_stacks=4, 
    num_layers=8,
    transformer_heads=8,
    transformer_layers=4
).to(device)

criterion = nn.CrossEntropyLoss()            # maps logits [N,2] + labels [N] → scalar
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = optim.lr_scheduler.StepLR(
    optimizer, step_size=20, gamma=0.5
)

# --- 3) Training & evaluation functions ---
def train_epoch():
    model.train()
    total_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        logits = model(batch.x, batch.edge_index)    # [total_nodes, 2]
        loss   = criterion(logits, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

@torch.no_grad()
def evaluate(loader):
    model.eval()
    correct = 0
    total   = 0
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch.x, batch.edge_index)
        preds  = logits.argmax(dim=1)
        correct += (preds == batch.y).sum().item()
        total   += batch.num_nodes
    return correct / total

# --- 4) Run training loop ---
num_epochs = 50
for epoch in range(1, num_epochs+1):
    loss     = train_epoch()
    val_acc  = evaluate(val_loader)
    scheduler.step()
    print(f"Epoch {epoch:02d} | Train Loss: {loss:.4f} | Val Acc: {val_acc:.4f}")

# --- 5) After training: get attacked IDs on a new graph ---
# data = your test Data object
# attacked_ids = predict_attacked_buses(model, data, threshold=0.5)
