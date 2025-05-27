import torch
from torch_geometric.loader import DataLoader
from model.ARMA_Transformer import GNNArmaTransformer
from torch.utils.data import random_split

from dataset import FDIADataset
from dataset_generators.functions import visualizeLossValid

# TODO: delete at the final stage
import sys
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# -- Load the saved state --
checkpoint = torch.load('./saved_grads/checkpoint2025_05_26.pth.tar')
config = checkpoint['config']

# -- Prepare the dataset --
dataset = FDIADataset(config["dataset_root"])

# 4/6 1/6 1/6 split
train_len = int(4/6 * len(dataset))
val_len   = int(1/6 * len(dataset))
test_len   = int(1/6 * len(dataset))

_, _, test_dataset = random_split(
    dataset,
    [train_len, val_len, test_len],
    generator=torch.Generator().manual_seed(123)  # for reproducibility
)


# -- Instantiate model --
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True
print(f"Device selected: {device}")

in_feats = dataset[0].x.size(1)
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

model.load_state_dict(checkpoint['model_state_dict'])


# -- Get loader for the dataset --
test_loader = DataLoader(test_dataset, batch_size=1)


# # # # # # # # # # #
# ---- Testing ---- #
# # # # # # # # # # #
model.eval()
strict_correct = 0
total   = len(test_loader)

for batch in test_loader:
    batch = batch.to(device)
    logits = model(batch.x, batch.edge_index, weights=batch.edge_attr, batch=batch.batch)
    preds  = logits.argmax(dim=1)
    current_correct = (preds == batch.y).sum().item()
    
    if(current_correct == len(batch.y)):
        strict_correct +=1

print(f"Test score: {((strict_correct / total) * 100):.2f}%")

visualizeLossValid(checkpoint["prec"], checkpoint["rec"], checkpoint["f1"], checkpoint["accuracies"])

