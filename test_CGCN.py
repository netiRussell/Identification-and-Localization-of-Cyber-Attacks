import torch
from torch_geometric.loader import DataLoader
from model.CGCN import CGCN
from torch.utils.data import random_split
import random
import numpy as np

from dataset import FDIADataset
from dataset_generators.functions import visualizeLossValid

# TODO: delete at the final stage
import sys
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# -- Load the saved state --
checkpoint = torch.load('./saved_grads/checkpoint2025_06_07.pth.tar', weights_only=False)
config = checkpoint['config']

# -- Prepare the dataset --
# Enable reproducibility
torch.backends.cudnn.deterministic = True
random.seed(123)
torch.manual_seed(123)
torch.cuda.manual_seed(123)
np.random.seed(123)

# Definition of the lists containing indices of Ad ans As samples
Ad_indices = list(range(config["Ad_start"], config["Ad_end"]))
As_indices = list(range(config["As_start"], config["As_end"]))

# Definition of the list containing indices of not attacked(normal) samples
normal_indices = list(range(int(config["total_num_of_samples"]/2), config["total_num_of_samples"]))

# 4/6 1/6 1/6 split
train_len = int(4/6 * config["total_num_of_samples"])
val_len   = int(1/6 * config["total_num_of_samples"]) + train_len

# Get test indices: 1.5k of Ad + 1.5k of As + 3k of norm = 1/6 of 36k samples or 6k samples total
test_indices = (Ad_indices[config["Ad_train"]+config["Ad_val"]:] + 
               As_indices[config["As_train"]+config["As_val"]:] + 
               normal_indices[config["norm_train"]+config["norm_val"]:])

# Shuffle the indices:
random.shuffle(test_indices)

# Get the final PyG dataset
test_dataset = FDIADataset(test_indices, config["dataset_root"])


# -- Instantiate model --
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True
print(f"Device selected: {device}")

in_feats = test_dataset[0].x.size(1)
"""
model = GNNArmaTransformer(
    in_channels=in_feats,
    hidden_channels=config["hidden_channels"],
    out_channels=config["out_channels"],
    num_stacks=config["num_stacks"], 
    num_layers=config["num_layers"],
    transformer_heads=config["transformer_heads"],
    transformer_layers=config["transformer_layers"]
)
"""
#model = torch.compile(model, backend="aot_eager")
model = CGCN(
    in_channels=in_feats,
    u=32,
    Ks=5    
)
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
    logits = torch.sigmoid(logits)
    logits[0] = 1 if logits[0] > 0.5 else 0
    
    print(int(logits[0]), batch.y_graph.item())
    if(int(logits[0]) == int(batch.y_graph.item())):
        strict_correct +=1

print(f"Test score: {((strict_correct / total) * 100):.2f}%")

visualizeLossValid(checkpoint["fa"], checkpoint["rec"], checkpoint["f1"], checkpoint["accuracies"])

