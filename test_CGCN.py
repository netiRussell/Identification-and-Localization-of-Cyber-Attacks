import torch
from torch_geometric.loader import DataLoader
from model.CGCN import CGCN
from torch.utils.data import random_split
import random
import numpy as np

from dataset import FDIADataset
from dataset_generators.functions import visualizeLossValid, preparePyGDataset, selectDevice

# TODO: delete at the final stage
import sys
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Enable reproducibility
torch.backends.cudnn.deterministic = True
random.seed(123)
torch.manual_seed(123)
torch.cuda.manual_seed(123)
np.random.seed(123)


test_config = {
    "checkpoint_name": "checkpoint2025_06_11.pth.tar"
    }


# -- Load the saved state --
checkpoint = torch.load(f"./saved_grads/{test_config['checkpoint_name']}", weights_only=False)
config = checkpoint['config']

# -- Prepare the dataset --
_, _, test_loader, in_feats = preparePyGDataset(config, FDIADataset)


# -- Select device --
device = selectDevice()

# -- Instantiate model --
#model = torch.compile(model, backend="aot_eager")
model = CGCN(
    in_channels=in_feats,
    u=config["u"],
    Ks=config["Ks"],
    dropout=config["dropout"],
    num_nodes = config["num_nodes"]
)
model = model.to(device)
model.load_state_dict(checkpoint['model_state_dict'])


# # # # # # # # # # #
# ---- Testing ---- #
# # # # # # # # # # #
model.eval()
strict_correct = 0
total   = len(test_loader)

visualizeLossValid(checkpoint["fa"], checkpoint["rec"], checkpoint["f1"], checkpoint["accuracies"])

for sample_id, sample in enumerate(test_loader):
    sample = sample.to(device)
    _, logits = model(sample.x, sample.edge_index, weights=sample.edge_attr, batch=sample.batch)
    probs = torch.sigmoid(logits)
    classification = 1 if probs[0] > 0.5 else 0
    
    if sample_id < 1500:
        print("Ad")
    elif sample_id < 3000:
        print("As")
    else:
        print("No attak")
    
    print(f"Output: {probs.item()}({classification}), Expected: {sample.y_graph.item()}\n")
    if(int(classification) == int(sample.y_graph.item())):
        strict_correct +=1

print(f"Test score: {((strict_correct / total) * 100):.2f}%")

