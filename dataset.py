import os
import numpy as np
import torch
from torch_geometric.data import Data
import random

# TODO: to be deleted
import sys

# Enable reproducibility
torch.backends.cudnn.deterministic = True
random.seed(123)
torch.manual_seed(123)
torch.cuda.manual_seed(123)
np.random.seed(123)

# The PyG dataset class
class FDIADataset(torch.utils.data.Dataset):
    def __init__(self, indices, root_dir):
        # Folder where the dataset is stored at
        self.root = root_dir
        
        # Corresponding X_{i}.npy files
        self.indices = indices
        
        # Save repetitive components just once
        # edge_index: shape [2, num_edges]
        ei = np.load(self.root + "/" + "edge_indices.npy", mmap_mode='r')
        self.edge_index = torch.tensor(ei, dtype=torch.long)
        
        # edge weights (impedances): shape [num_edges]
        w = np.load(os.path.join(self.root, "weights.npy"), mmap_mode='r')
        
        # turn into shape [num_edges, 1] so it can be Data.edge_attr
        self.edge_attr = torch.tensor(w, dtype=torch.float).unsqueeze(-1)
        

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # Get the real index of the sample selected
        i = self.indices[idx]
        
        # Retreive the nodes
        x = torch.tensor(np.load(os.path.join(self.root, f"x_{i}.npy")),
                         dtype=torch.float)
        
        # Get raw target with both node-level and graph-level labels
        raw_y = torch.tensor(np.load(os.path.join(self.root, f"target_{i}.npy")),
                         dtype=torch.float)
        # Node-level labels: 0/1 per–node target array of shape [2848]
        y = raw_y[:-1]
        # Graph-level labels: 0 - no attack present, 1 - an attack took place
        y_graph = raw_y[-1]
    

        return Data(x=x, edge_index=self.edge_index, edge_attr=self.edge_attr, y=y, y_graph = y_graph)
    
    
    
    
    