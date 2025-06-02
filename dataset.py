import os
import numpy as np
import torch
from torch_geometric.data import Data
import joblib 

class FDIADataset(torch.utils.data.Dataset):
    def __init__(self, root_dir):
        # Folder where the dataset is stored at
        self.root = root_dir
        
        # Save repetitive components just once
        # edge_index: shape [2, num_edges]
        ei = np.load(self.root + "/" + "edge_indices.npy", mmap_mode='r')
        self.edge_index = torch.tensor(ei, dtype=torch.long)
        
        # edge weights (impedances): shape [num_edges]
        w = np.load(os.path.join(self.root, "weights.npy"), mmap_mode='r')
        
        # turn into shape [num_edges, 1] so it can be Data.edge_attr
        self.edge_attr = torch.tensor(w, dtype=torch.float).unsqueeze(-1)
        

    def __len__(self):
        return 36000

    def __getitem__(self, idx):
        # node features: shape [2848, num_node_feats]
        x = torch.tensor(np.load(os.path.join(self.root, f"x_{idx}.npy")),
                         dtype=torch.float)
        
        # Get a range of min and max values
        min_vals = x.min(dim=0).values
        max_vals = x.max(dim=0).values
        range_vals = (max_vals - min_vals).clamp(min=1e-8)
        
        # Apply normal scaling [0,1]
        x = (x - min_vals) / range_vals
        
        
        # labels: 0/1 per–node target array of shape [2848]
        y = torch.tensor(np.load(os.path.join(self.root, f"target_{idx}.npy")),
                         dtype=torch.long)
        return Data(x=x, edge_index=self.edge_index, edge_attr=self.edge_attr, y=y)