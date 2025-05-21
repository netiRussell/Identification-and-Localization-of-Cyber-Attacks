import os
import numpy as np
import torch
from torch_geometric.data import Data

class FDIADataset(torch.utils.data.Dataset):
    def __init__(self, root_dir):
        self.root = root_dir
        # find all sample indices by looking for “x{idx}.npy”

    def __len__(self):
        return 36000

    def __getitem__(self, idx):
        # 1) edge_index: shape [2, num_edges]
        edge_index = np.load(self.root + "/" + "edge_indices.npy")
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        
        # 2) edge weights (impedances): shape [num_edges]
        w = np.load(os.path.join(self.root, "weights.npy"))
        # turn into shape [num_edges, 1] so it can be Data.edge_attr
        edge_attr = torch.tensor(w, dtype=torch.float).unsqueeze(-1)
        
        # 3) node features: shape [2848, num_node_feats]
        x = torch.tensor(np.load(os.path.join(self.root, f"x_{idx}.npy")),
                         dtype=torch.float)
        
        # 4) labels: your 0/1 per–node target array
        #    we assume here you have y_{idx}.npy of shape [2848]
        y = torch.tensor(np.load(os.path.join(self.root, f"target_{idx}.npy")),
                         dtype=torch.long)
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)