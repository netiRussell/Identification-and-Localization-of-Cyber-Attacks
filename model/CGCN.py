import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import ChebConv, GraphNorm, global_mean_pool
import torch.nn.utils as utils

# TODO: to be deleted
import sys

class CGCN(nn.Module):
    """
      Chebushev GCN model from "Cyberattack_Detection_in_Large-Scale_Smart_Grids_using_Chebyshev_Graph_Convolutional_Networks"

      in_channels: Number of channels/features for each node.
      u: Number of units/neurons in each graph convolutional layer (or just hidden channels). 
      Ks: Spatial kernel size.
      droupout: During training, randomly zeroes some of the elements of the input tensor with this probability
      edge_index: The edge indices representing the connectivity of the graph.
      
      Batch normalization layers help
    """
    def __init__(self, in_channels, u, Ks, dropout=0.1 ):
        super(CGCN, self).__init__()
        
        self.chebConv1 = ChebConv(in_channels, u, Ks)
        self.gn1 = GraphNorm(u)
        self.dropout1 = nn.Dropout(dropout)

        self.chebConv2 = ChebConv(u, u, Ks)
        self.gn2 = GraphNorm(u)
        self.dropout2 = nn.Dropout(dropout)

        self.chebConv3 = ChebConv(u, u, Ks)
        self.gn3 = GraphNorm(u)
        self.dropout3 = nn.Dropout(dropout)

        self.chebConv4 = ChebConv(u, u, Ks)
        self.gn4 = GraphNorm(u)
        self.dropout4 = nn.Dropout(dropout)
        
        """
        self.chebConv5 = ChebConv(u, u, Ks)
        self.gn5 = GraphNorm(u)
        self.dropout5 = nn.Dropout(dropout)
        """
        
        self.dense = nn.Linear(u, 1) 

    def forward(self, x, edge_index, weights, batch):
        # 4 CGCN layers
        # Relu and Dropout are applied after each one of them
        x = self.chebConv1(x, edge_index, weights)
        x = self.gn1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        #if torch.isnan(x).any() or torch.isinf(x).any():
        #    raise RuntimeError("NaN/Inf in x → after chebConv1")
        
        x = self.chebConv2(x, edge_index, weights)
        x = self.gn2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        #if torch.isnan(x).any() or torch.isinf(x).any():
        #    raise RuntimeError("NaN/Inf in x → after chebConv2")
        
        x = self.chebConv3(x, edge_index, weights)
        x = self.gn3(x)
        x = F.relu(x)
        x = self.dropout3(x)
        #if torch.isnan(x).any() or torch.isinf(x).any():
        #    raise RuntimeError("NaN/Inf in x → after chebConv3")
        
        x = self.chebConv4(x, edge_index, weights)
        x = self.gn4(x)
        x = F.relu(x)
        x = self.dropout4(x)
        #if torch.isnan(x).any() or torch.isinf(x).any():
        #    raise RuntimeError("NaN/Inf in x → after chebConv4")
        
        """
        x = self.chebConv5(x, edge_index, weights)
        x = self.gn5(x)
        x = F.relu(x)
        x = self.dropout5(x)
        #if torch.isnan(x).any() or torch.isinf(x).any():
        #    raise RuntimeError("NaN/Inf in x → after chebConv4")
        """
        
        # Collapse nodes to a graph representation
        x = global_mean_pool(x, batch) # (batch_size, hidden)
        # Avoid NaN, inf
        x = x.nan_to_num(0.0, posinf=1e6, neginf=-1e6)

        return self.dense(x).squeeze(-1) # (batch_size,)
