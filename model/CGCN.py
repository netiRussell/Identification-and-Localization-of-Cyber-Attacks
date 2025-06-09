import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import ChebConv, BatchNorm
import torch.nn.utils as utils
from torch_geometric.utils import to_dense_batch

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
    def __init__(self, in_channels, u, Ks, dropout=0.1, num_nodes=2848 ):
        super(CGCN, self).__init__()
        
        self.num_nodes = num_nodes
        
        self.chebConv1 = ChebConv(in_channels, u, Ks)
        self.bn1 = BatchNorm(u)
        self.dropout1 = nn.Dropout(dropout)

        self.chebConv2 = ChebConv(u, u, Ks)
        self.bn2 = BatchNorm(u)
        self.dropout2 = nn.Dropout(dropout)

        self.chebConv3 = ChebConv(u, u, Ks)
        self.bn3 = BatchNorm(u)
        self.dropout3 = nn.Dropout(dropout)

        self.chebConv4 = ChebConv(u, u, Ks)
        self.bn4 = BatchNorm(u)
        self.dropout4 = nn.Dropout(dropout)
        
        """
        self.chebConv5 = ChebConv(u, u, Ks)
        self.bn5 = BatchNorm(u)
        self.dropout5 = nn.Dropout(dropout)
        """
        
        self.flatten = nn.Flatten(start_dim=1)
        
        self.dense = nn.Linear(u*num_nodes, 1) 

    def forward(self, x, edge_index, weights, batch):
        # 4 CGCN layers
        # Relu and Dropout are applied after each one of them
        x = self.chebConv1(x, edge_index, weights)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        #if torch.isnan(x).any() or torch.isinf(x).any():
        #    raise RuntimeError("NaN/Inf in x → after chebConv1")
        
        x = self.chebConv2(x, edge_index, weights)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        #if torch.isnan(x).any() or torch.isinf(x).any():
        #    raise RuntimeError("NaN/Inf in x → after chebConv2")
        
        x = self.chebConv3(x, edge_index, weights)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout3(x)
        #if torch.isnan(x).any() or torch.isinf(x).any():
        #    raise RuntimeError("NaN/Inf in x → after chebConv3")
        
        x = self.chebConv4(x, edge_index, weights)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.dropout4(x)
        #if torch.isnan(x).any() or torch.isinf(x).any():
        #    raise RuntimeError("NaN/Inf in x → after chebConv4")
        
        """
        x = self.chebConv5(x, edge_index, weights)
        x = self.bn5(x)
        x = F.relu(x)
        x = self.dropout5(x)
        #if torch.isnan(x).any() or torch.isinf(x).any():
        #    raise RuntimeError("NaN/Inf in x → after chebConv4")
        """
        
        # from [total_nodes, u] → [batch_size, num_nodes, u]
        x, mask = to_dense_batch(x, batch, max_num_nodes=self.num_nodes)
        
        # flatten per-graph: [B, N, u] → [B, N·u]
        x = self.flatten(x)

        # Avoid NaN, inf
        #x = x.nan_to_num(0.0, posinf=1e6, neginf=-1e6)

        return self.dense(x).squeeze(-1) # (batch_size,)
       
