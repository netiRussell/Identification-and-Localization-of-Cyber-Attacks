import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import ChebConv
import torch.nn.utils as utils
from torch_geometric.utils import to_dense_batch
from torch_geometric.utils import get_laplacian

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
    def __init__(self, in_channels, u, Ks, dropout=0.1, num_nodes=300 ):
        super(CGCN, self).__init__()
        
        # Local parameter
        self.num_nodes = num_nodes
     
        
        # Chebuchev Conv layers
        self.chebConv1 = ChebConv(in_channels, u, Ks)
        self.chebConv2 = ChebConv(u, u, Ks)
        self.chebConv3 = ChebConv(u, u, Ks)
        self.chebConv4 = ChebConv(u, u, Ks)
        # self.chebConv5 = ChebConv(u, u, Ks)
    
        
        # Layer Normalizations
        self.layerNorm1 = nn.LayerNorm(u)
        self.layerNorm3 = nn.LayerNorm(u)
        
        
        # Dropouts
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)
        #self.dropout5 = nn.Dropout(dropout)
        
        
        self.classifier  = nn.Linear(u, 1) # Transforms each node's into a single feature node

    def forward(self, x, edge_index, weights, batch):        
        # 4 CGCN layers
        # Relu and Dropout are applied after each one of them
        x = self.chebConv1(x, edge_index, weights)
        x = self.layerNorm1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        #if torch.isnan(x).any() or torch.isinf(x).any():
        #    raise RuntimeError("NaN/Inf in x → after chebConv1")
        
        x = self.chebConv2(x, edge_index, weights)
        x = F.relu(x)
        x = self.dropout2(x)
        #if torch.isnan(x).any() or torch.isinf(x).any():
        #    raise RuntimeError("NaN/Inf in x → after chebConv2")
        
        x = self.chebConv3(x, edge_index, weights)
        x = self.layerNorm3(x)
        x = F.relu(x)
        x = self.dropout3(x)
        #if torch.isnan(x).any() or torch.isinf(x).any():
        #    raise RuntimeError("NaN/Inf in x → after chebConv3")
        
        x = self.chebConv4(x, edge_index, weights)
        x = F.relu(x)
        x = self.dropout4(x)
        #if torch.isnan(x).any() or torch.isinf(x).any():
        #    raise RuntimeError("NaN/Inf in x → after chebConv4")
        
        """
        x = self.chebConv5(x, self.edge_index_norm, self.weight_norm)
        x = self.bn5(x)
        x = F.relu(x)
        x = self.dropout5(x)
        #if torch.isnan(x).any() or torch.isinf(x).any():
        #    raise RuntimeError("NaN/Inf in x → after chebConv4")
        """
        
        # from [total_nodes, u] → [batch_size, num_nodes, u]
        x, _ = to_dense_batch(x, batch, max_num_nodes=self.num_nodes)
        
        # Classification logits per node (Localizing an attack)
        logits_nodes = self.classifier(x).squeeze(-1)              # [batch_size, num_nodes]
        if not torch.isfinite(x).all():
            raise ValueError("Classifier output NaNs/Infs")
        
        # Classification logits per graph (Identifying an attack)
        logits_graph = logits_nodes.max(dim=1).values # choose max one
        

        return logits_nodes, logits_graph
       
