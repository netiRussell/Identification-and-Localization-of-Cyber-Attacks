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
    def __init__(self, in_channels, num_CGCN_layers, u, Ks, dropout=0.1, num_nodes=300 ):
        super(CGCN, self).__init__()
        
        # Local parameter
        self.num_nodes = num_nodes
     
        
        # Chebuchev Conv layers (Dynamic definition)
        self.chebConvs = nn.ModuleList()
        for index in range(num_CGCN_layers):
            in_channels = in_channels if index == 0 else u
            
            self.chebConvs.append( ChebConv(in_channels, u, Ks) )
    
        
        # Layer Normalizations
        self.layerNorms = nn.ModuleList()
        for index in range(num_CGCN_layers // 2):
            self.layerNorms.append( nn.LayerNorm(u) )
        
        # Dropouts
        self.dropouts = nn.ModuleList()
        for index in range(num_CGCN_layers):
            self.dropouts.append( nn.Dropout(dropout) )
        
        
        self.classifier  = nn.Linear(u, 1) # Transforms each node's into a single feature node

    def forward(self, x, edge_index, weights, batch): 
        # Dynamically go over each ChebConv with layerNorm added after every 2nd chebConv
        for idx, chebConv in enumerate(self.chebConvs):
            x = chebConv(x, edge_index, weights)
            
            if (idx+1) % 2 == 0:
                x = self.layerNorms[idx//2](x) # every 2nd iteration has this layer
            
            x = F.relu(x)
            x = self.dropouts[idx](x)
            
            if not torch.isfinite(x).all():
                raise ValueError(f"Layer#{idx} output NaNs/Infs")
        
        # from [total_nodes, u] → [batch_size, num_nodes, u]
        x, mask = to_dense_batch(x, batch, max_num_nodes=self.num_nodes)
        
        # Classification logits per node (Localizing an attack)
        logits_nodes = self.classifier(x).squeeze(-1)              # [batch_size, num_nodes]
        if not torch.isfinite(logits_nodes).all():
            raise ValueError("Classifier output NaNs/Infs")
        
        # Classification logits per graph (Identifying an attack): max over valid nodes only (mask padded nodes)
        logits_graph = logits_nodes.masked_fill(~mask, float('-inf')).max(dim=1).values
        

        return logits_nodes, logits_graph
       
