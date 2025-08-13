import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import ARMAConv, BatchNorm
from torch_geometric.utils import to_dense_batch

# to be deleted
import sys

class GNNArma(nn.Module):
    """
    GNN encoder using ARMA filters.
    """

    def __init__(self, in_channels=2, num_ARMA_layers=3, hidden_channels=32, 
                 num_stacks=3, num_layers=5, dropout=0.5, num_nodes=300):
        super(GNNArma, self).__init__()
        
        self.num_nodes = num_nodes
        
        # ARMA convolutional graph layers (Dynamic definition)
        self.ARMAconvs = nn.ModuleList()
        for index in range(num_ARMA_layers):
            in_channels = in_channels if index == 0 else hidden_channels
            out_channels = 1 if index == num_ARMA_layers-1 else hidden_channels
            
            self.ARMAconvs.append( ARMAConv(in_channels, out_channels,
                                  num_stacks=num_stacks, num_layers=num_layers) )
        
        # Layer Normalizations
        self.layerNorms = nn.ModuleList()
        for index in range(num_ARMA_layers // 2):
            u = 1 if (index+1)*2 == num_ARMA_layers else hidden_channels
            self.layerNorms.append( nn.LayerNorm(u) )
        
        # Dropouts
        self.dropouts = nn.ModuleList()
        for index in range(num_ARMA_layers):
            self.dropouts.append( nn.Dropout(dropout) )
            
        # Classifier
        self.classifier = nn.Linear(1, 1)
        

    def forward(self, x, edge_index, weights, batch):
        # Dynamically go over each ArmaConv with layerNorm added after every 2nd ArmaConv
        for idx, ARMALayer in enumerate(self.ARMAconvs):
            x = ARMALayer(x, edge_index, weights)
            
            if (idx+1) % 2 == 0:
                x = self.layerNorms[idx//2](x) # every 2nd iteration has this layer
            
            x = F.relu(x)
            x = self.dropouts[idx](x)
            
            if not torch.isfinite(x).all():
                raise ValueError(f"Layer#{idx} output NaNs/Infs")
        
        # from [total_nodes, u] → [batch_size, num_nodes, u]
        x, _ = to_dense_batch(x, batch, max_num_nodes=self.num_nodes)
        
        # Classification logits per node (Localizing an attack)
        logits_nodes = self.classifier(x).squeeze(-1)              # [batch_size, num_nodes]
        if not torch.isfinite(logits_nodes).all():
            raise ValueError("Classifier output NaNs/Infs")
        
        # Classification logits per graph (Identifying an attack)
        logits_graph = logits_nodes.max(dim=1).values # choose max one
        
        return logits_nodes, logits_graph
    
    
    
    
