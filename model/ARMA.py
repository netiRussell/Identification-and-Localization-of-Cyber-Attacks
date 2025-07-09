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
    # TODO: add batch norm and resid. Make sure there are no inf or nan by
    """
    if torch.isnan(x).any() or torch.isinf(x).any():
        raise RuntimeError("NaN/Inf in x → after chebConv3")
        
        and
        
    x = x.nan_to_num(0.0, posinf=1e6, neginf=-1e6)
    """
    def __init__(self, in_channels=2, hidden_channels=32, 
                 num_stacks=3, num_layers=5, dropout=0.5, num_nodes=2848):
        super(GNNArma, self).__init__()
        
        self.num_nodes = num_nodes
        
        # ARMA convolutional graph layers
        self.ARMAconv1 = ARMAConv(in_channels, hidden_channels,
                              num_stacks=num_stacks, num_layers=num_layers)
        
        self.ARMAconv2 = ARMAConv(hidden_channels, hidden_channels,
                              num_stacks=num_stacks, num_layers=num_layers)
        
        self.ARMAconv3 = ARMAConv(hidden_channels, 1,
                              num_stacks=num_stacks, num_layers=num_layers)
        
        # Classifier
        self.classifier = nn.Linear(1, 1)
        
        # Layer Normalizations
        #self.layerNorm = nn.LayerNorm(1)
        
        # Dropouts
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        

    def forward(self, x, edge_index, weights, batch):
        # First ARMA layer
        x = self.ARMAconv1(x, edge_index, weights)
        x = F.relu(x)
        x = self.dropout1(x)
        if not torch.isfinite(x).all():
            raise ValueError("First layer output NaNs/Infs")
        
        # Second ARMA layer
        x = self.ARMAconv2(x, edge_index, weights)
        x = F.relu(x)
        x = self.dropout2(x)
        if not torch.isfinite(x).all():
            raise ValueError("Second layer output NaNs/Infs")
        
        # Third ARMA layer
        x = self.ARMAconv3(x, edge_index, weights)
        #x = self.layerNorm(x)
        x = F.relu(x)
        x = self.dropout3(x)
        if not torch.isfinite(x).all():
            raise ValueError("Third layer output NaNs/Infs")
        
        # from [total_nodes, u] → [batch_size, num_nodes, u]
        x, _ = to_dense_batch(x, batch, max_num_nodes=self.num_nodes)
        
        # Classification logits per node (Localizing an attack)
        logits_nodes = self.classifier(x).squeeze(-1)              # [batch_size, num_nodes]
        if not torch.isfinite(x).all():
            raise ValueError("Classifier output NaNs/Infs")
        
        # Classification logits per graph (Identifying an attack)
        logits_graph = logits_nodes.max(dim=1).values # choose max one
        
        return logits_nodes, logits_graph
    
    
    
    
