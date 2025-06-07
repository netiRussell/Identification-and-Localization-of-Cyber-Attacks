import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import ARMAConv, GraphNorm

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
                 num_stacks=3, num_layers=5, dropout=0.1):
        super(GNNArma, self).__init__()
        
        self.ARMAconv1 = ARMAConv(in_channels, hidden_channels,
                              num_stacks=num_stacks, num_layers=num_layers)
        self.gn1 = GraphNorm(hidden_channels)
        self.dropout1 = nn.Dropout(dropout)
        
        self.ARMAconv2 = ARMAConv(hidden_channels, hidden_channels,
                              num_stacks=num_stacks, num_layers=num_layers)
        self.gn2 = GraphNorm(hidden_channels)
        self.dropout1 = nn.Dropout(dropout)
        
        self.ARMAconv3 = ARMAConv(hidden_channels, 1,
                              num_stacks=num_stacks, num_layers=num_layers)
        self.gn3 = GraphNorm(hidden_channels)
        self.dropout3 = nn.Dropout(dropout)
        
        
        # Classifier
        self.classifier = nn.Linear(1, 1)

    def forward(self, x, edge_index, weights):
        # First ARMA layer
        x = self.ARMAconv1(x, edge_index, weights)
        x = self.gn1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        # Second ARMA layer
        x = self.ARMAconv2(x, edge_index, weights)
        x = self.gn2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        # Third ARMA layer
        x = self.ARMAconv3(x, edge_index, weights)
        x = self.gn3(x)
        x = F.relu(x)
        x = self.dropout3(x)
        
        # Classification logits per node (Localizing an attack)
        logits_nodes = self.classifier(x).squeeze(-1)              # [num_nodes]
        
        # Classification logits per graph (Identifying an attack)
        logits_graph = torch.max(logits_nodes) # choose max one
        
        return logits_nodes, logits_graph
