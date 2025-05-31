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
        
        self.conv1 = ARMAConv(in_channels, hidden_channels,
                              num_stacks=num_stacks, num_layers=num_layers)
        self.gn1 = GraphNorm(hidden_channels)
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = ARMAConv(hidden_channels, hidden_channels,
                              num_stacks=num_stacks, num_layers=num_layers)
        self.gn2 = GraphNorm(hidden_channels)
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv3 = ARMAConv(hidden_channels, hidden_channels,
                              num_stacks=num_stacks, num_layers=num_layers)
        self.gn3 = GraphNorm(hidden_channels)
        self.dropout3 = nn.Dropout(dropout)
        
        
        # Classifier
        self.classifier = nn.Linear(hidden_channels, 2)

    def forward(self, x, edge_index, weights):
        # First ARMA layer
        x = self.conv1(x, edge_index, weights)
        x = self.gn1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        # Second ARMA layer
        x = self.conv2(x, edge_index, weights)
        x = self.gn2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        # Third ARMA layer
        x = self.conv3(x, edge_index, weights)
        x = self.gn3(x)
        x = F.relu(x)
        x = self.dropout3(x)
        
        # Classification logits per node
        logits = self.classifier(x)                  # [num_nodes, 2]
        
        return logits


"""
Training Initialization:
    
model = GNNArmaTransformer(
    in_channels=in_feats,
    hidden_channels=config["hidden_channels"],
    out_channels=config["out_channels"],
    num_stacks=config["num_stacks"], 
    num_layers=config["num_layers"],
    transformer_heads=config["transformer_heads"],
    transformer_layers=config["transformer_layers"]
)
"""
