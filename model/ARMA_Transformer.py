import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import ARMAConv

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
    def __init__(self, in_channels, hidden_channels, out_channels, 
                 num_stacks=3, num_layers=2):
        super(GNNArma, self).__init__()
        self.conv1 = ARMAConv(in_channels, hidden_channels,
                              num_stacks=num_stacks, num_layers=num_layers)
        
        self.conv2 = ARMAConv(hidden_channels, hidden_channels,
                              num_stacks=num_stacks, num_layers=num_layers)
        
        self.conv3 = ARMAConv(hidden_channels, out_channels,
                              num_stacks=num_stacks, num_layers=num_layers)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x, edge_index, weights):
        # First ARMA layer
        x = self.conv1(x, edge_index, weights)
        x = F.relu(x)
        
        # Second ARMA layer
        x = self.conv2(x, edge_index, weights)
        x = F.relu(x)
        
        # Third ARMA layer
        x = self.conv3(x, edge_index, weights)
        x = self.bn(x)
        x = F.relu(x)
        return x

class GNNArmaTransformer(nn.Module):
    """
    Combines the GNN ARMA encoder with a vanilla Transformer encoder
    and a classification head.
    
    out_channels = output features dimension for GNNArma
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_stacks=3, 
                 num_layers=2, transformer_heads=4, transformer_layers=2):
        super(GNNArmaTransformer, self).__init__()
        
        self.gnn_arma = GNNArma(in_channels, hidden_channels, out_channels, num_stacks, num_layers)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=out_channels,
            nhead=transformer_heads,
            dim_feedforward=2048,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=transformer_layers
        )
        
        # Classifier
        self.classifier = nn.Linear(out_channels, 2)
        

    def forward(self, x, edge_index, weights, batch):
        # 1) GNN ARMA encoding
        out = self.gnn_arma(x, edge_index, weights)              # [num_nodes, out_channels]
        
        # 2) reshape to [B, N, F] so transformer works per-graph
        B = int(batch.max().item()) + 1
        N = out.size(0) // B
        out = out.view(B, N, -1)                  # [B, N, F]
        out = self.transformer(out)               # [B, N, F]
        out = out.view(-1, out.size(-1))          # [B * N, F]
        
        # 3) Classification logits per node
        logits = self.classifier(out)                  # [num_nodes, 2]
        
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
