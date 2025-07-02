import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import ARMAConv

class GNNArma(nn.Module):
    """
    GNN encoder using ARMA filters.
    """
    def __init__(self, in_channels, hidden_channels, out_channels, 
                 num_stacks=3, num_layers=2, dropout=0.3, leakyRelu_rate=0.1):
        super(GNNArma, self).__init__()
        self.conv1 = ARMAConv(in_channels, hidden_channels,
                              num_stacks=num_stacks, num_layers=num_layers)
        
        self.conv2 = ARMAConv(hidden_channels, hidden_channels,
                              num_stacks=num_stacks, num_layers=num_layers)
        
        self.conv3 = ARMAConv(hidden_channels, out_channels,
                              num_stacks=num_stacks, num_layers=num_layers)
        
        # Dropouts:
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        # Activation functions:
        self.leakyRelu = nn.LeakyReLU(leakyRelu_rate)

    def forward(self, x, edge_index, weights):
        # First ARMA layer
        x = self.conv1(x, edge_index, weights)
        x = self.leakyRelu(x)
        x = self.dropout1(x)
        
        # Second ARMA layer
        x = self.conv2(x, edge_index, weights)
        x = self.leakyRelu(x)
        
        # Third ARMA layer
        x = self.conv3(x, edge_index, weights)
        x = self.leakyRelu(x)
        x = self.dropout2(x)
        return x

class GNNArmaTransformer(nn.Module):
    """
    Combines the GNN ARMA encoder with a vanilla Transformer encoder
    and a classification head.
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_stacks=3, 
                 num_layers=2, transformer_heads=4, transformer_layers=2,
                 dropout=0.3, num_nodes=300, leakyRelu_rate=0.1):
        super(GNNArmaTransformer, self).__init__()
        
        # Safe parameters to be reused later
        self.num_nodes = num_nodes
        
        # ARMA
        self.gnn_arma = GNNArma(in_channels, hidden_channels, out_channels, num_stacks, num_layers, dropout, leakyRelu_rate)
        
        # Transformer-Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=out_channels,
            dropout=dropout,
            nhead=transformer_heads,
            dim_feedforward=2048,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=transformer_layers,
        )
        
        # Classifier
        self.classifier = nn.Linear(out_channels, 1)
        
        # Activation functions:
        self.leakyRelu = nn.LeakyReLU(leakyRelu_rate)
        

    def forward(self, x, edge_index, weights, batch):  
        # TODO: to be deleted
        if torch.isnan(x).any() or torch.isnan(weights).any():
            print("Found NaN in inputs")
        
        # Get the batch size
        current_batchSize = int(batch.max().item()) + 1
        
        # ARMA layers
        h = self.gnn_arma(x, edge_index, weights)

        # Reshape so that transformer works per-graph → [B, N, F]
        h = h.view(current_batchSize, self.num_nodes, -1)

        # Transformer Encoder
        h = self.transformer(h)
        h = self.leakyRelu(h)

        # Classification logits per node (Localizing an attack)
        logits_nodes = self.classifier(h)

        # Squeeze => [B, N]
        logits_nodes = logits_nodes.squeeze(-1)

        # Classification logits per graph (Identifying an attack)
        logits_graph = logits_nodes.max(dim=1).values

        return logits_nodes, logits_graph
