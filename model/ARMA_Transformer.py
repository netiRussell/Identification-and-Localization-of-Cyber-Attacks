import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import ARMAConv

class GNNArma(nn.Module):
    """
    GNN encoder using ARMA filters.
    """
    def __init__(self, in_channels, num_ARMA_layers, hidden_channels, out_channels, 
                 num_stacks=3, num_layers=2, dropout=0.3, leakyRelu_rate=0.1):
        super(GNNArma, self).__init__()
        
        # ARMA convolutional graph layers (Dynamic definition)
        self.ARMAconvs = nn.ModuleList()
        for index in range(num_ARMA_layers):
            in_channels = in_channels if index == 0 else hidden_channels
            out_channels = out_channels if index == num_ARMA_layers-1 else hidden_channels
            
            self.ARMAconvs.append( ARMAConv(in_channels, out_channels,
                                  num_stacks=num_stacks, num_layers=num_layers) )
        
        # Layer Normalizations
        self.layerNorms = nn.ModuleList()
        for index in range(num_ARMA_layers // 2):
            self.layerNorms.append( nn.LayerNorm(out_channels) )
        
        # Dropouts
        self.dropouts = nn.ModuleList()
        for index in range(num_ARMA_layers):
            self.dropouts.append( nn.Dropout(dropout) )
        
        # Activation functions
        self.leakyRelu = nn.LeakyReLU(leakyRelu_rate)

    def forward(self, x, edge_index, weights):
        # Dynamically go over each ArmaConv with layerNorm added after every 2nd ArmaConv
        for idx, ARMALayer in enumerate(self.ARMAconvs):
            x = ARMALayer(x, edge_index, weights)
            
            if (idx+1) % 2 == 0:
                x = self.layerNorms[idx//2](x) # every 2nd iteration has this layer
            
            x = self.leakyRelu(x)
            x = self.dropouts[idx](x)
            
            if not torch.isfinite(x).all():
                raise ValueError(f"Layer#{idx} output NaNs/Infs")
            
        return x

class GNNArmaTransformer(nn.Module):
    """
    Combines the GNN ARMA encoder with a vanilla Transformer encoder
    and a classification head.
    """
    def __init__(self, in_channels, num_ARMA_layers, hidden_channels, d_model_channels, num_stacks=3, 
                 num_layers=2, transformer_heads=4, transformer_layers=2,
                 dropout=0.3, num_nodes=300, leakyRelu_rate=0.1):
        super(GNNArmaTransformer, self).__init__()
        
        # Safe parameters to be reused later
        self.num_nodes = num_nodes
        
        # ARMA
        self.gnn_arma = GNNArma(in_channels, num_ARMA_layers, hidden_channels, d_model_channels, num_stacks, num_layers, dropout, leakyRelu_rate)
        
        # Transformer-Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model_channels,
            dropout=dropout,
            nhead=transformer_heads,
            dim_feedforward=128,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=transformer_layers,
        )
        
        # Classifier
        self.classifier = nn.Linear(d_model_channels, 1)
        
        # Activation functions:
        self.leakyRelu = nn.LeakyReLU(leakyRelu_rate)
        

    def forward(self, x, edge_index, weights, batch):  
        # TODO: to be deleted
        if torch.isnan(x).any() or torch.isnan(weights).any():
            raise RuntimeError("Found NaN in inputs")
        
        # Get the batch size
        current_batchSize = int(batch.max().item()) + 1
        
        # ARMA layers
        h = self.gnn_arma(x, edge_index, weights)
        
        if not torch.isfinite(h).all():
            raise RuntimeError("NaN or inf in ARMA output => add a small GraphNorm/BatchNorm after each ARMAConv")

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
