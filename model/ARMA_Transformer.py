import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import ARMAConv
import math

class PositionalEncoder(nn.Module):
  def __init__( self, d_model: int, max_path_len: int, dropout: float ):
    super().__init__()
    self.d_model = d_model
    self.max_path_len = max_path_len
    self.dropout = nn.Dropout(dropout)

    # Matrix of shape (max_path_len, d_model) to cover all embedding vectors
    pos_enc = torch.zeros(max_path_len, d_model, requires_grad=False)

    # 0 to max_path_len-1 Vector of shape (max_path_len, 1)
    position = torch.arange(0, max_path_len, dtype=torch.float).unsqueeze(1)

    # Division term from the paper
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

    # Apply sin to even positions (every vector, every even value from embedding vector)
    pos_enc[:, 0::2] = torch.sin(position * div_term)

    # Apply cos to odd positions (every vector, every odd value from embedding vector)
    pos_enc[:, 1::2] = torch.cos(position * div_term)

    # Cache the positional encodings, so that it is retrievable via model.state_dict()
    self.register_buffer('pos_enc', pos_enc)
  
  def forward( self, x, node_ids ):
    # Safely wrap positions if node_ids >= max_path_len
    idx = node_ids % self.max_path_len
    pe = self.pos_enc.index_select(0, idx)   # (B*N, d_model)
    return self.dropout(x + pe)
    

class GNNArma(nn.Module):
    """
    GNN encoder using ARMA filters.
    """
    def __init__(self, in_channels, num_ARMA_layers, hidden_channels, d_model_channels, 
                 num_stacks=3, num_layers=2, dropout=0.3, leakyRelu_rate=0.1):
        super(GNNArma, self).__init__()
        
        # ARMA convolutional graph layers (Dynamic definition)
        self.ARMAconvs = nn.ModuleList()
        for index in range(num_ARMA_layers):
            in_channels = in_channels if index == 0 else hidden_channels
            out_channels = d_model_channels if index == num_ARMA_layers-1 else hidden_channels
            self.ARMAconvs.append( ARMAConv(in_channels, out_channels,
                                  num_stacks=num_stacks, num_layers=num_layers) )
        
        # Layer Normalizations
        self.layerNorms = nn.ModuleList()
        for index in range(num_ARMA_layers // 2):
            u = d_model_channels if (index+1)*2 == num_ARMA_layers else hidden_channels
            self.layerNorms.append( nn.LayerNorm(u) )
        
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
    def __init__(self, in_channels, num_ARMA_layers, hidden_channels, d_model_channels, dim_feedforward, num_stacks=3, 
                 num_layers=2, transformer_heads=4, transformer_layers=2,
                 dropout=0.3, num_nodes=300, leakyRelu_rate=0.1):
        super(GNNArmaTransformer, self).__init__()
        
        # Safe parameters to be reused later
        self.num_nodes = num_nodes
        
        # Projecting to d_model
        self.project_to_hidden = nn.Linear(in_channels, hidden_channels)
        
        # Positional Encoder with learnable parameters
        self.node_pos_embed = nn.Embedding(num_nodes, hidden_channels)
        nn.init.normal_(self.node_pos_embed.weight, std=0.02) 
        
        self.pre_arma_dropout = nn.Dropout(dropout)
        
        # ARMA
        self.gnn_arma = GNNArma(hidden_channels, num_ARMA_layers, hidden_channels, d_model_channels, num_stacks, num_layers, dropout, leakyRelu_rate)
        
        """
        # Positional Encoder
        self.posenc = PositionalEncoder(d_model=d_model_channels,
                                        max_path_len=num_nodes,
                                       dropout=0.1) 
        """
        
        # Transformer-Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model_channels,
            dropout=dropout,
            nhead=transformer_heads,
            dim_feedforward=dim_feedforward,
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
        
        # Create a sequence for the Positional Encoder
        node_ids = torch.arange(self.num_nodes, device=x.device).repeat(current_batchSize)
        
        # Project the node features to hidden_channels(d_model in the paper)
        x = self.project_to_hidden(x)
        
        # PE with Residual Connection for stability
        x = x + self.node_pos_embed(node_ids)
        
        # Pre ARMAConv Dropout
        x = self.pre_arma_dropout(x)
        
        # ARMA layers
        h = self.gnn_arma(x, edge_index, weights)
        
        """
        # PE
        node_ids = torch.arange(self.num_nodes, device=x.device).repeat(current_batchSize)  # [0..N-1] for each graph
        h = self.posenc(h, node_ids)
        """
        
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
