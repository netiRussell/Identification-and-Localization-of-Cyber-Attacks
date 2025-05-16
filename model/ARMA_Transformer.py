import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import ARMAConv

class GNNArmaEncoder(nn.Module):
    """
    GNN encoder using ARMA filters.
    """
    def __init__(self, in_channels, hidden_channels, out_channels, 
                 num_stacks=3, num_layers=2):
        super(GNNArmaEncoder, self).__init__()
        self.conv1 = ARMAConv(in_channels, hidden_channels,
                              num_stacks=num_stacks, num_layers=num_layers)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        
        self.conv2 = ARMAConv(hidden_channels, out_channels,
                              num_stacks=num_stacks, num_layers=num_layers)
        self.bn2 = nn.BatchNorm1d(out_channels)

    def forward(self, x, edge_index):
        # First ARMA layer
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        
        # Second ARMA layer
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        return x

class GNNArmaTransformer(nn.Module):
    """
    Combines the GNN ARMA encoder with a vanilla Transformer encoder
    and a classification head.
    """
    def __init__(self, in_channels, hidden_channels, gnn_out_channels,
                 transformer_heads=4, transformer_layers=2):
        super(GNNArmaTransformer, self).__init__()
        self.encoder = GNNArmaEncoder(in_channels, hidden_channels, gnn_out_channels)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=gnn_out_channels,
            nhead=transformer_heads,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=transformer_layers
        )
        
        # Binary classifier (attack vs. safe)
        self.classifier = nn.Linear(gnn_out_channels, 2)

    def forward(self, x, edge_index):
        # 1) GNN ARMA encoding
        h = self.encoder(x, edge_index)              # [num_nodes, feat_dim]
        
        # 2) Prep for transformer: [seq_len, batch_size=1, feat_dim]
        h = h.unsqueeze(1)
        h = self.transformer(h)
        h = h.squeeze(1)                             # [num_nodes, feat_dim]
        
        # 3) Classification logits per node
        logits = self.classifier(h)                  # [num_nodes, 2]
        return logits

def predict_attacked_buses(model, data, threshold=0.5):
    """
    Runs model inference on a single PyG Data object and
    returns a list of node indices predicted as attacked.
    """
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)     # [num_nodes, 2]
        probs = F.softmax(logits, dim=1)[:, 1]       # P(attack)
        attacked_mask = probs > threshold
        attacked_ids = attacked_mask.nonzero(as_tuple=False).view(-1).tolist()
    return attacked_ids

# Example usage:
# model = GNNArmaTransformer(in_channels=F_in, hidden_channels=64, gnn_out_channels=128)
# logits = model(data.x, data.edge_index)
# attacked = predict_attacked_buses(model, data)
