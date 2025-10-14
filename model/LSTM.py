import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch

# to be deleted
import sys

# units = 64, layers = 2

class LSTM(nn.Module):
    """
    LSTM
    """

    def __init__(self, in_channels=2, num_LSTM_layers=2, units=64, 
                 dropout=0.5, num_nodes=300):
        super(LSTM, self).__init__()
        
        self.num_nodes = num_nodes
        
        # LSTM layers
        self.lstm = nn.LSTM(in_channels, units, num_layers=num_LSTM_layers, 
                            bias=True, batch_first=True, dropout=dropout)
            
        # Normalization layer
        self.layerNorm = nn.LayerNorm(units)
        
        # Dropout
        self.dropout = nn.Dropout(units)
        
        # Classifier
        self.classifier = nn.Linear(units, 1)
        

    def forward(self, x, edge_index, weights, batch):
        # from [total_nodes, u] → [batch_size, num_nodes, u]
        x, _ = to_dense_batch(x, batch, max_num_nodes=self.num_nodes)
        
        # LSTM main logic
        x = self.lstm(x)
        x = self.layerNorm(x)
        x = self.dropout(x)

        # Classification logits per node (Localizing an attack)
        logits_nodes = self.classifier(x).squeeze(-1)              # [batch_size, num_nodes]
        if not torch.isfinite(logits_nodes).all():
            raise ValueError("Classifier output NaNs/Infs")
        
        # Classification logits per graph (Identifying an attack)
        logits_graph = logits_nodes.max(dim=1).values # choose max one
        
        return logits_nodes, logits_graph
    
    
    
    
