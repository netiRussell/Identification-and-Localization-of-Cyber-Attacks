import torch
import torch.nn as nn 
from torch.nn import Conv1d, Linear, GroupNorm
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch

# to be deleted
import sys

class CNN(nn.Module):
    """
    CNN.
    """

    def __init__(self, in_channels=2, num_CNN_layers=3, Ks=3, units=64, padding="same", dropout=0.5, num_nodes=300):
        super(CNN, self).__init__()
        self.in_channels = in_channels
        self.num_nodes = num_nodes
        
        # CNN layers (Dynamic definition)
        self.CNNconvs = nn.ModuleList()
        for index in range(num_CNN_layers):
            in_channels = in_channels if index == 0 else units
            self.CNNconvs.append( Conv1d(in_channels, units,
                                  kernel_size=Ks, padding=padding) )
        
        # Group Normalizations
        self.groupNorms = nn.ModuleList()
        for index in range(num_CNN_layers // 2):
            self.groupNorms.append( GroupNorm(int(units/8), units) )
        
        # Dropouts
        self.dropouts = nn.ModuleList()
        for index in range(num_CNN_layers):
            self.dropouts.append( nn.Dropout(dropout) )
        
        # Classifier
        self.classifier = Linear(units, 1)
        

    def forward(self, x, edge_index, weights, batch):
        # x: [total_nodes, in_channels] -> [B, N, C]
        x, _ = to_dense_batch(x, batch, max_num_nodes=self.num_nodes)
        
        # For Conv1d, go to [B, C, N]
        x = x.permute(0, 2, 1).contiguous()
        
        # Dynamically go over each CNN with layerNorm added after every 2nd CNN
        for idx, CNNLayer in enumerate(self.CNNconvs):
            x = CNNLayer(x)
            
            
            if (idx+1) % 2 == 0:
                x = self.groupNorms[idx//2](x) # every 2nd iteration has this layer
            
            
            x = F.relu(x)
            x = self.dropouts[idx](x)
            
            if not torch.isfinite(x).all():
                raise ValueError(f"Layer#{idx} output NaNs/Infs")
        
        # Back to [B, N, C]
        x = x.permute(0, 2, 1).contiguous()
        
        # Classification logits per node (Localizing an attack)
        logits_nodes = self.classifier(x).squeeze(-1)              # [batch_size, num_nodes]
        if not torch.isfinite(logits_nodes).all():
            raise ValueError("Classifier output NaNs/Infs")
        
        # Classification logits per graph (Identifying an attack)
        logits_graph = logits_nodes.max(dim=1).values # choose max one
        
        return logits_nodes, logits_graph
    
    
    
    
