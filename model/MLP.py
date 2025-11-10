import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import Linear
from torch_geometric.utils import to_dense_batch

class MLP(nn.Module):
    """
    GNN encoder using ARMA filters.
    """

    def __init__(self, in_channels=2, num_MLP_layers=3, units=128, 
                 dropout=0.5, num_nodes=300):
        super(MLP, self).__init__()
        
        self.num_nodes = num_nodes
        
        # MLP layers (Dynamic definition)
        self.MLPs = nn.ModuleList()
        for index in range(num_MLP_layers):
            in_channels = in_channels if index == 0 else units
            
            self.MLPs.append( Linear(in_channels, units, bias=True) )
        
        # Layer Normalizations
        self.layerNorms = nn.ModuleList()
        for index in range(num_MLP_layers // 2):
            self.layerNorms.append( nn.LayerNorm(units) )
        
        # Dropouts
        self.dropouts = nn.ModuleList()
        for index in range(num_MLP_layers):
            self.dropouts.append( nn.Dropout(dropout) )
            
        # Classifier
        self.classifier = nn.Linear(units, 1)
        

    def forward(self, x, edge_index, weights, batch):
        # Dynamically go over each MLP with layerNorm added after every 2nd MLP
        for idx, MLPLayer in enumerate(self.MLPs):
            x = MLPLayer(x)
            
            if (idx+1) % 2 == 0:
                x = self.layerNorms[idx//2](x) # every 2nd iteration has this layer
            
            x = F.relu(x)
            x = self.dropouts[idx](x)
            
            if not torch.isfinite(x).all():
                raise ValueError(f"Layer#{idx} output NaNs/Infs") 
        
        # from [total_nodes, u] → [batch_size, num_nodes, u]
        x, mask = to_dense_batch(x, batch, max_num_nodes=self.num_nodes)

        # Classification logits per node (Localizing an attack)
        logits_nodes = self.classifier(x).squeeze(-1)              # [batch_size, num_nodes]
        if not torch.isfinite(logits_nodes).all():
            raise ValueError("Classifier output NaNs/Infs")
        
        # Classification logits per graph (Identifying an attack): max over valid nodes only (mask padded nodes)
        logits_graph = logits_nodes.masked_fill(~mask, float('-inf')).max(dim=1).values
        
        return logits_nodes, logits_graph
    
    
    
    
