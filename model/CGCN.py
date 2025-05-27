import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import ChebConv

class GNNArmaTransformer(nn.Module):
    """
      Chebushev GCN model from "Cyberattack_Detection_in_Large-Scale_Smart_Grids_using_Chebyshev_Graph_Convolutional_Networks"

      in_channels: Number of channels/features for each node.
      u: Number of units/neurons in each graph convolutional layer.
      Ks: Spatial kernel size.
      droupout: During training, randomly zeroes some of the elements of the input tensor with this probability
      edge_index: The edge indices representing the connectivity of the graph.
    """
    def __init__(self, in_channels, u, Ks, dropout=0.1 ):
        super(GNNArmaTransformer, self).__init__()
        
        self.chebConv1 = ChebConv(in_channels, u, Ks)
        self.dropout1 = nn.Dropout(dropout)

        self.chebConv2 = ChebConv(in_channels, u, Ks)
        self.dropout2 = nn.Dropout(dropout)

        self.chebConv3 = ChebConv(in_channels, u, Ks)
        self.dropout3 = nn.Dropout(dropout)

        self.chebConv4 = ChebConv(in_channels, u, Ks)
        self.dropout4 = nn.Dropout(dropout)

        self.flatten = nn.Flatten()

        # TODO: make sure that the in_channels is correct
        self.dense = nn.Linear(in_channels, 1) 

    def forward(self, x, edge_index, weights, batch):
        # TODO: add relu between cheb and dropout

        # 4 CGCN layers
        # Relu and Dropout are applied after each one of them
        out = self.chebConv1(x, edge_index, weights)
        out = F.relu(out)
        out = self.dropout1(out)

        out = self.chebConv2(out, edge_index, weights)
        out = F.relu(out)
        out = self.dropout2(out)

        out = self.chebConv3(out, edge_index, weights)
        out = F.relu(out)
        out = self.dropout3(out)

        out = self.chebConv4(out, edge_index, weights)
        out = F.relu(out)
        out = self.dropout4(out)

        # Flatten and Dense layers
        out = self.flatten(out)
        out = self.dense(out)

        # Final sigmoid for a probabilistic binary output
        out = torch.sigmoid(out)
        
        return out
