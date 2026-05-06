import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

class GNNBackbone(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNNBackbone, self).__init__()
        # 两层经典的 GCN，提取拓扑结构特征 (h_topo)
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.conv2(x, edge_index)
        return x