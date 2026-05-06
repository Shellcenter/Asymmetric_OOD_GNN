import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class MLPDynamicProjector(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=768):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class AsymmetricGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels=128, out_channels=768):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.projector = MLPDynamicProjector(hidden_channels, hidden_channels * 2, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        h_topo = self.conv2(x, edge_index)
        z_topo = self.projector(h_topo)
        return z_topo

class SupConDistillationLoss(nn.Module):
    def __init__(self, margin=2.0):
        super().__init__()
        self.margin = margin

    def forward(self, z_topo, z_sem, labels):
        z_topo = F.normalize(z_topo, p=2, dim=1)
        z_sem = F.normalize(z_sem, p=2, dim=1)
        dist = F.pairwise_distance(z_topo, z_sem, p=2)

        id_mask = (labels == 0)
        loss_id = torch.mean(dist[id_mask] ** 2) if id_mask.sum() > 0 else torch.tensor(0.0, device=z_topo.device)

        ood_mask = (labels == 1)
        loss_ood = torch.mean(F.relu(self.margin - dist[ood_mask]) ** 2) if ood_mask.sum() > 0 else torch.tensor(0.0, device=z_topo.device)

        return loss_id + loss_ood

def compute_free_energy(z_topo, temperature=1.0):
    return -temperature * torch.logsumexp(z_topo / temperature, dim=1)