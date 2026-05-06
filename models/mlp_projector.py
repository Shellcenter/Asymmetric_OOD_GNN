import torch.nn as nn

class MLPProjector(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=768):
        super(MLPProjector, self).__init__()
        # 将 GNN 的特征映射到 768 维，去对齐 BERT 的 Z_sem
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, h_topo):
        return self.net(h_topo)