import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
from sklearn.metrics import roc_auc_score


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class BaselineGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes):
        super(BaselineGCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x


def main():
    parser = argparse.ArgumentParser(description="Baseline OOD Detection (MSP)")
    parser.add_argument('--dataset', type=str, default='Cora')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    raw_dataset = Planetoid(root=f'./data/{args.dataset}', name=args.dataset)
    data = raw_dataset[0].to(device)

    id_classes = [0, 1, 2, 3]
    labels = torch.ones(data.num_nodes, dtype=torch.long, device=device)
    for c in id_classes:
        labels[data.y == c] = 0

    id_mask = (labels == 0)
    ood_mask = (labels == 1)

    id_indices = torch.where(id_mask)[0]
    perm = torch.randperm(id_indices.size(0))
    train_size = int(len(id_indices) * 0.6)

    train_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
    train_mask[id_indices[perm[:train_size]]] = True

    test_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
    test_mask[id_indices[perm[train_size:]]] = True
    test_mask[ood_mask] = True

    model = BaselineGCN(raw_dataset.num_features, 64, len(id_classes)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    model.train()
    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[train_mask], data.y[train_mask])
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        torch.cuda.synchronize()
        start_time = time.time()

        logits = model(data.x, data.edge_index)
        probs = F.softmax(logits, dim=1)
        msp_scores = 1.0 - probs.max(dim=1)[0]

        torch.cuda.synchronize()
        end_time = time.time()

    test_ind_idx = torch.where(test_mask & id_mask)[0]
    test_ood_idx = torch.where(test_mask & ood_mask)[0]

    y_true = torch.cat(
        [torch.zeros_like(msp_scores[test_ind_idx]), torch.ones_like(msp_scores[test_ood_idx])]).cpu().numpy()
    y_scores = torch.cat([msp_scores[test_ind_idx], msp_scores[test_ood_idx]]).cpu().numpy()

    auroc = roc_auc_score(y_true, y_scores)
    latency_ms = (end_time - start_time) * 1000

    print(f"--- Baseline: GCN (MSP) on {args.dataset} ---")
    print(f"AUROC: {auroc:.4f}")
    print(f"Total Latency: {latency_ms:.4f} ms")


if __name__ == "__main__":
    main()