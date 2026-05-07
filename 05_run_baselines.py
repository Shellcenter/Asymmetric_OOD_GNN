"""Traditional GCN + MSP baseline under the same Cora leave-out protocol."""

from __future__ import annotations

import argparse
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv

from core_model import evaluate_ood_metrics


ID_CLASSES = (0, 1, 2, 3)
OOD_CLASSES = (4, 5, 6)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def build_leave_out_masks(y: torch.Tensor, train_ratio: float, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    labels = torch.ones_like(y, dtype=torch.long)
    id_mask = torch.zeros_like(y, dtype=torch.bool)
    for cls in ID_CLASSES:
        id_mask |= y == cls
    labels[id_mask] = 0

    ood_mask = torch.zeros_like(y, dtype=torch.bool)
    for cls in OOD_CLASSES:
        ood_mask |= y == cls

    generator = torch.Generator(device=y.device)
    generator.manual_seed(seed)
    id_indices = torch.where(id_mask)[0]
    perm = torch.randperm(id_indices.numel(), generator=generator, device=y.device)
    train_size = int(train_ratio * id_indices.numel())

    train_mask = torch.zeros_like(y, dtype=torch.bool)
    train_mask[id_indices[perm[:train_size]]] = True

    eval_id_mask = torch.zeros_like(y, dtype=torch.bool)
    eval_id_mask[id_indices[perm[train_size:]]] = True

    assert labels[train_mask].sum().item() == 0, "Data leakage: train_mask contains OOD nodes."
    return train_mask, eval_id_mask, ood_mask


class BaselineGCN(torch.nn.Module):
    """A standard two-layer GCN classifier trained only on ID classes."""

    def __init__(self, in_channels: int, hidden_channels: int, num_classes: int, dropout: float = 0.5):
        super().__init__()
        self.dropout = dropout
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x, edge_index)
        h = F.relu(h, inplace=True)
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.conv2(h, edge_index)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GCN MSP baseline for Cora OOD detection.")
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--hidden_channels", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--train_ratio", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = Planetoid(root=os.path.join(args.data_root, "Cora"), name="Cora")
    data = dataset[0].to(device)

    train_mask, eval_id_mask, eval_ood_mask = build_leave_out_masks(data.y, args.train_ratio, args.seed)
    train_mask = train_mask.to(device)
    eval_id_mask = eval_id_mask.to(device)
    eval_ood_mask = eval_ood_mask.to(device)

    model = BaselineGCN(dataset.num_features, args.hidden_channels, len(ID_CLASSES)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print("=== Baseline: GCN + Maximum Softmax Probability ===")
    print(f"Train ID nodes: {int(train_mask.sum())} | Eval ID nodes: {int(eval_id_mask.sum())} | OOD nodes: {int(eval_ood_mask.sum())}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(data.x, data.edge_index)
        loss = F.cross_entropy(logits[train_mask], data.y[train_mask])
        loss.backward()
        optimizer.step()

        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(f"Epoch {epoch:03d}/{args.epochs} | CE Loss: {loss.item():.6f}")

    model.eval()
    if device.type == "cuda":
        torch.cuda.synchronize()
    start_time = time.perf_counter()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        probs = F.softmax(logits, dim=1)
        anomaly_scores = 1.0 - probs.max(dim=1).values
    if device.type == "cuda":
        torch.cuda.synchronize()
    latency_ms = (time.perf_counter() - start_time) * 1000.0

    metrics = evaluate_ood_metrics(anomaly_scores[eval_id_mask], anomaly_scores[eval_ood_mask])
    print(f"AUROC: {metrics['AUROC']:.4f}")
    print(f"AUPR: {metrics['AUPR']:.4f}")
    print(f"FPR@95TPR: {metrics['FPR95']:.4f}")
    print(f"Full-graph latency: {latency_ms:.4f} ms")


if __name__ == "__main__":
    main()
