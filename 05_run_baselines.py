"""Baseline: standard GCN + MSP OOD detection (Cora + ArXiv)."""

from __future__ import annotations

import argparse
import logging
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from core_model import evaluate_ood_metrics
from data_loader import DatasetName, load_dataset


LOGGER = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class BaselineGCN(torch.nn.Module):
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


def parse_args():
    parser = argparse.ArgumentParser(description="GCN MSP baseline for OOD detection.")
    parser.add_argument("--dataset", type=str, default="cora", choices=("cora", "arxiv"))
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--hidden_channels", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--train_ratio", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = load_dataset(
        name=args.dataset, data_root=args.data_root,
        train_ratio=args.train_ratio, seed=args.seed, device=device,
    )

    model = BaselineGCN(ds.num_features, args.hidden_channels, len(ds.id_classes)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    LOGGER.info("Baseline: GCN + MSP on %s", args.dataset)
    LOGGER.info("train=%d eval_id=%d eval_ood=%d",
                 int(ds.train_mask.sum()), int(ds.eval_id_mask.sum()), int(ds.eval_ood_mask.sum()))

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(ds.data.x, ds.data.edge_index)
        loss = F.cross_entropy(logits[ds.train_mask], ds.data.y[ds.train_mask])
        loss.backward()
        optimizer.step()
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            LOGGER.info("epoch=%03d/%03d ce_loss=%.6f", epoch, args.epochs, loss.item())

    model.eval()
    if device.type == "cuda":
        torch.cuda.synchronize()
    start_time = time.perf_counter()
    with torch.no_grad():
        logits = model(ds.data.x, ds.data.edge_index)
        probs = F.softmax(logits, dim=1)
        anomaly_scores = 1.0 - probs.max(dim=1).values
    if device.type == "cuda":
        torch.cuda.synchronize()
    latency_ms = (time.perf_counter() - start_time) * 1000.0

    metrics = evaluate_ood_metrics(anomaly_scores[ds.eval_id_mask], anomaly_scores[ds.eval_ood_mask])
    LOGGER.info("AUROC=%.4f AUPR=%.4f FPR95=%.4f latency_ms=%.4f",
                 metrics["AUROC"], metrics["AUPR"], metrics["FPR95"], latency_ms)


if __name__ == "__main__":
    main()
