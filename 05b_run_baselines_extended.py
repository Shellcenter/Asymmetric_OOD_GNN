"""Extended baselines: GCN trained once, evaluated with Entropy / Energy / MaxLogit."""

from __future__ import annotations

import argparse
import logging
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv

from core_model import compute_free_energy, evaluate_ood_metrics

ID_CLASSES = (0, 1, 2, 3)
OOD_CLASSES = (4, 5, 6)
LOGGER = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def build_leave_out_masks(y, train_ratio, seed):
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
    return train_mask, eval_id_mask, ood_mask


class BaselineGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = F.relu(h, inplace=True)
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.conv2(h, edge_index)


def score_entropy(logits):
    p = F.softmax(logits, dim=1)
    return -(p * torch.log(p + 1e-10)).sum(dim=1)


def score_msp(logits):
    return 1.0 - F.softmax(logits, dim=1).max(dim=1).values


def score_energy(logits, T=1.0):
    return compute_free_energy(logits, temperature=T)


def score_maxlogit(logits):
    return -logits.max(dim=1).values


SCORERS = {
    "msp": score_msp,
    "entropy": score_entropy,
    "energy": lambda logits: score_energy(logits, T=1.0),
    "maxlogit": score_maxlogit,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Extended GCN baselines.")
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--hidden_channels", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--train_ratio", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--methods", type=str, default="entropy,energy,maxlogit")
    parser.add_argument("--output_dir", type=str, default="logs/baseline_extended")
    parser.add_argument("--results_dir", type=str, default="results_data/baseline_extended")
    parser.add_argument("--weights_dir", type=str, default="weights/baseline_extended")
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = Planetoid(root=os.path.join(args.data_root, "Cora"), name="Cora")
    data = dataset[0].to(device)

    train_mask, eval_id_mask, eval_ood_mask = build_leave_out_masks(data.y, args.train_ratio, args.seed)
    train_mask = train_mask.to(device)
    eval_id_mask = eval_id_mask.to(device)
    eval_ood_mask = eval_ood_mask.to(device)

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.weights_dir, exist_ok=True)

    model = BaselineGCN(dataset.num_features, args.hidden_channels, len(ID_CLASSES)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(data.x, data.edge_index)
        loss = F.cross_entropy(logits[train_mask], data.y[train_mask])
        loss.backward()
        optimizer.step()

    # Save weights
    weight_path = os.path.join(args.weights_dir, f"seed{args.seed}.pth")
    torch.save(model.state_dict(), weight_path)

    model.eval()
    methods = [m.strip() for m in args.methods.split(",")]
    results = []

    for method in methods:
        if method not in SCORERS:
            LOGGER.warning("Unknown method %s, skipping", method)
            continue

        if device.type == "cuda":
            torch.cuda.synchronize()
        start_time = time.perf_counter()
        with torch.no_grad():
            logits = model(data.x, data.edge_index)
            scores = SCORERS[method](logits)
        if device.type == "cuda":
            torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        metrics = evaluate_ood_metrics(scores[eval_id_mask], scores[eval_ood_mask])
        LOGGER.info("method=%s seed=%d AUROC=%.4f AUPR=%.4f FPR95=%.4f latency_ms=%.4f",
                     method, args.seed, metrics["AUROC"], metrics["AUPR"], metrics["FPR95"], latency_ms)

        results.append({
            "method": f"GCN-{method.upper()}",
            "seed": args.seed,
            "AUROC": round(metrics["AUROC"], 4),
            "AUPR": round(metrics["AUPR"], 4),
            "FPR95": round(metrics["FPR95"], 4),
            "latency_ms": round(latency_ms, 4),
        })

    log_path = os.path.join(args.output_dir, f"seed{args.seed}.log")
    with open(log_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(f"method={r['method']} seed={r['seed']} "
                    f"AUROC={r['AUROC']} AUPR={r['AUPR']} "
                    f"FPR95={r['FPR95']} latency_ms={r['latency_ms']}\n")

    LOGGER.info("saved=%s", log_path)


if __name__ == "__main__":
    main()
