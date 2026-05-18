"""Diagnostic: measure Cora OOD performance ceiling across scoring rules.

Runs a single GCN training and evaluates ALL scoring rules on the same
model, revealing whether different scoring rules or auxiliary losses
can actually separate from the MSP baseline.

If all methods cluster around the same AUROC, the dataset ceiling has
been reached and further architecture tweaks won't help.
"""

from __future__ import annotations

import argparse
import logging
import os
import random

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


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)
        self.dropout = dropout

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = F.relu(h, inplace=True)
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.conv2(h, edge_index)


def build_masks(y, train_ratio, seed):
    id_mask = torch.zeros_like(y, dtype=torch.bool)
    for cls in ID_CLASSES:
        id_mask |= y == cls
    ood_mask = torch.zeros_like(y, dtype=torch.bool)
    for cls in OOD_CLASSES:
        ood_mask |= y == cls
    generator = torch.Generator(device=y.device)
    generator.manual_seed(seed)
    id_indices = torch.where(id_mask)[0]
    perm = torch.randperm(id_indices.numel(), generator=generator, device=y.device)
    train_size = int(train_ratio * id_indices.numel())
    train_id = torch.zeros_like(y, dtype=torch.bool)
    eval_id = torch.zeros_like(y, dtype=torch.bool)
    train_id[id_indices[perm[:train_size]]] = True
    eval_id[id_indices[perm[train_size:]]] = True
    return train_id, eval_id, ood_mask


def evaluate_all_scorers(logits, eval_id, eval_ood):
    """Compare many OOD scoring rules on the same logits."""
    results = {}
    probs = F.softmax(logits, dim=1)

    # 1. Maximum Softmax Probability (MSP)
    msp_scores = 1.0 - probs.max(dim=1).values
    results["MSP"] = evaluate_ood_metrics(msp_scores[eval_id], msp_scores[eval_ood])

    # 2. Entropy
    entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=1)
    results["Entropy"] = evaluate_ood_metrics(entropy[eval_id], entropy[eval_ood])

    # 3. Max Logit
    maxlogit = -logits.max(dim=1).values
    results["MaxLogit"] = evaluate_ood_metrics(maxlogit[eval_id], maxlogit[eval_ood])

    # 4. Energy (multiple temperatures)
    for T in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
        energy = compute_free_energy(logits, temperature=T)
        results[f"Energy(T={T})"] = evaluate_ood_metrics(energy[eval_id], energy[eval_ood])

    return results


def main():
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--hidden_channels", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--train_ratio", type=float, default=0.6)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = Planetoid(root=os.path.join(args.data_root, "Cora"), name="Cora")
    data = dataset[0].to(device)

    all_results = {}
    for seed in args.seeds:
        set_seed(seed)
        train_id, eval_id, eval_ood = build_masks(data.y, args.train_ratio, seed)
        train_id = train_id.to(device)
        eval_id = eval_id.to(device)
        eval_ood = eval_ood.to(device)

        model = GCN(dataset.num_features, args.hidden_channels, len(ID_CLASSES)).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        for _ in range(1, args.epochs + 1):
            model.train()
            optimizer.zero_grad()
            loss = F.cross_entropy(model(data.x, data.edge_index)[train_id], data.y[train_id])
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(data.x, data.edge_index)
            results = evaluate_all_scorers(logits, eval_id, eval_ood)

        for name, metrics in results.items():
            all_results.setdefault(name, []).append(metrics)

    # Summary
    LOGGER.info("=" * 60)
    LOGGER.info("OOD SCORING RULE DIAGNOSTICS")
    LOGGER.info(f"Seeds: {args.seeds}, hidden={args.hidden_channels}")
    LOGGER.info("-" * 60)
    LOGGER.info(f"{'Method':<20} {'AUROC':>12} {'AUPR':>12} {'FPR95':>12}")
    LOGGER.info("-" * 60)

    best_auroc = 0.0
    best_method = ""
    for name in sorted(all_results):
        aurocs = [m["AUROC"] for m in all_results[name]]
        auprs = [m["AUPR"] for m in all_results[name]]
        fpr95s = [m["FPR95"] for m in all_results[name]]
        mean_auroc = np.mean(aurocs)
        std_auroc = np.std(aurocs)
        mean_aupr = np.mean(auprs)
        mean_fpr95 = np.mean(fpr95s)
        LOGGER.info(
            f"{name:<20} {mean_auroc:.4f}+/-{std_auroc:.4f} "
            f"{mean_aupr:.4f}       {mean_fpr95:.4f}"
        )
        if mean_auroc > best_auroc:
            best_auroc = mean_auroc
            best_method = name

    LOGGER.info("-" * 60)
    LOGGER.info("Best: %s (AUROC=%.4f)", best_method, best_auroc)

    # Check if all methods are within noise
    all_aurocs = []
    for name in sorted(all_results):
        all_aurocs.extend([m["AUROC"] for m in all_results[name]])
    spread = np.max(all_aurocs) - np.min(all_aurocs)
    LOGGER.info("Total AUROC spread across all methods: %.4f", spread)
    if spread < 0.02:
        LOGGER.warning(
            "AUROC spread < 0.02: Cora OOD performance is SATURATED. "
            "All scoring rules cluster at the same ceiling. "
            "The dataset is too small/simple to differentiate methods. "
            "Consider adding a larger dataset (e.g. OGB-ArXiv) or a "
            "more challenging OOD split."
        )


if __name__ == "__main__":
    main()
