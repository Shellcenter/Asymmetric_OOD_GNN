"""Semantic Feature Augmentation for Graph OOD Detection.

Concatenates pre-computed text embeddings (sentence-transformer) to
GNN input features. The GNN learns to use both topological and semantic
information for OOD detection, without requiring text at inference time.
"""

from __future__ import annotations

import argparse, logging, os, random, time
import numpy as np
import torch, torch.nn.functional as F
from torch_geometric.nn import GCNConv

from data_loader import DatasetName, load_dataset
from core_model import compute_free_energy, evaluate_ood_metrics

LOGGER = logging.getLogger(__name__)


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class SemanticGCN(torch.nn.Module):
    """3-layer GCN that takes concat(raw_features, semantic_anchor) as input."""
    def __init__(self, in_channels, hidden_channels, num_classes, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.cls  = GCNConv(hidden_channels, num_classes)
        self.dropout = dropout

    def forward(self, x, edge_index):
        h = F.relu(self.conv1(x, edge_index))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.relu(self.conv2(h, edge_index))
        return self.cls(h, edge_index)


def evaluate_all_scorers(logits, eval_id, eval_ood):
    """Return dict of {score_name: {AUROC, AUPR, FPR95}}."""
    probs = F.softmax(logits, dim=1)
    scores = {
        "MSP": 1.0 - probs.max(dim=1).values,
        "Entropy": -(probs * torch.log(probs + 1e-10)).sum(dim=1),
        "Energy(T=1)": compute_free_energy(logits, temperature=1.0),
        "MaxLogit": -logits.max(dim=1).values,
    }
    return {name: evaluate_ood_metrics(s[eval_id], s[eval_ood]) for name, s in scores.items()}


def parse_args():
    p = argparse.ArgumentParser(description="Semantic Feature Augmentation for OOD")
    p.add_argument("--dataset", type=str, default="arxiv", choices=("cora", "arxiv"))
    p.add_argument("--data_root", type=str, default="./data")
    p.add_argument("--anchor_path", type=str, default=None)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--train_ratio", type=float, default=0.6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_semantic", action="store_true",
                   help="Run baseline without semantic features.")
    p.add_argument("--save_dir", type=str, default="./weights/semantic_augmented")
    return p.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── load data & anchors ──
    ds = load_dataset(args.dataset, data_root=args.data_root,
                      train_ratio=args.train_ratio, seed=args.seed, device=device)

    anchor_path = args.anchor_path or ds.semantic_anchor_path
    use_semantic = not args.no_semantic

    if use_semantic and os.path.exists(anchor_path):
        anchors = torch.load(anchor_path, map_location=device).float()
        x_feat = torch.cat([ds.data.x, anchors], dim=1)
        LOGGER.info("Using semantic augmentation: %d -> %d dims",
                     ds.data.x.shape[1], x_feat.shape[1])
    else:
        x_feat = ds.data.x
        if args.no_semantic:
            LOGGER.info("Baseline mode: raw features only (%d dims)", x_feat.shape[1])
        else:
            LOGGER.warning("Anchor not found at %s, falling back to raw features", anchor_path)

    # ── train ──
    model = SemanticGCN(x_feat.shape[1], args.hidden, len(ds.id_classes)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for ep in range(1, args.epochs + 1):
        model.train(); opt.zero_grad()
        logits = model(x_feat, ds.data.edge_index)
        F.cross_entropy(logits[ds.train_mask], ds.data.y[ds.train_mask]).backward()
        opt.step()
        if ep == 1 or ep % 50 == 0:
            LOGGER.info("epoch=%03d/%03d", ep, args.epochs)

    # ── evaluate ──
    model.eval()
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        logits = model(x_feat, ds.data.edge_index)
    if device.type == "cuda":
        torch.cuda.synchronize()
    latency = (time.perf_counter() - t0) * 1000.0

    metrics = evaluate_all_scorers(logits, ds.eval_id_mask, ds.eval_ood_mask)

    LOGGER.info("=" * 50)
    mode = "semantic_augmented" if use_semantic else "baseline"
    LOGGER.info("Results: %s | seed=%d", mode, args.seed)
    for name, m in metrics.items():
        LOGGER.info("  %-15s AUROC=%.4f  AUPR=%.4f  FPR95=%.4f",
                     name, m["AUROC"], m["AUPR"], m["FPR95"])
    LOGGER.info("  latency_ms=%.2f", latency)

    # ── save ──
    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, f"{args.dataset}_{mode}_seed{args.seed}.pth")
    torch.save({
        "model_state_dict": model.state_dict(),
        "in_channels": x_feat.shape[1],
        "hidden_channels": args.hidden,
        "num_classes": len(ds.id_classes),
        "use_semantic": use_semantic,
        "seed": args.seed,
        "metrics": {k: v for k, v in metrics.items()},
    }, save_path)
    LOGGER.info("saved=%s", save_path)

    return metrics


if __name__ == "__main__":
    main()
