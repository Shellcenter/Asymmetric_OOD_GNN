"""Extended baselines: GCN trained once, evaluated with Entropy / Energy / MaxLogit.
Supports both Cora and ArXiv via unified data loader."""

from __future__ import annotations

import argparse, logging, os, random, time
import numpy as np
import torch, torch.nn.functional as F
from torch_geometric.nn import GCNConv

from core_model import compute_free_energy, evaluate_ood_metrics
from data_loader import DatasetName, load_dataset

LOGGER = logging.getLogger(__name__)


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


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


SCORERS = {
    "msp": lambda logits: 1.0 - F.softmax(logits, dim=1).max(dim=1).values,
    "entropy": lambda logits: -(F.softmax(logits, dim=1) * torch.log(F.softmax(logits, dim=1) + 1e-10)).sum(dim=1),
    "energy": lambda logits: compute_free_energy(logits, temperature=1.0),
    "maxlogit": lambda logits: -logits.max(dim=1).values,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Extended GCN baselines.")
    parser.add_argument("--dataset", type=str, default="cora", choices=("cora", "arxiv"))
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--hidden_channels", type=int, default=128)
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
    ds = load_dataset(
        name=args.dataset, data_root=args.data_root,
        train_ratio=args.train_ratio, seed=args.seed, device=device,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.weights_dir, exist_ok=True)

    model = BaselineGCN(ds.num_features, args.hidden_channels, len(ds.id_classes)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for epoch in range(1, args.epochs + 1):
        model.train(); optimizer.zero_grad()
        logits = model(ds.data.x, ds.data.edge_index)
        F.cross_entropy(logits[ds.train_mask], ds.data.y[ds.train_mask]).backward()
        optimizer.step()

    weight_path = os.path.join(args.weights_dir, f"{args.dataset}_seed{args.seed}.pth")
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
            logits = model(ds.data.x, ds.data.edge_index)
            scores = SCORERS[method](logits)
        if device.type == "cuda":
            torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        metrics = evaluate_ood_metrics(scores[ds.eval_id_mask], scores[ds.eval_ood_mask])
        LOGGER.info("dataset=%s method=%s seed=%d AUROC=%.4f AUPR=%.4f FPR95=%.4f latency_ms=%.4f",
                     args.dataset, method, args.seed, metrics["AUROC"], metrics["AUPR"], metrics["FPR95"], latency_ms)
        results.append({
            "dataset": args.dataset, "method": f"GCN-{method.upper()}", "seed": args.seed,
            "AUROC": round(metrics["AUROC"], 4), "AUPR": round(metrics["AUPR"], 4),
            "FPR95": round(metrics["FPR95"], 4), "latency_ms": round(latency_ms, 4),
        })

    log_path = os.path.join(args.output_dir, f"{args.dataset}_seed{args.seed}.log")
    with open(log_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(f"dataset={r['dataset']} method={r['method']} seed={r['seed']} "
                    f"AUROC={r['AUROC']} AUPR={r['AUPR']} FPR95={r['FPR95']} latency_ms={r['latency_ms']}\n")
    LOGGER.info("saved=%s", log_path)


if __name__ == "__main__":
    main()
