"""Compare OOD scoring rules on existing asymmetric GNN checkpoints.

This script does not train or modify model weights. It reloads checkpoints,
reconstructs the same leave-out split from the stored training seed, and reports
which online OOD score is strongest for the current learned representation.
"""

from __future__ import annotations

import argparse
import csv
import glob
import logging
import os
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid

from core_model import (
    AsymmetricGNN,
    compute_free_energy,
    compute_mahalanobis_logits,
    compute_prototype_logits,
    evaluate_ood_metrics,
    fit_mahalanobis_statistics,
)


ID_CLASSES = (0, 1, 2, 3)
OOD_CLASSES = (4, 5, 6)
LOGGER = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_masks(y: torch.Tensor, train_ratio: float, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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


def load_checkpoint(path: str, in_channels: int, device: torch.device) -> tuple[AsymmetricGNN, dict]:
    checkpoint = torch.load(path, map_location=device)
    if "model_state_dict" not in checkpoint:
        raise KeyError(f"{path} is not a phase-2 checkpoint with model_state_dict.")
    model = AsymmetricGNN(
        in_channels=checkpoint.get("in_channels", in_channels),
        hidden_channels=checkpoint.get("hidden_channels", 128),
        out_channels=checkpoint["out_channels"],
        num_classes=checkpoint.get("num_classes", len(ID_CLASSES)),
    ).to(device)
    missing, _ = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    checkpoint["has_classifier"] = not any(key.startswith("classifier_conv.") for key in missing)
    model.eval()
    return model, checkpoint


def aggregate(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=0))


def score_metrics(
    z_topo: torch.Tensor,
    class_logits: torch.Tensor | None,
    y: torch.Tensor,
    train_id: torch.Tensor,
    eval_id: torch.Tensor,
    eval_ood: torch.Tensor,
    prototypes: torch.Tensor,
    logit_scale: float,
    covariance_eps: float,
    has_classifier: bool,
) -> dict[str, dict[str, float]]:
    logits = compute_prototype_logits(z_topo, prototypes, logit_scale=logit_scale)
    probs = F.softmax(logits, dim=1)
    cosine = F.normalize(z_topo, p=2, dim=1) @ F.normalize(prototypes, p=2, dim=1).t()

    train_labels = y[train_id].long()
    means, precision = fit_mahalanobis_statistics(z_topo[train_id], train_labels, ID_CLASSES, covariance_eps)
    mahal_logits = compute_mahalanobis_logits(z_topo, means, precision)

    scores = {
        "prototype_energy": compute_free_energy(logits, temperature=1.0),
        "prototype_msp": 1.0 - probs.max(dim=1).values,
        "negative_max_logit": -logits.max(dim=1).values,
        "cosine_distance": 1.0 - cosine.max(dim=1).values,
        "mahalanobis_energy": compute_free_energy(mahal_logits, temperature=1.0),
        "mahalanobis_min_distance": -mahal_logits.max(dim=1).values,
    }
    if has_classifier and class_logits is not None:
        class_probs = F.softmax(class_logits, dim=1)
        scores.update(
            {
                "classifier_energy": compute_free_energy(class_logits, temperature=1.0),
                "classifier_msp": 1.0 - class_probs.max(dim=1).values,
                "negative_max_classifier_logit": -class_logits.max(dim=1).values,
            }
        )

    return {
        name: evaluate_ood_metrics(score[eval_id], score[eval_ood])
        for name, score in scores.items()
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose OOD scores for existing GNN checkpoints.")
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--weights_glob", type=str, required=True)
    parser.add_argument("--output_csv", type=str, default="./results_data/diagnostics/score_diagnostics.csv")
    parser.add_argument("--covariance_eps", type=float, default=1e-4)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    args = parse_args()
    paths = sorted(glob.glob(args.weights_glob))
    if not paths:
        raise FileNotFoundError(f"No checkpoints matched: {args.weights_glob}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = Planetoid(root=os.path.join(args.data_root, "Cora"), name="Cora")
    data = dataset[0].to(device)

    rows: list[dict[str, object]] = []
    by_method: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for path in paths:
        model, checkpoint = load_checkpoint(path, dataset.num_features, device)
        seed = int(checkpoint.get("seed", 42))
        train_ratio = float(checkpoint.get("train_ratio", 0.6))
        set_seed(seed)
        train_id, eval_id, eval_ood = build_masks(data.y, train_ratio, seed)
        train_id = train_id.to(device)
        eval_id = eval_id.to(device)
        eval_ood = eval_ood.to(device)

        if "id_prototypes" not in checkpoint:
            raise KeyError(f"{path} does not contain id_prototypes.")
        prototypes = checkpoint["id_prototypes"].to(device).float()
        logit_scale = float(checkpoint.get("logit_scale", 10.0))

        with torch.no_grad():
            h_topo = model.encode(data.x, data.edge_index)
            z_topo = model.project(h_topo)
            class_logits = model.classify(data.x, data.edge_index) if checkpoint.get("has_classifier", False) else None
            metrics_by_score = score_metrics(
                z_topo,
                class_logits,
                data.y,
                train_id,
                eval_id,
                eval_ood,
                prototypes,
                logit_scale,
                args.covariance_eps,
                bool(checkpoint.get("has_classifier", False)),
            )

        for method, metrics in metrics_by_score.items():
            row = {
                "checkpoint": path,
                "train_seed": seed,
                "method": method,
                "AUROC": metrics["AUROC"],
                "AUPR": metrics["AUPR"],
                "FPR95": metrics["FPR95"],
            }
            rows.append(row)
            for metric_name, value in metrics.items():
                by_method[method][metric_name].append(value)

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    with open(args.output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["checkpoint", "train_seed", "method", "AUROC", "AUPR", "FPR95"])
        writer.writeheader()
        writer.writerows(rows)

    LOGGER.info("saved=%s", args.output_csv)
    for method in sorted(by_method):
        auroc_mean, auroc_std = aggregate(by_method[method]["AUROC"])
        aupr_mean, aupr_std = aggregate(by_method[method]["AUPR"])
        fpr_mean, fpr_std = aggregate(by_method[method]["FPR95"])
        LOGGER.info(
            "%s AUROC=%.4f+/-%.4f AUPR=%.4f+/-%.4f FPR95=%.4f+/-%.4f",
            method,
            auroc_mean,
            auroc_std,
            aupr_mean,
            aupr_std,
            fpr_mean,
            fpr_std,
        )


if __name__ == "__main__":
    main()
