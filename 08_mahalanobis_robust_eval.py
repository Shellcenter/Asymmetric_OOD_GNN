"""Robust multi-seed OOD evaluation with a Mahalanobis energy head.

Protocol guarantees:
- GNN training still uses only ID train nodes (already done in Phase 2).
- Mahalanobis statistics are fit only on ID train nodes of each split.
- No semantic features are used in this online evaluation stage.
"""

from __future__ import annotations

import argparse
import logging
import os
import random
from dataclasses import dataclass

import numpy as np
import torch
from torch_geometric.datasets import Planetoid

from core_model import (
    AsymmetricGNN,
    compute_free_energy,
    compute_mahalanobis_logits,
    evaluate_ood_metrics,
    fit_mahalanobis_statistics,
)


ID_CLASSES = (0, 1, 2, 3)
OOD_CLASSES = (4, 5, 6)
LOGGER = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Set random seeds for evaluation."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class SplitMasks:
    """Boolean masks for Mahalanobis calibration splits."""

    train_id: torch.Tensor
    val_id: torch.Tensor
    test_id: torch.Tensor
    val_ood: torch.Tensor
    test_ood: torch.Tensor


def _split_indices(indices: torch.Tensor, first_ratio: float, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """Split indices with a deterministic generator."""
    perm = torch.randperm(indices.numel(), generator=generator, device=indices.device)
    first_size = int(first_ratio * indices.numel())
    return indices[perm[:first_size]], indices[perm[first_size:]]


def build_protocol_masks(y: torch.Tensor, train_ratio: float, val_ratio: float, seed: int) -> SplitMasks:
    """Build train, validation, and test masks."""
    id_mask = torch.zeros_like(y, dtype=torch.bool)
    for cls in ID_CLASSES:
        id_mask |= y == cls

    ood_mask = torch.zeros_like(y, dtype=torch.bool)
    for cls in OOD_CLASSES:
        ood_mask |= y == cls

    generator = torch.Generator(device=y.device)
    generator.manual_seed(seed)

    id_indices = torch.where(id_mask)[0]
    train_id_idx, remain_id_idx = _split_indices(id_indices, train_ratio, generator)
    val_id_idx, test_id_idx = _split_indices(remain_id_idx, val_ratio, generator)

    ood_indices = torch.where(ood_mask)[0]
    val_ood_idx, test_ood_idx = _split_indices(ood_indices, val_ratio, generator)

    split = SplitMasks(
        train_id=torch.zeros_like(y, dtype=torch.bool),
        val_id=torch.zeros_like(y, dtype=torch.bool),
        test_id=torch.zeros_like(y, dtype=torch.bool),
        val_ood=torch.zeros_like(y, dtype=torch.bool),
        test_ood=torch.zeros_like(y, dtype=torch.bool),
    )
    split.train_id[train_id_idx] = True
    split.val_id[val_id_idx] = True
    split.test_id[test_id_idx] = True
    split.val_ood[val_ood_idx] = True
    split.test_ood[test_ood_idx] = True
    return split


def load_model(weights_path: str, in_channels: int, device: torch.device) -> AsymmetricGNN:
    """Load a distilled GNN checkpoint."""
    checkpoint = torch.load(weights_path, map_location=device)
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        hidden_channels = checkpoint.get("hidden_channels", 128)
        out_channels = checkpoint["out_channels"]
        model_in_channels = checkpoint.get("in_channels", in_channels)
    else:
        state_dict = checkpoint
        hidden_channels = 128
        out_channels = state_dict["projector.net.4.weight"].size(0)
        model_in_channels = in_channels

    model = AsymmetricGNN(
        in_channels=model_in_channels,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        num_classes=len(ID_CLASSES),
    ).to(device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def aggregate(values: list[float]) -> tuple[float, float]:
    """Return mean and population standard deviation."""
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=0))


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Robust Mahalanobis OOD evaluation on Cora.")
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--weights_path", type=str, default="./weights/cora_gnn.pth")
    parser.add_argument("--base_seed", type=int, default=42)
    parser.add_argument("--num_seeds", type=int, default=7)
    parser.add_argument("--train_ratio", type=float, default=0.6)
    parser.add_argument("--val_ratio", type=float, default=0.5)
    parser.add_argument("--temp_min", type=float, default=0.1)
    parser.add_argument("--temp_max", type=float, default=5.0)
    parser.add_argument("--temp_steps", type=int, default=60)
    parser.add_argument("--lambda_auroc", type=float, default=0.2, help="score = FPR95 + lambda*(1-AUROC)")
    parser.add_argument("--covariance_eps", type=float, default=1e-4)
    parser.add_argument("--save_path", type=str, default="./weights/cora_mahalanobis_robust.pt")
    return parser.parse_args()


def main() -> None:
    """Evaluate Mahalanobis OOD scoring over multiple splits."""
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    args = parse_args()
    set_seed(args.base_seed)
    if not os.path.exists(args.weights_path):
        raise FileNotFoundError(f"Missing GNN checkpoint: {args.weights_path}")
    if args.num_seeds <= 0:
        raise ValueError("num_seeds must be positive.")
    if args.temp_steps < 2:
        raise ValueError("temp_steps must be >= 2.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = Planetoid(root=os.path.join(args.data_root, "Cora"), name="Cora")
    data = dataset[0].to(device)
    model = load_model(args.weights_path, dataset.num_features, device)

    with torch.no_grad():
        z_topo = model(data.x, data.edge_index)

    temp_grid = torch.linspace(args.temp_min, args.temp_max, args.temp_steps, dtype=torch.float32).tolist()
    chosen_temps: list[float] = []
    seed_results: list[dict] = []

    final_test_aurocs: list[float] = []
    final_test_auprs: list[float] = []
    final_test_fpr95s: list[float] = []

    for offset in range(args.num_seeds):
        split_seed = args.base_seed + offset
        masks = build_protocol_masks(data.y, args.train_ratio, args.val_ratio, split_seed)
        train_id_labels = data.y[masks.train_id].long()
        means, precision = fit_mahalanobis_statistics(
            z_topo[masks.train_id],
            train_id_labels,
            ID_CLASSES,
            covariance_eps=args.covariance_eps,
        )
        logits = compute_mahalanobis_logits(z_topo, means, precision)

        best_temp = None
        best_val_metrics = None
        best_score = None
        for temp in temp_grid:
            energy = compute_free_energy(logits, temperature=temp)
            val_metrics = evaluate_ood_metrics(energy[masks.val_id], energy[masks.val_ood])
            score = val_metrics["FPR95"] + args.lambda_auroc * (1.0 - val_metrics["AUROC"])
            if best_score is None or score < best_score:
                best_score = score
                best_temp = temp
                best_val_metrics = val_metrics

        chosen_temps.append(best_temp)
        energy_best = compute_free_energy(logits, temperature=best_temp)
        test_metrics = evaluate_ood_metrics(energy_best[masks.test_id], energy_best[masks.test_ood])
        seed_results.append(
            {
                "split_seed": split_seed,
                "selected_temperature": float(best_temp),
                "val_metrics": best_val_metrics,
                "test_metrics": test_metrics,
            }
        )

    recommended_temperature = float(np.median(np.asarray(chosen_temps, dtype=np.float64)))

    for offset in range(args.num_seeds):
        split_seed = args.base_seed + offset
        masks = build_protocol_masks(data.y, args.train_ratio, args.val_ratio, split_seed)
        train_id_labels = data.y[masks.train_id].long()
        means, precision = fit_mahalanobis_statistics(
            z_topo[masks.train_id],
            train_id_labels,
            ID_CLASSES,
            covariance_eps=args.covariance_eps,
        )
        logits = compute_mahalanobis_logits(z_topo, means, precision)
        energy = compute_free_energy(logits, temperature=recommended_temperature)
        test_metrics = evaluate_ood_metrics(energy[masks.test_id], energy[masks.test_ood])
        final_test_aurocs.append(test_metrics["AUROC"])
        final_test_auprs.append(test_metrics["AUPR"])
        final_test_fpr95s.append(test_metrics["FPR95"])

    auroc_mean, auroc_std = aggregate(final_test_aurocs)
    aupr_mean, aupr_std = aggregate(final_test_auprs)
    fpr95_mean, fpr95_std = aggregate(final_test_fpr95s)

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    torch.save(
        {
            "method": "mahalanobis_energy",
            "recommended_temperature": recommended_temperature,
            "num_seeds": args.num_seeds,
            "base_seed": args.base_seed,
            "lambda_auroc": args.lambda_auroc,
            "covariance_eps": args.covariance_eps,
            "seed_results": seed_results,
            "aggregate_test_metrics": {
                "AUROC_mean": auroc_mean,
                "AUROC_std": auroc_std,
                "AUPR_mean": aupr_mean,
                "AUPR_std": aupr_std,
                "FPR95_mean": fpr95_mean,
                "FPR95_std": fpr95_std,
            },
            "weights_path": args.weights_path,
        },
        args.save_path,
    )

    LOGGER.info("Phase 8: Mahalanobis robust evaluation")
    LOGGER.info("seed_range=%d-%d", args.base_seed, args.base_seed + args.num_seeds - 1)
    LOGGER.info("recommended_temperature=%.4f", recommended_temperature)
    LOGGER.info(
        "test AUROC=%.4f+/-%.4f AUPR=%.4f+/-%.4f FPR95=%.4f+/-%.4f",
        auroc_mean,
        auroc_std,
        aupr_mean,
        aupr_std,
        fpr95_mean,
        fpr95_std,
    )
    LOGGER.info("saved=%s", args.save_path)


if __name__ == "__main__":
    main()
