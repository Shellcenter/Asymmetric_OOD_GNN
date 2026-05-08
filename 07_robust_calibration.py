"""Robust calibration via multi-seed temperature + logit-scale search.

This script performs a stable post-hoc calibration procedure:
1) Split held-out nodes into validation/test by multiple random seeds.
2) Jointly search (temperature, logit_scale) on each validation split.
3) Aggregate the chosen parameters across seeds.
4) Evaluate the aggregated parameter pair on test splits.
"""

from __future__ import annotations

import argparse
import os
import random
from dataclasses import dataclass

import numpy as np
import torch
from torch_geometric.datasets import Planetoid

from core_model import AsymmetricGNN, compute_free_energy, compute_prototype_logits, evaluate_ood_metrics


ID_CLASSES = (0, 1, 2, 3)
OOD_CLASSES = (4, 5, 6)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class SplitMasks:
    train_id: torch.Tensor
    val_id: torch.Tensor
    test_id: torch.Tensor
    val_ood: torch.Tensor
    test_ood: torch.Tensor


def _split_indices(indices: torch.Tensor, first_ratio: float, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    perm = torch.randperm(indices.numel(), generator=generator, device=indices.device)
    first_size = int(first_ratio * indices.numel())
    return indices[perm[:first_size]], indices[perm[first_size:]]


def build_protocol_masks(y: torch.Tensor, train_ratio: float, val_ratio: float, seed: int) -> SplitMasks:
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

    masks = SplitMasks(
        train_id=torch.zeros_like(y, dtype=torch.bool),
        val_id=torch.zeros_like(y, dtype=torch.bool),
        test_id=torch.zeros_like(y, dtype=torch.bool),
        val_ood=torch.zeros_like(y, dtype=torch.bool),
        test_ood=torch.zeros_like(y, dtype=torch.bool),
    )
    masks.train_id[train_id_idx] = True
    masks.val_id[val_id_idx] = True
    masks.test_id[test_id_idx] = True
    masks.val_ood[val_ood_idx] = True
    masks.test_ood[test_ood_idx] = True
    return masks


def load_model(checkpoint_path: str, in_channels: int, device: torch.device) -> tuple[AsymmetricGNN, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" not in checkpoint:
        raise KeyError("Expected checkpoint with `model_state_dict` and prototype metadata.")
    if "id_prototypes" not in checkpoint:
        raise KeyError("Checkpoint lacks `id_prototypes`. Re-run `python 02_train_distill.py`.")

    model = AsymmetricGNN(
        in_channels=checkpoint.get("in_channels", in_channels),
        hidden_channels=checkpoint.get("hidden_channels", 128),
        out_channels=checkpoint["out_channels"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def parse_float_list(raw: str) -> list[float]:
    values = [float(x.strip()) for x in raw.split(",") if x.strip()]
    if not values:
        raise ValueError("Received empty float list.")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Robust free-energy calibration for asymmetric OOD detection.")
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--weights_path", type=str, default="./weights/cora_gnn.pth")
    parser.add_argument("--base_seed", type=int, default=42)
    parser.add_argument("--num_seeds", type=int, default=7)
    parser.add_argument("--train_ratio", type=float, default=0.6)
    parser.add_argument("--val_ratio", type=float, default=0.5)
    parser.add_argument("--temp_min", type=float, default=0.1)
    parser.add_argument("--temp_max", type=float, default=5.0)
    parser.add_argument("--temp_steps", type=int, default=50)
    parser.add_argument("--logit_scales", type=str, default="5,10,15,20")
    parser.add_argument("--lambda_auroc", type=float, default=0.2, help="score = FPR95 + lambda*(1-AUROC)")
    parser.add_argument("--save_path", type=str, default="./weights/cora_robust_calibration.pt")
    return parser.parse_args()


def aggregate_stats(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=0))


def main() -> None:
    args = parse_args()
    set_seed(args.base_seed)

    if not os.path.exists(args.weights_path):
        raise FileNotFoundError(f"Missing checkpoint: {args.weights_path}")
    if args.num_seeds <= 0:
        raise ValueError("num_seeds must be positive.")
    if args.temp_steps < 2:
        raise ValueError("temp_steps must be >= 2.")

    candidate_scales = parse_float_list(args.logit_scales)
    temp_grid = torch.linspace(args.temp_min, args.temp_max, args.temp_steps, dtype=torch.float32).tolist()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = Planetoid(root=os.path.join(args.data_root, "Cora"), name="Cora")
    data = dataset[0].to(device)

    model, checkpoint = load_model(args.weights_path, dataset.num_features, device)
    id_prototypes = checkpoint["id_prototypes"].to(device).float()

    with torch.no_grad():
        z_topo = model(data.x, data.edge_index)

    seed_results: list[dict] = []
    chosen_temps: list[float] = []
    chosen_scales: list[float] = []

    for offset in range(args.num_seeds):
        split_seed = args.base_seed + offset
        masks = build_protocol_masks(data.y, args.train_ratio, args.val_ratio, split_seed)

        best_score = None
        best_pair = None
        best_val_metrics = None

        for scale in candidate_scales:
            logits = compute_prototype_logits(z_topo, id_prototypes, logit_scale=scale)
            for temp in temp_grid:
                energy = compute_free_energy(logits, temperature=temp)
                val_metrics = evaluate_ood_metrics(energy[masks.val_id], energy[masks.val_ood])
                robust_score = val_metrics["FPR95"] + args.lambda_auroc * (1.0 - val_metrics["AUROC"])
                if best_score is None or robust_score < best_score:
                    best_score = robust_score
                    best_pair = (scale, temp)
                    best_val_metrics = val_metrics

        chosen_scale, chosen_temp = best_pair
        chosen_scales.append(chosen_scale)
        chosen_temps.append(chosen_temp)

        logits_best = compute_prototype_logits(z_topo, id_prototypes, logit_scale=chosen_scale)
        energy_best = compute_free_energy(logits_best, temperature=chosen_temp)
        test_metrics = evaluate_ood_metrics(energy_best[masks.test_id], energy_best[masks.test_ood])

        seed_results.append(
            {
                "split_seed": split_seed,
                "chosen_logit_scale": float(chosen_scale),
                "chosen_temperature": float(chosen_temp),
                "val_metrics": best_val_metrics,
                "test_metrics": test_metrics,
            }
        )

    # Use robust aggregate settings for a final pass over all test splits.
    recommended_scale = float(np.median(np.asarray(chosen_scales, dtype=np.float64)))
    recommended_temperature = float(np.median(np.asarray(chosen_temps, dtype=np.float64)))

    final_test_aurocs: list[float] = []
    final_test_auprs: list[float] = []
    final_test_fpr95s: list[float] = []

    for offset in range(args.num_seeds):
        split_seed = args.base_seed + offset
        masks = build_protocol_masks(data.y, args.train_ratio, args.val_ratio, split_seed)
        logits = compute_prototype_logits(z_topo, id_prototypes, logit_scale=recommended_scale)
        energy = compute_free_energy(logits, temperature=recommended_temperature)
        metrics = evaluate_ood_metrics(energy[masks.test_id], energy[masks.test_ood])
        final_test_aurocs.append(metrics["AUROC"])
        final_test_auprs.append(metrics["AUPR"])
        final_test_fpr95s.append(metrics["FPR95"])

    auroc_mean, auroc_std = aggregate_stats(final_test_aurocs)
    aupr_mean, aupr_std = aggregate_stats(final_test_auprs)
    fpr95_mean, fpr95_std = aggregate_stats(final_test_fpr95s)

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    torch.save(
        {
            "recommended_temperature": recommended_temperature,
            "recommended_logit_scale": recommended_scale,
            "num_seeds": args.num_seeds,
            "base_seed": args.base_seed,
            "lambda_auroc": args.lambda_auroc,
            "seed_results": seed_results,
            "aggregate_test_metrics": {
                "AUROC_mean": auroc_mean,
                "AUROC_std": auroc_std,
                "AUPR_mean": aupr_mean,
                "AUPR_std": aupr_std,
                "FPR95_mean": fpr95_mean,
                "FPR95_std": fpr95_std,
            },
            "search_space": {
                "logit_scales": candidate_scales,
                "temp_min": args.temp_min,
                "temp_max": args.temp_max,
                "temp_steps": args.temp_steps,
            },
            "weights_path": args.weights_path,
        },
        args.save_path,
    )

    print("=== Phase 7: Robust Multi-Seed Calibration ===")
    print(f"Seeds: {args.base_seed} .. {args.base_seed + args.num_seeds - 1}")
    print(f"Recommended logit_scale (median): {recommended_scale:.4f}")
    print(f"Recommended temperature (median): {recommended_temperature:.4f}")
    print(
        "Aggregate test metrics | "
        f"AUROC: {auroc_mean:.4f} ± {auroc_std:.4f}, "
        f"AUPR: {aupr_mean:.4f} ± {aupr_std:.4f}, "
        f"FPR@95TPR: {fpr95_mean:.4f} ± {fpr95_std:.4f}"
    )
    print(f"Saved robust calibration artifact to: {args.save_path}")


if __name__ == "__main__":
    main()
