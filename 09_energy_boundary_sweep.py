"""Sweep ID-only energy-boundary training configurations.

This script avoids accidental checkpoint overwrites by training each setting
into a separate file, then evaluating prototype-energy OOD performance under a
multi-seed robust calibration protocol.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid

from core_model import (
    AsymmetricGNN,
    IDEnergyBoundaryLoss,
    SemanticAlignmentLoss,
    compute_class_prototypes,
    compute_free_energy,
    compute_prototype_logits,
    evaluate_ood_metrics,
)


ID_CLASSES = (0, 1, 2, 3)
OOD_CLASSES = (4, 5, 6)
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class EnergyConfig:
    """Energy-boundary hyperparameter configuration."""

    name: str
    energy_weight: float
    energy_margin: float
    energy_compact_weight: float
    semantic_weight: float = 0.2
    prototype_weight: float = 1.0
    logit_scale: float = 10.0


@dataclass
class SplitMasks:
    """Boolean masks for fixed train/evaluation splits."""

    train_id: torch.Tensor
    eval_id: torch.Tensor
    eval_ood: torch.Tensor


@dataclass
class RobustSplitMasks:
    """Boolean masks for robust calibration splits."""

    val_id: torch.Tensor
    test_id: torch.Tensor
    val_ood: torch.Tensor
    test_ood: torch.Tensor


def set_seed(seed: int) -> None:
    """Set random seeds for sweep reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def parse_config(raw: str) -> EnergyConfig:
    """Parse name:energy_weight:energy_margin[:compact_weight]."""

    parts = raw.split(":")
    if len(parts) not in (3, 4):
        raise ValueError(f"Invalid config `{raw}`. Expected name:weight:margin[:compact].")
    compact = float(parts[3]) if len(parts) == 4 else 0.05
    return EnergyConfig(
        name=parts[0],
        energy_weight=float(parts[1]),
        energy_margin=float(parts[2]),
        energy_compact_weight=compact,
    )


def build_masks(y: torch.Tensor, train_ratio: float, seed: int) -> SplitMasks:
    """Build the fixed train/evaluation split."""
    id_mask = torch.zeros_like(y, dtype=torch.bool)
    for cls in ID_CLASSES:
        id_mask |= y == cls

    ood_mask = torch.zeros_like(y, dtype=torch.bool)
    for cls in OOD_CLASSES:
        ood_mask |= y == cls

    labels = torch.ones_like(y, dtype=torch.long)
    labels[id_mask] = 0

    generator = torch.Generator(device=y.device)
    generator.manual_seed(seed)
    id_indices = torch.where(id_mask)[0]
    perm = torch.randperm(id_indices.numel(), generator=generator, device=y.device)
    train_size = int(train_ratio * id_indices.numel())

    train_id = torch.zeros_like(y, dtype=torch.bool)
    eval_id = torch.zeros_like(y, dtype=torch.bool)
    train_id[id_indices[perm[:train_size]]] = True
    eval_id[id_indices[perm[train_size:]]] = True

    if labels[train_id].sum().item() != 0:
        raise RuntimeError("Training split contains OOD nodes.")
    return SplitMasks(train_id=train_id, eval_id=eval_id, eval_ood=ood_mask)


def _split_indices(indices: torch.Tensor, first_ratio: float, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """Split indices with a deterministic generator."""
    perm = torch.randperm(indices.numel(), generator=generator, device=indices.device)
    first_size = int(first_ratio * indices.numel())
    return indices[perm[:first_size]], indices[perm[first_size:]]


def build_robust_masks(y: torch.Tensor, train_ratio: float, val_ratio: float, seed: int) -> RobustSplitMasks:
    """Build validation and test masks for robust calibration."""
    id_mask = torch.zeros_like(y, dtype=torch.bool)
    for cls in ID_CLASSES:
        id_mask |= y == cls

    ood_mask = torch.zeros_like(y, dtype=torch.bool)
    for cls in OOD_CLASSES:
        ood_mask |= y == cls

    generator = torch.Generator(device=y.device)
    generator.manual_seed(seed)
    id_indices = torch.where(id_mask)[0]
    _, remain_id_idx = _split_indices(id_indices, train_ratio, generator)
    val_id_idx, test_id_idx = _split_indices(remain_id_idx, val_ratio, generator)

    ood_indices = torch.where(ood_mask)[0]
    val_ood_idx, test_ood_idx = _split_indices(ood_indices, val_ratio, generator)

    masks = RobustSplitMasks(
        val_id=torch.zeros_like(y, dtype=torch.bool),
        test_id=torch.zeros_like(y, dtype=torch.bool),
        val_ood=torch.zeros_like(y, dtype=torch.bool),
        test_ood=torch.zeros_like(y, dtype=torch.bool),
    )
    masks.val_id[val_id_idx] = True
    masks.test_id[test_id_idx] = True
    masks.val_ood[val_ood_idx] = True
    masks.test_ood[test_ood_idx] = True
    return masks


def train_one_config(
    data: Data,
    num_features: int,
    anchors: torch.Tensor,
    masks: SplitMasks,
    config: EnergyConfig,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[AsymmetricGNN, torch.Tensor, dict[str, float]]:
    model = AsymmetricGNN(
        in_channels=num_features,
        hidden_channels=args.hidden_channels,
        out_channels=anchors.size(1),
    ).to(device)
    distill_criterion = SemanticAlignmentLoss()
    energy_criterion = IDEnergyBoundaryLoss(
        margin=config.energy_margin,
        compact_weight=config.energy_compact_weight,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    labels = torch.zeros(data.num_nodes, dtype=torch.long, device=device)
    train_class_labels = data.y[masks.train_id].long()
    last_losses = {}

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        z_topo = model(data.x, data.edge_index)
        train_z = z_topo[masks.train_id]

        semantic_loss = distill_criterion(train_z, anchors[masks.train_id])
        train_prototypes = compute_class_prototypes(train_z, train_class_labels, ID_CLASSES)
        prototype_logits = compute_prototype_logits(train_z, train_prototypes, logit_scale=config.logit_scale)
        prototype_loss = F.cross_entropy(prototype_logits, train_class_labels)
        train_energy = compute_free_energy(prototype_logits, temperature=args.energy_temperature)
        energy_loss, boundary_loss, compact_loss = energy_criterion(train_energy)

        loss = (
            config.semantic_weight * semantic_loss
            + config.prototype_weight * prototype_loss
            + config.energy_weight * energy_loss
        )
        loss.backward()
        optimizer.step()

        if epoch == args.epochs:
            last_losses = {
                "loss": float(loss.item()),
                "semantic": float(semantic_loss.item()),
                "prototype": float(prototype_loss.item()),
                "energy_boundary": float(boundary_loss.item()),
                "energy_compact": float(compact_loss.item()),
            }

    model.eval()
    with torch.no_grad():
        z_topo = model(data.x, data.edge_index)
        id_prototypes = compute_class_prototypes(z_topo[masks.train_id], train_class_labels, ID_CLASSES).detach()
    return model, id_prototypes, last_losses


def evaluate_single_split(
    model: AsymmetricGNN,
    data: Data,
    prototypes: torch.Tensor,
    masks: SplitMasks,
    logit_scale: float,
    temperature: float,
) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        z_topo = model(data.x, data.edge_index)
        logits = compute_prototype_logits(z_topo, prototypes, logit_scale=logit_scale)
        energy = compute_free_energy(logits, temperature=temperature)
    return evaluate_ood_metrics(energy[masks.eval_id], energy[masks.eval_ood])


def robust_calibrate(
    model: AsymmetricGNN,
    data: Data,
    prototypes: torch.Tensor,
    logit_scales: list[float],
    temp_grid: list[float],
    args: argparse.Namespace,
) -> tuple[dict[str, float], dict[str, float]]:
    model.eval()
    with torch.no_grad():
        z_topo = model(data.x, data.edge_index)

    chosen_temps = []
    chosen_scales = []
    for offset in range(args.num_seeds):
        seed = args.base_seed + offset
        masks = build_robust_masks(data.y, args.train_ratio, args.val_ratio, seed)
        best_score = None
        best_pair = None
        for scale in logit_scales:
            logits = compute_prototype_logits(z_topo, prototypes, logit_scale=scale)
            for temp in temp_grid:
                energy = compute_free_energy(logits, temperature=temp)
                val_metrics = evaluate_ood_metrics(energy[masks.val_id], energy[masks.val_ood])
                score = val_metrics["FPR95"] + args.lambda_auroc * (1.0 - val_metrics["AUROC"])
                if best_score is None or score < best_score:
                    best_score = score
                    best_pair = (scale, temp)
        chosen_scales.append(best_pair[0])
        chosen_temps.append(best_pair[1])

    recommended = {
        "logit_scale": float(np.median(np.asarray(chosen_scales, dtype=np.float64))),
        "temperature": float(np.median(np.asarray(chosen_temps, dtype=np.float64))),
    }

    aurocs, auprs, fpr95s = [], [], []
    for offset in range(args.num_seeds):
        seed = args.base_seed + offset
        masks = build_robust_masks(data.y, args.train_ratio, args.val_ratio, seed)
        logits = compute_prototype_logits(z_topo, prototypes, logit_scale=recommended["logit_scale"])
        energy = compute_free_energy(logits, temperature=recommended["temperature"])
        metrics = evaluate_ood_metrics(energy[masks.test_id], energy[masks.test_ood])
        aurocs.append(metrics["AUROC"])
        auprs.append(metrics["AUPR"])
        fpr95s.append(metrics["FPR95"])

    aggregate = {
        "AUROC_mean": float(np.mean(aurocs)),
        "AUROC_std": float(np.std(aurocs)),
        "AUPR_mean": float(np.mean(auprs)),
        "AUPR_std": float(np.std(auprs)),
        "FPR95_mean": float(np.mean(fpr95s)),
        "FPR95_std": float(np.std(fpr95s)),
    }
    return recommended, aggregate


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Sweep energy-boundary training settings.")
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--anchor_path", type=str, default="./embeddings/cora_semantic_anchor.pt")
    parser.add_argument("--output_dir", type=str, default="./sweep_results/energy_boundary")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--hidden_channels", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--energy_temperature", type=float, default=1.0)
    parser.add_argument("--train_ratio", type=float, default=0.6)
    parser.add_argument("--val_ratio", type=float, default=0.5)
    parser.add_argument("--base_seed", type=int, default=42)
    parser.add_argument("--num_seeds", type=int, default=7)
    parser.add_argument("--temp_min", type=float, default=0.05)
    parser.add_argument("--temp_max", type=float, default=8.0)
    parser.add_argument("--temp_steps", type=int, default=120)
    parser.add_argument("--logit_scales", type=str, default="5,10,15,20")
    parser.add_argument("--lambda_auroc", type=float, default=0.2)
    parser.add_argument(
        "--configs",
        nargs="+",
        default=[
            "ew02_m-6:0.2:-6.0:0.05",
            "ew01_m-7:0.1:-7.0:0.05",
            "ew03_m-6:0.3:-6.0:0.05",
            "ew05_m-5:0.5:-5.0:0.05",
        ],
        help="Configs as name:energy_weight:energy_margin[:compact_weight].",
    )
    return parser.parse_args()


def main() -> None:
    """Run the energy-boundary sweep."""
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    args = parse_args()
    set_seed(args.base_seed)
    if not os.path.exists(args.anchor_path):
        raise FileNotFoundError(f"Missing anchors: {args.anchor_path}. Run `python download_cora_vocab.py` then `python 01_extract_llm.py` first.")

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = Planetoid(root=os.path.join(args.data_root, "Cora"), name="Cora")
    data = dataset[0].to(device)
    anchors = torch.load(args.anchor_path, map_location=device).float()
    masks = build_masks(data.y, args.train_ratio, args.base_seed)

    configs = [parse_config(raw) for raw in args.configs]
    temp_grid = torch.linspace(args.temp_min, args.temp_max, args.temp_steps, dtype=torch.float32).tolist()
    logit_scales = [float(x.strip()) for x in args.logit_scales.split(",") if x.strip()]

    rows = []
    for config in configs:
        LOGGER.info(
            "sweep_config=%s energy_weight=%.4f energy_margin=%.4f",
            config.name,
            config.energy_weight,
            config.energy_margin,
        )
        set_seed(args.base_seed)
        model, prototypes, losses = train_one_config(data, dataset.num_features, anchors, masks, config, args, device)
        single_metrics = evaluate_single_split(
            model,
            data,
            prototypes,
            masks,
            logit_scale=config.logit_scale,
            temperature=args.energy_temperature,
        )
        recommended, robust_metrics = robust_calibrate(model, data, prototypes, logit_scales, temp_grid, args)

        checkpoint_path = os.path.join(args.output_dir, f"{config.name}.pth")
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "id_prototypes": prototypes.detach().cpu(),
                "in_channels": dataset.num_features,
                "hidden_channels": args.hidden_channels,
                "out_channels": anchors.size(1),
                "id_classes": ID_CLASSES,
                "ood_classes": OOD_CLASSES,
                "seed": args.base_seed,
                "train_ratio": args.train_ratio,
                "config": asdict(config),
                "recommended": recommended,
                "robust_metrics": robust_metrics,
            },
            checkpoint_path,
        )

        row = {
            "name": config.name,
            "energy_weight": config.energy_weight,
            "energy_margin": config.energy_margin,
            "energy_compact_weight": config.energy_compact_weight,
            "single_AUROC": single_metrics["AUROC"],
            "single_AUPR": single_metrics["AUPR"],
            "single_FPR95": single_metrics["FPR95"],
            "recommended_logit_scale": recommended["logit_scale"],
            "recommended_temperature": recommended["temperature"],
            **robust_metrics,
            **{f"final_{k}": v for k, v in losses.items()},
            "checkpoint": checkpoint_path,
        }
        rows.append(row)
        LOGGER.info(
            "%s single_AUROC=%.4f single_AUPR=%.4f single_FPR95=%.4f "
            "robust_AUROC=%.4f+/-%.4f robust_AUPR=%.4f+/-%.4f "
            "robust_FPR95=%.4f+/-%.4f",
            config.name,
            single_metrics["AUROC"],
            single_metrics["AUPR"],
            single_metrics["FPR95"],
            robust_metrics["AUROC_mean"],
            robust_metrics["AUROC_std"],
            robust_metrics["AUPR_mean"],
            robust_metrics["AUPR_std"],
            robust_metrics["FPR95_mean"],
            robust_metrics["FPR95_std"],
        )

    json_path = os.path.join(args.output_dir, "summary.json")
    csv_path = os.path.join(args.output_dir, "summary.csv")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    best = min(rows, key=lambda item: (item["FPR95_mean"], -item["AUROC_mean"]))
    LOGGER.info(
        "best_config=%s AUROC=%.4f+/-%.4f AUPR=%.4f+/-%.4f FPR95=%.4f+/-%.4f",
        best["name"],
        best["AUROC_mean"],
        best["AUROC_std"],
        best["AUPR_mean"],
        best["AUPR_std"],
        best["FPR95_mean"],
        best["FPR95_std"],
    )
    LOGGER.info("saved_json=%s saved_csv=%s", json_path, csv_path)


if __name__ == "__main__":
    main()
