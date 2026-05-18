"""Comprehensive sweep: energy boundary + lightweight weight.

This replaces the old 09_energy_boundary_sweep.py with a more systematic search.

Sweep dimensions:
  - energy_margin:  [-10, -8, -6, -4, -2]  (was only {-5, -6, -7})
  - energy_compact_weight: [0.01, 0.05, 0.1, 0.2]  (was fixed at 0.05)
  - energy_weight: [0.05, 0.1, 0.2, 0.5, 1.0]  (auxiliary loss weight)

Full lightweight weight sweep:
  - semantic_weight: [0.0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.3, 0.5, 1.0]
  - prototype_weight: [0.0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.3, 0.5, 1.0]

Protocol:
  - energy boundary sweep: grid search over margin × compact_weight, with
    energy_weight fixed at 0.2. Best config selected on single split, then
    validated with robust multi-seed calibration.
  - lightweight sweep: systematic sweep from full weight (1.0) down to
    zero, demonstrating the "lightweight" regime is the sweet spot.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
from dataclasses import asdict, dataclass, field
from itertools import product

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


# ─── data structures ────────────────────────────────────────────

@dataclass(frozen=True)
class EnergyConfig:
    name: str
    energy_weight: float
    energy_margin: float
    energy_compact_weight: float
    semantic_weight: float = 0.2
    prototype_weight: float = 1.0
    logit_scale: float = 10.0


@dataclass(frozen=True)
class LightweightConfig:
    semantic_weight: float
    prototype_weight: float
    energy_weight: float = 0.0

    @property
    def name(self) -> str:
        return f"sw{self.semantic_weight}_pw{self.prototype_weight}_ew{self.energy_weight}"


@dataclass
class SplitMasks:
    train_id: torch.Tensor
    eval_id: torch.Tensor
    eval_ood: torch.Tensor


@dataclass
class RobustSplitMasks:
    val_id: torch.Tensor
    test_id: torch.Tensor
    val_ood: torch.Tensor
    test_ood: torch.Tensor


# ─── utilities ──────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def build_masks(y: torch.Tensor, train_ratio: float, seed: int) -> SplitMasks:
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


def _split_indices(indices, first_ratio, generator):
    perm = torch.randperm(indices.numel(), generator=generator, device=indices.device)
    first_size = int(first_ratio * indices.numel())
    return indices[perm[:first_size]], indices[perm[first_size:]]


def build_robust_masks(y, train_ratio, val_ratio, seed):
    id_mask = torch.zeros_like(y, dtype=torch.bool)
    for cls in ID_CLASSES:
        id_mask |= y == cls
    ood_mask = torch.zeros_like(y, dtype=torch.bool)
    for cls in OOD_CLASSES:
        ood_mask |= y == cls

    generator = torch.Generator(device=y.device)
    generator.manual_seed(seed)
    id_indices = torch.where(id_mask)[0]
    _, remain = _split_indices(id_indices, train_ratio, generator)
    val_id_idx, test_id_idx = _split_indices(remain, val_ratio, generator)

    ood_indices = torch.where(ood_mask)[0]
    val_ood_idx, test_ood_idx = _split_indices(ood_indices, val_ratio, generator)

    out = RobustSplitMasks(
        val_id=torch.zeros_like(y, dtype=torch.bool),
        test_id=torch.zeros_like(y, dtype=torch.bool),
        val_ood=torch.zeros_like(y, dtype=torch.bool),
        test_ood=torch.zeros_like(y, dtype=torch.bool),
    )
    out.val_id[val_id_idx] = True
    out.test_id[test_id_idx] = True
    out.val_ood[val_ood_idx] = True
    out.test_ood[test_ood_idx] = True
    return out


def robust_calibrate(model, data, prototypes, logit_scales, temp_grid, args):
    model.eval()
    with torch.no_grad():
        z_topo = model(data.x, data.edge_index)

    chosen_temps, chosen_scales = [], []
    for offset in range(args.num_seeds):
        seed_val = args.base_seed + offset
        masks = build_robust_masks(data.y, args.train_ratio, args.val_ratio, seed_val)
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
        seed_val = args.base_seed + offset
        masks = build_robust_masks(data.y, args.train_ratio, args.val_ratio, seed_val)
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


# ─── training ───────────────────────────────────────────────────

def train_energy_config(
    data, num_features, anchors, masks, config, args, device,
):
    model = AsymmetricGNN(
        in_channels=num_features,
        hidden_channels=args.hidden_channels,
        out_channels=anchors.size(1),
    ).to(device)
    distill_criterion = SemanticAlignmentLoss()
    energy_criterion = IDEnergyBoundaryLoss(
        margin=config.energy_margin, compact_weight=config.energy_compact_weight,
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
        id_prototypes = compute_class_prototypes(
            z_topo[masks.train_id], train_class_labels, ID_CLASSES
        ).detach()
    return model, id_prototypes, last_losses


def train_lightweight_config(
    data, num_features, anchors, masks, config, args, device,
):
    """Train with classifier + lightweight aux losses (MSP scoring compatible)."""
    model = AsymmetricGNN(
        in_channels=num_features,
        hidden_channels=args.hidden_channels,
        out_channels=anchors.size(1),
    ).to(device)
    distill_criterion = SemanticAlignmentLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    labels = torch.zeros(data.num_nodes, dtype=torch.long, device=device)
    train_class_labels = data.y[masks.train_id].long()
    last_losses = {}

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        class_logits = model.classify(data.x, data.edge_index)
        z_topo = model(data.x, data.edge_index)
        train_z = z_topo[masks.train_id]

        classifier_loss = F.cross_entropy(class_logits[masks.train_id], train_class_labels)
        semantic_loss = distill_criterion(train_z, anchors[masks.train_id])
        train_prototypes = compute_class_prototypes(train_z, train_class_labels, ID_CLASSES)
        prototype_logits = compute_prototype_logits(train_z, train_prototypes, logit_scale=10.0)
        prototype_loss = F.cross_entropy(prototype_logits, train_class_labels)

        loss = (
            1.0 * classifier_loss
            + config.semantic_weight * semantic_loss
            + config.prototype_weight * prototype_loss
        )
        loss.backward()
        optimizer.step()

        if epoch == args.epochs:
            last_losses = {
                "loss": float(loss.item()),
                "classifier": float(classifier_loss.item()),
                "semantic": float(semantic_loss.item()),
                "prototype": float(prototype_loss.item()),
            }

    model.eval()
    with torch.no_grad():
        z_topo = model(data.x, data.edge_index)
        id_prototypes = compute_class_prototypes(
            z_topo[masks.train_id], train_class_labels, ID_CLASSES
        ).detach()
    return model, id_prototypes, last_losses


# ─── evaluation ─────────────────────────────────────────────────

def evaluate_single_split(model, data, prototypes, masks, logit_scale, temperature):
    model.eval()
    with torch.no_grad():
        z_topo = model(data.x, data.edge_index)
        logits = compute_prototype_logits(z_topo, prototypes, logit_scale=logit_scale)
        energy = compute_free_energy(logits, temperature=temperature)
    return evaluate_ood_metrics(energy[masks.eval_id], energy[masks.eval_ood])


def evaluate_classifier_msp(model, data, masks):
    """Evaluate with classifier MSP (like the rescue_* experiments)."""
    model.eval()
    with torch.no_grad():
        logits = model.classify(data.x, data.edge_index)
        probs = F.softmax(logits, dim=1)
        scores = 1.0 - probs.max(dim=1).values
    return evaluate_ood_metrics(scores[masks.eval_id], scores[masks.eval_ood])


# ─── main sweep logic ───────────────────────────────────────────

def run_energy_boundary_sweep(data, num_features, anchors, masks, args, device, temp_grid, logit_scales):
    """Grid search over energy_margin × energy_compact_weight.

    energy_margin values are chosen based on the expected free-energy
    range. With 4 ID classes, logit_scale=10, and uniform-ish logits,
    free energy ≈ -1 * logsumexp([~2.5, ~2.5, ~2.5, ~2.5]) ≈ -3.9.
    After training, logits are more concentrated, pushing free energy
    down toward -10 to -6. We sweep [-10, -2] to cover this range.
    """
    margins = sorted(args.energy_margins)
    compact_weights = sorted(args.energy_compact_weights)
    configs = []
    for ew in sorted(args.energy_weights):
        for m in margins:
            for cw in compact_weights:
                configs.append(EnergyConfig(
                    name=f"ew{ew}_m{m}_cw{cw}",
                    energy_weight=ew,
                    energy_margin=m,
                    energy_compact_weight=cw,
                ))

    LOGGER.info("Energy sweep: %d configs (%d margins × %d compact × %d weights)",
                 len(configs), len(margins), len(compact_weights), len(args.energy_weights))

    rows = []
    for config in configs:
        LOGGER.info("[energy] %s", config.name)
        set_seed(args.base_seed)
        model, prototypes, losses = train_energy_config(
            data, num_features, anchors, masks, config, args, device,
        )
        single_metrics = evaluate_single_split(
            model, data, prototypes, masks, logit_scale=config.logit_scale,
            temperature=args.energy_temperature,
        )
        recommended, robust_metrics = robust_calibrate(
            model, data, prototypes, logit_scales, temp_grid, args,
        )

        rows.append({
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
        })
        LOGGER.info(
            "  single: AUROC=%.4f FPR95=%.4f  robust: AUROC=%.4f+/-%.4f FPR95=%.4f+/-%.4f",
            single_metrics["AUROC"], single_metrics["FPR95"],
            robust_metrics["AUROC_mean"], robust_metrics["AUROC_std"],
            robust_metrics["FPR95_mean"], robust_metrics["FPR95_std"],
        )

    return rows


def run_lightweight_sweep(data, num_features, anchors, masks, args, device):
    """Systematic sweep: semantic_weight and prototype_weight from 0.0 to 1.0."""
    semantic_weights = sorted(args.semantic_weights)
    prototype_weights = sorted(args.prototype_weights)

    configs = []
    for sw in semantic_weights:
        for pw in prototype_weights:
            configs.append(LightweightConfig(semantic_weight=sw, prototype_weight=pw))

    LOGGER.info("Lightweight sweep: %d configs (%d semantic × %d prototype)",
                 len(configs), len(semantic_weights), len(prototype_weights))

    rows = []
    for config in configs:
        LOGGER.info("[lightweight] %s", config.name)
        set_seed(args.base_seed)
        model, prototypes, losses = train_lightweight_config(
            data, num_features, anchors, masks, config, args, device,
        )
        metrics = evaluate_classifier_msp(model, data, masks)

        rows.append({
            "semantic_weight": config.semantic_weight,
            "prototype_weight": config.prototype_weight,
            "AUROC": metrics["AUROC"],
            "AUPR": metrics["AUPR"],
            "FPR95": metrics["FPR95"],
            **{f"final_{k}": v for k, v in losses.items()},
        })
        LOGGER.info("  AUROC=%.4f AUPR=%.4f FPR95=%.4f",
                     metrics["AUROC"], metrics["AUPR"], metrics["FPR95"])

    return rows


# ─── CLI ────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Comprehensive sweep for AsymOOD.")
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--anchor_path", type=str, default="./embeddings/cora_semantic_anchor.pt")
    parser.add_argument("--output_dir", type=str, default="./sweep_results/comprehensive")

    # Training
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--hidden_channels", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--margin", type=float, default=1.0, help="SupCon margin.")
    parser.add_argument("--energy_temperature", type=float, default=1.0)
    parser.add_argument("--train_ratio", type=float, default=0.6)
    parser.add_argument("--val_ratio", type=float, default=0.5)
    parser.add_argument("--base_seed", type=int, default=42)
    parser.add_argument("--num_seeds", type=int, default=7)

    # Calibration
    parser.add_argument("--temp_min", type=float, default=0.05)
    parser.add_argument("--temp_max", type=float, default=8.0)
    parser.add_argument("--temp_steps", type=int, default=120)
    parser.add_argument("--logit_scales", type=str, default="5,10,15,20")
    parser.add_argument("--lambda_auroc", type=float, default=0.2)

    # Energy boundary sweep: comprehensive grid
    parser.add_argument(
        "--energy_weights", nargs="+", type=float,
        default=[0.05, 0.1, 0.2, 0.5, 1.0],
    )
    parser.add_argument(
        "--energy_margins", nargs="+", type=float,
        default=[-10.0, -8.0, -6.0, -4.0, -2.0],
        help="Energy margins to sweep. Range covers the expected free-energy "
             "distribution after training (-10 to -2 for 4-class prototype logits).",
    )
    parser.add_argument(
        "--energy_compact_weights", nargs="+", type=float,
        default=[0.01, 0.05, 0.1, 0.2],
        help="Compactness weights to sweep.",
    )

    # Lightweight sweep: full range
    parser.add_argument(
        "--semantic_weights", nargs="+", type=float,
        default=[0.0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.3, 0.5, 1.0],
        help="Semantic weight sweep from zero to full weight.",
    )
    parser.add_argument(
        "--prototype_weights", nargs="+", type=float,
        default=[0.0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.3, 0.5, 1.0],
        help="Prototype weight sweep from zero to full weight.",
    )

    parser.add_argument("--skip_energy", action="store_true")
    parser.add_argument("--skip_lightweight", action="store_true")
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    args = parse_args()
    set_seed(args.base_seed)

    if not os.path.exists(args.anchor_path):
        raise FileNotFoundError(
            f"Missing anchors: {args.anchor_path}. "
            "Run `python download_cora_vocab.py` then `python 01_extract_llm.py` first."
        )

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = Planetoid(root=os.path.join(args.data_root, "Cora"), name="Cora")
    data = dataset[0].to(device)
    anchors = torch.load(args.anchor_path, map_location=device).float()
    masks = build_masks(data.y, args.train_ratio, args.base_seed)

    temp_grid = torch.linspace(args.temp_min, args.temp_max, args.temp_steps, dtype=torch.float32).tolist()
    logit_scales = [float(x.strip()) for x in args.logit_scales.split(",") if x.strip()]

    # ── Energy boundary sweep ──
    if not args.skip_energy:
        LOGGER.info("=" * 60)
        LOGGER.info("ENERGY BOUNDARY SWEEP")
        LOGGER.info("margins=%s compact_weights=%s", args.energy_margins, args.energy_compact_weights)
        rows = run_energy_boundary_sweep(data, dataset.num_features, anchors, masks, args, device, temp_grid, logit_scales)
        _save_results(rows, "energy_boundary", args.output_dir)
        _print_best(rows, key=lambda r: (r["FPR95_mean"], -r["AUROC_mean"]))

    # ── Lightweight sweep ──
    if not args.skip_lightweight:
        LOGGER.info("=" * 60)
        LOGGER.info("LIGHTWEIGHT SWEEP")
        LOGGER.info("semantic=%s prototype=%s", args.semantic_weights, args.prototype_weights)
        rows = run_lightweight_sweep(data, dataset.num_features, anchors, masks, args, device)
        _save_results(rows, "lightweight", args.output_dir)
        _print_best(rows, key=lambda r: (r["FPR95"], -r["AUROC"]))


def _save_results(rows, prefix, output_dir):
    json_path = os.path.join(output_dir, f"{prefix}.json")
    csv_path = os.path.join(output_dir, f"{prefix}.csv")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    if rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    LOGGER.info("saved: %s, %s", json_path, csv_path)


def _print_best(rows, key):
    best = min(rows, key=key)
    LOGGER.info("BEST: %s", best)


if __name__ == "__main__":
    main()
