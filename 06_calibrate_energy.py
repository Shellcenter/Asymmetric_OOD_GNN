"""Temperature calibration for prototype-based free-energy OOD scoring.

This script keeps the asymmetric constraint:
- no semantic features are loaded;
- no OOD node is used in model training;
- temperature is tuned on a held-out validation split and reported on test.
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

from core_model import AsymmetricGNN, compute_free_energy, compute_prototype_logits, evaluate_ood_metrics


ID_CLASSES = (0, 1, 2, 3)
OOD_CLASSES = (4, 5, 6)
LOGGER = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Set random seeds for calibration."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class SplitMasks:
    """Boolean masks for calibration splits."""

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

    train_id = torch.zeros_like(y, dtype=torch.bool)
    val_id = torch.zeros_like(y, dtype=torch.bool)
    test_id = torch.zeros_like(y, dtype=torch.bool)
    val_ood = torch.zeros_like(y, dtype=torch.bool)
    test_ood = torch.zeros_like(y, dtype=torch.bool)
    train_id[train_id_idx] = True
    val_id[val_id_idx] = True
    test_id[test_id_idx] = True
    val_ood[val_ood_idx] = True
    test_ood[test_ood_idx] = True
    return SplitMasks(train_id=train_id, val_id=val_id, test_id=test_id, val_ood=val_ood, test_ood=test_ood)


def load_model(checkpoint_path: str, in_channels: int, device: torch.device) -> tuple[AsymmetricGNN, dict]:
    """Load a prototype-calibrated GNN checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" not in checkpoint:
        raise KeyError("Expected checkpoint with `model_state_dict` and prototype metadata.")
    model = AsymmetricGNN(
        in_channels=checkpoint.get("in_channels", in_channels),
        hidden_channels=checkpoint.get("hidden_channels", 128),
        out_channels=checkpoint["out_channels"],
        num_classes=checkpoint.get("num_classes", len(ID_CLASSES)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()
    return model, checkpoint


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Calibrate free-energy temperature on Cora.")
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--weights_path", type=str, default="./weights/cora_gnn.pth")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_ratio", type=float, default=0.6, help="Must match training protocol.")
    parser.add_argument("--val_ratio", type=float, default=0.5, help="Split ratio on held-out ID/OOD.")
    parser.add_argument("--temp_min", type=float, default=0.1)
    parser.add_argument("--temp_max", type=float, default=5.0)
    parser.add_argument("--temp_steps", type=int, default=50)
    parser.add_argument("--save_path", type=str, default="./weights/cora_energy_calibration.pt")
    return parser.parse_args()


def main() -> None:
    """Calibrate free-energy temperature."""
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    args = parse_args()
    set_seed(args.seed)

    if not os.path.exists(args.weights_path):
        raise FileNotFoundError(f"Missing checkpoint: {args.weights_path}. Run `python 02_train_distill.py` first.")
    if args.temp_steps < 2:
        raise ValueError("temp_steps must be >= 2.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = Planetoid(root=os.path.join(args.data_root, "Cora"), name="Cora")
    data = dataset[0].to(device)
    masks = build_protocol_masks(data.y, args.train_ratio, args.val_ratio, args.seed)

    model, checkpoint = load_model(args.weights_path, dataset.num_features, device)
    if "id_prototypes" not in checkpoint:
        raise KeyError("Checkpoint lacks `id_prototypes`. Re-run `python 02_train_distill.py`.")

    id_prototypes = checkpoint["id_prototypes"].to(device).float()
    logit_scale = float(checkpoint.get("logit_scale", 10.0))

    with torch.no_grad():
        z_topo = model(data.x, data.edge_index)
        logits = compute_prototype_logits(z_topo, id_prototypes, logit_scale=logit_scale)

    temp_grid = torch.linspace(args.temp_min, args.temp_max, args.temp_steps, dtype=torch.float32)
    best_temperature = None
    best_metrics = None
    best_score = None

    for temp in temp_grid.tolist():
        energy = compute_free_energy(logits, temperature=float(temp))
        val_metrics = evaluate_ood_metrics(energy[masks.val_id], energy[masks.val_ood])
        score_tuple = (val_metrics["FPR95"], -val_metrics["AUROC"])
        if best_score is None or score_tuple < best_score:
            best_score = score_tuple
            best_temperature = float(temp)
            best_metrics = val_metrics

    calibrated_energy = compute_free_energy(logits, temperature=best_temperature)
    test_metrics = evaluate_ood_metrics(calibrated_energy[masks.test_id], calibrated_energy[masks.test_ood])

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    torch.save(
        {
            "temperature": best_temperature,
            "val_metrics": best_metrics,
            "test_metrics": test_metrics,
            "seed": args.seed,
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "weights_path": args.weights_path,
        },
        args.save_path,
    )

    LOGGER.info("Phase 6: energy temperature calibration")
    LOGGER.info(
        "train_id=%d val_id=%d test_id=%d val_ood=%d test_ood=%d",
        int(masks.train_id.sum()),
        int(masks.val_id.sum()),
        int(masks.test_id.sum()),
        int(masks.val_ood.sum()),
        int(masks.test_ood.sum()),
    )
    LOGGER.info("selected_temperature=%.4f", best_temperature)
    LOGGER.info(
        "split=validation AUROC=%.4f AUPR=%.4f FPR95=%.4f",
        best_metrics["AUROC"],
        best_metrics["AUPR"],
        best_metrics["FPR95"],
    )
    LOGGER.info(
        "split=test AUROC=%.4f AUPR=%.4f FPR95=%.4f",
        test_metrics["AUROC"],
        test_metrics["AUPR"],
        test_metrics["FPR95"],
    )
    LOGGER.info("saved=%s", args.save_path)


if __name__ == "__main__":
    main()
