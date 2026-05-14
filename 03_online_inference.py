"""Phase 3: online asymmetric inference without LLM dependencies."""

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

from core_model import AsymmetricGNN, compute_free_energy, compute_prototype_logits, evaluate_ood_metrics


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


def build_eval_masks(y: torch.Tensor, train_ratio: float, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Build evaluation masks under the leave-out protocol."""
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

    eval_id_mask = torch.zeros_like(y, dtype=torch.bool)
    eval_id_mask[id_indices[perm[train_size:]]] = True
    return eval_id_mask, ood_mask


def load_model(checkpoint_path: str, dataset_num_features: int, device: torch.device) -> tuple[AsymmetricGNN, dict]:
    """Load a distilled GNN checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" in checkpoint:
        config = checkpoint
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
        config = {
            "in_channels": dataset_num_features,
            "hidden_channels": 128,
            "out_channels": state_dict["projector.net.4.weight"].size(0),
            "seed": 42,
            "train_ratio": 0.6,
        }

    model = AsymmetricGNN(
        in_channels=config.get("in_channels", dataset_num_features),
        hidden_channels=config.get("hidden_channels", 128),
        out_channels=config["out_channels"],
        num_classes=config.get("num_classes", len(ID_CLASSES)),
    ).to(device)
    missing, _ = model.load_state_dict(state_dict, strict=False)
    config["has_classifier"] = not any(key.startswith("classifier_conv.") for key in missing)
    model.eval()
    return model, config


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Online asymmetric OOD inference on Cora.")
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--weights_path", type=str, default="./weights/cora_gnn.pth")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--score_method",
        type=str,
        default="auto",
        choices=("auto", "classifier_energy", "classifier_msp", "prototype_energy", "prototype_msp"),
    )
    return parser.parse_args()


def main() -> None:
    """Run online asymmetric inference."""
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    args = parse_args()
    set_seed(args.seed)

    if not os.path.exists(args.weights_path):
        raise FileNotFoundError(f"Missing GNN weights at {args.weights_path}. Run `python 02_train_distill.py` first.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = Planetoid(root=os.path.join(args.data_root, "Cora"), name="Cora")
    data = dataset[0].to(device)

    model, config = load_model(args.weights_path, dataset.num_features, device)
    if "id_prototypes" not in config:
        raise KeyError(
            "Checkpoint does not contain ID prototypes. Re-run `python 02_train_distill.py` "
            "to create a prototype-calibrated asymmetric checkpoint."
        )
    id_prototypes = config["id_prototypes"].to(device).float()
    logit_scale = float(config.get("logit_scale", 10.0))
    score_method = args.score_method
    if score_method == "auto":
        score_method = "classifier_energy" if config.get("has_classifier", False) else "prototype_energy"
    eval_id_mask, eval_ood_mask = build_eval_masks(
        data.y,
        train_ratio=float(config.get("train_ratio", 0.6)),
        seed=int(config.get("seed", args.seed)),
    )
    eval_id_mask = eval_id_mask.to(device)
    eval_ood_mask = eval_ood_mask.to(device)

    with torch.no_grad():
        for _ in range(args.warmup):
            h_topo = model.encode(data.x, data.edge_index)
            if score_method.startswith("classifier"):
                logits = model.classify(data.x, data.edge_index)
            else:
                z_topo = model.project(h_topo)
                logits = compute_prototype_logits(z_topo, id_prototypes, logit_scale=logit_scale)
            if score_method.endswith("energy"):
                _ = compute_free_energy(logits, temperature=args.temperature)
            else:
                _ = 1.0 - F.softmax(logits, dim=1).max(dim=1).values

    if device.type == "cuda":
        torch.cuda.synchronize()
    start_time = time.perf_counter()
    with torch.no_grad():
        for _ in range(args.runs):
            h_topo = model.encode(data.x, data.edge_index)
            if score_method.startswith("classifier"):
                logits = model.classify(data.x, data.edge_index)
            else:
                z_topo = model.project(h_topo)
                logits = compute_prototype_logits(z_topo, id_prototypes, logit_scale=logit_scale)
            if score_method.endswith("energy"):
                scores = compute_free_energy(logits, temperature=args.temperature)
            else:
                scores = 1.0 - F.softmax(logits, dim=1).max(dim=1).values
    if device.type == "cuda":
        torch.cuda.synchronize()
    latency_ms = (time.perf_counter() - start_time) * 1000.0 / args.runs

    metrics = evaluate_ood_metrics(scores[eval_id_mask], scores[eval_ood_mask])

    LOGGER.info("Phase 3: online asymmetric inference")
    LOGGER.info("score_method=%s", score_method)
    LOGGER.info("AUROC=%.4f", metrics["AUROC"])
    LOGGER.info("AUPR=%.4f", metrics["AUPR"])
    LOGGER.info("FPR95=%.4f", metrics["FPR95"])
    LOGGER.info("latency_ms=%.4f runs=%d", latency_ms, args.runs)


if __name__ == "__main__":
    main()
