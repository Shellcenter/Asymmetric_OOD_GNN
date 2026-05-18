"""Phase 3: online asymmetric OOD inference (Cora + ArXiv).

Evaluates a trained GNN checkpoint without requiring semantic features
at inference time.
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from core_model import AsymmetricGNN, compute_free_energy, compute_prototype_logits, evaluate_ood_metrics
from data_loader import DatasetName, load_dataset


LOGGER = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model(checkpoint_path: str, in_channels: int, num_id_classes: int, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" not in checkpoint:
        state_dict = checkpoint
        hidden_channels = 128
        out_channels = state_dict["projector.net.4.weight"].size(0)
    else:
        state_dict = checkpoint["model_state_dict"]
        hidden_channels = checkpoint.get("hidden_channels", 128)
        out_channels = checkpoint["out_channels"]

    model = AsymmetricGNN(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        num_classes=num_id_classes,
    ).to(device)
    missing, _ = model.load_state_dict(state_dict, strict=False)
    has_classifier = not any(k.startswith("classifier_conv.") for k in missing)
    model.eval()
    return model, checkpoint, has_classifier


def parse_args():
    parser = argparse.ArgumentParser(description="Online asymmetric OOD inference.")
    parser.add_argument("--dataset", type=str, default="cora", choices=("cora", "arxiv"))
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--weights_path", type=str, required=True)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--score_method", type=str, default="prototype_energy",
        choices=("auto", "classifier_energy", "classifier_msp", "prototype_energy", "prototype_msp"),
    )
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    args = parse_args()
    set_seed(args.seed)

    if not os.path.exists(args.weights_path):
        raise FileNotFoundError(f"Missing weights: {args.weights_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset using the seed stored in checkpoint (or CLI seed)
    ds = load_dataset(
        name=args.dataset, data_root=args.data_root,
        train_ratio=0.6, seed=args.seed, device=device,
    )

    model, config, has_classifier = load_model(
        args.weights_path, ds.num_features, len(ds.id_classes), device,
    )

    if "id_prototypes" not in config:
        raise KeyError("Checkpoint lacks id_prototypes. Re-run training.")
    id_prototypes = config["id_prototypes"].to(device).float()
    logit_scale = float(config.get("logit_scale", 10.0))

    score_method = args.score_method
    if score_method == "auto":
        score_method = "classifier_energy" if has_classifier else "prototype_energy"

    # ── warmup + inference ──
    with torch.no_grad():
        for _ in range(args.warmup):
            h_topo = model.encode(ds.data.x, ds.data.edge_index)
            if score_method.startswith("classifier"):
                logits = model.classify(ds.data.x, ds.data.edge_index)
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
            h_topo = model.encode(ds.data.x, ds.data.edge_index)
            if score_method.startswith("classifier"):
                logits = model.classify(ds.data.x, ds.data.edge_index)
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

    metrics = evaluate_ood_metrics(scores[ds.eval_id_mask], scores[ds.eval_ood_mask])

    LOGGER.info("Phase 3: online asymmetric inference on %s", args.dataset)
    LOGGER.info("score_method=%s", score_method)
    LOGGER.info("AUROC=%.4f AUPR=%.4f FPR95=%.4f latency_ms=%.4f x %d runs",
                 metrics["AUROC"], metrics["AUPR"], metrics["FPR95"], latency_ms, args.runs)


if __name__ == "__main__":
    main()
