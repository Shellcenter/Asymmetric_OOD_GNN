"""Phase 3: online asymmetric inference without LLM dependencies."""

from __future__ import annotations

import argparse
import os
import random
import time

import numpy as np
import torch
from torch_geometric.datasets import Planetoid

from core_model import AsymmetricGNN, compute_free_energy, evaluate_ood_metrics


ID_CLASSES = (0, 1, 2, 3)
OOD_CLASSES = (4, 5, 6)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_eval_masks(y: torch.Tensor, train_ratio: float, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
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
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Online asymmetric OOD inference on Cora.")
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--weights_path", type=str, default="./weights/cora_gnn.pth")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if not os.path.exists(args.weights_path):
        raise FileNotFoundError(f"Missing GNN weights at {args.weights_path}. Run `python 02_train_distill.py` first.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = Planetoid(root=os.path.join(args.data_root, "Cora"), name="Cora")
    data = dataset[0].to(device)

    model, config = load_model(args.weights_path, dataset.num_features, device)
    eval_id_mask, eval_ood_mask = build_eval_masks(
        data.y,
        train_ratio=float(config.get("train_ratio", 0.6)),
        seed=int(config.get("seed", args.seed)),
    )
    eval_id_mask = eval_id_mask.to(device)
    eval_ood_mask = eval_ood_mask.to(device)

    if device.type == "cuda":
        torch.cuda.synchronize()
    start_time = time.perf_counter()
    with torch.no_grad():
        # Online contract: the model receives only graph topology features and edge_index.
        z_topo = model(data.x, data.edge_index)
        energy = compute_free_energy(z_topo, temperature=args.temperature)
    if device.type == "cuda":
        torch.cuda.synchronize()
    latency_ms = (time.perf_counter() - start_time) * 1000.0

    metrics = evaluate_ood_metrics(energy[eval_id_mask], energy[eval_ood_mask])

    print("=== Phase 3: Online Asymmetric Inference ===")
    print("LLM features are not loaded. Input contract: model(x, edge_index) only.")
    print(f"AUROC: {metrics['AUROC']:.4f}")
    print(f"AUPR: {metrics['AUPR']:.4f}")
    print(f"FPR@95TPR: {metrics['FPR95']:.4f}")
    print(f"Full-graph latency: {latency_ms:.4f} ms")


if __name__ == "__main__":
    main()
