"""T-SNE visualization of distilled topological manifolds."""

from __future__ import annotations

import argparse
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from torch_geometric.datasets import Planetoid

from core_model import AsymmetricGNN


ID_CLASSES = (0, 1, 2, 3)
OOD_CLASSES = (4, 5, 6)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_binary_labels(y: torch.Tensor) -> torch.Tensor:
    labels = torch.ones_like(y, dtype=torch.long)
    id_mask = torch.zeros_like(y, dtype=torch.bool)
    for cls in ID_CLASSES:
        id_mask |= y == cls
    labels[id_mask] = 0
    return labels


def load_model(checkpoint_path: str, dataset_num_features: int, device: torch.device) -> AsymmetricGNN:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        hidden_channels = checkpoint.get("hidden_channels", 128)
        out_channels = checkpoint["out_channels"]
        in_channels = checkpoint.get("in_channels", dataset_num_features)
    else:
        state_dict = checkpoint
        hidden_channels = 128
        out_channels = state_dict["projector.net.4.weight"].size(0)
        in_channels = dataset_num_features

    model = AsymmetricGNN(in_channels=in_channels, hidden_channels=hidden_channels, out_channels=out_channels).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize GNN topology embeddings with T-SNE.")
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--weights_path", type=str, default="./weights/cora_gnn.pth")
    parser.add_argument("--output_path", type=str, default="./plots/tsne_visualization.pdf")
    parser.add_argument("--perplexity", type=float, default=30.0)
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
    labels = build_binary_labels(data.y).cpu().numpy()

    model = load_model(args.weights_path, dataset.num_features, device)
    with torch.no_grad():
        z_topo = model(data.x, data.edge_index).detach().cpu().numpy()

    print("Running T-SNE on distilled topological embeddings...")
    try:
        tsne = TSNE(n_components=2, perplexity=args.perplexity, max_iter=1000, init="pca", random_state=args.seed)
    except TypeError:
        tsne = TSNE(n_components=2, perplexity=args.perplexity, n_iter=1000, init="pca", random_state=args.seed)
    z_2d = tsne.fit_transform(z_topo)

    id_mask = labels == 0
    ood_mask = labels == 1

    plt.figure(figsize=(7.0, 6.0))
    plt.scatter(z_2d[id_mask, 0], z_2d[id_mask, 1], s=12, c="#1f77b4", alpha=0.72, label="ID Nodes")
    plt.scatter(z_2d[ood_mask, 0], z_2d[ood_mask, 1], s=12, c="#d62728", alpha=0.72, label="OOD Nodes")
    plt.title("T-SNE of Distilled Topological Manifold")
    plt.legend(frameon=True)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    plt.savefig(args.output_path, format="pdf", dpi=300, bbox_inches="tight")
    print(f"Saved paper-ready visualization to: {args.output_path}")


if __name__ == "__main__":
    main()
