"""Phase 1: offline semantic anchor extraction for Cora.

This script simulates a frozen large language model with a fixed linear
projection. The generated semantic anchors are persisted to disk and are the
only artifact consumed by the distillation phase.
"""

from __future__ import annotations

import argparse
import os
import random

import numpy as np
import torch
from torch_geometric.datasets import Planetoid


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline semantic anchor extraction on Cora.")
    parser.add_argument("--data_root", type=str, default="./data", help="Root directory for PyG datasets.")
    parser.add_argument("--output_path", type=str, default="./embeddings/cora_llm_anchor.pt")
    parser.add_argument("--anchor_dim", type=int, default=64, help="Dimension of the simulated LLM anchor.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = Planetoid(root=os.path.join(args.data_root, "Cora"), name="Cora")
    data = dataset[0].to(device)

    # A frozen linear layer stands in for a very large pretrained LLM encoder.
    llm_surrogate = torch.nn.Linear(dataset.num_features, args.anchor_dim, bias=False).to(device)
    llm_surrogate.requires_grad_(False)
    llm_surrogate.eval()

    with torch.no_grad():
        z_sem_anchor = llm_surrogate(data.x).detach().cpu()

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    torch.save(z_sem_anchor, args.output_path)

    print("=== Phase 1: Offline Semantic Anchor Generation ===")
    print(f"Dataset: Cora | Nodes: {data.num_nodes} | Input dim: {dataset.num_features}")
    print(f"Frozen anchor shape: {tuple(z_sem_anchor.shape)}")
    print(f"Saved semantic anchors to: {args.output_path}")
    print("The simulated LLM is now unloaded from all downstream phases.")


if __name__ == "__main__":
    main()
