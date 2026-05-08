"""Phase 2: label-based cross-modal distillation on Cora.

The training split follows a strict label-based leave-out protocol:
classes 0-3 are in-distribution (ID), classes 4-6 are out-of-distribution
(OOD), and the GNN is optimized only on 60% of ID nodes.
"""

from __future__ import annotations

import argparse
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid

from core_model import AsymmetricGNN, SupConDistillationLoss, compute_class_prototypes, compute_prototype_logits


ID_CLASSES = (0, 1, 2, 3)
OOD_CLASSES = (4, 5, 6)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def build_leave_out_masks(y: torch.Tensor, train_ratio: float, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    labels = torch.ones_like(y, dtype=torch.long)
    id_mask = torch.zeros_like(y, dtype=torch.bool)

    for cls in ID_CLASSES:
        id_mask |= y == cls
    labels[id_mask] = 0

    ood_mask = torch.zeros_like(y, dtype=torch.bool)
    for cls in OOD_CLASSES:
        ood_mask |= y == cls

    generator = torch.Generator(device=y.device)
    generator.manual_seed(seed)
    id_indices = torch.where(id_mask)[0]
    perm = torch.randperm(id_indices.numel(), generator=generator, device=y.device)
    train_size = int(train_ratio * id_indices.numel())

    train_mask = torch.zeros_like(y, dtype=torch.bool)
    train_mask[id_indices[perm[:train_size]]] = True

    test_mask = torch.zeros_like(y, dtype=torch.bool)
    test_mask[id_indices[perm[train_size:]]] = True
    test_mask[ood_mask] = True

    # Academic leakage fuse: no OOD node can enter the optimization split.
    assert labels[train_mask].sum().item() == 0, "Data leakage: train_mask contains OOD nodes."
    return labels, train_mask, test_mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cross-modal distillation for asymmetric graph OOD detection.")
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--anchor_path", type=str, default="./embeddings/cora_llm_anchor.pt")
    parser.add_argument("--save_path", type=str, default="./weights/cora_gnn.pth")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--hidden_channels", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--semantic_weight", type=float, default=0.2)
    parser.add_argument("--prototype_weight", type=float, default=1.0)
    parser.add_argument("--logit_scale", type=float, default=10.0)
    parser.add_argument("--train_ratio", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = Planetoid(root=os.path.join(args.data_root, "Cora"), name="Cora")
    data = dataset[0].to(device)

    labels, train_mask, test_mask = build_leave_out_masks(data.y, args.train_ratio, args.seed)
    labels = labels.to(device)
    train_mask = train_mask.to(device)
    test_mask = test_mask.to(device)

    if not os.path.exists(args.anchor_path):
        raise FileNotFoundError(
            f"Missing semantic anchors at {args.anchor_path}. Run `python 01_extract_llm.py` first."
        )
    z_sem_anchor = torch.load(args.anchor_path, map_location=device).float()
    if z_sem_anchor.size(0) != data.num_nodes:
        raise ValueError("Semantic anchor count does not match the number of Cora nodes.")

    model = AsymmetricGNN(
        in_channels=dataset.num_features,
        hidden_channels=args.hidden_channels,
        out_channels=z_sem_anchor.size(1),
    ).to(device)
    criterion = SupConDistillationLoss(margin=args.margin)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_class_labels = data.y[train_mask].long()

    print("=== Phase 2: Cross-Modal Distillation ===")
    print(f"ID classes: {ID_CLASSES} | OOD classes: {OOD_CLASSES}")
    print(f"Train ID nodes: {int(train_mask.sum())} | Evaluation nodes: {int(test_mask.sum())}")
    print("LLM anchors are loaded as frozen tensors; no LLM module is attached to training.")

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        z_topo = model(data.x, data.edge_index)

        # Strict leave-out: only pure ID train nodes participate in optimization.
        semantic_loss = criterion(z_topo[train_mask], z_sem_anchor[train_mask], labels[train_mask])
        train_prototypes = compute_class_prototypes(z_topo[train_mask], train_class_labels, ID_CLASSES)
        prototype_logits = compute_prototype_logits(
            z_topo[train_mask],
            train_prototypes,
            logit_scale=args.logit_scale,
        )
        prototype_loss = F.cross_entropy(prototype_logits, train_class_labels)
        loss = args.semantic_weight * semantic_loss + args.prototype_weight * prototype_loss
        loss.backward()
        optimizer.step()

        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(
                f"Epoch {epoch:03d}/{args.epochs} | "
                f"Loss: {loss.item():.6f} | "
                f"Semantic: {semantic_loss.item():.6f} | "
                f"Prototype CE: {prototype_loss.item():.6f}"
            )

    model.eval()
    with torch.no_grad():
        z_topo = model(data.x, data.edge_index)
        id_prototypes = compute_class_prototypes(z_topo[train_mask], train_class_labels, ID_CLASSES).detach().cpu()

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "in_channels": dataset.num_features,
            "hidden_channels": args.hidden_channels,
            "out_channels": z_sem_anchor.size(1),
            "id_classes": ID_CLASSES,
            "ood_classes": OOD_CLASSES,
            "id_prototypes": id_prototypes,
            "logit_scale": args.logit_scale,
            "seed": args.seed,
            "train_ratio": args.train_ratio,
        },
        args.save_path,
    )
    print(f"Saved distilled asymmetric GNN to: {args.save_path}")


if __name__ == "__main__":
    main()
