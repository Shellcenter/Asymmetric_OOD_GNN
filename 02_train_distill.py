"""Phase 2: semantic distillation training (Cora + ArXiv).

Semantic anchors (sentence-transformer embeddings of paper text) are
distilled into a GNN encoder via supervised contrastive learning.

Cora:  label-based ID/OOD (classes 0-3 vs 4-6).
ArXiv: time-based ID/OOD (<=2015 vs >=2018).
"""

from __future__ import annotations

import argparse
import logging
import os
import random

import numpy as np
import torch
import torch.nn.functional as F

from core_model import (
    AsymmetricGNN,
    IDEnergyBoundaryLoss,
    SemanticAlignmentLoss,
    compute_class_prototypes,
    compute_free_energy,
    compute_prototype_logits,
)
from data_loader import DatasetName, load_dataset


LOGGER = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Semantic distillation for graph OOD detection.")
    parser.add_argument("--dataset", type=str, default="cora", choices=("cora", "arxiv"))
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--anchor_path", type=str, default=None,
                        help="Override default anchor path.")
    parser.add_argument("--save_path", type=str, default=None,
                        help="Override default save path.")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--hidden_channels", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--semantic_weight", type=float, default=0.2)
    parser.add_argument("--classifier_weight", type=float, default=1.0)
    parser.add_argument("--prototype_weight", type=float, default=1.0)
    parser.add_argument("--energy_weight", type=float, default=0.2)
    parser.add_argument("--energy_compact_weight", type=float, default=0.05)
    parser.add_argument("--energy_margin", type=float, default=-6.0)
    parser.add_argument("--energy_temperature", type=float, default=1.0)
    parser.add_argument("--logit_scale", type=float, default=10.0)
    parser.add_argument("--train_ratio", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── load dataset ──
    ds = load_dataset(
        name=args.dataset,
        data_root=args.data_root,
        train_ratio=args.train_ratio,
        seed=args.seed,
        device=device,
    )

    anchor_path = args.anchor_path or ds.semantic_anchor_path
    if not os.path.exists(anchor_path):
        raise FileNotFoundError(
            f"Missing semantic anchors at {anchor_path}. "
            f"Run {'01b_extract_arxiv_anchors.py' if args.dataset == 'arxiv' else '01_extract_llm.py'} first."
        )

    z_sem_anchor = torch.load(anchor_path, map_location=device).float()
    if z_sem_anchor.size(0) != ds.data.num_nodes:
        raise ValueError(
            f"Anchor count {z_sem_anchor.size(0)} != node count {ds.data.num_nodes}"
        )

    num_id_classes = len(ds.id_classes) if ds.id_classes else 1

    # ── build model ──
    model = AsymmetricGNN(
        in_channels=ds.num_features,
        hidden_channels=args.hidden_channels,
        out_channels=z_sem_anchor.size(1),
        num_classes=num_id_classes,
    ).to(device)

    criterion = SemanticAlignmentLoss()
    energy_criterion = IDEnergyBoundaryLoss(
        margin=args.energy_margin, compact_weight=args.energy_compact_weight
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_class_labels = ds.data.y[ds.train_mask].long()
    train_mask = ds.train_mask

    LOGGER.info("Phase 2: semantic distillation on %s", args.dataset)
    LOGGER.info("nodes=%d features=%d id_classes=%d anchor_dim=%d",
                 ds.data.num_nodes, ds.num_features, num_id_classes, z_sem_anchor.size(1))
    LOGGER.info("train_id=%d eval_id=%d eval_ood=%d",
                 int(train_mask.sum()), int(ds.eval_id_mask.sum()), int(ds.eval_ood_mask.sum()))

    # ── training loop ──
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()

        class_logits = model.classify(ds.data.x, ds.data.edge_index)
        h_topo = model.encode(ds.data.x, ds.data.edge_index)
        z_topo = model.project(h_topo)

        # Semantic distillation (ID-only pull toward anchors)
        semantic_loss = criterion(z_topo[train_mask], z_sem_anchor[train_mask])

        # Prototype alignment
        train_prototypes = compute_class_prototypes(
            z_topo[train_mask], train_class_labels, ds.id_classes,
        )
        prototype_logits = compute_prototype_logits(
            z_topo[train_mask], train_prototypes, logit_scale=args.logit_scale,
        )
        prototype_loss = F.cross_entropy(prototype_logits, train_class_labels)

        # Classifier
        classifier_loss = F.cross_entropy(class_logits[train_mask], train_class_labels)

        # Energy regularizer (ID-only)
        train_energy = compute_free_energy(class_logits[train_mask], temperature=args.energy_temperature)
        energy_loss, energy_boundary_loss, energy_compact_loss = energy_criterion(train_energy)

        loss = (
            args.classifier_weight * classifier_loss
            + args.semantic_weight * semantic_loss
            + args.prototype_weight * prototype_loss
            + args.energy_weight * energy_loss
        )
        loss.backward()
        optimizer.step()

        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            LOGGER.info(
                "epoch=%03d/%03d loss=%.6f clf=%.6f sem=%.6f proto=%.6f "
                "energy_bound=%.6f energy_compact=%.6f",
                epoch, args.epochs, loss.item(),
                classifier_loss.item(), semantic_loss.item(), prototype_loss.item(),
                energy_boundary_loss.item(), energy_compact_loss.item(),
            )

    # ── save checkpoint ──
    model.eval()
    with torch.no_grad():
        z_topo = model(ds.data.x, ds.data.edge_index)
        id_prototypes = compute_class_prototypes(
            z_topo[train_mask], train_class_labels, ds.id_classes
        ).detach().cpu()

    save_path = args.save_path or f"./weights/{args.dataset}_gnn_seed{args.seed}.pth"
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "in_channels": ds.num_features,
            "hidden_channels": args.hidden_channels,
            "out_channels": z_sem_anchor.size(1),
            "num_classes": num_id_classes,
            "id_classes": ds.id_classes,
            "ood_classes": ds.ood_classes,
            "id_prototypes": id_prototypes,
            "logit_scale": args.logit_scale,
            "energy_margin": args.energy_margin,
            "energy_temperature": args.energy_temperature,
            "classifier_weight": args.classifier_weight,
            "energy_weight": args.energy_weight,
            "energy_compact_weight": args.energy_compact_weight,
            "dataset": args.dataset,
            "seed": args.seed,
            "train_ratio": args.train_ratio,
        },
        save_path,
    )
    LOGGER.info("saved=%s", save_path)


if __name__ == "__main__":
    main()
