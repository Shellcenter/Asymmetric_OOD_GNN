"""Unified data loading for Cora and OGB-ArXiv with ID/OOD splits.

Cora: label-based split (classes 0-3 = ID, 4-6 = OOD, as before).
ArXiv: time-based split (papers published <= 2015 = ID, >= 2018 = OOD).
       Papers from 2016-2017 are excluded to create a clear distribution gap.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid


LOGGER = logging.getLogger(__name__)

DatasetName = Literal["cora", "arxiv"]


@dataclass
class OODDataset:
    """Container for OOD detection datasets with pre-built masks."""

    name: DatasetName
    data: Data                # PyG data object on the target device
    num_features: int
    id_classes: tuple[int, ...]
    ood_classes: tuple[int, ...]     # only meaningful for Cora
    train_mask: torch.Tensor   # ID training nodes
    eval_id_mask: torch.Tensor # ID held-out nodes
    eval_ood_mask: torch.Tensor # OOD nodes
    labels: torch.Tensor       # binary: 0=ID, 1=OOD
    semantic_anchor_path: str  # path to pre-computed anchor .pt file
    # ArXiv-specific
    id_year_boundary: int | None = None
    ood_year_boundary: int | None = None


# ─── Cora ───────────────────────────────────────────────────────

ID_CLASSES_CORA = (0, 1, 2, 3)
OOD_CLASSES_CORA = (4, 5, 6)


def _build_cora_masks(
    y: torch.Tensor,
    train_ratio: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    id_mask = torch.zeros_like(y, dtype=torch.bool)
    for cls in ID_CLASSES_CORA:
        id_mask |= y == cls

    ood_mask = torch.zeros_like(y, dtype=torch.bool)
    for cls in OOD_CLASSES_CORA:
        ood_mask |= y == cls

    binary_labels = torch.ones_like(y, dtype=torch.long)
    binary_labels[id_mask] = 0

    generator = torch.Generator(device=y.device)
    generator.manual_seed(seed)
    id_indices = torch.where(id_mask)[0]
    perm = torch.randperm(id_indices.numel(), generator=generator, device=y.device)
    train_size = int(train_ratio * id_indices.numel())

    train_mask = torch.zeros_like(y, dtype=torch.bool)
    eval_id_mask = torch.zeros_like(y, dtype=torch.bool)
    train_mask[id_indices[perm[:train_size]]] = True
    eval_id_mask[id_indices[perm[train_size:]]] = True

    return binary_labels, train_mask, eval_id_mask, ood_mask


def load_cora(
    data_root: str = "./data",
    train_ratio: float = 0.6,
    seed: int = 42,
    device: torch.device | None = None,
) -> OODDataset:
    """Load Cora with label-based ID/OOD split."""
    dataset = Planetoid(root=os.path.join(data_root, "Cora"), name="Cora")
    data = dataset[0]
    if device is not None:
        data = data.to(device)

    labels, train_mask, eval_id, eval_ood = _build_cora_masks(
        data.y, train_ratio, seed,
    )
    labels = labels.to(data.y.device)

    return OODDataset(
        name="cora",
        data=data,
        num_features=dataset.num_features,
        id_classes=ID_CLASSES_CORA,
        ood_classes=OOD_CLASSES_CORA,
        train_mask=train_mask,
        eval_id_mask=eval_id,
        eval_ood_mask=eval_ood,
        labels=labels,
        semantic_anchor_path="./embeddings/cora_semantic_anchor.pt",
    )


# ─── ArXiv (time-based OOD) ─────────────────────────────────────

ARXIV_ID_YEAR = 2015   # published <= 2015 → ID
ARXIV_OOD_YEAR = 2018  # published >= 2018 → OOD (skip 2016-2017 gap)


def _build_arxiv_masks(
    node_years: np.ndarray,
    n_nodes: int,
    id_year: int,
    ood_year: int,
    train_ratio: float,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    years = node_years.flatten()
    id_mask = torch.from_numpy(years <= id_year).to(device)
    ood_mask = torch.from_numpy(years >= ood_year).to(device)
    gap_mask = ~id_mask & ~ood_mask  # 2016-2017

    binary_labels = torch.ones(n_nodes, dtype=torch.long, device=device)
    binary_labels[id_mask] = 0
    binary_labels[gap_mask] = -1  # exclude from evaluation

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    id_indices = torch.where(id_mask)[0]
    perm = torch.randperm(id_indices.numel(), generator=generator, device=device)
    train_size = int(train_ratio * id_indices.numel())

    train_mask = torch.zeros(n_nodes, dtype=torch.bool, device=device)
    train_mask[id_indices[perm[:train_size]]] = True
    eval_id_mask = torch.zeros(n_nodes, dtype=torch.bool, device=device)
    eval_id_mask[id_indices[perm[train_size:]]] = True

    return binary_labels, train_mask, eval_id_mask, ood_mask


def _patch_torch_load():
    """Force weights_only=False for OGB compatibility with PyTorch >=2.6."""
    import torch
    _orig = torch.load
    def _patched(f, map_location=None, weights_only=False, **kw):
        return _orig(f, map_location=map_location, weights_only=False, **kw)
    torch.load = _patched


def load_arxiv(
    data_root: str = "./data",
    train_ratio: float = 0.6,
    seed: int = 42,
    device: torch.device | None = None,
) -> OODDataset:
    """Load OGB-ArXiv with time-based ID/OOD split."""
    _patch_torch_load()
    from ogb.nodeproppred import NodePropPredDataset

    ogb_root = os.path.join(data_root, "ogb")
    ogb_dataset = NodePropPredDataset(name="ogbn-arxiv", root=ogb_root)
    graph_dict, labels = ogb_dataset[0]

    # Build edge_index
    edge_index = torch.from_numpy(graph_dict["edge_index"]).long()
    node_feat = torch.from_numpy(graph_dict["node_feat"]).float()
    node_years = graph_dict["node_year"]
    node_labels = torch.from_numpy(labels.flatten()).long()
    n_nodes = int(graph_dict["num_nodes"])
    n_features = int(node_feat.shape[1])

    # Build masks
    mask_device = device if device is not None else torch.device("cpu")
    binary_labels, train_mask, eval_id, eval_ood = _build_arxiv_masks(
        node_years, n_nodes, ARXIV_ID_YEAR, ARXIV_OOD_YEAR, train_ratio, seed, mask_device,
    )

    data = Data(x=node_feat, edge_index=edge_index, y=node_labels)
    if device is not None:
        data = data.to(device)
        binary_labels = binary_labels.to(device)
        train_mask = train_mask.to(device)
        eval_id = eval_id.to(device)
        eval_ood = eval_ood.to(device)

    n_id = int(train_mask.sum() + eval_id.sum())
    n_ood = int(eval_ood.sum())
    n_gap = int((binary_labels == -1).sum())
    LOGGER.info(
        "ArXiv loaded: nodes=%d id=%d ood=%d gap(2016-2017)=%d train_id=%d",
        n_nodes, n_id, n_ood, n_gap, int(train_mask.sum()),
    )

    return OODDataset(
        name="arxiv",
        data=data,
        num_features=n_features,
        id_classes=tuple(range(40)),  # all classes are ID (OOD is temporal)
        ood_classes=(),
        train_mask=train_mask,
        eval_id_mask=eval_id,
        eval_ood_mask=eval_ood,
        labels=binary_labels,
        semantic_anchor_path="./embeddings/arxiv_semantic_anchor.pt",
        id_year_boundary=ARXIV_ID_YEAR,
        ood_year_boundary=ARXIV_OOD_YEAR,
    )


# ─── Unified loader ─────────────────────────────────────────────

def load_dataset(
    name: DatasetName,
    data_root: str = "./data",
    train_ratio: float = 0.6,
    seed: int = 42,
    device: torch.device | None = None,
) -> OODDataset:
    """Load an OOD dataset by name.

    Args:
        name: 'cora' or 'arxiv'.
        data_root: Root directory for datasets.
        train_ratio: Fraction of ID nodes used for training.
        seed: Random seed for reproducible train/eval splits.
        device: Torch device to place the data on.

    Returns:
        OODDataset with pre-built masks and metadata.
    """
    loaders = {"cora": load_cora, "arxiv": load_arxiv}
    if name not in loaders:
        raise ValueError(f"Unknown dataset: {name}. Choose from {list(loaders)}.")
    return loaders[name](data_root=data_root, train_ratio=train_ratio, seed=seed, device=device)
