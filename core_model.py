"""Core components for asymmetric graph OOD detection.

This module intentionally contains no dataset-specific logic. The LLM semantic
anchors are consumed only during distillation; online inference uses the GNN
with graph features and ``edge_index`` only.
"""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score, roc_curve
from torch_geometric.nn import GCNConv


class MLPDynamicProjector(nn.Module):
    """Projection head that maps GNN topology features to the anchor space."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AsymmetricGNN(nn.Module):
    """Two-layer GCN encoder followed by a lightweight projection MLP."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        dropout: float = 0.5,
        projector_dropout: float = 0.2,
    ):
        super().__init__()
        self.dropout = dropout
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.projector = MLPDynamicProjector(
            input_dim=hidden_channels,
            hidden_dim=hidden_channels * 2,
            output_dim=out_channels,
            dropout=projector_dropout,
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x, edge_index)
        h = F.relu(h, inplace=True)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.conv2(h, edge_index)
        h = F.relu(h, inplace=True)
        return self.projector(h)


class SupConDistillationLoss(nn.Module):
    """Supervised contrastive distillation with an ID pull and OOD margin push.

    ``labels`` follows the binary OOD convention: 0 for ID nodes and 1 for OOD
    nodes. If a pure ID training split is provided, the OOD term is exactly zero,
    which preserves strict leave-out evaluation.
    """

    def __init__(self, margin: float = 1.0, ood_weight: float = 1.0):
        super().__init__()
        self.margin = margin
        self.ood_weight = ood_weight

    def forward(
        self,
        z_topo: torch.Tensor,
        z_sem: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        if z_topo.shape != z_sem.shape:
            raise ValueError(f"Shape mismatch: z_topo={tuple(z_topo.shape)}, z_sem={tuple(z_sem.shape)}")

        z_topo = F.normalize(z_topo, p=2, dim=1)
        z_sem = F.normalize(z_sem, p=2, dim=1)
        dist = F.pairwise_distance(z_topo, z_sem, p=2)

        id_mask = labels == 0
        ood_mask = labels == 1
        zero = z_topo.new_tensor(0.0)

        pull_id = dist[id_mask].pow(2).mean() if id_mask.any() else zero
        push_ood = F.relu(self.margin - dist[ood_mask]).pow(2).mean() if ood_mask.any() else zero
        return pull_id + self.ood_weight * push_ood


def compute_free_energy(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Compute the thermodynamic free energy score.

    Higher returned values are treated as more OOD-like. The implementation uses
    ``-T * logsumexp`` from energy-based OOD detection; downstream code compares
    the resulting scalar scores directly.
    """

    if temperature <= 0:
        raise ValueError("temperature must be positive.")
    return -temperature * torch.logsumexp(logits / temperature, dim=1)


def compute_class_prototypes(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    class_ids: Iterable[int],
) -> torch.Tensor:
    """Compute one normalized prototype per ID class.

    The caller is responsible for passing only leakage-free training nodes.
    """

    prototypes = []
    for class_id in class_ids:
        class_mask = labels == int(class_id)
        if not class_mask.any():
            raise ValueError(f"Class {class_id} has no samples for prototype construction.")
        prototype = embeddings[class_mask].mean(dim=0)
        prototypes.append(prototype)
    return F.normalize(torch.stack(prototypes, dim=0), p=2, dim=1)


def compute_prototype_logits(
    embeddings: torch.Tensor,
    prototypes: torch.Tensor,
    logit_scale: float = 10.0,
) -> torch.Tensor:
    """Convert embeddings into cosine-similarity logits against ID prototypes."""

    embeddings = F.normalize(embeddings, p=2, dim=1)
    prototypes = F.normalize(prototypes, p=2, dim=1)
    return logit_scale * embeddings @ prototypes.t()


def fit_mahalanobis_statistics(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    class_ids: Iterable[int],
    covariance_eps: float = 1e-4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Estimate class means and shared precision matrix from ID train nodes."""

    class_ids = tuple(class_ids)
    means = []
    centered_all = []
    for class_id in class_ids:
        class_mask = labels == int(class_id)
        if not class_mask.any():
            raise ValueError(f"Class {class_id} has no samples for Mahalanobis fitting.")
        class_emb = embeddings[class_mask]
        class_mean = class_emb.mean(dim=0)
        means.append(class_mean)
        centered_all.append(class_emb - class_mean)

    means = torch.stack(means, dim=0)
    centered = torch.cat(centered_all, dim=0)
    feature_dim = centered.size(1)
    dof = max(centered.size(0) - len(class_ids), 1)
    covariance = (centered.t() @ centered) / float(dof)
    covariance = covariance + covariance_eps * torch.eye(feature_dim, device=embeddings.device, dtype=embeddings.dtype)
    precision = torch.linalg.pinv(covariance)
    return means, precision


def compute_mahalanobis_logits(
    embeddings: torch.Tensor,
    means: torch.Tensor,
    precision: torch.Tensor,
) -> torch.Tensor:
    """Return negative squared Mahalanobis distance logits for each ID class."""

    diff = embeddings.unsqueeze(1) - means.unsqueeze(0)  # [N, C, D]
    md2 = torch.einsum("ncd,df,ncf->nc", diff, precision, diff)
    return -0.5 * md2


def _to_numpy(x: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().float().numpy()
    return np.asarray(x, dtype=np.float64)


def evaluate_ood_metrics(energy_ind: torch.Tensor | np.ndarray, energy_ood: torch.Tensor | np.ndarray) -> Dict[str, float]:
    """Return AUROC, AUPR, and FPR@95TPR for OOD scores.

    The metric convention is binary: ID nodes are negatives (0), OOD nodes are
    positives (1), and larger scores indicate stronger OOD evidence.
    """

    ind_scores = _to_numpy(energy_ind)
    ood_scores = _to_numpy(energy_ood)
    y_true = np.concatenate([np.zeros_like(ind_scores), np.ones_like(ood_scores)])
    y_score = np.concatenate([ind_scores, ood_scores])

    auroc = roc_auc_score(y_true, y_score)
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    aupr = auc(recall, precision)
    fpr, tpr, _ = roc_curve(y_true, y_score)

    if np.any(tpr >= 0.95):
        fpr95 = float(fpr[np.argmax(tpr >= 0.95)])
    else:
        fpr95 = 1.0

    return {"AUROC": float(auroc), "AUPR": float(aupr), "FPR95": fpr95}