"""Fair comparison on ArXiv: baseline vs semantic distillation.

All conditions use hidden=128, same architecture depth.
Evaluated with BOTH classifier_energy and prototype_energy scoring.
"""

from __future__ import annotations

import random, os, sys
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

sys.path.insert(0, "E:/AsymOOD")
from data_loader import load_dataset
from core_model import (
    AsymmetricGNN, SemanticAlignmentLoss,
    compute_class_prototypes, compute_prototype_logits,
    compute_free_energy, evaluate_ood_metrics,
)

SEED = 42
HIDDEN = 128  # ← unified
EPOCHS = 200
LR = 0.01
WD = 5e-4
LOGIT_SCALE = 10.0


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def evaluate_all(model, data, train_mask, eval_id, eval_ood,
                 id_classes, logit_scale):
    """Evaluate with both classifier_energy and prototype_energy scoring."""
    model.eval()
    with torch.no_grad():
        h = model.encode(data.x, data.edge_index)
        if hasattr(model, 'project'):
            z = model.project(h)
        else:
            z = h
        cls_logits = model.classify(data.x, data.edge_index)

    results = {}

    # classifier_energy scoring
    cls_energy = compute_free_energy(cls_logits, temperature=1.0)
    results["classifier_energy"] = evaluate_ood_metrics(
        cls_energy[eval_id], cls_energy[eval_ood])

    # prototype_energy scoring
    train_labels = data.y[train_mask].long()
    prototypes = compute_class_prototypes(z[train_mask], train_labels, id_classes)
    proto_logits = compute_prototype_logits(z, prototypes, logit_scale=logit_scale)
    proto_energy = compute_free_energy(proto_logits, temperature=1.0)
    results["prototype_energy"] = evaluate_ood_metrics(
        proto_energy[eval_id], proto_energy[eval_ood])

    return results


def pretty(m):
    return f"cls_E={m['classifier_energy']['AUROC']:.4f}  proto_E={m['prototype_energy']['AUROC']:.4f}"


# ── load ──
set_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

ds = load_dataset("arxiv", data_root="E:/AsymOOD/data", train_ratio=0.6, seed=SEED, device=device)
anchors = torch.load("E:/AsymOOD/embeddings/arxiv_semantic_anchor.pt", map_location=device).float()
id_classes = tuple(range(40))
print(f"Nodes: {ds.data.num_nodes}  Anchors: {anchors.shape[1]}d  Train: {ds.train_mask.sum().item()}")
print()

results_summary = {}

# ═══════════════════════════════════════════════════
# 1. PlainGCN baseline (CE only)
# ═══════════════════════════════════════════════════
print("=== 1/3: PlainGCN (CE only) ===")
set_seed(SEED)

class PlainGCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(128, HIDDEN)
        self.conv2 = GCNConv(HIDDEN, HIDDEN)
        self.cls = GCNConv(HIDDEN, len(id_classes))
    def encode(self, x, ei):
        h = F.relu(self.conv1(x, ei))
        h = F.dropout(h, p=0.5, training=self.training)
        h = F.relu(self.conv2(h, ei))
        return h
    def classify(self, x, ei):
        return self.cls(self.encode(x, ei), ei)
    def forward(self, x, ei):
        return self.cls(self.encode(x, ei), ei)

model1 = PlainGCN().to(device)
opt = torch.optim.Adam(model1.parameters(), lr=LR, weight_decay=WD)
for ep in range(1, EPOCHS + 1):
    model1.train(); opt.zero_grad()
    logits = model1(ds.data.x, ds.data.edge_index)
    F.cross_entropy(logits[ds.train_mask], ds.data.y[ds.train_mask]).backward()
    opt.step()
    if ep == 1 or ep % 50 == 0: print(f"  ep {ep:03d}")

m1 = evaluate_all(model1, ds.data, ds.train_mask, ds.eval_id_mask, ds.eval_ood_mask, id_classes, LOGIT_SCALE)
results_summary["PlainGCN"] = m1
print(f"  {pretty(m1)}")

# ═══════════════════════════════════════════════════
# 2. AsymmetricGNN + CE + prototype (no semantic)
# ═══════════════════════════════════════════════════
print("=== 2/3: AsymGNN + CE + prototype (no semantic) ===")
set_seed(SEED)

model2 = AsymmetricGNN(128, HIDDEN, anchors.size(1), num_classes=len(id_classes)).to(device)
opt2 = torch.optim.Adam(model2.parameters(), lr=LR, weight_decay=WD)

# proto_weight reduced to 0.1 to avoid gradient competition with classifier
PROTO_W = 0.1

for ep in range(1, EPOCHS + 1):
    model2.train(); opt2.zero_grad()
    cls_logits = model2.classify(ds.data.x, ds.data.edge_index)
    h = model2.encode(ds.data.x, ds.data.edge_index)
    z = model2.project(h)
    cls_loss = F.cross_entropy(cls_logits[ds.train_mask], ds.data.y[ds.train_mask])
    proto = compute_class_prototypes(z[ds.train_mask], ds.data.y[ds.train_mask].long(), id_classes)
    proto_logits = compute_prototype_logits(z[ds.train_mask], proto, logit_scale=LOGIT_SCALE)
    proto_loss = F.cross_entropy(proto_logits, ds.data.y[ds.train_mask])
    (1.0 * cls_loss + PROTO_W * proto_loss).backward()
    opt2.step()
    if ep == 1 or ep % 50 == 0: print(f"  ep {ep:03d}")

m2 = evaluate_all(model2, ds.data, ds.train_mask, ds.eval_id_mask, ds.eval_ood_mask, id_classes, LOGIT_SCALE)
results_summary["AsymGNN+proto"] = m2
print(f"  {pretty(m2)}")

# ═══════════════════════════════════════════════════
# 3. AsymmetricGNN + CE + prototype + semantic
# ═══════════════════════════════════════════════════
print("=== 3/3: AsymGNN + CE + proto + semantic ===")
set_seed(SEED)

model3 = AsymmetricGNN(128, HIDDEN, anchors.size(1), num_classes=len(id_classes)).to(device)
opt3 = torch.optim.Adam(model3.parameters(), lr=LR, weight_decay=WD)
sem_criterion = SemanticAlignmentLoss()

SEM_W = 0.2

for ep in range(1, EPOCHS + 1):
    model3.train(); opt3.zero_grad()
    cls_logits = model3.classify(ds.data.x, ds.data.edge_index)
    h = model3.encode(ds.data.x, ds.data.edge_index)
    z = model3.project(h)
    cls_loss = F.cross_entropy(cls_logits[ds.train_mask], ds.data.y[ds.train_mask])
    proto = compute_class_prototypes(z[ds.train_mask], ds.data.y[ds.train_mask].long(), id_classes)
    proto_logits = compute_prototype_logits(z[ds.train_mask], proto, logit_scale=LOGIT_SCALE)
    proto_loss = F.cross_entropy(proto_logits, ds.data.y[ds.train_mask])
    sem_loss = sem_criterion(z[ds.train_mask], anchors[ds.train_mask])
    (1.0 * cls_loss + PROTO_W * proto_loss + SEM_W * sem_loss).backward()
    opt3.step()
    if ep == 1 or ep % 50 == 0: print(f"  ep {ep:03d}")

m3 = evaluate_all(model3, ds.data, ds.train_mask, ds.eval_id_mask, ds.eval_ood_mask, id_classes, LOGIT_SCALE)
results_summary["AsymGNN+proto+sem"] = m3
print(f"  {pretty(m3)}")

# ── Summary ──
print()
print("=" * 65)
print(f"{'Method':<25} {'cls_energy':>10} {'proto_energy':>12}")
print("-" * 50)
for name in ["PlainGCN", "AsymGNN+proto", "AsymGNN+proto+sem"]:
    m = results_summary[name]
    print(f"{name:<25} {m['classifier_energy']['AUROC']:>10.4f} {m['prototype_energy']['AUROC']:>12.4f}")
print("-" * 50)
d_cls = results_summary["AsymGNN+proto+sem"]["classifier_energy"]["AUROC"] - results_summary["PlainGCN"]["classifier_energy"]["AUROC"]
d_proto = results_summary["AsymGNN+proto+sem"]["prototype_energy"]["AUROC"] - results_summary["AsymGNN+proto"]["prototype_energy"]["AUROC"]
print(f"Semantic distill gain (classifier_energy): {d_cls:+.4f}")
print(f"Semantic distill gain (prototype_energy): {d_proto:+.4f}")
