"""Fair test: concat(raw, semantic) vs raw-only. Same 3-layer GCN architecture."""
import random, sys, torch, torch.nn.functional as F, numpy as np
from torch_geometric.nn import GCNConv

sys.path.insert(0, "E:/AsymOOD")
from data_loader import load_dataset
from core_model import compute_free_energy, evaluate_ood_metrics

SEED = 42
HIDDEN = 128
EPOCHS = 200
LR = 0.01
WD = 5e-4

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

set_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ds = load_dataset("arxiv", data_root="E:/AsymOOD/data", train_ratio=0.6, seed=SEED, device=device)
anchors = torch.load("E:/AsymOOD/embeddings/arxiv_semantic_anchor.pt", map_location=device).float()
id_classes = tuple(range(40))

for label, x_feat in [
    ("raw_only (128d)", ds.data.x),
    ("raw+sem (512d)", torch.cat([ds.data.x, anchors], dim=1)),
]:
    in_dim = x_feat.shape[1]
    print(f"\n=== {label} ===")
    set_seed(SEED)

    # Same 3-layer GCN as PlainGCN in run_comparison.py
    class GCN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = GCNConv(in_dim, HIDDEN)
            self.conv2 = GCNConv(HIDDEN, HIDDEN)
            self.cls  = GCNConv(HIDDEN, len(id_classes))
        def forward(self, x, ei):
            h = F.relu(self.conv1(x, ei))
            h = F.dropout(h, p=0.5, training=self.training)
            h = F.relu(self.conv2(h, ei))
            return self.cls(h, ei)

    model = GCN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)

    for ep in range(1, EPOCHS + 1):
        model.train(); opt.zero_grad()
        logits = model(x_feat, ds.data.edge_index)
        F.cross_entropy(logits[ds.train_mask], ds.data.y[ds.train_mask]).backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        logits = model(x_feat, ds.data.edge_index)
        energy = compute_free_energy(logits, temperature=1.0)
    m = evaluate_ood_metrics(energy[ds.eval_id_mask], energy[ds.eval_ood_mask])
    print(f"  AUROC={m['AUROC']:.4f}  AUPR={m['AUPR']:.4f}  FPR95={m['FPR95']:.4f}")
