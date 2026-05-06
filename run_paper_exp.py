import time
import torch
import torch.nn as nn
from torch_geometric.datasets import Planetoid
from sklearn.metrics import roc_auc_score
from core_model import AsymmetricGNN, SupConDistillationLoss, compute_free_energy


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load Dataset
    dataset = Planetoid(root='./data/Cora', name='Cora')
    data = dataset[0].to(device)
    num_nodes = data.num_nodes

    # Label-based OOD Setup (ID: 0-3, OOD: 4-6)
    id_classes = [0, 1, 2, 3]
    labels = torch.ones(num_nodes, dtype=torch.long, device=device)
    for c in id_classes:
        labels[data.y == c] = 0

    id_mask = (labels == 0)
    ood_mask = (labels == 1)

    id_indices = torch.where(id_mask)[0]
    perm = torch.randperm(id_indices.size(0))
    train_size = int(len(id_indices) * 0.6)

    train_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    train_mask[id_indices[perm[:train_size]]] = True

    test_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    test_mask[id_indices[perm[train_size:]]] = True
    test_mask[ood_mask] = True

    test_ind_idx = torch.where(test_mask & id_mask)[0]
    test_ood_idx = torch.where(test_mask & ood_mask)[0]

    # Offline LLM Anchor Simulation
    torch.manual_seed(42)
    frozen_projector = nn.Linear(dataset.num_features, 768).to(device)
    frozen_projector.requires_grad_(False)
    z_sem_anchor = frozen_projector(data.x).detach()

    # Initialization
    model = AsymmetricGNN(in_channels=dataset.num_features, hidden_channels=128, out_channels=768).to(device)
    criterion = SupConDistillationLoss(margin=2.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)

    # Training
    epochs = 150
    model.train()
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        z_topo = model(data.x, data.edge_index)
        loss = criterion(z_topo[train_mask], z_sem_anchor[train_mask], labels[train_mask])
        loss.backward()
        optimizer.step()

    # Inference & Evaluation
    model.eval()
    with torch.no_grad():
        torch.cuda.synchronize()
        start_time = time.time()

        z_topo_all = model(data.x, data.edge_index)
        energy_ind = compute_free_energy(z_topo_all[test_ind_idx], temperature=1.0)
        energy_ood = compute_free_energy(z_topo_all[test_ood_idx], temperature=1.0)

        torch.cuda.synchronize()
        end_time = time.time()

    # Metrics
    y_true = torch.cat([torch.zeros_like(energy_ind), torch.ones_like(energy_ood)]).cpu().numpy()
    y_scores = torch.cat([energy_ind, energy_ood]).cpu().numpy()

    auroc = roc_auc_score(y_true, y_scores)
    if auroc < 0.5:
        auroc = 1.0 - auroc

    latency_ms = (end_time - start_time) * 1000
    test_node_count = test_ind_idx.size(0) + test_ood_idx.size(0)
    latency_per_node = latency_ms / test_node_count

    print(f"Test Nodes: {test_node_count}")
    print(f"AUROC: {auroc:.4f}")
    print(f"Total Latency: {latency_ms:.2f} ms")
    print(f"Latency per node: {latency_per_node:.4f} ms")


if __name__ == "__main__":
    main()