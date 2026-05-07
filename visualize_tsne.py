import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch_geometric.datasets import Planetoid
from core_model import AsymmetricGNN


def main():
    parser = argparse.ArgumentParser(description="T-SNE Visualization")
    parser.add_argument('--dataset', type=str, default='Cora')
    parser.add_argument('--weights_path', type=str, default='./weights/Cora_asym_gnn.pth')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = Planetoid(root=f'./data/{args.dataset}', name=args.dataset)
    data = dataset[0].to(device)

    id_classes = [0, 1, 2, 3]
    labels = torch.ones(data.num_nodes, dtype=torch.long, device=device)
    for c in id_classes:
        labels[data.y == c] = 0

    state_dict = torch.load(args.weights_path, map_location=device)
    out_channels = state_dict['projector.net.3.weight'].size(0)

    model = AsymmetricGNN(in_channels=dataset.num_features, hidden_channels=128, out_channels=out_channels).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    with torch.no_grad():
        z_topo = model(data.x, data.edge_index).cpu().numpy()
        labels_np = labels.cpu().numpy()

    print("Running T-SNE dimensionality reduction...")
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
    z_tsne = tsne.fit_transform(z_topo)

    plt.figure(figsize=(8, 8))
    id_mask = (labels_np == 0)
    ood_mask = (labels_np == 1)

    plt.scatter(z_tsne[id_mask, 0], z_tsne[id_mask, 1], c='#1f77b4', label='ID Nodes', alpha=0.6, s=15,
                edgecolors='none')
    plt.scatter(z_tsne[ood_mask, 0], z_tsne[ood_mask, 1], c='#d62728', label='OOD Nodes', alpha=0.6, s=15,
                edgecolors='none')

    plt.title('T-SNE Visualization of Topological Manifolds', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=12)
    plt.axis('off')

    os.makedirs('./plots', exist_ok=True)
    plt.tight_layout()
    plt.savefig(f'./plots/{args.dataset}_tsne.pdf', format='pdf', dpi=300)
    print(f"Visualization saved to ./plots/{args.dataset}_tsne.pdf")


if __name__ == "__main__":
    main()