import time
import argparse
import torch
from torch_geometric.datasets import Planetoid
from sklearn.metrics import roc_auc_score
from core_model import AsymmetricGNN, compute_free_energy


def main():
    parser = argparse.ArgumentParser(description="Standalone Inference Profiling")
    parser.add_argument('--dataset', type=str, default='Cora')
    parser.add_argument('--weights_path', type=str, default='./weights/Cora_asym_gnn.pth')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = Planetoid(root=f'./data/{args.dataset}', name=args.dataset)
    data = dataset[0].to(device)

    # Determine out_channels dynamically based on weights
    state_dict = torch.load(args.weights_path, map_location=device)
    out_channels = state_dict['projector.net.3.weight'].size(0)

    model = AsymmetricGNN(in_channels=dataset.num_features, hidden_channels=128, out_channels=out_channels).to(device)
    model.load_state_dict(state_dict)

    model.eval()
    with torch.no_grad():
        torch.cuda.synchronize()
        start_time = time.time()

        z_topo_all = model(data.x, data.edge_index)
        energy_all = compute_free_energy(z_topo_all, temperature=1.0)

        torch.cuda.synchronize()
        end_time = time.time()

    latency_ms = (end_time - start_time) * 1000
    print(f"[Inference Profile] Dataset: {args.dataset}")
    print(f"Processed Nodes: {data.num_nodes}")
    print(f"Total Latency: {latency_ms:.4f} ms")
    print(f"Latency per node: {latency_ms / data.num_nodes:.4f} ms")


if __name__ == "__main__":
    main()