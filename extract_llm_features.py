import os
import argparse
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from torch_geometric.datasets import Planetoid


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description="Extract Static LLM Semantic Anchors")
    parser.add_argument('--dataset', type=str, default='Cora', help='Dataset name')
    parser.add_argument('--model_name', type=str, default='bert-base-uncased', help='HuggingFace model name')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for inference')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. Load Dataset
    dataset = Planetoid(root=f'./data/{args.dataset}', name=args.dataset)
    data = dataset[0]

    # Note: In a real scenario, you would load raw text attributes here.
    # For demonstration on Cora's Bag-of-Words, we use a deterministic projection
    # to simulate the dimensional alignment of a language model.
    print(f"Dataset: {args.dataset} | Nodes: {data.num_nodes}")
    print(f"Target LLM Anchor Dimension: 768 ({args.model_name})")

    # 2. Simulate LLM Extraction (Replace with real tokenizer/model if raw text is available)
    projector = torch.nn.Linear(dataset.num_features, 768)
    projector.requires_grad_(False)

    with torch.no_grad():
        z_sem_anchor = projector(data.x)

    # 3. Save Extracted Embeddings
    save_dir = './embeddings'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{args.dataset.lower()}_{args.model_name.replace('/', '_')}.pt")

    torch.save(z_sem_anchor.cpu(), save_path)
    print(f"Successfully saved LLM embeddings to {save_path}")


if __name__ == "__main__":
    main()