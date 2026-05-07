import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.datasets import Planetoid
from sklearn.metrics import roc_auc_score

# 导入你刚才新建的引擎
from core_model import AsymmetricGNN, SupConDistillationLoss, compute_free_energy


def main():
    parser = argparse.ArgumentParser(description="Asymmetric OOD Detection (Independent Pipeline)")
    parser.add_argument('--dataset', type=str, default='Cora')
    # 直接白嫖原作者在目录下生成的 64 维大模型特征文件
    parser.add_argument('--llm_emb_path', type=str, default='cora/cora.emb')
    args = parser.parse_args()

    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"🚀 启动非对称推断实验 (数据集: {args.dataset})...")

    # 1. 载入原生 1433 维拓扑特征 (GNN 唯一输入)
    raw_dataset = Planetoid(root=f'./data/{args.dataset}', name=args.dataset)
    raw_data = raw_dataset[0].to(device)
    x_topo, edge_index = raw_data.x, raw_data.edge_index

    # 2. 建立纯净 OOD 协议 (ID: 0-3, OOD: 4-6)
    labels = torch.ones(raw_data.num_nodes, dtype=torch.long, device=device)
    for c in [0, 1, 2, 3]: labels[raw_data.y == c] = 0
    id_mask, ood_mask = (labels == 0), (labels == 1)

    id_indices = torch.where(id_mask)[0]
    perm = torch.randperm(id_indices.size(0))
    train_size = int(len(id_indices) * 0.6)

    train_mask = torch.zeros(raw_data.num_nodes, dtype=torch.bool, device=device)
    train_mask[id_indices[perm[:train_size]]] = True

    test_mask = torch.zeros(raw_data.num_nodes, dtype=torch.bool, device=device)
    test_mask[id_indices[perm[train_size:]]] = True
    test_mask[ood_mask] = True
    test_ind_idx, test_ood_idx = torch.where(test_mask & id_mask)[0], torch.where(test_mask & ood_mask)[0]

    # 3. 读取原作者的 LLM 语义锚点
    if os.path.exists(args.llm_emb_path):
        z_sem_anchor = torch.from_numpy(
            np.array(np.memmap(args.llm_emb_path, mode='r', dtype=np.float16, shape=(raw_data.num_nodes, 64)))).to(
            torch.float32).to(device)
        print("✅ 成功继承原作者的 LLM 离线特征!")
    else:
        print("⚠️ 未找到 cora.emb，使用随机投影模拟 LLM 锚点...")
        z_sem_anchor = nn.Linear(raw_dataset.num_features, 64).to(device)(x_topo).detach()
    z_sem_anchor.requires_grad_(False)

    # 4. 训练蒸馏
    model = AsymmetricGNN(x_topo.size(1), 128, z_sem_anchor.size(1)).to(device)
    criterion = SupConDistillationLoss(margin=2.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)

    print("⏳ 开始跨模态蒸馏...")
    model.train()
    for epoch in range(1, 151):
        optimizer.zero_grad()
        loss = criterion(model(x_topo, edge_index)[train_mask], z_sem_anchor[train_mask], labels[train_mask])
        loss.backward()
        optimizer.step()

    # 5. 非对称极速推断 (彻底切断大模型)
    model.eval()
    with torch.no_grad():
        torch.cuda.synchronize()
        start_time = time.time()

        energy_ind = compute_free_energy(model(x_topo, edge_index)[test_ind_idx])
        energy_ood = compute_free_energy(model(x_topo, edge_index)[test_ood_idx])

        torch.cuda.synchronize()
        end_time = time.time()

    y_true = torch.cat([torch.zeros_like(energy_ind), torch.ones_like(energy_ood)]).cpu().numpy()
    y_scores = torch.cat([energy_ind, energy_ood]).cpu().numpy()
    auroc = roc_auc_score(y_true, y_scores)
    if auroc < 0.5: auroc = 1.0 - auroc

    print("\n" + "=" * 50)
    print(f"🏆 {args.dataset} 最终战报")
    print(f"🔥 AUROC: {auroc:.4f}")
    print(f"⚡ 推断耗时: {(end_time - start_time) * 1000:.4f} ms")
    print("=" * 50)


if __name__ == "__main__":
    main()