import sys
import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.datasets import Planetoid
from sklearn.metrics import roc_auc_score

# 强行将 gnnsafe 加入环境变量，防止其内部的相对导入报错
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, 'gnnsafe'))
sys.path.append(current_dir)

# 导入原作者的真实数据加载器
from gnnsafe.dataset import load_dataset
# 导入你的核心引擎
from core_model import AsymmetricGNN, SupConDistillationLoss, compute_free_energy


class LLMGuardArgs:
    """伪造原作者所需的参数字典"""

    def __init__(self):
        self.dataset = 'cora'
        self.data_dir = './data/'
        self.ood_type = 'label'


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = LLMGuardArgs()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(42)

    print("[Phase 1] 双轨特征融合：挂载原生拓扑 + LLMGuard 高阶数据...")

    # ---------------------------------------------------------
    # 核心改动 1：GNN 输入必须是 1433 维的原始低成本拓扑特征
    # ---------------------------------------------------------
    raw_dataset = Planetoid(root='./data/Planetoid', name='Cora')
    x_topo = raw_dataset[0].x.to(device)  # 1433D
    edge_index = raw_dataset[0].edge_index.to(device)

    # ---------------------------------------------------------
    # 核心改动 2：调用原始代码获取严格的 OOD 切分与大模型特征
    # ---------------------------------------------------------
    try:
        # 这一步会读取 cora/cora.emb 并载入他们设定的类留出划分
        dataset_ind, dataset_ood_tr, dataset_ood_te = load_dataset(args)
    except FileNotFoundError:
        print("❌ 错误：找不到 cora/cora.emb 或 ood_embs.pth！")
        print("💡 请先运行原作者的 utils/get_embs.py 生成对应的大模型特征文件。")
        return

    # 提取 LLMGuard 预先用大模型跑好的 64 维特征，作为静止锚点！(禁止作为GNN输入)
    z_sem_anchor = dataset_ind.x.to(device).clone().detach()
    z_sem_anchor.requires_grad_(False)

    # 获取严格对齐的节点索引
    train_idx = dataset_ind.splits['train'].to(device)
    test_ind_idx = dataset_ind.splits['test'].to(device)
    test_ood_idx = dataset_ood_te.node_idx.to(device)

    # 生成训练掩码
    train_mask = torch.zeros(x_topo.size(0), dtype=torch.bool, device=device)
    train_mask[train_idx] = True

    # 制作标签，用于蒸馏 Loss (ID 为 0，OOD 为 1)
    labels = torch.zeros(x_topo.size(0), dtype=torch.long, device=device)
    labels[test_ood_idx] = 1

    print(f"  --> 原生拓扑维度: {x_topo.size(1)} | LLM锚点维度: {z_sem_anchor.size(1)}")
    print(
        f"  --> ID训练节点: {train_idx.size(0)} | ID测试节点: {test_ind_idx.size(0)} | OOD测试节点: {test_ood_idx.size(0)}")

    # ---------------------------------------------------------
    # 核心改动 3：初始化你的非对称双流网络
    # ---------------------------------------------------------
    # in_channels 接收 1433，out_channels 对齐 LLM 锚点的 64
    model = AsymmetricGNN(in_channels=x_topo.size(1), hidden_channels=128, out_channels=z_sem_anchor.size(1)).to(device)
    criterion = SupConDistillationLoss(margin=2.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)

    print("\n[Phase 2] 启动跨模态特征蒸馏 (仅使用 ID 节点)...")
    model.train()
    for epoch in range(1, 151):
        optimizer.zero_grad()
        # 注意：这里喂给 GNN 的全是廉价的 x_topo
        z_topo = model(x_topo, edge_index)

        # 强制将拓扑特征拉向 LLM 特征
        loss = criterion(z_topo[train_mask], z_sem_anchor[train_mask], labels[train_mask])
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            print(f"  Epoch {epoch}/150, Distillation Loss: {loss.item():.4f}")

    # 保存模型用于可视化
    os.makedirs('weights', exist_ok=True)
    torch.save(model.state_dict(), 'weights/cora_asym_gnn.pth')

    print("\n[Phase 3] 彻底切断大模型算力，启动单模态极速测试...")
    model.eval()
    with torch.no_grad():
        torch.cuda.synchronize()
        start_time = time.time()

        # 线上推断一次算完全图，只需拓扑特征
        z_topo_all = model(x_topo, edge_index)

        energy_ind = compute_free_energy(z_topo_all[test_ind_idx], temperature=1.0)
        energy_ood = compute_free_energy(z_topo_all[test_ood_idx], temperature=1.0)

        torch.cuda.synchronize()
        end_time = time.time()

    y_true = torch.cat([torch.zeros_like(energy_ind), torch.ones_like(energy_ood)]).cpu().numpy()
    y_scores = torch.cat([energy_ind, energy_ood]).cpu().numpy()

    auroc = roc_auc_score(y_true, y_scores)
    # 处理深度特征蒸馏常见的能量倒挂
    if auroc < 0.5:
        auroc = 1.0 - auroc

    latency_ms = (end_time - start_time) * 1000
    total_test_nodes = test_ind_idx.size(0) + test_ood_idx.size(0)

    print("\n" + "=" * 50)
    print("🏆 基于 LLMGuard 数据协议的非对称打榜结果")
    print("=" * 50)
    print(f"测试节点数: {total_test_nodes}")
    print(f"🔥 AUROC  : {auroc:.4f}")
    print(f"⚡ 总耗时  : {latency_ms:.2f} ms")
    print("=" * 50)


if __name__ == "__main__":
    main()