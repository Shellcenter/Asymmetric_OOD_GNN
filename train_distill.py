import torch
import torch.optim as optim
from models.gnn_encoder import GNNBackbone
from models.mlp_projector import MLPProjector
from utils.loss import SupervisedContrastiveDistillationLoss
from torch_geometric.data import Data

# ================= 1. 模拟数据环境准备 =================
print(">>> [环境构建] 正在生成模拟的图数据与 LLM 锚点...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_nodes = 100
input_feat_dim = 64
# 1. 模拟节点原始特征与结构
x = torch.randn((num_nodes, input_feat_dim)).to(device)
edge_index = torch.randint(0, num_nodes, (2, 300)).to(device)
# 2. 模拟真实标签 (0: ID 正常, 1: OOD 异常)
labels = torch.randint(0, 2, (num_nodes,)).to(device)
# 3. 模拟离线提取好的 LLM 静态锚点 Z_sem (也就是上一节生成的 .pt 文件)
# 实际科研中应为: z_sem_anchor = torch.load("embeddings/Z_sem_anchor.pt").to(device)
z_sem_anchor = torch.randn((num_nodes, 768)).to(device)

# ================= 2. 初始化网络与优化器 =================
print(">>> [网络初始化] 构建 GNN Backbone + MLP Projector...")
gnn = GNNBackbone(input_dim=input_feat_dim, hidden_dim=128, output_dim=128).to(device)
mlp = MLPProjector(input_dim=128, hidden_dim=256, output_dim=768).to(device)

optimizer = optim.Adam(list(gnn.parameters()) + list(mlp.parameters()), lr=0.001)
criterion = SupervisedContrastiveDistillationLoss(margin=5.0)

# ================= 3. 跨模态蒸馏训练循环 =================
print(">>> [Phase 2] 开始全监督跨模态蒸馏训练...")
epochs = 50
gnn.train()
mlp.train()

for epoch in range(1, epochs + 1):
    optimizer.zero_grad()

    # 动态侧：GNN + MLP 提取 Z_topo
    h_topo = gnn(x, edge_index)
    z_topo = mlp(h_topo)

    # 对齐侧：计算对比损失 (Z_topo 动态拟合，Z_sem 静止不动)
    loss = criterion(z_topo, z_sem_anchor, labels)

    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch:03d}/{epochs} | Distillation Loss: {loss.item():.4f}")

# 保存“出师”后的模型权重
torch.save(gnn.state_dict(), "models/gnn_trained.pth")
torch.save(mlp.state_dict(), "models/mlp_trained.pth")
print(">>> [Phase 2 完成] 模型已固化，LLM 功成身退！")