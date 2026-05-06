import torch
import torch.nn as nn
import torch.nn.functional as F


class SupervisedContrastiveDistillationLoss(nn.Module):
    def __init__(self, margin=2.0):
        super().__init__()
        self.margin = margin

    def forward(self, z_topo, z_sem, labels):
        """
        Z_topo: GNN动态输出 [N, 768]
        Z_sem: LLM静态锚点 [N, 768]
        labels: 0代表正常ID，1代表异常OOD
        """
        # 计算欧氏距离
        dist = F.pairwise_distance(z_topo, z_sem, p=2)

        # ID Pull: 正常节点，强迫拓扑特征死死咬住大模型的语义特征
        id_mask = (labels == 0)
        loss_id = torch.mean(dist[id_mask] ** 2) if id_mask.sum() > 0 else 0.0

        # OOD Push: 异常节点，将其推离大模型的正常语义流形
        ood_mask = (labels == 1)
        loss_ood = torch.mean(F.relu(self.margin - dist[ood_mask]) ** 2) if ood_mask.sum() > 0 else 0.0

        # 总损失
        return loss_id + loss_ood