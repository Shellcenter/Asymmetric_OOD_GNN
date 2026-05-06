import torch
from sklearn.metrics import roc_auc_score


def compute_free_energy(z_topo, temperature=1.0):
    """
    非对称推断阶段：基于能量的 OOD 打分器
    公式：E(v) = -T * log(sum(exp(Z / T)))
    """
    # 能量越低越像正常节点，能量越高越像 OOD 异常
    energy_scores = -temperature * torch.logsumexp(z_topo / temperature, dim=1)
    return energy_scores


def compute_auroc(energy_scores, labels):
    """
    根据能量得分计算 AUROC
    注意：能量越高代表越可能是 OOD（label=1）
    """
    # 转换为 numpy
    scores = energy_scores.detach().cpu().numpy()
    y_true = labels.detach().cpu().numpy()

    try:
        auroc = roc_auc_score(y_true, scores)
    except ValueError:
        auroc = 0.5  # 防止纯单一样本报错
    return auroc