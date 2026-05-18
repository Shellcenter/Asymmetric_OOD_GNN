# 当前状态摘要 — 请 Opus 4.6 结合实验数据决定下一步

## 项目背景

这是一个图 OOD 检测（Graph Out-of-Distribution Detection）的硕士论文实验项目。核心思路：

> 用论文的文本语义嵌入（sentence-transformer）作为蒸馏目标，训练一个 GNN，使其推理时不需要文本就能做 OOD 检测。

## 发现并修复的问题

### 1. LLM 造假 → 已修复
原代码用 `nn.Linear(1433, 64, bias=False)` 随机矩阵模拟"LLM"。已替换为 `all-MiniLM-L6-v2` 真实 384 维 sentence-transformer 嵌入。

### 2. 损失函数欺诈 → 已修复
原 `SupConDistillationLoss` 声称有 "OOD push" 项，但训练集从不含 OOD 节点，该项永远为 0。已重命名为 `SemanticAlignmentLoss`，只做 ID 节点的 L2 对齐。

### 3. Cora 性能饱和 → 已切换到 ArXiv
Cora 上所有方法 AUROC 挤在 0.805±0.016，无法区分。已迁移到 OGB-ArXiv（169K 节点），用年份做 ID/OOD 分割（≤2015 vs ≥2018）。

### 4. 推理打分与 baseline 相同 → 已修复
原论文表格用 `classifier_msp` 打分，和 GCN-MSP baseline 一样。已改为默认 `prototype_energy` 打分。

## 关键实验数据

### Cora（2708 节点，标签分割 ID/OOD）

| 方法 | 打分 | AUROC |
|---|---|---|
| GCN-MSP | MSP | 0.8092 ± 0.0166 |
| Classifier-only | MSP | 0.8053 ± 0.0165 |
| Ours-light | MSP | 0.8054 ± 0.0161 |
| Ours-energy-light | MSP | 0.8051 ± 0.0162 |

**问题：所有方法在标准差内完全重叠。数据集无法区分方法优劣。**

### ArXiv（169K 节点，年份分割 ID/OOD）

Extended baseline（同一个 GCN，不同打分）：

| 打分 | AUROC |
|---|---|
| MSP | 0.6034 |
| Entropy | 0.6104 |
| Energy (classifier logits) | 0.6094 |
| MaxLogit | 0.6112 |

**好消息：ArXiv 上 AUROC ~0.61，远离天花板，方法间有区分度。**

### 核心对比实验（ArXiv，prototype_energy 打分，单 seed 42）

| 方法 | 训练损失 | 打分 | AUROC |
|---|---|---|---|
| PlainGCN | CE only | prototype_energy | 0.5528 |
| Ours (no distill) | CE + prototype CE | prototype_energy | 0.5304 |
| Ours (full) | CE + prototype CE + semantic alignment | prototype_energy | 0.5319 |

**坏消息：our 方法的 prototype_energy 打分反而比 PlainGCN 差（0.53 vs 0.55），语义蒸馏无明显增益（0.5304 vs 0.5319）。**

注意：用 classifier_energy 打分时纯 GCN 可达 0.6094，说明 prototype_energy 本身可能就不适合 40 类场景（类太多，原型判别力弱）。

## 需要你做的判断

1. **语义蒸馏是否还有路可走？** 当前结果是负的——蒸馏没有帮助 OOD 检测。是改蒸馏策略（互信息？对比学习？），还是承认这个思路在当前设置下无效？

2. **应该用哪种 OOD 打分？** prototype_energy（0.55）远低于 classifier_energy（0.61）。如果 semantic alignment 对 classifier_energy 有帮助，那应该用 classifier_energy 重做对比。但这样"prototype"这条故事线就断了。

3. **这个论文还能成立吗？** 如果最终所有方法在 ArXiv 上也趋于一致（虽然目前看起来不会），或者语义蒸馏始终没有正增益，论文的贡献是什么？

4. **下一步实验怎么跑？**
   - 选项 A：用 classifier_energy 打分，重做三组对比，看 semantic distill 是否有正则化增益
   - 选项 B：改进蒸馏方式（contrastive / mutual information）
   - 选项 C：接受 prototype_energy 不如 classifier_energy，把蒸馏作为 aux loss，以 classifier MSP/Energy 为主结果
   - 选项 D：多跑几个 seed（当前只跑了 seed 42），确认负结果是稳定的

请结合以上全部信息，决定下一步实验方案。
