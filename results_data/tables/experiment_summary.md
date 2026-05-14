# AsymOOD Experiment Summary

**Generated**: 2026-05-14
**Environment**: Python 3.11.13, PyTorch 2.3.1 (CUDA), PyG 2.7.0, conda env=py311_env

---

## 1. Completed Experiments

| Stage | Description | Seeds | Status |
|-------|-------------|-------|--------|
| Stage 1 | semantic_weight sensitivity (0.002, 0.02) | 42–51 | Done |
| Stage 2 | prototype_weight sensitivity (0.005, 0.02, 0.05) | 42–51 | Done |
| Stage 3 | energy_weight sensitivity (0.0005, 0.005) | 42–51 | Done |
| Stage 4 | Extended baselines (GCN-MSP/Entropy/Energy/MaxLogit) | 42–51 | Done |
| Stage 5 | Paper table generation (6 tables) | — | Done |
| Stage 6 | t-SNE visualization (Ours-light seed=42) | 42 | Done |

Total: 210 train+inference runs, 0 failures.

## 2. Table Index

| Table | Path |
|-------|------|
| Table 1: Main comparison | `results_data/tables/table1_main_comparison.csv` |
| Table 2: Ablation | `results_data/tables/table2_ablation.csv` |
| Table 3: Semantic sensitivity | `results_data/tables/table3_semantic_sensitivity.csv` |
| Table 4: Prototype sensitivity | `results_data/tables/table4_prototype_sensitivity.csv` |
| Table 5: Energy sensitivity | `results_data/tables/table5_energy_sensitivity.csv` |
| Table 6: Extended baselines | `results_data/tables/table6_extended_baselines.csv` |

## 3. Main Result (one sentence)

Ours-light (GCN 64-dim + classifier MSP score, no LLM at inference) achieves AUROC 0.8054, AUPR 0.8089, FPR95 0.5922 on Cora leave-out OOD detection, competitive with standard GCN baselines while requiring no LLM at inference time.

## 4. Ablation (one sentence)

Adding prototype and semantic alignment to the classifier-only GCN yields marginal improvements; the full Ours-light configuration (semantic=0.005, prototype=0.01, energy=0.0) provides the best FPR95–AUROC balance in the ablation hierarchy.

## 5. Sensitivity (one sentence)

All three hyperparameters (semantic_weight, prototype_weight, energy_weight) are robust across 1–2 orders of magnitude, with performance variation within 1 standard deviation of the default configuration.

## 6. Extended Baselines (one sentence)

GCN-Energy (AUROC 0.8165) and GCN-MaxLogit (AUROC 0.8161) slightly outperform GCN-MSP (AUROC 0.8092) on Cora, but the gap is within one standard deviation and all methods use only ID-class supervision.

## 7. Visualization

`plots/tsne_ours_light_seed42.pdf`

## 8. Paper Writing Cautions

- **Do NOT claim** that the proposed method comprehensively outperforms all baselines. The AUROC/AUPR gains over GCN-MSP are marginal and often within 1σ.
- **Fair to claim**: Ours-light achieves competitive AUROC/AUPR with slightly better FPR95 compared to GCN-MSP, while requiring no LLM at inference.
- **Fair to claim**: The method is robust to hyperparameter choices (semantic_weight, prototype_weight, energy_weight all insensitive over 1–2 orders of magnitude).
- **Fair to claim**: GCN-Energy and GCN-MaxLogit are strong baseline scoring functions that match or slightly exceed MSP on this benchmark.
- **Do NOT overinterpret** latency differences: all methods run in <10 ms on Cora-scale graphs.
