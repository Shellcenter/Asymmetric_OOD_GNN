# AsymOOD Experiment Archive

**Project**: E:\AsymOOD
**Archive date**: 2026-05-14
**Status**: All 6 experiment stages completed. No training failures.

---

## A. Method Code Files

| File | Purpose |
|------|---------|
| `core_model.py` | Core model components: `AsymmetricGNN` (2-layer GCN + MLP projector), `SupConDistillationLoss`, `IDEnergyBoundaryLoss`, energy/Mahalanobis scoring utilities, and `evaluate_ood_metrics` (AUROC/AUPR/FPR95). |
| `02_train_distill.py` | Phase 2 training: cross-modal distillation of GNN topology embeddings toward frozen LLM semantic anchors, with optional energy-boundary regularization. |
| `03_online_inference.py` | Phase 3 inference: loads a distilled GNN checkpoint and evaluates OOD detection under classifier-MSP or prototype-energy scoring, without LLM dependencies. |
| `06_calibrate_energy.py` | Energy score calibration: grid-search over temperature and logit_scale to optimize OOD detection thresholds on a validation split. |
| `07_robust_calibration.py` | Multi-seed robust calibration: repeated calibration across independent validation splits, reporting mean ± std of AUROC, AUPR, FPR95. |
| `08_mahalanobis_robust_eval.py` | Mahalanobis-distance OOD evaluation: fits class-conditional Gaussian statistics on ID embeddings and scores OOD nodes via Mahalanobis distance. |

## B. New Experiment Scripts

| File | Purpose |
|------|---------|
| `05b_run_baselines_extended.py` | **New.** Trains a standard 2-layer GCN on ID classes only, then evaluates OOD detection with 4 scoring functions: MSP, Entropy, Energy, and MaxLogit. One training per seed, all 3 extended scorers evaluated on the same model. Outputs per-seed logs to `logs/baseline_extended/`. |
| `11_score_diagnostics.py` | **New.** Evaluates different OOD scoring rules (prototype_energy, prototype_msp, classifier_energy, classifier_msp, Mahalanobis, etc.) and computes per-seed diagnostic statistics including score distributions, histograms, and ID/OOD separability metrics. |
| `12_build_paper_tables.py` | **New.** Generates paper-ready tables from log files and result CSVs. Parses experiment logs to extract AUROC/AUPR/FPR95/latency across multiple experiment directories, computes mean ± std, and writes formatted CSV tables for manuscript inclusion. |

## C. Result Artifacts

| Directory | Contents |
|-----------|----------|
| `logs/` | Per-seed training+inference logs for all sensitivity sweeps and baselines. Subdirectories: `sensitivity_semantic_*_h64/`, `sensitivity_prototype_*_h64/`, `sensitivity_energy_*_h64/`, `baseline/`, `baseline_extended/`, and legacy experiment logs. |
| `weights/` | Trained model checkpoints (.pth) mirroring the logs/ directory structure. |
| `results_data/tables/` | All 6 paper-ready CSV tables plus experiment_summary.md. |
| `results_data/baseline_extended/` | Per-seed baseline results in `all_baselines.csv`. |
| `results_data/diagnostics/` | Score distribution diagnostics CSVs. |
| `plots/` | t-SNE visualization: `tsne_ours_light_seed42.pdf`. |
| `sweep_results/` | Legacy energy-boundary sweep summaries (JSON + CSV). |

## D. Paper-Ready Tables

| File | Description |
|------|-------------|
| `results_data/tables/table1_main_comparison.csv` | Main comparison: GCN-MSP, Classifier-only, Ours-light, Ours-energy-light. AUROC/AUPR/FPR95/Latency. |
| `results_data/tables/table2_ablation.csv` | Ablation study: incremental addition of semantic, prototype, and energy components. |
| `results_data/tables/table3_semantic_sensitivity.csv` | Semantic weight sensitivity: 0.002, 0.005 (default), 0.02. All 10 seeds. |
| `results_data/tables/table4_prototype_sensitivity.csv` | Prototype weight sensitivity: 0.005, 0.01 (default), 0.02, 0.05. All 10 seeds. |
| `results_data/tables/table5_energy_sensitivity.csv` | Energy weight sensitivity: 0.0 (default), 0.0005, 0.001, 0.005. All 10 seeds. |
| `results_data/tables/table6_extended_baselines.csv` | Extended baselines: GCN-MSP, GCN-MaxLogit, GCN-Entropy, GCN-Energy. All 10 seeds. |
| `results_data/tables/experiment_summary.md` | Human-readable experiment summary with paper writing cautions. |

## E. Final Recommended Configuration

```
method            = Ours-light
hidden_channels   = 64
classifier_weight = 1.0
semantic_weight   = 0.005
prototype_weight  = 0.01
energy_weight     = 0.0
score_method      = classifier_msp
seeds             = 42, 43, 44, 45, 46, 47, 48, 49, 50, 51
LLM_at_inference  = No
```

**Main results (Ours-light, 10 seeds):**
- AUROC: 0.8054 ± 0.0161
- AUPR:  0.8089 ± 0.0205
- FPR95: 0.5922 ± 0.0300
- Latency: 2.29 ms

## F. Key Paper Claim (cautious wording)

> Ours-light achieves competitive AUROC/AUPR and slightly better FPR95 than the GCN-MSP baseline on Cora leave-out OOD detection, while avoiding any LLM calls during inference. Extended baselines such as GCN-Energy and GCN-MaxLogit remain strong on AUROC/AUPR metrics. The paper should not claim universal superiority over all baselines; instead, the key contribution is achieving competitive OOD detection performance without LLM dependence at inference time.

## G. Cautions for Paper Writing

1. **Do not claim universal superiority over all baselines.** The AUROC/AUPR gains over GCN-MSP are marginal and often within one standard deviation. GCN-Energy (AUROC 0.8165) and GCN-MaxLogit (AUROC 0.8161) actually achieve slightly higher AUROC than Ours-light (0.8054).

2. **Do not promote the old Prototype-energy-only method as the main result.** The main method is Ours-light with classifier_msp scoring. The prototype-energy-only variant is an ablation, not the primary contribution.

3. **Weak energy compactness regularization (energy_weight ≤ 0.001) did not improve FPR95.** Energy regularization at these levels had negligible or slightly negative impact on FPR95. This should be discussed as a supplementary ablation, not a core contribution. The energy_margin and energy_compact_weight results from the sweep are in `sweep_results/`.

4. **Latency results are Cora-scale only.** All methods complete inference in <10 ms on the Cora dataset (2,708 nodes). These latency figures should not be extrapolated to larger graphs without qualification.

5. **Cora is a single-dataset benchmark.** The leave-out protocol (classes 0-3 ID, 4-6 OOD) is well-defined but results may not generalize to other datasets or OOD definitions. This limitation should be acknowledged.

## H. Recommended Next Steps

1. **Freeze experiment results.** The current results are complete and internally consistent. Do not rerun any training or inference unless a reviewer explicitly requests it.
2. **Do not rerun training unless reviewer asks.** If a reviewer asks for additional seeds, a new dataset, or a new baseline, add only that specific experiment without touching existing results.
3. **Prepare manuscript sections from the six tables.** Tables 1–6 in `results_data/tables/` contain all numerical results needed for the main paper and supplementary material. Draft the Results section directly from these CSV files.
4. **Keep logs/ and weights/ locally; only include selected artifacts in the paper submission.** The full `logs/` and `weights/` directories are large. For the paper artifact, include only: the 6 paper tables (`results_data/tables/table*.csv`), the t-SNE plot (`plots/tsne_ours_light_seed42.pdf`), the experiment summary (`results_data/tables/experiment_summary.md`), and the relevant Python scripts (`05b_run_baselines_extended.py`, `11_score_diagnostics.py`, `12_build_paper_tables.py`).
5. **Avoid claiming universal superiority over all baselines.** Refer to Section G for detailed cautions. Frame the contribution as: competitive OOD detection without LLM at inference, with robust hyperparameters and strong baselines documented.
