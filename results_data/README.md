# Categorized Experiment Data

This directory stores cleaned experiment data exported from the energy-boundary sweep.

## Files

- `configs/energy_boundary_configs.csv`: Hyperparameter settings for each energy-boundary run.
- `metrics/single_split_metrics.csv`: Single-split OOD metrics.
- `metrics/robust_calibration_metrics.csv`: Multi-seed robust calibration metrics with mean and standard deviation.
- `training/final_losses.csv`: Final training losses for semantic distillation, prototype classification, and energy regularization.
- `checkpoints/checkpoint_index.csv`: Mapping from configuration names to saved checkpoints.
- `recommendations/main_table_recommendation.json`: Recommended main-table configuration and reproduction commands.

## Recommended Main Result

Use `ew03_m-6` as the main-table method:

- AUROC: `0.8521 +/- 0.0135`
- AUPR: `0.8760 +/- 0.0189`
- FPR@95TPR: `0.6332 +/- 0.0228`

Reproduce with:

```bash
python 02_train_distill.py --energy_weight 0.3 --energy_margin -6.0
python 07_robust_calibration.py --temp_min 0.05 --temp_max 8.0 --temp_steps 120 --logit_scales 5,10,15,20 --num_seeds 7
```

To regenerate these categorized files after a new sweep:

```bash
python 10_export_results_data.py
```
