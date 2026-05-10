"""Export experiment summaries into categorized data files.

The sweep script stores a wide summary table. This utility splits that table
into focused CSV/JSON files that are easier to use for paper tables, plots, and
checkpoint bookkeeping.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Any


MAIN_TABLE_CONFIG = "ew03_m-6"


def read_rows(path: str) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def as_float(value: Any) -> float:
    return float(value)


def select_fields(rows: list[dict[str, Any]], fields: list[str]) -> list[dict[str, Any]]:
    return [{field: row[field] for field in fields} for row in rows]


def write_csv(path: str, rows: list[dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: str, payload: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def format_metric(mean: str, std: str) -> str:
    return f"{as_float(mean):.4f} +/- {as_float(std):.4f}"


def build_main_recommendation(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_name = {row["name"]: row for row in rows}
    if MAIN_TABLE_CONFIG not in by_name:
        raise KeyError(f"Missing main-table config: {MAIN_TABLE_CONFIG}")

    main_row = by_name[MAIN_TABLE_CONFIG]
    best_fpr_row = min(rows, key=lambda row: (as_float(row["FPR95_mean"]), -as_float(row["AUROC_mean"])))
    return {
        "main_table_config": MAIN_TABLE_CONFIG,
        "selection_reason": (
            "ew03_m-6 has the strongest AUROC/AUPR and nearly identical FPR95 to the "
            "best-FPR configuration, with lower FPR95 variance."
        ),
        "recommended_training_command": "python 02_train_distill.py --energy_weight 0.3 --energy_margin -6.0",
        "recommended_calibration_command": (
            "python 07_robust_calibration.py --temp_min 0.05 --temp_max 8.0 "
            "--temp_steps 120 --logit_scales 5,10,15,20 --num_seeds 7"
        ),
        "main_metrics": {
            "AUROC": format_metric(main_row["AUROC_mean"], main_row["AUROC_std"]),
            "AUPR": format_metric(main_row["AUPR_mean"], main_row["AUPR_std"]),
            "FPR95": format_metric(main_row["FPR95_mean"], main_row["FPR95_std"]),
            "single_split_AUROC": as_float(main_row["single_AUROC"]),
            "single_split_AUPR": as_float(main_row["single_AUPR"]),
            "single_split_FPR95": as_float(main_row["single_FPR95"]),
        },
        "best_fpr_config": {
            "name": best_fpr_row["name"],
            "AUROC": format_metric(best_fpr_row["AUROC_mean"], best_fpr_row["AUROC_std"]),
            "AUPR": format_metric(best_fpr_row["AUPR_mean"], best_fpr_row["AUPR_std"]),
            "FPR95": format_metric(best_fpr_row["FPR95_mean"], best_fpr_row["FPR95_std"]),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export categorized experiment result files.")
    parser.add_argument("--summary_csv", type=str, default="./sweep_results/energy_boundary/summary.csv")
    parser.add_argument("--output_dir", type=str, default="./results_data")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_rows(args.summary_csv)

    configs = select_fields(
        rows,
        ["name", "energy_weight", "energy_margin", "energy_compact_weight"],
    )
    single_metrics = select_fields(
        rows,
        ["name", "single_AUROC", "single_AUPR", "single_FPR95"],
    )
    robust_metrics = select_fields(
        rows,
        [
            "name",
            "recommended_logit_scale",
            "recommended_temperature",
            "AUROC_mean",
            "AUROC_std",
            "AUPR_mean",
            "AUPR_std",
            "FPR95_mean",
            "FPR95_std",
        ],
    )
    training_losses = select_fields(
        rows,
        [
            "name",
            "final_loss",
            "final_semantic",
            "final_prototype",
            "final_energy_boundary",
            "final_energy_compact",
        ],
    )
    checkpoints = select_fields(
        rows,
        ["name", "checkpoint"],
    )

    write_csv(os.path.join(args.output_dir, "configs", "energy_boundary_configs.csv"), configs)
    write_csv(os.path.join(args.output_dir, "metrics", "single_split_metrics.csv"), single_metrics)
    write_csv(os.path.join(args.output_dir, "metrics", "robust_calibration_metrics.csv"), robust_metrics)
    write_csv(os.path.join(args.output_dir, "training", "final_losses.csv"), training_losses)
    write_csv(os.path.join(args.output_dir, "checkpoints", "checkpoint_index.csv"), checkpoints)
    write_json(
        os.path.join(args.output_dir, "recommendations", "main_table_recommendation.json"),
        build_main_recommendation(rows),
    )

    print(f"Exported categorized result files to: {args.output_dir}")


if __name__ == "__main__":
    main()
