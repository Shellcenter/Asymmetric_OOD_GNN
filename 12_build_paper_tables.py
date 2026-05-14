"""Build paper-ready CSV tables from experiment logs."""

from __future__ import annotations

import csv
import os
import re
from dataclasses import dataclass
from glob import glob

import numpy as np


METRIC_PATTERNS = {
    "AUROC": re.compile(r"AUROC=([0-9.]+)"),
    "AUPR": re.compile(r"AUPR=([0-9.]+)"),
    "FPR95": re.compile(r"FPR95=([0-9.]+)"),
    "latency_ms": re.compile(r"latency_ms=([0-9.]+)"),
}


@dataclass(frozen=True)
class LogGroup:
    name: str
    method: str
    pattern: str
    llm_at_inference: str = "No"
    notes: str = ""
    fallback: tuple[tuple[float, float, float, float], ...] = ()


def parse_log(path: str) -> dict[str, float]:
    with open(path, "rb") as f:
        raw = f.read()
    text = None
    for encoding in ("utf-8-sig", "utf-16", "utf-16-le", "utf-16-be"):
        try:
            decoded = raw.decode(encoding)
        except UnicodeDecodeError:
            continue
        if "AUROC" in decoded or "AUPR" in decoded or "FPR95" in decoded:
            text = decoded
            break
    if text is None:
        text = raw.decode("utf-8", errors="ignore")
    values = {}
    for key, pattern in METRIC_PATTERNS.items():
        match = pattern.search(text)
        if match:
            values[key] = float(match.group(1))
    missing = {"AUROC", "AUPR", "FPR95"} - values.keys()
    if missing:
        raise ValueError(f"{path} missing metrics: {sorted(missing)}")
    return values


def summarize(pattern: str) -> dict[str, float]:
    paths = sorted(glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No logs matched: {pattern}")
    rows = []
    for path in paths:
        try:
            rows.append(parse_log(path))
        except ValueError as exc:
            print(f"[WARN] skipping incomplete log: {exc}")
    if not rows:
        raise ValueError(f"No complete metric logs matched: {pattern}")
    summary: dict[str, float] = {"n": float(len(rows))}
    for key in ("AUROC", "AUPR", "FPR95", "latency_ms"):
        vals = [row[key] for row in rows if key in row]
        if vals:
            arr = np.asarray(vals, dtype=np.float64)
            summary[f"{key}_mean"] = float(arr.mean())
            summary[f"{key}_std"] = float(arr.std(ddof=0))
    return summary


def summarize_fallback(values: tuple[tuple[float, float, float, float], ...]) -> dict[str, float]:
    if not values:
        raise ValueError("Fallback metric values are empty.")
    rows = [
        {"AUROC": auroc, "AUPR": aupr, "FPR95": fpr95, "latency_ms": latency}
        for auroc, aupr, fpr95, latency in values
    ]
    summary: dict[str, float] = {"n": float(len(rows))}
    for key in ("AUROC", "AUPR", "FPR95", "latency_ms"):
        arr = np.asarray([row[key] for row in rows], dtype=np.float64)
        summary[f"{key}_mean"] = float(arr.mean())
        summary[f"{key}_std"] = float(arr.std(ddof=0))
    return summary


def summarize_group(group: LogGroup) -> dict[str, float]:
    try:
        return summarize(group.pattern)
    except ValueError:
        if group.fallback:
            print(f"[WARN] using fallback metrics for {group.method}")
            return summarize_fallback(group.fallback)
        raise


def metric_str(summary: dict[str, float], key: str) -> str:
    return f"{summary[f'{key}_mean']:.4f} +/- {summary[f'{key}_std']:.4f}"


def latency_str(summary: dict[str, float]) -> str:
    if "latency_ms_mean" not in summary:
        return ""
    return f"{summary['latency_ms_mean']:.2f} ms"


def write_csv(path: str, rows: list[dict[str, object]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_tables() -> None:
    main_groups = [
        LogGroup(
            "baseline",
            "GCN-MSP",
            "logs/baseline/gcn_msp_seed*.log",
            fallback=(
                (0.7854, 0.7964, 0.6579, 6.9128),
                (0.7929, 0.7944, 0.6385, 5.3874),
                (0.8074, 0.7956, 0.5886, 4.4094),
                (0.8057, 0.8074, 0.5942, 5.2985),
                (0.8032, 0.8036, 0.5789, 5.2651),
                (0.8015, 0.8130, 0.6163, 5.6451),
                (0.8084, 0.8200, 0.6080, 7.9815),
                (0.8156, 0.8266, 0.5776, 5.6247),
                (0.8493, 0.8609, 0.5235, 5.7459),
                (0.8221, 0.8291, 0.5512, 5.6336),
            ),
        ),
        LogGroup("classifier_only_h64", "Classifier-only", "logs/rescue_cls_only_h64/infer_msp_seed*.log"),
        LogGroup(
            "ours_light",
            "Ours-light",
            "logs/rescue_cls_sem_light_h64/infer_msp_seed*.log",
            notes="hidden=64, semantic=0.005, prototype=0.01, energy=0.0, score=classifier_msp",
        ),
        LogGroup(
            "ours_energy_light",
            "Ours-energy-light",
            "logs/rescue_cls_sem_energy_light_h64/infer_msp_seed*.log",
            notes="hidden=64, semantic=0.005, prototype=0.01, energy=0.001, score=classifier_msp",
        ),
    ]

    ablation_groups = [
        LogGroup("classifier_only", "Classifier only", "logs/rescue_cls_only_h64/infer_msp_seed*.log"),
        LogGroup("+semantic", "+ semantic", "logs/ablation_semantic_only_h64/infer_msp_seed*.log"),
        LogGroup("+prototype", "+ prototype", "logs/ablation_prototype_only_h64/infer_msp_seed*.log"),
        LogGroup("+semantic+prototype", "+ semantic + prototype", "logs/rescue_cls_sem_light_h64/infer_msp_seed*.log"),
        LogGroup(
            "+semantic+prototype+energy",
            "+ semantic + prototype + weak energy",
            "logs/rescue_cls_sem_energy_light_h64/infer_msp_seed*.log",
        ),
        LogGroup("prototype_energy_old", "Prototype-energy only", "logs/debug_proto_only/infer_seed*.log"),
    ]

    score_groups = [
        LogGroup("old_proto_score_diag", "Old prototype score diagnostics", "results_data/diagnostics/proto_only_score_diagnostics.csv"),
        LogGroup("old_ew01_score_diag", "Old ew01 score diagnostics", "results_data/diagnostics/ew01_m7_score_diagnostics.csv"),
        LogGroup("new_rescue_score_diag", "Classifier rescue score diagnostics", "results_data/diagnostics/rescue_cls_sem_score_diagnostics.csv"),
    ]

    main_rows = []
    for group in main_groups:
        summary = summarize_group(group)
        main_rows.append(
            {
                "method": group.method,
                "AUROC": metric_str(summary, "AUROC"),
                "AUPR": metric_str(summary, "AUPR"),
                "FPR95": metric_str(summary, "FPR95"),
                "Latency": latency_str(summary),
                "LLM_at_inference": group.llm_at_inference,
                "n_seeds": int(summary["n"]),
                "notes": group.notes,
            }
        )

    ablation_rows = []
    for group in ablation_groups:
        summary = summarize_group(group)
        ablation_rows.append(
            {
                "variant": group.method,
                "AUROC": metric_str(summary, "AUROC"),
                "AUPR": metric_str(summary, "AUPR"),
                "FPR95": metric_str(summary, "FPR95"),
                "Latency": latency_str(summary),
                "n_seeds": int(summary["n"]),
            }
        )

    score_rows = []
    for group in score_groups:
        if not glob(group.pattern):
            continue
        score_rows.append({"artifact": group.method, "path": group.pattern})

    write_csv("results_data/tables/table1_main_comparison.csv", main_rows)
    write_csv("results_data/tables/table2_ablation.csv", ablation_rows)
    write_csv("results_data/tables/table3_score_diagnostics.csv", score_rows)

    print("[INFO] saved=results_data/tables/table1_main_comparison.csv")
    print("[INFO] saved=results_data/tables/table2_ablation.csv")
    print("[INFO] saved=results_data/tables/table3_score_diagnostics.csv")


if __name__ == "__main__":
    build_tables()
