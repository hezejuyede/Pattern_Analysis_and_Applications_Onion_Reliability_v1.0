"""Recompute compact-release evidence without source image bytes."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
AUDITOR_PATH = ROOT / "audit" / "independent_result_verification" / "independent_verify.py"
SEED = 20260930


def load_auditor():
    spec = importlib.util.spec_from_file_location("slim_independent_auditor", AUDITOR_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def compare_aggregate(raw_path: Path, aggregate_path: Path, group_columns: list[str], extra: dict[str, str]) -> tuple[int, float]:
    raw = pd.read_csv(raw_path)
    reported = pd.read_csv(aggregate_path)
    grouped = raw.groupby(group_columns, sort=True)
    rows = []
    for keys, part in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_columns, keys))
        values = part["value"].to_numpy(float)
        row.update(
            mean=float(np.mean(values)),
            standard_deviation=float(np.std(values, ddof=1)),
            median=float(np.median(values)),
            p025=float(np.quantile(values, 0.025)),
            p975=float(np.quantile(values, 0.975)),
            repetitions=int(part["repetition"].nunique()),
        )
        for output_name, raw_name in extra.items():
            row[output_name] = float(part[raw_name].mean())
        rows.append(row)
    recomputed = pd.DataFrame(rows)
    merged = reported.merge(recomputed, on=group_columns, suffixes=("_reported", "_recomputed"), validate="one_to_one")
    numeric = ["mean", "standard_deviation", "median", "p025", "p975", "repetitions", *extra]
    differences = []
    for name in numeric:
        differences.append((merged[f"{name}_reported"] - merged[f"{name}_recomputed"]).abs().max())
    return len(merged), float(max(differences))


def main() -> None:
    auditor = load_auditor()
    manifest = pd.read_csv(ROOT / "manifests" / "unified_manifest.csv")
    comparison, checks, predictions = auditor.verify_predictions(manifest)
    _, paired_comparison = auditor.paired_bootstrap(predictions)
    paired_max = float(paired_comparison.filter(like="_abs_diff").max().max())

    split = pd.read_csv(ROOT / "results" / "reliability_benchmark_v1" / "split_manifests" / "tom_acquisition_day_outer_folds.csv")
    y = split.foliar_binary.astype(int).to_numpy()
    groups = split.acquisition_day_utc.astype(str).to_numpy()
    recreated = np.full(len(split), -1, dtype=int)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    for fold, (_, test) in enumerate(splitter.split(np.zeros(len(y)), y, groups)):
        recreated[test] = fold
    fold_checks = [
        not split.sample_id.duplicated().any(),
        split.groupby("acquisition_day_utc").outer_fold.nunique().eq(1).all(),
        split.groupby("outer_fold").foliar_binary.nunique().eq(2).all(),
        np.array_equal(recreated, split.outer_fold.to_numpy()),
    ]

    primary = pd.read_csv(ROOT / "results" / "reliability_benchmark_v1" / "audit" / "primary_development_external_sha256.csv")
    included = primary[primary.analysis_inclusion.isin(["development_included", "external_included"])]
    tom = set(included.loc[included.dataset.eq("TOM2024_A_BurkinaFaso"), "analysis_sha256"])
    cold = set(included.loc[included.dataset.eq("COLD_raw_India"), "analysis_sha256"])
    ledger_checks = [len(tom & cold) == 0, len(tom) == 1643, len(cold) == 813]

    cold_rows, cold_max = compare_aggregate(
        ROOT / "results/cold_augmented_leakage_ablation_v1/tables/repeated_split_metrics.csv",
        ROOT / "results/cold_augmented_leakage_ablation_v1/tables/aggregate_split_summary.csv",
        ["split_scheme", "feature_model", "metric"],
        {"mean_test_files": "test_files", "mean_test_families": "test_families", "mean_contamination_fraction": "test_family_contamination_fraction"},
    )
    tom_rows, tom_max = compare_aggregate(
        ROOT / "results/tom_split_severity_sensitivity_v1/tables/repeated_split_metrics.csv",
        ROOT / "results/tom_split_severity_sensitivity_v1/tables/aggregate_split_summary.csv",
        ["split_scheme", "metric"],
        {"mean_test_files": "test_files", "mean_source_contamination": "test_source_contamination_fraction", "mean_day_contamination": "test_day_contamination_fraction"},
    )
    check_frame = pd.DataFrame(checks)
    passed = bool(
        check_frame.passed.all()
        and all(fold_checks)
        and all(ledger_checks)
        and paired_max <= 1e-12
        and cold_max <= 1e-12
        and tom_max <= 1e-12
    )
    result = {
        "status": "PASS" if passed else "FAIL",
        "scope": "Frozen numerical evidence in the compact release; source image bytes are not redistributed.",
        "metric_rows_recomputed": int(len(comparison)),
        "metric_max_estimate_abs_difference": float(comparison.estimate_abs_diff.max()),
        "metric_max_interval_abs_difference": float(max(comparison.ci_low_abs_diff.max(), comparison.ci_high_abs_diff.max())),
        "prediction_checks_passed": int(check_frame.passed.sum()),
        "prediction_checks_total": int(len(check_frame)),
        "paired_cold_max_abs_difference": paired_max,
        "locked_fold_checks_passed": int(sum(fold_checks)),
        "locked_fold_checks_total": len(fold_checks),
        "hash_ledger_checks_passed": int(sum(ledger_checks)),
        "hash_ledger_checks_total": len(ledger_checks),
        "cold_ablation_aggregate_rows": cold_rows,
        "cold_ablation_aggregate_max_abs_difference": cold_max,
        "tom_sensitivity_aggregate_rows": tom_rows,
        "tom_sensitivity_aggregate_max_abs_difference": tom_max,
        "image_byte_hash_verification": "NOT_RUN_OFFLINE",
    }
    output = ROOT / "verification_output" / "slim_evidence_recomputation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
