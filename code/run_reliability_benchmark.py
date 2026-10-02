"""Leakage-controlled cross-country onion reliability benchmark.

The design is intentionally asymmetric. TOM2024 (Burkina Faso) is the only
development domain. Model selection, probability calibration, and threshold
selection use acquisition-day-grouped validation within TOM2024. COLD (India)
is opened only for the locked external evaluation. Digital Green onion images
are reported as a small operational case series, and the non-onion sample is
used only as an open-set/OOD stress test.

No augmented image collection is used in the primary external validation:
filename-defined augmentation families are auditable, but those images are
derived and non-independent, and the raw-parent crosswalk is incomplete for
one IYSV family. A separate family-safe leakage ablation audits that archive.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "reliability_benchmark_v1"
SEED = 20260930
C_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
MODEL_SPECS = {
    "prior_prevalence": {"feature": None, "kind": "prior", "role": "baseline"},
    "color_shortcut_logit": {"feature": "color_shortcut", "kind": "logit", "role": "shortcut_baseline"},
    "handcrafted_logit": {"feature": "handcrafted", "kind": "logit", "role": "classical_baseline"},
    "resnet18_logit": {"feature": "resnet18", "kind": "logit", "role": "a_priori_primary"},
    "efficientnet_b0_logit": {
        "feature": "efficientnet_b0",
        "kind": "logit",
        "role": "secondary_architecture_sensitivity",
    },
    "convnext_tiny_logit": {
        "feature": "convnext_tiny",
        "kind": "logit",
        "role": "secondary_architecture_sensitivity",
    },
    "swin_t_logit": {"feature": "swin_t", "kind": "logit", "role": "secondary_transformer_sensitivity"},
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_dump(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_feature(name: str, expected_ids: list[str]) -> np.ndarray:
    path = ROOT / "features" / f"{name}.npz"
    archive = np.load(path, allow_pickle=False)
    identifiers = archive["sample_ids"].astype(str).tolist()
    if identifiers != expected_ids:
        raise RuntimeError(f"{name} feature order does not match the frozen manifest")
    matrix = archive["features"].astype(np.float64, copy=False)
    if not np.isfinite(matrix).all():
        raise RuntimeError(f"{name} contains non-finite features")
    return matrix


def acquisition_day(source_group: pd.Series) -> pd.Series:
    timestamps = source_group.str.extract(r"tom_timestamp:(1\d{12})$", expand=False)
    if timestamps.isna().any():
        missing = source_group[timestamps.isna()].head().tolist()
        raise RuntimeError(f"cannot parse TOM2024 source timestamps: {missing}")
    return pd.to_datetime(timestamps.astype("int64"), unit="ms", utc=True).dt.strftime("%Y-%m-%d")


def fixed_group_folds(y: np.ndarray, groups: np.ndarray, n_splits: int, seed: int):
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = list(splitter.split(np.zeros(len(y)), y, groups))
    for fold_index, (train, test) in enumerate(folds):
        train_groups = set(groups[train])
        test_groups = set(groups[test])
        overlap = train_groups.intersection(test_groups)
        if overlap:
            raise RuntimeError(f"group leakage in fold {fold_index}: {sorted(overlap)[:3]}")
        if len(np.unique(y[train])) != 2 or len(np.unique(y[test])) != 2:
            raise RuntimeError(f"fold {fold_index} lacks a binary class")
    return folds


def make_pipeline(c_value: float = 1.0) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=c_value,
                    penalty="l2",
                    solver="liblinear",
                    class_weight="balanced",
                    random_state=SEED,
                    max_iter=10000,
                ),
            ),
        ]
    )


@dataclass
class FittedModel:
    name: str
    feature_name: str | None
    estimator: Pipeline | None
    calibrator: LogisticRegression | None
    threshold: float
    source_prior: float
    oof_score: np.ndarray
    oof_probability: np.ndarray
    oof_threshold: np.ndarray
    oof_fold: np.ndarray
    final_c: float | None
    outer_best_c: list[float]

    def predict(self, matrix: np.ndarray | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.estimator is None:
            if matrix is None:
                raise ValueError("a length-carrying matrix is required for the prior model")
            probability = np.full(matrix.shape[0], self.source_prior, dtype=float)
            score = np.full(matrix.shape[0], math.log(self.source_prior / (1.0 - self.source_prior)), dtype=float)
        else:
            score = self.estimator.decision_function(matrix)
            probability = self.calibrator.predict_proba(np.asarray(score).reshape(-1, 1))[:, 1]
        prediction = (probability >= self.threshold).astype(int)
        return np.asarray(score), np.asarray(probability), prediction


def save_model_artifact(model: FittedModel, path: Path) -> None:
    """Serialize a plain mapping so the artifact loads outside ``__main__``."""
    payload = {
        "artifact_schema": "onion_reliability_fitted_model_v2",
        "state": {
            "name": model.name,
            "feature_name": model.feature_name,
            "estimator": model.estimator,
            "calibrator": model.calibrator,
            "threshold": model.threshold,
            "source_prior": model.source_prior,
            "oof_score": model.oof_score,
            "oof_probability": model.oof_probability,
            "oof_threshold": model.oof_threshold,
            "oof_fold": model.oof_fold,
            "final_c": model.final_c,
            "outer_best_c": model.outer_best_c,
        },
    }
    joblib.dump(payload, path, compress=3)


def load_model_artifact(path: Path) -> FittedModel:
    payload = joblib.load(path)
    if not isinstance(payload, dict) or payload.get("artifact_schema") != "onion_reliability_fitted_model_v2":
        raise RuntimeError(f"unsupported model artifact schema: {path}")
    return FittedModel(**payload["state"])


def tune_pipeline(
    matrix: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    folds,
    n_jobs: int,
) -> tuple[Pipeline, float, pd.DataFrame]:
    grid = GridSearchCV(
        estimator=make_pipeline(),
        param_grid={"classifier__C": list(C_GRID)},
        scoring={"balanced_accuracy": "balanced_accuracy", "roc_auc": "roc_auc"},
        refit="balanced_accuracy",
        cv=folds,
        n_jobs=n_jobs,
        return_train_score=False,
        error_score="raise",
    )
    grid.fit(matrix, y, groups=groups)
    results = pd.DataFrame(grid.cv_results_)
    keep = [
        "param_classifier__C",
        "mean_test_balanced_accuracy",
        "std_test_balanced_accuracy",
        "rank_test_balanced_accuracy",
        "mean_test_roc_auc",
        "std_test_roc_auc",
        "rank_test_roc_auc",
        "mean_fit_time",
        "mean_score_time",
    ]
    return grid.best_estimator_, float(grid.best_params_["classifier__C"]), results[keep]


def fit_nested_model(
    name: str,
    feature_name: str,
    matrix: np.ndarray,
    y: np.ndarray,
    day_groups: np.ndarray,
    outer_folds,
    n_jobs: int,
) -> tuple[FittedModel, list[pd.DataFrame], pd.DataFrame]:
    oof_score = np.full(len(y), np.nan, dtype=float)
    oof_probability = np.full(len(y), np.nan, dtype=float)
    oof_threshold = np.full(len(y), np.nan, dtype=float)
    oof_fold = np.full(len(y), -1, dtype=int)
    outer_best_c: list[float] = []
    outer_tables: list[pd.DataFrame] = []
    for fold_number, (outer_train, outer_test) in enumerate(outer_folds):
        inner_groups = day_groups[outer_train]
        inner_y = y[outer_train]
        inner_folds_local = fixed_group_folds(inner_y, inner_groups, n_splits=4, seed=SEED + 101 + fold_number)
        estimator, best_c, table = tune_pipeline(
            matrix[outer_train], inner_y, inner_groups, inner_folds_local, n_jobs=n_jobs
        )
        # Generate calibration scores using only the outer-training partition.
        # Each score is produced by a model that did not see that inner fold;
        # the outer test days enter neither base fitting nor calibration nor
        # threshold selection.
        inner_oof_score = np.full(len(outer_train), np.nan, dtype=float)
        for inner_train, inner_test in inner_folds_local:
            inner_estimator = make_pipeline(best_c)
            inner_estimator.fit(matrix[outer_train][inner_train], inner_y[inner_train])
            inner_oof_score[inner_test] = inner_estimator.decision_function(matrix[outer_train][inner_test])
        if not np.isfinite(inner_oof_score).all():
            raise RuntimeError(f"incomplete inner OOF calibration scores for {name}, fold {fold_number}")
        fold_calibrator = LogisticRegression(C=1e6, solver="lbfgs", random_state=SEED, max_iter=10000)
        fold_calibrator.fit(inner_oof_score.reshape(-1, 1), inner_y)
        inner_probability = fold_calibrator.predict_proba(inner_oof_score.reshape(-1, 1))[:, 1]
        inner_fpr, inner_tpr, inner_thresholds = roc_curve(inner_y, inner_probability)
        inner_finite = np.isfinite(inner_thresholds)
        inner_best = np.flatnonzero(inner_finite)[np.argmax((inner_tpr - inner_fpr)[inner_finite])]
        fold_threshold = float(inner_thresholds[inner_best])

        outer_score = estimator.decision_function(matrix[outer_test])
        oof_score[outer_test] = outer_score
        oof_probability[outer_test] = fold_calibrator.predict_proba(outer_score.reshape(-1, 1))[:, 1]
        oof_threshold[outer_test] = fold_threshold
        oof_fold[outer_test] = fold_number
        outer_best_c.append(best_c)
        table.insert(0, "outer_fold", fold_number)
        table.insert(1, "selected", table["param_classifier__C"].astype(float).eq(best_c))
        outer_tables.append(table)
    if (
        not np.isfinite(oof_score).all()
        or not np.isfinite(oof_probability).all()
        or not np.isfinite(oof_threshold).all()
        or (oof_fold < 0).any()
    ):
        raise RuntimeError(f"incomplete nested OOF predictions for {name}")

    # Platt scaling sees only predictions made for validation days that were not
    # used to fit the corresponding base model.
    calibrator = LogisticRegression(C=1e6, solver="lbfgs", random_state=SEED, max_iter=10000)
    calibrator.fit(oof_score.reshape(-1, 1), y)
    final_calibration_probability = calibrator.predict_proba(oof_score.reshape(-1, 1))[:, 1]
    fpr, tpr, thresholds = roc_curve(y, final_calibration_probability)
    finite = np.isfinite(thresholds)
    best_index = np.flatnonzero(finite)[np.argmax((tpr - fpr)[finite])]
    threshold = float(thresholds[best_index])

    final_folds = fixed_group_folds(y, day_groups, n_splits=5, seed=SEED + 909)
    final_estimator, final_c, final_table = tune_pipeline(matrix, y, day_groups, final_folds, n_jobs=n_jobs)
    final_table.insert(0, "outer_fold", "final_all_TOM")
    final_table.insert(1, "selected", final_table["param_classifier__C"].astype(float).eq(final_c))
    return (
        FittedModel(
            name=name,
            feature_name=feature_name,
            estimator=final_estimator,
            calibrator=calibrator,
            threshold=threshold,
            source_prior=float(y.mean()),
            oof_score=oof_score,
            oof_probability=oof_probability,
            oof_threshold=oof_threshold,
            oof_fold=oof_fold,
            final_c=final_c,
            outer_best_c=outer_best_c,
        ),
        outer_tables,
        final_table,
    )


def fit_prior(y: np.ndarray, outer_folds) -> FittedModel:
    oof_probability = np.full(len(y), np.nan, dtype=float)
    oof_fold = np.full(len(y), -1, dtype=int)
    for fold_number, (train, test) in enumerate(outer_folds):
        oof_probability[test] = float(y[train].mean())
        oof_fold[test] = fold_number
    oof_score = np.log(np.clip(oof_probability, 1e-9, 1 - 1e-9) / np.clip(1 - oof_probability, 1e-9, 1))
    return FittedModel(
        name="prior_prevalence",
        feature_name=None,
        estimator=None,
        calibrator=None,
        threshold=0.5,
        source_prior=float(y.mean()),
        oof_score=oof_score,
        oof_probability=oof_probability,
        oof_threshold=np.full(len(y), 0.5, dtype=float),
        oof_fold=oof_fold,
        final_c=None,
        outer_best_c=[],
    )


def expected_calibration_error(y: np.ndarray, probability: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    memberships = np.minimum(np.digitize(probability, edges[1:-1], right=True), bins - 1)
    value = 0.0
    for bin_number in range(bins):
        mask = memberships == bin_number
        if mask.any():
            value += mask.mean() * abs(float(y[mask].mean()) - float(probability[mask].mean()))
    return float(value)


def binary_metrics(y: np.ndarray, probability: np.ndarray, threshold: float | np.ndarray) -> dict[str, float]:
    prediction = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if tp + fn else np.nan
    specificity = tn / (tn + fp) if tn + fp else np.nan
    clipped = np.clip(probability, 1e-7, 1 - 1e-7)
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "macro_f1": float(f1_score(y, prediction, average="macro", zero_division=0)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "auroc": float(roc_auc_score(y, probability)),
        "average_precision": float(average_precision_score(y, probability)),
        "brier": float(brier_score_loss(y, probability)),
        "nll": float(log_loss(y, clipped, labels=[0, 1])),
        "ece_10": expected_calibration_error(y, probability, bins=10),
    }


def group_bootstrap_indices(
    y: np.ndarray, groups: np.ndarray, replicates: int, rng: np.random.Generator
) -> Iterable[np.ndarray]:
    table = pd.DataFrame({"position": np.arange(len(y)), "y": y, "group": groups.astype(str)})
    unique_groups = table.group.drop_duplicates().tolist()
    group_to_positions = table.groupby("group")["position"].apply(lambda s: s.to_numpy()).to_dict()
    completed = 0
    attempts = 0
    while completed < replicates:
        attempts += 1
        if attempts > replicates * 20:
            raise RuntimeError("could not obtain binary-class cluster-bootstrap replicates")
        sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        indices = np.concatenate([group_to_positions[group] for group in sampled])
        if len(np.unique(y[indices])) != 2:
            continue
        completed += 1
        yield indices


def metric_rows(
    analysis_set: str,
    model_name: str,
    y: np.ndarray,
    probability: np.ndarray,
    threshold: float | np.ndarray,
    groups: np.ndarray,
    bootstrap_replicates: int,
    bootstrap_unit: str,
) -> list[dict]:
    threshold_array = np.asarray(threshold)
    if threshold_array.ndim and len(threshold_array) != len(y):
        raise RuntimeError("per-image threshold length mismatch")
    point = binary_metrics(y, probability, threshold)
    distributions = {name: [] for name in point}
    rng = np.random.default_rng(SEED + sum(ord(char) for char in analysis_set + model_name))
    for indices in group_bootstrap_indices(y, groups, bootstrap_replicates, rng):
        replicate_threshold = threshold_array[indices] if threshold_array.ndim else float(threshold_array)
        values = binary_metrics(y[indices], probability[indices], replicate_threshold)
        for metric, value in values.items():
            if np.isfinite(value):
                distributions[metric].append(value)
    rows = []
    for metric, estimate in point.items():
        distribution = np.asarray(distributions[metric], dtype=float)
        if metric == "ece_10":
            ci_low = np.nan
            ci_high = np.nan
            replicate_count = 0
        else:
            ci_low = float(np.quantile(distribution, 0.025))
            ci_high = float(np.quantile(distribution, 0.975))
            replicate_count = int(len(distribution))
        rows.append(
            {
                "analysis_set": analysis_set,
                "model": model_name,
                "metric": metric,
                "estimate": estimate,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "bootstrap_replicates": replicate_count,
                "bootstrap_unit": bootstrap_unit,
                "n_images": int(len(y)),
                "n_groups": int(pd.Series(groups).nunique()),
                "threshold_locked_from_TOM_nested_oof": float(threshold_array) if threshold_array.ndim == 0 else np.nan,
                "threshold_scheme": "single_final_TOM_OOF_threshold" if threshold_array.ndim == 0 else "outer_fold_specific_inner_OOF_threshold",
            }
        )
    return rows


def fpr_at_95_tpr(y_ood: np.ndarray, ood_score: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y_ood, ood_score)
    eligible = np.flatnonzero(tpr >= 0.95)
    return float(fpr[eligible[0]]) if len(eligible) else 1.0


def open_set_classification_rate(
    known_y: np.ndarray,
    known_prediction: np.ndarray,
    known_acceptance: np.ndarray,
    unknown_acceptance: np.ndarray,
) -> float:
    thresholds = np.unique(np.concatenate([known_acceptance, unknown_acceptance]))
    thresholds = np.concatenate([[np.inf], thresholds[::-1], [-np.inf]])
    correct = known_prediction == known_y
    points = []
    for threshold in thresholds:
        ccr = np.mean(correct & (known_acceptance >= threshold))
        unknown_fpr = np.mean(unknown_acceptance >= threshold)
        points.append((unknown_fpr, ccr))
    frame = pd.DataFrame(points, columns=["fpr", "ccr"]).groupby("fpr", as_index=False).ccr.max().sort_values("fpr")
    return float(np.trapezoid(frame.ccr.to_numpy(), frame.fpr.to_numpy()))


def selective_risk_rows(
    analysis_set: str,
    model_name: str,
    y: np.ndarray,
    prediction: np.ndarray,
    confidence: np.ndarray,
) -> list[dict]:
    order = np.argsort(-confidence, kind="stable")
    errors = (prediction[order] != y[order]).astype(float)
    cumulative_risk = np.cumsum(errors) / np.arange(1, len(errors) + 1)
    rows = []
    for coverage in (0.25, 0.50, 0.75, 1.00):
        count = max(1, int(math.ceil(coverage * len(y))))
        rows.append(
            {
                "analysis_set": analysis_set,
                "model": model_name,
                "coverage": coverage,
                "risk": float(cumulative_risk[count - 1]),
                "accepted_images": count,
            }
        )
    rows.append(
        {
            "analysis_set": analysis_set,
            "model": model_name,
            "coverage": "AURC",
            "risk": float(cumulative_risk.mean()),
            "accepted_images": len(y),
        }
    )
    return rows


def audit_naive_tom_splits(tom_all_manifest: Path, repetitions: int = 100) -> pd.DataFrame:
    frame = pd.read_csv(tom_all_manifest)
    required = {"source_group", "label"}
    if not required.issubset(frame.columns):
        return pd.DataFrame()
    rows = []
    for repetition in range(repetitions):
        rng = np.random.default_rng(SEED + repetition)
        test_positions = []
        for _, part in frame.groupby("label"):
            positions = part.index.to_numpy()
            count = max(1, int(round(0.2 * len(positions))))
            test_positions.extend(rng.choice(positions, size=count, replace=False).tolist())
        test_mask = frame.index.isin(test_positions)
        train_groups = set(frame.loc[~test_mask, "source_group"])
        leaked_test = frame.loc[test_mask, "source_group"].isin(train_groups)
        rows.append(
            {
                "repetition": repetition,
                "test_images": int(test_mask.sum()),
                "test_images_with_group_mate_in_train": int(leaked_test.sum()),
                "test_contamination_fraction": float(leaked_test.mean()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--n-jobs", type=int, default=max(1, min(4, (os.cpu_count() or 2) - 1)))
    parser.add_argument("--reuse-models", action="store_true", help="Reuse previously fitted joblib models after a reporting-only failure")
    args = parser.parse_args()
    if args.bootstrap < 100:
        raise ValueError("use at least 100 bootstrap replicates")

    for directory in (
        RESULTS,
        RESULTS / "models",
        RESULTS / "predictions",
        RESULTS / "split_manifests",
        RESULTS / "tables",
        RESULTS / "figures",
        RESULTS / "audit",
    ):
        directory.mkdir(parents=True, exist_ok=True)

    manifest_path = ROOT / "manifests" / "unified_manifest.csv"
    manifest = pd.read_csv(manifest_path)
    if manifest.sample_id.duplicated().any():
        raise RuntimeError("duplicate sample IDs")
    if manifest.source_group.isna().any():
        raise RuntimeError("missing source groups")
    missing_files = [path for path in manifest.relative_path if not (ROOT / path).is_file()]
    if missing_files:
        raise RuntimeError(f"missing image files: {missing_files[:3]}")

    ids = manifest.sample_id.astype(str).tolist()
    handcrafted = load_feature("handcrafted", ids)
    resnet18 = load_feature("resnet18", ids)
    efficientnet_b0 = load_feature("efficientnet_b0", ids)
    convnext_tiny = load_feature("convnext_tiny", ids)
    swin_t_features = load_feature("swin_t", ids)
    # The first 9 channel blocks each contain 16 histogram bins and 5 moments.
    color_shortcut = handcrafted[:, : 9 * (16 + 5)]
    feature_matrices = {
        "color_shortcut": color_shortcut,
        "handcrafted": handcrafted,
        "resnet18": resnet18,
        "efficientnet_b0": efficientnet_b0,
        "convnext_tiny": convnext_tiny,
        "swin_t": swin_t_features,
    }

    tom_mask = manifest.dataset.eq("TOM2024_A_BurkinaFaso") & manifest.foliar_binary_eligible.eq(1)
    cold_mask = manifest.dataset.eq("COLD_raw_India") & manifest.foliar_binary_eligible.eq(1)
    case_mask = manifest.dataset.eq("DigitalGreen_onion_operational") & manifest.foliar_binary_eligible.eq(1)
    ood_mask = manifest.dataset.eq("DigitalGreen_non_onion_OOD")
    tom_positions_all = np.flatnonzero(tom_mask.to_numpy())
    cold_positions = np.flatnonzero(cold_mask.to_numpy())
    case_positions = np.flatnonzero(case_mask.to_numpy())
    ood_positions = np.flatnonzero(ood_mask.to_numpy())
    tom_all = manifest.iloc[tom_positions_all].copy()
    cold = manifest.iloc[cold_positions].copy()
    tom_all["analysis_sha256"] = [sha256_file(ROOT / path) for path in tom_all.relative_path]
    cold["analysis_sha256"] = [sha256_file(ROOT / path) for path in cold.relative_path]
    removed_tom_exact = tom_all[tom_all.analysis_sha256.duplicated(keep="first")].copy()
    keep_tom = ~tom_all.analysis_sha256.duplicated(keep="first")
    tom_positions = tom_positions_all[keep_tom.to_numpy()]
    tom = tom_all.loc[keep_tom].reset_index(drop=True)
    cold = cold.reset_index(drop=True)
    if cold.analysis_sha256.duplicated().any():
        raise RuntimeError("COLD locked external set still contains exact duplicates")
    if set(tom.analysis_sha256).intersection(cold.analysis_sha256):
        raise RuntimeError("exact image overlap between TOM development and COLD external sets")
    sha_audit = pd.concat(
        [
            tom.assign(analysis_inclusion="development_included"),
            removed_tom_exact.assign(analysis_inclusion="removed_TOM_exact_duplicate"),
            cold.assign(analysis_inclusion="external_included"),
        ],
        ignore_index=True,
    )
    sha_audit[
        ["sample_id", "dataset", "relative_path", "source_group", "original_label", "analysis_sha256", "analysis_inclusion"]
    ].to_csv(RESULTS / "audit" / "primary_development_external_sha256.csv", index=False)
    case = manifest.iloc[case_positions].copy().reset_index(drop=True)
    ood = manifest.iloc[ood_positions].copy().reset_index(drop=True)
    tom["acquisition_day_utc"] = acquisition_day(tom.source_group)
    y_tom = tom.foliar_binary.astype(int).to_numpy()
    y_cold = cold.foliar_binary.astype(int).to_numpy()
    y_case = case.foliar_binary.astype(int).to_numpy()
    day_groups = tom.acquisition_day_utc.astype(str).to_numpy()

    # One row was retained per conflict-free timestamp/source group before this
    # script. Day-level grouping is coarser and therefore also protects the raw
    # source groups.
    if tom.source_group.duplicated().any():
        raise RuntimeError("TOM development manifest contains repeated raw-photo source groups")
    if cold.source_group.duplicated().any():
        raise RuntimeError("COLD external manifest contains repeated exact-SHA groups")
    if set(tom.source_group).intersection(cold.source_group):
        raise RuntimeError("development/external source-group overlap")
    outer_folds = fixed_group_folds(y_tom, day_groups, n_splits=5, seed=SEED)
    split_manifest = tom[
        [
            "sample_id",
            "relative_path",
            "source_group",
            "analysis_sha256",
            "acquisition_day_utc",
            "original_label",
            "foliar_binary",
        ]
    ].copy()
    split_manifest["outer_fold"] = -1
    for fold_number, (_, test) in enumerate(outer_folds):
        split_manifest.loc[test, "outer_fold"] = fold_number
    if (split_manifest.outer_fold < 0).any():
        raise RuntimeError("unassigned TOM rows")
    split_manifest.to_csv(RESULTS / "split_manifests" / "tom_acquisition_day_outer_folds.csv", index=False)

    fitted: dict[str, FittedModel] = {"prior_prevalence": fit_prior(y_tom, outer_folds)}
    save_model_artifact(fitted["prior_prevalence"], RESULTS / "models" / "prior_prevalence.joblib")
    cv_tables: list[pd.DataFrame] = []
    for model_name in (
        "color_shortcut_logit",
        "handcrafted_logit",
        "resnet18_logit",
        "efficientnet_b0_logit",
        "convnext_tiny_logit",
        "swin_t_logit",
    ):
        artifact_path = RESULTS / "models" / f"{model_name}.joblib"
        if args.reuse_models and artifact_path.is_file():
            fitted[model_name] = load_model_artifact(artifact_path)
            continue
        feature_name = MODEL_SPECS[model_name]["feature"]
        matrix = feature_matrices[feature_name][tom_positions]
        model, outer_tables, final_table = fit_nested_model(
            model_name, feature_name, matrix, y_tom, day_groups, outer_folds, n_jobs=args.n_jobs
        )
        fitted[model_name] = model
        for table in outer_tables + [final_table]:
            table.insert(0, "model", model_name)
            cv_tables.append(table)
        save_model_artifact(model, artifact_path)
    if cv_tables:
        pd.concat(cv_tables, ignore_index=True).to_csv(RESULTS / "tables" / "nested_cv_hyperparameters.csv", index=False)

    metric_table: list[dict] = []
    risk_table: list[dict] = []
    prediction_tables: dict[str, list[pd.DataFrame]] = {
        "tom_nested_oof": [],
        "cold_locked_external": [],
        "digigreen_onion_case_series": [],
    }
    cached_predictions: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    set_definitions = [
        ("tom_nested_oof", tom, tom_positions, y_tom, day_groups, "acquisition_day_utc"),
        ("cold_locked_external", cold, cold_positions, y_cold, cold.source_group.astype(str).to_numpy(), "raw_SHA_group"),
        (
            "digigreen_onion_case_series",
            case,
            case_positions,
            y_case,
            case.source_group.astype(str).to_numpy(),
            "farmer_image",
        ),
    ]
    for model_name, model in fitted.items():
        for set_name, frame, positions, y, groups, bootstrap_unit in set_definitions:
            if set_name == "tom_nested_oof":
                score = model.oof_score
                probability = model.oof_probability
                evaluation_threshold = model.oof_threshold
                prediction = (probability >= evaluation_threshold).astype(int)
                fold_values = model.oof_fold
                prediction_stream = "TOM_outer_crossfit_calibrator_and_threshold"
            else:
                feature_name = model.feature_name or "resnet18"
                score, probability, prediction = model.predict(feature_matrices[feature_name][positions])
                evaluation_threshold = model.threshold
                fold_values = np.full(len(frame), -1)
                prediction_stream = "final_TOM_all_nested_OOF_calibrator_and_threshold"
            cached_predictions[(model_name, set_name)] = (score, probability, prediction)
            output = frame.copy()
            output.insert(0, "model", model_name)
            output["uncalibrated_score"] = score
            output["calibrated_probability_disorder"] = probability
            output["locked_threshold"] = evaluation_threshold
            output["predicted_disorder"] = prediction
            output["tom_outer_fold"] = fold_values
            output["prediction_stream"] = prediction_stream
            prediction_tables[set_name].append(output)
            metric_table.extend(
                metric_rows(
                    set_name,
                    model_name,
                    y,
                    probability,
                    evaluation_threshold,
                    groups,
                    args.bootstrap,
                    bootstrap_unit,
                )
            )
            risk_table.extend(selective_risk_rows(set_name, model_name, y, prediction, np.maximum(probability, 1 - probability)))

    for set_name, tables in prediction_tables.items():
        pd.concat(tables, ignore_index=True).to_csv(RESULTS / "predictions" / f"{set_name}.csv", index=False)
    # Reload every portable artifact and prove its serialized OOF arrays match
    # the exact per-image TOM prediction stream written above.
    consistency_rows = []
    tom_prediction_table = pd.concat(prediction_tables["tom_nested_oof"], ignore_index=True)
    for model_name in MODEL_SPECS:
        artifact_path = RESULTS / "models" / f"{model_name}.joblib"
        restored = load_model_artifact(artifact_path)
        part = tom_prediction_table[tom_prediction_table.model.eq(model_name)]
        if part.sample_id.astype(str).tolist() != tom.sample_id.astype(str).tolist():
            raise RuntimeError(f"serialized consistency sample order mismatch for {model_name}")
        checks = {
            "oof_score_max_abs_error": float(np.max(np.abs(restored.oof_score - part.uncalibrated_score.to_numpy()))),
            "oof_probability_max_abs_error": float(
                np.max(np.abs(restored.oof_probability - part.calibrated_probability_disorder.to_numpy()))
            ),
            "oof_threshold_max_abs_error": float(np.max(np.abs(restored.oof_threshold - part.locked_threshold.to_numpy()))),
            "oof_fold_exact_match": bool(np.array_equal(restored.oof_fold, part.tom_outer_fold.to_numpy())),
        }
        if max(checks[key] for key in checks if key.endswith("max_abs_error")) > 1e-12 or not checks["oof_fold_exact_match"]:
            raise RuntimeError(f"serialized OOF mismatch for {model_name}: {checks}")
        consistency_rows.append(
            {
                "model": model_name,
                "artifact_sha256": sha256_file(artifact_path),
                **checks,
                "status": "PASS",
            }
        )
    pd.DataFrame(consistency_rows).to_csv(RESULTS / "audit" / "serialized_model_prediction_consistency.csv", index=False)
    metrics = pd.DataFrame(metric_table)
    metrics.to_csv(RESULTS / "tables" / "binary_metrics_group_bootstrap.csv", index=False)
    pd.DataFrame(risk_table).to_csv(RESULTS / "tables" / "risk_coverage.csv", index=False)

    # Per-diagnosis case-series counts remain descriptive because n=24 and the
    # operational labels include mixed biotic/abiotic findings.
    case_prediction = pd.concat(prediction_tables["digigreen_onion_case_series"], ignore_index=True)
    case_prediction["correct_binary_label"] = case_prediction.predicted_disorder.eq(case_prediction.foliar_binary)
    case_summary = (
        case_prediction.groupby(["model", "original_label"], as_index=False)
        .agg(n=("sample_id", "size"), correct=("correct_binary_label", "sum"), mean_probability=("calibrated_probability_disorder", "mean"))
    )
    case_summary["accuracy_descriptive"] = case_summary.correct / case_summary.n
    case_summary.to_csv(RESULTS / "tables" / "digigreen_case_series_by_annotation.csv", index=False)

    # OOD/open-set stress test: Digital Green non-onion crops are unknowns.
    ood_rows: list[dict] = []
    ood_score_rows: list[pd.DataFrame] = []
    for model_name, model in fitted.items():
        feature_name = model.feature_name or "resnet18"
        source_matrix = feature_matrices[feature_name][tom_positions]
        unknown_matrix = feature_matrices[feature_name][ood_positions]
        unknown_score, unknown_probability, unknown_prediction = model.predict(unknown_matrix)
        unknown_msp = np.maximum(unknown_probability, 1 - unknown_probability)
        for known_set, known_frame, known_positions, known_y in (
            ("cold_locked_external", cold, cold_positions, y_cold),
            ("digigreen_onion_case_series", case, case_positions, y_case),
        ):
            known_score, known_probability, known_prediction = cached_predictions[(model_name, known_set)]
            known_msp = np.maximum(known_probability, 1 - known_probability)
            detector_scores: dict[str, tuple[np.ndarray, np.ndarray]] = {"maximum_softmax_probability": (known_msp, unknown_msp)}
            if model.estimator is not None:
                scaler = model.estimator.named_steps["scale"]
                source_scaled = scaler.transform(source_matrix)
                known_scaled = scaler.transform(feature_matrices[feature_name][known_positions])
                unknown_scaled = scaler.transform(unknown_matrix)
                neighbors = NearestNeighbors(n_neighbors=5, metric="cosine", n_jobs=args.n_jobs).fit(source_scaled)
                known_distance = neighbors.kneighbors(known_scaled, return_distance=True)[0].mean(axis=1)
                unknown_distance = neighbors.kneighbors(unknown_scaled, return_distance=True)[0].mean(axis=1)
                detector_scores["source_5nn_cosine"] = (-known_distance, -unknown_distance)
            for detector, (known_acceptance, unknown_acceptance) in detector_scores.items():
                y_ood = np.concatenate([np.zeros(len(known_acceptance), dtype=int), np.ones(len(unknown_acceptance), dtype=int)])
                anomaly = -np.concatenate([known_acceptance, unknown_acceptance])
                ood_rows.append(
                    {
                        "known_set": known_set,
                        "unknown_set": "digigreen_non_onion_OOD",
                        "model": model_name,
                        "detector": detector,
                        "known_images": len(known_acceptance),
                        "unknown_images": len(unknown_acceptance),
                        "ood_auroc": float(roc_auc_score(y_ood, anomaly)),
                        "ood_average_precision": float(average_precision_score(y_ood, anomaly)),
                        "fpr_at_95pct_ood_tpr": fpr_at_95_tpr(y_ood, anomaly),
                        "oscr": open_set_classification_rate(
                            known_y, known_prediction, known_acceptance, unknown_acceptance
                        ),
                    }
                )
                known_output = pd.DataFrame(
                    {
                        "known_set": known_set,
                        "sample_id": known_frame.sample_id,
                        "model": model_name,
                        "detector": detector,
                        "is_ood": 0,
                        "id_acceptance_score": known_acceptance,
                        "ood_anomaly_score": -known_acceptance,
                        "binary_probability": known_probability,
                        "binary_prediction": known_prediction,
                        "binary_truth": known_y,
                    }
                )
                unknown_output = pd.DataFrame(
                    {
                        "known_set": known_set,
                        "sample_id": ood.sample_id,
                        "model": model_name,
                        "detector": detector,
                        "is_ood": 1,
                        "id_acceptance_score": unknown_acceptance,
                        "ood_anomaly_score": -unknown_acceptance,
                        "binary_probability": unknown_probability,
                        "binary_prediction": unknown_prediction,
                        "binary_truth": np.nan,
                    }
                )
                ood_score_rows.extend([known_output, unknown_output])
    pd.DataFrame(ood_rows).to_csv(RESULTS / "tables" / "open_set_ood_metrics.csv", index=False)
    pd.concat(ood_score_rows, ignore_index=True).to_csv(RESULTS / "predictions" / "open_set_ood_scores.csv", index=False)

    # Structural leakage audit for the unreduced TOM file inventory. This is a
    # contamination-count audit only; it does not estimate performance inflation.
    source_inventory = ROOT / "audit" / "tom2024_category_a_onion" / "source_groups" / "source_group_members.csv"
    naive_audit = audit_naive_tom_splits(source_inventory)
    if not naive_audit.empty:
        naive_audit.to_csv(RESULTS / "audit" / "tom_naive_file_split_contamination.csv", index=False)

    # Figure 1: locked external performance with group-bootstrap intervals.
    plot_metrics = metrics[
        metrics.analysis_set.eq("cold_locked_external")
        & metrics.metric.isin(["balanced_accuracy", "auroc", "brier"])
    ].copy()
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    for axis, metric in zip(axes, ["balanced_accuracy", "auroc", "brier"]):
        part = plot_metrics[plot_metrics.metric.eq(metric)].copy()
        part["model"] = pd.Categorical(part.model, categories=list(MODEL_SPECS), ordered=True)
        part = part.sort_values("model")
        x = np.arange(len(part))
        axis.errorbar(
            x,
            part.estimate,
            yerr=np.vstack([part.estimate - part.ci_low, part.ci_high - part.estimate]),
            fmt="o",
            capsize=3,
            color="#1f4e79",
        )
        axis.set_xticks(x, [name.replace("_", "\n") for name in part.model.astype(str)], fontsize=7)
        axis.set_title(metric.replace("_", " ").title())
        axis.grid(axis="y", alpha=0.25)
        if metric != "brier":
            axis.set_ylim(0, 1)
            axis.axhline(0.5, color="0.6", linestyle="--", linewidth=0.8)
        else:
            axis.set_ylim(bottom=0)
    fig.suptitle("Locked COLD external evaluation (India); models developed only on TOM2024")
    fig.savefig(RESULTS / "figures" / "cold_external_metrics.png", dpi=400)
    fig.savefig(RESULTS / "figures" / "cold_external_metrics.pdf")
    plt.close(fig)

    # Figure 2: reliability curves on the untouched external set.
    fig, axis = plt.subplots(figsize=(5.2, 4.6), constrained_layout=True)
    for model_name in MODEL_SPECS:
        probability = cached_predictions[(model_name, "cold_locked_external")][1]
        observed, predicted = calibration_curve(y_cold, probability, n_bins=10, strategy="quantile")
        axis.plot(predicted, observed, marker="o", linewidth=1.2, label=model_name)
    axis.plot([0, 1], [0, 1], "--", color="0.5", linewidth=1)
    axis.set(xlabel="Mean predicted probability", ylabel="Observed disorder fraction", xlim=(0, 1), ylim=(0, 1))
    axis.legend(fontsize=7)
    axis.grid(alpha=0.2)
    fig.savefig(RESULTS / "figures" / "cold_external_reliability.png", dpi=400)
    fig.savefig(RESULTS / "figures" / "cold_external_reliability.pdf")
    plt.close(fig)

    model_summary = []
    for name, model in fitted.items():
        model_summary.append(
            {
                "model": name,
                "feature": model.feature_name,
                "role": MODEL_SPECS[name]["role"],
                "final_C_selected_with_TOM_grouped_CV": model.final_c,
                "outer_fold_best_C": model.outer_best_c,
                "calibration": "Platt scaling on acquisition-day nested OOF scores" if model.calibrator else "source prevalence",
                "threshold": model.threshold,
                "threshold_source": "TOM acquisition-day nested OOF" if model.calibrator else "fixed 0.5",
            }
        )
    json_dump(RESULTS / "models" / "model_summary.json", model_summary)
    compute_rows = [
        {
            "model": "resnet18_logit",
            "role": "a_priori_primary",
            "backbone": "ResNet18",
            "weights": "ImageNet1K_V1",
            "parameters": 11689512,
            "giga_operations_per_image": 1.814,
            "feature_dimension": 512,
            "extraction_elapsed_seconds": np.nan,
            "images_per_second": np.nan,
        }
    ]
    for feature_name, model_name in (
        ("efficientnet_b0", "efficientnet_b0_logit"),
        ("convnext_tiny", "convnext_tiny_logit"),
        ("swin_t", "swin_t_logit"),
    ):
        metadata = json.loads((ROOT / "features" / f"{feature_name}.json").read_text(encoding="utf-8"))
        compute_rows.append(
            {
                "model": model_name,
                "role": MODEL_SPECS[model_name]["role"],
                "backbone": feature_name,
                "weights": metadata["weights"],
                "parameters": metadata["official_weight_metadata"]["parameter_count"],
                "giga_operations_per_image": metadata["official_weight_metadata"]["giga_operations_per_image"],
                "feature_dimension": metadata["feature_dimension"],
                "extraction_elapsed_seconds": metadata["extraction"]["elapsed_seconds"],
                "images_per_second": metadata["extraction"]["images_per_second"],
            }
        )
    pd.DataFrame(compute_rows).to_csv(RESULTS / "tables" / "backbone_compute_metadata.csv", index=False)

    integrity = {
        "manifest_sha256": sha256_file(manifest_path),
        "handcrafted_features_sha256": sha256_file(ROOT / "features" / "handcrafted.npz"),
        "resnet18_features_sha256": sha256_file(ROOT / "features" / "resnet18.npz"),
        "efficientnet_b0_features_sha256": sha256_file(ROOT / "features" / "efficientnet_b0.npz"),
        "convnext_tiny_features_sha256": sha256_file(ROOT / "features" / "convnext_tiny.npz"),
        "swin_t_features_sha256": sha256_file(ROOT / "features" / "swin_t.npz"),
        "script_sha256": sha256_file(Path(__file__)),
        "samples": {
            "TOM2024_development_primary": len(tom),
            "TOM2024_primary_exact_duplicates_removed": len(removed_tom_exact),
            "TOM2024_acquisition_days": int(tom.acquisition_day_utc.nunique()),
            "COLD_locked_external": len(cold),
            "DigitalGreen_onion_case_series": len(case),
            "DigitalGreen_non_onion_OOD": len(ood),
        },
        "assertions": {
            "TOM_one_representative_per_raw_source_group": True,
            "TOM_primary_exact_hashes_unique_after_filter": True,
            "outer_fold_acquisition_day_overlap": 0,
            "TOM_COLD_source_group_overlap": 0,
            "TOM_COLD_exact_hash_overlap": 0,
            "COLD_used_for_model_selection": False,
            "COLD_used_for_calibration": False,
            "COLD_used_for_threshold_selection": False,
            "augmented_collections_used": False,
            "challenger_models_used_COLD_for_selection_or_tuning": False,
        },
        "known_limitations": [
            "COLD filenames do not expose acquisition day or biological specimen identity; external uncertainty is bootstrapped by deduplicated raw SHA group.",
            "TOM timestamps are acquisition events, not verified biological plant identities; day grouping is the coarser leakage barrier.",
            "The healthy-versus-dataset-labelled-foliar-disorder endpoint harmonizes heterogeneous diagnoses and does not establish pathogen-specific transfer.",
            "Digital Green onion n=24 is a descriptive operational case series and is not a confirmatory external test.",
            "Digital Green non-onion images test crop-level open-set rejection, not unknown onion-disease rejection.",
        ],
    }
    json_dump(RESULTS / "audit" / "data_and_leakage_integrity.json", integrity)
    run_metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "bootstrap_replicates": args.bootstrap,
        "n_jobs": args.n_jobs,
        "C_grid": C_GRID,
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "command": f"python {Path(__file__).as_posix()} --bootstrap {args.bootstrap} --n-jobs {args.n_jobs}",
    }
    json_dump(RESULTS / "run_metadata.json", run_metadata)

    # Compact factual report, deliberately separating the release decision from
    # numerical performance.
    ext = metrics[metrics.analysis_set.eq("cold_locked_external")]
    key = ext[ext.metric.isin(["balanced_accuracy", "auroc", "brier", "ece_10"])].pivot(
        index="model", columns="metric", values="estimate"
    )
    primary_model = "resnet18_logit"  # fixed a priori; never selected by COLD performance
    report = [
        "# Leakage-controlled onion reliability benchmark: execution report",
        "",
        "## Locked design",
        "",
        f"TOM2024 contributed {len(tom):,} conflict-free, exact-deduplicated source representatives from {tom.acquisition_day_utc.nunique()} acquisition days ({len(removed_tom_exact)} redundant exact files were removed at analysis time). All model and hyperparameter choices, Platt calibration, and decision thresholds were derived within TOM2024 using acquisition-day-grouped nested validation. COLD contributed {len(cold):,} exact-deduplicated raw images and was not used for tuning, calibration, threshold selection, or feature learning.",
        "",
        "The primary endpoint is binary healthy versus dataset-labelled foliar disorder. It is deliberately broader than pathogen diagnosis because COLD and TOM2024 do not provide a validated one-to-one disease ontology. The analysis does not claim pathogen-specific cross-country transfer.",
        "",
        "TOM internal metrics use an outer-fold prediction stream in which each fold has a calibrator and threshold derived only from inner out-of-fold scores in the remaining acquisition days. The separate final calibrator and threshold use all TOM nested out-of-fold scores and are applied only to COLD, Digital Green, and OOD images. This prevents calibration or threshold reuse from making the TOM internal estimate optimistic.",
        "",
        "ResNet18 was fixed as the sole a-priori primary model. EfficientNet-B0, ConvNeXt-Tiny, and Swin-T were prelisted before their features or scores were computed and are retained as secondary architecture sensitivities regardless of their COLD results. Because COLD had already been opened for the primary ResNet analysis before this expansion, the challenger comparison is post-primary sensitivity evidence rather than a new pristine confirmation; no challenger uses COLD for fitting, calibration, threshold selection, retention, or claims.",
        "",
        "## External result",
        "",
        f"The a-priori primary frozen-feature model, **{primary_model}**, obtained locked COLD balanced accuracy {key.loc[primary_model, 'balanced_accuracy']:.3f}, AUROC {key.loc[primary_model, 'auroc']:.3f}, Brier {key.loc[primary_model, 'brier']:.3f}, and ECE {key.loc[primary_model, 'ece_10']:.3f}. The color shortcut, full handcrafted, and prevalence-prior comparators are reported concurrently; no model is selected using COLD. Confidence intervals and every baseline are in `tables/binary_metrics_group_bootstrap.csv`. This is a reliability result, not evidence that a new architecture has been invented.",
        "",
        "## Leakage controls",
        "",
        "- Conflicting TOM timestamp groups were removed upstream; one image per retained timestamp group entered the candidate pool, followed by an exact-hash deduplication at analysis time.",
        "- Outer and inner validation folds are disjoint by UTC acquisition day, a coarser unit than the timestamp/raw-photo group.",
        "- The 4,502-image COLD augmented collection was excluded from primary external validation because it contains derived, non-independent siblings; filename families are auditable, but the raw-parent crosswalk is incomplete for one IYSV family. It is evaluated only in a separate family-safe leakage ablation.",
        "- COLD remained locked until the final TOM-only models, calibrators, and thresholds were fixed.",
        "",
        "## Interpretation limits",
        "",
        "Digital Green onion images (n=24) are reported as a case series only. Their annotations include healthy labels with textual stress observations and mixed pest/disease labels. The non-onion Digital Green subset probes rejection of other crops; it does not validate rejection of novel onion diseases. COLD lacks acquisition-day and plant-identity metadata, so its interval uses the exact-deduplicated SHA group as the resampling unit.",
        "",
        "## Machine-readable outputs",
        "",
        "The split manifest, per-image probabilities, grouped-bootstrap intervals, OOD scores, selected hyperparameters, fitted models, hashes, and environment versions are stored next to this report. The report should not be converted into a submission claim until the external result and its uncertainty are assessed against the journal release gate.",
        "",
    ]
    (RESULTS / "EXPERIMENT_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({"results": str(RESULTS), "a_priori_primary_model": primary_model, "key_metrics": key.to_dict(orient="index")}, indent=2))


if __name__ == "__main__":
    main()
