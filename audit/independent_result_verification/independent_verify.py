"""Independent verification of the onion reliability benchmark.

This script reads, but never writes to, ``results/reliability_benchmark_v1``.
All derived audit artifacts are written beside this file.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import __main__
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold


AUDIT_DIR = Path(__file__).resolve().parent
ROOT = AUDIT_DIR.parents[1]
RESULTS = ROOT / "results" / "reliability_benchmark_v1"
SEED = 20260930
PREDICTION_FILES = {
    "tom_nested_oof": RESULTS / "predictions" / "tom_nested_oof.csv",
    "cold_locked_external": RESULTS / "predictions" / "cold_locked_external.csv",
    "digigreen_onion_case_series": RESULTS / "predictions" / "digigreen_onion_case_series.csv",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ece(y: np.ndarray, probability: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    membership = np.minimum(np.digitize(probability, edges[1:-1], right=True), bins - 1)
    value = 0.0
    for number in range(bins):
        mask = membership == number
        if mask.any():
            value += mask.mean() * abs(float(y[mask].mean()) - float(probability[mask].mean()))
    return float(value)


def metrics(y: np.ndarray, probability: np.ndarray, threshold: np.ndarray) -> dict[str, float]:
    prediction = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    clipped = np.clip(probability, 1e-7, 1 - 1e-7)
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "macro_f1": float(f1_score(y, prediction, average="macro", zero_division=0)),
        "sensitivity": float(tp / (tp + fn)),
        "specificity": float(tn / (tn + fp)),
        "auroc": float(roc_auc_score(y, probability)),
        "average_precision": float(average_precision_score(y, probability)),
        "brier": float(brier_score_loss(y, probability)),
        "nll": float(log_loss(y, clipped, labels=[0, 1])),
        "ece_10": ece(y, probability),
    }


def group_bootstrap_indices(y: np.ndarray, groups: np.ndarray, replicates: int, rng: np.random.Generator):
    frame = pd.DataFrame({"position": np.arange(len(y)), "y": y, "group": groups.astype(str)})
    names = frame.group.drop_duplicates().tolist()
    positions = frame.groupby("group").position.apply(lambda values: values.to_numpy()).to_dict()
    completed = 0
    while completed < replicates:
        sampled = rng.choice(names, size=len(names), replace=True)
        indices = np.concatenate([positions[name] for name in sampled])
        if len(np.unique(y[indices])) != 2:
            continue
        completed += 1
        yield indices


def input_snapshot() -> pd.DataFrame:
    paths = [
        ROOT / "code" / "run_reliability_benchmark.py",
        ROOT / "code" / "analyze_reliability_results.py",
        ROOT / "code" / "extract_features.py",
        ROOT / "manifests" / "unified_manifest.csv",
    ]
    paths.extend(sorted(path for path in (ROOT / "features").glob("*.npz") if path.is_file()))
    paths.extend(sorted(path for path in (ROOT / "features").glob("*.json") if path.is_file()))
    paths.extend(sorted(path for path in RESULTS.rglob("*") if path.is_file()))
    rows = []
    for path in paths:
        stat = path.stat()
        rows.append(
            {
                "path": path.relative_to(ROOT).as_posix(),
                "bytes": stat.st_size,
                "mtime_utc": pd.Timestamp(stat.st_mtime, unit="s", tz="UTC").isoformat(),
                "sha256": sha256_file(path),
            }
        )
    return pd.DataFrame(rows).drop_duplicates("path").sort_values("path")


def verify_predictions(manifest: pd.DataFrame) -> tuple[pd.DataFrame, list[dict], dict[str, pd.DataFrame]]:
    reported = pd.read_csv(RESULTS / "tables" / "binary_metrics_group_bootstrap.csv")
    recomputed_rows: list[dict] = []
    checks: list[dict] = []
    prediction_frames: dict[str, pd.DataFrame] = {}
    manifest_by_id = manifest.set_index("sample_id")

    for set_name, path in PREDICTION_FILES.items():
        long = pd.read_csv(path)
        prediction_frames[set_name] = long
        models = sorted(long.model.unique())
        id_sets = {model: set(long.loc[long.model.eq(model), "sample_id"]) for model in models}
        checks.append(
            {
                "check": f"{set_name}: model sample-ID sets identical",
                "passed": len({frozenset(value) for value in id_sets.values()}) == 1,
                "detail": f"models={models}; n={len(next(iter(id_sets.values())))}",
            }
        )
        for model in models:
            part = long[long.model.eq(model)].copy()
            duplicate_count = int(part.sample_id.duplicated().sum())
            missing_from_manifest = sorted(set(part.sample_id) - set(manifest.sample_id))
            aligned = manifest_by_id.loc[part.sample_id]
            truth_match = np.array_equal(part.foliar_binary.astype(int).to_numpy(), aligned.foliar_binary.astype(int).to_numpy())
            source_match = np.array_equal(part.source_group.astype(str).to_numpy(), aligned.source_group.astype(str).to_numpy())
            path_match = np.array_equal(part.relative_path.astype(str).to_numpy(), aligned.relative_path.astype(str).to_numpy())
            probability = part.calibrated_probability_disorder.to_numpy(float)
            threshold = part.locked_threshold.to_numpy(float)
            predicted = (probability >= threshold).astype(int)
            y = part.foliar_binary.astype(int).to_numpy()
            checks.extend(
                [
                    {"check": f"{set_name}/{model}: one row per sample", "passed": duplicate_count == 0, "detail": f"duplicates={duplicate_count}"},
                    {"check": f"{set_name}/{model}: all IDs in manifest", "passed": not missing_from_manifest, "detail": f"missing={len(missing_from_manifest)}"},
                    {"check": f"{set_name}/{model}: truth matches manifest", "passed": truth_match, "detail": ""},
                    {"check": f"{set_name}/{model}: source groups match manifest", "passed": source_match, "detail": ""},
                    {"check": f"{set_name}/{model}: paths match manifest", "passed": path_match, "detail": ""},
                    {"check": f"{set_name}/{model}: finite probabilities in [0,1]", "passed": bool(np.isfinite(probability).all() and ((probability >= 0) & (probability <= 1)).all()), "detail": ""},
                    {"check": f"{set_name}/{model}: predictions equal probability>=threshold", "passed": np.array_equal(predicted, part.predicted_disorder.astype(int).to_numpy()), "detail": ""},
                ]
            )
            for metric, value in metrics(y, probability, threshold).items():
                recomputed_rows.append({"analysis_set": set_name, "model": model, "metric": metric, "recomputed_estimate": value})

            groups = (
                pd.to_datetime(part.source_group.str.extract(r"(1\d{12})$", expand=False).astype("int64"), unit="ms", utc=True)
                .dt.strftime("%Y-%m-%d")
                .to_numpy()
                if set_name == "tom_nested_oof"
                else part.source_group.astype(str).to_numpy()
            )
            rng = np.random.default_rng(SEED + sum(ord(char) for char in set_name + model))
            distributions = {name: [] for name in metrics(y, probability, threshold)}
            for indices in group_bootstrap_indices(y, groups, 1000, rng):
                values = metrics(y[indices], probability[indices], threshold[indices])
                for metric, value in values.items():
                    distributions[metric].append(value)
            for row in recomputed_rows[-9:]:
                values = np.asarray(distributions[row["metric"]], dtype=float)
                if row["metric"] == "ece_10":
                    row.update(recomputed_ci_low=np.nan, recomputed_ci_high=np.nan)
                else:
                    row.update(
                        recomputed_ci_low=float(np.quantile(values, 0.025)),
                        recomputed_ci_high=float(np.quantile(values, 0.975)),
                    )

    recomputed = pd.DataFrame(recomputed_rows)
    comparison = reported.merge(recomputed, on=["analysis_set", "model", "metric"], how="outer", validate="one_to_one")
    comparison["estimate_abs_diff"] = (comparison.estimate - comparison.recomputed_estimate).abs()
    comparison["ci_low_abs_diff"] = (comparison.ci_low - comparison.recomputed_ci_low).abs()
    comparison["ci_high_abs_diff"] = (comparison.ci_high - comparison.recomputed_ci_high).abs()
    comparison["estimate_match_1e-12"] = comparison.estimate_abs_diff <= 1e-12
    comparison["ci_match_1e-12"] = (
        comparison.metric.eq("ece_10")
        | ((comparison.ci_low_abs_diff <= 1e-12) & (comparison.ci_high_abs_diff <= 1e-12))
    )
    checks.append(
        {
            "check": "all published point estimates reproduce at 1e-12",
            "passed": bool(comparison["estimate_match_1e-12"].all()),
            "detail": f"max_abs_diff={comparison.estimate_abs_diff.max():.3g}",
        }
    )
    checks.append(
        {
            "check": "all published bootstrap limits reproduce at 1e-12",
            "passed": bool(comparison["ci_match_1e-12"].all()),
            "detail": f"max_low={comparison.ci_low_abs_diff.max():.3g}; max_high={comparison.ci_high_abs_diff.max():.3g}",
        }
    )
    return comparison, checks, prediction_frames


def verify_splits_and_hashes(manifest: pd.DataFrame, predictions: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, list[dict], dict]:
    checks: list[dict] = []
    split = pd.read_csv(RESULTS / "split_manifests" / "tom_acquisition_day_outer_folds.csv")
    day_fold_counts = split.groupby("acquisition_day_utc").outer_fold.nunique()
    source_fold_counts = split.groupby("source_group").outer_fold.nunique()
    hash_fold_counts = split.groupby("analysis_sha256").outer_fold.nunique()
    checks.extend(
        [
            {"check": "TOM acquisition day occurs in one outer fold", "passed": bool(day_fold_counts.max() == 1), "detail": f"days={len(day_fold_counts)}"},
            {"check": "TOM source group occurs in one outer fold", "passed": bool(source_fold_counts.max() == 1), "detail": f"groups={len(source_fold_counts)}"},
            {"check": "TOM exact hash occurs in one outer fold", "passed": bool(hash_fold_counts.max() == 1), "detail": f"hashes={len(hash_fold_counts)}"},
            {"check": "TOM split manifest has unique sample IDs", "passed": not split.sample_id.duplicated().any(), "detail": f"n={len(split)}"},
            {"check": "TOM split manifest has both classes in every fold", "passed": bool(split.groupby("outer_fold").foliar_binary.nunique().eq(2).all()), "detail": str(split.groupby("outer_fold").size().to_dict())},
        ]
    )
    # Recreate the exact outer partition from the declared seed.
    y = split.foliar_binary.astype(int).to_numpy()
    groups = split.acquisition_day_utc.astype(str).to_numpy()
    recreated = np.full(len(split), -1, dtype=int)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    for fold, (_, test) in enumerate(splitter.split(np.zeros(len(y)), y, groups)):
        recreated[test] = fold
    checks.append(
        {"check": "stored TOM outer folds recreate from seed", "passed": np.array_equal(recreated, split.outer_fold.to_numpy()), "detail": "StratifiedGroupKFold(seed=20260930)"}
    )

    primary = pd.read_csv(RESULTS / "audit" / "primary_development_external_sha256.csv")
    included = primary[primary.analysis_inclusion.isin(["development_included", "external_included"])].copy()
    hash_rows = []
    for row in included.itertuples(index=False):
        actual = sha256_file(ROOT / row.relative_path)
        hash_rows.append(
            {
                "sample_id": row.sample_id,
                "dataset": row.dataset,
                "relative_path": row.relative_path,
                "reported_sha256": row.analysis_sha256,
                "recomputed_sha256": actual,
                "match": actual == row.analysis_sha256,
            }
        )
    hash_frame = pd.DataFrame(hash_rows)
    tom_hash = set(hash_frame.loc[hash_frame.dataset.eq("TOM2024_A_BurkinaFaso"), "recomputed_sha256"])
    cold_hash = set(hash_frame.loc[hash_frame.dataset.eq("COLD_raw_India"), "recomputed_sha256"])
    checks.extend(
        [
            {"check": "all primary image SHA-256 values independently reproduce", "passed": bool(hash_frame.match.all()), "detail": f"n={len(hash_frame)}"},
            {"check": "no exact image hash overlap TOM vs COLD", "passed": not bool(tom_hash & cold_hash), "detail": f"overlap={len(tom_hash & cold_hash)}"},
            {"check": "TOM included hashes unique", "passed": len(tom_hash) == int((hash_frame.dataset == "TOM2024_A_BurkinaFaso").sum()), "detail": f"unique={len(tom_hash)}"},
            {"check": "COLD included hashes unique", "passed": len(cold_hash) == int((hash_frame.dataset == "COLD_raw_India").sum()), "detail": f"unique={len(cold_hash)}"},
        ]
    )
    removed = primary[primary.analysis_inclusion.eq("removed_TOM_exact_duplicate")]
    duplicate_label_conflicts = 0
    for digest, part in primary[primary.dataset.eq("TOM2024_A_BurkinaFaso")].groupby("analysis_sha256"):
        if len(part) > 1 and part.original_label.nunique() > 1:
            duplicate_label_conflicts += 1
    checks.append(
        {"check": "removed TOM exact duplicates do not cross labels", "passed": duplicate_label_conflicts == 0, "detail": f"removed={len(removed)}; conflicting_hash_groups={duplicate_label_conflicts}"}
    )
    summary = {
        "tom_included": int((hash_frame.dataset == "TOM2024_A_BurkinaFaso").sum()),
        "cold_included": int((hash_frame.dataset == "COLD_raw_India").sum()),
        "tom_days": int(split.acquisition_day_utc.nunique()),
        "tom_removed_exact_duplicates": int(len(removed)),
        "tom_cold_exact_hash_overlap": int(len(tom_hash & cold_hash)),
        "cold_unique_source_groups": int(predictions["cold_locked_external"].drop_duplicates("sample_id").source_group.nunique()),
    }
    return hash_frame, checks, summary


def load_saved_models() -> tuple[dict[str, object], object]:
    source = ROOT / "code" / "run_reliability_benchmark.py"
    spec = importlib.util.spec_from_file_location("independent_loaded_benchmark", source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    __main__.FittedModel = module.FittedModel
    models = {
        name: module.load_model_artifact(RESULTS / "models" / f"{name}.joblib")
        for name in module.MODEL_SPECS
    }
    return models, module


def verify_saved_models(manifest: pd.DataFrame, predictions: dict[str, pd.DataFrame]) -> list[dict]:
    checks: list[dict] = []
    models, module = load_saved_models()
    ids = manifest.sample_id.astype(str).tolist()
    handcrafted_archive = np.load(ROOT / "features" / "handcrafted.npz", allow_pickle=False)
    resnet_archive = np.load(ROOT / "features" / "resnet18.npz", allow_pickle=False)
    checks.append({"check": "handcrafted feature IDs equal manifest order", "passed": handcrafted_archive["sample_ids"].astype(str).tolist() == ids, "detail": f"n={len(ids)}"})
    checks.append({"check": "ResNet feature IDs equal manifest order", "passed": resnet_archive["sample_ids"].astype(str).tolist() == ids, "detail": f"n={len(ids)}"})
    handcrafted = handcrafted_archive["features"].astype(float)
    feature = {
        "prior_prevalence": resnet_archive["features"].astype(float),
        "color_shortcut_logit": handcrafted[:, : 9 * (16 + 5)],
        "handcrafted_logit": handcrafted,
        "resnet18_logit": resnet_archive["features"].astype(float),
    }
    # Challenger embeddings, when present, use the feature stem declared by
    # the frozen benchmark module. The COLD predictions remain read-only here.
    for name, specification in module.MODEL_SPECS.items():
        if name in feature:
            continue
        feature_name = specification["feature"]
        archive = np.load(ROOT / "features" / f"{feature_name}.npz", allow_pickle=False)
        checks.append(
            {
                "check": f"{feature_name} feature IDs equal manifest order",
                "passed": archive["sample_ids"].astype(str).tolist() == ids,
                "detail": f"n={len(ids)}",
            }
        )
        feature[name] = archive["features"].astype(float)
    cold_positions = np.flatnonzero((manifest.dataset.eq("COLD_raw_India") & manifest.foliar_binary_eligible.eq(1)).to_numpy())
    for name, model in models.items():
        tom_part = predictions["tom_nested_oof"][predictions["tom_nested_oof"].model.eq(name)]
        cold_part = predictions["cold_locked_external"][predictions["cold_locked_external"].model.eq(name)]
        oof_equal = (
            # CSV round-tripping changes one large-magnitude score by 1.42e-14.
            np.allclose(model.oof_score, tom_part.uncalibrated_score.to_numpy(), rtol=0, atol=5e-14)
            and np.allclose(model.oof_probability, tom_part.calibrated_probability_disorder.to_numpy(), rtol=0, atol=1e-14)
            and np.allclose(model.oof_threshold, tom_part.locked_threshold.to_numpy(), rtol=0, atol=1e-14)
            and np.array_equal(model.oof_fold, tom_part.tom_outer_fold.to_numpy())
        )
        score, probability, prediction = model.predict(feature[name][cold_positions])
        external_equal = (
            np.allclose(score, cold_part.uncalibrated_score.to_numpy(), rtol=0, atol=1e-12)
            and np.allclose(probability, cold_part.calibrated_probability_disorder.to_numpy(), rtol=0, atol=1e-12)
            and np.array_equal(prediction, cold_part.predicted_disorder.to_numpy())
        )
        checks.extend(
            [
                {"check": f"{name}: serialized OOF fields equal TOM prediction CSV", "passed": bool(oof_equal), "detail": f"oof_n={len(model.oof_score)}"},
                {"check": f"{name}: serialized final model exactly regenerates COLD predictions", "passed": bool(external_equal), "detail": f"C={model.final_c}; threshold={model.threshold:.15g}"},
                {"check": f"{name}: final calibrator/threshold derived from TOM-sized OOF stream", "passed": len(model.oof_score) == len(tom_part), "detail": f"n={len(model.oof_score)}"},
            ]
        )
    return checks


def paired_bootstrap(predictions: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    long = predictions["cold_locked_external"].sort_values(["sample_id", "model"])
    p = long.pivot(index="sample_id", columns="model", values="calibrated_probability_disorder")
    t = long.pivot(index="sample_id", columns="model", values="locked_threshold").loc[p.index]
    y = long.groupby("sample_id").foliar_binary.first().astype(int).loc[p.index].to_numpy()
    groups = long.groupby("sample_id").source_group.first().astype(str).loc[p.index].to_numpy()
    reference = "resnet18_logit"
    original_path = RESULTS / "tables" / "cold_paired_model_differences.csv"
    if original_path.exists():
        original = pd.read_csv(original_path)
        comparators = original.comparator_model.drop_duplicates().tolist()
        replicate_values = original.bootstrap_replicates.drop_duplicates().astype(int).tolist()
        if len(replicate_values) != 1:
            raise RuntimeError(f"paired table has inconsistent replicate counts: {replicate_values}")
        paired_replicates = replicate_values[0]
    else:
        original = None
        comparators = [name for name in p.columns if name != reference]
        paired_replicates = 5000

    def one(metric: str, probability: np.ndarray, threshold: float, indices: np.ndarray | None = None) -> float:
        if indices is not None:
            probability = probability[indices]
            yy = y[indices]
        else:
            yy = y
        if metric == "balanced_accuracy":
            return float(balanced_accuracy_score(yy, probability >= threshold))
        if metric == "auroc":
            return float(roc_auc_score(yy, probability))
        if metric == "brier":
            return float(brier_score_loss(yy, probability))
        raise ValueError(metric)

    point: dict[tuple[str, str], float] = {}
    distributions = {(comp, metric): [] for comp in comparators for metric in ("balanced_accuracy", "auroc", "brier")}
    for comp in comparators:
        for metric in ("balanced_accuracy", "auroc", "brier"):
            ref = one(metric, p[reference].to_numpy(), float(t[reference].iloc[0]))
            cmp = one(metric, p[comp].to_numpy(), float(t[comp].iloc[0]))
            point[(comp, metric)] = cmp - ref if metric == "brier" else ref - cmp
    rng = np.random.default_rng(SEED + 4001)
    group_frame = pd.DataFrame({"position": np.arange(len(groups)), "group": groups.astype(str)})
    position_groups = group_frame.groupby("group", sort=False).position.apply(lambda values: values.to_numpy()).tolist()
    completed = 0
    while completed < paired_replicates:
        sampled_groups = rng.integers(0, len(position_groups), size=len(position_groups))
        indices = np.concatenate([position_groups[index] for index in sampled_groups])
        if len(np.unique(y[indices])) != 2:
            continue
        completed += 1
        for comp in comparators:
            for metric in ("balanced_accuracy", "auroc", "brier"):
                ref = one(metric, p[reference].to_numpy(), float(t[reference].iloc[0]), indices)
                cmp = one(metric, p[comp].to_numpy(), float(t[comp].iloc[0]), indices)
                distributions[(comp, metric)].append(cmp - ref if metric == "brier" else ref - cmp)
    rows = []
    for (comp, metric), values in distributions.items():
        values = np.asarray(values)
        rows.append(
            {
                "reference_model": reference,
                "comparator_model": comp,
                "metric": metric,
                "improvement_positive_favors_reference": point[(comp, metric)],
                "ci_low": float(np.quantile(values, 0.025)),
                "ci_high": float(np.quantile(values, 0.975)),
                "bootstrap_replicates": len(values),
                "tail_fraction_at_or_below_zero": float(np.mean(values <= 0)),
            }
        )
    recomputed = pd.DataFrame(rows)
    if original is None:
        return recomputed, None
    merged = original.merge(recomputed, on=["reference_model", "comparator_model", "metric"], suffixes=("_reported", "_recomputed"), validate="one_to_one")
    for column in ("improvement_positive_favors_reference", "ci_low", "ci_high", "tail_fraction_at_or_below_zero"):
        merged[f"{column}_abs_diff"] = (merged[f"{column}_reported"] - merged[f"{column}_recomputed"]).abs()
    return recomputed, merged


def main() -> None:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    snapshot = input_snapshot()
    snapshot.to_csv(AUDIT_DIR / "audited_input_checksums.csv", index=False)
    manifest = pd.read_csv(ROOT / "manifests" / "unified_manifest.csv")
    comparison, checks, predictions = verify_predictions(manifest)
    comparison.to_csv(AUDIT_DIR / "recomputed_metric_comparison.csv", index=False)
    hash_frame, hash_checks, data_summary = verify_splits_and_hashes(manifest, predictions)
    hash_frame.to_csv(AUDIT_DIR / "recomputed_primary_image_hashes.csv", index=False)
    checks.extend(hash_checks)
    checks.extend(verify_saved_models(manifest, predictions))
    paired, paired_comparison = paired_bootstrap(predictions)
    paired.to_csv(AUDIT_DIR / "recomputed_cold_paired_model_differences.csv", index=False)
    if paired_comparison is not None:
        paired_comparison.to_csv(AUDIT_DIR / "reported_vs_recomputed_cold_paired_differences.csv", index=False)
        paired_max_diff = float(paired_comparison.filter(like="_abs_diff").max().max())
        checks.append({"check": "reported paired COLD bootstrap table reproduces", "passed": paired_max_diff <= 1e-12, "detail": f"max_abs_diff={paired_max_diff:.3g}"})

    check_frame = pd.DataFrame(checks)
    check_frame.to_csv(AUDIT_DIR / "verification_checks.csv", index=False)
    summary = {
        "audit_snapshot_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "input_script_sha256": sha256_file(ROOT / "code" / "run_reliability_benchmark.py"),
        "all_mechanical_checks_pass": bool(check_frame.passed.all()),
        "checks_passed": int(check_frame.passed.sum()),
        "checks_total": int(len(check_frame)),
        "data": data_summary,
        "metric_rows": int(len(comparison)),
        "metric_max_estimate_abs_diff": float(comparison.estimate_abs_diff.max()),
        "metric_max_ci_low_abs_diff": float(comparison.ci_low_abs_diff.max()),
        "metric_max_ci_high_abs_diff": float(comparison.ci_high_abs_diff.max()),
        "cold_groups_are_singletons": data_summary["cold_unique_source_groups"] == data_summary["cold_included"],
    }
    (AUDIT_DIR / "mechanical_verification_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
