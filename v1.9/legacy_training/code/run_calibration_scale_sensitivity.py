"""Post hoc source-only calibration construction sensitivity.

This diagnostic preserves the published primary results.  It compares Platt
calibration of a mixture of fold-selected-C OOF scores with calibration of
OOF scores generated at the already selected all-TOM C.  Both calibrators
operate on the SAME final all-TOM head. No external label selects a pipeline.
The source probabilities used to fit calibration are not validation scores.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / "results/reliability_benchmark_v1"
RESULTS = ROOT / "results/calibration_scale_sensitivity_v1"
SEED = 20260930
MODEL = "resnet18_logit"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def head(c: float) -> Pipeline:
    return Pipeline([
        ("scale", StandardScaler()),
        ("classifier", LogisticRegression(
            C=c, penalty="l2", solver="liblinear", class_weight="balanced",
            random_state=SEED, max_iter=10000)),
    ])


def calibrate(score: np.ndarray, y: np.ndarray):
    model = LogisticRegression(C=1e6, solver="lbfgs", random_state=SEED, max_iter=10000)
    model.fit(score.reshape(-1, 1), y)
    probability = model.predict_proba(score.reshape(-1, 1))[:, 1]
    fpr, tpr, thresholds = roc_curve(y, probability)
    finite = np.flatnonzero(np.isfinite(thresholds))
    threshold = float(thresholds[finite[np.argmax((tpr - fpr)[finite])]])
    return model, threshold, probability


def metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, float]:
    prediction = p >= threshold
    positive, negative = y == 1, y == 0
    sensitivity = float(prediction[positive].mean())
    specificity = float((~prediction[negative]).mean())
    tp, fp = int(np.sum(prediction & positive)), int(np.sum(prediction & negative))
    fn, tn = int(np.sum(~prediction & positive)), int(np.sum(~prediction & negative))
    pclip = np.clip(p, 1e-7, 1 - 1e-7)
    bins = np.minimum(np.digitize(p, np.linspace(0, 1, 11)[1:-1], right=True), 9)
    ece = sum(float((bins == b).mean()) * abs(float(y[bins == b].mean()) - float(p[bins == b].mean()))
              for b in range(10) if (bins == b).any())
    return {
        "balanced_accuracy": (sensitivity + specificity) / 2,
        "sensitivity": sensitivity, "specificity": specificity,
        "macro_f1": .5 * (2 * tp / max(1, 2 * tp + fp + fn) + 2 * tn / max(1, 2 * tn + fp + fn)),
        "auroc": float(roc_auc_score(y, p)),
        "average_precision": float(average_precision_score(y, p)),
        "brier": float(np.mean((p - y) ** 2)),
        "nll": float(-np.mean(y * np.log(pclip) + (1 - y) * np.log(1 - pclip))),
        "ece_10": float(ece),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=RESULTS)
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if args.bootstrap < 100:
        raise ValueError("at least 100 bootstrap replicates are required")
    out.mkdir(parents=True, exist_ok=True)
    if (out / "prespecified_protocol.json").exists():
        raise FileExistsError("preserve the existing diagnostic; choose a new --output-dir for a rerun")
    paths = {
        "features": ROOT / "features/resnet18.npz",
        "manifest": ROOT / "manifests/unified_manifest.csv",
        "model_summary": ORIGINAL / "models/model_summary.json",
        "folds": ORIGINAL / "split_manifests/tom_acquisition_day_outer_folds.csv",
        "source_predictions": ORIGINAL / "predictions/tom_nested_oof.csv",
        "external_predictions": ORIGINAL / "predictions/cold_locked_external.csv",
        "script": Path(__file__).resolve(),
    }
    summaries = json.loads(paths["model_summary"].read_text(encoding="utf-8"))
    summary = next(row for row in summaries if row["model"] == MODEL)
    final_c = float(summary["final_C_selected_with_TOM_grouped_CV"])
    fold_c = [float(v) for v in summary["outer_fold_best_C"]]
    protocol = {
        "frozen_utc_before_model_fitting": datetime.now(timezone.utc).isoformat(),
        "analysis_status": "post_hoc_source_only_protocol_sensitivity_not_new_confirmation",
        "external_results_already_known_before_design": True,
        "model": MODEL, "final_C_already_selected": final_c,
        "outer_fold_C_already_selected": fold_c,
        "comparison": ["reconstructed_legacy_mixed_C", "fixed_final_C_OOF"],
        "same_all_TOM_final_head_for_both_arms": True,
        "no_retuning_no_target_label_selection": True,
        "source_calibration_training_probabilities_are_not_unbiased_source_validation": True,
        "all_results_retained": True,
        "bootstrap_replicates": args.bootstrap,
        "bootstrap_unit": "exact_deduplicated_COLD_image_conditional_on_fitted_models",
        "seed": SEED,
        "inputs": {key: {"path": str(value.relative_to(ROOT)) if value.is_relative_to(ROOT) else str(value),
                          "sha256": sha(value)} for key, value in paths.items()},
        "environment": {"python": sys.version, "platform": platform.platform(),
                        "numpy": np.__version__, "pandas": pd.__version__,
                        "scipy": scipy.__version__, "scikit_learn": sklearn.__version__},
    }
    write_json(out / "prespecified_protocol.json", protocol)

    source_all = pd.read_csv(paths["source_predictions"])
    source = source_all.loc[source_all.model.eq(MODEL)].reset_index(drop=True)
    external_all = pd.read_csv(paths["external_predictions"])
    external = external_all.loc[external_all.model.eq(MODEL)].reset_index(drop=True)
    folds = pd.read_csv(paths["folds"])
    manifest = pd.read_csv(paths["manifest"])
    archive = np.load(paths["features"], allow_pickle=False)
    feature_ids = archive["sample_ids"].astype(str).tolist()
    if feature_ids != manifest.sample_id.astype(str).tolist():
        raise RuntimeError("feature order does not match original manifest")
    lookup = {sample: position for position, sample in enumerate(feature_ids)}
    if len(lookup) != len(feature_ids):
        raise RuntimeError("duplicate feature sample IDs")
    x_source = archive["features"][[lookup[v] for v in source.sample_id.astype(str)]].astype(np.float64)
    x_external = archive["features"][[lookup[v] for v in external.sample_id.astype(str)]].astype(np.float64)
    if len(source) != 1643 or len(external) != 813 or not np.isfinite(x_source).all() or not np.isfinite(x_external).all():
        raise RuntimeError("unexpected analysis population or nonfinite features")
    if not source.sample_id.astype(str).equals(folds.sample_id.astype(str)):
        raise RuntimeError("published source predictions and fold IDs differ")
    if not np.array_equal(source.tom_outer_fold.to_numpy(), folds.outer_fold.to_numpy()):
        raise RuntimeError("published source predictions and fold numbers differ")
    if source.analysis_sha256.duplicated().any() or external.analysis_sha256.duplicated().any():
        raise RuntimeError("duplicated exact images in frozen evaluation streams")
    if set(source.analysis_sha256).intersection(external.analysis_sha256):
        raise RuntimeError("source/external exact overlap")
    y = source.foliar_binary.astype(int).to_numpy()
    legacy_score, fixed_score = np.full(len(y), np.nan), np.full(len(y), np.nan)
    fold_rows = []
    for k, c in enumerate(fold_c):
        test = source.tom_outer_fold.eq(k).to_numpy()
        train = ~test
        if set(source.loc[train, "acquisition_day_utc"]).intersection(source.loc[test, "acquisition_day_utc"]):
            raise RuntimeError("acquisition-day overlap")
        for target, c_value in ((legacy_score, c), (fixed_score, final_c)):
            fitted = head(c_value).fit(x_source[train], y[train])
            target[test] = fitted.decision_function(x_source[test])
        delta = fixed_score[test] - legacy_score[test]
        fold_rows.append({
            "outer_fold": k, "n_train": int(train.sum()), "n_test": int(test.sum()),
            "legacy_C": c, "fixed_C": final_c, "changed_C": c != final_c,
            "legacy_score_mean": float(legacy_score[test].mean()),
            "legacy_score_std": float(legacy_score[test].std(ddof=1)),
            "fixed_score_mean": float(fixed_score[test].mean()),
            "fixed_score_std": float(fixed_score[test].std(ddof=1)),
            "fixed_minus_legacy_mean": float(delta.mean()),
            "fixed_minus_legacy_mean_abs": float(np.abs(delta).mean()),
            "fixed_minus_legacy_max_abs": float(np.abs(delta).max()),
            "reconstructed_minus_published_score_max_abs": float(np.max(np.abs(
                legacy_score[test] - source.loc[test, "uncalibrated_score"].to_numpy()))),
        })
    if not np.isfinite(legacy_score).all() or not np.isfinite(fixed_score).all():
        raise RuntimeError("incomplete OOF source scores")
    pd.DataFrame(fold_rows).to_csv(out / "source_fold_score_comparison.csv", index=False)
    final_head = head(final_c).fit(x_source, y)
    common_external_score = final_head.decision_function(x_external)
    source_output = source[["sample_id", "analysis_sha256", "acquisition_day_utc", "tom_outer_fold", "foliar_binary"]].copy()
    source_output["published_legacy_score"] = source.uncalibrated_score
    source_output["reconstructed_legacy_score"] = legacy_score
    source_output["fixed_final_C_score"] = fixed_score
    calibration_rows, prediction_rows = [], []
    predictions = {}
    for arm, score in (("reconstructed_legacy_mixed_C", legacy_score), ("fixed_final_C_OOF", fixed_score)):
        calibrator, threshold, source_probability = calibrate(score, y)
        slope, intercept = float(calibrator.coef_[0, 0]), float(calibrator.intercept_[0])
        if slope <= 0:
            raise RuntimeError("unexpected nonpositive Platt slope; inspect instead of silently reversing ranking")
        score_threshold = (np.log(threshold / (1 - threshold)) - intercept) / slope
        calibration_rows.append({"arm": arm, "final_head_C": final_c,
                                 "platt_slope": slope, "platt_intercept": intercept,
                                 "probability_threshold": threshold, "raw_score_threshold": float(score_threshold),
                                 "calibration_source_n": len(y), "interpretation": "source_fitted_not_source_validation"})
        source_output[f"{arm}_calibration_training_probability"] = source_probability
        probability = calibrator.predict_proba(common_external_score.reshape(-1, 1))[:, 1]
        predictions[arm] = (probability, threshold)
        prediction = external[["sample_id", "analysis_sha256", "original_label", "foliar_binary"]].copy()
        prediction.insert(0, "arm", arm)
        prediction["common_final_head_raw_score"] = common_external_score
        prediction["calibrated_probability_disorder"] = probability
        prediction["source_only_threshold"] = threshold
        prediction["predicted_disorder"] = (probability >= threshold).astype(int)
        prediction_rows.append(prediction)
    pd.DataFrame(calibration_rows).to_csv(out / "calibration_parameters.csv", index=False)
    source_output.to_csv(out / "source_calibration_scores.csv", index=False)
    pd.concat(prediction_rows, ignore_index=True).to_csv(out / "external_predictions.csv", index=False)

    # External labels are used only below, to report EVERY frozen comparison.
    y_ext = external.foliar_binary.astype(int).to_numpy()
    point = {arm: metrics(y_ext, probability, threshold) for arm, (probability, threshold) in predictions.items()}
    dist = {arm: {metric: [] for metric in values if metric != "ece_10"} for arm, values in point.items()}
    rng = np.random.default_rng(SEED + 202)
    completed = 0
    while completed < args.bootstrap:
        take = rng.choice(len(y_ext), len(y_ext), replace=True)
        if np.unique(y_ext[take]).size != 2:
            continue
        for arm, (probability, threshold) in predictions.items():
            result = metrics(y_ext[take], probability[take], threshold)
            for metric in dist[arm]:
                dist[arm][metric].append(result[metric])
        completed += 1
    metric_rows, difference_rows = [], []
    for arm, values in point.items():
        for metric, estimate in values.items():
            samples = dist[arm].get(metric, [])
            lo, hi = np.quantile(samples, [.025, .975]) if samples else (np.nan, np.nan)
            metric_rows.append({"arm": arm, "metric": metric, "estimate": estimate, "ci_low": lo,
                                "ci_high": hi, "bootstrap_replicates": len(samples), "n_images": len(y_ext),
                                "bootstrap_unit": "exact_deduplicated_COLD_image_conditional_on_fitted_models"})
    legacy, fixed = "reconstructed_legacy_mixed_C", "fixed_final_C_OOF"
    for metric in point[legacy]:
        samples = np.asarray(dist[fixed].get(metric, [])) - np.asarray(dist[legacy].get(metric, []))
        lo, hi = np.quantile(samples, [.025, .975]) if len(samples) else (np.nan, np.nan)
        difference_rows.append({"contrast": "fixed_C_minus_reconstructed_legacy", "metric": metric,
                                "difference": point[fixed][metric] - point[legacy][metric],
                                "ci_low": lo, "ci_high": hi, "paired_bootstrap_replicates": len(samples)})
    pd.DataFrame(metric_rows).to_csv(out / "external_metrics.csv", index=False)
    pd.DataFrame(difference_rows).to_csv(out / "paired_external_differences.csv", index=False)
    published_probability = external.calibrated_probability_disorder.to_numpy()
    old_threshold = float(external.locked_threshold.iloc[0])
    old_metrics = metrics(y_ext, published_probability, old_threshold)
    pd.DataFrame([{"metric": metric, "published": value, "reconstructed": point[legacy][metric],
                   "reconstructed_minus_published": point[legacy][metric] - value}
                  for metric, value in old_metrics.items()]).to_csv(out / "published_reconstruction_metrics.csv", index=False)
    reconstruction = {
        "source_legacy_score_max_abs_error": float(np.max(np.abs(legacy_score - source.uncalibrated_score.to_numpy()))),
        "external_raw_score_max_abs_error": float(np.max(np.abs(common_external_score - external.uncalibrated_score.to_numpy()))),
        "external_probability_max_abs_error": float(np.max(np.abs(predictions[legacy][0] - published_probability))),
        "threshold_absolute_difference": abs(predictions[legacy][1] - old_threshold),
        "external_label_decision_disagreements": int(np.sum((predictions[legacy][0] >= predictions[legacy][1]) != external.predicted_disorder.to_numpy())),
        "shared_external_raw_score_two_arms": True,
        "old_results_modified": False,
    }
    # Recheck every original input after the diagnostic, proving non-mutation.
    reconstruction["input_hashes_unchanged"] = all(sha(value) == protocol["inputs"][key]["sha256"] for key, value in paths.items())
    if not reconstruction["input_hashes_unchanged"]:
        raise RuntimeError("an original input changed during this diagnostic")
    write_json(out / "reconstruction_audit.json", reconstruction)
    report = f"""# Source-only calibration construction sensitivity

This is a post hoc sensitivity analysis designed after the published COLD results were known. It is not a new independently locked external confirmation. The protocol and original input hashes were saved before fitting. No architecture, hyperparameter, calibration family, threshold objective, or retained result was selected with COLD labels.

The fixed primary ResNet18 feature matrix, 1,643 TOM samples, five acquisition-day folds, and already selected final C={final_c:g} were reused. The original fold-selected C values were {fold_c}. Two source score streams were reconstructed in one environment: each fold's published C, or the already selected final C for every fold. Standardization was fitted on the corresponding fold's training images. Each stream fitted a logistic Platt calibrator and a Youden threshold. Both pipelines used the same final logistic head fitted to all TOM samples at C={final_c:g}; only source score construction, calibration and the resulting threshold differed. The probabilities used to train these calibrators are not reported as unbiased source-validation results.

The paired external comparison uses the same 813 exact-deduplicated COLD images. The {args.bootstrap:,} paired image bootstrap replicates hold both fitted pipelines fixed and do not incorporate model-training, plant, farm, or dataset uncertainty. ECE has no interval. The original frozen metrics remain the primary analysis.

| Construction | Balanced accuracy | AUROC | Brier | Sensitivity | Specificity |
|---|---:|---:|---:|---:|---:|
| Reconstructed legacy mixed-C calibration | {point[legacy]['balanced_accuracy']:.6f} | {point[legacy]['auroc']:.6f} | {point[legacy]['brier']:.6f} | {point[legacy]['sensitivity']:.6f} | {point[legacy]['specificity']:.6f} |
| Fixed-final-C OOF calibration | {point[fixed]['balanced_accuracy']:.6f} | {point[fixed]['auroc']:.6f} | {point[fixed]['brier']:.6f} | {point[fixed]['sensitivity']:.6f} | {point[fixed]['specificity']:.6f} |

The unchanged common external scores imply identical AUROC under the positive-slope Platt calibrators; AUROC equality is not an additional empirical replication. The analysis diagnoses probability scale and threshold construction, not a new representation or a remedy for external data shift. Refitting from 80% fold-training samples to all TOM samples can still cause score-scale change in both arms, and the fixed-C comparison does not eliminate that possibility.

Reconstruction in scikit-learn {sklearn.__version__} is compared with the published predictions rather than assumed identical. Maximum original/reconstructed differences: source raw score {reconstruction['source_legacy_score_max_abs_error']:.12g}; external raw score {reconstruction['external_raw_score_max_abs_error']:.12g}; external probability {reconstruction['external_probability_max_abs_error']:.12g}. Original/reconstructed locked decisions differ for {reconstruction['external_label_decision_disagreements']} of 813 external images. See `reconstruction_audit.json` and `published_reconstruction_metrics.csv` before interpreting small contrasts.

All source score streams, fitted calibration coefficients, external probabilities, metric summaries and paired contrasts are retained. No old model was deserialized and no published input or result was modified.
"""
    (out / "METHODS_AND_RESULTS.md").write_text(report, encoding="utf-8")
    outputs = [{"path": p.name, "bytes": p.stat().st_size, "sha256": sha(p)}
               for p in sorted(out.iterdir()) if p.is_file() and p.name != "output_sha256.csv"]
    pd.DataFrame(outputs).to_csv(out / "output_sha256.csv", index=False)
    print(json.dumps({"output": str(out), "reconstruction": reconstruction, "external_metrics": point}, indent=2))


if __name__ == "__main__":
    main()
