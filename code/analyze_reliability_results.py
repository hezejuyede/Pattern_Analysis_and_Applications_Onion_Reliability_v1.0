"""Secondary, prespecified-style interpretation of the locked benchmark outputs.

This script never refits a model. It quantifies paired baseline differences,
the internal-to-external transport gap, per-diagnosis behavior, and outer-fold
variability from frozen per-image predictions.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "reliability_benchmark_v1"
SEED = 20260930
REPLICATES = 3000
REFERENCE = "resnet18_logit"
COMPARATORS = [
    "prior_prevalence",
    "color_shortcut_logit",
    "handcrafted_logit",
    "efficientnet_b0_logit",
    "convnext_tiny_logit",
    "swin_t_logit",
]


def metric_value(metric: str, y: np.ndarray, probability: np.ndarray, threshold: float | np.ndarray) -> float:
    if metric == "balanced_accuracy":
        return float(balanced_accuracy_score(y, probability >= threshold))
    if metric == "auroc":
        return float(roc_auc_score(y, probability))
    if metric == "brier":
        return float(brier_score_loss(y, probability))
    raise ValueError(metric)


def grouped_positions(groups: np.ndarray) -> list[np.ndarray]:
    frame = pd.DataFrame({"position": np.arange(len(groups)), "group": groups.astype(str)})
    return frame.groupby("group", sort=False).position.apply(lambda s: s.to_numpy()).tolist()


def grouped_resample(position_groups: list[np.ndarray], rng: np.random.Generator) -> np.ndarray:
    sampled = rng.integers(0, len(position_groups), size=len(position_groups))
    return np.concatenate([position_groups[index] for index in sampled])


def aligned_predictions(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    long = pd.read_csv(path)
    long = long.sort_values(["sample_id", "model"])
    probability = long.pivot(index="sample_id", columns="model", values="calibrated_probability_disorder")
    threshold = long.pivot(index="sample_id", columns="model", values="locked_threshold").loc[probability.index]
    truth = long.groupby("sample_id").foliar_binary.first().astype(int).loc[probability.index]
    groups = long.groupby("sample_id").source_group.first().astype(str).loc[probability.index]
    metadata = long.drop_duplicates("sample_id").set_index("sample_id").loc[probability.index]
    return long, probability, threshold, truth, groups, metadata


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total == 0:
        return np.nan, np.nan
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


def main() -> None:
    tables = RESULTS / "tables"
    audit = RESULTS / "audit"
    cold_long, cold_p, cold_threshold_matrix, cold_y_s, cold_groups_s, cold_metadata = aligned_predictions(
        RESULTS / "predictions" / "cold_locked_external.csv"
    )
    tom_long, tom_p, tom_threshold_matrix, tom_y_s, _, tom_metadata = aligned_predictions(
        RESULTS / "predictions" / "tom_nested_oof.csv"
    )
    if (cold_threshold_matrix.nunique(axis=0) != 1).any():
        raise RuntimeError("COLD predictions do not use one final TOM-locked threshold per model")
    thresholds = cold_threshold_matrix.iloc[0]
    cold_y = cold_y_s.to_numpy()
    cold_groups = cold_groups_s.to_numpy()
    cold_position_groups = grouped_positions(cold_groups)

    # Paired group bootstrap on the locked COLD images. For Brier, positive
    # improvement means the reference has lower error.
    rng = np.random.default_rng(SEED + 4001)
    point_differences: dict[tuple[str, str], float] = {}
    distributions = {(comparator, metric): [] for comparator in COMPARATORS for metric in ("balanced_accuracy", "auroc", "brier")}
    for comparator in COMPARATORS:
        for metric in ("balanced_accuracy", "auroc", "brier"):
            ref = metric_value(metric, cold_y, cold_p[REFERENCE].to_numpy(), float(thresholds[REFERENCE]))
            comp = metric_value(metric, cold_y, cold_p[comparator].to_numpy(), float(thresholds[comparator]))
            point_differences[(comparator, metric)] = comp - ref if metric == "brier" else ref - comp
    completed = 0
    while completed < REPLICATES:
        indices = grouped_resample(cold_position_groups, rng)
        if len(np.unique(cold_y[indices])) != 2:
            continue
        completed += 1
        for comparator in COMPARATORS:
            for metric in ("balanced_accuracy", "auroc", "brier"):
                ref = metric_value(metric, cold_y[indices], cold_p[REFERENCE].to_numpy()[indices], float(thresholds[REFERENCE]))
                comp = metric_value(metric, cold_y[indices], cold_p[comparator].to_numpy()[indices], float(thresholds[comparator]))
                improvement = comp - ref if metric == "brier" else ref - comp
                distributions[(comparator, metric)].append(improvement)
    paired_rows = []
    for key, values in distributions.items():
        comparator, metric = key
        values = np.asarray(values)
        paired_rows.append(
            {
                "reference_model": REFERENCE,
                "comparator_model": comparator,
                "metric": metric,
                "improvement_positive_favors_reference": point_differences[key],
                "ci_low": float(np.quantile(values, 0.025)),
                "ci_high": float(np.quantile(values, 0.975)),
                "bootstrap_replicates": len(values),
                "bootstrap_unit": "COLD exact-deduplicated raw SHA group",
                "tail_fraction_at_or_below_zero": float(np.mean(values <= 0)),
            }
        )
    pd.DataFrame(paired_rows).to_csv(tables / "cold_paired_model_differences.csv", index=False)

    # Transport gap uses independent cluster bootstraps: TOM acquisition day and
    # COLD exact-SHA group. External-minus-internal is negative for accuracy/AUC;
    # for Brier, a positive value denotes worse external calibration/error.
    tom_days = pd.to_datetime(
        tom_metadata.source_group.str.extract(r"(1\d{12})$", expand=False).astype("int64"), unit="ms", utc=True
    ).dt.strftime("%Y-%m-%d").to_numpy()
    tom_y = tom_y_s.to_numpy()
    tom_day_position_groups = grouped_positions(tom_days)
    rng = np.random.default_rng(SEED + 4002)
    transport_rows = []
    for model in cold_p.columns:
        for metric in ("balanced_accuracy", "auroc", "brier"):
            internal = metric_value(metric, tom_y, tom_p[model].to_numpy(), tom_threshold_matrix[model].to_numpy())
            external = metric_value(metric, cold_y, cold_p[model].to_numpy(), float(thresholds[model]))
            gaps = []
            while len(gaps) < REPLICATES:
                tom_indices = grouped_resample(tom_day_position_groups, rng)
                cold_indices = grouped_resample(cold_position_groups, rng)
                if len(np.unique(tom_y[tom_indices])) != 2 or len(np.unique(cold_y[cold_indices])) != 2:
                    continue
                internal_b = metric_value(
                    metric,
                    tom_y[tom_indices],
                    tom_p[model].to_numpy()[tom_indices],
                    tom_threshold_matrix[model].to_numpy()[tom_indices],
                )
                external_b = metric_value(
                    metric, cold_y[cold_indices], cold_p[model].to_numpy()[cold_indices], float(thresholds[model])
                )
                gaps.append(external_b - internal_b)
            transport_rows.append(
                {
                    "model": model,
                    "metric": metric,
                    "TOM_nested_oof": internal,
                    "COLD_locked_external": external,
                    "external_minus_internal": external - internal,
                    "ci_low": float(np.quantile(gaps, 0.025)),
                    "ci_high": float(np.quantile(gaps, 0.975)),
                    "bootstrap_replicates": len(gaps),
                    "internal_bootstrap_unit": "TOM acquisition day",
                    "external_bootstrap_unit": "COLD exact-deduplicated raw SHA group",
                }
            )
    pd.DataFrame(transport_rows).to_csv(tables / "internal_external_transport_gap.csv", index=False)

    # Outer-fold variability and per-diagnosis behavior reveal what aggregate
    # balanced accuracy hides.
    fold_rows = []
    for (model, fold), part in tom_long.groupby(["model", "tom_outer_fold"]):
        y = part.foliar_binary.astype(int).to_numpy()
        probability = part.calibrated_probability_disorder.to_numpy()
        threshold = part.locked_threshold.to_numpy()
        fold_rows.append(
            {
                "model": model,
                "outer_fold": int(fold),
                "images": len(part),
                "acquisition_days": pd.to_datetime(
                    part.source_group.str.extract(r"(1\d{12})$", expand=False).astype("int64"), unit="ms", utc=True
                ).dt.strftime("%Y-%m-%d").nunique(),
                "balanced_accuracy": balanced_accuracy_score(y, probability >= threshold),
                "auroc": roc_auc_score(y, probability),
                "brier": brier_score_loss(y, probability),
            }
        )
    pd.DataFrame(fold_rows).to_csv(tables / "tom_outer_fold_variability.csv", index=False)

    subgroup_rows = []
    for analysis_set, long in (("TOM_nested_oof", tom_long), ("COLD_locked_external", cold_long)):
        for (model, label), part in long.groupby(["model", "original_label"]):
            truth = int(part.foliar_binary.iloc[0])
            predicted = part.predicted_disorder.astype(int)
            correct = int((predicted == truth).sum())
            low, high = wilson(correct, len(part))
            subgroup_rows.append(
                {
                    "analysis_set": analysis_set,
                    "model": model,
                    "dataset_label": label,
                    "binary_truth": truth,
                    "n_images": len(part),
                    "correct_images": correct,
                    "correct_rate": correct / len(part),
                    "wilson_95_low": low,
                    "wilson_95_high": high,
                    "mean_predicted_disorder_probability": part.calibrated_probability_disorder.mean(),
                    "median_predicted_disorder_probability": part.calibrated_probability_disorder.median(),
                }
            )
    pd.DataFrame(subgroup_rows).to_csv(tables / "per_dataset_label_performance.csv", index=False)
    cold_error = (
        cold_long.groupby(["model", "original_label", "foliar_binary", "predicted_disorder"], as_index=False)
        .agg(images=("sample_id", "size"))
    )
    cold_error["class_total"] = cold_error.groupby(["model", "original_label"]).images.transform("sum")
    cold_error["within_class_fraction"] = cold_error.images / cold_error.class_total
    cold_error["prediction_status"] = cold_error.predicted_disorder.map({0: "predicted_healthy", 1: "predicted_disorder"})
    cold_error.to_csv(tables / "cold_original_class_by_predicted_status.csv", index=False)

    metrics = pd.read_csv(tables / "binary_metrics_group_bootstrap.csv")
    def get(set_name: str, model: str, metric: str) -> pd.Series:
        return metrics[(metrics.analysis_set == set_name) & (metrics.model == model) & (metrics.metric == metric)].iloc[0]

    res_ba = get("cold_locked_external", REFERENCE, "balanced_accuracy")
    res_auc = get("cold_locked_external", REFERENCE, "auroc")
    res_sens = get("cold_locked_external", REFERENCE, "sensitivity")
    dg_ba = get("digigreen_onion_case_series", REFERENCE, "balanced_accuracy")
    ood = pd.read_csv(tables / "open_set_ood_metrics.csv")
    ood_row = ood[
        (ood.known_set == "cold_locked_external")
        & (ood.model == REFERENCE)
        & (ood.detector == "source_5nn_cosine")
    ].iloc[0]
    transport = pd.read_csv(tables / "internal_external_transport_gap.csv")
    res_gap = transport[(transport.model == REFERENCE) & (transport.metric == "balanced_accuracy")].iloc[0]
    gate = {
        "status": "HOLD_NOT_SUBMISSION_READY",
        "locked_external_validation_design": "PASS_WITH_METADATA_LIMITATION",
        "strong_baselines": "PASS",
        "development_leakage_controls": "PASS",
        "cross_domain_discrimination": "INSUFFICIENT_FOR_DEPLOYMENT_CLAIM",
        "operational_case_series": "FAIL_AS_CONFIRMATORY_VALIDATION",
        "open_set_rejection": "FAIL_AS_DEPLOYMENT_SAFETY_EVIDENCE",
        "key_evidence": {
            "COLD_resnet18_balanced_accuracy": float(res_ba.estimate),
            "COLD_resnet18_balanced_accuracy_CI": [float(res_ba.ci_low), float(res_ba.ci_high)],
            "COLD_resnet18_AUROC": float(res_auc.estimate),
            "COLD_resnet18_sensitivity": float(res_sens.estimate),
            "TOM_to_COLD_balanced_accuracy_gap": float(res_gap.external_minus_internal),
            "DigitalGreen_case_series_balanced_accuracy": float(dg_ba.estimate),
            "DigitalGreen_case_series_n": int(dg_ba.n_images),
            "non_onion_OOD_5NN_AUROC": float(ood_row.ood_auroc),
            "non_onion_OOD_FPR95": float(ood_row.fpr_at_95pct_ood_tpr),
        },
        "release_blockers": [
            "The a-priori primary model's external balanced accuracy is moderate and disorder sensitivity is below 0.50 at the TOM-locked threshold.",
            "The internal-to-external performance loss is large, directly contradicting any broad robustness or deployment claim.",
            "The 24-image Digital Green onion set is a heterogeneous descriptive case series and performs near chance at the locked threshold.",
            "Non-onion OOD rejection remains weak at the operating point required for 95% unknown detection.",
            "Neither public dataset exposes verified plant-level IDs, and COLD lacks acquisition-day grouping metadata.",
            "No independent phytopathologist re-adjudication or prospective field cohort is available.",
        ],
        "publishable_core_if_reframed": "A negative, leakage-controlled cross-country reliability audit may be scientifically useful, but it requires a journal fit check, pathology review of label harmonization, and stronger independent field evidence before a full JPDP submission package is defensible.",
    }
    (audit / "release_gate.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")
    report = f"""# Evidence gate after locked cross-country evaluation

**Decision: HOLD — the current evidence does not justify a submission-ready or deployment-level claim.**

The design passed the core independence test: all choices were made on TOM2024 acquisition-day-grouped validation, and COLD remained computationally excluded from fitting. The a-priori primary ResNet18 plus regularized logistic regression reached COLD balanced accuracy {res_ba.estimate:.3f} (95% group-bootstrap CI {res_ba.ci_low:.3f}–{res_ba.ci_high:.3f}) and AUROC {res_auc.estimate:.3f}. Its disorder sensitivity was only {res_sens.estimate:.3f}. Balanced accuracy fell by {abs(res_gap.external_minus_internal):.3f} from TOM nested validation to COLD; this is evidence of material transport failure, not broad cross-country robustness. EfficientNet-B0, ConvNeXt-Tiny, and Swin-T are reported as secondary architecture sensitivities and do not replace the primary based on COLD.

The 24-image Digital Green onion case series produced balanced accuracy {dg_ba.estimate:.3f} for the same locked ResNet model and has a wide interval. It remains descriptive. For non-onion OOD images, source-neighbour distance reached AUROC {ood_row.ood_auroc:.3f}, but FPR at 95% OOD sensitivity was {ood_row.fpr_at_95pct_ood_tpr:.3f}; this operating point is not suitable for a safety claim.

The defensible contribution is therefore a leakage-controlled reliability audit showing how high acquisition-day-grouped internal performance fails to transport across public onion datasets. A full plant-disease journal paper still needs pathology review of the broad endpoint and a sufficiently large, prospectively collected or independently re-adjudicated field cohort. Language polishing or a more complex network cannot repair these evidence gaps.

Machine-readable paired comparisons, transport gaps, outer-fold results, and label-level detection rates are in the `tables` directory.
"""
    (RESULTS / "EVIDENCE_GATE.md").write_text(report, encoding="utf-8")
    print(json.dumps(gate, indent=2))


if __name__ == "__main__":
    main()
