"""Build publication/deployment gates and freeze final evidence hashes."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PRIMARY = ROOT / "results" / "reliability_benchmark_v1"
ABLATION = ROOT / "results" / "cold_augmented_leakage_ablation_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metric(frame: pd.DataFrame, analysis_set: str, model: str, name: str) -> pd.Series:
    rows = frame[(frame.analysis_set == analysis_set) & (frame.model == model) & (frame.metric == name)]
    if len(rows) != 1:
        raise RuntimeError(f"missing/duplicate metric {analysis_set}, {model}, {name}")
    return rows.iloc[0]


def main() -> None:
    tables = PRIMARY / "tables"
    audit = PRIMARY / "audit"
    metrics = pd.read_csv(tables / "binary_metrics_group_bootstrap.csv")
    transport = pd.read_csv(tables / "internal_external_transport_gap.csv")
    paired = pd.read_csv(tables / "cold_paired_model_differences.csv")
    ood = pd.read_csv(tables / "open_set_ood_metrics.csv")
    compute = pd.read_csv(tables / "backbone_compute_metadata.csv")
    compute["timing_status"] = "measured_during_frozen_feature_extraction"
    compute["timing_note"] = "CPU wall time measured by extract_challenger_features.py"
    resnet_mask = compute.model.eq("resnet18_logit")
    compute.loc[resnet_mask, "timing_status"] = "not_recorded_for_legacy_primary_extraction"
    compute.loc[resnet_mask, "timing_note"] = (
        "Blank timing fields are missing, not zero: ResNet18 features were extracted before timing instrumentation. "
        "Parameter count and GOP/image use torchvision official weight metadata."
    )
    compute.to_csv(tables / "backbone_compute_metadata.csv", index=False)

    primary_model = "resnet18_logit"
    primary_ba = metric(metrics, "cold_locked_external", primary_model, "balanced_accuracy")
    primary_auc = metric(metrics, "cold_locked_external", primary_model, "auroc")
    primary_sensitivity = metric(metrics, "cold_locked_external", primary_model, "sensitivity")
    primary_brier = metric(metrics, "cold_locked_external", primary_model, "brier")
    primary_ece = metric(metrics, "cold_locked_external", primary_model, "ece_10")
    dg_ba = metric(metrics, "digigreen_onion_case_series", primary_model, "balanced_accuracy")
    transport_primary = transport[
        (transport.model == primary_model) & (transport.metric == "balanced_accuracy")
    ].iloc[0]
    primary_ood = ood[
        (ood.known_set == "cold_locked_external")
        & (ood.model == primary_model)
        & (ood.detector == "source_5nn_cosine")
    ].iloc[0]

    ablation_summary = pd.read_csv(ABLATION / "tables" / "aggregate_split_summary.csv")
    ablation_paired = pd.read_csv(ABLATION / "tables" / "paired_split_inflation.csv")
    def ablation_value(scheme: str, feature: str, metric_name: str) -> float:
        return float(
            ablation_summary[
                (ablation_summary.split_scheme == scheme)
                & (ablation_summary.feature_model == feature)
                & (ablation_summary.metric == metric_name)
            ].iloc[0]["mean"]
        )

    challenger_models = ["efficientnet_b0_logit", "convnext_tiny_logit", "swin_t_logit"]
    challenger_evidence = {}
    for model in challenger_models:
        challenger_evidence[model] = {
            name: float(metric(metrics, "cold_locked_external", model, name).estimate)
            for name in ("balanced_accuracy", "auroc", "brier", "ece_10", "sensitivity", "specificity")
        }

    gate = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "primary_model": primary_model,
        "model_selection_rule": (
            "ResNet18 remains the sole a-priori primary. All three challengers are secondary sensitivities and are "
            "reported regardless of COLD performance; no external ranking changes the claim."
        ),
        "deployment_claim_gate": "FAIL",
        "publication_reliability_science_gate": "PASS_WITH_EXPLICIT_LIMITS",
        "submission_package_gate": "PASS_SCIENTIFIC_EVIDENCE_WITH_EXPLICIT_LIMITS",
        "criteria": {
            "pattern_analysis_scope": "PASS_FOR_RELIABILITY_BENCHMARK_FRAMING",
            "substantive_original_contribution": (
                "PASS_WITH_CAVEAT: augmentation-family inflation, acquisition-day grouping, cross-country transport, "
                "calibration, selective risk, and OOD are integrated; no new architecture is claimed."
            ),
            "independent_validation": (
                "PASS_WITH_LIMITATIONS: COLD is country/dataset independent and excluded from all fits, but it is public, "
                "lacks acquisition-day/plant IDs, and had been opened for the primary before challenger expansion."
            ),
            "strong_baselines": (
                "PASS: prevalence, color shortcut, handcrafted, ResNet18 primary, EfficientNet-B0, ConvNeXt-Tiny, and "
                "Swin-T all use the same TOM-only nested protocol."
            ),
            "leakage_prevention": (
                "PASS_WITH_METADATA_LIMITATION: exact duplicates removed; TOM inner/outer splits grouped by acquisition "
                "day; augmented COLD assessed separately by 816 filename families; plant identity remains unavailable."
            ),
            "data_conclusion_match": (
                "PASS only for a negative reliability/transport conclusion; FAIL for accurate field diagnosis, "
                "pathogen-specific transfer, safety, or deployment claims."
            ),
            "reproducibility": (
                "PASS_INTERNALLY: frozen features, split manifests, portable fitted artifacts, exact per-image predictions, "
                "hashes, and commands are present; independent audit is still required for final submission release."
            ),
        },
        "primary_locked_external_evidence": {
            "balanced_accuracy": float(primary_ba.estimate),
            "balanced_accuracy_95_group_bootstrap": [float(primary_ba.ci_low), float(primary_ba.ci_high)],
            "auroc": float(primary_auc.estimate),
            "sensitivity": float(primary_sensitivity.estimate),
            "brier": float(primary_brier.estimate),
            "ece_10_point_only": float(primary_ece.estimate),
            "TOM_to_COLD_balanced_accuracy_change": float(transport_primary.external_minus_internal),
            "DigitalGreen_case_series_balanced_accuracy_n24": float(dg_ba.estimate),
            "non_onion_5NN_OOD_AUROC": float(primary_ood.ood_auroc),
            "non_onion_5NN_OOD_FPR95": float(primary_ood.fpr_at_95pct_ood_tpr),
        },
        "challenger_secondary_sensitivity_evidence": challenger_evidence,
        "augmentation_family_leakage_evidence": {
            "files": 4502,
            "families": 816,
            "purple_blotch_files": 735,
            "purple_blotch_families": 18,
            "file_split_test_sibling_contamination_fraction": float(
                ablation_summary[ablation_summary.split_scheme == "ordinary_file_random"].mean_contamination_fraction.mean()
            ),
            "resnet_file_split_balanced_accuracy": ablation_value(
                "ordinary_file_random", "resnet18", "balanced_accuracy"
            ),
            "resnet_family_split_balanced_accuracy": ablation_value(
                "augmentation_family_grouped", "resnet18", "balanced_accuracy"
            ),
        },
        "deployment_failures_preserved": [
            "Primary disorder sensitivity is below 0.50 at the source-locked threshold.",
            "The primary internal-to-external balanced-accuracy loss is large.",
            "The 24-image operational onion set is descriptive and near chance at the locked threshold.",
            "OOD FPR95 is too high for a safety/deployment claim.",
            "There is no prospective cohort, plant-level identity, or independent phytopathologist re-adjudication."
        ],
        "publication_release_conditions": [
            "Manuscript title, abstract, and conclusions must state reliability audit/transport failure rather than deployment.",
            "All challenger models must remain secondary; COLD cannot select a replacement primary.",
            "COLD broad binary labels must not be rewritten as pathogen-confirmed ground truth.",
            "Complete target-journal format, citation, figure, equation, author, and portal-metadata QA.",
            "Independent rerun/hash audit must pass after this evidence freeze."
        ],
    }
    (audit / "release_gate.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")
    (audit / "publication_evidence_gate.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")

    report = f"""# Publication evidence gate and deployment boundary

**Publication reliability science gate: PASS WITH EXPLICIT LIMITS.**  
**Manuscript-evidence gate: PASS WITH EXPLICIT LIMITS.**  
**Deployment, safety, and pathogen-specific diagnostic gates: FAIL.**

The a-priori primary ResNet18 model reached locked COLD balanced accuracy {primary_ba.estimate:.3f} (95% exact-SHA-group bootstrap interval {primary_ba.ci_low:.3f}–{primary_ba.ci_high:.3f}), AUROC {primary_auc.estimate:.3f}, and sensitivity {primary_sensitivity.estimate:.3f}. Its balanced accuracy changed by {transport_primary.external_minus_internal:.3f} from acquisition-day nested TOM validation to COLD. This supports a cross-dataset reliability-failure paper; it does not support field deployment.

EfficientNet-B0, ConvNeXt-Tiny, and Swin-T were listed before their features or scores were computed. They use identical TOM-only nested fitting, fold-local preprocessing, cross-fitted internal calibration/thresholds, and a final TOM-only calibrator/threshold for COLD. Every challenger is retained as a secondary sensitivity. COLD had already been opened for the primary ResNet analysis, so these additions are post-primary sensitivity evidence rather than a second pristine confirmation.

The separate augmented-COLD experiment shows the methodological contribution directly: 4,502 files represent only 816 families, and 735 purple-blotch images represent 18 families. Ordinary file splitting placed augmentation siblings across partitions for nearly every test file and inflated fixed ResNet balanced accuracy relative to family-grouped splitting. These results diagnose evaluation inflation and do not validate deployment.

The manuscript may proceed only with the negative reliability framing and the limitations in `audit/publication_evidence_gate.json`. ECE is reported as a point estimate without percentile confidence bounds because nonlinear clustered bootstrap intervals were not used for that metric.
"""
    (PRIMARY / "PUBLICATION_EVIDENCE_GATE.md").write_text(report, encoding="utf-8")

    # Freeze all primary and leakage-ablation evidence plus the inputs required
    # to reproduce their fitted-feature analyses. The manifest excludes itself.
    targets: list[tuple[str, Path]] = []
    for path in sorted(PRIMARY.rglob("*")):
        if path.is_file() and path.name not in {"evidence_freeze_manifest.csv", "evidence_freeze_summary.json"}:
            targets.append(("primary_result", path))
    for path in sorted(ABLATION.rglob("*")):
        if path.is_file():
            targets.append(("augmentation_ablation_result", path))
    extra = [
        ROOT / "code" / "run_reliability_benchmark.py",
        ROOT / "code" / "analyze_reliability_results.py",
        ROOT / "code" / "extract_challenger_features.py",
        ROOT / "code" / "run_augmented_family_leakage_ablation.py",
        ROOT / "code" / "build_publication_gate_and_freeze.py",
        ROOT / "manifests" / "unified_manifest.csv",
        ROOT / "audit" / "cold_augmented_lineage" / "augmented_family_manifest.csv",
        ROOT / "audit" / "cold_augmented_lineage" / "resnet18_embeddings.npz",
        ROOT / "features" / "cold_augmented_color_shortcut.npz",
        ROOT / "features" / "handcrafted.npz",
        ROOT / "features" / "resnet18.npz",
        ROOT / "features" / "efficientnet_b0.npz",
        ROOT / "features" / "convnext_tiny.npz",
        ROOT / "features" / "swin_t.npz",
    ]
    for path in extra:
        targets.append(("code_or_frozen_input", path))
    unique: dict[str, tuple[str, Path]] = {}
    for category, path in targets:
        unique[str(path.resolve())] = (category, path)
    rows = []
    for category, path in unique.values():
        if not path.is_file():
            raise RuntimeError(f"freeze target missing: {path}")
        rows.append(
            {
                "category": category,
                "path_relative_to_onion_root": path.resolve().relative_to(ROOT.resolve()).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    manifest = pd.DataFrame(rows).sort_values(["category", "path_relative_to_onion_root"])
    manifest_path = audit / "evidence_freeze_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "freeze_status": "PUBLIC_REPOSITORY_NORMALIZED_EVIDENCE_FREEZE",
        "files_hashed_excluding_manifest_itself": len(manifest),
        "total_bytes_hashed": int(manifest.bytes.sum()),
        "manifest_sha256": sha256_file(manifest_path),
        "primary_run_command": (
            "python code/run_reliability_benchmark.py --bootstrap 1000 --n-jobs 4"
        ),
        "derived_analysis_command": "python code/analyze_reliability_results.py",
        "augmentation_ablation_command": (
            "python code/run_augmented_family_leakage_ablation.py --repetitions 30"
        ),
        "primary_script_sha256": sha256_file(ROOT / "code" / "run_reliability_benchmark.py"),
        "analysis_script_sha256": sha256_file(ROOT / "code" / "analyze_reliability_results.py"),
        "augmentation_ablation_script_sha256": sha256_file(
            ROOT / "code" / "run_augmented_family_leakage_ablation.py"
        ),
        "publication_gate": gate["publication_reliability_science_gate"],
        "submission_package_gate": gate["submission_package_gate"],
        "deployment_gate": gate["deployment_claim_gate"],
    }
    (audit / "evidence_freeze_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
