"""Diagnose evaluation inflation from augmentation-family leakage in COLD.

This is a separate sensitivity analysis. It does not alter, tune, or extend the
locked TOM2024-to-raw-COLD external result. The comparison holds feature
extractor, classifier, regularization, seed series, and test fraction fixed;
only the split unit changes from files to augmentation families.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize


ROOT = Path(__file__).resolve().parents[1]
LINEAGE = ROOT / "audit" / "cold_augmented_lineage" / "augmented_family_manifest.csv"
EMBEDDINGS = ROOT / "audit" / "cold_augmented_lineage" / "resnet18_embeddings.npz"
RESULTS = ROOT / "results" / "cold_augmented_leakage_ablation_v1"
COLOR_CACHE = ROOT / "features" / "cold_augmented_color_shortcut.npz"
SEED = 20260930
TEST_FRACTION = 0.20
FIXED_C = 0.01


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def one_color_feature(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"cannot read {path}")
    image = cv2.resize(image, (128, 128), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    parts = []
    for array, ranges in ((image, ((0, 256),) * 3), (hsv, ((0, 180), (0, 256), (0, 256))), (lab, ((0, 256),) * 3)):
        for channel, limits in enumerate(ranges):
            histogram = cv2.calcHist([array], [channel], None, [16], list(limits)).ravel()
            histogram = histogram / max(histogram.sum(), 1.0)
            values = array[:, :, channel].astype(np.float32).ravel()
            parts.extend([histogram, np.array([values.mean(), values.std(), *np.quantile(values, [0.1, 0.5, 0.9])])])
    vector = np.concatenate(parts).astype(np.float32)
    if vector.shape != (189,):
        raise RuntimeError(f"unexpected color feature shape {vector.shape}")
    return vector


def load_features(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    archive = np.load(EMBEDDINGS, allow_pickle=False)
    path_to_row = {str(path).replace("\\", "/"): index for index, path in enumerate(archive["paths"].astype(str))}
    missing = [path for path in frame.local_path if path.replace("\\", "/") not in path_to_row]
    if missing:
        raise RuntimeError(f"missing augmented ResNet embeddings: {missing[:3]}")
    rows = [path_to_row[path.replace("\\", "/")] for path in frame.local_path]
    resnet = archive["features"][rows].astype(np.float64)
    if COLOR_CACHE.is_file():
        color_archive = np.load(COLOR_CACHE, allow_pickle=False)
        cached_paths = color_archive["paths"].astype(str).tolist()
        if cached_paths != frame.local_path.astype(str).tolist():
            raise RuntimeError("color feature cache path order differs from lineage manifest")
        color = color_archive["features"].astype(np.float64)
    else:
        color = np.stack([one_color_feature(ROOT / path) for path in frame.local_path])
        # Use a fixed-width Unicode array so the frozen cache remains loadable
        # with allow_pickle=False. Pandas to_numpy() would otherwise preserve an
        # object dtype and make a clean rerun fail at cache loading.
        cache_paths = np.asarray(frame.local_path.astype(str).tolist(), dtype=str)
        np.savez_compressed(COLOR_CACHE, features=color.astype(np.float32), paths=cache_paths)
    return {"color_shortcut": color, "resnet18": resnet}


def file_random_split(y: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=TEST_FRACTION, random_state=seed)
    return next(splitter.split(np.zeros(len(y)), y))


def family_grouped_split(labels: np.ndarray, families: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    family_table = pd.DataFrame({"family": families, "label": labels}).drop_duplicates()
    if family_table.family.duplicated().any():
        raise RuntimeError("one augmentation family has conflicting class labels")
    rng = np.random.default_rng(seed)
    test_families: list[str] = []
    for _, part in family_table.groupby("label"):
        values = part.family.to_numpy()
        count = max(1, int(round(TEST_FRACTION * len(values))))
        test_families.extend(rng.choice(values, size=count, replace=False).tolist())
    test_mask = np.isin(families, test_families)
    train = np.flatnonzero(~test_mask)
    test = np.flatnonzero(test_mask)
    if set(families[train]).intersection(families[test]):
        raise RuntimeError("augmentation-family leakage in grouped split")
    if set(np.unique(labels[train])) != set(np.unique(labels)) or set(np.unique(labels[test])) != set(np.unique(labels)):
        raise RuntimeError("grouped split dropped a class")
    return train, test


def make_model() -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=FIXED_C,
                    penalty="l2",
                    solver="lbfgs",
                    class_weight="balanced",
                    max_iter=10000,
                    random_state=SEED,
                ),
            ),
        ]
    )


def evaluation_metrics(y: np.ndarray, probability: np.ndarray, prediction: np.ndarray, classes: np.ndarray) -> dict[str, float]:
    one_hot = label_binarize(y, classes=classes)
    return {
        "accuracy": float(accuracy_score(y, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "macro_f1": float(f1_score(y, prediction, average="macro", zero_division=0)),
        "macro_ovr_auroc": float(roc_auc_score(one_hot, probability, average="macro", multi_class="ovr")),
        "multiclass_log_loss": float(log_loss(y, np.clip(probability, 1e-9, 1 - 1e-9), labels=classes)),
    }


def family_aggregate_metrics(
    y: np.ndarray, probability: np.ndarray, families: np.ndarray, classes: np.ndarray
) -> dict[str, float]:
    probability_columns = [f"p_{value}" for value in classes]
    table = pd.DataFrame(probability, columns=probability_columns)
    table["family"] = families
    table["truth"] = y
    if (table.groupby("family").truth.nunique() > 1).any():
        raise RuntimeError("family aggregation found conflicting labels")
    aggregate = table.groupby("family", as_index=False).agg(
        {**{column: "mean" for column in probability_columns}, "truth": "first"}
    )
    family_probability = aggregate[probability_columns].to_numpy()
    family_y = aggregate.truth.to_numpy()
    family_prediction = classes[np.argmax(family_probability, axis=1)]
    return {
        f"family_{name}": value
        for name, value in evaluation_metrics(family_y, family_probability, family_prediction, classes).items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repetitions", type=int, default=30)
    args = parser.parse_args()
    if args.repetitions < 10:
        raise ValueError("use at least 10 repeated splits")
    for directory in (RESULTS, RESULTS / "tables", RESULTS / "figures", RESULTS / "split_manifests", RESULTS / "audit"):
        directory.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(LINEAGE)
    required = {"local_path", "label", "family_id", "original_filename", "parent_index"}
    if not required.issubset(frame.columns):
        raise RuntimeError(f"lineage manifest missing columns {sorted(required - set(frame.columns))}")
    if len(frame) != 4502 or frame.family_id.nunique() != 816:
        raise RuntimeError("unexpected augmented lineage counts")
    if frame.local_path.duplicated().any() or frame.family_id.isna().any():
        raise RuntimeError("invalid augmented family manifest")
    encoder = LabelEncoder().fit(frame.label)
    y = encoder.transform(frame.label)
    families = frame.family_id.astype(str).to_numpy()
    classes = np.arange(len(encoder.classes_))
    features = load_features(frame)

    metric_rows: list[dict] = []
    recall_rows: list[dict] = []
    membership_rows: list[pd.DataFrame] = []
    for repetition in range(args.repetitions):
        seed = SEED + repetition
        splits = {
            "ordinary_file_random": file_random_split(y, seed),
            "augmentation_family_grouped": family_grouped_split(y, families, seed),
        }
        for split_name, (train, test) in splits.items():
            train_families = set(families[train])
            test_families = set(families[test])
            family_overlap = train_families.intersection(test_families)
            contaminated_test = np.isin(families[test], list(train_families))
            membership = frame[["row_idx", "local_path", "label", "family_id", "parent_index"]].copy()
            membership["repetition"] = repetition
            membership["seed"] = seed
            membership["split_scheme"] = split_name
            membership["partition"] = "train"
            membership.loc[test, "partition"] = "test"
            membership["family_present_in_other_partition"] = membership.family_id.isin(family_overlap)
            membership_rows.append(membership)
            for feature_name, matrix in features.items():
                estimator = make_model()
                estimator.fit(matrix[train], y[train])
                probability = estimator.predict_proba(matrix[test])
                prediction = estimator.predict(matrix[test])
                metrics = evaluation_metrics(y[test], probability, prediction, classes)
                metrics.update(family_aggregate_metrics(y[test], probability, families[test], classes))
                for metric, value in metrics.items():
                    metric_rows.append(
                        {
                            "repetition": repetition,
                            "seed": seed,
                            "split_scheme": split_name,
                            "feature_model": feature_name,
                            "metric": metric,
                            "value": value,
                            "train_files": len(train),
                            "test_files": len(test),
                            "train_families": len(train_families),
                            "test_families": len(test_families),
                            "overlap_families": len(family_overlap),
                            "test_files_with_family_in_train": int(contaminated_test.sum()),
                            "test_family_contamination_fraction": float(contaminated_test.mean()),
                        }
                    )
                recalls = recall_score(y[test], prediction, labels=classes, average=None, zero_division=0)
                for class_index, recall in zip(classes, recalls):
                    recall_rows.append(
                        {
                            "repetition": repetition,
                            "seed": seed,
                            "split_scheme": split_name,
                            "feature_model": feature_name,
                            "label": encoder.inverse_transform([class_index])[0],
                            "test_files": int(np.sum(y[test] == class_index)),
                            "test_families": int(pd.Series(families[test][y[test] == class_index]).nunique()),
                            "recall": float(recall),
                        }
                    )

    metrics = pd.DataFrame(metric_rows)
    recalls = pd.DataFrame(recall_rows)
    metrics.to_csv(RESULTS / "tables" / "repeated_split_metrics.csv", index=False)
    recalls.to_csv(RESULTS / "tables" / "repeated_per_class_recall.csv", index=False)
    pd.concat(membership_rows, ignore_index=True).to_csv(
        RESULTS / "split_manifests" / "all_repeated_split_membership.csv", index=False
    )

    summary = (
        metrics.groupby(["split_scheme", "feature_model", "metric"], as_index=False)
        .agg(
            mean=("value", "mean"),
            standard_deviation=("value", "std"),
            median=("value", "median"),
            p025=("value", lambda values: np.quantile(values, 0.025)),
            p975=("value", lambda values: np.quantile(values, 0.975)),
            repetitions=("value", "size"),
            mean_test_files=("test_files", "mean"),
            mean_test_families=("test_families", "mean"),
            mean_contamination_fraction=("test_family_contamination_fraction", "mean"),
        )
    )
    summary.to_csv(RESULTS / "tables" / "aggregate_split_summary.csv", index=False)

    pivot = metrics.pivot(index=["repetition", "feature_model", "metric"], columns="split_scheme", values="value").reset_index()
    pivot["file_random_minus_family_grouped"] = (
        pivot["ordinary_file_random"] - pivot["augmentation_family_grouped"]
    )
    paired = (
        pivot.groupby(["feature_model", "metric"], as_index=False)
        .agg(
            mean_inflation=("file_random_minus_family_grouped", "mean"),
            standard_deviation=("file_random_minus_family_grouped", "std"),
            median_inflation=("file_random_minus_family_grouped", "median"),
            p025=("file_random_minus_family_grouped", lambda values: np.quantile(values, 0.025)),
            p975=("file_random_minus_family_grouped", lambda values: np.quantile(values, 0.975)),
            repetitions=("file_random_minus_family_grouped", "size"),
        )
    )
    paired.to_csv(RESULTS / "tables" / "paired_split_inflation.csv", index=False)

    family_counts = (
        frame.groupby("label", as_index=False)
        .agg(files=("local_path", "size"), effective_families=("family_id", "nunique"), min_family_size=("family_id", lambda x: frame.loc[x.index].groupby("family_id").size().min()), max_family_size=("family_id", lambda x: frame.loc[x.index].groupby("family_id").size().max()))
    )
    family_counts["files_per_effective_family"] = family_counts.files / family_counts.effective_families
    family_counts.to_csv(RESULTS / "audit" / "effective_sample_size_by_label.csv", index=False)

    # Paired distribution figure for the two most interpretable metrics.
    figure_data = metrics[metrics.metric.isin(["balanced_accuracy", "family_balanced_accuracy"])].copy()
    models = ["color_shortcut", "resnet18"]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0), constrained_layout=True)
    for axis, metric in zip(axes, ["balanced_accuracy", "family_balanced_accuracy"]):
        positions = []
        data = []
        labels = []
        position = 1
        for model in models:
            for split_name in ("ordinary_file_random", "augmentation_family_grouped"):
                values = figure_data[
                    (figure_data.feature_model == model)
                    & (figure_data.split_scheme == split_name)
                    & (figure_data.metric == metric)
                ].value.to_numpy()
                data.append(values)
                positions.append(position)
                labels.append(f"{model}\n{'file' if split_name.startswith('ordinary') else 'family'}")
                position += 1
            position += 0.5
        axis.boxplot(data, positions=positions, widths=0.65, showfliers=True)
        axis.set_xticks(positions, labels, fontsize=7)
        axis.set_ylim(0, 1.02)
        axis.set_ylabel(metric.replace("_", " ").title())
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("COLD augmented-set split sensitivity (30 fixed repeated seeds)")
    fig.savefig(RESULTS / "figures" / "file_vs_family_split.png", dpi=400)
    fig.savefig(RESULTS / "figures" / "file_vs_family_split.pdf")
    plt.close(fig)

    key = summary[summary.metric.eq("balanced_accuracy")].set_index(["feature_model", "split_scheme"])
    resnet_random = key.loc[("resnet18", "ordinary_file_random"), "mean"]
    resnet_grouped = key.loc[("resnet18", "augmentation_family_grouped"), "mean"]
    color_random = key.loc[("color_shortcut", "ordinary_file_random"), "mean"]
    color_grouped = key.loc[("color_shortcut", "augmentation_family_grouped"), "mean"]
    contamination = metrics[metrics.split_scheme.eq("ordinary_file_random")].test_family_contamination_fraction.mean()
    report = f"""# COLD augmentation-family leakage sensitivity analysis

This analysis is separate from the locked TOM2024-to-raw-COLD external evaluation. It diagnoses how the reported performance of a fixed classifier changes when the unit of splitting is corrected; it does not validate deployment.

The augmented archive contains 4,502 files but only 816 filename-defined augmentation families. The most extreme class is purple blotch: 735 files arise from 18 families. Under ordinary stratified file splitting, a mean {contamination:.1%} of test files had a sibling from the same family in training. Family-grouped splitting enforced zero overlap.

Across {args.repetitions} fixed repeated seeds, the frozen ResNet18 plus fixed-C regularized logistic model had mean file-level balanced accuracy {resnet_random:.3f} with ordinary file splits and {resnet_grouped:.3f} with family-grouped splits. The fixed color-shortcut model changed from {color_random:.3f} to {color_grouped:.3f}. `tables/paired_split_inflation.csv` gives the seed-paired difference distribution, and `tables/repeated_per_class_recall.csv` exposes class-specific effects.

Family IDs are taken directly from the complete `dr_<parent>_<random>.jpg` filename structure. No uncertain raw-image crosswalk is used. The IYSV archive has 282 augmented families versus 281 raw files; this mismatch does not affect within-augmented family separation and is not resolved by forced parent matching.
"""
    (RESULTS / "ABLATION_REPORT.md").write_text(report, encoding="utf-8")
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "split-unit leakage sensitivity only; not external validation",
        "repetitions": args.repetitions,
        "seed_start": SEED,
        "test_fraction": TEST_FRACTION,
        "fixed_logistic_C": FIXED_C,
        "families": int(frame.family_id.nunique()),
        "files": len(frame),
        "lineage_manifest_sha256": sha256_file(LINEAGE),
        "embedding_archive_sha256": sha256_file(EMBEDDINGS),
        "color_feature_cache_sha256": sha256_file(COLOR_CACHE),
        "script_sha256": sha256_file(Path(__file__)),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
    }
    (RESULTS / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"results": str(RESULTS), "resnet_file_random_balanced_accuracy": resnet_random, "resnet_family_grouped_balanced_accuracy": resnet_grouped, "mean_file_split_contamination": contamination}, indent=2))


if __name__ == "__main__":
    main()
