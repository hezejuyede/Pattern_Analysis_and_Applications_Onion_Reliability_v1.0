"""TOM2024 split-unit sensitivity using a fixed frozen ResNet18 baseline.

The analysis compares file-random, raw-photo/acquisition-event-grouped, and
acquisition-day-grouped splits after removing conflict groups and exact image
duplicates. Acquisition day is a batch proxy, not a verified plant identity.
This diagnostic is separate from the locked TOM-to-COLD external test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18


ROOT = Path(__file__).resolve().parents[1]
SOURCE_MEMBERS = ROOT / "audit" / "tom2024_category_a_onion" / "source_groups" / "source_group_members.csv"
IMAGE_INVENTORY = ROOT / "audit" / "tom2024_category_a_onion" / "image_inventory.csv"
FEATURE_CACHE = ROOT / "features" / "tom_all_foliar_nonconflict_exactdedup_resnet18.npz"
RESULTS = ROOT / "results" / "tom_split_severity_sensitivity_v1"
SEED = 20260930
TEST_FRACTION = 0.20
FIXED_C = 0.01
ELIGIBLE_LABELS = {"Healthy_leaf", "Alternaria_D", "Fusarium-D", "Virosis-D"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Images(Dataset):
    def __init__(self, paths: list[Path], transform):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            return self.transform(image.convert("RGB"))


def load_or_extract_features(frame: pd.DataFrame, batch_size: int) -> np.ndarray:
    expected_paths = frame.analysis_path.astype(str).tolist()
    if FEATURE_CACHE.is_file():
        archive = np.load(FEATURE_CACHE, allow_pickle=False)
        if archive["paths"].astype(str).tolist() != expected_paths:
            raise RuntimeError("TOM feature cache does not match analysis manifest")
        return archive["features"].astype(np.float64)
    weights = ResNet18_Weights.DEFAULT
    base = resnet18(weights=weights)
    model = torch.nn.Sequential(*list(base.children())[:-1], torch.nn.Flatten(1)).eval()
    dataset = Images([ROOT / path for path in expected_paths], weights.transforms())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    pieces = []
    torch.set_num_threads(max(1, (os.cpu_count() or 2) - 2))
    with torch.inference_mode():
        for images in loader:
            pieces.append(model(images).cpu().numpy().astype(np.float32))
    features = np.concatenate(pieces)
    if features.shape != (len(frame), 512):
        raise RuntimeError(f"unexpected feature shape {features.shape}")
    np.savez_compressed(FEATURE_CACHE, features=features, paths=np.asarray(expected_paths))
    return features.astype(np.float64)


def build_manifest() -> tuple[pd.DataFrame, pd.DataFrame]:
    groups = pd.read_csv(SOURCE_MEMBERS)
    inventory = pd.read_csv(IMAGE_INVENTORY)[["relative_path", "sha256"]]
    frame = groups.merge(inventory, on="relative_path", how="left", validate="one_to_one")
    if frame.sha256.isna().any():
        raise RuntimeError("missing TOM hashes after inventory join")
    frame = frame[(frame.label_count == 1) & frame.label.isin(ELIGIBLE_LABELS)].copy()
    frame["analysis_path"] = "data/tom2024_category_a/CATA-English/" + frame.relative_path.astype(str)
    frame["acquisition_day_utc"] = pd.to_datetime(frame.utc_datetime, format="mixed", utc=True).dt.strftime("%Y-%m-%d")
    frame["foliar_binary"] = frame.label.ne("Healthy_leaf").astype(int)
    duplicates = frame[frame.sha256.duplicated(keep="first")].copy()
    frame = frame.drop_duplicates("sha256", keep="first").reset_index(drop=True)
    if frame.sha256.duplicated().any():
        raise RuntimeError("exact duplicates remain")
    for path in frame.analysis_path:
        if not (ROOT / path).is_file():
            raise RuntimeError(f"missing TOM image {path}")
    return frame, duplicates


def file_split(y: np.ndarray, seed: int):
    return next(StratifiedShuffleSplit(n_splits=1, test_size=TEST_FRACTION, random_state=seed).split(np.zeros(len(y)), y))


def source_group_split(y: np.ndarray, source_groups: np.ndarray, seed: int):
    family = pd.DataFrame({"group": source_groups, "y": y}).drop_duplicates()
    if family.group.duplicated().any():
        raise RuntimeError("conflicting source group remains")
    rng = np.random.default_rng(seed)
    chosen = []
    for _, part in family.groupby("y"):
        values = part.group.to_numpy()
        count = max(1, int(round(TEST_FRACTION * len(values))))
        chosen.extend(rng.choice(values, size=count, replace=False).tolist())
    mask = np.isin(source_groups, chosen)
    return np.flatnonzero(~mask), np.flatnonzero(mask)


def day_group_split(y: np.ndarray, day_groups: np.ndarray, seed: int):
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    candidates = []
    prevalence = y.mean()
    for train, test in splitter.split(np.zeros(len(y)), y, day_groups):
        objective = abs(len(test) / len(y) - TEST_FRACTION) + abs(y[test].mean() - prevalence)
        candidates.append((objective, train, test))
    _, train, test = min(candidates, key=lambda item: item[0])
    return train, test


def make_model():
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=FIXED_C,
                    penalty="l2",
                    solver="liblinear",
                    class_weight="balanced",
                    max_iter=10000,
                    random_state=SEED,
                ),
            ),
        ]
    )


def metrics(y: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    prediction = probability >= 0.5
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "macro_f1": float(f1_score(y, prediction, average="macro")),
        "auroc": float(roc_auc_score(y, probability)),
        "brier": float(brier_score_loss(y, probability)),
    }


def source_aggregate(y: np.ndarray, probability: np.ndarray, groups: np.ndarray) -> dict[str, float]:
    table = pd.DataFrame({"y": y, "probability": probability, "group": groups})
    if (table.groupby("group").y.nunique() > 1).any():
        raise RuntimeError("source group contains mixed labels")
    aggregate = table.groupby("group", as_index=False).agg(y=("y", "first"), probability=("probability", "mean"))
    return {f"source_group_{name}": value for name, value in metrics(aggregate.y.to_numpy(), aggregate.probability.to_numpy()).items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repetitions", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.repetitions < 10:
        raise ValueError("use at least 10 repetitions")
    for directory in (RESULTS, RESULTS / "tables", RESULTS / "figures", RESULTS / "audit", RESULTS / "split_manifests"):
        directory.mkdir(parents=True, exist_ok=True)
    frame, removed_duplicates = build_manifest()
    frame.to_csv(RESULTS / "audit" / "analysis_manifest.csv", index=False)
    removed_duplicates.to_csv(RESULTS / "audit" / "removed_exact_duplicates.csv", index=False)
    features = load_or_extract_features(frame, args.batch_size)
    y = frame.foliar_binary.to_numpy()
    source_groups = frame.source_group.astype(str).to_numpy()
    day_groups = frame.acquisition_day_utc.astype(str).to_numpy()

    result_rows = []
    membership_rows = []
    for repetition in range(args.repetitions):
        seed = SEED + repetition
        splits = {
            "naive_file_random": file_split(y, seed),
            "raw_photo_grouped": source_group_split(y, source_groups, seed),
            "acquisition_day_grouped": day_group_split(y, day_groups, seed),
        }
        for scheme, (train, test) in splits.items():
            source_overlap = set(source_groups[train]).intersection(source_groups[test])
            day_overlap = set(day_groups[train]).intersection(day_groups[test])
            source_contaminated = np.isin(source_groups[test], list(set(source_groups[train])))
            day_contaminated = np.isin(day_groups[test], list(set(day_groups[train])))
            if scheme == "raw_photo_grouped" and source_overlap:
                raise RuntimeError("raw-photo group leakage")
            if scheme == "acquisition_day_grouped" and (source_overlap or day_overlap):
                raise RuntimeError("day-group leakage")
            model = make_model().fit(features[train], y[train])
            probability = model.predict_proba(features[test])[:, 1]
            values = metrics(y[test], probability)
            values.update(source_aggregate(y[test], probability, source_groups[test]))
            for metric, value in values.items():
                result_rows.append(
                    {
                        "repetition": repetition,
                        "seed": seed,
                        "split_scheme": scheme,
                        "metric": metric,
                        "value": value,
                        "train_files": len(train),
                        "test_files": len(test),
                        "train_source_groups": len(set(source_groups[train])),
                        "test_source_groups": len(set(source_groups[test])),
                        "source_group_overlap": len(source_overlap),
                        "test_source_contamination_fraction": float(source_contaminated.mean()),
                        "train_days": len(set(day_groups[train])),
                        "test_days": len(set(day_groups[test])),
                        "day_overlap": len(day_overlap),
                        "test_day_contamination_fraction": float(day_contaminated.mean()),
                    }
                )
            membership = frame[["analysis_path", "sha256", "source_group", "acquisition_day_utc", "label", "foliar_binary"]].copy()
            membership["repetition"] = repetition
            membership["seed"] = seed
            membership["split_scheme"] = scheme
            membership["partition"] = "train"
            membership.loc[test, "partition"] = "test"
            membership_rows.append(membership)

    results = pd.DataFrame(result_rows)
    results.to_csv(RESULTS / "tables" / "repeated_split_metrics.csv", index=False)
    pd.concat(membership_rows, ignore_index=True).to_csv(
        RESULTS / "split_manifests" / "all_repeated_split_membership.csv", index=False
    )
    summary = (
        results.groupby(["split_scheme", "metric"], as_index=False)
        .agg(
            mean=("value", "mean"),
            standard_deviation=("value", "std"),
            median=("value", "median"),
            p025=("value", lambda x: np.quantile(x, 0.025)),
            p975=("value", lambda x: np.quantile(x, 0.975)),
            repetitions=("value", "size"),
            mean_test_files=("test_files", "mean"),
            mean_source_contamination=("test_source_contamination_fraction", "mean"),
            mean_day_contamination=("test_day_contamination_fraction", "mean"),
        )
    )
    summary.to_csv(RESULTS / "tables" / "aggregate_split_summary.csv", index=False)

    fig, axis = plt.subplots(figsize=(7.5, 4.2), constrained_layout=True)
    schemes = ["naive_file_random", "raw_photo_grouped", "acquisition_day_grouped"]
    values = [
        results[(results.split_scheme == scheme) & (results.metric == "balanced_accuracy")].value.to_numpy()
        for scheme in schemes
    ]
    axis.boxplot(values, labels=["File random", "Raw-photo group", "Acquisition day"])
    axis.set_ylim(0, 1.0)
    axis.set_ylabel("Balanced accuracy")
    axis.grid(axis="y", alpha=0.25)
    axis.set_title("TOM2024 split-unit sensitivity: fixed frozen ResNet18 + logistic")
    fig.savefig(RESULTS / "figures" / "split_severity_balanced_accuracy.png", dpi=400)
    fig.savefig(RESULTS / "figures" / "split_severity_balanced_accuracy.pdf")
    plt.close(fig)

    key = summary[summary.metric.eq("balanced_accuracy")].set_index("split_scheme")
    report = f"""# TOM2024 split-severity sensitivity

This diagnostic uses all {len(frame):,} nonconflict, exact-deduplicated foliar files from {frame.source_group.nunique():,} timestamp/raw-photo groups and {frame.acquisition_day_utc.nunique()} UTC acquisition days. It uses a fixed frozen ResNet18 feature extractor, fixed-C regularized logistic regression, and fold-local standardization. No result from COLD is used.

Across {args.repetitions} fixed repeated seeds, mean balanced accuracy was {key.loc['naive_file_random', 'mean']:.3f} for naive file splits, {key.loc['raw_photo_grouped', 'mean']:.3f} for raw-photo-group splits, and {key.loc['acquisition_day_grouped', 'mean']:.3f} for acquisition-day-group splits. Mean test source-group contamination and acquisition-day contamination are reported beside every metric in `tables/aggregate_split_summary.csv`.

Exact duplicate bytes were collapsed before every scheme, so the contrast concerns residual acquisition-event and batch dependence rather than identical-file leakage. A filename timestamp is treated as an acquisition event, and UTC day as a coarser acquisition-batch proxy. Neither is a verified plant identity; therefore even the day-grouped result may retain plant- or site-level dependence.
"""
    (RESULTS / "SENSITIVITY_REPORT.md").write_text(report, encoding="utf-8")
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "TOM split-unit sensitivity; separate from locked external evaluation",
        "files_after_exact_deduplication": len(frame),
        "exact_duplicate_files_removed": len(removed_duplicates),
        "source_groups": int(frame.source_group.nunique()),
        "acquisition_days": int(frame.acquisition_day_utc.nunique()),
        "repetitions": args.repetitions,
        "seed_start": SEED,
        "fixed_logistic_C": FIXED_C,
        "source_manifest_sha256": sha256_file(SOURCE_MEMBERS),
        "inventory_sha256": sha256_file(IMAGE_INVENTORY),
        "feature_cache_sha256": sha256_file(FEATURE_CACHE),
        "script_sha256": sha256_file(Path(__file__)),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": __import__("torchvision").__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
    }
    (RESULTS / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"results": str(RESULTS), "balanced_accuracy": key["mean"].to_dict()}, indent=2))


if __name__ == "__main__":
    main()
