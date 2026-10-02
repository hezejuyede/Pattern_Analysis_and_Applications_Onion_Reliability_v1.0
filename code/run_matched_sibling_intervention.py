"""Post hoc fixed-budget sibling-substitution diagnostic; never external validation.

Freeze the protocol and input hashes before fitting. All reported seeds, feature
sets, endpoints and contamination doses are retained, irrespective of outcome.
"""
from __future__ import annotations

import hashlib
import json
import platform
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "matched_sibling_intervention_v1"
INPUTS = {
    "lineage": ROOT / "audit/cold_augmented_lineage/augmented_family_manifest.csv",
    "resnet18": ROOT / "audit/cold_augmented_lineage/resnet18_embeddings.npz",
    "color_shortcut": ROOT / "features/cold_augmented_color_shortcut.npz",
}
SEED_START = 20261002
REPETITIONS = 30
DOSES = ("0", "0.5", "1")
CLASS_NAMES = ("healthy", "iris_yellow_virus", "purple_blotch", "stemphylium_colletotrichum_leaf_blight")
METRICS = ("balanced_accuracy", "accuracy", "macro_f1", "log_loss", "brier", "auroc")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def freeze_protocol() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "design.json").exists():
        raise RuntimeError("Refusing to overwrite a frozen design/result; use a new version for changes.")
    design = {
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_status": "post_hoc_diagnostic_designed_after_existing_results_were_known",
        "purpose": "Estimate the fixed-budget same-class sibling-substitution effect within one augmented archive.",
        "not_estimated": ["prospective generalization", "new independent external validation", "biological causal effect", "plant-level replication"],
        "repetitions": REPETITIONS,
        "seed_start": SEED_START,
        "seed_rule": "20261002 + repetition, repetition = 0,...,29; retain every seed",
        "test_family_fraction_by_class": 0.2,
        "test_family_count_rule": "max(1, Python round(0.2 * n_families_in_class))",
        "test_families_by_class": dict(zip(CLASS_NAMES, (85, 56, 4, 18))),
        "sentinel_family_fraction_by_class": 0.1,
        "sentinel_families_by_class": dict(zip(CLASS_NAMES, (43, 28, 2, 9))),
        "test_images": 245,
        "probe_test_images": 163,
        "sentinel_test_images": 82,
        "test_view_selection": "one random anchor and two other distinct sibling views per test family, shared across all conditions and endpoints",
        "sentinel_selection": "After selecting probe P families, select round(0.1 * original_class_family_count) sentinel S families from the remaining class; one anchor each, all S families excluded from all training conditions.",
        "clean_training": "Every remaining D family contributes two random distinct views; 571 families, 1142 images.",
        "donor_matching": "Each probe P family is matched without replacement to one random D donor family from the same class; donor family contributes its two selected views.",
        "doses": list(DOSES),
        "dose_rule": "Within each class, randomize matched pair order; replace round(dose * n_test_families) donor pairs by the corresponding two test-family siblings. Record actual exposure rate.",
        "controls": ["identical test anchor images", "identical training image budget", "identical per-class image counts", "identical training family count", "exactly two training views per included family", "identical feature extractors", "identical logistic hyperparameters", "training-only standardization"],
        "models": {"resnet18": "frozen ImageNet 512 dimensions", "color_shortcut": "fixed color histogram/statistic 189 dimensions"},
        "classifier": {"name": "LogisticRegression", "C": 0.01, "regularization": "L2 library default", "solver": "lbfgs", "class_weight": "balanced", "max_iter": 10000, "random_state": SEED_START},
        "endpoints": {"four_class": "primary; original four archive classes", "binary_healthy_affected": "secondary; healthy=0, all three affected archive classes=1; no pathology equivalence claim"},
        "primary_metric": "four-class balanced accuracy; full exposure minus zero exposure separately in P probes and S sentinels; localization contrast = delta_P minus delta_S, paired within seed and feature set",
        "secondary_metrics": list(METRICS),
        "brier_definition": "four-class sum_k (p_k - onehot_k)^2 without dividing by classes; binary (p_affected - y)^2",
        "uncertainty": "Across-seed SD and 2.5/97.5 percentiles describe split variation, not independent biological confidence intervals; no seed-level significance tests.",
        "interpretation": "The intervention replaces same-class unrelated source families with related families. It controls quantity and class composition, but intentionally changes training identity and dependence. Its magnitude is conditional on this two-view budget and archive.",
        "prohibited": ["raw-parent inferred crosswalk", "target-label hyperparameter tuning", "seed selection", "changing primary model after inspecting results", "representing the diagnostic as preregistered before the original study"],
        "input_sha256": {name: digest(path) for name, path in INPUTS.items()},
        "script_sha256": digest(Path(__file__)),
        "software": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__, "scikit_learn": sklearn.__version__},
        "execution": "CPU only, one numerical library thread; no cv2, torch, image decoding, new features, or new image data",
    }
    write_json(OUT / "design.json", design)
    (OUT / "DESIGN.md").write_text(
        "# Fixed-budget matched sibling substitution\n\n"
        f"Protocol frozen before this experiment was run: {design['frozen_utc']}.\n\n"
        "This is an explicitly **post hoc** diagnostic designed after the original split-sensitivity and external results were known. It is not a new confirmatory external evaluation.\n\n"
        "For each of 30 seeds (20261002–20261031), select 20% of each class's families as probe P families (85 healthy, 56 virus, 4 purple blotch, 18 blight). Draw one probe anchor and two distinct siblings from each. Then select 10% of the original class's families as sentinel S families from the remainder (43, 28, 2, 9; 82 total); draw one anchor each and exclude all sentinel-family views from every training condition. Every remaining D family contributes two training views. Pair each probe family without replacement with a random same-class D donor family. At exposure doses 0, 0.5 and 1, replace the chosen donor's two training views with the paired probe family's siblings. All 245 P/S test anchors, training images (1142), training families (571), class counts and two views per training family are fixed across conditions. Half exposure uses integer rounding and reports the realized proportion.\n\n"
        "Use the existing frozen ResNet18 and color features, training-only standardization and class-balanced L2 logistic regression with C=0.01. Primary endpoint: original four-class archive labels. Secondary endpoint: healthy versus any affected label with exactly the same memberships. Report all seeds and outcomes. Compare full minus zero exposure in four-class balanced accuracy separately in probes and sentinels; the paired localization contrast is delta_P minus delta_S. Seed percentiles are split-variation ranges, not confidence intervals for biological replication.\n\n"
        "Same-class substitution controls training quantity and label composition. It intentionally changes source identity and dependence, so the effect is conditional on the specified archive and two-view training budget. Filename families are source-image groups, not identified plants. No inferred raw-image crosswalk is used. No external model, calibration or threshold is altered.\n\n"
        "Machine-readable input and script hashes, endpoint and model settings are frozen in `design.json`.\n",
        encoding="utf-8",
    )
    return design


def load_data() -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    frame = pd.read_csv(INPUTS["lineage"]).reset_index(drop=True)
    assert len(frame) == 4502 and frame.family_id.nunique() == 816
    assert not frame.local_path.duplicated().any() and not frame.row_idx.duplicated().any()
    assert not frame.sha256.duplicated().any()
    assert frame.groupby("family_id").label.nunique().max() == 1
    assert frame.groupby("family_id").size().min() >= 3
    assert set(frame.label) == set(CLASS_NAMES)
    matrices = {}
    wanted = [str(x).replace("\\", "/") for x in frame.local_path]
    for name in ("resnet18", "color_shortcut"):
        archive = np.load(INPUTS[name], allow_pickle=False)
        paths = [str(x).replace("\\", "/") for x in archive["paths"]]
        assert len(paths) == len(set(paths))
        positions = {p: i for i, p in enumerate(paths)}
        matrix = np.asarray(archive["features"][[positions[p] for p in wanted]], dtype=np.float64)
        assert np.isfinite(matrix).all()
        assert matrix.shape == (4502, 512 if name == "resnet18" else 189)
        matrices[name] = matrix
    return frame, matrices


def make_memberships(frame: pd.DataFrame, repetition: int) -> tuple[np.ndarray, dict[str, np.ndarray], list[dict], list[dict]]:
    rng = np.random.default_rng(SEED_START + repetition)
    family_rows = {str(f): np.asarray(g.index, dtype=int) for f, g in frame.groupby("family_id", sort=True)}
    family_labels = frame.groupby("family_id", sort=True).label.first()
    records = {}
    test_rows = []
    for label in CLASS_NAMES:
        ids = family_labels.index[family_labels.eq(label)].to_numpy(dtype=str)
        order = rng.permutation(ids)
        n_test = max(1, round(0.2 * len(ids)))
        n_sentinel = max(1, round(0.1 * len(ids)))
        tests = order[:n_test]
        sentinels = set(order[n_test:n_test + n_sentinel])
        donors = order[n_test + n_sentinel:2 * n_test + n_sentinel]
        active_half = set(rng.permutation(tests)[:round(0.5 * n_test)])
        pairs = dict(zip(tests, donors))
        inverse = {d: t for t, d in pairs.items()}
        for family in ids:
            is_test = family in pairs
            is_sentinel = family in sentinels
            drawn = rng.choice(family_rows[family], 3 if is_test else (1 if is_sentinel else 2), replace=False)
            anchor = int(drawn[0]) if is_test or is_sentinel else -1
            views = drawn[1:] if is_test else (np.array([-1, -1]) if is_sentinel else drawn)
            if is_test or is_sentinel:
                test_rows.append(anchor)
            partner = pairs.get(family, inverse.get(family, ""))
            role = "probe" if is_test else ("sentinel" if is_sentinel else ("donor" if family in inverse else "base"))
            treatment_test = family if is_test else inverse.get(family, "")
            half_swapped = treatment_test in active_half
            records[family] = {
                "repetition": repetition, "seed": SEED_START + repetition,
                "family_id": family, "original_label": label, "role": role,
                "paired_family": partner,
                "test_anchor_row_idx": int(frame.at[anchor, "row_idx"]) if is_test or is_sentinel else -1,
                "view1_row_idx": int(frame.at[int(views[0]), "row_idx"]) if not is_sentinel else -1,
                "view2_row_idx": int(frame.at[int(views[1]), "row_idx"]) if not is_sentinel else -1,
                "train_0": int(not is_test and not is_sentinel),
                "train_0.5": int(not is_sentinel and (half_swapped if is_test else not (role == "donor" and half_swapped))),
                "train_1": int(role not in ("donor", "sentinel")),
                "_views": np.asarray(views, dtype=int),
            }
    test = np.sort(np.asarray(test_rows, dtype=int))
    assert len(test) == 245
    panel = np.array([records[f]["role"] for f in frame.iloc[test].family_id])
    assert np.sum(panel == "probe") == 163 and np.sum(panel == "sentinel") == 82
    train_by_dose, checks = {}, []
    clean_counts = None
    for dose in DOSES:
        train = np.sort(np.concatenate([x["_views"] for x in records.values() if x[f"train_{dose}"]]))
        assert len(train) == len(set(train)) == 1142
        assert len(np.intersect1d(train, test)) == 0
        assert not set(frame.iloc[train].sha256).intersection(frame.iloc[test].sha256)
        train_families = frame.iloc[train].family_id
        assert train_families.nunique() == 571
        assert train_families.value_counts().eq(2).all()
        counts = frame.iloc[train].label.value_counts().reindex(CLASS_NAMES).to_numpy()
        if clean_counts is None:
            clean_counts = counts.copy()
        assert np.array_equal(counts, clean_counts)
        contaminated = frame.iloc[test].family_id.isin(set(train_families))
        expected = sum(round(float(dose) * n) for n in (85, 56, 4, 18))
        assert int(contaminated.sum()) == expected
        assert not contaminated.to_numpy()[panel == "sentinel"].any()
        assert np.array_equal(counts, [596, 396, 24, 126])
        train_by_dose[dose] = train
        checks.append({
            "repetition": repetition, "dose": dose, "train_images": len(train),
            "train_families": train_families.nunique(), "test_anchors": len(test),
            "test_file_overlap": 0, "test_sha_overlap": 0,
            "train_views_per_family": 2,
            "train_class_counts": ";".join(f"{x}:{n}" for x, n in zip(CLASS_NAMES, counts)),
            "test_anchor_sha256": hashlib.sha256("\n".join(frame.iloc[test].sha256).encode()).hexdigest(),
            "probe_anchor_sha256": hashlib.sha256("\n".join(frame.iloc[test[panel == "probe"]].sha256).encode()).hexdigest(),
            "sentinel_anchor_sha256": hashlib.sha256("\n".join(frame.iloc[test[panel == "sentinel"]].sha256).encode()).hexdigest(),
            "probe_test_families": 163, "sentinel_test_families": 82, "exposed_sentinel_families": 0,
            "exposed_test_families": int(contaminated.sum()),
            "actual_exposure_fraction": int(contaminated.sum()) / 163, "status": "PASS",
        })
    return test, train_by_dose, [{k: v for k, v in x.items() if k != "_views"} for x in records.values()], checks


def fit_predict(x: np.ndarray, y: np.ndarray, train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, float, list[dict], int]:
    model = Pipeline([
        ("scale", StandardScaler()),
        ("classifier", LogisticRegression(C=0.01, solver="lbfgs", class_weight="balanced", max_iter=10000, random_state=SEED_START)),
    ])
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(x[train], y[train])
        probability = model.predict_proba(x[test])
    warning_records = [{"category": x.category.__name__, "message": str(x.message)} for x in caught]
    if any(issubclass(x.category, ConvergenceWarning) for x in caught):
        raise RuntimeError("Convergence warning: frozen analysis halted without changing its model settings.")
    assert model.named_steps["scale"].n_samples_seen_ == len(train)
    assert np.array_equal(model.named_steps["classifier"].classes_, np.unique(y))
    assert np.isfinite(probability).all() and np.allclose(probability.sum(axis=1), 1)
    return probability, time.perf_counter() - started, warning_records, int(np.max(model.named_steps["classifier"].n_iter_))


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    pred = np.argmax(p, axis=1)
    binary = p.shape[1] == 2
    truth = np.eye(p.shape[1])[y]
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "log_loss": float(log_loss(y, p, labels=np.arange(p.shape[1]))),
        "brier": float(np.mean((p[:, 1] - y) ** 2) if binary else np.mean(np.sum((p - truth) ** 2, axis=1))),
        "auroc": float(roc_auc_score(y, p[:, 1]) if binary else roc_auc_score(y, p, multi_class="ovr", average="macro")),
    }


def main() -> None:
    started = time.perf_counter()
    design = freeze_protocol()
    frame, matrices = load_data()
    y_four = frame.label.map(dict(zip(CLASS_NAMES, range(4)))).to_numpy(dtype=int)
    endpoints = {"four_class": y_four, "binary_healthy_affected": (y_four != 0).astype(int)}
    metric_rows, pred_rows, recall_rows, member_rows, check_rows, warning_rows = [], [], [], [], [], []
    fits = 0
    with threadpool_limits(limits=1):
        for repetition in range(REPETITIONS):
            test, train_by_dose, members, checks = make_memberships(frame, repetition)
            member_rows.extend(members)
            check_rows.extend(checks)
            roles = {m["family_id"]: m["role"] for m in members}
            panels = np.asarray([roles[f] for f in frame.iloc[test].family_id])
            for dose, train in train_by_dose.items():
                actual = next(x["actual_exposure_fraction"] for x in checks if x["dose"] == dose)
                for feature_name, x in matrices.items():
                    for endpoint, y in endpoints.items():
                        p, seconds, caught, n_iter = fit_predict(x, y, train, test)
                        fits += 1
                        key = {"repetition": repetition, "seed": SEED_START + repetition, "dose": dose, "feature_model": feature_name, "endpoint": endpoint}
                        warning_rows.extend([{**key, **w} for w in caught])
                        predicted = np.argmax(p, axis=1)
                        for evaluation_panel in ("probe", "sentinel"):
                            selected = panels == evaluation_panel
                            metric_rows.append({**key, "evaluation_panel": evaluation_panel, "actual_exposure_fraction": actual if evaluation_panel == "probe" else 0.0, **metrics(y[test][selected], p[selected]), "fit_predict_seconds": seconds, "n_iter": n_iter})
                            for value in np.unique(y):
                                mask = (y[test] == value) & selected
                                recall_rows.append({**key, "evaluation_panel": evaluation_panel, "class": int(value), "class_name": CLASS_NAMES[value] if endpoint == "four_class" else ("healthy" if value == 0 else "affected"), "test_families": int(mask.sum()), "recall": float(np.mean(predicted[mask] == value))})
                        for local, pos in enumerate(test):
                            pred_rows.append({**key, "evaluation_panel": panels[local], "row_idx": int(frame.at[pos, "row_idx"]), "family_id": frame.at[pos, "family_id"], "original_label": frame.at[pos, "label"], "truth": int(y[pos]), "predicted": int(predicted[local]), **{f"p{k}": float(p[local, k]) for k in range(p.shape[1])}})
            print(f"Completed repetition {repetition + 1}/{REPETITIONS}; fits={fits}; elapsed={time.perf_counter()-started:.1f}s", flush=True)

    table = pd.DataFrame(metric_rows)
    table.to_csv(OUT / "repeated_metrics.csv", index=False, float_format="%.12g")
    pd.DataFrame(pred_rows).to_csv(OUT / "predictions.csv.gz", index=False, float_format="%.12g", compression={"method": "gzip", "mtime": 0})
    pd.DataFrame(member_rows).to_csv(OUT / "family_membership.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    pd.DataFrame(recall_rows).to_csv(OUT / "class_recall.csv", index=False, float_format="%.12g")
    pd.DataFrame(check_rows).to_csv(OUT / "execution_checks.csv", index=False)
    write_json(OUT / "warnings.json", {"warning_count": len(warning_rows), "warnings": warning_rows})
    summaries, pairs, pair_summaries = [], [], []
    for (feature, endpoint, dose, panel), part in table.groupby(["feature_model", "endpoint", "dose", "evaluation_panel"], sort=True):
        for name in METRICS:
            v = part[name].to_numpy()
            summaries.append({"feature_model": feature, "endpoint": endpoint, "dose": dose, "evaluation_panel": panel, "metric": name, "mean": float(v.mean()), "split_sd": float(v.std(ddof=1)), "split_p025": float(np.quantile(v, 0.025)), "split_p975": float(np.quantile(v, 0.975)), "repetitions": len(v)})
    for (feature, endpoint, panel), part in table.groupby(["feature_model", "endpoint", "evaluation_panel"], sort=True):
        clean = part[part.dose.eq("0")].set_index("repetition")
        for dose in ("0.5", "1"):
            contaminated = part[part.dose.eq(dose)].set_index("repetition").loc[clean.index]
            for name in METRICS:
                differences = contaminated[name] - clean[name]
                for rep, value in differences.items():
                    pairs.append({"feature_model": feature, "endpoint": endpoint, "evaluation_panel": panel, "dose": dose, "repetition": rep, "metric": name, "exposed_minus_clean": float(value)})
                pair_summaries.append({"feature_model": feature, "endpoint": endpoint, "evaluation_panel": panel, "dose": dose, "metric": name, "mean_exposed_minus_clean": float(differences.mean()), "split_sd": float(differences.std(ddof=1)), "split_p025": float(differences.quantile(.025)), "split_p975": float(differences.quantile(.975)), "positive_seeds": int((differences > 0).sum()), "negative_seeds": int((differences < 0).sum()), "zero_seeds": int((differences == 0).sum()), "repetitions": len(differences)})
    pd.DataFrame(summaries).to_csv(OUT / "aggregate_metrics.csv", index=False, float_format="%.12g")
    pd.DataFrame(pairs).to_csv(OUT / "paired_differences.csv", index=False, float_format="%.12g")
    summary = pd.DataFrame(pair_summaries)
    summary.to_csv(OUT / "paired_summary.csv", index=False, float_format="%.12g")
    pair_table = pd.DataFrame(pairs)
    local = pair_table.pivot(index=["feature_model", "endpoint", "dose", "repetition", "metric"], columns="evaluation_panel", values="exposed_minus_clean").reset_index()
    local["localization_delta_probe_minus_delta_sentinel"] = local.probe - local.sentinel
    local.to_csv(OUT / "localization_contrasts.csv", index=False, float_format="%.12g")
    local_summaries = []
    for (feature, endpoint, dose, metric), part in local.groupby(["feature_model", "endpoint", "dose", "metric"]):
        v = part.localization_delta_probe_minus_delta_sentinel
        local_summaries.append({"feature_model": feature, "endpoint": endpoint, "dose": dose, "metric": metric, "mean_localization_contrast": float(v.mean()), "split_sd": float(v.std(ddof=1)), "split_p025": float(v.quantile(.025)), "split_p975": float(v.quantile(.975)), "positive_seeds": int((v > 0).sum()), "negative_seeds": int((v < 0).sum()), "zero_seeds": int((v == 0).sum()), "repetitions": len(v)})
    local_summary = pd.DataFrame(local_summaries)
    local_summary.to_csv(OUT / "localization_summary.csv", index=False, float_format="%.12g")
    checks = pd.DataFrame(check_rows)
    assert fits == 360 and len(pred_rows) == 88200 and len(member_rows) == 24480
    assert checks.groupby("repetition").test_anchor_sha256.nunique().eq(1).all()
    assert all(digest(INPUTS[k]) == v for k, v in design["input_sha256"].items())
    assert digest(Path(__file__)) == design["script_sha256"]
    elapsed = time.perf_counter() - started
    report = ["# Matched sibling-substitution diagnostic", "", "**Status: completed; explicitly post hoc.** Every prespecified seed, exposure dose, feature model and endpoint is retained. This diagnostic does not alter the original locked external evaluation.", "", "All 90 membership checks passed. Each condition has 163 identical probe anchors and 82 identical sentinel anchors, 1142 training images, 571 training families, exactly two views per training family and identical class counts. There is zero exact-file or SHA overlap between training and test. Realized probe-family exposure fractions are 0, 81/163 (0.49693), and 1; sentinel exposure remains zero.", "", "| Endpoint | Features | Panel | Clean BA | Full-exposure BA | Paired difference | Split 2.5–97.5% range |", "|---|---|---|---:|---:|---:|---|" ]
    for feature in matrices:
        for endpoint in endpoints:
            for panel in ("probe", "sentinel"):
                part = table[(table.feature_model == feature) & (table.endpoint == endpoint) & (table.evaluation_panel == panel)]
                clean = part[part.dose.eq("0")].balanced_accuracy.mean()
                full = part[part.dose.eq("1")].balanced_accuracy.mean()
                s = summary[(summary.feature_model == feature) & (summary.endpoint == endpoint) & summary.evaluation_panel.eq(panel) & summary.dose.eq("1") & summary.metric.eq("balanced_accuracy")].iloc[0]
                report.append(f"| {endpoint} | {feature} | {panel} | {clean:.6f} | {full:.6f} | {s.mean_exposed_minus_clean:+.6f} | {s.split_p025:+.6f} to {s.split_p975:+.6f} |")
    report += ["", "| Endpoint | Features | Localization contrast delta_P minus delta_S | Split 2.5–97.5% range |", "|---|---|---:|---|"]
    for _, s in local_summary[(local_summary.dose == "1") & (local_summary.metric == "balanced_accuracy")].iterrows():
        report.append(f"| {s.endpoint} | {s.feature_model} | {s.mean_localization_contrast:+.6f} | {s.split_p025:+.6f} to {s.split_p975:+.6f} |")
    report += ["", "The percentile ranges describe variation across overlapping repeated splits, **not** confidence intervals based on independent biological observations. Four purple-blotch probe families and two sentinel families per seed limit class-specific precision. Related-family replacement changes source identity and dependence while controlling quantity and class composition; it does not establish a universal effect size or validate disease diagnosis. Results from the two-view budget must not replace or be merged numerically with the original all-view file-versus-family contrast. All lineage keys are archive filename families; the inferred raw-parent crosswalk was not used.", "", f"CPU runtime: {elapsed:.2f} s; 360 fits; {len(warning_rows)} warnings; no convergence warnings. scikit-learn {sklearn.__version__}; no new image decoding or feature extraction.", ""]
    (OUT / "RESULTS.md").write_text("\n".join(report), encoding="utf-8")
    write_json(OUT / "run_metadata.json", {"status": "PASS_POST_HOC_DIAGNOSTIC", "completed_utc": datetime.now(timezone.utc).isoformat(), "elapsed_seconds": elapsed, "fits": fits, "prediction_rows": len(pred_rows), "membership_family_rows": len(member_rows), "membership_checks": len(check_rows), "warnings": len(warning_rows), "convergence_warnings": 0, "design_sha256": digest(OUT / "design.json"), "script_sha256": digest(Path(__file__)), "software": design["software"], "input_sha256_unchanged": True, "original_results_untouched": True})
    files = [p for p in sorted(OUT.iterdir()) if p.is_file() and p.name != "manifest.csv"]
    pd.DataFrame([{"path": p.name, "bytes": p.stat().st_size, "sha256": digest(p)} for p in files]).to_csv(OUT / "manifest.csv", index=False)
    print("\n".join(report), flush=True)


if __name__ == "__main__":
    main()
