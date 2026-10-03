"""Post hoc within-archive regional robustness, with a frozen source-only protocol.

No plant identity or pathological confirmation is inferred from archive metadata.
Selected-C OOF scores train calibration; they are not unbiased internal validation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
REGIONS = ["Centre-Ouest", "Centre-Sud", "Plateau-Central"]
VARIANTS = ["region_day_disjoint", "region_only"]
MODELS = ["prior_prevalence", "color_shortcut_logit", "handcrafted_logit", "resnet18_logit"]
C_GRID = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
SEED = 20261002
INNER_SEED = 20261839
METRICS = ["balanced_accuracy", "roc_auc", "brier", "sensitivity", "specificity"]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def pipeline(c=1.0):
    return make_pipeline(StandardScaler(), LogisticRegression(C=c, solver="liblinear", class_weight="balanced", random_state=20260930, max_iter=10000))


def load_inputs(args):
    df = pd.read_csv(args.metadata, keep_default_na=False)
    df["foliar_binary"] = df.foliar_binary.astype(float).astype(int)
    assert len(df) == 1643 and not df.sample_id.duplicated().any()
    known = df.recovered_region.isin(REGIONS)
    assert known.sum() == 1416 and (~known).sum() == 227
    assert df.loc[known, "region_match_status"].eq("unique_index_match").all()
    assert set(df.foliar_binary) == {0, 1}
    for key in ["source_group", "analysis_sha256"]:
        assert not df[key].duplicated().any(), key
    matrices = {}
    for name in ["handcrafted", "resnet18"]:
        with np.load(args.features_dir / (name + ".npz"), allow_pickle=False) as a:
            ids = a["sample_ids"].astype(str)
            assert len(set(ids)) == len(ids)
            index = {s: i for i, s in enumerate(ids)}
            matrices[name] = a["features"][[index[s] for s in df.sample_id]].astype(np.float64)
            assert np.isfinite(matrices[name]).all()
    assert matrices["handcrafted"].shape[1] == 1957 and matrices["resnet18"].shape[1] == 512
    matrices["color_shortcut"] = matrices["handcrafted"][:, :189]
    return df, matrices


def freeze(args, df):
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    if (out / "protocol_freeze.json").exists():
        raise RuntimeError("Freeze already exists; use --mode run to reuse it.")
    memberships, inner_rows, support = [], [], []
    for variant in VARIANTS:
        for region_index, region in enumerate(REGIONS):
            test = df.recovered_region.eq(region)
            known = df.recovered_region.isin(REGIONS)
            candidate = known & ~test
            same_day = df.acquisition_day_utc.isin(df.loc[test, "acquisition_day_utc"])
            purged = candidate & same_day if variant == "region_day_disjoint" else pd.Series(False, index=df.index)
            train = candidate & ~purged
            role = np.where(~known, "unknown_excluded", np.where(test, "test", np.where(train, "train", "purged")))
            split = df.copy()
            split.insert(0, "variant", variant)
            split.insert(1, "region", region)
            split["role"] = role
            memberships.append(split)
            for key in ["sample_id", "source_group", "analysis_sha256", "recovered_region"]:
                assert not set(df.loc[train, key]) & set(df.loc[test, key]), (variant, region, key)
            if variant == "region_day_disjoint":
                assert not set(df.loc[train, "acquisition_day_utc"]) & set(df.loc[test, "acquisition_day_utc"])
            tr = df.loc[train].reset_index(drop=True)
            folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=INNER_SEED + region_index).split(np.zeros(len(tr)), tr.foliar_binary, tr.acquisition_day_utc))
            for fold, (fit, val) in enumerate(folds):
                assert set(tr.iloc[fit].foliar_binary) == set(tr.iloc[val].foliar_binary) == {0, 1}
                assert not set(tr.iloc[fit].acquisition_day_utc) & set(tr.iloc[val].acquisition_day_utc)
                for i in val:
                    inner_rows.append(dict(variant=variant, region=region, sample_id=tr.iloc[i].sample_id, calibration_fold=fold, acquisition_day_utc=tr.iloc[i].acquisition_day_utc))
            for (split_role, label), group in split.groupby(["role", "original_label"], sort=True):
                support.append(dict(variant=variant, region=region, role=split_role, original_label=label, n=len(group), days=group.acquisition_day_utc.nunique()))
    pd.concat(memberships, ignore_index=True).to_csv(out / "outer_memberships.csv", index=False)
    pd.DataFrame(inner_rows).to_csv(out / "inner_fold_assignments.csv", index=False)
    pd.DataFrame(support).to_csv(out / "class_support.csv", index=False)
    df.to_csv(out / "analysis_metadata.csv", index=False)
    inputs = {"metadata": str(args.metadata.resolve()), "handcrafted_features": str((args.features_dir / "handcrafted.npz").resolve()), "resnet18_features": str((args.features_dir / "resnet18.npz").resolve()), "implementation": str(Path(__file__).resolve())}
    protocol = {
        "schema": "regional_robustness_v1", "created_utc": datetime.now(timezone.utc).isoformat(),
        "overall_status": "post_hoc_within_existing_archive_not_new_independent_collection",
        "freeze_timing": "Before fitting these regional models; original archive outcomes were already known.",
        "regions": REGIONS, "variants": VARIANTS, "models": MODELS, "C_grid": C_GRID,
        "inner_folds": 5, "inner_seed_base": INNER_SEED, "fit_seed": 20260930, "bootstrap_seed": SEED,
        "bootstrap_replicates": 2000, "primary": "region_day_disjoint per-region BA, equal-region mean BA, minimum-region BA",
        "calibration": "Selected-C source 5-fold OOF scores fit Platt C=1e6 and finite Youden threshold; not unbiased source validation.",
        "prior": "Training prevalence probability and fixed 0.5 decision threshold; no fitted calibrator.",
        "bootstrap": "Per-region UTC-day cluster resampling; aggregate joint resampling of globally unique UTC days retains cross-region dependence; reject replicates lacking either state in any represented region. Conditional on fitted models and these three regions.",
        "sensitivity_limit": "Region-only and day-purged training differ in sample size/composition; their difference is not a causal leakage effect.",
        "identity_limit": "Filename timestamp UTC day and recovered archive region are not independently verified collection dates, farms or plant identities. Source folder names are not confirmed diagnoses.",
        "features": "Previously frozen ImageNet features / fixed handcrafted features; no target adaptation.",
        "inputs": {k: {"path": v, "sha256": sha(v)} for k, v in inputs.items()},
        "frozen_files": {name: sha(out / name) for name in ["outer_memberships.csv", "inner_fold_assignments.csv", "class_support.csv", "analysis_metadata.csv"]},
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__},
    }
    save_json(out / "protocol_freeze.json", protocol)
    print("FROZEN", out / "protocol_freeze.json", flush=True)


def verify_freeze(args):
    p = json.loads((args.output_dir / "protocol_freeze.json").read_text(encoding="utf-8"))
    paths = {"metadata": args.metadata, "handcrafted_features": args.features_dir / "handcrafted.npz", "resnet18_features": args.features_dir / "resnet18.npz", "implementation": Path(__file__)}
    for key, path in paths.items():
        assert sha(path) == p["inputs"][key]["sha256"], key + " hash changed"
    for name, digest in p["frozen_files"].items():
        assert sha(args.output_dir / name) == digest, name + " hash changed"
    return p


def point_metrics(y, p, pred):
    return dict(balanced_accuracy=float(balanced_accuracy_score(y, pred)), roc_auc=float(roc_auc_score(y, p)), brier=float(brier_score_loss(y, p)), sensitivity=float(pred[y == 1].mean()), specificity=float((pred[y == 0] == 0).mean()))


def fit_and_predict(args, df, matrices):
    out = args.output_dir
    verify_freeze(args)
    if (out / "predictions.csv").exists():
        raise RuntimeError("Predictions already exist; do not overwrite. Use --mode bootstrap.")
    membership = pd.read_csv(out / "outer_memberships.csv", keep_default_na=False)
    inner = pd.read_csv(out / "inner_fold_assignments.csv")
    index = {s: i for i, s in enumerate(df.sample_id)}
    all_predictions, all_metrics, all_oof, all_cv, all_recall = [], [], [], [], []
    (out / "models").mkdir(exist_ok=True)
    for variant in VARIANTS:
        for region in REGIONS:
            split = membership[(membership.variant == variant) & (membership.region == region)]
            tr = split[split.role == "train"].copy().reset_index(drop=True)
            te = split[split.role == "test"].copy().reset_index(drop=True)
            tr_pos = np.array([index[s] for s in tr.sample_id])
            te_pos = np.array([index[s] for s in te.sample_id])
            y = tr.foliar_binary.to_numpy(dtype=int)
            yt = te.foliar_binary.to_numpy(dtype=int)
            assignments = inner[(inner.variant == variant) & (inner.region == region)].set_index("sample_id").loc[tr.sample_id, "calibration_fold"].to_numpy()
            folds = [(np.flatnonzero(assignments != f), np.flatnonzero(assignments == f)) for f in range(5)]
            for model in MODELS:
                estimator = calibrator = None
                selected_c = None
                source_prior = float(y.mean())
                if model == "prior_prevalence":
                    threshold = 0.5
                    score = np.full(len(te), np.log(source_prior / (1 - source_prior)))
                    probability = np.full(len(te), source_prior)
                else:
                    feature = model.removesuffix("_logit")
                    x = matrices[feature][tr_pos]
                    xt = matrices[feature][te_pos]
                    grid = GridSearchCV(pipeline(), {"logisticregression__C": C_GRID}, scoring={"balanced_accuracy": "balanced_accuracy", "roc_auc": "roc_auc"}, refit="balanced_accuracy", cv=folds, n_jobs=args.n_jobs, error_score="raise", return_train_score=False)
                    with joblib.parallel_backend("loky", inner_max_num_threads=1):
                        grid.fit(x, y)
                    estimator = grid.best_estimator_
                    selected_c = float(grid.best_params_["logisticregression__C"])
                    table = pd.DataFrame(grid.cv_results_)
                    keep = [c for c in table if c == "param_logisticregression__C" or ("test_" in c and not c.startswith("rank_"))]
                    table = table[keep].copy()
                    table.insert(0, "model", model); table.insert(0, "region", region); table.insert(0, "variant", variant)
                    table["selected"] = table.param_logisticregression__C.astype(float).eq(selected_c)
                    all_cv.append(table)
                    oof = np.full(len(y), np.nan)
                    for fit, val in folds:
                        head = pipeline(selected_c).fit(x[fit], y[fit])
                        oof[val] = head.decision_function(x[val])
                    assert np.isfinite(oof).all()
                    calibrator = LogisticRegression(C=1e6, solver="lbfgs", random_state=20260930, max_iter=10000).fit(oof[:, None], y)
                    poof = calibrator.predict_proba(oof[:, None])[:, 1]
                    fpr, tpr, thresholds = roc_curve(y, poof)
                    finite = np.flatnonzero(np.isfinite(thresholds))
                    threshold = float(thresholds[finite[np.argmax((tpr - fpr)[finite])]])
                    score = estimator.decision_function(xt)
                    probability = calibrator.predict_proba(score[:, None])[:, 1]
                    calibration = tr[["sample_id", "acquisition_day_utc", "original_label"]].copy()
                    calibration.insert(0, "model", model); calibration.insert(0, "region", region); calibration.insert(0, "variant", variant)
                    calibration["y_true"] = y; calibration["calibration_fold"] = assignments
                    calibration["score"] = oof; calibration["probability"] = poof
                    calibration["threshold"] = threshold; calibration["selected_c"] = selected_c
                    all_oof.append(calibration)
                prediction = (probability >= threshold).astype(int)
                assert np.isfinite(probability).all()
                artifact = dict(schema="regional_head_v1", variant=variant, region=region, model=model, estimator=estimator, calibrator=calibrator, threshold=threshold, selected_c=selected_c, source_prior=source_prior, train_ids=tr.sample_id.tolist(), test_ids=te.sample_id.tolist(), protocol_sha256=sha(out / "protocol_freeze.json"))
                joblib.dump(artifact, out / "models" / f"{variant}__{region}__{model}.joblib", compress=3)
                pred = te[["sample_id", "acquisition_day_utc", "original_label"]].copy()
                pred.insert(0, "model", model); pred.insert(0, "region", region); pred.insert(0, "variant", variant)
                pred["y_true"] = yt; pred["score"] = score; pred["probability"] = probability
                pred["prediction"] = prediction; pred["threshold"] = threshold
                all_predictions.append(pred)
                values = point_metrics(yt, probability, prediction)
                all_metrics.append(dict(variant=variant, region=region, model=model, n=len(te), train_n=len(tr), train_days=tr.acquisition_day_utc.nunique(), test_days=te.acquisition_day_utc.nunique(), selected_c=selected_c, threshold=threshold, train_prevalence=source_prior, test_prevalence=float(yt.mean()), **values))
                for label, sub in pred.groupby("original_label"):
                    all_recall.append(dict(variant=variant, region=region, model=model, original_label=label, n=len(sub), days=sub.acquisition_day_utc.nunique(), recall=float(sub.prediction.eq(sub.y_true).mean())))
                print(f"FIT {variant} / {region} / {model}: BA={values['balanced_accuracy']:.6f}; C={selected_c}", flush=True)
                pd.concat(all_predictions, ignore_index=True).to_csv(out / "predictions_partial.csv", index=False)
    pd.concat(all_predictions, ignore_index=True).to_csv(out / "predictions.csv", index=False)
    pd.DataFrame(all_metrics).to_csv(out / "metrics.csv", index=False)
    pd.concat(all_oof, ignore_index=True).to_csv(out / "source_calibration_oof.csv", index=False)
    pd.concat(all_cv, ignore_index=True).to_csv(out / "cv_selection.csv", index=False)
    pd.DataFrame(all_recall).to_csv(out / "source_label_recall.csv", index=False)


def bootstrap_counts(days, ysets, rng, B=2000):
    """Joint day weights; each (positions, y) must retain both states."""
    accepted = []
    attempts = 0
    while sum(len(a) for a in accepted) < B:
        batch = rng.multinomial(len(days), np.full(len(days), 1.0 / len(days)), size=min(B, 500))
        valid = np.ones(len(batch), dtype=bool)
        for positions, y in ysets:
            valid &= batch[:, positions[y == 0]].sum(axis=1) > 0
            valid &= batch[:, positions[y == 1]].sum(axis=1) > 0
        accepted.append(batch[valid])
        attempts += len(batch)
        if attempts > B * 40:
            raise RuntimeError("Bootstrap class-preserving resampling exhausted")
    return np.concatenate(accepted)[:B].astype(np.uint16), attempts


def weighted_metrics(frame, weights):
    y = frame.y_true.to_numpy(dtype=int)
    p = frame.probability.to_numpy(dtype=float)
    pred = frame.prediction.to_numpy(dtype=int)
    pos = weights[:, y == 1].sum(axis=1)
    neg = weights[:, y == 0].sum(axis=1)
    sens = (weights * ((y == 1) & (pred == 1))).sum(axis=1) / pos
    spec = (weights * ((y == 0) & (pred == 0))).sum(axis=1) / neg
    brier = (weights * np.square(p - y)).sum(axis=1) / (pos + neg)
    order = np.argsort(p, kind="stable")
    unique_starts = np.r_[0, np.flatnonzero(np.diff(p[order])) + 1]
    weighted_positive = np.add.reduceat(weights[:, order] * y[order], unique_starts, axis=1)
    weighted_negative = np.add.reduceat(weights[:, order] * (1 - y[order]), unique_starts, axis=1)
    below_negative = np.cumsum(weighted_negative, axis=1) - 0.5 * weighted_negative
    auc = (weighted_positive * below_negative).sum(axis=1) / (pos * neg)
    return dict(balanced_accuracy=(sens + spec) / 2, roc_auc=auc, brier=brier, sensitivity=sens, specificity=spec)


def bootstrap(args):
    out = args.output_dir
    verify_freeze(args)
    predictions = pd.read_csv(out / "predictions.csv")
    metrics = pd.read_csv(out / "metrics.csv")
    strict = predictions[predictions.variant == "region_day_disjoint"]
    interval_rows, contrast_rows, bootstrap_states, count_archives, day_orders = [], [], {}, {}, {}
    point = metrics[metrics.variant == "region_day_disjoint"].set_index(["region", "model"])
    primary_model = "resnet18_logit"
    def add_interval(region, model, metric, estimate, values):
        lo, hi = np.quantile(values, [0.025, 0.975])
        interval_rows.append(dict(variant="region_day_disjoint", region=region, model=model, metric=metric, estimate=float(estimate), lower=float(lo), upper=float(hi)))
    def add_contrasts(region, states, estimates):
        for other in MODELS[:-1]:
            for metric in METRICS:
                values = states[primary_model][metric] - states[other][metric]
                lo, hi = np.quantile(values, [0.025, 0.975])
                contrast_rows.append(dict(region=region, comparator=other, metric=metric, estimate=float(estimates[primary_model][metric] - estimates[other][metric]), lower=float(lo), upper=float(hi)))
    for ri, region in enumerate(REGIONS):
        reference = strict[(strict.region == region) & (strict.model == primary_model)].sort_values("sample_id").reset_index(drop=True)
        days = sorted(reference.acquisition_day_utc.unique())
        lookup = {d: j for j, d in enumerate(days)}
        positions = np.array([lookup[d] for d in reference.acquisition_day_utc])
        counts, attempts = bootstrap_counts(days, [(positions, reference.y_true.to_numpy())], np.random.default_rng(SEED + ri))
        weights = counts[:, positions].astype(float)
        count_archives["region_" + str(ri)] = counts
        day_orders[region] = dict(days=days, seed=SEED + ri, attempts=attempts)
        states = {}
        for model in MODELS:
            frame = strict[(strict.region == region) & (strict.model == model)].sort_values("sample_id").reset_index(drop=True)
            assert frame.sample_id.equals(reference.sample_id) and frame.y_true.equals(reference.y_true)
            states[model] = weighted_metrics(frame, weights)
            for metric in METRICS:
                add_interval(region, model, metric, point.loc[(region, model), metric], states[model][metric])
        add_contrasts(region, states, {m: point.loc[(region, m), METRICS].to_dict() for m in MODELS})
    reference = strict[strict.model == primary_model].sort_values("sample_id").reset_index(drop=True)
    global_days = sorted(reference.acquisition_day_utc.unique())
    lookup = {d: j for j, d in enumerate(global_days)}
    region_frames = {r: reference[reference.region == r].copy().reset_index(drop=True) for r in REGIONS}
    pos = {r: np.array([lookup[d] for d in region_frames[r].acquisition_day_utc]) for r in REGIONS}
    counts, attempts = bootstrap_counts(global_days, [(pos[r], region_frames[r].y_true.to_numpy()) for r in REGIONS], np.random.default_rng(SEED + 100))
    count_archives["joint_global"] = counts
    day_orders["joint_global"] = dict(days=global_days, seed=SEED + 100, attempts=attempts)
    joint = {m: {} for m in MODELS}
    for model in MODELS:
        by_region = []
        for region in REGIONS:
            frame = strict[(strict.region == region) & (strict.model == model)].sort_values("sample_id").reset_index(drop=True)
            assert frame.sample_id.equals(region_frames[region].sample_id)
            by_region.append(weighted_metrics(frame, counts[:, pos[region]].astype(float)))
        for metric in METRICS:
            stacked = np.stack([v[metric] for v in by_region])
            joint[model][metric] = stacked.mean(axis=0)
            add_interval("equal_region_mean", model, metric, point.xs(model, level="model")[metric].mean(), stacked.mean(axis=0))
            add_interval("minimum_region", model, metric, point.xs(model, level="model")[metric].min(), stacked.min(axis=0))
    add_contrasts("equal_region_mean", joint, {m: point.xs(m, level="model")[METRICS].mean().to_dict() for m in MODELS})
    pd.DataFrame(interval_rows).to_csv(out / "intervals.csv", index=False)
    pd.DataFrame(contrast_rows).to_csv(out / "paired_contrasts.csv", index=False)
    np.savez_compressed(out / "bootstrap_day_counts.npz", **count_archives)
    save_json(out / "bootstrap_day_orders.json", day_orders)
    aggregates = []
    for (variant, model), group in metrics.groupby(["variant", "model"]):
        for aggregation in ["equal_region_mean", "minimum_region"]:
            aggregates.append(dict(variant=variant, region=aggregation, model=model, **{metric: float(group[metric].mean() if aggregation == "equal_region_mean" else group[metric].min()) for metric in METRICS}))
    pd.DataFrame(aggregates).to_csv(out / "aggregate_metrics.csv", index=False)
    files = sorted(p for p in out.rglob("*") if p.is_file() and p.name not in ["completion_manifest.json", "predictions_partial.csv"])
    save_json(out / "completion_manifest.json", dict(status="COMPLETE", completed_utc=datetime.now(timezone.utc).isoformat(), protocol_sha256=sha(out / "protocol_freeze.json"), model_cases=24, learned_models=18, predictions=len(predictions), bootstrap_replicates=2000, files={str(p.relative_to(out)).replace("\\", "/"): sha(p) for p in files}))
    print("COMPLETE", out, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=ROOT / "metadata" / "tom_region_recovery" / "analysis_image_region_join.csv")
    parser.add_argument("--features-dir", type=Path, default=ROOT / "features")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "regional_robustness_v1")
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--mode", choices=["freeze", "run", "bootstrap", "all"], default="all")
    args = parser.parse_args()
    if not 1 <= args.n_jobs <= 4:
        parser.error("--n-jobs must be 1..4")
    if args.mode in ["freeze", "all", "run"]:
        df, matrices = load_inputs(args)
    if args.mode in ["freeze", "all"]:
        freeze(args, df)
    if args.mode in ["run", "all"]:
        fit_and_predict(args, df, matrices)
    if args.mode in ["bootstrap", "all"]:
        bootstrap(args)


if __name__ == "__main__":
    main()
