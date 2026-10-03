"""Post hoc correction using reviewed isolation components and supported views.

Default is preparation only. --execute requires a hash-bound independent review.
Components block all known connected content; intervention_source_id identifies
the one supported common reference used for the selected views. Neither is a
plant identity. No image is regrouped or silently dropped by this program.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import platform
import time
import warnings
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


@contextmanager
def csv_writer(path, fields):
    """Deterministic gzip header; CSV values retain full float precision."""
    path = Path(path)
    if path.suffix == ".gz":
        with path.open("wb") as raw:
            with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as gz:
                with io.TextIOWrapper(gz, encoding="utf-8", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=fields)
                    writer.writeheader()
                    yield writer
    else:
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            yield writer


def path_key(value):
    return str(value).replace("\\", "/")


def load_inputs(args):
    frame = pd.read_csv(args.manifest, dtype=str, keep_default_na=False)
    required = {"row_idx", "label", "local_path", "sha256", "component_id"}
    require(required.issubset(frame.columns), f"Missing manifest columns: {required-set(frame.columns)}")
    frame = frame.reset_index(drop=True)
    for col in required:
        require(frame[col].str.len().gt(0).all(), f"Empty {col}")
    for col in ("row_idx", "local_path", "sha256"):
        require(not frame[col].duplicated().any(), f"Duplicate {col}; resolve in reviewed manifest")
    require(frame.sha256.str.fullmatch(r"[a-fA-F0-9]{64}").all(), "Invalid SHA256")
    require(frame.groupby("component_id").label.nunique().eq(1).all(), "Cross-label component")
    require(frame.groupby("component_id").size().ge(3).all(), "Component has fewer than three unique selected views")
    has_source = "intervention_source_id" in frame.columns
    if has_source:
        require(frame.intervention_source_id.str.len().gt(0).all(), "Empty intervention source")
        require(frame.groupby("component_id").intervention_source_id.nunique().eq(1).all(), "Multiple intervention sources inside component")
        require(frame.groupby("intervention_source_id").component_id.nunique().eq(1).all(), "Common intervention source crosses components")
    else:
        frame["intervention_source_id"] = "UNVERIFIED_NO_SOURCE_ID"
    require(args.healthy_label in set(frame.label), "Healthy label absent")
    labels = [args.healthy_label] + sorted(set(frame.label)-{args.healthy_label})
    require(len(labels) >= 3, "Multiclass research purpose requires at least three labels; no automatic binary-only fallback")
    matrices, feature_info = {}, {}
    wanted = frame.local_path.map(path_key).tolist()
    require(len(set(wanted)) == len(wanted), "Normalized manifest paths collide")
    for name, path in (("resnet_logit", args.resnet_features), ("colour_logit", args.colour_features)):
        with np.load(path, allow_pickle=False) as archive:
            require({"features", "paths"}.issubset(archive.files), f"Invalid NPZ {path}")
            paths = [path_key(x) for x in archive["paths"].astype(str)]
            require(len(paths) == len(set(paths)), f"Duplicate cache paths: {name}")
            positions = {p: i for i, p in enumerate(paths)}
            require(set(wanted).issubset(positions), f"Missing selected paths in {name}")
            features = archive["features"]
            require(features.ndim == 2 and features.shape[0] == len(paths), "Feature shape mismatch")
            matrix = np.asarray(features[[positions[p] for p in wanted]], dtype=np.float64)
        require(np.isfinite(matrix).all(), f"Nonfinite features: {name}")
        matrices[name] = matrix
        feature_info[name] = {"dimension": matrix.shape[1], "cache_rows": len(paths), "selected_rows": len(matrix), "alignment": "normalized exact path"}
    return frame, matrices, labels, has_source, feature_info


def create_plan(frame, labels, seed, repetition, allocation_pairs):
    rng = np.random.default_rng(seed)
    components = {str(cid): group.index.to_numpy(dtype=int) for cid, group in frame.groupby("component_id", sort=True)}
    component_label = frame.groupby("component_id", sort=True).label.first()
    records, matched_pairs, odd, test_positions = {}, [], set(), []
    for label in labels:
        ids = component_label.index[component_label.eq(label)].to_numpy(dtype=str)
        n_p, n_s = round(0.2*len(ids)), round(0.1*len(ids))
        n_d = len(ids)-n_p-n_s
        require(n_p >= 2 and n_s >= 1 and n_d >= n_p,
                f"HOLD_MULTICLASS_SUPPORT: {label}: N={len(ids)}, P={n_p}, S={n_s}, D={n_d}")
        order = rng.permutation(ids)
        probes = order[:n_p]
        sentinels = set(order[n_p:n_p+n_s])
        donors = order[n_p+n_s:n_p+n_s+n_p]
        donor_for = dict(zip(probes, donors))
        probe_for = {d: p for p, d in donor_for.items()}
        pair_order = rng.permutation(probes)
        for pos in range(0, len(pair_order)-1, 2):
            matched_pairs.append((str(pair_order[pos]), str(pair_order[pos+1])))
        if len(pair_order) % 2:
            odd.add(str(pair_order[-1]))
        for cid in ids:
            role = "probe" if cid in donor_for else ("sentinel" if cid in sentinels else ("donor" if cid in probe_for else "base"))
            chosen = rng.choice(components[cid], 3 if role == "probe" else (1 if role == "sentinel" else 2), replace=False)
            anchor = int(chosen[0]) if role in ("probe", "sentinel") else None
            views = chosen[1:] if role == "probe" else (np.array([], dtype=int) if role == "sentinel" else chosen)
            if anchor is not None:
                test_positions.append(anchor)
            records[str(cid)] = {"repetition": repetition, "seed": seed, "component_id": str(cid),
                "intervention_source_id": str(frame.iloc[components[cid][0]].intervention_source_id),
                "label": label, "role": role, "paired_donor_or_probe": str(donor_for.get(cid, probe_for.get(cid, ""))),
                "probe_pair_id": "", "odd_unpaired_probe": int(cid in odd),
                "test_anchor_row_idx": "" if anchor is None else str(frame.iloc[anchor].row_idx),
                "view1_row_idx": str(frame.iloc[views[0]].row_idx) if len(views) else "",
                "view2_row_idx": str(frame.iloc[views[1]].row_idx) if len(views) else "",
                "available_view_count": len(components[cid]), "_views": views}
    for pair_id, (left, right) in enumerate(matched_pairs):
        records[left]["probe_pair_id"] = str(pair_id)
        records[right]["probe_pair_id"] = str(pair_id)
    exposure = {"dose_0": set()}
    for allocation in range(allocation_pairs):
        coins = rng.integers(0, 2, size=len(matched_pairs))
        a = {pair[int(coin)] for pair, coin in zip(matched_pairs, coins)}
        b = {pair[1-int(coin)] for pair, coin in zip(matched_pairs, coins)}
        exposure[f"allocation_{allocation}_A"] = a
        exposure[f"allocation_{allocation}_B"] = b
        require(not (a & b) and len(a) == len(b), "Complementary allocation broken")
    exposure["dose_1"] = {cid for cid, r in records.items() if r["role"] == "probe"}
    test = np.sort(np.array(test_positions, dtype=int))
    test_components = frame.iloc[test].component_id.to_numpy(dtype=str)
    panels = np.array([records[cid]["role"] for cid in test_components])
    paired = np.array([records[cid]["role"] == "probe" and cid not in odd for cid in test_components])
    train_by_condition, checks, assignment_rows = {}, [], []
    clean_counts = None
    n_d = sum(r["role"] in ("donor", "base") for r in records.values())
    quota_by_class = {label: sum(records[left]["label"] == label for left, _ in matched_pairs) for label in labels}
    source_ids_verified = frame.intervention_source_id.ne("UNVERIFIED_NO_SOURCE_ID").all()
    for condition, exposed in exposure.items():
        train_components = set()
        for cid, record in records.items():
            active = (record["role"] in ("donor", "base") and record["paired_donor_or_probe"] not in exposed) or (record["role"] == "probe" and cid in exposed)
            if active:
                train_components.add(cid)
            assignment_rows.append({"repetition": repetition, "seed": seed, "condition": condition, "component_id": cid,
                "intervention_source_id": record["intervention_source_id"], "label": record["label"], "role": record["role"],
                "in_training": int(active), "own_probe_exposed": int(cid in exposed),
                "paired_probe": int(record["role"] == "probe" and cid not in odd)})
        train = np.sort(np.concatenate([records[cid]["_views"] for cid in sorted(train_components)]))
        counts = frame.iloc[train].label.value_counts().reindex(labels, fill_value=0).to_numpy()
        if clean_counts is None:
            clean_counts = counts.copy()
        require(len(train) == len(set(train)) == 2*n_d, "Training file budget mismatch")
        require(len(train_components) == n_d, "Training component budget mismatch")
        require(frame.iloc[train].component_id.value_counts().eq(2).all(), "Not two selected views per training component")
        require(np.array_equal(counts, clean_counts), "Training class support changed")
        require(not set(train).intersection(test), "Shared train/test row")
        require(not set(frame.iloc[train].sha256).intersection(frame.iloc[test].sha256), "Shared train/test image SHA")
        sentinel_components = set(test_components[panels == "sentinel"])
        require(not train_components.intersection(sentinel_components), "Exposed sentinel isolation component")
        actual_exposed = set(test_components).intersection(train_components)
        require(actual_exposed == exposed, "Exposure mismatch")
        expected_sources = {records[cid]["intervention_source_id"] for cid in exposed}
        shared_sources = set(frame.iloc[train].intervention_source_id).intersection(frame.iloc[test].intervention_source_id)
        sentinel_sources = set(frame.iloc[test[panels == "sentinel"]].intervention_source_id)
        sentinel_source_overlap = set(frame.iloc[train].intervention_source_id).intersection(sentinel_sources)
        if source_ids_verified:
            require(shared_sources == expected_sources, "Train/test intervention-source intersection is not exactly the exposed probe sources")
            require(not sentinel_source_overlap, "Sentinel intervention source enters training")
            if condition == "dose_0":
                require(not shared_sources, "Zero-exposure arm shares an intervention source with test")
        exposure_by_class = {label: sum(records[c]["label"] == label for c in exposed) for label in labels}
        if condition.startswith("allocation_"):
            require(not odd.intersection(exposed), "Odd probe exposed in fixed-quota experiment")
            require(len(exposed) == len(matched_pairs), "Quota count mismatch")
            require(exposure_by_class == quota_by_class, "Original-class fixed quota mismatch in A/B arm")
        train_by_condition[condition] = train
        checks.append({"repetition": repetition, "seed": seed, "condition": condition, "status": "PASS",
            "training_files": len(train), "training_components": n_d, "views_per_training_component": 2,
            "probe_n": int((panels == "probe").sum()), "paired_probe_n": int(paired.sum()), "sentinel_n": int((panels == "sentinel").sum()),
            "exposed_probe_n": len(exposed), "exposed_paired_probe_n": len(exposed-odd),
            "actual_exposure_fraction_all_probes": len(exposed)/int((panels == "probe").sum()),
            "actual_exposure_fraction_paired_probes": len(exposed-odd)/int(paired.sum()),
            "training_class_counts": json.dumps(dict(zip(labels, counts.tolist())), sort_keys=True),
            "exposed_class_counts": json.dumps(exposure_by_class, sort_keys=True),
            "fixed_quota_by_original_class": json.dumps(quota_by_class, sort_keys=True),
            "exact_file_overlap": 0, "sentinel_component_overlap": 0,
            "source_id_checks": "PASS" if source_ids_verified else "UNVERIFIED_PREPARATION_ONLY",
            "train_test_intervention_source_overlap": len(shared_sources) if source_ids_verified else None,
            "expected_exposed_intervention_sources": len(expected_sources) if source_ids_verified else None,
            "sentinel_intervention_source_overlap": len(sentinel_source_overlap) if source_ids_verified else None,
            "test_anchor_sha256": hashlib.sha256("\n".join(frame.iloc[test].sha256).encode()).hexdigest()})
    return {"test": test, "train": train_by_condition, "exposure": exposure, "records": records, "panels": panels,
            "paired": paired, "checks": checks, "assignments": assignment_rows}


def metric_values(y, p):
    k = p.shape[1]
    pred = p.argmax(axis=1)
    require(set(np.unique(y)) == set(range(k)), "Evaluation panel lacks a prespecified endpoint class")
    recalls = [float(np.mean(pred[y == c] == c)) for c in range(k)]
    return {"n": len(y), "balanced_accuracy": float(np.mean(recalls)), "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, labels=np.arange(k), average="macro", zero_division=0)),
        "log_loss": float(log_loss(y, p, labels=np.arange(k))),
        "brier": float(np.mean((p[:, 1]-y)**2) if k == 2 else np.mean(np.sum((p-np.eye(k)[y])**2, axis=1))),
        "auroc": float(roc_auc_score(y, p[:, 1]) if k == 2 else roc_auc_score(y, p, average="macro", multi_class="ovr"))}


def predict(x, y, train, test, seed, model_name, cosine_x):
    if model_name == "resnet_cosine_1nn":
        nearest = train[(cosine_x[test] @ cosine_x[train].T).argmax(axis=1)]
        return np.eye(len(np.unique(y)))[y[nearest]], nearest, [], 0
    model = Pipeline([("scale", StandardScaler()), ("classifier", LogisticRegression(C=0.01,
        solver="lbfgs", class_weight="balanced", max_iter=10000, random_state=seed))])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(x[train], y[train])
        p = model.predict_proba(x[test])
    require(not any(issubclass(w.category, ConvergenceWarning) for w in caught), "Convergence warning; halted without changing frozen settings")
    require(model.named_steps["scale"].n_samples_seen_ == len(train), "Scaler support mismatch")
    require(np.array_equal(model.named_steps["classifier"].classes_, np.arange(len(np.unique(y)))), "Classifier class order mismatch")
    require(np.isfinite(p).all() and np.allclose(p.sum(axis=1), 1), "Invalid probabilities")
    return p, None, [{"category": w.category.__name__, "message": str(w.message)} for w in caught], int(model.named_steps["classifier"].n_iter_.max())


def class_mean(values, y):
    return float(np.mean([np.mean(values[y == c]) for c in np.unique(y)]))


def summarize(frame, groups, value_columns):
    rows = []
    for key, part in frame.groupby(groups, sort=True, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        for col in value_columns:
            values = part[col].to_numpy(dtype=float)
            rows.append({**dict(zip(groups, key)), "metric": col, "mean": float(values.mean()),
                "split_sd": float(values.std(ddof=1)) if len(values)>1 else None,
                "split_p025": float(np.quantile(values, .025)), "split_p975": float(np.quantile(values, .975)),
                "split_count": len(values)})
    return pd.DataFrame(rows)


def execute(args, frame, matrices, labels, plans, freeze):
    started = time.perf_counter()
    out = Path(args.output_dir)
    mapping = dict(zip(labels, range(len(labels))))
    ys = {"multiclass": frame.label.map(mapping).to_numpy(dtype=int),
          "binary_healthy_other": frame.label.ne(args.healthy_label).to_numpy(dtype=int)}
    xdeep = matrices["resnet_logit"]
    norm = np.linalg.norm(xdeep, axis=1)
    require(np.all(norm > 0), "Zero norm ResNet vector prevents defined cosine-1NN")
    cosine_x = xdeep/norm[:, None]
    model_names = ["resnet_logit", "colour_logit", "resnet_cosine_1nn"]
    pred_fields = ["repetition", "seed", "condition", "model", "endpoint", "panel", "paired_probe", "own_probe_exposed",
        "row_idx", "component_id", "intervention_source_id", "original_label", "truth", "predicted", "nearest_train_row_idx"] + [f"p{i}" for i in range(len(labels))]
    contrast_fields = ["repetition", "seed", "allocation_pair", "model", "endpoint", "row_idx", "component_id", "intervention_source_id",
        "original_label", "truth", "exposed_in_A", "correct_exposed", "correct_unexposed", "d_correct",
        "p_true_exposed_minus_unexposed", "log_loss_exposed_minus_unexposed", "brier_exposed_minus_unexposed", "A_B_total_variation", "A_B_prediction_flip"]
    metric_rows, fit_rows, allocation_rows, sentinel_rows, warning_rows = [], [], [], [], []
    prediction_count, individual_count, fits = 0, 0, 0
    with csv_writer(out/"predictions.csv.gz", pred_fields) as pred_writer, csv_writer(out/"individual_allocation_contrasts.csv.gz", contrast_fields) as d_writer, threadpool_limits(limits=1):
        for rep, plan in enumerate(plans):
            seed = args.seed_start+rep
            test, panels, paired = plan["test"], plan["panels"], plan["paired"]
            test_components = frame.iloc[test].component_id.to_numpy(dtype=str)
            evaluation_panels = {"all_probe": panels == "probe", "paired_probe": paired, "sentinel": panels == "sentinel"}
            for model_name in model_names:
                x = matrices["colour_logit" if model_name == "colour_logit" else "resnet_logit"]
                for endpoint, y in ys.items():
                    probability = {}
                    key = {"repetition": rep, "seed": seed, "model": model_name, "endpoint": endpoint}
                    for condition, train in plan["train"].items():
                        t0 = time.perf_counter()
                        p, nearest, caught, n_iter = predict(x, y, train, test, seed, model_name, cosine_x)
                        fits += int(model_name != "resnet_cosine_1nn")
                        probability[condition] = p
                        warning_rows.extend([{**key, "condition": condition, **w} for w in caught])
                        fit_rows.append({**key, "condition": condition, "n_train": len(train), "n_test": len(test),
                            "seconds": time.perf_counter()-t0, "n_iter": n_iter, "warnings": len(caught)})
                        for panel_name, mask in evaluation_panels.items():
                            metric_rows.append({**key, "condition": condition, "evaluation_panel": panel_name, **metric_values(y[test][mask], p[mask])})
                        for i, pos in enumerate(test):
                            item = frame.iloc[pos]
                            pred_writer.writerow({**key, "condition": condition, "panel": panels[i], "paired_probe": int(paired[i]),
                                "own_probe_exposed": int(item.component_id in plan["exposure"][condition]), "row_idx": item.row_idx,
                                "component_id": item.component_id, "intervention_source_id": item.intervention_source_id,
                                "original_label": item.label, "truth": int(y[pos]), "predicted": int(p[i].argmax()),
                                "nearest_train_row_idx": "" if nearest is None else frame.iloc[nearest[i]].row_idx,
                                **{f"p{c}": float(p[i, c]) for c in range(p.shape[1])}})
                            prediction_count += 1
                    truth = y[test]
                    q0 = probability["dose_0"]
                    for allocation in range(args.allocation_pairs):
                        ca, cb = f"allocation_{allocation}_A", f"allocation_{allocation}_B"
                        pa, pb = probability[ca], probability[cb]
                        sign = np.array([1 if c in plan["exposure"][ca] else -1 for c in test_components])
                        a_correct, b_correct = (pa.argmax(1) == truth).astype(int), (pb.argmax(1) == truth).astype(int)
                        own_d = sign*(a_correct-b_correct)
                        pt_a, pt_b = pa[np.arange(len(test)), truth], pb[np.arange(len(test)), truth]
                        probability_d = sign*(pt_a-pt_b)
                        loss_a = -np.log(np.clip(pt_a, np.finfo(float).eps, 1))
                        loss_b = -np.log(np.clip(pt_b, np.finfo(float).eps, 1))
                        onehot = np.eye(pa.shape[1])[truth]
                        ba = (pa[:, 1]-truth)**2 if pa.shape[1] == 2 else np.sum((pa-onehot)**2, axis=1)
                        bb = (pb[:, 1]-truth)**2 if pb.shape[1] == 2 else np.sum((pb-onehot)**2, axis=1)
                        tv = .5*np.abs(pa-pb).sum(1)
                        flip = pa.argmax(1) != pb.argmax(1)
                        for i in np.flatnonzero(paired):
                            item = frame.iloc[test[i]]
                            d_writer.writerow({**key, "allocation_pair": allocation, "row_idx": item.row_idx, "component_id": item.component_id,
                                "intervention_source_id": item.intervention_source_id, "original_label": item.label, "truth": int(truth[i]),
                                "exposed_in_A": int(sign[i] == 1), "correct_exposed": int(a_correct[i] if sign[i] == 1 else b_correct[i]),
                                "correct_unexposed": int(b_correct[i] if sign[i] == 1 else a_correct[i]), "d_correct": int(own_d[i]),
                                "p_true_exposed_minus_unexposed": float(probability_d[i]),
                                "log_loss_exposed_minus_unexposed": float(sign[i]*(loss_a[i]-loss_b[i])),
                                "brier_exposed_minus_unexposed": float(sign[i]*(ba[i]-bb[i])),
                                "A_B_total_variation": float(tv[i]), "A_B_prediction_flip": int(flip[i])})
                            individual_count += 1
                        for stratum in ["all_paired"] + labels:
                            mask = paired if stratum == "all_paired" else paired & frame.iloc[test].label.eq(stratum).to_numpy()
                            require(mask.any(), "No paired probe in required original-label stratum")
                            allocation_rows.append({**key, "allocation_pair": allocation, "stratum": stratum, "n": int(mask.sum()),
                                "own_exposure_BA_contrast": class_mean(own_d[mask], truth[mask]),
                                "own_exposure_accuracy_contrast": float(own_d[mask].mean()),
                                "own_exposure_true_probability_contrast": float(probability_d[mask].mean()),
                                "own_exposure_log_loss_contrast": float((sign*(loss_a-loss_b))[mask].mean()),
                                "own_exposure_brier_contrast": float((sign*(ba-bb))[mask].mean()),
                                "A_B_total_variation": float(tv[mask].mean()), "A_B_prediction_flip": float(flip[mask].mean())})
                        for panel_name, mask in evaluation_panels.items():
                            ma, mb, m0 = metric_values(truth[mask], pa[mask]), metric_values(truth[mask], pb[mask]), metric_values(truth[mask], q0[mask])
                            sentinel_rows.append({**key, "allocation_pair": allocation, "evaluation_panel": panel_name, "n": int(mask.sum()),
                                "prediction_disagreement_rate": float(flip[mask].mean()), "mean_total_variation": float(tv[mask].mean()),
                                "mean_absolute_true_probability_difference": float(np.abs(pt_a-pt_b)[mask].mean()),
                                "mean_absolute_positive_probability_difference": float(np.abs(pa[:, 1]-pb[:, 1])[mask].mean()) if pa.shape[1] == 2 else None,
                                "mean_AB_BA_minus_q0": .5*(ma["balanced_accuracy"]+mb["balanced_accuracy"])-m0["balanced_accuracy"],
                                "mean_AB_Brier_minus_q0": .5*(ma["brier"]+mb["brier"])-m0["brier"],
                                "p95_total_variation": float(np.quantile(tv[mask], .95))})
            print(f"Completed seed {rep+1}/{args.repetitions}; logistic fits={fits}; elapsed={time.perf_counter()-started:.1f}s", flush=True)
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(out/"condition_metrics.csv", index=False)
    pd.DataFrame(fit_rows).to_csv(out/"fit_log.csv", index=False)
    allocations = pd.DataFrame(allocation_rows)
    allocations.to_csv(out/"allocation_pair_summary.csv", index=False)
    diagnostics = pd.DataFrame(sentinel_rows)
    diagnostics.to_csv(out/"sentinel_and_global_diagnostics.csv", index=False)
    allocation_values = [c for c in allocations if c.startswith("own_") or c.startswith("A_B_")]
    seed_alloc = allocations.groupby(["repetition", "seed", "model", "endpoint", "stratum"], as_index=False)[allocation_values].mean()
    seed_alloc["allocation_pairs_averaged"] = args.allocation_pairs
    seed_alloc.to_csv(out/"allocation_seed_means.csv", index=False)
    summarize(seed_alloc, ["model", "endpoint", "stratum"], allocation_values).to_csv(out/"allocation_split_summary.csv", index=False)
    diag_values = ["prediction_disagreement_rate", "mean_total_variation", "mean_absolute_true_probability_difference", "mean_AB_BA_minus_q0", "mean_AB_Brier_minus_q0", "p95_total_variation"]
    seed_diag = diagnostics.groupby(["repetition", "seed", "model", "endpoint", "evaluation_panel"], as_index=False)[diag_values].mean()
    seed_diag.to_csv(out/"diagnostic_seed_means.csv", index=False)
    summarize(seed_diag, ["model", "endpoint", "evaluation_panel"], diag_values).to_csv(out/"diagnostic_split_summary.csv", index=False)
    # Dose 0.5 is exactly allocation 0 A, not an extra fit or a selected result.
    dose_map = {"dose_0": "0", "allocation_0_A": "0.5", "dose_1": "1"}
    doses = metrics[metrics.condition.isin(dose_map)].copy()
    doses["dose"] = doses.condition.map(dose_map)
    doses.to_csv(out/"dose_metrics.csv", index=False)
    metric_cols = ["balanced_accuracy", "accuracy", "macro_f1", "log_loss", "brier", "auroc"]
    summarize(doses, ["model", "endpoint", "dose", "evaluation_panel"], metric_cols).to_csv(out/"dose_split_summary.csv", index=False)
    id_cols = ["repetition", "seed", "model", "endpoint", "evaluation_panel"]
    zero = doses[doses.dose.eq("0")].set_index(id_cols)
    delta_rows = []
    for dose in ("0.5", "1"):
        other = doses[doses.dose.eq(dose)].set_index(id_cols).loc[zero.index]
        delta = other[metric_cols]-zero[metric_cols]
        delta = delta.reset_index()
        delta["dose"] = dose
        delta_rows.append(delta)
    deltas = pd.concat(delta_rows, ignore_index=True)
    deltas.to_csv(out/"dose_minus_zero_by_seed.csv", index=False)
    summarize(deltas, ["model", "endpoint", "dose", "evaluation_panel"], metric_cols).to_csv(out/"dose_minus_zero_summary.csv", index=False)
    warning_count = len(warning_rows)
    dump(out/"warnings.json", {"count": warning_count, "items": warning_rows})
    for name, info in freeze["inputs"].items():
        require(sha(info["path_at_execution"]) == info["sha256"], f"Input changed during execution: {name}")
    require(sha(__file__) == freeze["script_sha256"], "Code changed during execution")
    expected_fits = args.repetitions*(2+2*args.allocation_pairs)*2*2
    require(fits == expected_fits, "Fit count mismatch")
    dump(out/"completion.json", {"status": "COMPLETED_LIMITED_CORRECTION_NOT_SUBMISSION_READY", "completed_utc": datetime.now(timezone.utc).isoformat(),
        "logistic_fits": fits, "expected_logistic_fits": expected_fits, "prediction_rows": prediction_count,
        "individual_allocation_contrast_rows": individual_count, "warning_count": warning_count,
        "elapsed_seconds": time.perf_counter()-started, "protocol_sha256": sha(out/"protocol_freeze.json"),
        "all_input_and_code_hashes_unchanged": True, "uncertainty": "Split/randomization variation, not biological confidence intervals", "scientific_submission_gate": "HOLD"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--resnet-features", required=True, type=Path)
    parser.add_argument("--colour-features", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--lineage-audit", type=Path)
    parser.add_argument("--healthy-label", default="healthy")
    parser.add_argument("--repetitions", default=30, type=int)
    parser.add_argument("--allocation-pairs", default=5, type=int)
    parser.add_argument("--seed-start", default=20261002, type=int)
    parser.add_argument("--execute", action="store_true", help="Fit only after a reviewed input manifest passes the lineage gate")
    args = parser.parse_args()
    require(args.repetitions >= 1 and args.allocation_pairs >= 1, "Invalid repetition count")
    require(not args.output_dir.exists(), "Output directory must be new; frozen runs are never overwritten")
    frame, matrices, labels, has_source, feature_info = load_inputs(args)
    inputs = {"manifest": args.manifest, "resnet_features": args.resnet_features, "colour_features": args.colour_features}
    review = None
    if args.lineage_audit:
        review = json.loads(args.lineage_audit.read_text(encoding="utf-8"))
        require(review.get("manifest_sha256") == sha(args.manifest), "Independent review does not match manifest")
        inputs["lineage_audit"] = args.lineage_audit
    if args.execute:
        require(has_source, "HOLD: no intervention_source_id; component membership alone does not establish sibling views")
        require(review is not None, "HOLD: independent lineage audit required for fitting")
        require(review.get("status") == "ALLOW_LIMITED_CORRECTION_COMPUTATION_NOT_SCIENTIFIC_OR_SUBMISSION_READY",
                "HOLD: lineage review does not authorize limited correction computation")
        require(all(item.get("pass") is True for item in review.get("checks", [])) and len(review.get("checks", [])) > 0, "Lineage review contains failed/missing checks")
    plans = [create_plan(frame, labels, args.seed_start+i, i, args.allocation_pairs) for i in range(args.repetitions)]
    # All memberships and invariants are materialized before any model is fitted.
    args.output_dir.mkdir(parents=True)
    frame.to_csv(args.output_dir/"selected_manifest.csv", index=False)
    members = [{k: v for k, v in r.items() if not k.startswith("_")} for plan in plans for r in plan["records"].values()]
    with csv_writer(args.output_dir/"component_membership.csv.gz", list(members[0])) as writer:
        writer.writerows(members)
    assignments = [row for plan in plans for row in plan["assignments"]]
    with csv_writer(args.output_dir/"allocation_membership.csv.gz", list(assignments[0])) as writer:
        writer.writerows(assignments)
    pd.DataFrame([row for plan in plans for row in plan["checks"]]).to_csv(args.output_dir/"support_checks.csv", index=False)
    counts = frame.groupby("component_id").label.first().value_counts().reindex(labels)
    freeze = {"status": "FROZEN_BEFORE_FITTING" if args.execute else "PREPARED_WITHOUT_FITTING", "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_status": "post_hoc_correction_after_old_outcomes_and_lineage_failure_were_known",
        "inputs": {k: {"path_at_execution": str(p.resolve()), "sha256": sha(p), "bytes": p.stat().st_size} for k, p in inputs.items()},
        "script_sha256": sha(__file__), "repetitions": args.repetitions, "seed_start": args.seed_start, "allocation_pairs_per_seed": args.allocation_pairs,
        "selected_images": len(frame), "isolation_components": frame.component_id.nunique(), "components_per_label": counts.to_dict(),
        "labels_multiclass": labels, "binary_labels": [args.healthy_label, "other_dataset_label"], "feature_info": feature_info,
        "intervention_source_id_available": has_source, "P_rule": "Python round(0.2*N_class), require >=2", "S_rule": "Python round(0.1*N_class), require >=1",
        "D_rule": "all remaining components; >=P; two fixed supported views per training component", "class_fallback": "none; insufficient multiclass support halts",
        "fixed_quota_rule": "Random pairs within original class; one exposed per pair in A, other in B; odd probe fixed unexposed for all allocations",
        "dose_half_definition": "allocation_0_A; actual exposed/ALL and exposed/PAIRED denominators reported", "dose_full_definition": "all P including odd exposed",
        "primary_estimand": "Class-balanced mean of (2*z_A-1)*(correct_A-correct_B) on PAIRED probes under this fixed-quota allocation policy",
        "secondary_weighting": "own_exposure_accuracy_contrast, true-probability, log-loss, Brier, TV and flip contrasts are image means within their stated stratum; only own_exposure_BA_contrast is endpoint-class-balanced",
        "interference": "Other families change exposure too; not a pure direct effect holding remaining training data fixed",
        "sentinel_diagnostic": "A/B prediction disagreement, absolute probability differences and mean A/B metric minus q0; signed A-B cancellation is not evidence of no global effect",
        "models": {"resnet_logit": "C=.01, balanced, lbfgs, standardization within training arm", "colour_logit": "same classifier protocol", "resnet_cosine_1nn": "untuned memory-sensitive reference; cosine on fixed embeddings; ties take first ordered training row"},
        "uncertainty": "Average 5 allocation pairs within seed, then describe 30 split seeds; no p-values or biological independence claims",
        "correction_scope": "Recompute on reviewed conservative components; old filename-token outputs are preserved but not treated as valid source-separated estimates",
        "source_limit": "One audited common-reference view set per blocking component; unselected connected material is quarantined; neither component nor reference ID is a plant ID",
        "scientific_submission_gate": "HOLD", "software": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__, "scikit_learn": sklearn.__version__},
        "prepared_file_sha256": {p.name: sha(p) for p in sorted(args.output_dir.iterdir()) if p.is_file()}}
    dump(args.output_dir/"protocol_freeze.json", freeze)
    if args.execute:
        try:
            execute(args, frame, matrices, labels, plans, freeze)
        except Exception as error:
            dump(args.output_dir/"FAILED.json", {"status": "FAILED_NOT_COMPLETE", "error": f"{type(error).__name__}: {error}", "scientific_submission_gate": "HOLD"})
            raise
    else:
        print(f"Prepared only: {len(frame)} images, {frame.component_id.nunique()} components, {len(plans)} seeds; no model fitted.")
    files = [p for p in sorted(args.output_dir.iterdir()) if p.is_file() and p.name != "artifact_manifest.csv"]
    pd.DataFrame([{"path": p.name, "bytes": p.stat().st_size, "sha256": sha(p)} for p in files]).to_csv(args.output_dir/"artifact_manifest.csv", index=False)


if __name__ == "__main__":
    main()
