"""Frozen single-source E/H replacements at the v17 zero-exposure background.

prepare writes assignments without fitting; complete preparation and a hash-bound
independent PASS review are mandatory before run. All fixed models are retained.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "vendor"))
import run_corrected_component_experiment as base
import run_matched_replacement_controls as matched

D18 = HERE.parent
D17 = D18.parent / "Onion_Deep_Revision_20261003_v17"
MODELS = ["resnet_logit", "colour_logit", "dinov2_logit",
          "resnet_cosine_1nn", "dinov2_cosine_1nn"]
SCHEMES = ("all_original_classes", "common_classes_n_ge_30")
SEED_START, N_SPLITS = 20261003, 10


def defaults(dataset):
    experiment = (D17 / "matched_replacement_onion" if dataset == "onion" else
                  D17 / "independent_task/potato/matched_replacement_experiment")
    source = (D17 / "onion_inputs" if dataset == "onion" else
              D17 / "independent_task/potato")
    return dict(manifest=source / "analysis_manifest.csv",
                resnet_features=source / ("resnet18.npz" if dataset == "onion" else "features/resnet18.npz"),
                colour_features=source / ("colour189.npz" if dataset == "onion" else "features/colour189.npz"),
                membership=experiment / "component_membership.csv",
                joint_predictions=experiment / "predictions.csv.gz",
                dino_features=D18 / "features" / f"{dataset}_dinov2_vits14.npz",
                output=D18 / "single_source" / dataset)


def sha_strings(values):
    return hashlib.sha256(("\n".join(sorted(map(str, values))) + "\n").encode()).hexdigest()


def load_dino(path, frame):
    with np.load(path, allow_pickle=False) as z:
        required = {"features", "paths", "row_idx", "image_sha256"}
        base.require(required.issubset(z.files), "DINO cache lacks features/paths/row_idx/image_sha256")
        paths = [base.path_key(x) for x in z["paths"].astype(str)]
        base.require(len(paths) == len(set(paths)), "Duplicated DINO path")
        index = {p: i for i, p in enumerate(paths)}
        wanted = frame.local_path.map(base.path_key).tolist()
        base.require(set(wanted).issubset(index), "Missing DINO manifest path")
        pos = np.array([index[p] for p in wanted])
        x = np.asarray(z["features"][pos], dtype=np.float64)
        rid = z["row_idx"].astype(str)[pos]
        ish = z["image_sha256"].astype(str)[pos]
        base.require(np.array_equal(rid, frame.row_idx.to_numpy(str)), "DINO row identity mismatch")
        base.require(np.array_equal(ish, frame.sha256.to_numpy(str)), "DINO image hash mismatch")
    base.require(x.ndim == 2 and x.shape[0] == len(frame) and np.isfinite(x).all(), "Invalid DINO feature matrix")
    return x, dict(dimension=x.shape[1], rows=len(x), alignment="path, row_idx and image_sha256")


def build_plans(frame, labels, membership):
    saved = pd.read_csv(membership, dtype=str, keep_default_na=False)
    subset = saved[saved.repetition.astype(int).lt(N_SPLITS)].copy()
    base.require(set(subset.repetition.astype(int)) == set(range(N_SPLITS)), "Missing first ten original splits")
    plans, audit, members, target_rows = [], [], [], []
    source_labels = frame.groupby("component_id").label.first()
    common = sorted(source_labels.value_counts()[lambda x: x.ge(30)].index.tolist())
    base.require(len(common) >= 2, "Common-class sensitivity has fewer than two labels")
    for rep in range(N_SPLITS):
        seed = SEED_START + rep
        original = matched.plan_one(frame, labels, seed, rep)
        records = original["records"]
        part = subset[subset.repetition.astype(int).eq(rep)].set_index("component_id")
        base.require(set(part.index) == set(records), "Saved/reconstructed component identifiers differ")
        for cid, rec in records.items():
            for col, value in rec.items():
                base.require(col in part.columns or col == "component_id", f"Missing saved membership {col}")
                if col != "component_id":
                    base.require(str(part.loc[cid, col]) == str(value), f"v17 record parity failure: {rep} {cid} {col}")
        # All execution uses the validated saved source and view records.
        rr = part.reset_index().to_dict("records")
        records = {r["component_id"]: r for r in rr}
        for r in records.values():
            for field in ("paired", "anchor_position", "view1_position", "view2_position"):
                r[field] = int(r[field])
        test = np.array(sorted(r["anchor_position"] for r in records.values() if r["anchor_position"] >= 0))
        tc = frame.iloc[test].component_id.to_numpy(str)
        panels = np.array([records[c]["role"] for c in tc])
        train0c = {c for c, r in records.items() if r["role"] in ("core", "remove")}
        train0 = np.array(sorted(j for c in train0c for j in (records[c]["view1_position"], records[c]["view2_position"])))
        expected_counts = frame.iloc[train0].label.value_counts().to_dict()
        targets = sorted(c for c, r in records.items() if r["paired"] == 1)
        plan = dict(repetition=rep, seed=seed, records=records, test=test, tc=tc,
                    panels=panels, train0=train0, train0c=train0c, targets=[], common=common)
        for cid in sorted(train0c):
            r = records[cid]
            members.append(dict(repetition=rep, seed=seed, target_component="", policy="zero", component_id=cid,
                                source_id=r["source_id"], label=r["label"], view1_position=r["view1_position"], view2_position=r["view2_position"]))
        for target in targets:
            r = records[target]
            donor, alternate = r["remove"], r["alternate"]
            base.require(donor in train0c and alternate not in train0c and target not in train0c, "Invalid mapped donor/alternate")
            base.require(len({target, donor, alternate}) == 3, "Mapped sources not distinct")
            base.require(records[donor]["label"] == records[alternate]["label"] == r["label"], "Cross-label replacement")
            common_background = train0c - {donor}
            target_plan = dict(target=target, label=r["label"], donor=donor, alternate=alternate)
            for policy, insertion in (("exposure", target), ("sham", alternate)):
                sources = common_background | {insertion}
                train = np.array(sorted(j for c in sources for j in (records[c]["view1_position"], records[c]["view2_position"])))
                target_plan[policy] = train
                target_plan[policy + "_sources"] = sources
                overlap = sources & set(tc)
                expected_overlap = {target} if policy == "exposure" else set()
                source_overlap = set(frame.iloc[train].intervention_source_id) & set(frame.iloc[test].intervention_source_id)
                expected_source_overlap = {r["source_id"]} if policy == "exposure" else set()
                base.require(len(train) == len(train0) == len(set(train)), "Training budget changed")
                base.require(frame.iloc[train].label.value_counts().to_dict() == expected_counts, "Training original-label counts changed")
                base.require(overlap == expected_overlap and source_overlap == expected_source_overlap, "Unexpected exposure")
                base.require(not set(train) & set(test), "Anchor entered training")
                base.require(not set(frame.iloc[train].sha256) & set(frame.iloc[test].sha256), "Exact image train/test leakage")
                base.require(not sources & set(tc[panels == "sentinel"]), "Sentinel entered training")
                audit.append(dict(repetition=rep, seed=seed, target_component=target, target_label=r["label"],
                                  donor_component=donor, alternate_component=alternate, policy=policy,
                                  training_sources=len(sources), training_images=len(train), test_anchors=len(test),
                                  exposed_sources=len(overlap), sentinel_overlap=0, common_background_sha256=sha_strings(common_background),
                                  train_sources_sha256=sha_strings(sources), training_row_idx_sha256=sha_strings(frame.iloc[train].row_idx),
                                  original_label_counts=json.dumps(expected_counts, sort_keys=True), status="PASS"))
                for cid in sorted(sources):
                    s = records[cid]
                    members.append(dict(repetition=rep, seed=seed, target_component=target, policy=policy, component_id=cid,
                                        source_id=s["source_id"], label=s["label"], view1_position=s["view1_position"], view2_position=s["view2_position"]))
            base.require(target_plan["exposure_sources"] - {target} == target_plan["sham_sources"] - {alternate} == common_background,
                         "More than one source differs between exposure/sham")
            target_rows.append(dict(repetition=rep, seed=seed, target_component=target, target_label=r["label"],
                                    donor_component=donor, alternate_component=alternate, anchor_position=r["anchor_position"],
                                    own_view1_position=r["view1_position"], own_view2_position=r["view2_position"],
                                    common_class=int(r["label"] in common)))
            plan["targets"].append(target_plan)
        plans.append(plan)
    return plans, pd.DataFrame(audit), members, pd.DataFrame(target_rows), common


def input_paths(args):
    paths = {k: Path(getattr(args, k)) for k in ("manifest", "resnet_features", "colour_features", "membership", "joint_predictions")}
    paths.update(code=Path(__file__).resolve(), shared_numerics=Path(base.__file__), original_assignment_code=Path(matched.__file__))
    if args.dino_features.exists():
        paths["dino_features"] = args.dino_features
    # These hash records bind the extraction/checkpoint provenance to the run.
    for name in ("DINOV2_PRE_EXTRACTION_FREEZE.json", "DINOV2_OFFICIAL_SOURCE_BLOB_VERIFICATION.json", "dinov2_source_file_hashes.json"):
        p = args.dino_features.parent / name
        if p.exists():
            paths["dino_provenance_" + name] = p
    return paths


def input_hashes(paths):
    return {k: {"path": str(p.resolve()), "sha256": base.sha(p)} for k, p in paths.items()}


def prepare(args, frame, matrices, labels, feature_info, plans, audit, members, targets, common):
    out = args.output
    exists = out.exists()
    if exists:
        old = json.loads((out / "protocol_freeze.json").read_text(encoding="utf-8"))
        base.require(old["status"] == "PREPARED_INPUTS_INCOMPLETE", "Complete frozen preparation cannot be overwritten")
        base.require(not (out / "predictions.csv.gz").exists(), "Executed output cannot be overwritten")
        for name, sha in old["prepared_hashes"].items():
            base.require(base.sha(out / name) == sha, "Incomplete preparation changed")
        for name, item in old["inputs"].items():
            base.require(base.sha(item["path"]) == item["sha256"], "Existing preparation input/code changed; create a fresh output")
    else:
        out.mkdir(parents=True)
        frame.to_csv(out / "manifest.csv", index=False)
        audit.to_csv(out / "support_checks.csv", index=False)
        targets.to_csv(out / "targets.csv", index=False)
        with base.csv_writer(out / "training_membership.csv.gz", list(members[0])) as writer:
            writer.writerows(members)
        pd.read_csv(args.membership, dtype=str, keep_default_na=False).query("repetition in @reps", local_dict={"reps": list(map(str, range(N_SPLITS)))}).to_csv(out / "component_membership_first10.csv", index=False)
    # The feature cache is not allowed to be substituted with a different model.
    complete = args.dino_features.exists()
    paths = input_paths(args)
    names = ["manifest.csv", "support_checks.csv", "targets.csv", "training_membership.csv.gz", "component_membership_first10.csv"]
    protocol = dict(status="PREPARED_BEFORE_NEW_FITS" if complete else "PREPARED_INPUTS_INCOMPLETE",
                    frozen_utc=datetime.now(timezone.utc).isoformat(), dataset=args.dataset,
                    chronology="Post hoc extension after v17 outcomes; not prospectively registered and not a new biological cohort.",
                    inputs=input_hashes(paths), prepared_hashes={name: base.sha(out / name) for name in names},
                    class_order=labels, common_class_rule="Original manifest source count >=30; same fitted full-class models.", common_classes=common,
                    models=MODELS, endpoint="multiclass", seeds=list(range(SEED_START, SEED_START + N_SPLITS)),
                    target_count=len(targets), target_counts_per_split=targets.groupby("repetition").size().to_dict(),
                    feature_info=feature_info,
                    numerical_model="Reviewed v17 base.predict: per-fit StandardScaler and balanced L2 LR C=.01, lbfgs, max_iter=10000. Both 1NN models explicitly route to resnet_cosine_1nn using their own normalized features.",
                    primary="Own-target E-H correctness; mean targets within target original class then equal target classes within seed.",
                    other_panels="All non-target probes, including odd unpaired probe, and sentinels. Within each target, original evaluation classes equal weight; target original classes subsequently equal weight.",
                    common_sensitivity="Restrict target classes and evaluation classes to manifest count >=30; retain unchanged models and all original predictions.",
                    uncertainty="Ten split means summarized by mean/min/max/sample SD; no biological CI, independent-source replication count or p-value.",
                    background_comparison="Same target and seed against mean of five existing selected joint E-H allocations; contextual sensitivity, not disease/ceiling mechanism or absence of interference.",
                    pretraining="DINO pretraining image exclusion is not established; the same frozen representation is used in every arm.",
                    scientific_submission_gate="HOLD_UNTIL_INDEPENDENT_REVIEW_AND_MANUSCRIPT_RECONCILIATION")
    base.dump(out / "protocol_freeze.json", protocol)
    print(json.dumps(dict(status=protocol["status"], output=str(out), target_count=len(targets), models=MODELS,
                          protocol_sha256=base.sha(out / "protocol_freeze.json"), code_sha256=base.sha(__file__)), ensure_ascii=False), flush=True)


def pair_metrics(y, q0, pe, ph, mask):
    yy, z, e, h = y[mask], q0[mask].argmax(1), pe[mask].argmax(1), ph[mask].argmax(1)
    base.require(len(yy) > 0, "Empty evaluation panel")
    cm = lambda a: base.class_mean(np.asarray(a, dtype=float), yy)
    out = matched.analyse_pair(y, q0.argmax(1), pe.argmax(1), ph.argmax(1), mask)
    out.update(zero_BA=cm(z == yy), sham_BA=cm(h == yy), exposure_BA=cm(e == yy),
               corrected_relative_to_sham=cm((h != yy) & (e == yy)), harmed_relative_to_sham=cm((h == yy) & (e != yy)),
               sham_error_headroom=cm(h != yy), p_true_difference=cm(pe[mask][np.arange(len(yy)), yy] - ph[mask][np.arange(len(yy)), yy]),
               total_variation=cm(.5 * np.abs(pe[mask] - ph[mask]).sum(1)))
    base.require(abs(out["policy_BA_difference"] - out["corrected_relative_to_sham"] + out["harmed_relative_to_sham"]) < 1e-12,
                 "E-H rescue/harm identity failed")
    base.require(out["corrected_relative_to_sham"] <= out["sham_error_headroom"] + 1e-12, "Rescue exceeded error mass")
    return out


def summarize_targets(rows, out):
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "target_panel_contrasts.csv", index=False)
    keys = ["repetition", "seed", "model", "endpoint", "panel", "weighting", "target_label"]
    metadata = keys + ["target_component", "n_evaluation", "n_evaluation_classes"]
    values = [c for c in frame.columns if c not in metadata]
    byclass = frame.groupby(keys, as_index=False)[values].mean()
    counts = frame.groupby(keys).size().rename("n_targets").reset_index()
    byclass = byclass.merge(counts, on=keys, validate="one_to_one")
    byclass.to_csv(out / "target_class_split_means.csv", index=False)
    splitkeys = [k for k in keys if k != "target_label"]
    splits = byclass.groupby(splitkeys, as_index=False)[values].mean()
    splits.to_csv(out / "split_means.csv", index=False)
    summary = []
    groupkeys = ["model", "endpoint", "panel", "weighting"]
    for key, part in splits.groupby(groupkeys, sort=True):
        base.require(len(part) == N_SPLITS, "Missing split in final summary")
        for metric in values:
            x = part[metric].to_numpy(float)
            summary.append({**dict(zip(groupkeys, key)), "metric": metric, "mean": float(x.mean()), "split_min": float(x.min()),
                            "split_max": float(x.max()), "split_sd": float(x.std(ddof=1)), "split_count": len(x)})
    pd.DataFrame(summary).to_csv(out / "split_summary.csv", index=False)
    return frame


def compare_saved_joint(args, singles, out):
    # Complete v17 predictions, not rounded summary tables, define this contrast.
    selected = []
    for chunk in pd.read_csv(args.joint_predictions, chunksize=150000):
        m = chunk.repetition.lt(N_SPLITS) & chunk.endpoint.eq("multiclass") & chunk.selected.eq(1)
        part = chunk.loc[m, ["repetition", "seed", "model", "condition", "component_id", "label", "truth", "predicted"]].copy()
        if len(part):
            part["correct"] = part.predicted.eq(part.truth).astype(int)
            parts = part.condition.str.split("_", expand=True)
            part["policy"], part["allocation_pair"], part["arm"] = parts[0], parts[1].astype(int), parts[2]
            selected.append(part)
    joint = pd.concat(selected, ignore_index=True)
    keys = ["repetition", "seed", "model", "component_id", "label", "allocation_pair", "arm"]
    paired = joint.pivot(index=keys, columns="policy", values="correct").reset_index()
    base.require(paired[["exposure", "sham"]].notna().all().all(), "Missing joint E/H match")
    paired["joint_E_minus_H"] = paired.exposure - paired.sham
    group = ["repetition", "seed", "model", "component_id", "label"]
    mean = paired.groupby(group).agg(joint_E_minus_H=("joint_E_minus_H", "mean"), joint_allocations=("joint_E_minus_H", "size")).reset_index()
    base.require(mean.joint_allocations.eq(5).all(), "Target was not selected once in every joint allocation pair")
    own = singles[singles.panel.eq("own_target") & singles.weighting.eq(SCHEMES[0])].copy()
    mean = mean.rename(columns={"component_id": "target_component", "label": "target_label"})
    merged = own.merge(mean, on=["repetition", "seed", "model", "target_component", "target_label"], how="left", validate="one_to_one")
    merged["single_minus_joint"] = merged.policy_BA_difference - merged.joint_E_minus_H
    merged["comparison_status"] = np.where(merged.joint_E_minus_H.notna(), "MATCHED_V17_JOINT_TARGET", "NO_JOINT_RESULT_IN_FROZEN_V17_INPUT")
    merged.to_csv(out / "single_vs_v17_joint_targets.csv", index=False)
    keys = ["repetition", "seed", "model", "target_label"]
    complete = merged[merged.joint_E_minus_H.notna()]
    classmeans = complete.groupby(keys, as_index=False)[["policy_BA_difference", "joint_E_minus_H", "single_minus_joint"]].mean()
    classmeans.groupby(["repetition", "seed", "model"], as_index=False)[["policy_BA_difference", "joint_E_minus_H", "single_minus_joint"]].mean().to_csv(out / "single_vs_v17_joint_split_means.csv", index=False)


def run(args, frame, matrices, labels, plans):
    out = args.output
    freeze_path = out / "protocol_freeze.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    base.require(freeze["status"] == "PREPARED_BEFORE_NEW_FITS", "Full DINO input freeze is required")
    base.require(input_hashes(input_paths(args)) == freeze["inputs"], "Input/code hash changed after prepare")
    for name, expected in freeze["prepared_hashes"].items():
        base.require(base.sha(out / name) == expected, "Prepared assignment changed")
    base.require(args.review is not None and args.review.is_file(), "Independent prefit review is required")
    review = json.loads(args.review.read_text(encoding="utf-8"))
    base.require(review.get("status") == "PASS" and review.get("protocol_sha256") == base.sha(freeze_path)
                 and review.get("code_sha256") == base.sha(__file__) and bool(review.get("reviewer")),
                 "Independent review must PASS this exact protocol and code")
    base.require(not (out / "predictions.csv.gz").exists() and not (out / "completion.json").exists(), "Cannot overwrite an executed run")
    base.dump(out / "accepted_prefit_review.json", review)
    truth_all = frame.label.map(dict(zip(labels, range(len(labels))))).to_numpy(int)
    cosine = {}
    for name in ("resnet_logit", "dinov2_logit"):
        norms = np.linalg.norm(matrices[name], axis=1)
        base.require((norms > 0).all(), "Zero vector prevents cosine 1NN")
        cosine[name] = matrices[name] / norms[:, None]
    fields = ["repetition", "seed", "model", "endpoint", "target_component", "target_label", "policy", "panel", "is_own_target",
              "component_id", "row_idx", "label", "truth", "predicted", "nearest_train_row_idx"] + [f"p{i}" for i in range(len(labels))]
    fits, allwarnings, contrast_rows = [], [], []
    n_predictions, started = 0, time.perf_counter()
    with base.csv_writer(out / "predictions.csv.gz", fields) as writer, threadpool_limits(limits=1):
        for plan in plans:
            rep, seed, test, tc = plan["repetition"], plan["seed"], plan["test"], plan["tc"]
            y = truth_all[test]
            testlabels = frame.iloc[test].label.to_numpy(str)
            common_eval = np.isin(testlabels, plan["common"])
            for model in MODELS:
                feature_key = ("dinov2_logit" if model.startswith("dinov2") else "colour_logit" if model == "colour_logit" else "resnet_logit")
                is_nn = model.endswith("cosine_1nn")
                route = "resnet_cosine_1nn" if is_nn else model
                x = matrices[feature_key]
                normed = cosine.get(feature_key)
                key = dict(repetition=rep, seed=seed, model=model, endpoint="multiclass")
                def fit_write(train, policy, target="", target_label=""):
                    nonlocal n_predictions
                    then = time.perf_counter()
                    p, nearest, warning, n_iter = base.predict(x, truth_all, train, test, seed, route, normed)
                    fits.append({**key, "target_component": target, "target_label": target_label, "policy": policy, "numerical_route": route,
                                 "n_train": len(train), "n_test": len(test), "n_iter": n_iter, "warnings": len(warning), "seconds": time.perf_counter()-then})
                    allwarnings.extend({**key, "target_component": target, "policy": policy, **w} for w in warning)
                    for j, pos in enumerate(test):
                        r = frame.iloc[pos]
                        writer.writerow({**key, "target_component": target, "target_label": target_label, "policy": policy,
                                         "panel": plan["panels"][j], "is_own_target": int(r.component_id == target), "component_id": r.component_id,
                                         "row_idx": r.row_idx, "label": r.label, "truth": int(y[j]), "predicted": int(p[j].argmax()),
                                         "nearest_train_row_idx": "" if nearest is None else frame.iloc[nearest[j]].row_idx,
                                         **{f"p{k}": float(p[j,k]) for k in range(len(labels))}})
                    n_predictions += len(test)
                    return p
                q0 = fit_write(plan["train0"], "zero")
                for target in plan["targets"]:
                    cid, label = target["target"], target["label"]
                    pe = fit_write(target["exposure"], "exposure", cid, label)
                    ph = fit_write(target["sham"], "sham", cid, label)
                    panels = {"own_target": tc == cid,
                              "all_unexposed_probe": (plan["panels"] == "probe") & (tc != cid),
                              "sentinel": plan["panels"] == "sentinel"}
                    for scheme in SCHEMES:
                        if scheme == SCHEMES[1] and label not in plan["common"]:
                            continue
                        for panel, raw_mask in panels.items():
                            mask = raw_mask if scheme == SCHEMES[0] else raw_mask & common_eval
                            contrast_rows.append({**key, "target_component": cid, "target_label": label, "panel": panel,
                                                  "weighting": scheme, "n_evaluation": int(mask.sum()), "n_evaluation_classes": len(np.unique(y[mask])),
                                                  **pair_metrics(y, q0, pe, ph, mask)})
            print(f"{args.dataset}: single-source split {rep+1}/{N_SPLITS}; {len(fits)} model conditions; {time.perf_counter()-started:.1f}s", flush=True)
    singles = summarize_targets(contrast_rows, out)
    compare_saved_joint(args, singles, out)
    pd.DataFrame(fits).to_csv(out / "fit_log.csv", index=False)
    base.dump(out / "warnings.json", dict(count=len(allwarnings), items=allwarnings))
    for item in freeze["inputs"].values():
        base.require(base.sha(item["path"]) == item["sha256"], "Frozen input/code changed during execution")
    base.dump(out / "completion.json", dict(status="COMPLETED_POST_HOC_SINGLE_SOURCE_CONTROL", dataset=args.dataset,
                                           protocol_sha256=base.sha(freeze_path), independent_review_sha256=base.sha(args.review),
                                           targets=sum(len(p["targets"]) for p in plans), model_conditions=len(fits),
                                           logistic_fits=sum(not r["model"].endswith("cosine_1nn") for r in fits),
                                           predictions=n_predictions, warnings=len(allwarnings), elapsed_seconds=time.perf_counter()-started,
                                           scientific_submission_gate="HOLD_UNTIL_INDEPENDENT_RESULT_AND_MANUSCRIPT_REVIEW"))
    pd.DataFrame([dict(path=p.name, bytes=p.stat().st_size, sha256=base.sha(p)) for p in sorted(out.iterdir()) if p.is_file()]).to_csv(out / "artifact_manifest.csv", index=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=("onion", "potato"), required=True)
    ap.add_argument("--mode", choices=("prepare", "run"), required=True)
    for name in ("manifest", "resnet-features", "colour-features", "membership", "joint-predictions", "dino-features", "output", "review"):
        ap.add_argument("--" + name, type=Path)
    args = ap.parse_args()
    for name, path in defaults(args.dataset).items():
        if getattr(args, name) is None:
            setattr(args, name, path)
    args.healthy_label = "healthy"
    frame, matrices, labels, verified_source, feature_info = base.load_inputs(args)
    base.require(verified_source, "Audited source IDs required")
    if args.dino_features.exists():
        matrices["dinov2_logit"], feature_info["dinov2_logit"] = load_dino(args.dino_features, frame)
    plans, audit, members, targets, common = build_plans(frame, labels, args.membership)
    if args.mode == "prepare":
        prepare(args, frame, matrices, labels, feature_info, plans, audit, members, targets, common)
    else:
        base.require("dinov2_logit" in matrices, "DINO features are unavailable; no model-dropping fallback")
        run(args, frame, matrices, labels, plans)


if __name__ == "__main__":
    main()
