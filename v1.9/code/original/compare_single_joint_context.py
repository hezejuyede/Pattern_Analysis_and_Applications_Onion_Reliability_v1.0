"""Compare single and joint E/H for identical targets and first ten seeds."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

D18 = Path(__file__).resolve().parent.parent
D17 = D18.parent / "Onion_Deep_Revision_20261003_v17"
OUT = D18 / "context_sensitivity"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024*1024), b""):
            h.update(b)
    return h.hexdigest()


def collect(path, nclass, single=False):
    probs = [f"p{i}" for i in range(nclass)]
    cols = ["repetition", "seed", "model", "endpoint", "component_id", "row_idx", "label", "truth", "predicted"] + probs
    cols += ["policy"] if single else ["condition", "selected"]
    zero, chosen = [], []
    for chunk in pd.read_csv(path, usecols=cols, dtype={"row_idx":str}, chunksize=200000):
        part = chunk[chunk.repetition.lt(10) & chunk.endpoint.eq("multiclass")].copy()
        z = part.policy.eq("zero") if single else part.condition.eq("zero")
        zero.append(part[z][cols[:9]+probs])
        if not single:
            part = part[part.selected.eq(1)].copy()
            if len(part):
                names = part.condition.str.split("_", expand=True)
                part["policy"], part["allocation_pair"], part["arm"] = names[0], names[1].astype(int), names[2]
                part["correct"] = part.predicted.eq(part.truth).astype(int)
                chosen.append(part)
    return pd.concat(zero, ignore_index=True), None if single else pd.concat(chosen, ignore_index=True)


def main():
    OUT.mkdir(exist_ok=True)
    allrows, parity, sources = [], [], []
    for dataset in ("onion", "potato"):
        single = D18 / "single_source" / dataset
        old = D17 / ("matched_replacement_onion" if dataset == "onion" else "independent_task/potato/matched_replacement_experiment")
        new = D18 / "joint_dinov2" / dataset
        paths = [single / "predictions.csv.gz", old / "predictions.csv.gz", new / "predictions.csv.gz"]
        for folder in (single, old, new):
            assert (folder / "completion.json").is_file(), f"Incomplete: {folder}"
        freeze = json.loads((single / "protocol_freeze.json").read_text(encoding="utf-8"))
        k = len(freeze["class_order"])
        sq0, _ = collect(paths[0], k, single=True)
        oq0, oj = collect(paths[1], k)
        nq0, nj = collect(paths[2], k)
        jq0 = pd.concat([oq0, nq0], ignore_index=True)
        qkeys = ["repetition", "seed", "model", "component_id", "row_idx", "label", "truth"]
        z = sq0.merge(jq0, on=qkeys, suffixes=("_single", "_joint"), validate="one_to_one", how="outer", indicator=True)
        assert z._merge.eq("both").all() and len(z) == len(sq0) == len(jq0)
        assert z.predicted_single.eq(z.predicted_joint).all()
        for model, part in z.groupby("model"):
            err = max(float(np.max(np.abs(part[f"p{i}_single"] - part[f"p{i}_joint"]))) for i in range(k))
            assert err < 1e-12
            parity.append(dict(dataset=dataset, model=model, zero_rows=len(part), decision_mismatches=0,
                               max_probability_difference=err, status="PASS"))
        j = pd.concat([oj, nj], ignore_index=True)
        keys = ["repetition", "seed", "model", "component_id", "label", "allocation_pair", "arm"]
        paired = j.pivot(index=keys, columns="policy", values="correct").reset_index()
        assert paired[["exposure", "sham"]].notna().all().all()
        paired["joint_E_minus_H"] = paired.exposure - paired.sham
        keys = ["repetition", "seed", "model", "component_id", "label"]
        joint = paired.groupby(keys).agg(joint_E_minus_H=("joint_E_minus_H", "mean"), n_joint_pairs=("joint_E_minus_H", "size")).reset_index()
        assert joint.n_joint_pairs.eq(5).all()
        joint = joint.rename(columns={"component_id":"target_component", "label":"target_label"})
        own = pd.read_csv(single / "target_panel_contrasts.csv")
        own = own[own.panel.eq("own_target") & own.weighting.eq("all_original_classes")]
        cols = ["repetition", "seed", "model", "target_component", "target_label"]
        m = own[cols + ["policy_BA_difference"]].merge(joint, on=cols, validate="one_to_one", how="outer", indicator=True)
        assert m._merge.eq("both").all() and len(m) == len(joint) == len(own)
        m = m.drop(columns="_merge").rename(columns={"policy_BA_difference":"single_E_minus_H"})
        m["single_minus_joint"] = m.single_E_minus_H - m.joint_E_minus_H
        m["dataset"] = dataset
        m["common_class"] = m.target_label.isin(freeze["common_classes"])
        allrows.append(m)
        for p in paths + [single / "target_panel_contrasts.csv"]:
            sources.append(dict(dataset=dataset, path=str(p), sha256=sha(p)))
    data = pd.concat(allrows, ignore_index=True)
    data.to_csv(OUT / "target_contrasts.csv", index=False)
    values = ["single_E_minus_H", "joint_E_minus_H", "single_minus_joint"]
    classrows, splitrows = [], []
    classkeys = ["dataset", "repetition", "seed", "model", "target_label"]
    for scheme in ("all_original_classes", "common_classes_n_ge_30"):
        part = data if scheme == "all_original_classes" else data[data.common_class]
        c = part.groupby(classkeys, as_index=False)[values].mean()
        c["weighting"] = scheme
        classrows.append(c)
        s = c.groupby(["dataset", "repetition", "seed", "model", "weighting"], as_index=False)[values].mean()
        splitrows.append(s)
    pd.concat(classrows).to_csv(OUT / "class_split_means.csv", index=False)
    splits = pd.concat(splitrows)
    splits.to_csv(OUT / "split_means.csv", index=False)
    rows = []
    for key, part in splits.groupby(["dataset", "model", "weighting"]):
        assert len(part) == 10
        for metric in values:
            x = part[metric].to_numpy(float)
            rows.append(dict(zip(["dataset", "model", "weighting"], key)) | dict(metric=metric, mean=x.mean(),
                        split_min=x.min(), split_max=x.max(), split_sd=x.std(ddof=1), split_count=10))
    pd.DataFrame(rows).to_csv(OUT / "summary.csv", index=False)
    qa = dict(status="PASS", matched_targets=len(data), q0_checks=parity, sources=sources,
              code_sha256=sha(__file__),
              interpretation="Same targets and seeds; joint contrast averages their five selected E/H allocations. Context and replacement extent change together. This is not an identified biological or algorithmic mechanism; all split summaries are design variation.")
    (OUT / "Q0_PARITY_QA.json").write_text(json.dumps(qa, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dict(status="PASS", matched_targets=len(data), maximum_q0_probability_difference=max(p["max_probability_difference"] for p in parity))))


if __name__ == "__main__":
    main()
