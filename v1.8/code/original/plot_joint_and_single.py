"""Scientific figures from saved split results; no fitting or result selection."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from PIL import Image

OUT = Path(__file__).resolve().parent
D18 = OUT.parent
D17 = D18.parent / "Onion_Deep_Revision_20261003_v17"
MODELS = ["resnet_logit", "dinov2_logit", "colour_logit", "resnet_cosine_1nn", "dinov2_cosine_1nn"]
LABELS = ["ResNet18 + LR", "DINOv2 + LR", "Colour + LR", "ResNet18 cosine 1NN", "DINOv2 cosine 1NN"]
SCHEMES = ["all_original_classes", "common_classes_n_ge_30"]
plt.rcParams.update({"font.family":"Arial", "font.size":8, "axes.labelsize":8,
                     "axes.titlesize":8.5, "xtick.labelsize":7.5, "ytick.labelsize":8,
                     "pdf.fonttype":42, "ps.fonttype":42, "axes.linewidth":.65,
                     "savefig.facecolor":"white", "axes.unicode_minus":False})


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def joint_source():
    rows, input_files = [], []
    for dataset in ("onion", "potato"):
        old = D17 / ("matched_replacement_onion" if dataset == "onion" else "independent_task/potato/matched_replacement_experiment")
        new = D18 / "joint_dinov2" / dataset
        paths = [old / "split_means.csv", new / "split_means.csv"]
        frame = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
        input_files.extend(paths)
        frame = frame[frame.endpoint.eq("multiclass") & frame.model.isin(MODELS)]
        protocol = json.loads((D18 / "single_source" / dataset / "protocol_freeze.json").read_text(encoding="utf-8"))
        for panel, metric in (("selected_probe", "policy_BA_difference"), ("sentinel", "turnover_difference")):
            part = frame[frame.panel.eq(panel)]
            for scheme in SCHEMES:
                if scheme == SCHEMES[0]:
                    s = part[part.stratum.eq("all_classes")][["repetition", "seed", "model", metric]].copy()
                else:
                    c = part[part.stratum.isin(protocol["common_classes"])]
                    assert c.groupby(["repetition", "model"]).size().eq(len(protocol["common_classes"])).all()
                    s = c.groupby(["repetition", "seed", "model"], as_index=False)[metric].mean()
                for _, r in s.iterrows():
                    rows.append(dict(dataset=dataset, design="joint", panel=panel, metric=metric,
                                     weighting=scheme, repetition=r.repetition, seed=r.seed, model=r.model, value=r[metric]))
    return pd.DataFrame(rows), input_files


def single_source():
    rows, input_files = [], []
    for dataset in ("onion", "potato"):
        p = D18 / "single_source" / dataset / "split_means.csv"
        input_files.append(p)
        frame = pd.read_csv(p)
        for panel, metric in (("own_target", "policy_BA_difference"), ("sentinel", "turnover_difference")):
            s = frame[frame.panel.eq(panel)]
            for _, r in s.iterrows():
                rows.append(dict(dataset=dataset, design="single", panel=panel, metric=metric,
                                 weighting=r.weighting, repetition=r.repetition, seed=r.seed, model=r.model, value=r[metric]))
    return pd.DataFrame(rows), input_files


def render(data, design, name):
    summaries = []
    groupcols = ["dataset", "design", "panel", "metric", "weighting", "model"]
    for key, part in data.groupby(groupcols):
        x = part.value.to_numpy(float) * 100
        assert len(x) == (30 if design == "joint" else 10)
        lo, hi = (np.quantile(x, [.025, .975]) if design == "joint" else (x.min(), x.max()))
        summaries.append(dict(zip(groupcols, key)) | dict(mean_pp=x.mean(), lower_pp=lo, upper_pp=hi, split_count=len(x)))
    summary = pd.DataFrame(summaries)
    data.to_csv(OUT / (name + "_split_source.csv"), index=False)
    summary.to_csv(OUT / (name + "_plot_source.csv"), index=False)
    fig, axes = plt.subplots(2, 2, figsize=(174/25.4, 124/25.4))
    fig.subplots_adjust(left=.225, right=.98, bottom=.19, top=.92, wspace=.18, hspace=.45)
    colours = {SCHEMES[0]:"#173D59", SCHEMES[1]:"#AD5B16"}
    y = np.arange(len(MODELS))[::-1]
    panels = ["selected_probe" if design == "joint" else "own_target", "sentinel"]
    for row, panel in enumerate(panels):
        rowdata = summary[summary.panel.eq(panel)]
        lo, hi = min(0, rowdata.lower_pp.min()), max(0, rowdata.upper_pp.max())
        pad = max(1.5, (hi-lo)*.08)
        for col, dataset in enumerate(("onion", "potato")):
            ax = axes[row, col]
            part = rowdata[rowdata.dataset.eq(dataset)]
            ax.axvline(0, color="#6C747A", lw=.7, linestyle="--", zorder=0)
            for index, model in enumerate(MODELS):
                for scheme in SCHEMES:
                    # Potato has no excluded class; both summaries are identical.
                    if dataset == "potato" and scheme == SCHEMES[1]:
                        continue
                    r = part[part.model.eq(model) & part.weighting.eq(scheme)].iloc[0]
                    yy = y[index] + ((.12 if scheme == SCHEMES[0] else -.12) if dataset == "onion" else 0)
                    ax.hlines(yy, r.lower_pp, r.upper_pp, color=colours[scheme], linewidth=1.3)
                    ax.plot(r.mean_pp, yy, marker="o" if scheme == SCHEMES[0] else "s", markersize=4.1,
                            markerfacecolor=colours[scheme] if scheme == SCHEMES[0] else "white",
                            markeredgecolor=colours[scheme], markeredgewidth=1, linestyle="none")
            ax.set_xlim(lo-pad, hi+pad)
            ax.set_ylim(-.55, 4.65)
            ax.set_yticks(y, LABELS if col == 0 else [""]*len(LABELS))
            ax.tick_params(axis="y", length=0, pad=7)
            ax.tick_params(axis="x", length=3, width=.6)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
            for side in ("top", "right", "left"):
                ax.spines[side].set_visible(False)
            ax.spines["bottom"].set_color("#727B82")
            label = "Local contrast" if row == 0 else "Sentinel turnover contrast"
            ax.set_title(f"({chr(97 + row*2 + col)}) {dataset.title()}: {label.lower()}", loc="left", pad=8)
            ax.set_xlabel("Exposure - sham (percentage points)", labelpad=5)
    handles = [Line2D([0],[0], color=colours[SCHEMES[0]], marker="o", markersize=4, lw=1.3, label="All original classes"),
               Line2D([0],[0], color=colours[SCHEMES[1]], marker="s", markerfacecolor="white", markersize=4, lw=1.3, label="Classes with at least 30 sources")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.54,.035), ncol=2, frameon=False, handlelength=1.8, columnspacing=1.3)
    for ext in ("pdf", "eps", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=600)
    plt.close(fig)
    # Journal figure RGB convention; preserve raster dimensions and resolution.
    p = OUT / f"{name}.png"
    with Image.open(p) as im:
        if im.mode != "RGB":
            im.convert("RGB").save(p, dpi=(600,600))
    return summary


def main():
    joint, jp = joint_source()
    single, sp = single_source()
    names = [(joint, "joint", "Fig_2_Joint_Replacement"), (single, "single", "Fig_3_Single_Source_Replacement")]
    for data, design, name in names:
        render(data, design, name)
    captions = """Fig. 2 Joint replacement contrasts for all five fixed models. Local contrast is selected-probe exposure-minus-sham balanced accuracy; sentinel contrast is the difference in correctness turnover, both relative to the same zero baseline. Markers are means of 30 split means; bars are their 2.5th-97.5th percentiles and describe design variation, not biological confidence intervals. Orange results reweight the unchanged multiclass predictions to original classes with at least 30 source units. All potato classes meet this rule, so their identical second series is omitted. The two tasks retain different provenance and acquisition conditions.\n\nFig. 3 Single-source replacement contrasts at the common zero-exposure training background. Only the target source versus its mapped unrelated same-class source differs between each exposure/sham pair. Own-target contrasts are first averaged within original target class and then equally across target classes. Sentinel metrics first balance evaluation classes within target, then average within and across target classes. Markers are means of the first ten split means; bars are the complete ten-split minimum-maximum range, not confidence intervals. The orange sensitivity filters both target and evaluation classes using the prespecified source-count rule without retraining. All potato classes meet this rule. This contrast is conditional on the fixed background and does not establish biological independence or field diagnostic validity.\n"""
    (OUT / "FIGURE_CAPTIONS_EN.txt").write_text(captions, encoding="utf-8")
    qa = dict(status="GENERATED_AWAITING_ACTUAL_PDF_VISUAL_REVIEW", width_mm=174, height_mm=124,
              plot_code_sha256=sha(__file__), inputs=[dict(path=str(p), sha256=sha(p)) for p in jp+sp],
              outputs=[dict(path=p.name, bytes=p.stat().st_size, sha256=sha(p)) for p in sorted(OUT.glob("Fig_*"))],
              numerical_scope="Plot sources are derived from all retained saved split means; no models or outcomes selected. Joint intervals and single ranges have explicitly different definitions.")
    (OUT / "FIGURE_BUILD_QA.json").write_text(json.dumps(qa, indent=2)+"\n", encoding="utf-8")
    print("Created two figures with full model coverage and saved exact plot sources.")


if __name__ == "__main__":
    main()
