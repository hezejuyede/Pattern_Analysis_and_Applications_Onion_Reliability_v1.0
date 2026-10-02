"""Generate Figure 5 from released regional metrics and source-label counts.

Only executed result tables are used. No model is fitted and no threshold is
selected here. Plot-source CSVs, input hashes and format metadata are retained.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

ONION = Path(__file__).resolve().parents[2]
OUT = ONION / "manuscript/publication_figures"
SOURCES = OUT / "source_data"
RESULTS = ONION / "results/regional_robustness_v1"
REGIONS = ["Centre-Ouest", "Centre-Sud", "Plateau-Central"]
MODELS = ["prior_prevalence", "color_shortcut_logit", "handcrafted_logit", "resnet18_logit"]
MODEL_LABELS = ["Prevalence", "Colour", "Handcrafted", "ResNet18"]
MODEL_COLOURS = ["#777777", "#D55E00", "#CC79A7", "#0072B2"]
LABELS = ["healthy", "Alternaria", "Fusarium", "virosis"]
LABEL_COLOURS = ["#009E73", "#E69F00", "#56B4E9", "#CC79A7"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extract_sources(results: Path) -> tuple[Path, Path, dict]:
    """Create plot-only extracts from complete executed tables without selection."""
    interval_path = results / "intervals.csv"
    support_path = results / "class_support.csv"
    intervals = pd.read_csv(interval_path)
    support = pd.read_csv(support_path)
    metrics = intervals[
        intervals.variant.eq("region_day_disjoint")
        & intervals.metric.eq("balanced_accuracy")
        & intervals.region.isin(REGIONS + ["equal_region_mean"])
    ][["region", "model", "estimate", "lower", "upper"]].rename(
        columns={"lower": "ci_low", "upper": "ci_high"})
    metrics["region"] = metrics.region.replace({"equal_region_mean": "Equal-region mean"})
    counts = support[
        support.variant.eq("region_day_disjoint") & support.role.isin(["train", "test"])
    ][["region", "role", "original_label", "n"]].rename(
        columns={"role": "split", "original_label": "label"})
    mapping = {"Healthy_leaf": "healthy", "Alternaria_D": "Alternaria",
               "Fusarium-D": "Fusarium", "Virosis-D": "virosis"}
    if not set(counts.label).issubset(mapping):
        raise ValueError("Unexpected original archive labels")
    counts["label"] = counts.label.map(mapping)
    SOURCES.mkdir(parents=True, exist_ok=True)
    metric_out = SOURCES / "Fig5_regional_balanced_accuracy.csv"
    counts_out = SOURCES / "Fig5_region_label_support.csv"
    metrics.to_csv(metric_out, index=False)
    counts.to_csv(counts_out, index=False)
    upstream = {str(p.resolve()): sha256(p) for p in (interval_path, support_path)}
    return metric_out, counts_out, upstream


def plot(metrics: pd.DataFrame, counts: pd.DataFrame) -> dict:
    expected = {(r, m) for r in REGIONS + ["Equal-region mean"] for m in MODELS}
    actual = set(zip(metrics.region, metrics.model))
    if actual != expected or len(metrics) != len(expected):
        raise ValueError("Figure 5 requires exactly four models and all three regions plus their equal-weight mean")
    vals = metrics[["estimate", "ci_low", "ci_high"]].to_numpy(float)
    if not np.isfinite(vals).all() or not ((vals >= 0) & (vals <= 1)).all():
        raise ValueError("Invalid balanced-accuracy point or interval")
    if (metrics.ci_low > metrics.ci_high).any():
        raise ValueError("Reversed intervals")
    if set(counts.region) != set(REGIONS) or set(counts.split) != {"train", "test"}:
        raise ValueError("Missing regional train/test support")
    if set(counts.label) - set(LABELS) or (counts.n < 0).any():
        raise ValueError("Invalid source-label support")

    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 8.5, "axes.labelsize": 8.5, "axes.titlesize": 8.5,
        "xtick.labelsize": 8.0, "ytick.labelsize": 8.0, "legend.fontsize": 8.0,
        "axes.linewidth": 0.6, "lines.linewidth": 1.0,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "path",
        "savefig.facecolor": "white", "figure.facecolor": "white",
    })
    fig = plt.figure(figsize=(178 / 25.4, 124 / 25.4))
    grid = fig.add_gridspec(2, 3, height_ratios=[2.5, 1.0], left=.10,
                            right=.985, bottom=.145, top=.89, hspace=.87, wspace=.52)
    ax = fig.add_subplot(grid[0, :])
    ax.text(-.10, 1.05, "a", transform=ax.transAxes, fontweight="bold", fontsize=10)
    order = REGIONS + ["Equal-region mean"]
    for mi, (model, label, colour) in enumerate(zip(MODELS, MODEL_LABELS, MODEL_COLOURS)):
        rows = metrics[metrics.model.eq(model)].set_index("region").loc[order]
        xx = np.arange(4) + (mi - 1.5) * .14
        # Draw intervals independently of the estimate; a percentile interval
        # need not mathematically contain the full-sample point estimate.
        ax.vlines(xx, rows.ci_low, rows.ci_high, color=colour, linewidth=.95)
        ax.hlines(rows.ci_low, xx-.025, xx+.025, color=colour, linewidth=.95)
        ax.hlines(rows.ci_high, xx-.025, xx+.025, color=colour, linewidth=.95)
        ax.scatter(xx, rows.estimate, s=21, marker=["s", "^", "D", "o"][mi],
                   c=colour, edgecolors="white", linewidths=.35, label=label, zorder=4)
    ax.axhline(.5, color="#AAAAAA", linestyle="--", linewidth=.6, zorder=0)
    ax.axvline(2.5, color="#DDDDDD", linewidth=.7, zorder=0)
    ax.set_xticks(range(4), ["Centre-Ouest", "Centre-Sud", "Plateau-Central", "Equal-region\nmean"])
    ax.set_xlim(-.48, 3.48)
    low = min(.45, float(metrics.ci_low.min())-.04)
    ax.set_ylim(max(0, low), 1.025)
    ax.set_ylabel("Balanced accuracy")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#EEEEEE", linewidth=.5, zorder=0)
    ax.legend(loc="upper center", bbox_to_anchor=(.5, 1.30), ncol=4,
              frameon=False, handletextpad=.35, columnspacing=1.35)

    legend_handles = []
    for ri, region in enumerate(REGIONS):
        bx = fig.add_subplot(grid[1, ri])
        if ri == 0:
            bx.text(-.32, 1.38, "b", transform=bx.transAxes, fontweight="bold", fontsize=10)
        totals = []
        for yi, split in enumerate(["train", "test"]):
            row = counts[counts.region.eq(region) & counts.split.eq(split)].set_index("label").n
            row = row.reindex(LABELS, fill_value=0)
            total = int(row.sum())
            if total <= 0:
                raise ValueError("Empty regional support bar")
            totals.append(total)
            left = 0.0
            for li, (label, colour) in enumerate(zip(LABELS, LABEL_COLOURS)):
                width = float(row[label]) / total
                bar = bx.barh(yi, width, left=left, height=.52, color=colour,
                             edgecolor="white", linewidth=.4)
                if ri == 0 and yi == 0:
                    legend_handles.append(bar[0])
                left += width
        bx.set_yticks([0, 1], [f"Train {totals[0]}", f"Test {totals[1]}"])
        bx.invert_yaxis()
        bx.set_title(region, pad=6)
        bx.set_xlim(0, 1)
        bx.set_xticks([0, .5, 1], ["0", "50", "100"])
        bx.set_xlabel("Image share (%)", labelpad=2)
        bx.spines[["top", "right", "left"]].set_visible(False)
        bx.tick_params(axis="y", length=0, pad=3)
    fig.legend(legend_handles, ["Healthy", "Alternaria", "Fusarium", "Virosis"],
               loc="lower center", bbox_to_anchor=(.53, .002), ncol=4, frameon=False,
               handlelength=1.25, columnspacing=1.25)

    OUT.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix in ("pdf", "eps", "svg", "png", "tif"):
        path = OUT / f"Fig5.{suffix}"
        kwargs = {"dpi": 600}
        if suffix == "tif":
            kwargs["pil_kwargs"] = {"compression": "tiff_lzw"}
        fig.savefig(path, **kwargs)
        paths.append(path)
    plt.close(fig)
    for suffix in ("png", "tif"):
        path = OUT / f"Fig5.{suffix}"
        with Image.open(path) as image:
            converted = image.convert("RGB")
        if suffix == "tif":
            converted.save(path, compression="tiff_lzw", dpi=(600, 600))
        else:
            converted.save(path, optimize=True, dpi=(600, 600))
    eps = OUT / "Fig5.eps"
    if not eps.read_bytes().rstrip().endswith(b"%%EOF"):
        with eps.open("ab") as stream:
            stream.write(b"\n%%EOF\n")
    return {p.name: {"bytes": p.stat().st_size, "sha256": sha256(p)} for p in paths}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=RESULTS,
                        help="Executed result directory; regenerates the plotting source extracts")
    parser.add_argument("--metrics", type=Path, default=SOURCES / "Fig5_regional_balanced_accuracy.csv")
    parser.add_argument("--counts", type=Path, default=SOURCES / "Fig5_region_label_support.csv")
    parser.add_argument("--use-extracted", action="store_true",
                        help="Read the explicitly supplied plot-source tables without extracting again")
    args = parser.parse_args()
    upstream = {}
    if not args.use_extracted:
        args.metrics, args.counts, upstream = extract_sources(args.results)
    metrics = pd.read_csv(args.metrics)
    counts = pd.read_csv(args.counts)
    SOURCES.mkdir(parents=True, exist_ok=True)
    outputs = plot(metrics, counts)
    manifest = {"figure": "Fig5", "dimensions_mm": [178, 124], "dpi_raster": 600,
                "inputs": {str(p.resolve()): sha256(p) for p in (args.metrics, args.counts)},
                "executed_result_inputs": upstream,
                "outputs": outputs, "inference": "Post hoc matched-region analysis; not new plants or independent pathology validation"}
    (SOURCES / "Fig5_BUILD_MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"figure": "Fig5", "formats": list(outputs)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
