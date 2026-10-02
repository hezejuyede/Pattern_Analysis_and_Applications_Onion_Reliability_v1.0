"""Generate publication figures from the frozen onion-study result tables.

The script never reads model objects or raw images.  Every plotted value comes
from a released CSV/JSON artifact, and the exact plot-source extracts are saved
next to the figures.  Output is written as PDF, EPS and 600-dpi RGB TIFF.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "reliability_benchmark_v1"
ABLATION = ROOT / "results" / "cold_augmented_leakage_ablation_v1"
INTERVENTION = ROOT / "results" / "matched_sibling_intervention_v1"
LINEAGE = ROOT / "audit" / "cold_augmented_lineage"
OUT = ROOT / "manuscript" / "publication_figures"
SOURCE = OUT / "source_data"

MODEL_ORDER = [
    "prior_prevalence",
    "color_shortcut_logit",
    "handcrafted_logit",
    "resnet18_logit",
    "efficientnet_b0_logit",
    "convnext_tiny_logit",
    "swin_t_logit",
]
MODEL_LABEL = {
    "prior_prevalence": "Prevalence",
    "color_shortcut_logit": "Colour shortcut",
    "handcrafted_logit": "Handcrafted",
    "resnet18_logit": "ResNet18",
    "efficientnet_b0_logit": "EfficientNet-B0",
    "convnext_tiny_logit": "ConvNeXt-Tiny",
    "swin_t_logit": "Swin-T",
}
BACKBONES = [
    "resnet18_logit",
    "efficientnet_b0_logit",
    "convnext_tiny_logit",
    "swin_t_logit",
]
PALETTE = {
    "prior_prevalence": "#777777",
    "color_shortcut_logit": "#D55E00",
    "handcrafted_logit": "#CC79A7",
    "resnet18_logit": "#0072B2",
    "efficientnet_b0_logit": "#009E73",
    "convnext_tiny_logit": "#E69F00",
    "swin_t_logit": "#6A3D9A",
}


def configure() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": 8.0,
            "axes.labelsize": 8.0,
            "axes.titlesize": 8.0,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "legend.fontsize": 7.0,
            "axes.linewidth": 0.6,
            "lines.linewidth": 1.1,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.10, 1.04, label, transform=ax.transAxes, fontweight="bold", va="bottom", ha="left", fontsize=9)


def save_all(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    SOURCE.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.03)
    eps = OUT / f"{stem}.eps"
    fig.savefig(eps, bbox_inches="tight", pad_inches=0.03)
    # Matplotlib's PS backend can omit the optional DSC end marker.  Appending
    # it makes the EPS self-terminating for stricter publisher preflight tools.
    with eps.open("rb+") as stream:
        stream.seek(0, 2)
        if stream.tell() > 0:
            stream.seek(-1, 2)
            last = stream.read(1)
        else:
            last = b""
        stream.seek(0, 2)
        stream.write((b"" if last in (b"\n", b"\r") else b"\n") + b"%%EOF\n")
    fig.savefig(
        OUT / f"{stem}.tif",
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.03,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)
    # Matplotlib writes an RGBA TIFF even on a white figure canvas.  Springer
    # requests 8-bit RGB; remove the unused alpha channel deterministically.
    tif = OUT / f"{stem}.tif"
    with Image.open(tif) as image:
        rgb = image.convert("RGB")
        rgb.save(tif, compression="tiff_lzw", dpi=(600, 600))


def rounded_box(ax: plt.Axes, xy: tuple[float, float], width: float, height: float, text: str, face: str, fontsize: float = 7.0) -> None:
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.018,rounding_size=0.018",
        linewidth=0.8,
        edgecolor="#333333",
        facecolor=face,
    )
    ax.add_patch(box)
    ax.text(xy[0] + width / 2, xy[1] + height / 2, text, ha="center", va="center", fontsize=fontsize, linespacing=1.2)


def arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    # Keep the arrowhead outside the target box.  Without an explicit end
    # shrink, short connectors can visually merge with the first text glyph
    # after tight PDF/EPS cropping.
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=8,
            linewidth=0.8,
            color="#444444",
            shrinkA=3.5,
            shrinkB=6.0,
        )
    )


def _intervention_ready() -> pd.DataFrame:
    status = json.loads((INTERVENTION / "run_metadata.json").read_text(encoding="utf-8"))
    if status["status"] != "PASS_POST_HOC_DIAGNOSTIC":
        raise ValueError("Intervention analysis has not completed successfully")
    checks = pd.read_csv(INTERVENTION / "execution_checks.csv")
    if len(checks) != 90 or not checks["status"].eq("PASS").all():
        raise ValueError("Expected all 90 intervention membership checks")
    for key, expected in {
        "train_images": 1142, "train_families": 571, "test_anchors": 245,
        "probe_test_families": 163, "sentinel_test_families": 82,
        "exposed_sentinel_families": 0,
    }.items():
        if not checks[key].eq(expected).all():
            raise ValueError(f"Unexpected frozen intervention count: {key}")
    return checks


def figure_1() -> None:
    checks = _intervention_ready()
    counts = pd.read_csv(ABLATION / "audit" / "effective_sample_size_by_label.csv")
    counts.to_csv(SOURCE / "Fig1_family_counts.csv", index=False)
    checks.to_csv(SOURCE / "Fig1_intervention_checks.csv", index=False)
    # The four-class intervention and binary source-transfer test are distinct
    # branches, not successive partitions of one nested experimental cohort.
    fig = plt.figure(figsize=(7.05, 5.60), constrained_layout=True)
    gs = fig.add_gridspec(3, 1, height_ratios=[1.18, 1.18, 1.70])
    lefts = [0.025, 0.275, 0.525, 0.775]
    width = 0.20
    rows = [
        (
            "(a)", "Controlled sibling exposure: four archive classes",
            ["COLD augmented\n4,502 files\n816 filename families",
             "Fixed family partitions\nProbe P: 163\nSentinel S: 82\nTraining D: 571",
             "Same-class substitution\nExposure: 0, 0.5, 1\n571 training families\n1,142 training files",
             "Fixed test anchors\nProbe P and sentinel S\nS never enters training\n30 prespecified seeds"],
            ["#E5EDF5", "#F1E9F4", "#F8EACF", "#E3F0E8"],
            "Within each seed: identical test images, class counts and training budget; two views per training family",
        ),
        (
            "(b)", "Locked source transfer: healthy versus visibly affected foliage",
            ["TOM2024 development\n1,643 unique images\n103 acquisition days",
             "Day-grouped fitting\n5 outer folds\n4 inner folds\nTraining-only scaling",
             "TOM-only decisions\nCross-fitted calibration\nOperating threshold\nFrozen before transfer",
             "Raw COLD external\n813 unique images\nRanking, calibration\nand threshold transfer"],
            ["#DCEAF7", "#E8F2E4", "#FFF1D6", "#FBE4DE"],
            "Separate analysis branch; filename families and acquisition days are not verified plant or plot identities",
        ),
    ]
    for row, (letter, heading, boxes, colours, note) in enumerate(rows):
        ax = fig.add_subplot(gs[row, 0])
        ax.set(xlim=(0, 1), ylim=(0, 1))
        ax.axis("off")
        ax.text(0.003, 0.98, letter, fontweight="bold", va="top", fontsize=9)
        ax.text(0.048, 0.98, heading, fontweight="bold", va="top", fontsize=8.0)
        for x, label, colour in zip(lefts, boxes, colours):
            rounded_box(ax, (x, 0.23), width, 0.53, label, colour, fontsize=8.0)
        for x in lefts[:-1]:
            arrow(ax, (x + width + 0.018, 0.495), (x + 0.25 - 0.018, 0.495))
        ax.text(0.50, 0.045, note, ha="center", va="bottom", fontsize=7.3, color="#444444")
    ax = fig.add_subplot(gs[2, 0])
    panel_label(ax, "(c)")
    label_map = {
        "healthy": "Healthy", "iris_yellow_virus": "IYSV",
        "purple_blotch": "Purple blotch",
        "stemphylium_colletotrichum_leaf_blight": "Pooled leaf blight",
    }
    counts = counts.assign(display=counts["label"].map(label_map))
    counts = counts.set_index("display").loc[["Healthy", "IYSV", "Pooled leaf blight", "Purple blotch"]].reset_index()
    y = np.arange(len(counts))
    h = 0.34
    ax.barh(y + h / 2, counts["files"], height=h, color="#7AA6C2", edgecolor="#333333", linewidth=0.4, label="Files")
    ax.barh(y - h / 2, counts["effective_families"], height=h, color="#F0B67F", hatch="//", edgecolor="#333333", linewidth=0.4, label="Filename families")
    for i, row in counts.iterrows():
        ax.text(row["files"] + 16, i + h / 2, f"{int(row['files']):,}", va="center", fontsize=7.2)
        ax.text(row["effective_families"] + 16, i - h / 2, f"{int(row['effective_families']):,}", va="center", fontsize=7.2)
    ax.set_yticks(y, counts["display"])
    ax.invert_yaxis()
    ax.set_xlabel("Number of distributed files or filename-defined families")
    ax.set_xlim(0, counts["files"].max() * 1.18)
    ax.grid(axis="x", color="#D9D9D9", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower right", ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    save_all(fig, "Fig1")


def _seed_summary(table: pd.DataFrame, value: str, keys: list[str]) -> pd.DataFrame:
    return table.groupby(keys, as_index=False)[value].agg(
        mean="mean", low=lambda x: float(np.quantile(x, 0.025)),
        high=lambda x: float(np.quantile(x, 0.975)), n="size"
    )


def _interval(ax: plt.Axes, x: np.ndarray, means: np.ndarray, low: np.ndarray,
              high: np.ndarray, *, color: str, marker: str, label: str,
              linestyle: str = "-") -> None:
    # Draw quantile endpoints directly: unlike a confidence interval for a
    # mean, a seed percentile range need not contain the arithmetic mean.
    ax.vlines(x, low, high, color=color, linewidth=0.85)
    ax.hlines(low, x - 0.012, x + 0.012, color=color, linewidth=0.85)
    ax.hlines(high, x - 0.012, x + 0.012, color=color, linewidth=0.85)
    ax.plot(x, means, color=color, marker=marker, markersize=4.0,
            markerfacecolor="white" if marker == "s" else color,
            linestyle=linestyle, label=label, zorder=3)


def figure_2() -> None:
    _intervention_ready()
    repeated = pd.read_csv(INTERVENTION / "repeated_metrics.csv")
    repeated = repeated[repeated.endpoint.eq("four_class")].copy()
    required = ["repetition", "seed", "dose", "feature_model", "evaluation_panel", "balanced_accuracy"]
    repeated = repeated[required]
    repeated["dose"] = repeated["dose"].astype(float)
    expected = pd.MultiIndex.from_product(
        [["resnet18", "color_shortcut"], ["probe", "sentinel"], [0.0, 0.5, 1.0], range(30)],
        names=["feature_model", "evaluation_panel", "dose", "repetition"],
    )
    observed = repeated.set_index(expected.names)
    if len(repeated) != 360 or observed.index.has_duplicates or not observed.index.sort_values().equals(expected.sort_values()):
        raise ValueError("Expected every feature, P/S panel, dose and seed exactly once")
    if not np.isfinite(repeated.balanced_accuracy).all():
        raise ValueError("Non-finite intervention metric")
    repeated.to_csv(SOURCE / "Fig2_controlled_balanced_accuracy_by_seed.csv", index=False)
    summary = _seed_summary(repeated, "balanced_accuracy", ["feature_model", "evaluation_panel", "dose"])
    summary.to_csv(SOURCE / "Fig2_controlled_balanced_accuracy_summary.csv", index=False)
    # Pair conditions by the actual repeated split; never subtract aggregate
    # quantiles or pretend repeated archive splits are biological replicates.
    wide = repeated.pivot(index=["feature_model", "evaluation_panel", "repetition"], columns="dose", values="balanced_accuracy")
    differences = pd.concat([
        (wide[dose] - wide[0.0]).rename("exposed_minus_clean").reset_index().assign(dose=dose)
        for dose in [0.5, 1.0]
    ], ignore_index=True)
    differences.to_csv(SOURCE / "Fig2_paired_changes_by_seed.csv", index=False)
    diff_summary = _seed_summary(differences, "exposed_minus_clean", ["feature_model", "evaluation_panel", "dose"])
    diff_summary.to_csv(SOURCE / "Fig2_paired_changes_summary.csv", index=False)
    localization = differences.pivot(index=["feature_model", "dose", "repetition"], columns="evaluation_panel", values="exposed_minus_clean").reset_index()
    localization["probe_minus_sentinel_change"] = localization.probe - localization.sentinel
    localization.to_csv(SOURCE / "Fig2_localization_by_seed.csv", index=False)
    local_summary = _seed_summary(localization, "probe_minus_sentinel_change", ["feature_model", "dose"])
    local_summary.to_csv(SOURCE / "Fig2_localization_summary.csv", index=False)
    # Independently check the new plot extracts against the analysis outputs.
    for path, computed, mean_name in [
        ("aggregate_metrics.csv", summary, "mean"),
        ("paired_summary.csv", diff_summary, "mean_exposed_minus_clean"),
        ("localization_summary.csv", local_summary, "mean_localization_contrast"),
    ]:
        released = pd.read_csv(INTERVENTION / path)
        released = released[released.endpoint.eq("four_class") & released.metric.eq("balanced_accuracy")]
        keys = ["feature_model", "dose"] + (["evaluation_panel"] if "evaluation_panel" in computed else [])
        check = computed.merge(released, on=keys, validate="one_to_one", suffixes=("_plot", "_released"))
        released_mean = "mean_released" if mean_name == "mean" else mean_name
        plot_mean = "mean_plot" if mean_name == "mean" else "mean"
        if len(check) != len(computed) or not np.allclose(check[plot_mean], check[released_mean], atol=2e-10, rtol=0):
            raise ValueError(f"Plot mean disagrees with released {path}")
        if not np.allclose(check.low, check.split_p025, atol=2e-10, rtol=0) or not np.allclose(check.high, check.split_p975, atol=2e-10, rtol=0):
            raise ValueError(f"Plot seed ranges disagree with released {path}")
    fig, axes = plt.subplots(2, 2, figsize=(7.05, 5.5), constrained_layout=True)
    panel_colours = {"probe": "#0072B2", "sentinel": "#D55E00"}
    panel_markers = {"probe": "o", "sentinel": "s"}
    panel_names = {"probe": "Probe P", "sentinel": "Unexposed S"}
    model_names = {"resnet18": "ResNet18", "color_shortcut": "Colour shortcut"}
    lo = max(0.0, float(summary.low.min()) - 0.035)
    hi = min(1.01, float(summary.high.max()) + 0.035)
    for ax, model, letter in zip(axes[0], ["resnet18", "color_shortcut"], ["(a)", "(b)"]):
        panel_label(ax, letter)
        for j, panel in enumerate(["probe", "sentinel"]):
            part = summary[summary.feature_model.eq(model) & summary.evaluation_panel.eq(panel)].sort_values("dose")
            _interval(ax, part.dose.to_numpy() + (j - 0.5) * 0.022,
                      part["mean"].to_numpy(), part.low.to_numpy(), part.high.to_numpy(),
                      color=panel_colours[panel], marker=panel_markers[panel], label=panel_names[panel],
                      linestyle="-" if panel == "probe" else "--")
        ax.set(xlim=(-0.09, 1.09), ylim=(lo, hi), xticks=[0, 0.5, 1],
               xlabel="Nominal fraction of P families exposed", ylabel="Four-class balanced accuracy")
        ax.text(0.02, 0.97, model_names[model], transform=ax.transAxes, va="top", fontweight="bold")
        if model == "resnet18":
            ax.legend(frameon=False, loc="lower right")
        else:
            ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.015, 0.905))
    ax = axes[1, 0]
    panel_label(ax, "(c)")
    names, positions = [], []
    for i, model in enumerate(["resnet18", "color_shortcut"]):
        for j, panel in enumerate(["probe", "sentinel"]):
            part = diff_summary[diff_summary.feature_model.eq(model) & diff_summary.evaluation_panel.eq(panel) & diff_summary.dose.eq(1.0)].iloc[0]
            y = i * 3 + j
            ax.hlines(y, part.low, part.high, color=panel_colours[panel], linewidth=1.1)
            ax.vlines([part.low, part.high], y - 0.12, y + 0.12, color=panel_colours[panel], linewidth=0.8)
            ax.plot(part["mean"], y, marker=panel_markers[panel], color=panel_colours[panel], markersize=4.6,
                    markerfacecolor="white" if panel == "sentinel" else panel_colours[panel])
            names.append(f"{model_names[model]} | {'P' if panel == 'probe' else 'S'}")
            positions.append(y)
    ax.axvline(0, color="#666666", linestyle="--", linewidth=0.8)
    ax.set(yticks=positions, yticklabels=names, ylim=(4.65, -0.65),
           xlabel="Paired BA change: full exposure minus clean")
    ax.text(0.98, 0.97, "Same test anchors", transform=ax.transAxes, va="top", ha="right", fontsize=7)
    ax = axes[1, 1]
    panel_label(ax, "(d)")
    for j, model in enumerate(["resnet18", "color_shortcut"]):
        part = local_summary[local_summary.feature_model.eq(model)].sort_values("dose")
        _interval(ax, part.dose.to_numpy() + (j - 0.5) * 0.020,
                  part["mean"].to_numpy(), part.low.to_numpy(), part.high.to_numpy(),
                  color=["#0072B2", "#009E73"][j], marker=["o", "s"][j], label=model_names[model])
    ax.axhline(0, color="#666666", linestyle="--", linewidth=0.8)
    ax.set(xlim=(0.40, 1.1), xticks=[0.5, 1], xlabel="Nominal fraction of P families exposed",
           ylabel="Paired BA change in P minus change in S")
    ax.legend(frameon=False, loc="upper left")
    for ax in axes.flat:
        ax.grid(axis="y" if ax is not axes[1, 0] else "x", color="#DEDEDE", linewidth=0.5)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    save_all(fig, "Fig2")

def metric_transport_panel(ax: plt.Axes, table: pd.DataFrame, metric: str, xlabel: str, show_y: bool) -> None:
    subset = table[table["metric"].eq(metric)].copy()
    y = np.arange(len(MODEL_ORDER))
    for i, model in enumerate(MODEL_ORDER):
        part = subset[subset["model"].eq(model)].set_index("analysis_set")
        vals = []
        for set_name, marker, face in [
            ("tom_nested_oof", "o", "white"),
            ("cold_locked_external", "s", PALETTE[model]),
        ]:
            row = part.loc[set_name]
            vals.append(float(row["estimate"]))
            ax.errorbar(
                float(row["estimate"]),
                i,
                xerr=[[float(row["estimate"]) - float(row["ci_low"])], [float(row["ci_high"]) - float(row["estimate"])]],
                fmt=marker,
                markersize=4.3,
                color=PALETTE[model],
                markerfacecolor=face,
                markeredgewidth=0.8,
                capsize=1.8,
                zorder=3,
            )
        ax.plot(vals, [i, i], color=PALETTE[model], linewidth=1.0, zorder=1)
    ax.set_yticks(y, [MODEL_LABEL[m] for m in MODEL_ORDER] if show_y else [""] * len(y))
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", color="#D9D9D9", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


def figure_3() -> None:
    metrics = pd.read_csv(RESULTS / "tables" / "binary_metrics_group_bootstrap.csv")
    source = metrics[
        metrics["analysis_set"].isin(["tom_nested_oof", "cold_locked_external"])
        & metrics["model"].isin(MODEL_ORDER)
        & metrics["metric"].isin(["balanced_accuracy", "auroc", "brier"])
    ].copy()
    source.to_csv(SOURCE / "Fig3_internal_external_metrics.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(7.05, 3.22), constrained_layout=True)
    metric_transport_panel(axes[0], source, "balanced_accuracy", "Balanced accuracy", True)
    metric_transport_panel(axes[1], source, "auroc", "AUROC", False)
    metric_transport_panel(axes[2], source, "brier", "Brier score (lower is better)", False)
    axes[0].set_xlim(0.39, 1.01)
    axes[1].set_xlim(0.39, 1.01)
    axes[2].set_xlim(0.0, 0.35)
    for ax, label in zip(axes, ["(a)", "(b)", "(c)"]):
        panel_label(ax, label)
    handles = [
        mpl.lines.Line2D([], [], marker="o", color="#333333", markerfacecolor="white", linestyle="none", label="TOM nested OOF"),
        mpl.lines.Line2D([], [], marker="s", color="#333333", markerfacecolor="#333333", linestyle="none", label="COLD external"),
    ]
    axes[1].legend(handles=handles, frameon=False, loc="lower left")
    save_all(fig, "Fig3")


def calibration_points(frame: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    edges = np.linspace(0, 1, bins + 1)
    frame = frame.copy()
    frame["bin"] = np.minimum(np.digitize(frame["calibrated_probability_disorder"], edges[1:-1], right=True), bins - 1)
    out = frame.groupby(["model", "bin"], as_index=False).agg(
        mean_probability=("calibrated_probability_disorder", "mean"),
        observed_fraction=("foliar_binary", "mean"),
        n=("foliar_binary", "size"),
    )
    return out


def calibration_panel(ax: plt.Axes, points: pd.DataFrame) -> None:
    ax.plot([0, 1], [0, 1], color="#777777", linestyle="--", linewidth=0.8)
    for model in BACKBONES:
        part = points[points["model"].eq(model)]
        ax.plot(
            part["mean_probability"],
            part["observed_fraction"],
            marker="o",
            markersize=2.8,
            color=PALETTE[model],
            label=MODEL_LABEL[model],
        )
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed disorder fraction")
    ax.grid(color="#E0E0E0", linewidth=0.45)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


def figure_4() -> None:
    tom = pd.read_csv(RESULTS / "predictions" / "tom_nested_oof.csv")
    cold = pd.read_csv(RESULTS / "predictions" / "cold_locked_external.csv")
    tom_points = calibration_points(tom[tom["model"].isin(BACKBONES)])
    cold_points = calibration_points(cold[cold["model"].isin(BACKBONES)])
    tom_points.assign(analysis_set="tom_nested_oof").to_csv(SOURCE / "Fig4_calibration_TOM.csv", index=False)
    cold_points.assign(analysis_set="cold_locked_external").to_csv(SOURCE / "Fig4_calibration_COLD.csv", index=False)

    per_label = pd.read_csv(RESULTS / "tables" / "per_dataset_label_performance.csv")
    per_label = per_label[
        per_label["analysis_set"].eq("COLD_locked_external") & per_label["model"].isin(BACKBONES)
    ].copy()
    per_label.to_csv(SOURCE / "Fig4_COLD_class_performance.csv", index=False)
    class_order = ["healthy", "iris_yellow_virus", "stemphylium_colletotrichum_leaf_blight", "purple_blotch"]
    heat = per_label.pivot(index="model", columns="dataset_label", values="correct_rate").loc[BACKBONES, class_order]

    risk = pd.read_csv(RESULTS / "tables" / "risk_coverage.csv")
    risk = risk[
        risk["analysis_set"].eq("cold_locked_external")
        & risk["model"].isin(BACKBONES)
        & risk["coverage"].isin(["0.25", "0.5", "0.75", "1.0"])
    ].copy()
    risk["coverage"] = risk["coverage"].astype(float)
    risk.to_csv(SOURCE / "Fig4_COLD_risk_coverage.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(7.05, 5.55), constrained_layout=True)
    calibration_panel(axes[0, 0], tom_points)
    calibration_panel(axes[0, 1], cold_points)
    for ax, label in zip(axes[0], ["(a)", "(b)"]):
        panel_label(ax, label)
    axes[0, 0].legend(frameon=False, loc="upper left")

    ax = axes[1, 0]
    panel_label(ax, "(c)")
    image = ax.imshow(heat.to_numpy(), vmin=0, vmax=1, cmap="viridis", aspect="auto")
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            value = heat.iloc[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", color="white" if value < 0.62 else "black", fontsize=6.8)
    ax.set_yticks(np.arange(len(BACKBONES)), [MODEL_LABEL[m] for m in BACKBONES])
    ax.set_xticks(np.arange(4), ["Healthy", "IYSV", "Pooled leaf\nblight", "Purple\nblotch"])
    ax.set_xlabel("COLD label")
    ax.set_ylabel("Backbone")
    cbar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.02)
    cbar.set_label("Correct fraction")

    ax = axes[1, 1]
    panel_label(ax, "(d)")
    for model in BACKBONES:
        part = risk[risk["model"].eq(model)].sort_values("coverage")
        ax.plot(part["coverage"], part["risk"], marker="o", markersize=3.2, color=PALETTE[model], label=MODEL_LABEL[model])
    ax.set_xlim(0.22, 1.03)
    ax.set_ylim(0, max(0.48, risk["risk"].max() * 1.08))
    ax.set_xlabel("Retained coverage")
    ax.set_ylabel("Error risk")
    ax.grid(color="#E0E0E0", linewidth=0.45)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper left")
    save_all(fig, "Fig4")


def figure_s1() -> None:
    ood = pd.read_csv(RESULTS / "tables" / "open_set_ood_metrics.csv")
    part = ood[
        ood["known_set"].eq("cold_locked_external")
        & ood["detector"].eq("source_5nn_cosine")
        & ood["model"].isin(MODEL_ORDER[1:])
    ].copy()
    part.to_csv(SOURCE / "FigS1_non_onion_OOD.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.65), constrained_layout=True)
    y = np.arange(len(part))
    labels = [MODEL_LABEL[m] for m in part["model"]]
    axes[0].barh(y, part["ood_auroc"], color=[PALETTE[m] for m in part["model"]], edgecolor="#333333", linewidth=0.4)
    axes[0].set_xlim(0.45, 1.0)
    axes[0].set_xlabel("OOD AUROC")
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    axes[1].barh(y, part["fpr_at_95pct_ood_tpr"], color=[PALETTE[m] for m in part["model"]], edgecolor="#333333", linewidth=0.4)
    axes[1].set_xlim(0, 1.0)
    axes[1].set_xlabel("FPR at 95% OOD TPR")
    axes[1].set_yticks(y, [""] * len(y))
    axes[1].invert_yaxis()
    for ax, label in zip(axes, ["(a)", "(b)"]):
        panel_label(ax, label)
        ax.grid(axis="x", color="#D9D9D9", linewidth=0.5)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    save_all(fig, "FigS1")


def write_manifest() -> None:
    manifest = {
        "source_root": str(ROOT),
        "figures": {
            "Fig1": "separate controlled sibling-exposure and locked source-transfer branches; COLD file-to-family counts",
            "Fig2": "30-seed fixed-budget sibling exposure: fixed probe/sentinel accuracy, paired changes and localization contrast",
            "Fig3": "TOM internal versus COLD external transport across all prespecified models",
            "Fig4": "calibration, class-specific correctness and selective risk for four backbones",
            "FigS1": "descriptive non-onion open-set results; not unknown-onion-disease validation",
        },
        "formats": ["PDF vector", "EPS vector", "TIFF RGB 600 dpi with LZW compression"],
        "generation_rule": "No raw image or model object is read; all marks derive from frozen CSV/JSON artifacts.",
        "Fig2_uncertainty": "2.5th and 97.5th percentiles across 30 prespecified seeds; not biological confidence intervals",
        "Fig2_primary_endpoint": "four original COLD archive classes; binary results are a separate sensitivity",
        "source_sha256": {
            str(p.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [INTERVENTION / "design.json", INTERVENTION / "repeated_metrics.csv", INTERVENTION / "execution_checks.csv",
                      *sorted(SOURCE.glob("Fig1*.csv")), *sorted(SOURCE.glob("Fig2_controlled*.csv")),
                      *sorted(SOURCE.glob("Fig2_paired*.csv")), *sorted(SOURCE.glob("Fig2_localization*.csv"))]
        },
    }
    (OUT / "FIGURE_MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures", nargs="+", choices=["Fig1", "Fig2", "Fig3", "Fig4", "FigS1"],
                        default=["Fig1", "Fig2", "Fig3", "Fig4", "FigS1"])
    args = parser.parse_args()
    configure()
    OUT.mkdir(parents=True, exist_ok=True)
    SOURCE.mkdir(parents=True, exist_ok=True)
    functions = {"Fig1": figure_1, "Fig2": figure_2, "Fig3": figure_3, "Fig4": figure_4, "FigS1": figure_s1}
    for name in args.figures:
        functions[name]()
    write_manifest()
    print(json.dumps({"output": str(OUT), "figures": args.figures}, indent=2))


if __name__ == "__main__":
    main()
