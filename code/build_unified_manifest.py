"""Build the frozen analysis manifest without copying image data."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def cold_rows() -> list[dict]:
    inventory = pd.read_csv(ROOT / "audit" / "cold_raw_hf" / "image_audit" / "image_inventory.csv")
    inventory = inventory.sort_values("relative_path").drop_duplicates("sha256", keep="first")
    rows = []
    for item in inventory.to_dict(orient="records"):
        label = str(item["class"])
        rows.append(
            {
                "dataset": "COLD_raw_India",
                "relative_path": f"data/cold_raw_hf/{item['relative_path']}",
                "source_group": f"cold_sha:{item['sha256']}",
                "original_label": label,
                "foliar_binary": 0 if label == "healthy" else 1,
                "foliar_binary_eligible": 1,
                "alternaria_binary": 1 if label == "purple_blotch" else (0 if label == "healthy" else ""),
                "alternaria_binary_eligible": int(label in {"healthy", "purple_blotch"}),
                "role": "development_or_external_source",
                "country": "India",
                "species": "Allium cepa",
            }
        )
    return rows


def tom_rows() -> list[dict]:
    manifest = pd.read_csv(ROOT / "audit" / "tom2024_category_a_onion" / "clean_source_representatives.csv")
    eligible = {"Healthy_leaf", "Alternaria_D", "Fusarium-D", "Virosis-D"}
    rows = []
    for item in manifest.itertuples(index=False):
        label = item.label
        rows.append(
            {
                "dataset": "TOM2024_A_BurkinaFaso",
                "relative_path": f"data/tom2024_category_a/CATA-English/{item.relative_path}",
                "source_group": f"tom_timestamp:{item.source_group}",
                "original_label": label,
                "foliar_binary": 0 if label == "Healthy_leaf" else (1 if label in eligible else ""),
                "foliar_binary_eligible": int(label in eligible),
                "alternaria_binary": 1 if label == "Alternaria_D" else (0 if label == "Healthy_leaf" else ""),
                "alternaria_binary_eligible": int(label in {"Healthy_leaf", "Alternaria_D"}),
                "role": "development_or_external_source",
                "country": "Burkina Faso",
                "species": "Allium cepa",
            }
        )
    return rows


def digigreen_onion_rows() -> list[dict]:
    manifest = pd.read_csv(ROOT / "audit" / "digigreen_onion" / "download_manifest.csv")
    rows = []
    for item in manifest.itertuples(index=False):
        diagnosis = item.diagnosis
        rows.append(
            {
                "dataset": "DigitalGreen_onion_operational",
                "relative_path": f"data/digigreen_onion/{item.local_file}",
                "source_group": f"digigreen_file:{item.image_file}",
                "original_label": diagnosis,
                "foliar_binary": 0 if diagnosis == "Healthy" else 1,
                "foliar_binary_eligible": 1,
                "alternaria_binary": 1 if "Purple Blotch" in diagnosis else (0 if diagnosis == "Healthy" else ""),
                "alternaria_binary_eligible": int(diagnosis == "Healthy" or "Purple Blotch" in diagnosis),
                "role": "small_operational_case_series",
                "country": item.country,
                "species": "Allium cepa",
            }
        )
    return rows


def digigreen_ood_rows() -> list[dict]:
    manifest = pd.read_csv(ROOT / "audit" / "digigreen_non_onion_ood" / "download_manifest.csv")
    rows = []
    for item in manifest.itertuples(index=False):
        rows.append(
            {
                "dataset": "DigitalGreen_non_onion_OOD",
                "relative_path": f"data/digigreen_non_onion_ood/{item.local_file}",
                "source_group": f"digigreen_file:{item.image_file}",
                "original_label": f"{item.crop}::{item.diagnosis}",
                "foliar_binary": "",
                "foliar_binary_eligible": 0,
                "alternaria_binary": "",
                "alternaria_binary_eligible": 0,
                "role": "non_Allium_open_set",
                "country": item.country,
                "species": "non-Allium",
            }
        )
    return rows


def main() -> None:
    rows = cold_rows() + tom_rows() + digigreen_onion_rows() + digigreen_ood_rows()
    for index, row in enumerate(rows):
        row["sample_id"] = f"sample_{index:05d}"
        absolute = ROOT / row["relative_path"]
        if not absolute.exists():
            raise FileNotFoundError(absolute)
        row["bytes"] = absolute.stat().st_size
    fields = [
        "sample_id", "dataset", "relative_path", "source_group", "original_label",
        "foliar_binary", "foliar_binary_eligible", "alternaria_binary",
        "alternaria_binary_eligible", "role", "country", "species", "bytes",
    ]
    output = ROOT / "manifests"
    output.mkdir(parents=True, exist_ok=True)
    with (output / "unified_manifest.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    frame = pd.DataFrame(rows)
    summary = {
        "samples": len(frame),
        "by_dataset": frame.groupby("dataset").size().astype(int).to_dict(),
        "foliar_binary_eligible": frame.groupby("dataset")["foliar_binary_eligible"].sum().astype(int).to_dict(),
        "alternaria_binary_eligible": frame.groupby("dataset")["alternaria_binary_eligible"].sum().astype(int).to_dict(),
    }
    (output / "unified_manifest_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
