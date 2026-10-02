"""Audit acquisition-day and class dependence in cleaned TOM2024 onion data."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import normalized_mutual_info_score


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "audit" / "tom2024_category_a_onion" / "clean_source_representatives.csv"
OUTPUT = ROOT / "audit" / "tom2024_category_a_onion" / "day_confounding"
ELIGIBLE = ["Healthy_leaf", "Alternaria_D", "Fusarium-D", "Virosis-D"]


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(INPUT)
    frame["acquisition_day_utc"] = pd.to_datetime(
        frame["utc_datetime"], utc=True, format="mixed"
    ).dt.date.astype(str)
    frame = frame.loc[frame["label"].isin(ELIGIBLE)].copy()
    frame["foliar_binary"] = (frame["label"] != "Healthy_leaf").astype(int)

    table = pd.crosstab(frame["acquisition_day_utc"], frame["label"])
    for name in ELIGIBLE:
        if name not in table:
            table[name] = 0
    table = table[ELIGIBLE]
    table["total"] = table.sum(axis=1)
    table.to_csv(OUTPUT / "class_by_acquisition_day.csv")

    binary_table = pd.crosstab(frame["acquisition_day_utc"], frame["foliar_binary"])
    summary = {
        "eligible_source_representatives": int(len(frame)),
        "acquisition_days": int(frame["acquisition_day_utc"].nunique()),
        "days_per_label": {
            key: int(value)
            for key, value in frame.groupby("label")["acquisition_day_utc"].nunique().items()
        },
        "normalized_mutual_information_day_multiclass": float(
            normalized_mutual_info_score(frame["acquisition_day_utc"], frame["label"])
        ),
        "normalized_mutual_information_day_binary": float(
            normalized_mutual_info_score(frame["acquisition_day_utc"], frame["foliar_binary"])
        ),
        "pure_multiclass_days": int(((table[ELIGIBLE] > 0).sum(axis=1) == 1).sum()),
        "pure_binary_days": int(((binary_table > 0).sum(axis=1) == 1).sum()),
        "interpretation": (
            "Class labels are partly tied to acquisition day; ordinary file-random splitting can "
            "share day-specific context across train and test. Acquisition-day grouped validation "
            "is therefore the primary internal design."
        ),
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

