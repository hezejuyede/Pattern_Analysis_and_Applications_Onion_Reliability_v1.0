"""Audit TOM2024 onion images at the acquisition-source level.

TOM2024 filenames contain a 13-digit millisecond timestamp after the first
underscore.  Files sharing that timestamp are treated as crops/resizes/copies
of one acquisition event.  This is a conservative leakage group; it is not
claimed to identify a biological plant or field plot.
"""

from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


TIMESTAMP = re.compile(r"_(?P<timestamp>1[0-9]{12})(?:_|\(|\.|$)")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data_root = root / "data" / "tom2024_category_a" / "CATA-English"
    audit_root = root / "audit" / "tom2024_category_a_onion" / "source_groups"
    audit_root.mkdir(parents=True, exist_ok=True)
    paths = sorted((data_root / "onion_diseases").rglob("*.jpg")) + sorted((data_root / "onion_pests").rglob("*.jpg"))

    records = []
    groups: dict[str, list[dict]] = defaultdict(list)
    unparsed = []
    for path in paths:
        relative = path.relative_to(data_root).as_posix()
        label = relative.split("/")[1]
        match = TIMESTAMP.search(path.stem)
        if not match:
            unparsed.append({"relative_path": relative, "label": label})
            continue
        timestamp = match.group("timestamp")
        record = {
            "source_group": timestamp,
            "utc_datetime": datetime.fromtimestamp(int(timestamp) / 1000, tz=timezone.utc).isoformat(),
            "label": label,
            "relative_path": relative,
        }
        records.append(record)
        groups[timestamp].append(record)

    group_rows = []
    member_rows = []
    for source_group, members in sorted(groups.items()):
        labels = sorted({member["label"] for member in members})
        summary = {
            "source_group": source_group,
            "utc_datetime": members[0]["utc_datetime"],
            "member_count": len(members),
            "label_count": len(labels),
            "labels": ";".join(labels),
        }
        group_rows.append(summary)
        for member in members:
            member_rows.append({**summary, "label": member["label"], "relative_path": member["relative_path"]})

    def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
        with path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    write_csv(audit_root / "source_group_members.csv", list(member_rows[0]), member_rows)
    write_csv(audit_root / "source_groups.csv", list(group_rows[0]), group_rows)
    write_csv(audit_root / "unparsed_files.csv", ["relative_path", "label"], unparsed)
    cross_label = [row for row in member_rows if row["label_count"] > 1]
    write_csv(audit_root / "cross_label_source_group_members.csv", list(member_rows[0]), cross_label)
    summary = {
        "files": len(paths),
        "parsed_files": len(records),
        "unparsed_files": len(unparsed),
        "source_groups": len(groups),
        "files_in_multi_member_groups": sum(len(members) for members in groups.values() if len(members) > 1),
        "multi_member_groups": sum(len(members) > 1 for members in groups.values()),
        "cross_label_groups": sum(len({member['label'] for member in members}) > 1 for members in groups.values()),
        "files_in_cross_label_groups": sum(
            len(members) for members in groups.values() if len({member['label'] for member in members}) > 1
        ),
    }
    (audit_root / "source_group_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
