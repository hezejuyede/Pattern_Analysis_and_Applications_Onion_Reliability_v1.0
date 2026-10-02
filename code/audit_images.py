"""Create an auditable inventory and duplicate report for image folders."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError


COPY_SUFFIX = re.compile(r"(?:\s*-\s*Copy)+$", re.IGNORECASE)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dhash_hex(image: Image.Image, width: int = 16, height: int = 16) -> str:
    gray = image.convert("L").resize((width + 1, height), Image.Resampling.LANCZOS)
    values = np.asarray(gray, dtype=np.int16)
    bits = values[:, 1:] > values[:, :-1]
    packed = np.packbits(bits.reshape(-1))
    return packed.tobytes().hex()


def normalised_source_id(path: Path) -> str:
    return COPY_SUFFIX.sub("", path.stem)


def write_rows(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    paths = sorted(
        path for path in args.root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
    )
    rows: list[dict] = []
    errors: list[dict] = []
    exact_groups: dict[str, list[str]] = defaultdict(list)
    perceptual_groups: dict[str, list[str]] = defaultdict(list)
    source_groups: dict[str, list[str]] = defaultdict(list)

    for path in paths:
        relative = path.relative_to(args.root).as_posix()
        digest = file_sha256(path)
        try:
            with Image.open(path) as image:
                image.load()
                row = {
                    "relative_path": relative,
                    "class": relative.split("/")[0] if "/" in relative else "",
                    "bytes": path.stat().st_size,
                    "width": image.width,
                    "height": image.height,
                    "mode": image.mode,
                    "format": image.format,
                    "sha256": digest,
                    "dhash256": dhash_hex(image),
                    "source_id": normalised_source_id(path),
                }
        except (OSError, UnidentifiedImageError) as exc:
            errors.append({"relative_path": relative, "error": str(exc)})
            continue
        rows.append(row)
        exact_groups[row["sha256"]].append(relative)
        perceptual_groups[row["dhash256"]].append(relative)
        source_groups[row["source_id"]].append(relative)

    duplicate_rows: list[dict] = []
    for kind, groups in (
        ("exact_sha256", exact_groups),
        ("identical_dhash256", perceptual_groups),
        ("normalised_filename", source_groups),
    ):
        group_number = 0
        for key, members in sorted(groups.items()):
            if len(members) < 2:
                continue
            group_number += 1
            for member in members:
                duplicate_rows.append(
                    {"kind": kind, "group": group_number, "key": key, "relative_path": member, "group_size": len(members)}
                )

    fieldnames = list(rows[0]) if rows else ["relative_path"]
    write_rows(args.output / "image_inventory.csv", fieldnames, rows)
    write_rows(
        args.output / "duplicate_groups.csv",
        ["kind", "group", "key", "relative_path", "group_size"],
        duplicate_rows,
    )
    write_rows(args.output / "read_errors.csv", ["relative_path", "error"], errors)

    summary = {
        "root": str(args.root.resolve()),
        "image_files": len(paths),
        "readable_images": len(rows),
        "read_errors": len(errors),
        "classes": {
            label: sum(1 for row in rows if row["class"] == label)
            for label in sorted({row["class"] for row in rows})
        },
        "unique_sha256": len(exact_groups),
        "unique_dhash256": len(perceptual_groups),
        "unique_normalised_source_ids": len(source_groups),
        "files_in_exact_duplicate_groups": sum(len(items) for items in exact_groups.values() if len(items) > 1),
        "files_in_identical_dhash_groups": sum(len(items) for items in perceptual_groups.values() if len(items) > 1),
        "files_in_normalised_filename_groups": sum(len(items) for items in source_groups.values() if len(items) > 1),
    }
    (args.output / "audit_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
