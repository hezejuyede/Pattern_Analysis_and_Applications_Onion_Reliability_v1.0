"""Build a deterministic non-Allium operational OOD set from Digital Green."""

from __future__ import annotations

import concurrent.futures
import csv
import hashlib
import json
import os
import random
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path


BASE = "https://huggingface.co/datasets/DigiGreen/Crop_Disease_Images/resolve/main/"


def fetch(item: dict, destination_root: Path, retries: int = 8) -> dict:
    relative = Path(item["image_file"])
    destination = destination_root / relative.name
    url = BASE + urllib.parse.quote(relative.as_posix(), safe="/") + "?download=true"
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=120) as response:
                data = response.read()
            if not data.startswith(b"\xff\xd8"):
                raise RuntimeError("download is not a JPEG")
            digest = hashlib.sha256(data).hexdigest()
            temporary = destination.with_suffix(".jpg.tmp")
            temporary.write_bytes(data)
            os.replace(temporary, destination)
            return {
                "image_file": item["image_file"], "local_file": destination.name,
                "crop": item["crop"], "diagnosis": item["diagnosis"],
                "country": item["country"], "state": item["state"],
                "bytes": len(data), "sha256": digest, "url": url,
            }
        except Exception as exc:
            if attempt == retries:
                raise RuntimeError(f"failed {item['image_file']}") from exc
            time.sleep(min(2**attempt, 30))
    raise AssertionError("unreachable")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    raw_rows = json.loads((root / "metadata" / "digigreen_rows.json").read_text(encoding="utf-8"))
    by_file = {}
    for wrapper in raw_rows:
        item = wrapper["row"]
        if item["crop"].casefold() != "onion":
            by_file.setdefault(item["image_file"], item)
    by_crop: dict[str, list[dict]] = defaultdict(list)
    for item in by_file.values():
        by_crop[item["crop"]].append(item)
    selected: list[dict] = []
    for crop in sorted(by_crop):
        pool = sorted(by_crop[crop], key=lambda item: (item["diagnosis"], item["image_file"]))
        random.Random(f"20260930:{crop}").shuffle(pool)
        selected.extend(pool[:5])

    destination = root / "data" / "digigreen_non_onion_ood"
    audit = root / "audit" / "digigreen_non_onion_ood"
    destination.mkdir(parents=True, exist_ok=True)
    audit.mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
        records = list(executor.map(lambda item: fetch(item, destination), selected))
    records.sort(key=lambda item: item["image_file"])
    fields = ["image_file", "local_file", "crop", "diagnosis", "country", "state", "bytes", "sha256", "url"]
    with (audit / "download_manifest.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    summary = {
        "selection_rule": "up to five distinct image files per non-Onion crop; deterministic crop-specific shuffle",
        "images": len(records), "crops": len({item["crop"] for item in records}),
        "diagnoses": len({item["diagnosis"] for item in records}),
        "unique_sha256": len({item["sha256"] for item in records}),
    }
    (audit / "download_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
