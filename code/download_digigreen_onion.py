"""Download the onion-only subset of Digital Green's expert annotations."""

from __future__ import annotations

import concurrent.futures
import csv
import hashlib
import json
import os
import time
import urllib.parse
import urllib.request
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
                "image_file": item["image_file"],
                "local_file": destination.name,
                "diagnosis": item["diagnosis"],
                "country": item["country"],
                "state": item["state"],
                "details": item["details"],
                "bytes": len(data),
                "sha256": digest,
                "url": url,
            }
        except Exception as exc:
            if attempt == retries:
                raise RuntimeError(f"failed {item['image_file']}") from exc
            time.sleep(min(2**attempt, 30))
    raise AssertionError("unreachable")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    items = json.loads((root / "metadata" / "digigreen_onion_rows.json").read_text(encoding="utf-8"))
    destination = root / "data" / "digigreen_onion"
    audit = root / "audit" / "digigreen_onion"
    destination.mkdir(parents=True, exist_ok=True)
    audit.mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        records = list(pool.map(lambda item: fetch(item, destination), items))
    records.sort(key=lambda item: item["image_file"])
    fields = ["image_file", "local_file", "diagnosis", "country", "state", "details", "bytes", "sha256", "url"]
    with (audit / "download_manifest.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    summary = {
        "images": len(records),
        "unique_sha256": len({item["sha256"] for item in records}),
        "diagnoses": {},
        "states": {},
    }
    for key, field in (("diagnoses", "diagnosis"), ("states", "state")):
        for value in sorted({item[field] for item in records}):
            summary[key][value] = sum(item[field] == value for item in records)
    (audit / "download_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
