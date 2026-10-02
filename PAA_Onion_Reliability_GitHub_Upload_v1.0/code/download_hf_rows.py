"""Materialize the public COLD raw image split through the HF rows API.

This route is used because the repository's parquet host can be unreachable in
some networks while the public dataset server remains available.  The script
records every row index, class label, signed source URL and final SHA-256.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import os
import time
import urllib.parse
import urllib.request
from pathlib import Path


DATASET = "Project-AgML/COLD_onion_leaf_disease_classification"
SPLIT = "train"
LABELS = {
    0: "iris_yellow_virus",
    1: "stemphylium_colletotrichum_leaf_blight",
    2: "healthy",
    3: "purple_blotch",
}


def get_json(url: str, retries: int = 6) -> dict:
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=90) as response:
                return json.load(response)
        except Exception:
            if attempt == retries:
                raise
            time.sleep(min(2**attempt, 20))
    raise AssertionError("unreachable")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def download_one(record: dict, output: Path, retries: int = 8) -> dict:
    label_name = LABELS[int(record["label"])]
    destination = output / label_name / f"row_{int(record['row_idx']):04d}.jpg"
    destination.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(record["url"], timeout=90) as response:
                data = response.read()
            if not data.startswith(b"\xff\xd8"):
                raise RuntimeError("response is not a JPEG")
            digest = sha256_bytes(data)
            if destination.exists() and hashlib.sha256(destination.read_bytes()).hexdigest() == digest:
                status = "reused"
            else:
                temporary = destination.with_suffix(".jpg.tmp")
                temporary.write_bytes(data)
                os.replace(temporary, destination)
                status = "downloaded"
            return {
                **record,
                "label_name": label_name,
                "relative_path": destination.relative_to(output).as_posix(),
                "bytes": len(data),
                "sha256": digest,
                "status": status,
            }
        except Exception as exc:
            if attempt == retries:
                raise RuntimeError(f"failed row {record['row_idx']}") from exc
            time.sleep(min(2**attempt, 20))
    raise AssertionError("unreachable")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", choices=("raw", "augmented"), default="raw")
    parser.add_argument("--expected-rows", type=int, default=None)
    parser.add_argument("--output-name", default=None)
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    expected_rows = args.expected_rows or (815 if args.config == "raw" else 4502)
    output_name = args.output_name or f"cold_{args.config}_hf"
    root = Path(__file__).resolve().parents[1]
    output = root / "data" / output_name
    audit = root / "audit" / output_name
    audit.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    for offset in range(0, expected_rows, 100):
        query = urllib.parse.urlencode(
            {"dataset": DATASET, "config": args.config, "split": SPLIT, "offset": offset, "length": min(100, expected_rows - offset)}
        )
        payload = get_json(f"https://datasets-server.huggingface.co/rows?{query}")
        for item in payload["rows"]:
            records.append(
                {"row_idx": int(item["row_idx"]), "label": int(item["row"]["label"]), "url": item["row"]["image"]["src"]}
            )
    if len(records) != expected_rows or len({r["row_idx"] for r in records}) != expected_rows:
        raise RuntimeError(f"row API returned {len(records)} non-unique rows, expected {expected_rows}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        completed = list(pool.map(lambda record: download_one(record, output), records))
    completed.sort(key=lambda item: item["row_idx"])

    fields = ["row_idx", "label", "label_name", "relative_path", "bytes", "sha256", "status", "url"]
    with (audit / "download_manifest.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(completed)
    summary = {
        "dataset": DATASET,
        "config": args.config,
        "split": SPLIT,
        "rows": len(completed),
        "classes": {LABELS[label]: sum(item["label"] == label for item in completed) for label in LABELS},
        "total_bytes": sum(item["bytes"] for item in completed),
        "unique_sha256": len({item["sha256"] for item in completed}),
    }
    (audit / "download_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
