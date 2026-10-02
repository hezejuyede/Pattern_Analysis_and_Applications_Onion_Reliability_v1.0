"""Verify downloaded TOM and COLD image bytes against the frozen ledger."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "results" / "reliability_benchmark_v1" / "audit" / "primary_development_external_sha256.csv"
OUTPUT = ROOT / "verification_output" / "downloaded_image_verification.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    ledger = pd.read_csv(LEDGER)
    rows = []
    for item in ledger.itertuples(index=False):
        path = ROOT / item.relative_path
        exists = path.is_file()
        actual = sha256_file(path) if exists else None
        rows.append({"sample_id": item.sample_id, "exists": exists, "match": exists and actual == item.analysis_sha256})
    frame = pd.DataFrame(rows)
    included = ledger[ledger.analysis_inclusion.isin(["development_included", "external_included"])]
    tom = set(included.loc[included.dataset.eq("TOM2024_A_BurkinaFaso"), "analysis_sha256"])
    cold = set(included.loc[included.dataset.eq("COLD_raw_India"), "analysis_sha256"])
    result = {
        "ledger_rows": int(len(ledger)),
        "files_present": int(frame.exists.sum()),
        "hashes_matching": int(frame.match.sum()),
        "all_present": bool(frame.exists.all()),
        "all_hashes_match": bool(frame.match.all()),
        "tom_cold_exact_hash_overlap": int(len(tom & cold)),
        "status": "PASS" if frame.match.all() and not (tom & cold) else "FAIL",
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
