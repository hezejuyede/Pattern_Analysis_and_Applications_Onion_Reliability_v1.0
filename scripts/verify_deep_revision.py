"""Verify v1.1 scientific inputs and results without loading model libraries."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    checks = []
    inputs = json.loads((ROOT / "DEEP_REVISION_INPUTS.json").read_text(encoding="utf-8"))
    for entry in inputs["files"]:
        path = ROOT / entry["path"]
        checks.append({"check": "input:" + entry["path"], "pass": path.is_file() and path.stat().st_size == entry["bytes"] and sha(path) == entry["sha256"]})
    for folder, manifest_name, path_column in [
        ("results/matched_sibling_intervention_v1", "manifest.csv", "path"),
        ("results/calibration_scale_sensitivity_v1", "output_sha256.csv", "file"),
    ]:
        base = ROOT / folder
        with (base / manifest_name).open(newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                relative = row.get(path_column) or row.get("path") or row.get("file")
                if relative is None:
                    raise ValueError(f"Missing file path in {manifest_name}: {row}")
                path = base / relative
                checks.append({"check": f"result:{folder}/{relative}", "pass": path.is_file() and sha(path) == row["sha256"]})
    matched = json.loads((ROOT / "results/matched_sibling_intervention_v1/design.json").read_text())
    checks.append({"check": "matched_frozen_script", "pass": sha(ROOT / "code/run_matched_sibling_intervention.py") == matched["script_sha256"]})
    calibration = json.loads((ROOT / "results/calibration_scale_sensitivity_v1/prespecified_protocol.json").read_text())
    for key, item in calibration["inputs"].items():
        relative = item["path"].replace("\\", "/")
        path = ROOT / relative
        checks.append({"check": "calibration_protocol:" + key, "pass": path.is_file() and sha(path) == item["sha256"]})
    independent = json.loads((ROOT / "results/deep_revision_independent_audit_v1.json").read_text())
    checks.append({"check": "independent_audit_status", "pass": independent.get("status") in {"PASS", "PASS_WITH_EXPLICIT_INFERENCE_LIMITS"}})
    large = [p.relative_to(ROOT).as_posix() for p in ROOT.rglob("*") if p.is_file() and ".git" not in p.parts and p.stat().st_size >= 25 * 1024 * 1024]
    checks.append({"check": "individual_file_size_below_25_MiB", "pass": not large})
    failures = [x for x in checks if not x["pass"]]
    result = {"status": "PASS" if not failures else "FAIL", "checks": len(checks), "failures": failures, "oversized_files": large, "scope": "scientific inputs, immutable protocols and result checksums; not editorial readiness or acceptance"}
    result["scientific_submission_status"] = "HOLD_AUGMENTED_LINEAGE_CORRECTION"
    result["interpretation"] = "A PASS here verifies historical bytes and reruns, not source-identity assumptions. Read audit/v16_research_gate/CORRECTION_NOTICE_20261002.txt before using the augmented experiments."
    print(json.dumps(result, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
