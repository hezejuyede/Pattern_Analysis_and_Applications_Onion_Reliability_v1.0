"""Verify hashes and GitHub browser-upload constraints."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "SHA256SUMS"
MAX_BYTES = 25 * 1024 * 1024
MAX_FILES = 100
IGNORED = {".git", ".venv", "venv", "verification_output", "data", "downloads"}
REQUIRED = {
    "README.md",
    "REPRODUCING.md",
    "OMITTED_DERIVED_ARTIFACTS.csv",
    "manifests/unified_manifest.csv",
    "results/reliability_benchmark_v1/predictions/tom_nested_oof.csv",
    "results/reliability_benchmark_v1/predictions/cold_locked_external.csv",
    "results/reliability_benchmark_v1/tables/binary_metrics_group_bootstrap.csv",
    "results/cold_augmented_leakage_ablation_v1/tables/repeated_split_metrics.csv",
    "results/tom_split_severity_sensitivity_v1/tables/repeated_split_metrics.csv",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    declared = {}
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        if line.strip():
            digest, relative = line.split("  ", 1)
            declared[relative] = digest
    present_paths = [
        path for path in ROOT.rglob("*")
        if path.is_file()
        and path != MANIFEST
        and path.relative_to(ROOT).parts[0] not in IGNORED
        and "__pycache__" not in path.parts
    ]
    present = {path.relative_to(ROOT).as_posix() for path in present_paths}
    mismatch = [
        relative for relative, expected in declared.items()
        if not (ROOT / relative).is_file() or sha256_file(ROOT / relative) != expected
    ]
    unlisted = sorted(present - set(declared))
    missing_required = sorted(REQUIRED - present)
    oversized = {
        path.relative_to(ROOT).as_posix(): path.stat().st_size
        for path in present_paths if path.stat().st_size >= MAX_BYTES
    }
    forbidden = sorted(
        path.relative_to(ROOT).as_posix() for path in present_paths
        if path.suffix.lower() in {".zip", ".7z", ".rar", ".jpg", ".jpeg", ".png"}
        and "figures" not in path.parts
    )
    status = "PASS" if not mismatch and not unlisted and not missing_required and not oversized and not forbidden and len(present_paths) + 1 <= MAX_FILES else "FAIL"
    result = {
        "status": status,
        "files_including_SHA256SUMS": len(present_paths) + 1,
        "github_browser_file_limit": MAX_FILES,
        "largest_file_bytes": max(path.stat().st_size for path in present_paths),
        "github_browser_per_file_limit_bytes": MAX_BYTES,
        "declared_hashes": len(declared),
        "hash_mismatches": mismatch,
        "unlisted_files": unlisted,
        "missing_required": missing_required,
        "oversized_files": oversized,
        "forbidden_raw_or_archive_files": forbidden,
        "publication_url_resolved": ("PUBLIC_" + "REPOSITORY_URL") not in (ROOT / "README.md").read_text(encoding="utf-8"),
    }
    print(json.dumps(result, indent=2))
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
