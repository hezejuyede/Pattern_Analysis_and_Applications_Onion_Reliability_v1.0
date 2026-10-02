"""Download public source images into the layout expected by the analysis.

Source images remain subject to their upstream licences. This script records
or checks hashes; it does not make those images part of this repository.
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOM_URL = "https://ppedmas.org/datasets/images/TOM2024/TOM2024-CATEGORYA-English.zip"
TOM_SHA256 = "6c110be15bc8bcdd4bc58aed277b3b81d66d8e2497505edb19639a60a8b76747"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(*parts: str) -> None:
    subprocess.run([sys.executable, *parts], cwd=ROOT, check=True)


def fetch_tom() -> None:
    archive = ROOT / "downloads" / "TOM2024-CATEGORYA-English.zip"
    if not archive.is_file():
        run("code/download_ranged.py", TOM_URL, str(archive))
    actual = sha256_file(archive)
    if actual != TOM_SHA256:
        raise RuntimeError(f"TOM2024 archive SHA-256 mismatch: {actual}")
    destination = ROOT / "data" / "tom2024_category_a"
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as source:
        bad = source.testzip()
        if bad is not None:
            raise RuntimeError(f"corrupt TOM2024 ZIP member: {bad}")
        source.extractall(destination)
    expected = destination / "CATA-English" / "onion_diseases"
    if not expected.is_dir():
        raise RuntimeError(f"unexpected TOM2024 archive layout: {expected} is absent")


def fetch_cold() -> None:
    run("code/download_hf_rows.py", "--config", "raw", "--expected-rows", "815", "--output-name", "cold_raw_hf")


def fetch_digigreen() -> None:
    run("code/download_digigreen_onion.py")
    run("code/download_digigreen_ood.py")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("all", "tom", "cold", "digigreen"), default="all")
    args = parser.parse_args()
    if args.dataset in {"all", "tom"}:
        fetch_tom()
    if args.dataset in {"all", "cold"}:
        fetch_cold()
    if args.dataset in {"all", "digigreen"}:
        fetch_digigreen()
    print("Public inputs downloaded. Run scripts/verify_downloaded_images.py next.")


if __name__ == "__main__":
    main()
