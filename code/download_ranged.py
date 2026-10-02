"""Download a public file in verified byte ranges.

The script is intentionally dependency-free.  Each range is retried and its
exact length is checked before the chunks are joined.  A final SHA-256 digest
is written next to the file so later analyses can cite the acquired object.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import time
import urllib.request
from pathlib import Path


def head_size(url: str) -> int:
    request = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(request, timeout=60) as response:
        value = response.headers.get("Content-Length")
    if not value:
        raise RuntimeError("Server did not provide Content-Length")
    return int(value)


def fetch_range(url: str, part: Path, start: int, end: int, retries: int = 8) -> dict:
    expected = end - start + 1
    if part.exists() and part.stat().st_size == expected:
        return {"start": start, "end": end, "bytes": expected, "status": "reused"}
    for attempt in range(1, retries + 1):
        try:
            request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(request, timeout=120) as response:
                status = getattr(response, "status", None)
                data = response.read()
            if status != 206:
                raise RuntimeError(f"expected HTTP 206, received {status}")
            if len(data) != expected:
                raise RuntimeError(f"expected {expected} bytes, received {len(data)}")
            temporary = part.with_suffix(part.suffix + ".tmp")
            temporary.write_bytes(data)
            os.replace(temporary, part)
            return {"start": start, "end": end, "bytes": expected, "status": "downloaded"}
        except Exception as exc:  # retry transient CDN/network failures
            if attempt == retries:
                raise RuntimeError(f"range {start}-{end} failed after {retries} attempts") from exc
            time.sleep(min(2**attempt, 30))
    raise AssertionError("unreachable")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("output", type=Path)
    parser.add_argument("--chunk-mib", type=int, default=4)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    size = head_size(args.url)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    parts_dir = args.output.with_suffix(args.output.suffix + ".parts")
    parts_dir.mkdir(parents=True, exist_ok=True)
    chunk_size = args.chunk_mib * 1024 * 1024
    ranges = [(start, min(size - 1, start + chunk_size - 1)) for start in range(0, size, chunk_size)]

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(fetch_range, args.url, parts_dir / f"{index:05d}.part", start, end): index
            for index, (start, end) in enumerate(ranges)
        }
        records = [future.result() for future in concurrent.futures.as_completed(futures)]

    temporary = args.output.with_suffix(args.output.suffix + ".joining")
    with temporary.open("wb") as destination:
        for index in range(len(ranges)):
            with (parts_dir / f"{index:05d}.part").open("rb") as source:
                while block := source.read(1024 * 1024):
                    destination.write(block)
    if temporary.stat().st_size != size:
        raise RuntimeError(f"joined size {temporary.stat().st_size} != expected {size}")
    os.replace(temporary, args.output)

    manifest = {
        "url": args.url,
        "bytes": size,
        "sha256": sha256_file(args.output),
        "ranges": sorted(records, key=lambda item: item["start"]),
    }
    args.output.with_suffix(args.output.suffix + ".download.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps({key: manifest[key] for key in ("bytes", "sha256")}, indent=2))


if __name__ == "__main__":
    main()
