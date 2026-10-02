"""Output-directory wrapper for the immutable matched-sibling experiment core."""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="New output directory; never overwrite a frozen design or existing result.")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error(f"Output directory is not empty: {output}")
    core_path = Path(__file__).resolve().with_name("run_matched_sibling_intervention.py")
    spec = importlib.util.spec_from_file_location("matched_sibling_frozen_core", core_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load the frozen experiment core")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.OUT = output
    module.main()


if __name__ == "__main__":
    main()
