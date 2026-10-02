# Compact repository release audit

Status: **PASS_ROOT_CORRECTION_READY**

- Intended use: replace the nested upload with these files at the repository root.
- Largest file: `results/reliability_benchmark_v1/predictions/tom_nested_oof.csv` (4.92 MiB).
- Raw third-party images: not redistributed.
- Retained evidence: source code, unified manifest, locked folds, per-image
  predictions, metric tables, repeated-split outputs, lineage mapping, hashes
  and independent verification summaries.
- Omitted content: regenerable feature caches, serialized estimator copies,
  expanded repeated-split membership ledgers and redundant audit intermediates.
- Reference offline recomputation: PASS; 189 metric rows and 152 prediction
  checks reproduced, with maximum metric/interval differences below 2.4e-14.
- Aggregate checks: 40 COLD ablation rows and 24 TOM sensitivity rows reproduced
  to floating-point precision.
- Public target: `https://github.com/hezejuyede/Pattern_Analysis_and_Applications_Onion_Reliability_v1.0`.
- Remaining action: verify this root tree in the pinned environment, push tag
  `v1.0.0`, and confirm anonymous access.

Run `python scripts/verify_slim_repository.py` and
`python scripts/recompute_slim_evidence.py` from the repository root.
