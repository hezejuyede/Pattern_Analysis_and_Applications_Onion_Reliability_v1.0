# Compact repository release audit

Status: **PASS_PUBLIC_RELEASE**

- Public repository: `https://github.com/hezejuyede/Pattern_Analysis_and_Applications_Onion_Reliability_v1.0`.
- Anonymous root-level clone verified on 2 October 2026 at commit `2ed0a4ed8af7da0480f0b4d451b5c19f3355ed40`.
- Integrity verification: PASS; 97 release files, 96 declared SHA-256 hashes,
  no mismatch, missing file, unlisted file or oversized file.
- Pinned numerical recomputation: PASS; 189 metric rows and 152/152 prediction
  checks reproduced.
- Locked-fold and evidence-ledger checks: 4/4 and 3/3 passed.
- Maximum metric and interval differences: `6.66e-15` and `2.30e-14`.
- COLD ablation and TOM split-sensitivity aggregates reproduced to floating-point
  precision.
- Raw third-party images are not redistributed. Source code, manifests, locked
  folds, per-image predictions, metric tables, sensitivity outputs, lineage
  mapping, hashes and audit summaries are retained.
- Immutable release: tag `v1.0.0`.

Run `python scripts/verify_slim_repository.py` and
`python scripts/recompute_slim_evidence.py` from the repository root.
