# Evidence-unit validation under dataset shift in onion foliar images

This is the compact, browser-uploadable reproducibility repository for the
Pattern Analysis and Applications submission. It tests how file-level splitting
can overstate apparent performance when augmented siblings share a source image,
then measures transport of a model locked on acquisition-day-grouped TOM2024
data to exact-deduplicated COLD images.

The endpoint is the source dataset's assignment of healthy versus visibly
affected foliage. The study does not establish pathogen diagnosis, field safety,
treatment efficacy or causal country effects.

## Evidence retained in this compact release

- all analysis, data-retrieval, feature-extraction and figure code;
- the 5,960-row unified sample manifest and the locked TOM outer folds;
- per-image predictions for TOM, COLD and the descriptive DigiGreen analyses;
- reported metric, bootstrap, sensitivity and augmentation-family tables;
- the COLD augmentation-family lineage manifest and raw-family mapping;
- source metadata, SHA-256 ledgers and independent audit summaries; and
- an offline verifier that recomputes 189 metric rows, their group-bootstrap
  intervals, paired COLD comparisons and both repeated-split aggregate tables.

Large derived feature matrices, serialized estimator copies and two expanded
repeated-split membership ledgers are intentionally omitted. They are generated
outputs rather than unique observations. `OMITTED_DERIVED_ARTIFACTS.csv` records
every omitted file, its original SHA-256 and the reason for omission. The
public-data reconstruction commands remain available in `REPRODUCING.md`.

## One-command verification

```bash
python -m pip install -r requirements.txt
python scripts/verify_slim_repository.py
python scripts/recompute_slim_evidence.py
```

## Main reported findings

- COLD augmented: 4,502 files represented 816 filename-defined families; in
  ordinary file splits, 98.5% of test files had a sibling in training.
- ResNet18 mean balanced accuracy was 0.832 with file splitting and 0.675 with
  family grouping. This is interpreted as apparent split-unit inflation.
- Locked ResNet18-plus-logistic performance changed from 0.927 balanced
  accuracy in acquisition-day-grouped TOM validation to 0.655 on COLD.
- On COLD, the locked model had sensitivity 0.444, specificity 0.866 and AUROC
  0.735. The conclusion is limited to cross-dataset reliability.

## Repository status

The public release is available at `https://github.com/hezejuyede/Pattern_Analysis_and_Applications_Onion_Reliability_v1.0`. This compact tree has
passed anonymous integrity and pinned-environment numerical checks. Use the immutable
release tag `v1.0.0`: `https://github.com/hezejuyede/Pattern_Analysis_and_Applications_Onion_Reliability_v1.0/tree/v1.0.0`.
