# Additional descriptive analyses in v1.9

These analyses were requested after the v1.8 outcomes and the review checklist were known. The 13 descriptive diagnostics had already been inspected before formal integration into v1.9. `ADDITIONAL_ANALYSIS_FREEZE.json` records this chronology. It is an input/specification freeze, not a preregistration made before outcomes were known.

The original v17/v18 fits, allocations, endpoints and predictions remain unchanged. The all-class estimate remains the original analysis. The at-least-30-source and leave-one-class-out results change evaluation weights while retaining the original prediction space; they do not fit a reduced-class classifier. Source-frequency weights describe the archive and are not agricultural prevalence estimates. Single-source summaries average within target class and then equally across target classes; joint summaries average allocation/arm class results before equal class weighting.

The 14 tables cover class support, unique recurring targets, cross-split overlap, class-specific recall/correction/harm/error headroom, repeated-context confusion counts, all-class/common-class/all-LOCO results, two frequency-weighted accuracy summaries, and conditional allocation MCSE. Event counts are recurring target-and-training-context records, not independent biological samples. Headroom ratios divide equally weighted aggregate quantities, not average per-row ratios. Fractions are used unless the column explicitly says `pp` or `percent`.

`conditional_allocation_mcse.csv` has 20 rows: two tasks, two designs and five models, with all original classes retained. The MCSE is the sample split SD (`ddof=1`) divided by the square root of the saved split count (30 joint, 10 single), in percentage points. It describes the fixed-archive allocation calculation under the recorded random-allocation design. It is not a plant-population confidence interval, a hypothesis test or evidence of an independent cohort. No p-values or confidence intervals have been added.

Run from any directory after installing the top-level requirements:

```bash
python /path/to/v1.9/code/portable/reproduce_additional.py --root /path/to/v1.9 --out /path/to/rebuilt-additional
```

The portable implementation was written separately and does not import `code/original/rebuild_additional_descriptive.py`. It scans the four complete joint prediction files and two complete single-source prediction files, reconstructs single-target E/H and zero outcomes from raw decisions, checks those against the existing derived target table, and calculates all 14 tables before opening the expected tables for equality comparison. It performs no model fitting.

`INPUT_BINDINGS.json` maps the immutable historical input hashes to actual release objects. Some CSVs use byte-preserving gzip and prediction files use the previously checked, exact string-cell Parquet transport. Both original and distributed SHA256 values remain recorded. Historical absolute paths in the specification are provenance, not external file requirements. `input_validation` contains only small historical comparison views needed to bind the original specification; it includes no photographs. Input hashes are checked before and after reproduction.

`expected` contains the formal v1.9 output tables, with hashes in `EXPECTED_TABLE_HASHES.json`; these values are comparison targets, not inputs to numerical reconstruction. `verification/ADDITIONAL_PORTABLE_VERIFICATION.json` records the actual independent run, all 14 comparisons, scanned rows and software versions. The final manuscript and supplement table numbering is maintained by the manuscript index, not inferred from file order.

This directory improves interpretation and reproducibility of a fixed-archive reanalysis. It supplies no new field sampling, whole-plant identity, pathological adjudication, author approval or editorial decision. Read the top-level data licence notice before reusing derived records.
