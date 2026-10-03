# Matched replacement controls: reproducible research release v1.9

This research release contains the corrected source-component experiment, a second public archive task, a frozen DINOv2 representation sensitivity, and one-source-at-a-time replacement controls. It is an incremental research artifact, not a claim that the manuscript has passed editorial review. No author approval of a new manuscript or journal acceptance is implied.

**Scope:** the primary portable commands cover the v17/v18 matched-replacement and single-source studies. [Supplementary context](supplementary_context/README.md) provides a separate, tested reconstruction of the corrected transfer summaries from fixed predictions. [Legacy training](legacy_training/README.md) supplies the earlier regional/calibration scripts, cached inputs and a path adapter: imports, input integrity and alignment were checked, but complete historical model training was not repeated for this release. [Source curation](source_curation/README.md) preserves acquisition, grouping and image-identity evidence; repeating image-level curation requires the original images. These are distinct reproducibility scopes, not a claim that all training and raw-image acquisition were rerun.

The scientific correction in the repository's 2 October 2026 notice remains in force: filename prefixes and augmented-file groups were insufficient evidence of independent source photographs. Earlier exposure estimates and historical manuscript tags must not be treated as current validated results. This release uses the saved corrected component/leaf assignments. It does not turn archive units into independently sampled plants, verify disease diagnoses, or provide a new field cohort.


The v1.9 additions are **comment-prompted descriptive analyses performed after the v1.8 results were known**. They retain all original model fits, predictions and main endpoints. They add complete leave-one-class-out and frequency-weighted summaries, class-specific correction/harm and error headroom, unique-target and overlap counts, and conditional allocation Monte Carlo standard errors. They do not add a new field cohort, plant identities, pathological adjudication, population confidence intervals or hypothesis tests. The original v17/v18 freezes and numerical objects remain unchanged.

The dedicated [additional-analysis README](additional_analysis/README.md) states the estimands, chronology, units and reproducibility checks. A separately implemented portable program reads the original decisions, reconstructs all 14 new tables, checks all expected values, and does not import the original descriptive-analysis implementation:

```bash
python /path/to/v1.9/code/portable/reproduce_additional.py --root /path/to/v1.9 --out /path/to/rebuilt-additional
```

The earlier main reconstruction command below still rebuilds the original matched-replacement results. `provenance/baseline_v1.8/COPY_LEDGER.csv` records unchanged baseline objects and the manuscript/figure files replaced for this release. The historical v1.8 manifest in that namespace is not the v1.9 distribution manifest.

## What is included

- All six complete prediction files: original ResNet18 logistic, colour logistic and ResNet18 cosine 1NN joint controls, plus DINOv2 logistic and cosine 1NN joint controls, and all five models in the single-source design. The older secondary binary endpoint is retained. There are 4,704,230 prediction rows in total.
- Frozen feature matrices, manifests, split and replacement assignments, training membership, fit logs, output tables, original numerical scripts, protocol freezes and technical verification records.
- A portable analysis implementation that independently reconstructs every retained joint arm/split summary and every single-source target/class/split summary, checks zero-baseline parity, rebuilds same-target context comparisons and same-selected-mask baseline tables, and optionally refits a deterministic small set of cases.
- Lossless Parquet conversions of predictions and large single-source training membership tables. **Every column remains a string**: original decimal text, blank fields and row order were preserved and compared against every source CSV cell. Casting is explicit in the portable reader. The original gzip SHA and the distributed Parquet SHA are different identifiers, both recorded in the conversion ledger.
- Large derived CSV tables use byte-preserving gzip transport. Decompression restores the exact original CSV bytes and SHA; the transport ledger keeps both identifiers. This reduces package size without dropping data or changing frozen records.

No original archive photographs, email correspondence, journal decision letters, private editorial reports, or pretrained model checkpoint files are distributed here. Data-derived artifacts retain upstream terms; the MIT license applies to original project code, not automatically to every file. Read [DATA_LICENSES_AND_ATTRIBUTION.md](DATA_LICENSES_AND_ATTRIBUTION.md).

## Reproduce from any directory

Use Python 3.12 with the recorded numerical versions. A virtual environment is recommended. No network access or original image paths are needed after installing dependencies.

```bash
python -m pip install -r /path/to/v1.9/requirements.txt
python /path/to/v1.9/code/portable/verify_release.py --root /path/to/v1.9
python /path/to/v1.9/code/portable/reproduce.py --root /path/to/v1.9 --out /path/to/rebuilt --refit smoke
python /path/to/v1.9/code/portable/plot_primary.py --tables /path/to/rebuilt/primary_tables --out /path/to/rebuilt/figures
```

Quote paths containing spaces on Windows. `--refit none` reconstructs all tables without training. `--refit smoke` (the default) refits the first split's exposure-0-A and sham-0-A joint conditions for all five representations/heads, plus the first target identifier in the first split for the single-source design; the selection does not depend on performance. Both nearest-neighbour routes are checked against their recorded neighbour row identifiers. `--refit full-audit` repeats the previously fixed larger independent audit sample, not the complete experiment. `--dataset onion` or `--dataset potato`, and `--experiment v17|v18_joint|v18_single`, allow individual checks.

The default smoke run rebuilds the full statistics and performs 24 logistic refits plus 16 nearest-neighbour condition checks across both tasks. Original full experiment logs and the separate full independent audit reports are retained; these counts should not be confused with one another. Exact bitwise fitted probabilities can depend on numerical library versions. The probability tolerance is 1e-10, with zero predicted-label mismatches; all deterministic table comparisons use 1e-11 or tighter except plotted percentage-point values (1e-10).

The portable entry point resolves frozen historical input paths through `provenance/path_mapping.json`, always preferring the release object. Historical paths inside unchanged protocol JSON and NPZ image identifiers are inert provenance, not filesystem dependencies. The original frozen scripts are preserved byte-for-byte under `code/original`; they are records of the original environment. Run the portable entry point for this release, not those archival scripts. All original hashes remain unchanged.

## Results and interpretation

Joint controls use 30 saved splits, five complementary allocation pairs per split, and exposure/sham conditions with matched removed donors and class budgets. The selected-probe contrast is balanced-accuracy exposure minus sham. Sentinel contrasts separate correctness turnover from label churn. The percentile bars describe variation across the saved split design; they are not confidence intervals for independently sampled plants.

Single-source controls use the first ten saved splits. For each target, its mapped donor is removed; exposure inserts two related target views and sham inserts two views from its fixed unrelated same-class alternate. Other training sources are identical within that exposure/sham pair. Evaluation classes are averaged equally within target, targets within their original target class, and target classes within split. Ten-split minimum/maximum ranges are design ranges, not biological confidence intervals.

The class-count sensitivity (at least 30 archive source units) was chosen after the v17 outcomes were known. In joint controls it reweights evaluation classes; in single-source controls it restricts both target and evaluation classes. Models are not refitted for this sensitivity. These are not identically weighted estimands.

Compare the single and joint designs on the same first ten splits and the same target, averaging the five selected joint allocations for that target. Their difference is sensitivity to the surrounding training policy; it is not a pure interaction estimate or evidence of no interference. The potato findings are retained, including smaller effects. A frozen pretrained representation does not establish the absence of pretraining/archive overlap.

`reproduced/primary_tables` contains regenerated figure sources (percentage points), same-mask zero/sham/exposure balanced accuracies (fractions), and matched-target context tables (fractions). Every underlying model, class and retained endpoint remains accessible in the experiment-level reconstructed files. No p-values or artificial biological replication counts are added.

## Release structure

| Directory | Contents |
| --- | --- |
| `experiments/v17/{onion,potato}` | Three original fixed models, multiclass and secondary binary joint predictions |
| `experiments/v18_joint/{onion,potato}` | Two frozen DINOv2 joint models, multiclass |
| `experiments/v18_single/{onion,potato}` | All five models, single-source controls |
| `inputs/{onion,potato}` | Cached numerical feature matrices; no raw images |
| `code/portable` | Path-independent analysis, verification and plotting entry points |
| `code/original*` | Unmodified experimental and independent audit scripts |
| `provenance` | Original freezes, path/hash mappings, conversion and independent audit evidence |
| `expected` | Original figure/context tables used only after reconstruction for equality checks |
| `additional_analysis` | Comment-prompted descriptive specification, original implementation, input bindings, 14 expected tables and independent verification |
| `verification` | Historical v1.8 smoke checks and current v1.9 replay checks; the v1.9 replay used `--refit none` and performed no new fitting |
| `supplementary_context` | Replayed fixed-prediction transfer summaries, original inputs and uncertainty code |
| `legacy_training` | Earlier regional/calibration code and cached training inputs; input validation only in this release |
| `source_curation` | Original acquisition, image identity and grouping code, manifests and upstream records |
| `manuscript` | Current bilingual editable manuscripts, reading PDFs, supplementary material and document checks |
| `figures` | Vector and raster figures, exact source tables and captions |

Before using the data scientifically, read the recorded acquisition/identity limitations and the upstream source terms. This artifact improves reproducibility and control design; it does not substitute for external collection records, pathological review, or editorial assessment of originality.
