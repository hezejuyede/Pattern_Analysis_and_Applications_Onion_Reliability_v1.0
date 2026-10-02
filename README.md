# Onion image reliability: original evidence and post hoc diagnostics

> **Scientific correction — 2 October 2026: submission HOLD.** A member-level audit found that different augmented filename prefixes can refer to the same original photograph. The historical file-versus-family and matched-sibling analyses therefore do not establish zero source exposure or never-exposed sentinels. Their reproducible numerical outputs are preserved as historical results, not validated source-level intervention estimates. See [CORRECTION_NOTICE_20261002.txt](audit/v16_research_gate/CORRECTION_NOTICE_20261002.txt). Corrected grouping and affected analyses require fresh review; technical PASS records from v1.2.0 and manuscript v1.5 do not override this HOLD. Original Git tags remain unchanged.

This repository preserves the original onion image reliability study and adds explicitly post hoc diagnostics, including recovered-region validation. The original locked external results are retained. The new analyses were designed after those results were known and do not constitute a new independent validation cohort.

The original study distinguishes augmentation families, acquisition days and external image archives. Healthy versus affected is a source-dataset label, not confirmed pathogen diagnosis. Filename families are not verified plants. Neither the original nor extended work establishes field safety, treatment efficacy or causal country effects.

## Release scope

The original immutable public release is [v1.0.0](https://github.com/hezejuyede/Pattern_Analysis_and_Applications_Onion_Reliability_v1.0/tree/v1.0.0). The cumulative scientific release is **v1.2.0**. Its content scope is recorded in `release_metadata_v12.json`; the annotated Git tag identifies the immutable tree. The original README and checksum list are preserved in `audit/release_v1.0.0/`. Original numerical evidence, source manifests and audit records remain available. Historical v1.0 verification reports describe that release, not the enlarged v1.1 tree.

The first two extensions, preserved from v1.1.0, are:

- **Historical matched sibling substitution (interpretation under correction):** 30 fixed seeds, fixed probe and nominal sentinel anchors, fixed training budgets and same-class donor replacement. Both four-class and binary endpoints are retained. The filename-prefix split does not establish source-disjoint anchors.
- **Calibration construction sensitivity:** compare legacy mixed-regularization out-of-fold calibration with out-of-fold calibration using the already-selected final regularization. The final classification head is identical in both arms; source data determine the calibrators and thresholds. External labels do not choose an arm.

The historical matched experiment uses 1,142 training images from 571 filename groups and 245 test anchors per seed. Its complete rerun reproduced all 88,200 prediction rows exactly. ResNet18 four-class balanced accuracy rose by 0.153 on nominal probes and changed by −0.009 on nominal sentinels; the corresponding color-feature changes were 0.284 and 0.002. These are reproducible filename-grouping results, but the source-exposure interpretation is suspended. They are not corrected intervention estimates or observations from independent biological replicates. Full historical results remain available; no adverse seeds or endpoints are discarded.

The original 0.832-versus-0.675 file/family contrast changed both training and test composition and shares the filename-grouping defect. The later fixed-budget diagnostic controlled file and class counts but did not repair that source-identity defect. Neither is a validated source-disjoint intervention estimate. The separate locked TOM-to-raw-COLD binary point estimates remain 0.927 internally and 0.655 externally; that branch does not use augmented filename groups. Its image-level uncertainty and unknown biological identity require their own qualification.

## Reproduce the extended analyses

Use a separate environment with Python **3.12.14** and `requirements-deep-revision.txt`. The original suite remains pinned separately by `requirements.txt` (scikit-learn 1.6.1); the extensions use scikit-learn 1.8.0. No GPU, PyTorch, OpenCV, image download or new feature extraction is required for the extensions. The four necessary frozen feature archives for the cumulative extensions are included; each file is below 25 MiB.

```bash
python -m pip install -r requirements-deep-revision.txt
python scripts/verify_deep_revision.py
python code/run_matched_sibling_intervention_cli.py --output-dir verification_output/matched_sibling_rerun
python code/verify_matched_sibling_rerun.py --canonical results/matched_sibling_intervention_v1 --rerun verification_output/matched_sibling_rerun
python code/run_calibration_scale_sensitivity.py --output-dir verification_output/calibration_rerun --bootstrap 1000
```

Result directories must be new or empty. The frozen protocols refuse to overwrite previous experiments. See `code/README_DEEP_REVISION.md` for all matched-experiment counts, definitions, hashes and interpretation. Calibration details are in `results/calibration_scale_sensitivity_v1/METHODS_AND_RESULTS.md`.

The v1.0 commands and reconstruction instructions remain in `REPRODUCING.md`. Its historical `verify_slim_repository.py` includes a 100-file browser-upload cap and is appropriate for v1.0.0. The expanded release is published through Git; use `verify_deep_revision.py` for its inputs, protocol hashes and result manifests. `OMITTED_DERIVED_ARTIFACTS.csv` is the original omission ledger: selected caches and the model-summary JSON have now been restored for the extended analyses.

## Evidence and licenses

`results/deep_revision_independent_audit_v1.json` records the independent checks. The matched results include a full rerun comparison and all family memberships; the calibration results include independent metric checks. `DEEP_REVISION_INPUTS.json` lists required inputs and their source hashes. `DEEP_REVISION_SOURCE_COPY.json` checks copied research files against the working-source artifacts. These records do not replace inspection of the scientific assumptions.

The existing MIT `LICENSE` applies to repository code and original authored materials to the extent the authors hold those rights. Source datasets and their derived content retain their original attribution and license conditions; MIT does not relicense third-party images or confer additional data rights. Read `DATA_SOURCES.md`, `LICENSE_STATUS.md` and `audit/reproducibility/data_sources_and_licenses.md` before reusing data.

## Manuscript and figures

The revised English and Chinese manuscript sources and 38-entry bibliography are in `manuscript/final_paa/narrative/`. Deterministic plot code and captions are in `manuscript/figure_pipeline/`, and vector/600-dpi figures and point-level source tables are in `manuscript/publication_figures/`. Run `python manuscript/figure_pipeline/generate_publication_figures.py --figures Fig1 Fig2` to regenerate the two changed main figures. Manuscript v1.5 uses scientific release v1.2.0. Earlier narrative and verification records describe their named historical versions. Mechanical verification does not certify editorial acceptance or eliminate the study limitations.


## Recovered regional provenance and post hoc regional validation

Public search-index associations recover a region for 1,416 of the 1,643 TOM analysis images. All 227 unmatched images are excluded from regional training and testing. These associations and filename-derived UTC days are not verified plant, farm or capture-session identifiers. This is a new analysis of an existing archive, not a newly acquired cohort or a pathology review.

The fixed experiment withholds each of three regions in turn. Its primary variant also removes every source image sharing a UTC day with the regional test. A region-only sensitivity keeps the identical tests with larger source sets; differences between these variants cannot isolate a causal date-leakage effect. Four prespecified comparators use identical partitions. All learned models select C, standardize, calibrate and select their threshold within source images only. The selected-C OOF scores train calibration; they are not reported as unbiased internal validation. Fixed-prediction bootstrap intervals use days, and aggregate draws share global days across all three regions. Neither test-label reuse for fitting nor selective omission of an adverse region is allowed.

```bash
python code/run_regional_robustness.py --metadata metadata/tom_region_recovery/analysis_image_region_join.csv --output-dir verification_output/regional_rerun --n-jobs 4
```

The new command uses the existing `requirements-deep-revision.txt` environment and frozen `resnet18.npz` and `handcrafted.npz`; colour features are the first 189 handcrafted columns. The protocol, source and split hashes, per-image predictions, selected penalties, calibration records, supports and all model/region results are in `results/regional_robustness_v1`. Independent computational checks are in `audit/regional_validation_v1`. The full original image caches are not duplicated. Public-index recovery scripts, factual source-key index and audit records are included; original website HTML snapshots are retained in the local provenance archive, not relicensed as MIT material.

The primary paper's original TOM-to-COLD results remain unchanged. Regional and timestamp-day changes also alter training size and archive-label composition. In one regional test the purged source has no virosis-labelled images and only 15 Alternaria-labelled images. Read all regional supports before attributing an error pattern to geography. The two original immutable tags v1.0.0 and v1.1.0 remain unchanged. Technical reproducibility does not certify journal readiness, biological independence or acceptance.


For the lightweight numerical environment plus deterministic plots, install `requirements-analysis-and-plots.txt`. Regenerate the regional figure with `python manuscript/figure_pipeline/generate_regional_figure.py`; its separate source and output ledger is `manuscript/publication_figures/source_data/Fig5_BUILD_MANIFEST.json`.

The original public-index recovery script is retained as historical acquisition code with its original local output path. For an optional live refresh on another machine use `python code/recover_tom_regions_portable.py --output-dir verification_output/new_live_region_index`. This wrapper requires a new directory and marks the current query date. A changed website response is not the analyzed snapshot and must not replace frozen metadata in a claimed exact rerun. Network access is unnecessary for the released regional fits and their verification.
