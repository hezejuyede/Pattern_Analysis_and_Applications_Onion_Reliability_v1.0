# Onion image reliability: original evidence and post hoc diagnostics

This repository preserves the original onion image reliability study and adds two explicitly post hoc diagnostics. The original locked external results are retained. The new analyses were designed after those results were known and do not constitute a new independent validation cohort.

The original study distinguishes augmentation families, acquisition days and external image archives. Healthy versus affected is a source-dataset label, not confirmed pathogen diagnosis. Filename families are not verified plants. Neither the original nor extended work establishes field safety, treatment efficacy or causal country effects.

## Release scope

The original immutable public release is [v1.0.0](https://github.com/hezejuyede/Pattern_Analysis_and_Applications_Onion_Reliability_v1.0/tree/v1.0.0). The cumulative scientific release is **v1.1.0**. Its content scope is recorded in `release_metadata_v11.json`; the annotated Git tag identifies the immutable tree. The original README and checksum list are preserved in `audit/release_v1.0.0/`. Original numerical evidence, source manifests and audit records remain available. Historical v1.0 verification reports describe that release, not the enlarged v1.1 tree.

The two extensions are:

- **Matched sibling substitution:** 30 fixed seeds, fixed probe and unexposed sentinel anchors, fixed training budgets and same-class donor replacement. It tests whether gains concentrate in families exposed through training siblings. Both original four-class and binary endpoints are retained.
- **Calibration construction sensitivity:** compare legacy mixed-regularization out-of-fold calibration with out-of-fold calibration using the already-selected final regularization. The final classification head is identical in both arms; source data determine the calibrators and thresholds. External labels do not choose an arm.

The matched experiment uses 1,142 training images from 571 families and 245 test anchors per seed. Its complete rerun reproduced all 88,200 prediction rows exactly. ResNet18 four-class balanced accuracy rose by 0.153 on probes and changed by −0.009 on unexposed sentinels; the corresponding color-feature changes were 0.284 and 0.002. These are means across repeated archive splits, not estimates from independent biological replicates. Effects were smaller for the binary endpoint. Full results, variation ranges and limitations are in the result reports; no favorable seeds or endpoints are discarded.

The original 0.832-versus-0.675 file/family contrast changed both training and test composition. The new fixed-budget diagnostic addresses that limitation but must not be numerically pooled with the original contrast. Locked TOM-to-COLD binary balanced accuracy remains 0.927 internally and 0.655 externally.

## Reproduce the extended analyses

Use a separate environment with Python **3.12.14** and `requirements-deep-revision.txt`. The original suite remains pinned separately by `requirements.txt` (scikit-learn 1.6.1); the extensions use scikit-learn 1.8.0. No GPU, PyTorch, OpenCV, image download or new feature extraction is required for the extensions. The three necessary frozen feature archives are included; each file is below 25 MiB.

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

The revised English and Chinese manuscript sources and 38-entry bibliography are in `manuscript/final_paa/narrative/`. Deterministic plot code and captions are in `manuscript/figure_pipeline/`, and vector/600-dpi figures and point-level source tables are in `manuscript/publication_figures/`. Run `python manuscript/figure_pipeline/generate_publication_figures.py --figures Fig1 Fig2` to regenerate the two changed main figures. Manuscript v1.4 uses scientific release v1.1.0. Mechanical verification does not certify editorial acceptance or eliminate the study limitations.
