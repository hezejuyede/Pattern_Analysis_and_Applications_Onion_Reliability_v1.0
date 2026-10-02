# Independent audit of the onion reliability evidence

## Decision: PASS

The frozen results are internally consistent, independently reproducible, and adequate as the evidence base for a narrowly framed **cross-dataset reliability and split-unit sensitivity study**. This decision does not support field deployment, pathogen-specific diagnosis, or a claim that the classifier is robust across countries.

The audited primary script is SHA-256 `287f848b7f7b5d94c2ce7dbe5e7cb04ae0436de296ca5e34424c3b32232422b4`. The repaired augmentation-ablation script is SHA-256 `bb8f51e9579be0d930118f4a1bbce972c22d88510051529f2ca12c2b93214b54`. The audited internal 64-file evidence manifest was SHA-256 `66197d76c69db49a194c3701c72dca9861aed22ee6f10fe2bff8e287721f3115`.

For public release, compressed membership tables were expanded to ordinary CSV and the historical absolute run command was made repository-relative. The exact audited ablation script is retained as `original_run_augmented_family_leakage_ablation.py`; the current public-release manifest is `c0dcc9206b8ea6c836ba10c54b06f2152531efd9930563cf64674047104d9c47`. These packaging changes do not alter features, assignments, predictions or numerical estimates; see `PUBLIC_REPOSITORY_MIGRATION.md` at repository root.

## What was independently verified

The prediction-level audit passed all 190 checks. It covered 1,643 TOM2024 development images from 103 acquisition days, 813 exact-deduplicated COLD external images, 24 Digital Green onion cases, seven models, 189 reported metric rows, split membership, sample identities, labels, thresholds, serialized estimators, and exact image hashes.

All 189 point estimates independently recomputed from the per-image predictions. The largest absolute difference was `6.66e-15`. All reported percentile limits independently recomputed; the largest absolute difference was `2.30e-14`. These are decimal-to-binary floating-point round-trip effects. Serialized-model versus parsed-CSV scores differed by at most `1.42e-14`, probabilities by at most `1.11e-16`, and thresholds by exactly zero; every class prediction and outer-fold assignment matched exactly.

A clean isolated run refitted all seven models from the frozen features. Per-image TOM, COLD, and Digital Green predictions, primary metric tables, OOD tables, paired differences, transport gaps, fold summaries, and class-level summaries had zero numeric difference from the canonical results. Control files were byte-identical; nested-CV timing columns were excluded because they are wall-clock measurements.

The repaired 30-seed augmentation-family experiment also reproduced from zero. Every tabular value matched exactly, and the report and PNG figure were byte-identical. The repair and its pre/post hashes are documented in `ABLATION_REPAIR_AUDIT_TRAIL.md`.

Finally, all 64 files in the rebuilt freeze existed, all byte counts matched, all SHA-256 values matched, and their total size matched the declared 93,854,901 bytes.

## Leakage and model-selection audit

The primary model path uses only TOM2024 features, labels, and acquisition-day groups for inner hyperparameter selection, outer cross-fitting, Platt calibration, and threshold selection. Each TOM outer-test day is absent from the base fit, calibrator fit, and threshold calculation used for that prediction. The final external calibrator and threshold use the TOM nested out-of-fold score stream; COLD enters only after the final TOM estimator, calibrator, and threshold exist.

TOM acquisition days, source groups, and exact hashes each occur in one outer fold. Four additional TOM exact duplicates were removed before analysis; all four matched the retained label. No exact hash overlaps TOM and COLD. The stored outer folds recreate exactly from `StratifiedGroupKFold` with seed 20260930.

EfficientNet-B0, ConvNeXt-Tiny, and Swin-T follow the same TOM-only nested protocol as ResNet18. They remain secondary sensitivity analyses. The local prespecification correctly records that COLD had already been opened for the ResNet18 primary analysis before those challengers were added, so the challenger comparison is not a second pristine external confirmation. No challenger was retained, discarded, or promoted based on COLD.

The current outputs do not call the highest external value the selected or best model. This matters because Swin-T has higher COLD AUROC (0.782) than ResNet18 (0.735), but much worse source-locked balanced accuracy (0.562 versus 0.655), Brier score (0.282 versus 0.240), and disorder sensitivity (0.127 versus 0.444). That pattern is evidence of transport and operating-point instability, not grounds to replace the primary after viewing COLD.

## Statistical findings supported by the audit

The primary ResNet18-logistic model exceeded the three simple COLD baselines under the paired image bootstrap. Its balanced-accuracy differences were +0.155 versus prevalence (95% interval 0.125 to 0.185), +0.130 versus the colour shortcut (0.095 to 0.165), and +0.085 versus handcrafted features (0.045 to 0.124). EfficientNet-B0 was statistically indistinguishable from ResNet18 for balanced accuracy, AUROC, and Brier score in this external set.

The primary result remains a failure of transport: balanced accuracy changed from 0.927 in TOM nested validation to 0.655 in COLD, a difference of -0.272 (95% interval -0.306 to -0.235). COLD disorder sensitivity was 0.444. Digital Green onion performance was descriptive only (n=24), and crop-level non-onion OOD rejection had an unusably high FPR at 95% OOD sensitivity. These results support the negative reliability conclusion and contradict deployment or safety claims.

The augmented-set experiment found 4,502 files but only 816 filename-defined families. A conventional file split exposed 98.5% of test files to a sibling in training. Mean ResNet18 balanced accuracy was 0.832 with file splitting and 0.675 with family splitting. This difference must be described as **apparent split-unit inflation or split-scheme sensitivity**. It is not a pure causal estimate of leakage because changing the split unit also changes which families and family-size distribution enter the test set.

## Limits that must remain explicit

1. COLD has 813 source groups for 813 images. Its reported “exact-SHA-group bootstrap” is therefore an image-level bootstrap; it cannot account for dependence among images from the same plant, field session, location, or acquisition day. The manuscript should use that exact wording and avoid implying biological cluster uncertainty.
2. Neither public dataset provides verified plant identities, laboratory confirmation for this harmonized endpoint, or independent pathology re-adjudication. The outcome is dataset-assigned healthy versus visibly affected foliage, not pathogen diagnosis.
3. The filesystem supports the declared local analysis chronology but is not an external preregistration service. “Fixed in the local analysis plan” is defensible; an externally preregistered claim requires an external timestamped record.
4. ECE is a bin-dependent point estimate without a confidence interval. Brier score measures total probabilistic error, not calibration alone.
5. The paired bootstrap tail fraction is descriptive and should not be presented as a formal multiplicity-adjusted p-value.
6. The study can establish that common evaluation choices overstate apparent reliability in these public onion datasets. It cannot establish agronomic readiness, recommend treatment, identify a causal pathogen, or generalize to unseen farms.

## Audit artifacts

- `mechanical_verification_summary.json` and `verification_checks.csv`: 190/190 primary checks.
- `recomputed_metric_comparison.csv`: all 189 point estimates and intervals.
- `recomputed_cold_paired_model_differences.csv`: independent paired comparisons.
- `serialized_artifact_numeric_comparison.csv`: model/CSV round-trip differences.
- `deterministic_rerun_summary.json`: clean seven-model and derived-analysis comparison.
- `deterministic_ablation_rerun_summary.json`: clean 30-seed ablation comparison.
- `freeze_manifest_verification_summary.json`: 64/64 frozen-file verification.
- `ABLATION_REPAIR_AUDIT_TRAIL.md`: blocking issue, exact repair, and pre/post hashes.
