# Leakage-controlled onion reliability benchmark: execution report

## Locked design

TOM2024 contributed 1,643 conflict-free, exact-deduplicated source representatives from 103 acquisition days (4 redundant exact files were removed at analysis time). All model and hyperparameter choices, Platt calibration, and decision thresholds were derived within TOM2024 using acquisition-day-grouped nested validation. COLD contributed 813 exact-deduplicated raw images and was not used for tuning, calibration, threshold selection, or feature learning.

The primary endpoint is binary healthy versus dataset-labelled foliar disorder. It is deliberately broader than pathogen diagnosis because COLD and TOM2024 do not provide a validated one-to-one disease ontology. The analysis does not claim pathogen-specific cross-country transfer.

TOM internal metrics use an outer-fold prediction stream in which each fold has a calibrator and threshold derived only from inner out-of-fold scores in the remaining acquisition days. The separate final calibrator and threshold use all TOM nested out-of-fold scores and are applied only to COLD, Digital Green, and OOD images. This prevents calibration or threshold reuse from making the TOM internal estimate optimistic.

ResNet18 was fixed as the sole a-priori primary model. EfficientNet-B0, ConvNeXt-Tiny, and Swin-T were prelisted before their features or scores were computed and are retained as secondary architecture sensitivities regardless of their COLD results. Because COLD had already been opened for the primary ResNet analysis before this expansion, the challenger comparison is post-primary sensitivity evidence rather than a new pristine confirmation; no challenger uses COLD for fitting, calibration, threshold selection, retention, or claims.

## External result

The a-priori primary frozen-feature model, **resnet18_logit**, obtained locked COLD balanced accuracy 0.655, AUROC 0.735, Brier 0.240, and ECE 0.168. The color shortcut, full handcrafted, and prevalence-prior comparators are reported concurrently; no model is selected using COLD. Confidence intervals and every baseline are in `tables/binary_metrics_group_bootstrap.csv`. This is a reliability result, not evidence that a new architecture has been invented.

## Leakage controls

- Conflicting TOM timestamp groups were removed upstream; one image per retained timestamp group entered the candidate pool, followed by an exact-hash deduplication at analysis time.
- Outer and inner validation folds are disjoint by UTC acquisition day, a coarser unit than the timestamp/raw-photo group.
- The 4,502-image COLD augmented collection was excluded from primary external validation because it contains derived, non-independent siblings; filename families are auditable, but the raw-parent crosswalk is incomplete for one IYSV family. It is evaluated only in a separate family-safe leakage ablation.
- COLD remained locked until the final TOM-only models, calibrators, and thresholds were fixed.

## Interpretation limits

Digital Green onion images (n=24) are reported as a case series only. Their annotations include healthy labels with textual stress observations and mixed pest/disease labels. The non-onion Digital Green subset probes rejection of other crops; it does not validate rejection of novel onion diseases. COLD lacks acquisition-day and plant-identity metadata, so its interval uses the exact-deduplicated SHA group as the resampling unit.

## Machine-readable outputs

The split manifest, per-image probabilities, grouped-bootstrap intervals, OOD scores, selected hyperparameters, fitted models, hashes, and environment versions are stored next to this report. The report should not be converted into a submission claim until the external result and its uncertainty are assessed against the journal release gate.
