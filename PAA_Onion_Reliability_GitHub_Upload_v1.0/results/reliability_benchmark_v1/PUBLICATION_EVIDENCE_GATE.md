# Publication evidence gate and deployment boundary

**Publication reliability science gate: PASS WITH EXPLICIT LIMITS.**  
**Manuscript-evidence gate: PASS WITH EXPLICIT LIMITS.**  
**Deployment, safety, and pathogen-specific diagnostic gates: FAIL.**

The a-priori primary ResNet18 model reached locked COLD balanced accuracy 0.655 (95% exact-SHA-group bootstrap interval 0.626–0.685), AUROC 0.735, and sensitivity 0.444. Its balanced accuracy changed by -0.272 from acquisition-day nested TOM validation to COLD. This supports a cross-dataset reliability-failure paper; it does not support field deployment.

EfficientNet-B0, ConvNeXt-Tiny, and Swin-T were listed before their features or scores were computed. They use identical TOM-only nested fitting, fold-local preprocessing, cross-fitted internal calibration/thresholds, and a final TOM-only calibrator/threshold for COLD. Every challenger is retained as a secondary sensitivity. COLD had already been opened for the primary ResNet analysis, so these additions are post-primary sensitivity evidence rather than a second pristine confirmation.

The separate augmented-COLD experiment shows the methodological contribution directly: 4,502 files represent only 816 families, and 735 purple-blotch images represent 18 families. Ordinary file splitting placed augmentation siblings across partitions for nearly every test file and inflated fixed ResNet balanced accuracy relative to family-grouped splitting. These results diagnose evaluation inflation and do not validate deployment.

The manuscript may proceed only with the negative reliability framing and the limitations in `audit/publication_evidence_gate.json`. ECE is reported as a point estimate without percentile confidence bounds because nonlinear clustered bootstrap intervals were not used for that metric.
