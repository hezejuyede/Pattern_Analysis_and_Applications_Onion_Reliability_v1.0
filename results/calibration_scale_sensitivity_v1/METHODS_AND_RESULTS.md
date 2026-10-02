# Source-only calibration construction sensitivity

This is a post hoc sensitivity analysis designed after the published COLD results were known. It is not a new independently locked external confirmation. The protocol and original input hashes were saved before fitting. No architecture, hyperparameter, calibration family, threshold objective, or retained result was selected with COLD labels.

The fixed primary ResNet18 feature matrix, 1,643 TOM samples, five acquisition-day folds, and already selected final C=0.1 were reused. The original fold-selected C values were [0.01, 0.01, 0.1, 0.1, 0.1]. Two source score streams were reconstructed in one environment: each fold's published C, or the already selected final C for every fold. Standardization was fitted on the corresponding fold's training images. Each stream fitted a logistic Platt calibrator and a Youden threshold. Both pipelines used the same final logistic head fitted to all TOM samples at C=0.1; only source score construction, calibration and the resulting threshold differed. The probabilities used to train these calibrators are not reported as unbiased source-validation results.

The paired external comparison uses the same 813 exact-deduplicated COLD images. The 1,000 paired image bootstrap replicates hold both fitted pipelines fixed and do not incorporate model-training, plant, farm, or dataset uncertainty. ECE has no interval. The original frozen metrics remain the primary analysis.

| Construction | Balanced accuracy | AUROC | Brier | Sensitivity | Specificity |
|---|---:|---:|---:|---:|---:|
| Reconstructed legacy mixed-C calibration | 0.655321 | 0.734511 | 0.240272 | 0.444444 | 0.866197 |
| Fixed-final-C OOF calibration | 0.646150 | 0.734511 | 0.230968 | 0.397933 | 0.894366 |

The unchanged common external scores imply identical AUROC under the positive-slope Platt calibrators; AUROC equality is not an additional empirical replication. The analysis diagnoses probability scale and threshold construction, not a new representation or a remedy for external data shift. Refitting from 80% fold-training samples to all TOM samples can still cause score-scale change in both arms, and the fixed-C comparison does not eliminate that possibility.

Reconstruction in scikit-learn 1.8.0 is compared with the published predictions rather than assumed identical. Maximum original/reconstructed differences: source raw score 3.5527136788e-15; external raw score 1.7763568394e-15; external probability 2.22044604925e-16. Original/reconstructed locked decisions differ for 0 of 813 external images. See `reconstruction_audit.json` and `published_reconstruction_metrics.csv` before interpreting small contrasts.

All source score streams, fitted calibration coefficients, external probabilities, metric summaries and paired contrasts are retained. No old model was deserialized and no published input or result was modified.
