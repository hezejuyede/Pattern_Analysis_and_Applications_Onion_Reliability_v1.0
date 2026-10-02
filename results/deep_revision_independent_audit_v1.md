# Independent audit of the deep revision

**Status: PASS with explicit inference limits.** No numerical, membership, feature-order, or training-only scaling error was found in the audited diagnostics. This is an internal verification report, not a journal acceptance score.

The reviewer read both new scripts and frozen protocols, verified input/output hashes, reconstructed every matched membership, recomputed all 4,320 metric values from 88,200 saved prediction rows, checked 2,880 paired contrasts and 1,440 localization contrasts, and independently fitted all 12 model/endpoint/dose combinations in repetition 0.

Maximum prediction-derived metric discrepancy: 5e-12. Maximum independently refitted probability discrepancy: 5e-13.

## Matched sibling substitution

All 30 repetitions retain P=163 probe families, S=82 sentinel families and D=571 remaining families. The three sets are disjoint. Each probe is reciprocally matched to a unique same-class donor. Across all doses and both endpoints/features, probe and sentinel anchor files are identical. Every training condition has 1,142 distinct images from 571 families, two views per family, with class counts 596 healthy, 396 virus, 24 purple blotch and 126 blight. No exact anchor or SHA enters training. Probe sibling exposure is 0, 81 or 163 families; sentinel-family exposure is always zero. All 90 memberships passed independent reconstruction.

The scaler is inside the fitted pipeline and uses the training rows only. Twelve independent refits reproduced the saved predictions to numerical precision.

| Feature | Endpoint | Panel | Full-minus-zero BA | Split 2.5–97.5% range |
|---|---|---|---:|---|
| color_shortcut | binary_healthy_affected | probe | +0.079153 | +0.032289 to +0.143188 |
| color_shortcut | binary_healthy_affected | sentinel | -0.010048 | -0.063186 to +0.048151 |
| color_shortcut | four_class | probe | +0.283862 | +0.141888 to +0.413174 |
| color_shortcut | four_class | sentinel | +0.001932 | -0.150111 to +0.136805 |
| resnet18 | binary_healthy_affected | probe | +0.038074 | -0.000673 to +0.091073 |
| resnet18 | binary_healthy_affected | sentinel | +0.002137 | -0.042196 to +0.056485 |
| resnet18 | four_class | probe | +0.152736 | +0.024977 to +0.280478 |
| resnet18 | four_class | sentinel | -0.008842 | -0.131125 to +0.123807 |

| Feature | Endpoint | Localization: delta P minus delta S | Split 2.5–97.5% range |
|---|---|---:|---|
| color_shortcut | binary_healthy_affected | +0.089201 | +0.014295 to +0.178660 |
| color_shortcut | four_class | +0.281931 | +0.086081 to +0.505960 |
| resnet18 | binary_healthy_affected | +0.035938 | -0.024237 to +0.103191 |
| resnet18 | four_class | +0.161578 | -0.013612 to +0.355463 |

## Calibration construction

The old ResNet18 score/decision stream was reconstructed without deserializing old estimators. Source raw-score maximum error was 3.6e-15, external raw-score maximum error 1.8e-15, probability maximum error 2.3e-16, and all 813 external decisions matched. The two new arms share one fitted final-C head; they differ only in source OOF-score construction, resulting Platt mapping and source threshold. Sixteen metrics were independently recomputed. Fixed-C calibration gives external BA 0.646150 versus 0.655321 and Brier 0.230968 versus 0.240272. Its probability-quality improvement does not remove the external reliability deficit. AUROC equality is forced by the common scores and positive monotonic calibration, not a separate replication.

## Required interpretation boundaries

- **interpretation_required:** Four evidence units are complementary audit questions across separate datasets and endpoints, not an empirically identified four-stage causal or numerical decomposition. Replace claims that one common material forms a progressively nested measured chain; report augmented-archive interventions and binary external transport as separate complementary experiments.
- **interpretation_required:** Old same-number random split seeds do not match test images or training families. Call the original result a split-scheme contrast. Use the new fixed-anchor/fixed-budget intervention for the conditional within-archive sibling-substitution claim, without numerically merging effect sizes.
- **precision_limit:** Only four purple-blotch probe families and two sentinel families occur per seed; an individual error changes four-class balanced accuracy by 0.0625 or 0.125 respectively. Repeated seeds reuse the archive. Retain class support and split-variation ranges; do not treat 30 seeds as 30 biological replicates or claim precise pathology-specific effects.
- **interpretation_required:** Sentinel near-zero or sign-changing effects are compatible with split variation but do not establish equivalence or absence of global effects. Report all sentinel results and localization contrasts with their ranges. Do not use seed significance tests or declare equivalence without a justified margin and independent replication.
- **uncertainty_limit:** Bootstrap resampling keeps fitted models and prediction streams fixed; COLD exact-SHA groups are singleton images with unknown biological clusters. Describe conditional image/day uncertainty, not full training-procedure or biological uncertainty.
- **protocol_boundary:** Same-class donor matching deliberately uses augmented-archive class labels; both new analyses were designed after original target results were known. Call them post hoc diagnostic/sensitivity analyses; neither is label-free deployment, a new independent external validation, or a prospectively preregistered primary study.
- **remaining_calibration_limit:** Fixed-C OOF calibration removes mixed-C score construction but both arms still transfer from fold-trained to full-source-fitted heads. Interpret persistence of external failure as robustness to the tested construction only, not proof that all possible score-scale effects have been eliminated.

Both new protocols explicitly identify the work as post hoc, retain every declared result, and preserve the original frozen analysis. The additions strengthen a limited empirical mechanism argument. They do not create new field data, another independent external archive, verified plants, a validated cross-dataset pathology ontology, or a novel learning algorithm.
