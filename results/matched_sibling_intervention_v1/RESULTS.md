# Matched sibling-substitution diagnostic

**Status: completed; explicitly post hoc.** Every prespecified seed, exposure dose, feature model and endpoint is retained. This diagnostic does not alter the original locked external evaluation.

All 90 membership checks passed. Each condition has 163 identical probe anchors and 82 identical sentinel anchors, 1142 training images, 571 training families, exactly two views per training family and identical class counts. There is zero exact-file or SHA overlap between training and test. Realized probe-family exposure fractions are 0, 81/163 (0.49693), and 1; sentinel exposure remains zero.

| Endpoint | Features | Panel | Clean BA | Full-exposure BA | Paired difference | Split 2.5–97.5% range |
|---|---|---|---:|---:|---:|---|
| four_class | resnet18 | probe | 0.641869 | 0.794604 | +0.152736 | +0.024977 to +0.280478 |
| four_class | resnet18 | sentinel | 0.613440 | 0.604597 | -0.008842 | -0.131125 to +0.123807 |
| binary_healthy_affected | resnet18 | probe | 0.820671 | 0.858746 | +0.038074 | -0.000673 to +0.091073 |
| binary_healthy_affected | resnet18 | sentinel | 0.815643 | 0.817780 | +0.002137 | -0.042196 to +0.056485 |
| four_class | color_shortcut | probe | 0.416930 | 0.700792 | +0.283862 | +0.141888 to +0.413174 |
| four_class | color_shortcut | sentinel | 0.415252 | 0.417183 | +0.001932 | -0.150111 to +0.136805 |
| binary_healthy_affected | color_shortcut | probe | 0.640933 | 0.720085 | +0.079153 | +0.032289 to +0.143188 |
| binary_healthy_affected | color_shortcut | sentinel | 0.661151 | 0.651103 | -0.010048 | -0.063186 to +0.048151 |

| Endpoint | Features | Localization contrast delta_P minus delta_S | Split 2.5–97.5% range |
|---|---|---:|---|
| binary_healthy_affected | color_shortcut | +0.089201 | +0.014295 to +0.178660 |
| four_class | color_shortcut | +0.281931 | +0.086081 to +0.505960 |
| binary_healthy_affected | resnet18 | +0.035938 | -0.024237 to +0.103191 |
| four_class | resnet18 | +0.161578 | -0.013612 to +0.355463 |

The percentile ranges describe variation across overlapping repeated splits, **not** confidence intervals based on independent biological observations. Four purple-blotch probe families and two sentinel families per seed limit class-specific precision. Related-family replacement changes source identity and dependence while controlling quantity and class composition; it does not establish a universal effect size or validate disease diagnosis. Results from the two-view budget must not replace or be merged numerically with the original all-view file-versus-family contrast. All lineage keys are archive filename families; the inferred raw-parent crosswalk was not used.

CPU runtime: 23.24 s; 360 fits; 0 warnings; no convergence warnings. scikit-learn 1.8.0; no new image decoding or feature extraction.
