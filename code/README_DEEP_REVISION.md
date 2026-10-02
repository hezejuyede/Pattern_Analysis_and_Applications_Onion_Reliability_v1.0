# Deep revision: fixed-budget sibling substitution

This directory contains an additional, explicitly post hoc diagnostic. It was designed after the original study results were available, frozen before its own first model fit, and run without altering the original external models, predictions, thresholds or results. Its design is **not** retrospectively described as preregistration of the original study.

The diagnostic tests whether related training views improve only the test families whose siblings enter training, rather than improving performance on unrelated held-out families. It controls training image count, per-class count, family count, views per family and test anchors. It remains an experiment within one augmented image archive, not new external or biological validation.

## Reproduction environment

The verified run used Python **3.12.14**, NumPy **2.3.5**, pandas **3.0.1**, SciPy **1.18.1**, scikit-learn **1.8.0**, and threadpoolctl **3.7.0**, on Windows with one numerical-library thread. The matched-sibling suite is separate from the original v1.0 suite, whose scikit-learn 1.6.1 environment remains unchanged. Do not claim the two suites ran in the same software environment.

No image decoding, OpenCV, PyTorch, GPU, pretrained-weight download or network access is needed. The stored ResNet18 and color vectors are used directly. NumPy archives are loaded with `allow_pickle=False`.

From the `work/onion_crossdomain` project root, with the stated packages already available:

```powershell
python code/run_matched_sibling_intervention_cli.py --output-dir results/matched_sibling_intervention_reproduction
python code/verify_matched_sibling_rerun.py --canonical results/matched_sibling_intervention_v1 --rerun results/matched_sibling_intervention_reproduction
```

The output directory must be new or empty. The CLI wrapper changes only the output directory; it loads the frozen `run_matched_sibling_intervention.py` core without modifying its source or SHA-256. The core's default output is `results/matched_sibling_intervention_v1` and it refuses to overwrite an existing frozen design. There are no user parameters for selecting seeds, models, doses or favorable outcomes.

## Input files to preserve and publish

Paths below are relative to the project root. Keep their relative structure and contents exactly as supplied. `DEEP_REVISION_INPUTS.json` lists byte counts and SHA-256 for each required input and the executable scripts; the frozen `design.json` separately records the three scientific input hashes and immutable core script hash.

| File | Content | Required |
|---|---|---|
| `audit/cold_augmented_lineage/augmented_family_manifest.csv` | Original class, family ID, row ID, source path and image SHA-256 for all 4,502 augmented rows | Yes |
| `audit/cold_augmented_lineage/resnet18_embeddings.npz` | Frozen 512-dimensional vectors and paths; 5,317 rows include raw images, but only the 4,502 augmented paths in the manifest are selected | Yes |
| `features/cold_augmented_color_shortcut.npz` | Fixed 189-dimensional color vectors for 4,502 augmented rows | Yes |
| `code/run_matched_sibling_intervention.py` | Frozen experimental core | Yes |
| `code/run_matched_sibling_intervention_cli.py` | Reproduction entry point with `--output-dir` | Yes |
| `code/verify_matched_sibling_rerun.py` | Complete rerun comparator | For verification |

Raw image files are not required for reproducing this diagnostic from features. The inferred family-to-raw crosswalk is not used. Preserve the existing dataset attribution and license documentation in `audit/reproducibility/data_sources_and_licenses.md` with any release. This diagnostic does not create an alternative license for source data.

## Fixed design

There are 30 seeds, 20261002 through 20261031. Within each original class, select 20% of families as exposed probes P and 10% as always-unexposed sentinels S, using the rounding rules frozen in `design.json`. P comprises 85 healthy, 56 virus, 4 purple-blotch and 18 combined-blight families (163 total). S comprises 43, 28, 2 and 9 respectively (82 total). Each contributes one fixed test anchor. Only P families supply two extra sibling views for the intervention.

The remaining 571 D families each contribute two training views (1,142 images, with class counts 596/396/24/126). Match each P family without replacement to a same-class D donor. Across doses 0, 0.5 and 1, replace two donor views with two P sibling views. The realized P exposure fractions are 0, 81/163 and 1; S exposure is always zero. Training size, class counts, family count, two views per family and all 245 test anchors are invariant. Exact test files never enter training.

Two feature sets, two endpoints, three doses and 30 seeds produce 360 fits. Four-class classification is primary. Healthy versus any affected class is secondary and uses exactly the same memberships; it does not claim that the affected classes share a pathological diagnosis. Standardization is fitted on the training rows of each condition. Both models use class-balanced L2 logistic regression, `C=0.01`, `solver=lbfgs`, and `max_iter=10000`. There is no tuning on target labels.

## Outputs and interpretation

`predictions.csv.gz` stores all **88,200** predictions, including `evaluation_panel=probe|sentinel`. `family_membership.csv.gz` stores **24,480** family-level membership records and the exact row IDs of each anchor/view. `repeated_metrics.csv` has **720** panel-level result rows. `class_recall.csv`, `paired_summary.csv` and `localization_summary.csv` retain all endpoints, features and doses. The localization contrast is the probe performance change minus the simultaneous sentinel performance change.

The `brier` metric is the unnormalized sum of four one-hot squared errors for the four-class task, and the usual positive-class squared error for the binary task. These two scales must not be compared directly. Across-seed percentiles describe repeated-split variation, not confidence intervals from independent biological replicates. The same archive families recur across splits. Only four purple-blotch probe families and two sentinel families occur per seed.

All 90 membership conditions passed file-hash separation and budget checks. Both runs completed all 360 fits without convergence warnings. The first CPU run took **23.244 s**, and the fresh complete rerun took **23.968 s**; these are observed runtimes on this machine, not performance guarantees. The rerun comparator checks exact predictions and numerical results after parsing, excluding only timestamps and measured runtimes and the protocol hash changed by its timestamp. The results directory contains the retained reproduction report and file checksums.

Interpret the intervention as replacing unrelated same-class family views with related views under a specified finite training budget. It does not hold training identity fixed, and it does not establish a universal leakage effect. The four-class and binary effects differ and must be reported separately. Neither the mechanism experiment nor reproduction increases the number of independent external cohorts.
