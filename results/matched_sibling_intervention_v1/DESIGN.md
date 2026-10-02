# Fixed-budget matched sibling substitution

Protocol frozen before this experiment was run: 2026-10-02T07:50:44.082204+00:00.

This is an explicitly **post hoc** diagnostic designed after the original split-sensitivity and external results were known. It is not a new confirmatory external evaluation.

For each of 30 seeds (20261002–20261031), select 20% of each class's families as probe P families (85 healthy, 56 virus, 4 purple blotch, 18 blight). Draw one probe anchor and two distinct siblings from each. Then select 10% of the original class's families as sentinel S families from the remainder (43, 28, 2, 9; 82 total); draw one anchor each and exclude all sentinel-family views from every training condition. Every remaining D family contributes two training views. Pair each probe family without replacement with a random same-class D donor family. At exposure doses 0, 0.5 and 1, replace the chosen donor's two training views with the paired probe family's siblings. All 245 P/S test anchors, training images (1142), training families (571), class counts and two views per training family are fixed across conditions. Half exposure uses integer rounding and reports the realized proportion.

Use the existing frozen ResNet18 and color features, training-only standardization and class-balanced L2 logistic regression with C=0.01. Primary endpoint: original four-class archive labels. Secondary endpoint: healthy versus any affected label with exactly the same memberships. Report all seeds and outcomes. Compare full minus zero exposure in four-class balanced accuracy separately in probes and sentinels; the paired localization contrast is delta_P minus delta_S. Seed percentiles are split-variation ranges, not confidence intervals for biological replication.

Same-class substitution controls training quantity and label composition. It intentionally changes source identity and dependence, so the effect is conditional on the specified archive and two-view training budget. Filename families are source-image groups, not identified plants. No inferred raw-image crosswalk is used. No external model, calibration or threshold is altered.

Machine-readable input and script hashes, endpoint and model settings are frozen in `design.json`.
