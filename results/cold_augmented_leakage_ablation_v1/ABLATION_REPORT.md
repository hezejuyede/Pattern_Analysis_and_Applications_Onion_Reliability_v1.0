# COLD augmentation-family leakage sensitivity analysis

This analysis is separate from the locked TOM2024-to-raw-COLD external evaluation. It diagnoses how the reported performance of a fixed classifier changes when the unit of splitting is corrected; it does not validate deployment.

The augmented archive contains 4,502 files but only 816 filename-defined augmentation families. The most extreme class is purple blotch: 735 files arise from 18 families. Under ordinary stratified file splitting, a mean 98.5% of test files had a sibling from the same family in training. Family-grouped splitting enforced zero overlap.

Across 30 fixed repeated seeds, the frozen ResNet18 plus fixed-C regularized logistic model had mean file-level balanced accuracy 0.832 with ordinary file splits and 0.675 with family-grouped splits. The fixed color-shortcut model changed from 0.702 to 0.437. `tables/paired_split_inflation.csv` gives the seed-paired difference distribution, and `tables/repeated_per_class_recall.csv` exposes class-specific effects.

Family IDs are taken directly from the complete `dr_<parent>_<random>.jpg` filename structure. No uncertain raw-image crosswalk is used. The IYSV archive has 282 augmented families versus 281 raw files; this mismatch does not affect within-augmented family separation and is not resolved by forced parent matching.
