# TOM2024 split-severity sensitivity

This diagnostic uses all 2,053 nonconflict, exact-deduplicated foliar files from 1,643 timestamp/raw-photo groups and 103 UTC acquisition days. It uses a fixed frozen ResNet18 feature extractor, fixed-C regularized logistic regression, and fold-local standardization. No result from COLD is used.

Across 30 fixed repeated seeds, mean balanced accuracy was 0.939 for naive file splits, 0.936 for raw-photo-group splits, and 0.928 for acquisition-day-group splits. Mean test source-group contamination and acquisition-day contamination are reported beside every metric in `tables/aggregate_split_summary.csv`.

Exact duplicate bytes were collapsed before every scheme, so the contrast concerns residual acquisition-event and batch dependence rather than identical-file leakage. A filename timestamp is treated as an acquisition event, and UTC day as a coarser acquisition-batch proxy. Neither is a verified plant identity; therefore even the day-grouped result may retain plant- or site-level dependence.
