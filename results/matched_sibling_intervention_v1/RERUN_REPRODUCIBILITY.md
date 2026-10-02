# Complete rerun verification

**PASS: all 360 models were refitted from the frozen feature archives.** All 88,200 prediction rows, 24,480 family membership rows, 720 panel metric rows, 90 membership checks, class recalls, paired effects and localization contrasts are exactly equal after CSV parsing. No tolerance or metric exclusion was needed. Both original artifact manifests passed every file hash and size check.

Original runtime: 23.244 s; rerun: 23.968 s. Model and input hashes are unchanged.

Only measured per-fit/total runtime, execution timestamps, and the protocol JSON hash affected by its new timestamp are excluded. DESIGN/RESULTS narrative timestamps and runtime statements are descriptive, not numerical scientific outputs. The full retained per-artifact comparisons are in `rerun_comparison.csv`; the rerun's duplicate files can be deleted after verification without losing scientific results.
