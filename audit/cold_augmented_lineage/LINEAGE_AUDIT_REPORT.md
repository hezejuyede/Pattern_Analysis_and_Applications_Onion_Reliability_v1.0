# COLD augmented lineage audit and leakage-control decision

**Audit date:** 2026-09-30  
**Dataset:** Project-AgML/COLD_onion_leaf_disease_classification, Hugging Face `raw` and `augmented` configurations  
**Audit unit:** downloaded image row, archive filename, augmentation family, and raw exact-content cluster

## Decision

The fixed COLD augmented archive may be used only with an **augmentation-family-grouped split**. A random split of its 4,502 files is invalid for performance estimation because multiple transformed views of one source image would enter both training and test sets. Exact-file deduplication does not solve this problem: all 4,502 augmented files have unique SHA-256 and 256-bit dHash values, yet they collapse to only 816 archive-derived filename families.

The safest design for the primary experiment is to split the 815 raw files first and create augmentation on the training fold only. The fixed augmented archive is useful for a prespecified leakage-sensitivity experiment: compare a naive file split with the family-safe split and quantify the optimism caused by sibling leakage. It is not an independent external validation set.

The family key is an **archive-derived source-image family**, not a plant, plot, farm, acquisition event, or biological replicate. No metadata supports those stronger interpretations.

## What is directly observed

The parquet files preserve original image names. Every augmented filename matches `dr_<parent_index>_<random_suffix>.jpg`. The first integer is contiguous within every class and groups visually and feature-similar transformed views. All 4,502 augmented rows were assigned without a parse failure:

| Class | Raw files | Filename families | Augmented files | Files per family |
|---|---:|---:|---:|---|
| Iris yellow virus | 281 | 282 | 1,272 | 138 families with 4; 144 with 5 |
| Stemphylium/Colletotrichum leaf blight | 90 | 90 | 1,217 | 43 families with 13; 47 with 14 |
| Healthy | 426 | 426 | 1,278 | 426 families with 3 |
| Purple blotch | 18 | 18 | 735 | 1 family with 39; 1 with 40; 16 with 41 |
| **Total** | **815** | **816** | **4,502** | — |

The class-specific augmentation rate is extremely uneven. Purple blotch is expanded from 18 raw files to 735 transformed files, while healthy is expanded threefold. Consequently, 735 is not the effective purple-blotch sample size. Any uncertainty interval, split, weighting scheme, or error analysis that treats those files as independent would be anti-conservative.

Frozen ImageNet ResNet-18 features provide a separate content-consistency check. Median within-family mean pairwise cosine similarity is 0.912 for healthy, 0.917 for Iris yellow virus, 0.914 for the combined leaf-blight class, and 0.896 for purple blotch. These results support use of the filename prefix as a dependency group. They do not prove plant-level identity.

## Materialization integrity

The local row order was checked against the two parquet files. Hugging Face's cached image endpoint re-encoded many JPEGs, so byte-level equality is not an appropriate sole integrity test:

- 556 local files are byte-identical to parquet image bytes;
- 4,761 differ at the byte level after cached JPEG materialization;
- all 5,317 decoded images match the corresponding parquet row under conservative content checks;
- maximum RGB mean absolute error is 4.092 on a 0–255 scale;
- maximum 256-bit dHash Hamming distance is 9;
- no row failed the decoded-content correspondence gate.

The two raw exact-duplicate pairs are both in the Iris yellow virus class: `103.jpg`/`104.jpg` and `149.jpg`/`150.jpg`. Thus the 815 raw files contain 813 unique SHA-256 contents.

## Raw-parent crosswalk

The public archive contains no explicit family-to-raw crosswalk. A cautious inferential crosswalk was therefore attempted in two stages:

1. retrieve up to 30 same-class raw exact-content clusters using the cosine similarity of frozen ResNet-18 features to the family centroid;
2. geometrically verify candidates with SIFT matches and a RANSAC homography over up to five deterministic family members.

No one-to-one assignment was forced. Candidate links with weak geometry, small margins, or conflicts were retained as ambiguous. After a global collision safeguard:

| Raw-link status | Families | Interpretation |
|---|---:|---|
| High geometric support | 653 | Supported but still inferential |
| Moderate geometric support | 51 | Supported with weaker margin |
| Ambiguous candidate | 49 | Do not treat as a raw-parent identity |
| Ambiguous global collision | 22 | Multiple supported families selected a raw cluster beyond its exact-file capacity |
| Unresolved | 41 | No adequate geometric link |
| **Total** | **816** | **704 supported; 112 not supported** |

A complete one-to-one crosswalk is impossible from these public artifacts: the augmented Iris yellow virus set exposes 282 family indices but the raw configuration contains only 281 files. The supported 704 links are useful for audit and selected sensitivity checks; they must not be presented as source-provided ground truth.

## Required experimental controls

1. Use `family_id` from `augmented_family_manifest.csv` as the grouping variable for every fixed-archive split, inner validation split, bootstrap, and permutation test.
2. Never place a raw image and a confidently linked augmented family in different folds. Because 112 family-to-raw links remain uncertain, do not combine the entire raw and fixed augmented configurations in a primary split. Prefer raw-first splitting with on-the-fly training augmentation.
3. Report both file count and source-image-family count. Do not describe either count as biological replication.
4. Keep all hyperparameter selection and calibration inside training groups. The target-country or target-dataset test labels must remain untouched until the locked final evaluation.
5. Use a genuinely external source-disjoint dataset for the paper's main generalization claim. COLD raw versus COLD augmented is an augmentation study, not external validation.
6. For purple blotch, state that the base evidence is 18 source files. Use parent-level confidence intervals and sharply limit biological or deployment claims.
7. Include the naive-file versus family-safe result only as a leakage demonstration. Do not choose the final model from the naive result.

## Reproducible artifacts

- `../../code/audit_cold_augmented_lineage.py`: complete audit and cautious crosswalk reconstruction.
- `lineage_summary.json`: machine-readable counts, thresholds, software versions, and summary results.
- `augmented_family_manifest.csv`: every augmented row with archive filename, family key, local hash, and crosswalk status.
- `raw_parquet_manifest.csv`: raw rows, original filenames, exact-content hashes, and materialization checks.
- `family_to_raw_mapping.csv`: family-level raw candidates and confidence status.
- `top_candidate_scores.csv`: five strongest geometrically ranked candidates per family.
- `family_feature_consistency.csv`: within-family ResNet-18 similarity statistics.
- `basic_image_audit/`: exact-hash, dHash, dimensional, and readability audit.
- `HuggingFace_README_snapshot.md`: dataset-card snapshot used for the public row and class counts.
- `artifact_hashes.csv`: SHA-256 ledger for the two source parquets, audit script, report, and tabular outputs.

Re-run from the project root with:

```powershell
python work/onion_crossdomain/code/audit_cold_augmented_lineage.py --resnet-candidates 30 --sift-family-members 5
```

The audit requires Python, NumPy, Pillow, OpenCV with SIFT, PyArrow, PyTorch, and torchvision. Network access is needed only if the torchvision ResNet-18 weights are not already cached.

