# Online Resource 1 — methods and complete result tables

Attributing related-view gains in plant image evaluation with single-source and joint replacement controls

Xin Li, Bojian Guo and Asel Kartanova. Correspondence: lixin26@kstu.kg.

Sections S1-S6 document source identity, the original replacement controls and the retained transfer analyses. Section S7 specifies the frozen-representation and single-source extensions; S8 adds post-result descriptive support, weighting and error-headroom analyses. Original numerical tables are supplied as CSV files at saved precision; S5b is a four-decimal display table, with full precision in S5. The S2/S4 prose tables multiply the corresponding CSV fractions by 100 to display percentage points. File locations below are relative to this source. The computational and biological units must remain distinct.

## S1 Source evidence and inclusion

The onion source archive supplied 4,502 augmented files and 815 raw records representing 813 unique RGB contents. A known pair of filename prefixes shared an identical original photograph; filename prefixes therefore did not define source independence. The audit evaluated all augmented members against five retained candidates per member (22,510 comparisons), then added global frozen-ResNet cosine candidates across all raw labels. The global pass tested 16,985 previously untested augmented/raw pairs and 6,508 undirected raw/raw pairs, using per-member top-five raw retrieval and top-ten other-raw retrieval with stable content-hash tie ordering. Finite candidate retrieval cannot demonstrate the absence of further relationships.

SIFT [S1] used up to 1,000 features, contrast threshold 0.02, L2 nearest-neighbour matching and a 0.75 ratio test. Homography RANSAC [S2] used a four-pixel threshold. The corrected support records include normal and horizontally reflected query orientations. Geometric support required at least 15 inliers and an inlier fraction of at least 0.60. The stricter spatial screen required inlier convex-hull coverage of at least 0.05 in each image, median residual at most two pixels and 95th-percentile residual at most four pixels. Residuals refer to decoded grayscale image coordinates, without resizing. These are heuristic correspondence criteria, not plant identity or a complete augmentation ledger. The executable geometry and graph rules are preserved in supplement_provenance.

All positive links plus filename membership formed conservative blocking components. Cross-original-label components were quarantined, without majority relabelling. Within each remaining component, the raw content supporting the most distinct augmented members under the spatial screen was selected, breaking ties by lexical content hash. At least three unique supported views were required; only those supported by that reference entered the experiment. Other linked material remained excluded or blocked. This selected 481 components and 3,076 images. The selected set differs from the full archive; a component is not necessarily one photograph, leaf or plant. Selection did not consult classifier outcomes, but was designed after earlier archive analyses had revealed the lineage problem.

The independent PlantVillage potato task contains the complete eligible original-colour three-class subset: 538 author-identified leaves and 2,152 original photographs, four per leaf. Labels are 250 early-blight leaves, 250 late-blight leaves and 38 healthy leaves. The author leaf map at Hugging Face commit 9e97599868962bd0079b8db4b7f1efa9185fa1e7 has blob cb04e3723d2ce15b0411483d71a60b5d536bc7f6, identical to the original GitHub map at commit 40789680ba2e6382608dba47b397688fcbd0d04a. All photo/leaf assignments were independently matched to the original 2016 File Name and Leaf # CSV fields. All downloaded photo bytes matched the author Git objects at commit 7f7ecc7e1eaca78107e3affe7cb5abd9427e139a. All decoded RGB photos were unique. Processing variants were not new observations. These identifiers record leaves, not whole plants; there was no new pathology adjudication or prospective field acquisition.

## S2 Matched replacement, representations and uncertainty

The main text defines the selected-probe policy contrast and common-reference sentinel turnover. Frozen source assignments, exact removed donors and all actual training/test intersections accompany each task in its experiment directory. Each selected probe is mapped to distinct removable and reserved same-class sources. Exposure and sham delete the same donor; exposure inserts two related non-anchor views, while sham inserts two views from the reserved source. Reserved alternatives are absent from the zero fit. Class counts and training budgets are identical within each pair; all test anchors are fixed. Sources change together, so the estimate is a joint replacement-policy contrast allowing interference.

The onion task uses 97 probes (96 paired), 48 sentinels and 240 training sources/480 files per fitted arm; each allocation selects 48 paired probes. The potato task uses 108 paired probes, 54 sentinels and 268 training leaves/536 photos; each allocation selects 54 probes. Thirty seeds 20261003–20261032 each generate five complementary A/B pairs. Mean A/B contrasts are averaged over the five pairs within each split. The 30 split means and their 2.5th/97.5th percentiles describe design variation. They are neither independent biological replications nor plant-population confidence intervals. No test of significance, equivalence margin or universal leakage threshold was fitted.

The primary features are frozen torchvision ImageNet1K_V1 ResNet18 global-average-pool vectors (512 dimensions), using the standard evaluation transform. The colour comparator contains 189 features: each of nine BGR/HSV/Lab channels contributes a 16-bin histogram, mean, standard deviation and the 10th/50th/90th percentiles. Each logistic arm separately fits standardization and balanced L2 logistic regression with C=0.01, lbfgs and at most 10,000 iterations. Untuned cosine 1NN uses the same ResNet representation, with deterministic row order resolving ties. No task-specific test result selected a feature representation, penalty or retained model. Original multiclass labels are primary; healthy-versus-other refits are secondary. Endpoints differ in fitted tasks and class weights, so their difference is not a causal label-granularity effect.

Tables S1 retain every saved original three-model endpoint, panel, stratum and outcome; the new DINOv2 results are added in S7. Table S2 below gives both endpoints for all three models, in percentage points. Table S3 supplies the corresponding raw zero/sham/exposure BA on exactly the same selected masks. Error headroom is descriptive; no normalized benefit or between-task causal mechanism is inferred.

### Table S2. Selected-probe exposure-minus-sham contrast (percentage points)

| task | model | endpoint | difference | split_p025 | split_p975 |
| --- | --- | --- | --- | --- | --- |
| onion | colour_logit | binary_healthy_other | 12.481739 | 8.194565 | 17.788913 |
| onion | colour_logit | multiclass | 37.714989 | 25.064245 | 49.032838 |
| onion | resnet_cosine_1nn | binary_healthy_other | 26.391304 | 19.420870 | 32.147609 |
| onion | resnet_cosine_1nn | multiclass | 50.250038 | 33.817162 | 62.449027 |
| onion | resnet_logit | binary_healthy_other | 6.575072 | 2.721957 | 10.100870 |
| onion | resnet_logit | multiclass | 23.655187 | 8.336442 | 37.939559 |
| potato | colour_logit | binary_healthy_other | 3.830000 | -0.182500 | 10.632500 |
| potato | colour_logit | multiclass | 2.635556 | -0.731667 | 7.277500 |
| potato | resnet_cosine_1nn | binary_healthy_other | 1.721667 | -6.365000 | 13.300000 |
| potato | resnet_cosine_1nn | multiclass | 2.136667 | -3.350000 | 9.766667 |
| potato | resnet_logit | binary_healthy_other | 1.908333 | -0.327500 | 9.591250 |
| potato | resnet_logit | multiclass | 1.704444 | -0.170000 | 6.049167 |

### Table S4. Original-class contrasts for the primary multiclass model (percentage points)

| task | stratum | mean | split_p025 | split_p975 |
| --- | --- | --- | --- | --- |
| onion | healthy | 6.101449 | 1.619565 | 13.293478 |
| onion | iris_yellow_virus | 12.385965 | 4.539474 | 22.210526 |
| onion | purple_blotch | 52.333333 | 0.000000 | 100.000000 |
| onion | stemphylium_colletotrichum_leaf_blight | 23.800000 | 8.000000 | 45.650000 |
| potato | Potato___Early_blight | 0.213333 | -1.020000 | 1.310000 |
| potato | Potato___Late_blight | 0.733333 | -0.910000 | 2.910000 |
| potato | healthy | 4.166667 | 0.000000 | 18.187500 |

The onion purple-blotch stratum contains only 13 source components and one selected probe per allocation arm. Its large contrast and wide split range materially influence the equally weighted four-class estimate. These values do not represent field disease prevalence or evidence of a pathogen-specific mechanism. Original-label source support is given in main Table 1; full stratum outcomes remain in Tables S1.

## S3 Separately locked onion transfer

The separate binary transfer task contains 1,643 exact-deduplicated TOM2024 images from 103 filename-derived UTC days and 813 raw COLD images. Its binary endpoint is healthy versus dataset-labelled visibly affected foliage, not pathogen diagnosis. Five day-grouped outer folds estimate internal performance. Four inner folds choose C from 0.0001, 0.001, 0.01, 0.1, 1, 10 and 100. Each outer fold fits scaling, selection, Platt mapping and a Youden threshold solely within its development data. A separate five-fold source-only search selects the final C; the final head is refitted on all TOM. Its final calibrator and threshold use pooled outer out-of-fold TOM decision scores. The complete pipeline is then applied unchanged to COLD. A fixed-C calibration construction sensitivity is described in S4.

The primary ResNet representation and prevalence, colour and handcrafted baselines are retained concurrently. EfficientNet-B0, ConvNeXt-Tiny and Swin-T are post-primary architecture sensitivities: their COLD outcomes were retained irrespective of performance, and they did not use COLD for fitting, tuning, calibration or threshold selection. They are not another pristine external confirmation and are not substitutes for the three comparators in the matched replacement task.

Corrected external uncertainty uses two finite-retrieval dependence graphs: all positive geometric support (547 raw-content components) and the spatially screened graph (700 components). All 813 records enter both sensitivities. Each graph uses 3,000 fixed-prediction component bootstrap draws. Model contrasts use common paired weights; external-minus-internal gaps additionally resample TOM days independently. They are conditional uncertainty summaries for incomplete graphs and fixed models, not plant-level confidence intervals or full training uncertainty. The earlier image-unit interval appears only as a labelled illustration in Fig. S2 and is not the preferred uncertainty basis. Tables S5–S7 preserve all models, metrics, paired contrasts and transfer gaps.

### Table S5b. All seven external models under both corrected graphs

Entries are estimate [conditional lower, upper]; no model is omitted.

| graph | model | balanced_accuracy | auroc | brier |
| --- | --- | --- | --- | --- |
| all_high | color_shortcut_logit | 0.5253 [0.4971, 0.5583] | 0.6066 [0.5636, 0.6570] | 0.2752 [0.2517, 0.2958] |
| all_high | convnext_tiny_logit | 0.5829 [0.5579, 0.6051] | 0.7105 [0.6613, 0.7502] | 0.2507 [0.2237, 0.2822] |
| all_high | efficientnet_b0_logit | 0.6502 [0.6143, 0.6791] | 0.7205 [0.6739, 0.7564] | 0.2620 [0.2360, 0.2932] |
| all_high | handcrafted_logit | 0.5707 [0.5363, 0.6126] | 0.6028 [0.5604, 0.6486] | 0.2882 [0.2664, 0.3098] |
| all_high | prior_prevalence | 0.5000 [0.5000, 0.5000] | 0.5000 [0.5000, 0.5000] | 0.3208 [0.2978, 0.3440] |
| all_high | resnet18_logit | 0.6553 [0.6215, 0.6914] | 0.7345 [0.6868, 0.7749] | 0.2403 [0.2142, 0.2720] |
| all_high | swin_t_logit | 0.5621 [0.5450, 0.5790] | 0.7820 [0.7497, 0.8177] | 0.2824 [0.2499, 0.3179] |
| spatial_high | color_shortcut_logit | 0.5253 [0.4973, 0.5544] | 0.6066 [0.5643, 0.6503] | 0.2752 [0.2544, 0.2949] |
| spatial_high | convnext_tiny_logit | 0.5829 [0.5599, 0.6070] | 0.7105 [0.6714, 0.7517] | 0.2507 [0.2252, 0.2757] |
| spatial_high | efficientnet_b0_logit | 0.6502 [0.6177, 0.6829] | 0.7205 [0.6827, 0.7568] | 0.2620 [0.2356, 0.2890] |
| spatial_high | handcrafted_logit | 0.5707 [0.5343, 0.6076] | 0.6028 [0.5596, 0.6462] | 0.2882 [0.2657, 0.3106] |
| spatial_high | prior_prevalence | 0.5000 [0.5000, 0.5000] | 0.5000 [0.5000, 0.5000] | 0.3208 [0.3007, 0.3400] |
| spatial_high | resnet18_logit | 0.6553 [0.6209, 0.6891] | 0.7345 [0.6935, 0.7733] | 0.2403 [0.2155, 0.2668] |
| spatial_high | swin_t_logit | 0.5621 [0.5455, 0.5796] | 0.7820 [0.7463, 0.8157] | 0.2824 [0.2523, 0.3130] |

## S4 Source-only calibration construction

The sensitivity reuses the primary representation, the same five TOM day folds and the already selected final C=0.1. The legacy fold-selected values were [0.01, 0.01, 0.1, 0.1, 0.1]. One source score stream reproduces these fold penalties; the other uses final C=0.1 in every fold. Scaling is fitted within each fold's training data. Each source stream fits its own Platt map and Youden threshold. Both arms use the same final head trained on all TOM at C=0.1. Therefore this comparison varies score construction/calibration/thresholding, not the representation or final head. Source scores used to fit these calibrators are not presented as unbiased internal validation.

The external sample is identical in both arms. AUROC is identical because the common underlying score receives positive-slope monotone mappings; this is not an additional empirical replication. Fixed-C construction can still differ from a head fitted to all source data and does not remove all score-scale shift. This analysis was designed after earlier COLD results were known, without choosing a retained arm on COLD performance. Tables S8–S9 use the corrected two-graph paired component intervals, superseding the earlier image-bootstrap intervals.

### Table S9. Fixed-final-C calibration minus reconstructed legacy calibration

| variant | metric | estimate | lower | upper |
| --- | --- | --- | --- | --- |
| all_high | balanced_accuracy | -0.009171 | -0.021625 | 0.006271 |
| all_high | brier | -0.009304 | -0.012036 | -0.007194 |
| spatial_high | balanced_accuracy | -0.009171 | -0.022320 | 0.003295 |
| spatial_high | brier | -0.009304 | -0.011604 | -0.007150 |

## S5 Recovered-region sensitivity within TOM2024

Public-index metadata supplied unique regional associations for 1,416 images. The remaining 227 images with missing region were excluded from training and testing in every regional case. The recovered regions are Centre-Ouest, Centre-Sud and Plateau-Central. The primary sensitivity holds out each region and removes from training every image sharing a filename-derived UTC day with that test region. A region-only arm retains the identical tests but permits such training-day overlap. Training sample counts and composition therefore differ; the contrast is not a pure causal leakage effect.

All four models are retained: source prevalence, colour logistic, handcrafted logistic and frozen ResNet18 logistic. Within each development set, five day-grouped folds select C from the same seven-value grid, construct selected-C source out-of-fold scores, fit the Platt map and a finite Youden threshold. The source OOF scores used for calibration are not unbiased validation estimates. Source prevalence uses the training prevalence and a fixed 0.5 decision threshold. No held-out region labels select a model, parameter, calibrator or threshold. Fixed-prediction per-region intervals use 2,000 UTC-day cluster draws; aggregates jointly resample globally unique UTC days to retain cross-region dependence, rejecting draws that omit either endpoint state from a represented region. Inference is conditional on these three regions and fitted models.

Table S10 below includes all 24 region/variant/model cases. Tables S11–S15 retain equal-region/minimum-region summaries, their intervals, original-label recalls, class support and all paired contrasts. Region/day recovery is not a farm identifier, whole-plant identity, independently verified collection calendar or pathology review. This is a post hoc within-archive stress test, not a new external cohort.

### Table S10. All regional cases

| variant | region | model | train_n | n | balanced_accuracy | roc_auc | brier |
| --- | --- | --- | --- | --- | --- | --- | --- |
| region_day_disjoint | Centre-Ouest | prior_prevalence | 384 | 736 | 0.500000 | 0.500000 | 0.303774 |
| region_day_disjoint | Centre-Ouest | color_shortcut_logit | 384 | 736 | 0.767256 | 0.838138 | 0.173628 |
| region_day_disjoint | Centre-Ouest | handcrafted_logit | 384 | 736 | 0.675435 | 0.759347 | 0.191594 |
| region_day_disjoint | Centre-Ouest | resnet18_logit | 384 | 736 | 0.801487 | 0.953616 | 0.061066 |
| region_day_disjoint | Centre-Sud | prior_prevalence | 854 | 313 | 0.500000 | 0.500000 | 0.464941 |
| region_day_disjoint | Centre-Sud | color_shortcut_logit | 854 | 313 | 0.793376 | 0.879924 | 0.243571 |
| region_day_disjoint | Centre-Sud | handcrafted_logit | 854 | 313 | 0.749820 | 0.843092 | 0.290565 |
| region_day_disjoint | Centre-Sud | resnet18_logit | 854 | 313 | 0.874258 | 0.952150 | 0.148902 |
| region_day_disjoint | Plateau-Central | prior_prevalence | 760 | 367 | 0.500000 | 0.500000 | 0.210596 |
| region_day_disjoint | Plateau-Central | color_shortcut_logit | 760 | 367 | 0.794751 | 0.839485 | 0.128990 |
| region_day_disjoint | Plateau-Central | handcrafted_logit | 760 | 367 | 0.801152 | 0.893642 | 0.114369 |
| region_day_disjoint | Plateau-Central | resnet18_logit | 760 | 367 | 0.880165 | 0.960458 | 0.084670 |
| region_only | Centre-Ouest | prior_prevalence | 680 | 736 | 0.500000 | 0.500000 | 0.221366 |
| region_only | Centre-Ouest | color_shortcut_logit | 680 | 736 | 0.804363 | 0.894501 | 0.123817 |
| region_only | Centre-Ouest | handcrafted_logit | 680 | 736 | 0.741653 | 0.803223 | 0.141883 |
| region_only | Centre-Ouest | resnet18_logit | 680 | 736 | 0.855499 | 0.960455 | 0.054004 |
| region_only | Centre-Sud | prior_prevalence | 1103 | 313 | 0.500000 | 0.500000 | 0.450844 |
| region_only | Centre-Sud | color_shortcut_logit | 1103 | 313 | 0.786090 | 0.872639 | 0.231374 |
| region_only | Centre-Sud | handcrafted_logit | 1103 | 313 | 0.731854 | 0.842193 | 0.267883 |
| region_only | Centre-Sud | resnet18_logit | 1103 | 313 | 0.866905 | 0.950306 | 0.147819 |
| region_only | Plateau-Central | prior_prevalence | 1049 | 367 | 0.500000 | 0.500000 | 0.209140 |
| region_only | Plateau-Central | color_shortcut_logit | 1049 | 367 | 0.822079 | 0.859861 | 0.113869 |
| region_only | Plateau-Central | handcrafted_logit | 1049 | 367 | 0.802699 | 0.897340 | 0.107792 |
| region_only | Plateau-Central | resnet18_logit | 1049 | 367 | 0.883845 | 0.965010 | 0.072307 |

The primary day-purged ResNet equal-region mean BA is 0.851970, with a conditional interval [0.820712, 0.885926]. Its difference from colour is +0.066842 [0.019869, 0.119359], although the Centre-Ouest contrast crosses zero. Centre-Ouest healthy specificity is 57/88 despite AUROC 0.953616. These results separate ranking from a transferable operating threshold; they do not establish unseen-pathogen diagnosis. The original raw-COLD BA remains 0.655321.

## S6 Reproduction, accountability and figure

The retained original three-model, two-endpoint joint analysis produced 5,040 logistic fits, 2,520 nearest-neighbour endpoint evaluations and 1,160,460 prediction records without fitting warnings. Main model/endpoint contrasts were independently reconstructed from raw predictions; 24 selected logistic cases were independently refitted. This is a separate computational implementation, not a blinded human/pathology review and not a claim that every fit was independently refitted. Release v1.9.0 archives the current replacement experiments, frozen numerical inputs, predictions, code and verification records at https://github.com/hezejuyede/Pattern_Analysis_and_Applications_Onion_Reliability_v1.0/tree/v1.9.0/v1.9. The release README provides portable commands under code/portable; supplementary_context documents the retained regional, transfer and calibration analysis history. Original photographs must be obtained from their source archives under the recorded upstream terms. The portable smoke check covers replacement and single-source reconstruction with selected refits; it does not repeat all regional or calibration training. These limits are stated separately from the original experiment logs and the larger independent computational audit.

Figure S1 is supplied as ../figures/Fig_S1_Matched_Replacement_Binary.pdf and the corresponding EPS/600-dpi PNG. It displays the secondary healthy/other endpoint for all three original models and both tasks: selected-probe BA contrast and exposure-minus-sham sentinel correctness turnover. Points are means of 30 split means; bars are their 2.5th–97.5th percentiles, not biological confidence intervals. Both sentinel policies use the same zero-fit predictions as reference. It is not a causal comparison of binary and multiclass label granularity.

Provenance JSON files retain historical paths and hashes; relocating their bytes does not change when they were frozen. Geometry executable copies retain their historical dependencies. SUPPLEMENT_BUILD_v17_RETAINED.json identifies exact table origins, copied-byte hashes and derived table hashes. The original frozen scripts preserve the historical computation; the release's portable entry points resolve those paths through a recorded mapping to distributed numerical inputs.

## S7 Frozen-representation and single-source extension

### S7.1 Fixed DINOv2 representation

The extension was designed after the preceding results were known. One additional representation family was selected before new feature extraction or classifier outcomes: DINOv2 ViT-S/14 without register tokens, pretrained on LVD-142M. This is a different pretraining sensitivity, not exhaustive state-of-the-art coverage or a new representation-learning method. The official checkpoint SHA256 is `b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9`. All 190 frozen source files were subsequently verified byte-for-byte by Git object hash against official commit `7764ea0f912e53c92e82eb78a2a1631e92725fc8`, without changing extraction code or weights. The records are retained in supplement_provenance.

Images were converted to RGB, resized to a shortest side of 256 pixels using bicubic interpolation with antialiasing, centre-cropped to 224 by 224 pixels, scaled to floating-point tensors and normalized by ImageNet channel means (0.485, 0.456, 0.406) and standard deviations (0.229, 0.224, 0.225). The implementation was checked against the official classification-evaluation transform on fixed images and was pixel-identical. In evaluation mode, the final normalized CLS representation had 384 dimensions. No fine-tuning, gradients, stochastic augmentation or label-based image exclusions were used. The original output vectors were retained for logistic regression; only cosine 1NN uses unit-length vectors. Original image SHA256, manifest row identity and exact normalized path were checked for every cached row. All 5,228 selected images were retained. Re-extraction of 16 predetermined rows produced maximum absolute difference zero. Neither these checks nor use of a public checkpoint establishes that the study archives were absent from the pretraining corpus.

The DINOv2 joint experiment reused all 30 original source partitions, anchors, inserted views, mapped donor/alternative pairs and complementary allocations. It used the same per-arm standardization and balanced L2 logistic regression (C=0.01), alongside untuned cosine 1NN. Multiclass was the only new endpoint; the preserved three-model binary analysis in S2 was not silently extended. Feature dimension and pretraining family change together, so a difference from ResNet18 is not a causal estimate of either architectural or pretraining effects. Tables S16, S28 and S29 retain every new DINOv2 joint model, stratum, panel and metric at summary, arm and split levels.

### S7.2 Single-source replacement at a fixed background

The first ten original seeds, 20261003–20261012, were fixed before fitting the new analysis. Every paired probe was retained: 96 onion targets per split and 108 potato targets per split, giving 960 and 1,080 target/seed conditions. For each target separately, the experiment starts from the original zero-exposure training set. It removes only that target's mapped donor. Exposure inserts its two fixed non-anchor views; sham inserts the two fixed views of its previously reserved same-class alternative. All other training sources and views are identical within this E/H pair. The original image and class budgets remain 240 sources/480 images for onion and 268 leaves/536 images for potato. All probe and sentinel anchors stay fixed; an odd unpaired onion probe remains eligible as an unexposed evaluation anchor but is not an intervention target.

Five prespecified models were retained: ResNet18 logistic, colour logistic, DINOv2 logistic, ResNet18 cosine 1NN and DINOv2 cosine 1NN. All logistic parameters match S2. Both 1NN models explicitly use their respective normalized features and the same deterministic nearest-neighbour rule; no model is omitted because of its outcome. Multiclass was the only endpoint. Actual training memberships, supported source intersections, budgets, image hashes and complete probabilities are supplied in the indexed result files. Run commands require a complete input/code freeze and an independent prefit PASS review of that exact protocol hash.

The own-target result is the exposure-minus-sham correctness of one fixed anchor. It is averaged within original target class and then equally across target classes within a split. It becomes a class-balanced accuracy contrast only after this aggregation. For all other probes and for sentinels, original evaluation classes are weighted equally within target; targets are then averaged within target class and target classes receive equal weight. These steps prevent the numerous healthy/IYSV onion targets from automatically outweighing the rarer target classes. Tables S17–S20 retain target, target-class, split and summary outputs. Ten-split mean, minimum, maximum and sample standard deviation describe allocation variation; there are no plant-population confidence intervals or p-values.

### S7.3 Class-support and error-accounting sensitivities

The common-class rule uses at least 30 original manifest source units. This retains healthy, IYSV-labelled and pooled-blight-labelled onion classes and all three potato classes. It is a transparent support check under the approximately 20% probe/10% sentinel partition, not a theorem establishing adequate biological sample size. The threshold was fixed for the extension after earlier class results were already known; this is not a blinded confirmatory subgroup claim.

No reduced-class classifier was trained. In joint replacement, the sensitivity reweights the existing evaluated classes while retaining all original training/replacement sources. In single-source replacement, the reported common-class sensitivity restricts both eligible target classes and evaluation classes, with all full-task fits and probabilities unchanged. Full-class and per-class estimates remain available. The distinction matters particularly for sentinel comparisons, where changing the target-class mixture and the evaluation-class mixture are different operations. Orange series in main Figs. 2 and 3 use the corresponding definitions. For potato, the common-class and original-class results are identical.

Each target-panel record separately supplies sham-incorrect/exposure-correct and sham-correct/exposure-incorrect fractions. With identical class weights, their difference equals E-H balanced correctness; the former cannot exceed sham error mass. Raw zero, sham and exposure correctness, true-class probability changes and probability total variation are also retained. Turnover, gain, loss and wrong-to-wrong changes are each measured relative to the common zero fit. These identities and bounds were checked on every saved row. They do not establish why the tasks differ: limited error headroom, acquisition setting, labels and source relations are not independently manipulated.

### S7.4 Same-target comparison of replacement extents

For each target, model and seed in the first ten splits, the single-source E-H contrast was paired with the target's average contrast across the five joint allocations in which it was selected. There are 10,200 target/seed/model pairs across both tasks. Each joint allocation supplies exactly one selected arm per target. The zero predictions from both analyses agree at numerical precision for every model and anchor. Averaging follows the same target-class hierarchy in both arms. Tables S21–S24 contain every paired target, class, split and summary; Table S27 below is a rounded display in percentage points. Main Table 3 must use this matched ten-split comparison, not subtract the 30-split joint mean from the ten-split single mean.

### Table S27. Same-target comparison on the first ten splits (percentage points)

| task | model | single_pp | joint_pp | single_minus_joint_pp | difference_min_pp | difference_max_pp |
| --- | --- | --- | --- | --- | --- | --- |
| onion | Colour + LR | 30.1310 | 34.6841 | -4.5531 | -16.8021 | 10.9943 |
| onion | DINOv2 cosine 1NN | 44.1487 | 43.9506 | 0.1982 | -8.2815 | 12.1076 |
| onion | DINOv2 + LR | 18.4920 | 17.6899 | 0.8021 | -2.5835 | 8.4462 |
| onion | ResNet18 cosine 1NN | 51.1150 | 49.2413 | 1.8737 | -3.4153 | 8.6911 |
| onion | ResNet18 + LR | 25.1928 | 23.3383 | 1.8545 | -5.5858 | 9.0400 |
| potato | Colour + LR | 2.4167 | 1.6767 | 0.7400 | -5.6667 | 5.3333 |
| potato | DINOv2 cosine 1NN | 6.7833 | 7.7367 | -0.9533 | -4.0000 | 2.9000 |
| potato | DINOv2 + LR | 2.8333 | 2.5867 | 0.2467 | -3.4667 | 4.8667 |
| potato | ResNet18 cosine 1NN | 4.9500 | 4.5700 | 0.3800 | -3.5000 | 3.7667 |
| potato | ResNet18 + LR | 2.4833 | 1.7133 | 0.7700 | -2.1000 | 6.5333 |


The ResNet18 and DINOv2 onion single-minus-joint differences average +1.8545 and +0.8021 points; the colour difference is -4.5531 points. Potato differences are smaller in mean but vary in sign. Every model/task minimum-maximum interval for the difference spans zero. This does not support a universal claim that joint replacement amplifies or attenuates the target contrast, and it is not evidence of equivalence. The joint comparison also changes other selected sources' replacement policy, so the paired difference is a contextual sensitivity, not an isolated interaction mechanism. The single-source contrast itself is conditional on the recorded zero-exposure background, not a general causal effect of biological identity.

### S7.5 Execution, independent reconstruction and file map

The DINOv2 joint extension added 1,260 logistic fits, 1,260 nearest-neighbour model conditions and 386,820 predictions. The single-source extension added 12,300 logistic fits, 8,200 nearest-neighbour model conditions and 3,156,950 predictions. Both completed without fitting warnings. These are computational counts and do not add new plants, leaves, images or acquisition sites.

Independent implementations reconstructed all saved new contrasts from raw predictions. Predetermined checks independently refitted 12 DINOv2 joint logistic conditions and 84 single-source logistic conditions, and checked 12 joint and 56 single-source nearest-neighbour conditions. Reconstruction differences were at floating-point precision. This is computational cross-checking, not blinded human expert review; the 96 new logistic checks are a subset of fits, not all 13,560 new fits. The prior 24 logistic refits described in S6 concern earlier retained experiments.

`SUPPLEMENT_TABLE_INDEX.csv` gives the complete supplementary table count, row counts, sizes and hashes. `V18_COMPLETE_RESULT_FILE_INDEX.csv` indexes full prediction files, fit logs, allocation/source memberships and freezes without duplicating large prediction archives inside table files. That byte-preserved index records the historical v1.8 layout. V19_RETAINED_RESULT_LOCATIONS.csv maps each original file and hash to its actual object relative to the v1.9 public release root, including lossless transport formats. `SUPPLEMENT_BUILD_v18_RETAINED.json` records the unchanged 18 earlier CSVs and all new copied/derived evidence. Original sources and timing remain in the frozen JSON; historical paths are not rewritten as if they had always been portable. The public release preserves the original experimental scripts and their hashes under code/original. Use code/portable/reproduce.py for the retained replacement analyses and code/portable/reproduce_additional.py for S8; the release README specifies commands and verification scope. Historical scripts are provenance, not a claim of path-independent execution.

### S7.6 Retained transfer figure

Figure S2 is the earlier separate onion transfer context, preserved as `../figures/Fig_S2_External_Context.pdf` with EPS, 600-dpi PNG and source CSV. Its points use the same 813 locked COLD predictions throughout. The image-unit interval is an illustrative earlier analysis; the 547-component all-positive and 700-component spatial graphs provide the dependence sensitivities, each based on 3,000 fixed-prediction component bootstrap draws. These conditional intervals do not establish plant independence or a newly acquired field cohort. The underlying transfer results and all 18 earlier tables are retained; moving the figure to the supplement does not repair or conceal the external performance gap.

## S8 Post-result descriptive analysis of support, weighting and error headroom

### S8.1 Analysis chronology and repeated source units

This section was added after the v1.8 results and the subsequent revision comments were known. It reuses the original predictions, source assignments and fitted models; no classifier, representation, seed or eligibility threshold was selected or refitted. The input hashes and definitions were frozen on 3 October 2026 at 03:24:54 UTC. These are descriptive extensions, not a prospectively registered confirmation. All five models and both tasks are retained in Tables S40-S53.

The 960 onion target-within-split records in the single-source design involve 428 distinct components. The corresponding 1,080 potato records involve 480 author-recorded leaves. Individual targets recur in one to six onion splits and one to seven potato splits. The 20 purple-blotch target records come from ten components. These counts distinguish computational repetition from archive coverage; they do not establish independent plants. Table S40 gives class support and Table S41 retains each target's recurrence count. Probe counts include the odd, unpaired purple-blotch probe; intervention-target counts do not.

### Table S40 Source support and distinct single-source targets

The displayed class names abbreviate the original folder labels. “Target records” counts target-within-split occurrences over ten splits. Probes and sentinels are the fixed per-split panel sizes. The CSV also retains the paired-target counts and recurrence ranges.

| Task | Original class | Sources | Unique targets | Target records | Probes | Sentinels |
| --- | --- | --- | --- | --- | --- | --- |
| Onion | Healthy | 228 | 205 | 460 | 46 | 23 |
| Onion | IYSV-labelled | 190 | 171 | 380 | 38 | 19 |
| Onion | Purple blotch | 13 | 10 | 20 | 3 | 1 |
| Onion | Pooled leaf blight | 50 | 42 | 100 | 10 | 5 |
| Potato | Early blight | 250 | 222 | 500 | 50 | 25 |
| Potato | Late blight | 250 | 224 | 500 | 50 | 25 |
| Potato | Healthy | 38 | 34 | 80 | 8 | 4 |


### S8.2 Class recall, leave-one-class-out and frequency weighting

For each policy, class recall is calculated within its actual test panel before averaging complementary arms, allocation pairs and splits. In the single-source analysis, each target contributes its own correctness under its own fitted model pair; targets are averaged within their original class before classes receive equal weight. Tables S43-S44 give the resulting per-class recalls, replacement contrasts and error decomposition. Tables S45-S47 retain repeated-context event counts and joint confusion counts. The same source can occur in several contexts, so these counts must not be treated as independent subjects. Numeric predicted labels in S47 follow the class order in the corresponding frozen protocol.

The leave-one-class-out sensitivity excludes one original class from the evaluation average while leaving all training and predictions intact. It is applied to joint selected probes and single-source own targets, not to a newly trained reduced-class task. In particular, a prediction of the excluded class remains an error for a retained-class target. Tables S48-S49 retain all exclusions, including those that increase the mean contrast. For onion ResNet18, excluding purple blotch changes the 30-split joint contrast from 23.6552 to 14.0958 percentage points and the ten-split single-source contrast from 25.1928 to 13.5904 points. The matching DINOv2 contrasts are 21.1808 to 12.1300 and 18.4920 to 11.3227 points. These comparisons preserve the design and split set within each pair.

Two further descriptive averages use, respectively, original manifest source frequencies and the frequencies of actual selected targets. The latter is ordinary target-weighted accuracy, rather than class-balanced accuracy. For onion ResNet18, the joint contrasts are 11.6732 and 11.3958 points under these two weights; the single-source contrasts are 12.0953 and 11.7708 points. Tables S50-S51 contain every model and task. Archive class frequencies are not estimates of field prevalence. The common-class rule and all new weighting analyses remain post-result sensitivities; they do not replace the original full-class analysis.

### S8.3 Corrections, harms and available sham errors

For each paired prediction, correction means that ordinary same-class replacement was wrong and related-view insertion was correct. Harm means that ordinary replacement was correct and related-view insertion was wrong. Their difference is the net correctness contrast. Available error is the sham error fraction under exactly the same averaging weights. The correction fraction cannot exceed available error. Consequently, limited sham error places an upper bound on the positive net contrast, without identifying why the tasks have different error levels.

Table S52 reports both corrected/available-error and net-contrast/available-error ratios. Each is a ratio of the corresponding weighted aggregate quantities, not an average of per-target ratios. Corrected/available-error is a rescue fraction; the net ratio also accounts for newly harmed predictions. Both can be unstable when available error is small. Zero available error would leave the ratios undefined, not zero. No aggregate denominator in the displayed results was zero.

### Table S52 Weighted error headroom and correction decomposition

Available error and the last two columns are percentages. Corrected, harmed and net contrast are percentage points. Joint summaries average 30 complete splits; single-source summaries average ten. A zero single-source harm count is an observed property of these records, not a classifier monotonicity guarantee. The all-model and all-weight results are retained in S49.

| Task/design | Model | Available error (%) | Corrected (pp) | Harmed (pp) | Net (pp) | Corrected/error (%) | Net/error (%) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Onion / joint | DINOv2 + LR | 29.7667 | 21.6570 | 0.4762 | 21.1808 | 72.7556 | 71.1560 |
| Onion / joint | ResNet18 + LR | 42.9175 | 25.0327 | 1.3775 | 23.6552 | 58.3274 | 55.1178 |
| Onion / single | DINOv2 + LR | 25.8678 | 18.4920 | 0.0000 | 18.4920 | 71.4864 | 71.4864 |
| Onion / single | ResNet18 + LR | 41.5355 | 25.1928 | 0.0000 | 25.1928 | 60.6537 | 60.6537 |
| Potato / joint | DINOv2 + LR | 2.8667 | 2.0389 | 0.1089 | 1.9300 | 71.1240 | 67.3256 |
| Potato / joint | ResNet18 + LR | 3.2622 | 1.9056 | 0.2011 | 1.7044 | 58.4128 | 52.2480 |
| Potato / single | DINOv2 + LR | 4.2000 | 2.8333 | 0.0000 | 2.8333 | 67.4603 | 67.4603 |
| Potato / single | ResNet18 + LR | 3.8500 | 2.4833 | 0.0000 | 2.4833 | 64.5022 | 64.5022 |


For joint ResNet18, the fraction of available sham errors corrected is 58.3274% in onion and 58.4128% in potato, whereas net error reduction relative to available error is 55.1178% and 52.2480%. DINOv2 gives correction fractions of 72.7556% and 71.1240%, and net fractions of 71.1560% and 67.3256%. These numerically similar rescue fractions show why a small absolute potato contrast need not imply a small response among its remaining errors. They do not establish statistical equivalence or isolate a causal ceiling mechanism. Task provenance, class structure, image processing and fitted difficulty also differ.

### S8.4 Cross-split dependence and conditional Monte Carlo error

Table S42 compares all 45 pairs of the ten single-source splits. Onion splits share 12-26 intervention targets and 109-134 zero-exposure training components; potato splits share 16-29 targets and 119-145 training leaves. Moreover, 39-58 onion targets and 42-63 potato targets from one split belong to the zero-exposure training set of the other split. Thus repeated target rows are not independent biological observations, and clustering solely by target and split does not by itself establish a population variance model. Standard multiway-cluster theory requires independence for observations that share no clustering dimension and sufficiently many clusters [S4]; neither condition follows from these archive records. The overlapping-training issue is also relevant to naive cross-validation uncertainty [S5].

A separate, narrower uncertainty target is the mean response over the specified randomized allocation design with the archive, source graph, frozen features and numerical pipeline held fixed. Treating the independently initialized random streams as independent complete allocation draws, the Monte Carlo standard error of that mean is the sample standard deviation of the split summaries divided by the square root of their number [S3]. Table S53 reports this quantity in percentage points, using sample SD with denominator R-1, R=30 for joint replacement and R=10 for single-source replacement. It uses all original classes and includes every model and both tasks. Each complete split, including its within-split allocations and target averaging, is one Monte Carlo replicate; individual targets or allocation arms are not separate replicates for this calculation.

### Table S53 Conditional allocation Monte Carlo standard errors

Mean, split SD and MCSE are in percentage points. These MCSEs describe numerical allocation sampling on the fixed archive under the stated independent-stream assumption. They are not plant-population standard errors, confidence intervals or tests of statistical significance. The reported split percentile and minimum-maximum ranges elsewhere describe the distribution of allocation outcomes and serve a different purpose.

| Task/design | Model | Splits | Mean (pp) | Split SD (pp) | MCSE (pp) |
| --- | --- | --- | --- | --- | --- |
| Onion / joint | Colour + LR | 30 | 37.7150 | 7.9295 | 1.4477 |
| Onion / joint | DINOv2 1NN | 30 | 46.0879 | 8.2921 | 1.5139 |
| Onion / joint | DINOv2 + LR | 30 | 21.1808 | 8.3878 | 1.5314 |
| Onion / joint | ResNet18 1NN | 30 | 50.2500 | 8.6097 | 1.5719 |
| Onion / joint | ResNet18 + LR | 30 | 23.6552 | 9.2819 | 1.6946 |
| Onion / single | Colour + LR | 10 | 30.1310 | 9.5646 | 3.0246 |
| Onion / single | DINOv2 1NN | 10 | 44.1487 | 6.4354 | 2.0350 |
| Onion / single | DINOv2 + LR | 10 | 18.4920 | 9.5058 | 3.0060 |
| Onion / single | ResNet18 1NN | 10 | 51.1150 | 9.1119 | 2.8814 |
| Onion / single | ResNet18 + LR | 10 | 25.1928 | 8.2070 | 2.5953 |
| Potato / joint | Colour + LR | 30 | 2.6356 | 2.6407 | 0.4821 |
| Potato / joint | DINOv2 1NN | 30 | 6.4744 | 3.1868 | 0.5818 |
| Potato / joint | DINOv2 + LR | 30 | 1.9300 | 2.4106 | 0.4401 |
| Potato / joint | ResNet18 1NN | 30 | 2.1367 | 3.3762 | 0.6164 |
| Potato / joint | ResNet18 + LR | 30 | 1.7044 | 1.9442 | 0.3550 |
| Potato / single | Colour + LR | 10 | 2.4167 | 2.0295 | 0.6418 |
| Potato / single | DINOv2 1NN | 10 | 6.7833 | 3.4561 | 1.0929 |
| Potato / single | DINOv2 + LR | 10 | 2.8333 | 3.8976 | 1.2325 |
| Potato / single | ResNet18 1NN | 10 | 4.9500 | 4.8196 | 1.5241 |
| Potato / single | ResNet18 + LR | 10 | 2.4833 | 2.9994 | 0.9485 |


Repeating the random allocation can reduce Monte Carlo error without adding independent source acquisitions. In a total-variance decomposition, the conditional allocation variance of the mean decreases with the number of independent splits, whereas variability of the archive-specific mean across new datasets does not. The current calculation does not estimate that second component. It also does not propagate uncertainty in unrecorded plant identities, diagnosis, geometric retrieval completeness or pretrained-image overlap. With only ten single-source draws, estimated Monte Carlo precision itself should be interpreted cautiously; no p-values or population confidence intervals are added.

### S8.5 Recalculation and file units

The new analysis reads the four complete joint prediction files and the saved single-source target, membership and contrast records. It reconstructs paired correctness events, checks the original joint contrasts against the frozen figure source, and confirms that all original input hashes remain unchanged. No fitting occurs in this replay. A separate portable implementation reads all six retained prediction files, independently reconstructs the single-source own-target events, and reproduces all 14 added tables to numerical precision without importing the original analysis implementation. ADDITIONAL_ANALYSIS_FREEZE.json records the input set and chronology; READ_ONLY_SUMMARY_QA.json records the original replay checks, and ADDITIONAL_PORTABLE_VERIFICATION.json records the independent check and its input hashes. These checks reproduce recorded computations; they do not provide another biological sample or model fit.

Unless their column names explicitly state percentage points or percent, performance proportions in S43-S51 use the 0-1 scale. Identifier, seed and event-count fields are not proportions. S40-S42 contain counts or unitless overlap proportions. S52 explicitly names percentage units; S53 uses percentage points. The summary files retain means, sample split SDs and full ranges. None of the event tables supplies additional independent biological sample size. All 39 earlier CSV files and Sections S1-S7 are retained alongside these 14 additions.


## Supplementary method references

[S1] Lowe DG (2004) Distinctive Image Features from Scale-Invariant Keypoints. International Journal of Computer Vision 60:91-110. https://doi.org/10.1023/b:visi.0000029664.99615.94

[S2] Fischler MA, Bolles RC (1981) Random sample consensus: a paradigm for model fitting with applications to image analysis and automated cartography. Communications of the ACM 24:381-395. https://doi.org/10.1145/358669.358692

[S3] Owen AB (2013) Monte Carlo theory, methods and examples. Chapters 1-2: Introduction and Simple Monte Carlo. Author-maintained electronic book. https://artowen.su.domains/mc/

[S4] Cameron AC, Gelbach JB, Miller DL (2011) Robust Inference With Multiway Clustering. Journal of Business & Economic Statistics 29:238-249. https://doi.org/10.1198/jbes.2010.07136

[S5] Bengio Y, Grandvalet Y (2004) No Unbiased Estimator of the Variance of K-Fold Cross-Validation. Journal of Machine Learning Research 5:1089-1105. https://www.jmlr.org/papers/v5/grandvalet04a.html

## Complete machine-readable table index

- `supplement_tables/S10_regional_all_24_cases.csv`: 24 rows

- `supplement_tables/S11_regional_aggregates.csv`: 16 rows

- `supplement_tables/S12_regional_intervals.csv`: 100 rows

- `supplement_tables/S13_regional_source_label_recalls.csv`: 88 rows

- `supplement_tables/S14_regional_class_support.csv`: 81 rows

- `supplement_tables/S15_regional_paired_contrasts.csv`: 60 rows

- `supplement_tables/S16_onion_dinov2_joint_summary.csv`: 640 rows

- `supplement_tables/S16_potato_dinov2_joint_summary.csv`: 512 rows

- `supplement_tables/S17_onion_single_target_panel_contrasts.csv`: 28500 rows

- `supplement_tables/S17_potato_single_target_panel_contrasts.csv`: 32400 rows

- `supplement_tables/S18_onion_single_target_class_split_means.csv`: 1050 rows

- `supplement_tables/S18_potato_single_target_class_split_means.csv`: 900 rows

- `supplement_tables/S19_onion_single_split_means.csv`: 300 rows

- `supplement_tables/S19_potato_single_split_means.csv`: 300 rows

- `supplement_tables/S1_onion_all_model_endpoint_panel_class_summaries.csv`: 1920 rows

- `supplement_tables/S1_potato_all_model_endpoint_panel_class_summaries.csv`: 1536 rows

- `supplement_tables/S20_onion_single_summary.csv`: 720 rows

- `supplement_tables/S20_potato_single_summary.csv`: 720 rows

- `supplement_tables/S21_context_all_target_contrasts.csv`: 10200 rows

- `supplement_tables/S22_context_target_class_split_means.csv`: 650 rows

- `supplement_tables/S23_context_split_means.csv`: 200 rows

- `supplement_tables/S24_context_summary.csv`: 60 rows

- `supplement_tables/S25_joint_all_five_models_plot_source.csv`: 40 rows

- `supplement_tables/S26_single_all_five_models_plot_source.csv`: 40 rows

- `supplement_tables/S27_context_display.csv`: 10 rows

- `supplement_tables/S28_onion_dinov2_joint_arm_contrasts.csv`: 12000 rows

- `supplement_tables/S28_potato_dinov2_joint_arm_contrasts.csv`: 9600 rows

- `supplement_tables/S29_onion_dinov2_joint_split_means.csv`: 1200 rows

- `supplement_tables/S29_potato_dinov2_joint_split_means.csv`: 960 rows

- `supplement_tables/S2_selected_probe_policy_contrast.csv`: 12 rows

- `supplement_tables/S3_selected_probe_absolute_BA.csv`: 12 rows

- `supplement_tables/S40_source_target_support.csv`: 7 rows

- `supplement_tables/S41_unique_target_frequency.csv`: 908 rows

- `supplement_tables/S42_cross_split_overlap.csv`: 90 rows

- `supplement_tables/S43_single_per_class_split_recall_headroom.csv`: 350 rows

- `supplement_tables/S44_joint_per_class_split_recall_headroom.csv`: 1050 rows

- `supplement_tables/S45_single_event_counts_not_independent_samples.csv`: 350 rows

- `supplement_tables/S46_joint_repeated_context_event_counts.csv`: 1050 rows

- `supplement_tables/S47_joint_repeated_context_confusion_records.csv`: 5061 rows

- `supplement_tables/S48_all_common_loco_split_results.csv`: 2200 rows

- `supplement_tables/S49_all_common_loco_summary.csv`: 110 rows

- `supplement_tables/S4_primary_original_class_policy_contrasts.csv`: 7 rows

- `supplement_tables/S50_frequency_weighted_split_results.csv`: 800 rows

- `supplement_tables/S51_frequency_weighted_summary.csv`: 40 rows

- `supplement_tables/S52_headroom_interpretation_percentage_units.csv`: 8 rows

- `supplement_tables/S53_conditional_allocation_mcse.csv`: 20 rows

- `supplement_tables/S5_locked_transfer_metric_intervals.csv`: 63 rows

- `supplement_tables/S5a_locked_transfer_point_estimates.csv`: 42 rows

- `supplement_tables/S5b_locked_external_display_all_models.csv`: 14 rows

- `supplement_tables/S6_locked_transfer_paired_models.csv`: 36 rows

- `supplement_tables/S7_locked_transfer_gaps.csv`: 42 rows

- `supplement_tables/S8_calibration_arms.csv`: 8 rows

- `supplement_tables/S9_calibration_changes.csv`: 4 rows
