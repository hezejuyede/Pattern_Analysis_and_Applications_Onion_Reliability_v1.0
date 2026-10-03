# Portable replay of the retained external-dependence sensitivity

This namespace supplies the previously local D16 dependence sensitivity used for the retained locked onion transfer/calibration context. It contains six original input CSVs, seven original prepared objects, the unchanged original protocol freeze and analysis script, expected numerical outputs, and an explicit path adapter. It includes fixed predictions and metadata, not source photographs or trained classifier weights.

```bash
python /path/to/v1.8/supplementary_context/code/reproduce_external_context.py --root /path/to/v1.8/supplementary_context --out /path/to/new_empty_external_replay
```

Use the numerical environment in the parent `requirements.txt`. The output must be new or empty. The adapter copies the seven verified prepared files and original freeze, then calls the unchanged original `run` function. It replaces only that module's SHA-path resolver: each historical input path is mapped explicitly to the corresponding distributed file, and the original input SHA must still match. The original script's own hash and all prepared-file hashes remain enforced. No weight, metric, prediction, label, threshold or sampling algorithm is replaced. Historical freezes are retained byte-for-byte, not re-dated.

The replay evaluates all seven locked models on 813 external COLD images and 1,643 internal TOM out-of-fold images. It retains both dependence graphs (547 and 700 raw-content components), the 103 TOM acquisition-day groups, and all 3,000 frozen resampling draws per stream. It recomputes image-weighted point estimates, metric intervals, paired model contrasts, external-minus-internal gaps, and both calibration arms and their difference. It performs zero classifier fits. Seven CSV outputs and all nine bootstrap-distribution arrays are compared with the saved original results.

The executed verification in `verification/PORTABLE_EXTERNAL_CONTEXT_VERIFICATION.json` found zero numerical differences in every output CSV and bitwise equality in all nine distribution arrays. The original routine also performed 189 checks against scikit-learn's weighted metrics. This is a replay of the original statistics using preserved predictions and resampling weights; it is not an independent new implementation or a rerun of classifier training.

The dependence graphs are finite-retrieval, heuristic image-dependence graphs. The reported ranges are conditional sensitivity summaries for fixed predictions and incomplete dependence information. They are not confidence intervals for independently sampled plants, new field validation, or a pathology review. Conflicting labels remain recorded; neither labels nor point-estimate populations were changed.

## Regional and earlier model-training scope

Earlier regional model code, feature inputs and prediction records remain in the complete historical public repository. The regional entry point is `code/run_regional_robustness.py` there; see the [public repository](https://github.com/hezejuyede/Pattern_Analysis_and_Applications_Onion_Reliability_v1.0). This namespace does **not** rerun regional classifier training or reconstruct the earlier source-only calibration training process. It reproduces the frozen-prediction sensitivity described above. The original retained analysis history and its limitations must be read alongside this increment.

TOM- and COLD-derived records retain their respective source terms; the parent MIT code license is not a blanket data license. Consult the parent `DATA_LICENSES_AND_ATTRIBUTION.md` for attribution and source notices.
