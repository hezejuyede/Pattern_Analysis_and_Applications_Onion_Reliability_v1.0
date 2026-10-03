# Minimal cached-input closure for the retained regional and calibration training

This namespace makes the two earlier training entry points available within the release. It preserves their original code bytes, original relative directory structure, feature caches, required manifests/metadata, selected-C summary and source fold assignments. The regional protocol and its four prepared assignment tables are retained unchanged. No serialized model pickle is included. No original photograph or checkpoint is needed to fit these fixed feature-based heads.

The two large source/external prediction CSVs required by the calibration script already exist under the parent `supplementary_context/inputs`. They are shared rather than duplicated. `SHARED_PREDICTION_PATHS.json` binds their expected virtual training-tree paths to the actual release objects and records their original SHA-256. The adapter changes only resolution of these two files in the original calibration module's `sha` and `read_csv` operations. It does not change training, folds, model selection, calibration, thresholds or statistical functions. The regional entry point requires no numerical or file-reading replacement.

## Commands

Use the parent `requirements.txt` numerical environment, and quote paths containing spaces.

```bash
# Read-only verification: hashes, imports, every required feature matrix and identities.
python /path/to/v1.8/legacy_training/code/legacy_cli.py --root /path/to/v1.8/legacy_training --analysis check

# Actual regional refitting from the preserved original split/fold freeze, then bootstrap.
python /path/to/v1.8/legacy_training/code/legacy_cli.py --root /path/to/v1.8/legacy_training --analysis regional --out /path/to/new_empty_regional_rerun --mode all --n-jobs 1

# Actual source-only calibration reconstruction/refitting using shared fixed predictions.
python /path/to/v1.8/legacy_training/code/legacy_cli.py --root /path/to/v1.8/legacy_training --analysis calibration --out /path/to/new_empty_calibration_rerun --bootstrap 1000
```

The last two commands perform classifier fitting; they were **not executed again** while assembling this increment. The executed check only imported both unchanged scripts, ran their `--help` entry points, verified all input hashes (including the original regional freeze), aligned every cached matrix, checked finite values, and checked source/external identities and grouped source folds. See `INPUT_VALIDATION.json`. No new full-training equivalence claim follows from this input check.

The regional adapter copies the original prepared assignment files and protocol to a new output directory and uses the preserved `run` and `bootstrap` modes. `--mode run` fits heads only; `--mode bootstrap` can subsequently process the same output. The calibration script writes a new execution protocol for each rerun and refuses to overwrite an existing diagnostic. Its probabilities fitted to source out-of-fold scores are calibration-training values, not unbiased source-validation predictions.

The older calibration script reports its historical image-bootstrap comparison. For dependence-aware interval sensitivities, use the separate `supplementary_context` replay, which preserves the 547/700 component graphs. Do not interpret rerunning these existing archive analyses as a new external collection, independent plant cohort, or pathological confirmation.

`CACHED_INPUT_MANIFEST.json` lists the exact copied files, bytes and hashes. The archived scripts rely only on the listed numerical packages (`numpy`, `pandas`, `scipy`, `scikit-learn`, and `joblib`, installed with scikit-learn); there are no additional local Python modules. Source data-derived cache and metadata terms remain those in the parent data-licence notice. Original project code remains MIT licensed.
