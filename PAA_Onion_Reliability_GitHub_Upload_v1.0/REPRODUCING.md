# Reproducing the compact release

Run commands from the repository root with Python 3.13.3.

## Offline evidence verification

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/verify_slim_repository.py
python scripts/recompute_slim_evidence.py
```

The evidence script recomputes all binary metric point estimates and the stored
group-bootstrap limits from the released per-image predictions. It also checks
paired external comparisons, the locked TOM folds, dataset-hash separation and
the aggregate COLD augmentation-family and TOM split-severity tables. It does
not claim to re-hash source image bytes that are not redistributed.

## Reconstruction from the public image sources

```bash
python scripts/download_public_data.py --dataset all
python scripts/verify_downloaded_images.py
python code/extract_features.py --model handcrafted
python code/extract_features.py --model resnet18 --batch-size 32
python code/extract_challenger_features.py --model efficientnet_b0 --batch-size 32
python code/extract_challenger_features.py --model convnext_tiny --batch-size 32
python code/extract_challenger_features.py --model swin_t --batch-size 32
python code/run_reliability_benchmark.py --bootstrap 1000 --n-jobs 4
python code/analyze_reliability_results.py
python code/run_augmented_family_leakage_ablation.py --repetitions 30
python code/run_tom_split_severity_sensitivity.py --repetitions 30
```

PyTorch and torchvision use platform-specific wheels, so install the current
CPU or CUDA build compatible with the locked software versions before feature
extraction. Source datasets retain their own licences; see `DATA_SOURCES.md`.

The full internal archive also preserves frozen feature matrices and expanded
membership ledgers. Their omission here reduces upload size without changing
the released predictions, metric tables, identifiers or source hashes.
