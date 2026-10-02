# Data sources, versions, licences and integrity records

No third-party source image is included here. The repository distributes only
manifests, row metadata, hashes, derived numerical features, predictions and
aggregate outputs. Users who need to repeat feature extraction must retrieve
the public datasets from their owners and accept the upstream terms.

## TOM2024

- Dataset DOI: <https://doi.org/10.17632/3d4yg89rtr.1>
- Data article: <https://doi.org/10.1016/j.dib.2025.111357>
- Archive used: <https://ppedmas.org/datasets/images/TOM2024/TOM2024-CATEGORYA-English.zip>
- Archive SHA-256 recorded for this study:
  `6c110be15bc8bcdd4bc58aed277b3b81d66d8e2497505edb19639a60a8b76747`
- Licence recorded by DataCite: CC BY 4.0.
- Analysed endpoint: dataset-assigned healthy versus visibly affected onion
  foliage. It is not a laboratory-confirmed pathogen endpoint.

## COLD onion leaf data

- Data DOI: <https://doi.org/10.17632/7nxxn4gj5s.3>
- Data article: <https://doi.org/10.1016/j.dib.2024.110524>
- Machine-readable source used:
  <https://huggingface.co/datasets/Project-AgML/COLD_onion_leaf_disease_classification>
- Configuration used for locked external testing: `raw`, 815 downloaded rows;
  813 images remained after exact-hash deduplication.
- Configuration used only for the split-unit ablation: `augmented`, 4,502
  files mapped to 816 filename-defined augmentation families.
- Licence stated by the machine-readable source: CC BY 4.0.

## Digital Green Crop Disease Expert Annotations

- Dataset: <https://huggingface.co/datasets/DigiGreen/Crop_Disease_Images>
- Snapshot metadata retained under `metadata/`.
- Study subsets: 24 onion photographs for a descriptive case series and 225
  deterministically sampled non-onion photographs for an OOD stress test.
- Licence stated by the source: CC BY 4.0.

## Integrity records

- `results/reliability_benchmark_v1/audit/development_external_sha256.csv`
  records the public-source bytes and SHA-256 values used in the unified
  manifest.
- `results/reliability_benchmark_v1/audit/primary_development_external_sha256.csv`
  records the 1,643 TOM development images, four removed TOM duplicate files,
  and 813 COLD external images used at the primary analysis gate.
- `audit/*/download_manifest.csv` records the selected source rows and hashes.
- `results/reliability_benchmark_v1/audit/evidence_freeze_manifest.csv`
  freezes 64 code, feature and result artifacts in the normalized public
  layout. Its SHA-256 is
  `c0dcc9206b8ea6c836ba10c54b06f2152531efd9930563cf64674047104d9c47`.

Run `python scripts/download_public_data.py --dataset all` to reconstruct the
downloaded image layout. Network endpoints can change; the script fails rather
than silently accepting a TOM archive with a different SHA-256. After download,
run `python scripts/verify_downloaded_images.py` before extracting features.
