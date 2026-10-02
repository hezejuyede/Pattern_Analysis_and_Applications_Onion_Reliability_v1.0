# Data sources versions and permitted use

## TOM2024

- Dataset: TOM2024, tomato, onion and maize pest and disease images.
- Dataset DOI and version used for provenance metadata: https://doi.org/10.17632/3d4yg89rtr.1
- Data article: Appiah et al., *Data in Brief* 59 (2025) 111357, https://doi.org/10.1016/j.dib.2025.111357
- Download used in this analysis: `https://ppedmas.org/datasets/images/TOM2024/TOM2024-CATEGORYA-English.zip`
- Local archive SHA-256: `6c110be15bc8bcdd4bc58aed277b3b81d66d8e2497505edb19639a60a8b76747`
- Licence recorded by DataCite for the dataset DOI: Creative Commons Attribution 4.0 International.
- Relevant analysis subset: 3,044 onion files. The primary supervised analysis uses 2,402 deterministic raw-photo representatives after excluding 19 source groups with conflicting labels; 1,647 of those representatives are eligible for the foliar binary endpoint.
- Redistribution rule for the final package: prefer download instructions, the source manifest and hashes. If image redistribution is included, retain the dataset attribution and CC BY 4.0 notice.

## COLD onion leaf data

- Dataset article: Aishwarya MP and Reddy AP, *Data in Brief* 54 (2024) 110524, https://doi.org/10.1016/j.dib.2024.110524
- Primary data DOI: https://doi.org/10.17632/7nxxn4gj5s.3
- Machine-readable snapshot used for the experiment: `https://huggingface.co/datasets/Project-AgML/COLD_onion_leaf_disease_classification`
- Snapshot licence: Creative Commons Attribution 4.0 International.
- Raw configuration used as the locked external dataset: 815 rows, comprising 426 healthy, 281 Iris yellow virus, 90 pooled Stemphylium/Colletotrichum leaf blight and 18 purple blotch images.
- Integrity note: the machine-readable snapshot contains 815 raw images, while the data article reports 864. The manuscript must cite and report the exact snapshot rather than silently substituting the article total.
- Augmented configuration: 4,502 rows. It is excluded from inferential analyses unless every augmented file can be mapped to one raw parent with auditable certainty.
- Redistribution rule for the final package: provide the download script, version information, manifest and hashes; do not imply that the authors created or newly annotated these images.

## Digital Green Crop Disease Expert Annotations

- Dataset: `https://huggingface.co/datasets/DigiGreen/Crop_Disease_Images`
- Citation supplied by the dataset: Digital Green (2026), *Crop Disease Expert Annotations*.
- Licence: Creative Commons Attribution 4.0 International.
- Dataset context: 1,026 expert annotations over 989 farmer-supplied photographs, 74 crops, collected primarily in India from 2 December 2024 to 30 March 2025. Agronomists reviewed the images; 37 images received more than one expert review.
- Subsets used here: 24 onion images as a descriptive operational case series and 225 deterministically sampled non-onion images as an out-of-scope set.
- Interpretation rule: Digital Green “Healthy” records may contain described water, nutrient, salinity or tip-drying stress. They are not a clean pathogen-negative reference set and must not be pooled into primary disease-specific estimates.
- Redistribution rule for the final package: preserve Digital Green attribution and CC BY 4.0 terms. The analysis package should ordinarily include the selected-row manifest and download script rather than a second copy of all images.

## Pretrained feature extractor

- Model: torchvision ResNet18 with `ResNet18_Weights.IMAGENET1K_V1`.
- Local weight SHA-256: `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`.
- The model is used as a frozen generic feature extractor. It is not represented as a newly developed architecture.

