# Data licences and attribution

This notice accompanies the v1.9 research release and preserves the v1.8 data-source terms. It distinguishes the original analysis code from third-party data, source metadata, pretrained models and derived records. Raw photographs and model weight files are not redistributed in this increment. This notice does not assert journal publication, journal acceptance, external peer review or completed author approval of a manuscript.

The v1.9 descriptive tables reuse the same frozen archive-derived records. No new data licence, field collection, pathological diagnosis or pretrained model distribution is implied. The new portable analysis implementation is original project code under MIT; source-derived labels and grouping records retain the upstream terms below.

## Scope of the repository MIT licence

The repository MIT licence applies to original analysis code only, unless a file carries another notice. It does not relicense COLD or PlantVillage photographs, their labels or grouping records, their derived feature caches, or third-party model implementations and weights. The applicable upstream notices below must remain with any redistributed data records or adaptations. Dataset attribution is required even when only derived features or metadata are shared.

| Material | Attribution and applicable terms |
| --- | --- |
| COLD-derived features, row metadata and associated analysis records | Retain the COLD attribution and CC BY 4.0 notice below; identify the selection, grouping and feature-extraction changes. |
| TOM2024 out-of-fold predictions and acquisition-day metadata in `supplementary_context`; TOM-derived handcrafted/ResNet feature caches, unified manifest and region metadata in `legacy_training` | Retain the TOM2024 attribution and CC BY 4.0 notice below. These are derived records, not redistributed photographs or newly collected data. |
| PlantVillage-derived features, labels and leaf-grouping metadata | Retain the PlantVillage attribution and CC BY-SA 3.0 notice below. This increment preserves those terms for the distributed PlantVillage-derived data files; the repository MIT licence is not a substitute. |
| Row-level predictions and combined tables | These are new computations, not new diagnoses. Keep each row's dataset provenance. Where source labels or grouping records are incorporated, retain the corresponding source terms. Combining records does not erase their separate licences. |
| Original analysis code | MIT, as stated in the repository code licence. |
| DINOv2 implementation and checkpoint | Apache-2.0 upstream. The checkpoint is referenced by URL and checksum, not included. Upstream licence and copyright notices remain applicable to any redistributed source code. |

The legal classification of a particular numerical embedding or aggregate as an adaptation can depend on its content and use. This package does not claim that converting an image to numbers removes all upstream obligations. Keeping the data notices and data families separate avoids a blanket claim that all release contents are MIT licensed.

## COLD onion data

Original data credit: Aishwarya M P and A. Padmanabha Reddy. Associated article: *Dataset of chilli and onion plant leaf images for classification and detection*, Data in Brief 54 (2024), 110524, [article DOI](https://doi.org/10.1016/j.dib.2024.110524).

The actual machine-readable input is the Project-AgML conversion of COLD at fixed [Hugging Face revision 72774f4bbb98b550d9f3be8c0d0e1e3d4e435fba](https://huggingface.co/datasets/Project-AgML/COLD_onion_leaf_disease_classification/tree/72774f4bbb98b550d9f3be8c0d0e1e3d4e435fba). Its [fixed dataset card](https://huggingface.co/datasets/Project-AgML/COLD_onion_leaf_disease_classification/blob/72774f4bbb98b550d9f3be8c0d0e1e3d4e435fba/README.md) states CC BY 4.0 and cites the original [Onion dataset, Mendeley Data V2](https://doi.org/10.17632/7nxxn4gj5s.2). The V2 DataCite record separately confirms CC BY 4.0. Licence: [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/legalcode).

The input snapshot contains 815 raw records and 4,502 augmented files. The selected experiment uses 3,076 supported views in 481 conservatively blocked components. Selection, image hashing, correspondence screening, grouping, embeddings, split assignments and predictions are additions made by this study. A component is not a verified plant or leaf identity. The version-specific manifest and image checksums identify the exact included records. Credit for Project-AgML's machine-readable conversion is additional to, and does not replace, credit for the original dataset.

The article's publication licence and the dataset licence concern different objects. Do not infer data permissions from the article's Crossref licence field. The older local reference to dataset DOI suffix `.3` is not used here: the fixed card cites `.2`, and the V2 metadata was verified on 3 October 2026.

## TOM2024 records in the supplementary context and legacy training

Dataset credit, following the fixed V1 DataCite creator record: Obed Appiah, Kwame Oppong Hackman, Belko Abdoul Aziz Diallo, Kehinde O. Ogunjobi, Valentin Ouedraogo, Momo Bebe and Diakalia Son. The original dataset is [TOM2024, Mendeley Data V1](https://doi.org/10.17632/3d4yg89rtr.1), published in 2024. Its [version-specific DataCite record](https://api.datacite.org/dois/10.17632/3d4yg89rtr.1), rechecked on 3 October 2026, states [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/legalcode). The later associated article is Appiah et al., *TOM2024: Datasets of tomato, onion, and maize images for developing pests and diseases AI-based classification models*, Data in Brief 59 (2025), 111357, [article DOI](https://doi.org/10.1016/j.dib.2025.111357).

The previously recorded image archive is [TOM2024-CATEGORYA-English.zip](https://ppedmas.org/datasets/images/TOM2024/TOM2024-CATEGORYA-English.zip), with study-recorded SHA-256 `6c110be15bc8bcdd4bc58aed277b3b81d66d8e2497505edb19639a60a8b76747`. That checksum identifies the prior download; this notice does not claim a new archive download or a byte-equivalence check against a different hosting service.

The `supplementary_context` increment contains out-of-fold row predictions and acquisition-day metadata for the actual analysis set of 1,643 eligible onion records spanning 103 filename-derived UTC acquisition days. It contains no TOM photographs. Selection, deduplication, the binary endpoint mapping, day grouping, fitted predictions and uncertainty calculations are study transformations. Retain the original dataset attribution and licence when reusing these records, and identify further changes. Filename-derived acquisition days are not verified farm, plant or diagnosis identifiers. The target distinguishes source-labelled healthy from visibly affected foliage, not laboratory-confirmed pathogens.

The corresponding COLD external prediction and dependency records in `supplementary_context` concern 813 exact-content-deduplicated raw images. They retain the COLD CC BY 4.0 attribution above. New bootstrap summaries computed from fixed predictions do not replace either dataset's original ownership, labels or licence and do not establish a new acquisition cohort.

The additional `legacy_training` objects retain the TOM-derived handcrafted and ResNet feature caches, unified row manifest and regional metadata used by the original training scripts. They are selected, transformed records from the same licensed archive, not new observations. The TOM-derived rows retain CC BY 4.0 and this attribution; COLD-derived rows and the external ResNet cache retain the COLD terms above. Matrix construction, endpoint mapping and saved allocation changes are documented in the original code and freezes. The original code's MIT licence does not replace these data terms.

## PlantVillage potato data and historical leaf map

Original data credit: Sharada P. Mohanty, David P. Hughes and Marcel Salathe. Associated article: *Using Deep Learning for Image-Based Plant Disease Detection*, Frontiers in Plant Science 7 (2016), 1419, [article DOI](https://doi.org/10.3389/fpls.2016.01419).

The author-maintained [Hugging Face dataset card at revision 9e97599868962bd0079b8db4b7f1efa9185fa1e7](https://huggingface.co/datasets/mohanty/PlantVillage/blob/9e97599868962bd0079b8db4b7f1efa9185fa1e7/README.md) records CC BY-SA 3.0. Licence: [Creative Commons Attribution-ShareAlike 3.0 Unported](https://creativecommons.org/licenses/by-sa/3.0/legalcode). This provenance is specific to the author-maintained source; licences attached to unrelated mirrors are not substituted.

The accompanying third-party dataset loader, `source_curation/potato/author_metadata_snapshot/plant_village.py`, has its own Apache-2.0 copyright header for the HuggingFace Datasets authors and script contributor. Its retained header and `source_curation/THIRD_PARTY_APACHE_2_0_LICENSE.txt` govern that code; they do not change the dataset's CC BY-SA 3.0 terms or make the loader original MIT code of this study.

The image bytes used here were checked against the original author's [GitHub revision 7f7ecc7e1eaca78107e3affe7cb5abd9427e139a](https://github.com/spMohanty/PlantVillage-Dataset/tree/7f7ecc7e1eaca78107e3affe7cb5abd9427e139a). The historical leaf map was verified against [GitHub revision 40789680ba2e6382608dba47b397688fcbd0d04a](https://github.com/spMohanty/PlantVillage-Dataset/tree/40789680ba2e6382608dba47b397688fcbd0d04a); its Git blob `cb04e3723d2ce15b0411483d71a60b5d536bc7f6` matches the author Hugging Face map at the revision above.

This study uses 2,152 original-colour photographs grouped into 538 author-identified potato leaves. The subset, feature extraction, matched replacements and predictions are study additions. The original leaf identifiers are retained as source records; they do not establish whole-plant identity or a new pathology review. Reusers must credit the original authors, link the licence, state their changes and preserve the applicable ShareAlike terms when distributing adaptations.

## DINOv2 fixed representation

Credit: Meta Platforms, Inc. and affiliates, and the DINOv2 contributors. The representation is the official pretrained DINOv2 ViT-S/14 without register tokens. [Official code at revision 7764ea0f912e53c92e82eb78a2a1631e92725fc8](https://github.com/facebookresearch/dinov2/tree/7764ea0f912e53c92e82eb78a2a1631e92725fc8) and its [Apache-2.0 licence](https://github.com/facebookresearch/dinov2/blob/7764ea0f912e53c92e82eb78a2a1631e92725fc8/LICENSE) identify the upstream implementation. All 190 files used from the pre-existing local source cache were subsequently matched to official Git blobs; that provenance check did not change the frozen extraction code.

The [official ViT-S/14 checkpoint](https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth) has SHA-256 `b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9`. The official DINOv2 README states Apache-2.0 for DINOv2 code and weights. Other model families appearing in the same repository can have different licences; their terms must not be generalized to this checkpoint or vice versa.

The shared 384-dimensional arrays are computed representations of the study images, not DINOv2 weights. The extraction records specify the deterministic evaluation transform and input hashes. The source-image notices above remain attached to their respective arrays. Possible overlap between public evaluation images and the undisclosed full pretraining collection is unknown; the licence notice does not establish pretraining independence.

## Preserving attribution in further releases

Retain this notice, the source identifiers, versioned manifests and changes to preprocessing or grouping. Do not describe the source datasets as newly collected, independently re-diagnosed or owned by the present study authors. A release identifier or reproducibility check is not evidence of editorial acceptance or biological validation.
