# Source-curation code and evidence snapshot

This namespace preserves the reasoning-relevant inputs and original implementation behind the corrected source units, rather than publishing only the final analysis manifest. Original code, freeze JSON and original metadata bytes remain unchanged. Large CSV/JSON/text objects use reversible gzip transport; the manifest records the SHA-256 of both the original bytes and the distributed container.

The onion record contains the five original member/candidate/geometry/dependence-graph scripts, the corrected-component selection script, every non-image CSV/JSON from the archived member-lineage audit, and all corrected-selection records. It also includes the original family manifest required by the recorded selection inputs. Candidate lists, geometry outcomes, support edges, exclusion/selection decisions and the final selected manifest can therefore be inspected together. Contact-sheet manifests are retained, but photographs and contact-sheet raster images are excluded.

The potato record contains the original acquisition and feature-preparation code, metadata/author-Git verification scripts, the fixed author metadata snapshot and leaf map, and the acquisition/selection provenance. PlantVillage's `plant_village.py` is third-party Apache-2.0 code with its original header; the full licence accompanies it. Its source-data/leaf-map terms remain distinct from the original project code's MIT licence.

```bash
python /path/to/v1.8/source_curation/code/verify_source_curation.py --root /path/to/v1.8/source_curation
python /path/to/v1.8/source_curation/code/verify_source_curation.py --root /path/to/v1.8/source_curation --restore /path/to/new_empty_restored_curation
```

The second command restores exact uncompressed object bytes into the saved relative structure. It does not rewrite historical absolute paths inside the frozen files. `PATH_AND_TRANSPORT_MANIFEST.json` explicitly associates each object with its historical path, original SHA and transport SHA. `CODE_FREEZE_BINDINGS.json` identifies frozen script hashes that match included source files and flags historical scripts outside this curated subset.

**Reconstruction scope:** this is an archived evidence/code snapshot with verified byte transport, not a claim that the complete image-to-component pipeline was rerun offline. Recomputing retrieval, correspondence geometry or image features from raw data requires the upstream photographs, the relevant upstream fixed snapshots, compatible extraction/retrieval dependencies and explicit adaptation of historical input paths. Acquisition scripts require network access and upstream availability. Original curation scripts may depend on OpenCV/SIFT and other archived project files beyond the statistical-analysis requirements in the parent directory. The image data and pretrained weights are not included here.

The final matched-replacement statistics can be reproduced from the cached matrices and assignments in the parent release without those photographs. Image-derived dependence components remain conservative blocking units, not verified plants; author leaf IDs in the second archive are not whole-plant IDs. The source evidence does not establish new collection sites, pathological adjudication or absence of undetected dependencies.

See the parent data-licence notice for COLD and PlantVillage attribution. No source photograph, private email or journal correspondence is included.
