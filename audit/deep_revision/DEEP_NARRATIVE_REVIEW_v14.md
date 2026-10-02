# Deep narrative review v1.4

Audit UTC: 2026-10-02T08:04:00.518264+00:00

Read-only audit of a changing working snapshot; hashes are in the JSON companion. Only the audit files were written.

Status: MINOR_REFERENCE_ITEMS_REMAIN

## Verified
- English abstract: 224 whitespace-delimited words.
- References: 38 bibliography entries; all 38 cited in both languages; no missing, duplicate, unused or language-specific citation keys.
- New Apicella and Guignard author names, titles, years, volume/issue and DOI metadata match Crossref and publisher records.
- New intervention and calibration point estimates and percentile/bootstrap endpoints match the original CSV values at reported three-decimal precision.
- No remaining EUVP name or claim of first discovering leakage / introducing grouping theory; the two task branches and post hoc chronology are explicit.
- ResNet localization range crossing zero and the distinction between a small sentinel change and equivalence are retained.

## Concrete residual issues

Historical v1.0.0 may remain when explicitly identified as the original evidence. Current text correctly distinguishes the cumulative v1.1.0. Public verification of the new tag is a separate release check, not a narrative error.

All four equations now have valid numbered callouts in both languages, including the grouped English callout for equations (3) and (4).

### N08 P2 - full_text.md
Supplemental OOD Fig. S1 is not explicitly called out in the body.

Action: Add Online Resource 1, Fig. S1 / 在线资源 1，图 S1 to the crop-level OOD paragraph.

### N07 P3 - full_text_zh.md
Proposed Table 4 caption remains as an unused planning item; actual main text calls Tables 1-3 only.

Action: Remove the stale caption scaffold or explicitly map its contrasts into the stated supplementary table. Do not generate an uncited fourth main table.

### N08 P2 - full_text_zh.md
Supplemental OOD Fig. S1 is not explicitly called out in the body.

Action: Add Online Resource 1, Fig. S1 / 在线资源 1，图 S1 to the crop-level OOD paragraph.

## Numerical coverage

- PASS four-class ResNet probe dose 0.0: mean=0.642
- PASS four-class ResNet probe dose 1.0: mean=0.795
- PASS four-class ResNet sentinel dose 0.0: mean=0.613
- PASS four-class ResNet sentinel dose 1.0: mean=0.605
- PASS four-class ResNet localization: mean_localization_contrast=0.162, split_p025=-0.014, split_p975=0.355
- PASS four-class colour probe dose 0.0: mean=0.417
- PASS four-class colour probe dose 1.0: mean=0.701
- PASS four-class colour localization: mean_localization_contrast=0.282, split_p025=0.086, split_p975=0.506
- PASS binary ResNet probe gain: mean_exposed_minus_clean=0.038
- PASS binary resnet18 localization: mean_localization_contrast=0.036, split_p025=-0.024, split_p975=0.103
- PASS binary color_shortcut localization: mean_localization_contrast=0.089, split_p025=0.014, split_p975=0.179
- PASS reconstructed_legacy_mixed_C balanced_accuracy: estimate=0.655
- PASS fixed_final_C_OOF balanced_accuracy: estimate=0.646
- PASS calibration difference balanced_accuracy: difference=-0.009, ci_low=-0.022, ci_high=0.003
- PASS reconstructed_legacy_mixed_C brier: estimate=0.240
- PASS fixed_final_C_OOF brier: estimate=0.231
- PASS calibration difference brier: difference=-0.009, ci_low=-0.011, ci_high=-0.007

## Limits
This review does not assign a publication score or guarantee acceptance. It checks the bilingual scientific account and reference closure; rendered manuscript layout and eventual immutable repository availability remain separate release checks.
