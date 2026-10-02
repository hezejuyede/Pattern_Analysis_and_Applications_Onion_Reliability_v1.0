# Evidence gate after locked cross-country evaluation

**Decision: HOLD — the current evidence does not justify a submission-ready or deployment-level claim.**

The design passed the core independence test: all choices were made on TOM2024 acquisition-day-grouped validation, and COLD remained computationally excluded from fitting. The a-priori primary ResNet18 plus regularized logistic regression reached COLD balanced accuracy 0.655 (95% group-bootstrap CI 0.626–0.685) and AUROC 0.735. Its disorder sensitivity was only 0.444. Balanced accuracy fell by 0.272 from TOM nested validation to COLD; this is evidence of material transport failure, not broad cross-country robustness. EfficientNet-B0, ConvNeXt-Tiny, and Swin-T are reported as secondary architecture sensitivities and do not replace the primary based on COLD.

The 24-image Digital Green onion case series produced balanced accuracy 0.458 for the same locked ResNet model and has a wide interval. It remains descriptive. For non-onion OOD images, source-neighbour distance reached AUROC 0.795, but FPR at 95% OOD sensitivity was 0.873; this operating point is not suitable for a safety claim.

The defensible contribution is therefore a leakage-controlled reliability audit showing how high acquisition-day-grouped internal performance fails to transport across public onion datasets. A full plant-disease journal paper still needs pathology review of the broad endpoint and a sufficiently large, prospectively collected or independently re-adjudicated field cohort. Language polishing or a more complex network cannot repair these evidence gaps.

Machine-readable paired comparisons, transport gaps, outer-fold results, and label-level detection rates are in the `tables` directory.
