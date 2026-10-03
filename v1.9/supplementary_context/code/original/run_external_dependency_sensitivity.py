"""Fixed-prediction sensitivity to candidate-derived image dependence groups.

This does not fit models, change labels, alter point estimands, or claim plant-level
confidence intervals. Prepare freezes inputs, alignment and all resampling weights
before any metric is computed; run verifies that freeze before producing results.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, roc_auc_score
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = 'resnet18_logit'
MODELS = ('color_shortcut_logit', 'convnext_tiny_logit', 'efficientnet_b0_logit',
          'handcrafted_logit', 'prior_prevalence', REFERENCE, 'swin_t_logit')
ARMS = ('reconstructed_legacy_mixed_C', 'fixed_final_C_OOF')
METRICS = ('balanced_accuracy', 'auroc', 'brier')
VARIANTS = ('all_high', 'spatial_high')
SCOPE = ('Percentile resampling sensitivity conditional on frozen predictions and '
         'finite-retrieval, heuristic image-dependence graphs; not independent plant '
         'or field-cohort confidence intervals. Labels and image-weighted point '
         'estimates are retained. Unobserved dependencies remain possible.')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def align_predictions(path, expected_n):
    frame = pd.read_csv(path)
    assert set(frame.model) == set(MODELS)
    assert not frame.duplicated(['sample_id', 'model']).any()
    ids = sorted(frame.sample_id.unique())
    assert len(ids) == expected_n and len(frame) == expected_n * len(MODELS)
    for column in ['foliar_binary', 'relative_path', 'analysis_sha256', 'original_label']:
        assert frame.groupby('sample_id')[column].nunique(dropna=False).eq(1).all(), column
    metadata = frame.drop_duplicates('sample_id').set_index('sample_id').loc[ids].reset_index()
    y = metadata.foliar_binary.to_numpy(dtype=np.int8)
    assert set(y) == {0, 1}
    def matrix(column):
        return frame.pivot(index='sample_id', columns='model', values=column).loc[ids, list(MODELS)].to_numpy(dtype=np.float64).T
    probability, prediction, threshold = (matrix(c) for c in ['calibrated_probability_disorder', 'predicted_disorder', 'locked_threshold'])
    assert np.isfinite(probability).all() and np.isfinite(threshold).all()
    assert ((probability >= 0) & (probability <= 1)).all()
    assert np.array_equal(prediction, (probability >= threshold).astype(np.int8))
    return metadata, y, probability, prediction.astype(np.int8), threshold


def resampling_counts(groups, y, rng, replicates):
    order = sorted(set(groups))
    indexes = {g: i for i, g in enumerate(order)}
    codes = np.asarray([indexes[g] for g in groups], dtype=np.int32)
    group_y0 = np.bincount(codes, weights=(y == 0).astype(int), minlength=len(order))
    group_y1 = np.bincount(codes, weights=(y == 1).astype(int), minlength=len(order))
    counts = rng.multinomial(len(order), np.full(len(order), 1 / len(order)), size=replicates)
    rejected = 0
    invalid = (counts @ group_y0 <= 0) | (counts @ group_y1 <= 0)
    while invalid.any():
        rejected += int(invalid.sum())
        if rejected > 100000:
            raise RuntimeError('Too many single-class bootstrap draws')
        counts[invalid] = rng.multinomial(len(order), np.full(len(order), 1 / len(order)), size=int(invalid.sum()))
        invalid = (counts @ group_y0 <= 0) | (counts @ group_y1 <= 0)
    assert counts.max() <= np.iinfo(np.uint16).max
    return counts.astype(np.uint16), codes, order, rejected


def input_paths(args):
    root = args.root.resolve()
    assert args.raw_identity and args.all_high_components and args.spatial_high_components
    return {
        'external': root / 'results/reliability_benchmark_v1/predictions/cold_locked_external.csv',
        'internal': root / 'results/reliability_benchmark_v1/predictions/tom_nested_oof.csv',
        'calibration': root / 'results/calibration_scale_sensitivity_v1/external_predictions.csv',
        'raw_identity': args.raw_identity.resolve(),
        'all_high_components': args.all_high_components.resolve(),
        'spatial_high_components': args.spatial_high_components.resolve(),
    }


def prepare(args):
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'PROTOCOL_FREEZE.json').exists():
        raise RuntimeError('Refuse replacing a frozen sensitivity analysis')
    paths = input_paths(args)
    ext, ey, ep, ed, et = align_predictions(paths['external'], 813)
    tom, ty, tp, td, tt = align_predictions(paths['internal'], 1643)
    assert all(np.unique(row).size == 1 for row in et)
    days = tom.acquisition_day_utc.astype(str).to_numpy()
    timestamps = tom.source_group.str.extract(r'(1\d{12})$', expand=False).astype('int64')
    derived_days = pd.to_datetime(timestamps, unit='ms', utc=True).dt.strftime('%Y-%m-%d').to_numpy()
    assert np.array_equal(days, derived_days) and len(set(days)) == 103
    raw = pd.read_csv(paths['raw_identity'], dtype=str)
    assert not raw.local_path.duplicated().any()
    ext_align = ext[['sample_id', 'relative_path', 'original_label', 'foliar_binary', 'analysis_sha256']].merge(
        raw[['local_path', 'raw_content_id', 'local_sha256', 'label']],
        left_on='relative_path', right_on='local_path', how='left', validate='one_to_one')
    assert ext_align.raw_content_id.notna().all() and not ext_align.raw_content_id.duplicated().any()
    assert (ext_align.analysis_sha256 == ext_align.local_sha256).all()
    assert (ext_align.original_label == ext_align.label).all()
    ext_align['position'] = np.arange(len(ext_align))
    tom_align = tom[['sample_id', 'relative_path', 'original_label', 'foliar_binary', 'analysis_sha256', 'acquisition_day_utc']].copy()
    tom_align['position'] = np.arange(len(tom_align))
    calibration = pd.read_csv(paths['calibration'])
    assert set(calibration.arm) == set(ARMS) and not calibration.duplicated(['arm', 'sample_id']).any()
    cp, cd, ct = [], [], []
    for arm in ARMS:
        part = calibration[calibration.arm.eq(arm)].set_index('sample_id').loc[ext.sample_id]
        assert len(part) == 813 and np.array_equal(part.foliar_binary.to_numpy(dtype=np.int8), ey)
        assert np.array_equal(part.analysis_sha256.to_numpy(), ext.analysis_sha256.to_numpy())
        p = part.calibrated_probability_disorder.to_numpy(dtype=np.float64)
        d = part.predicted_disorder.to_numpy(dtype=np.int8)
        t = part.source_only_threshold.to_numpy(dtype=np.float64)
        assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
        assert np.array_equal(d, (p >= t).astype(np.int8))
        cp.append(p); cd.append(d); ct.append(t)
    assert np.max(np.abs(cp[0] - ep[MODELS.index(REFERENCE)])) <= 1e-12
    assert np.array_equal(cd[0], ed[MODELS.index(REFERENCE)])
    arrays = {'external_y': ey, 'external_probability': ep, 'external_prediction': ed, 'external_threshold': et,
              'internal_y': ty, 'internal_probability': tp, 'internal_prediction': td, 'internal_threshold': tt,
              'calibration_probability': np.asarray(cp), 'calibration_prediction': np.asarray(cd), 'calibration_threshold': np.asarray(ct)}
    weights, orders, rejected, component_records, conflict_records = {}, {}, {}, [], []
    for variant in VARIANTS:
        graph = pd.read_csv(paths[variant + '_components'], dtype=str)
        assert not graph.raw_content_id.duplicated().any()
        joined = ext_align.merge(graph[['raw_content_id', 'component_id']], on='raw_content_id', how='left', validate='one_to_one').sort_values('position')
        assert joined.component_id.notna().all() and np.array_equal(joined.sample_id, ext.sample_id)
        groups = joined.component_id.astype(str).to_numpy()
        counts, codes, order, invalid = resampling_counts(groups, ey, np.random.default_rng(args.seed), args.replicates)
        weights[variant + '_component_counts'] = counts
        arrays[variant + '_sample_component_codes'] = codes
        orders[variant] = order; rejected[variant] = invalid
        ext_align[variant + '_component_id'] = groups
        for cid, part in joined.groupby('component_id', sort=True):
            record = {'variant': variant, 'component_id': cid, 'n_images': len(part), 'n_healthy': int(part.foliar_binary.eq(0).sum()),
                      'n_affected': int(part.foliar_binary.eq(1).sum()), 'archive_label_count': int(part.original_label.nunique()),
                      'original_labels': ' | '.join(sorted(part.original_label.unique())), 'binary_conflict': part.foliar_binary.nunique() > 1}
            component_records.append(record)
        subset = [r for r in component_records if r['variant'] == variant]
        conflicting = [r for r in subset if r['binary_conflict']]
        conflict_records.append({'variant': variant, 'components': len(order), 'images': len(ext), 'binary_conflict_components': len(conflicting),
                                 'images_in_binary_conflict_components': sum(r['n_images'] for r in conflicting),
                                 'multiclass_conflict_components': sum(r['archive_label_count'] > 1 for r in subset),
                                 'labels_modified': 0, 'images_excluded': 0})
    # Independent stream for TOM days; shared across both graph sensitivities and all models.
    day_rng = np.random.default_rng(np.random.SeedSequence([args.seed, 1]))
    counts, codes, order, invalid = resampling_counts(days, ty, day_rng, args.replicates)
    weights['tom_day_counts'] = counts; arrays['tom_sample_day_codes'] = codes
    orders['tom_days'] = order; rejected['tom_days'] = invalid
    ext_align.to_csv(out / 'external_alignment.csv', index=False)
    tom_align.to_csv(out / 'internal_alignment.csv', index=False)
    pd.DataFrame(component_records).to_csv(out / 'component_composition.csv', index=False)
    pd.DataFrame(conflict_records).to_csv(out / 'binary_conflict_component_summary.csv', index=False)
    np.savez_compressed(out / 'aligned_predictions.npz', **arrays)
    np.savez_compressed(out / 'bootstrap_component_weights.npz', **weights)
    dump(out / 'bootstrap_group_orders.json', orders)
    prepared = ['external_alignment.csv', 'internal_alignment.csv', 'component_composition.csv', 'binary_conflict_component_summary.csv',
                'aligned_predictions.npz', 'bootstrap_component_weights.npz', 'bootstrap_group_orders.json']
    freeze = {'status': 'PREPARED_NOT_YET_ANALYZED', 'frozen_utc': datetime.now(timezone.utc).isoformat(), 'script_sha256': sha(Path(__file__)),
              'inputs': {k: {'path': str(p), 'sha256': sha(p)} for k, p in paths.items()},
              'prepared_files_sha256': {n: sha(out / n) for n in prepared}, 'seed': args.seed, 'replicates': args.replicates,
              'models': list(MODELS), 'calibration_arms': list(ARMS), 'metrics': list(METRICS), 'variants': list(VARIANTS),
              'external_rows': 813, 'internal_rows': 1643, 'internal_days': 103,
              'resampling': 'For each graph reset default_rng(seed=20261002), sample G whole components uniformly with replacement G times per replicate using multinomial counts. Every model and both calibration arms share each graph weight draw. TOM uses independent default_rng(SeedSequence([seed,1])) and samples all 103 days with replacement; the same TOM stream is paired with both external graphs.',
              'point_estimand': 'Original image-weighted metrics: one unit per all 813 locked COLD images and all 1643 TOM OOF images. No exclusions, relabeling, model refits or threshold changes.',
              'intervals': '2.5% and 97.5% empirical percentiles with numpy linear interpolation; labels-only degenerate draws replaced before freezing; fixed predictions, no refitting.',
              'rejected_single_class_draws': rejected, 'paired_sign': 'Reference minus comparator for all metrics; an additional improvement-positive column reverses Brier sign.',
              'gap_sign': 'External minus internal, from independent external-component and TOM-day draws.',
              'calibration_change': 'fixed_final_C_OOF minus reconstructed_legacy_mixed_C; BA and Brier only; shared external weights.',
              'scope': SCOPE, 'no_selection_between_graphs': True, 'classification_model_fits': 0,
              'software': {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__, 'sklearn': sklearn.__version__}}
    dump(out / 'PROTOCOL_FREEZE.json', freeze)
    print(json.dumps({'status': freeze['status'], 'output': str(out), 'groups': {k: len(v) for k, v in orders.items()}, 'rejected_draws': rejected}), flush=True)


def weighted_metrics(y, probability, prediction, weights):
    """Compute weighted BA/AUC/Brier, preserving equal-score AUC ties."""
    w = np.asarray(weights, dtype=np.float64)
    pos = y.astype(np.float64); neg = 1 - pos
    npos, nneg = w @ pos, w @ neg
    assert (npos > 0).all() and (nneg > 0).all()
    sensitivity = (w @ (pos * prediction)) / npos
    specificity = (w @ (neg * (1 - prediction))) / nneg
    ba = .5 * (sensitivity + specificity)
    brier = (w @ (probability - y) ** 2) / w.sum(axis=1)
    order = np.argsort(probability, kind='stable')
    sorted_scores = probability[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_scores) != 0) + 1]
    positive = np.add.reduceat(w[:, order] * y[order], starts, axis=1)
    negative = np.add.reduceat(w[:, order] * (1 - y[order]), starts, axis=1)
    before_negative = np.cumsum(negative, axis=1) - negative
    auc = np.sum(positive * (before_negative + .5 * negative), axis=1) / (npos * nneg)
    return np.column_stack([ba, auc, brier])


def quantiles(values):
    assert np.isfinite(values).all()
    return [float(x) for x in np.quantile(values, [.025, .975], method='linear')]


def run(args):
    out = args.output.resolve()
    protocol = json.loads((out / 'PROTOCOL_FREEZE.json').read_text(encoding='utf-8'))
    if (out / 'COMPLETION.json').exists():
        raise RuntimeError('Refuse overwriting complete statistics')
    assert sha(Path(__file__)) == protocol['script_sha256'], 'Script changed after weight freeze'
    for item in protocol['inputs'].values():
        assert sha(item['path']) == item['sha256'], item['path']
    for name, expected in protocol['prepared_files_sha256'].items():
        assert sha(out / name) == expected, name
    a = np.load(out / 'aligned_predictions.npz', allow_pickle=False)
    b = np.load(out / 'bootstrap_component_weights.npz', allow_pickle=False)
    B = protocol['replicates']; points, intervals, paired, gaps, calibration_rows, calibration_changes, checks = [], [], [], [], [], [], []
    distributions = {}
    cohorts = {'internal': (a['internal_y'], a['internal_probability'], a['internal_prediction']),
               'external': (a['external_y'], a['external_probability'], a['external_prediction'])}
    point = {}
    for cohort, (y, probs, preds) in cohorts.items():
        point[cohort] = np.asarray([weighted_metrics(y, p, d, np.ones((1, len(y))))[0] for p, d in zip(probs, preds)])
        for mi, model in enumerate(MODELS):
            for ki, metric in enumerate(METRICS):
                points.append({'cohort': cohort, 'model': model, 'metric': metric, 'estimate': float(point[cohort][mi, ki]), 'n_images': len(y)})
    def evaluate(name, y, probabilities, predictions, weights):
        values = np.asarray([weighted_metrics(y, p, d, weights) for p, d in zip(probabilities, predictions)])
        check_indices = sorted({0, B // 2, B - 1})
        for mi, model in enumerate(MODELS):
            for ri in check_indices:
                w = weights[ri].astype(float)
                actual = [balanced_accuracy_score(y, predictions[mi], sample_weight=w), roc_auc_score(y, probabilities[mi], sample_weight=w), brier_score_loss(y, probabilities[mi], sample_weight=w)]
                for ki, metric in enumerate(METRICS):
                    error = abs(float(values[mi, ri, ki]) - float(actual[ki]))
                    checks.append({'stream': name, 'model': model, 'replicate': ri, 'metric': metric, 'vectorized': float(values[mi, ri, ki]), 'sklearn': float(actual[ki]), 'absolute_error': error})
                    assert error <= 1e-12, (name, model, ri, metric, error)
        return values
    with threadpool_limits(limits=1):
        tw = b['tom_day_counts'][:, a['tom_sample_day_codes']]
        internal = evaluate('tom_days', *cohorts['internal'], tw)
        distributions['tom_days_metrics'] = internal
        for mi, model in enumerate(MODELS):
            for ki, metric in enumerate(METRICS):
                lo, hi = quantiles(internal[mi, :, ki]); intervals.append({'variant': 'tom_days', 'cohort': 'internal', 'model': model, 'metric': metric, 'estimate': float(point['internal'][mi, ki]), 'lower': lo, 'upper': hi, 'replicates': B})
        ref = MODELS.index(REFERENCE)
        for variant in VARIANTS:
            ew = b[variant + '_component_counts'][:, a[variant + '_sample_component_codes']]
            external = evaluate(variant, *cohorts['external'], ew)
            distributions[variant + '_external_metrics'] = external
            gap = external - internal; distributions[variant + '_transport_gaps'] = gap
            for mi, model in enumerate(MODELS):
                for ki, metric in enumerate(METRICS):
                    lo, hi = quantiles(external[mi, :, ki]); intervals.append({'variant': variant, 'cohort': 'external', 'model': model, 'metric': metric, 'estimate': float(point['external'][mi, ki]), 'lower': lo, 'upper': hi, 'replicates': B})
                    gl, gh = quantiles(gap[mi, :, ki]); gaps.append({'variant': variant, 'model': model, 'metric': metric, 'external_minus_internal': float(point['external'][mi, ki] - point['internal'][mi, ki]), 'lower': gl, 'upper': gh, 'replicates': B})
                    if mi != ref:
                        delta = external[ref, :, ki] - external[mi, :, ki]; dl, dh = quantiles(delta); sign = -1 if metric == 'brier' else 1
                        improvement = sign * delta; il, ih = quantiles(improvement)
                        paired.append({'variant': variant, 'reference': REFERENCE, 'comparator': model, 'metric': metric, 'reference_minus_comparator': float(point['external'][ref, ki] - point['external'][mi, ki]), 'lower': dl, 'upper': dh, 'improvement_positive_favors_reference': float(sign * (point['external'][ref, ki] - point['external'][mi, ki])), 'improvement_lower': il, 'improvement_upper': ih, 'replicates': B})
            cal = np.asarray([weighted_metrics(a['external_y'], p, d, ew)[:, [0, 2]] for p, d in zip(a['calibration_probability'], a['calibration_prediction'])])
            cal_point = np.asarray([weighted_metrics(a['external_y'], p, d, np.ones((1, 813)))[0, [0, 2]] for p, d in zip(a['calibration_probability'], a['calibration_prediction'])])
            distributions[variant + '_calibration_arm_metrics'] = cal
            distributions[variant + '_calibration_change'] = cal[1] - cal[0]
            for ai, arm in enumerate(ARMS):
                for ki, metric in enumerate(('balanced_accuracy', 'brier')):
                    lo, hi = quantiles(cal[ai, :, ki]); calibration_rows.append({'variant': variant, 'arm': arm, 'metric': metric, 'estimate': float(cal_point[ai, ki]), 'lower': lo, 'upper': hi, 'replicates': B})
            for ki, metric in enumerate(('balanced_accuracy', 'brier')):
                lo, hi = quantiles(cal[1, :, ki] - cal[0, :, ki]); calibration_changes.append({'variant': variant, 'metric': metric, 'contrast': ARMS[1] + '_minus_' + ARMS[0], 'estimate': float(cal_point[1, ki] - cal_point[0, ki]), 'lower': lo, 'upper': hi, 'replicates': B})
    for filename, rows in [('point_estimates.csv', points), ('metric_intervals.csv', intervals), ('paired_model_contrasts.csv', paired),
                           ('external_minus_internal_gaps.csv', gaps), ('calibration_arm_intervals.csv', calibration_rows),
                           ('calibration_changes.csv', calibration_changes), ('sklearn_numerical_crosscheck.csv', checks)]:
        pd.DataFrame(rows).to_csv(out / filename, index=False)
    np.savez_compressed(out / 'bootstrap_metric_distributions.npz', **distributions)
    summary = {'status': 'PASS_FIXED_PREDICTION_DEPENDENCY_SENSITIVITY', 'completed_utc': datetime.now(timezone.utc).isoformat(),
               'replicates': B, 'variants_reported': list(VARIANTS), 'models_reported': len(MODELS), 'reference': REFERENCE,
               'point_estimates_unchanged_image_weighted': True, 'labels_modified': 0, 'images_excluded': 0, 'model_fits': 0,
               'sklearn_crosschecks': len(checks), 'maximum_numerical_error': max(r['absolute_error'] for r in checks), 'scope': SCOPE,
               'binary_conflict_components': pd.read_csv(out / 'binary_conflict_component_summary.csv').to_dict('records'),
               'distribution_array_axes': {'metrics': ['model', 'replicate', 'metric'], 'calibration_arm_metrics': ['arm', 'replicate', 'BA_then_Brier'], 'calibration_change': ['replicate', 'BA_then_Brier']}}
    dump(out / 'SUMMARY.json', summary)
    dump(out / 'COMPLETION.json', {'status': summary['status'], 'files_sha256': {p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file()}})
    print(json.dumps(summary, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['prepare', 'run'], required=True)
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--raw-identity', type=Path)
    p.add_argument('--all-high-components', type=Path)
    p.add_argument('--spatial-high-components', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', type=int, default=20261002)
    p.add_argument('--replicates', type=int, default=3000)
    args = p.parse_args()
    if args.replicates < 1:
        p.error('--replicates must be positive')
    if args.mode == 'prepare':
        if not all([args.raw_identity, args.all_high_components, args.spatial_high_components]):
            p.error('prepare requires --raw-identity and both component CSV paths')
        prepare(args)
    else:
        run(args)


if __name__ == '__main__':
    main()
