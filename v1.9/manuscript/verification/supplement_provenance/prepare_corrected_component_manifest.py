"""Freeze a conservative, geometry-supported subset before outcome evaluation.

Blocking components and common-reference views have distinct roles. A component
is never assumed to be a single photograph or a biological replicate.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph-dir', type=Path, required=True)
    parser.add_argument('--original-manifest', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    inputs = {
        'graph': args.graph_dir / 'all_high_augmented_components.csv',
        'support': args.graph_dir / 'member_source_support.csv',
        'original_manifest': args.original_manifest,
        'graph_freeze': args.graph_dir / 'GRAPH_FREEZE.json',
    }
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError('Refuse to overwrite a frozen selection')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    freeze = {
        'frozen_utc': datetime.now(timezone.utc).isoformat(),
        'script_sha256': sha(__file__),
        'inputs': {name: {'path': str(p.resolve()), 'sha256': sha(p)} for name, p in inputs.items()},
        'rules': [
            'Use all-high evidence components for blocking; exclude every cross-original-label component.',
            'Within each component choose the raw-content reference supported spatially by the most distinct augmented members; ties use lexical raw-content hash order.',
            'Require at least three distinct augmented SHA256 values for that common reference.',
            'Admit only members supporting that chosen reference, with one intervention_source_id per blocking component.',
            'All other members remain blocked/excluded; no classifier output is read or used.',
        ],
        'limits': [
            'Candidate-retrieved spatial correspondence is evidence of common visual content, not a complete original-to-augmentation generation ledger.',
            'Components may conservatively merge distinct photographs; absence of a retrieved edge does not prove independence.',
            'Retained subset differs from the whole archive; plant identity and pathology remain unverified.',
            'Selection and design are post hoc relative to historical model results; frozen before these corrected fits.',
        ],
        'model_fits': 0,
    }
    (args.output_dir / 'SELECTION_FREEZE.json').write_text(json.dumps(freeze, indent=2), encoding='utf-8')
    graph = pd.read_csv(inputs['graph'])
    support = pd.read_csv(inputs['support'])
    old = pd.read_csv(inputs['original_manifest']).rename(columns={'row_idx': 'aug_row'})
    assert not graph.aug_row.duplicated().any() and not support.aug_row.duplicated().any()
    assert set(graph.aug_row) == set(support.aug_row) == set(old.aug_row)
    assert graph.shape[0] == 4502
    data = graph.drop(columns=['spatial_high_raw_content_ids']).merge(
        support[['aug_row', 'spatial_high_raw_content_ids']], on='aug_row', validate='one_to_one'
    ).merge(old[['aug_row', 'sha256']], on='aug_row', validate='one_to_one')
    assert not data.sha256.duplicated().any()
    selected, sources, ledger = [], [], []
    for component, rows in data.groupby('component_id', sort=True):
        label_count = int(rows.component_archive_label_count.iloc[0])
        assert rows.component_archive_label_count.nunique() == 1
        support_sets = {int(r.aug_row): set(json.loads(r.spatial_high_raw_content_ids)) for r in rows.itertuples()}
        counts = Counter(source for values in support_sets.values() for source in values)
        chosen = sorted(counts, key=lambda source: (-counts[source], source))[0] if counts else ''
        enough = bool(chosen) and counts[chosen] >= 3
        eligible = label_count == 1 and enough
        reason = 'accepted_common_reference_subset' if eligible else ('cross_label_blocking_component' if label_count != 1 else 'fewer_than_three_spatially_supported_views')
        sources.append({'component_id': component, 'archive_labels': '|'.join(sorted(rows.label.unique())), 'component_members': len(rows), 'component_label_count': label_count, 'intervention_source_id': chosen if eligible else '', 'chosen_reference_supported_views': counts.get(chosen, 0), 'eligible': eligible, 'decision': reason})
        for row in rows.itertuples():
            take = eligible and chosen in support_sets[int(row.aug_row)]
            ledger.append({'row_idx': int(row.aug_row), 'component_id': component, 'selected': take, 'reason': 'selected_common_reference_view' if take else ('different_or_unresolved_reference_within_blocked_component' if eligible else reason)})
            if take:
                selected.append({'row_idx': int(row.aug_row), 'label': row.label, 'local_path': row.local_path, 'sha256': row.sha256, 'component_id': component, 'intervention_source_id': chosen, 'historical_filename_family': row.family_id})
    manifest = pd.DataFrame(selected).sort_values(['component_id', 'row_idx'])
    assert manifest.groupby('component_id').intervention_source_id.nunique().max() == 1
    assert manifest.groupby('component_id').label.nunique().max() == 1
    assert manifest.groupby('intervention_source_id').component_id.nunique().max() == 1
    assert manifest.groupby('component_id').size().min() >= 3
    manifest.to_csv(args.output_dir / 'corrected_manifest.csv', index=False)
    pd.DataFrame(sources).to_csv(args.output_dir / 'component_eligibility.csv', index=False)
    pd.DataFrame(ledger).to_csv(args.output_dir / 'all_member_selection_ledger.csv', index=False)
    result = {'status': 'PREPARED_REQUIRES_INDEPENDENT_REVIEW', 'selected_images': len(manifest), 'selected_components': int(manifest.component_id.nunique()), 'components_by_label': manifest.drop_duplicates('component_id').label.value_counts().to_dict(), 'images_by_label': manifest.label.value_counts().to_dict(), 'all_component_decisions': dict(Counter(r['decision'] for r in sources)), 'model_fits': 0, 'files_sha256': {p.name: sha(p) for p in sorted(args.output_dir.iterdir()) if p.is_file()}}
    (args.output_dir / 'SELECTION_SUMMARY.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
