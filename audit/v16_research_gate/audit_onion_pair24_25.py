"""Reconstruct a cross-prefix source-overlap counterexample without refitting.

This reads historical manifests/memberships and existing image files only.
The contact sheet preserves source photographs; it is an audit illustration,
not synthetic research imagery or a new annotation of disease identity.
"""
from pathlib import Path
import hashlib
import json
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
AUDIT = ROOT / 'audit' / 'cold_augmented_lineage'
IDS = ['iris_yellow_virus:parent_0024', 'iris_yellow_virus:parent_0025']

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

mapping_path = AUDIT / 'family_to_raw_mapping.csv'
aug_path = AUDIT / 'augmented_family_manifest.csv'
raw_path = AUDIT / 'raw_parquet_manifest.csv'
membership_path = ROOT / 'results' / 'matched_sibling_intervention_v1' / 'family_membership.csv.gz'
m = pd.read_csv(mapping_path)
a = pd.read_csv(aug_path)
r = pd.read_csv(raw_path)
d = pd.read_csv(membership_path)
mapping = m[m.family_id.isin(IDS)].copy()
assert len(mapping) == 2 and mapping.best_raw_cluster.nunique() == 1
assert mapping.raw_link_confidence.eq('high_geometric_support').all()
assert mapping.best_raw_exact_cluster_size.eq(2).all()
raw_rows = r[r.row_idx.isin([562, 789])].sort_values('row_idx')
assert len(raw_rows) == 2 and raw_rows.sha256.nunique() == 1
aug_rows = a[a.family_id.isin(IDS)].sort_values(['family_id', 'row_idx'])
assert len(aug_rows) == 8
file_rows = []
for kind, frame in [('raw', raw_rows), ('augmented', aug_rows)]:
    for row in frame.to_dict('records'):
        path = ROOT / row['local_path']
        actual = sha(path)
        assert actual == row['sha256'], path
        file_rows.append({'kind': kind, 'row_idx': row['row_idx'],
                          'family_id': row.get('family_id', ''),
                          'original_filename': row['original_filename'],
                          'local_path': row['local_path'], 'sha256': actual})
pd.DataFrame(file_rows).to_csv(OUT / 'ONION_PAIR24_25_FILE_HASHES.csv', index=False, encoding='utf-8')
mapping.to_csv(OUT / 'ONION_PAIR24_25_EXISTING_GEOMETRY.csv', index=False, encoding='utf-8')

x = d[d.family_id.isin(IDS)].copy()
rows = []
for seed, g in x.groupby('seed', sort=True):
    assert len(g) == 2
    for test in g.to_dict('records'):
        if test['role'] not in ['probe', 'sentinel']:
            continue
        other = g[g.family_id.ne(test['family_id'])].iloc[0]
        for dose in ['0', '0.5', '1']:
            if int(other['train_' + dose]) == 1:
                rows.append({'seed': int(seed), 'dose': dose,
                             'test_family': test['family_id'], 'role': test['role'],
                             'test_anchor_row_idx': int(test['test_anchor_row_idx']),
                             'train_related_family': other.family_id,
                             'train_related_view1': int(other.view1_row_idx),
                             'train_related_view2': int(other.view2_row_idx),
                             'link_basis': 'same_raw_sha_cluster_in_existing_high_support_mapping'})
cross = pd.DataFrame(rows)
cross.to_csv(OUT / 'ONION_PAIR24_25_CROSS_PARTITION_RECORDS.csv', index=False, encoding='utf-8')
counts = cross.groupby(['dose','role']).size().reset_index(name='test_anchor_events')
assert int(counts.loc[counts.dose.eq('0') & counts.role.eq('probe'), 'test_anchor_events'].iloc[0]) == 8
assert int(counts.loc[counts.dose.eq('0') & counts.role.eq('sentinel'), 'test_anchor_events'].iloc[0]) == 1

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9})
fig, axes = plt.subplots(3, 4, figsize=(10.0, 8.4))
for ax in axes.flat:
    ax.axis('off')
for col, row in enumerate(raw_rows.to_dict('records')):
    axes[0,col].imshow(Image.open(ROOT / row['local_path']))
    axes[0,col].set_title(f"Raw {row['original_filename']} (row {row['row_idx']})\nSHA-256 identical", fontsize=9)
axes[0,2].text(0, .83, 'Existing audit links BOTH prefix groups\nto this same exact-content raw cluster.\n\nThe old collision rule allows 2 links\nbecause there are 2 duplicate raw files.\n\nFor source disjointness, 2 duplicate files\nrepresent 1 content unit, not 2.', va='top', linespacing=1.45)
for ri, fid in enumerate(IDS, start=1):
    for col, row in enumerate(aug_rows[aug_rows.family_id.eq(fid)].to_dict('records')):
        axes[ri,col].imshow(Image.open(ROOT / row['local_path']))
        axes[ri,col].set_title(f"{row['original_filename']} (row {row['row_idx']})\nFilename group {24 if ri == 1 else 25}", fontsize=9)
fig.suptitle('Onion counterexample: separate filename groups share the same source photograph', fontsize=12, y=.987)
fig.text(.025, .015, 'Audit contact sheet from existing files. Geometry is from the historical best-member audit; all eight derivatives were visually inspected.\nNo disease diagnosis or universal completeness of image matching is inferred. See paired file hashes and split records.', fontsize=8)
fig.tight_layout(rect=(0, .06, 1, .96), h_pad=1.3)
fig.savefig(OUT / 'ONION_PAIR24_25_VISUAL_LINK.png', dpi=200)
fig.savefig(OUT / 'ONION_PAIR24_25_VISUAL_LINK.pdf')
plt.close(fig)

summary = {
  'status': 'COUNTEREXAMPLE_TO_SOURCE_DISJOINTNESS_CONFIRMED_FROM_EXISTING_ARTIFACTS',
  'date': '2026-10-02',
  'scope': 'No downloads, no model fitting, no mutation of original evidence.',
  'families': IDS,
  'raw_exact_content_sha256': raw_rows.sha256.iloc[0],
  'raw_filenames': raw_rows.original_filename.tolist(),
  'raw_file_count': 2, 'unique_raw_content_count': 1,
  'old_collision_capacity': 2,
  'old_supported_family_links': 2,
  'old_downgrade_condition_len_gt_capacity': False,
  'why_invalid_for_source_disjointness': 'Exact duplicate raw records are one source-content unit, regardless of file multiplicity. Both groups remain separated by the existing experiment.',
  'cross_partition_event_counts': counts.to_dict('records'),
  'nominal_zero_seeds_affected': int(cross[cross.dose.eq('0')].seed.nunique()),
  'sentinel_zero_seed': 20261028,
  'geometry_scope': 'Existing family-level scores select the best of sampled members. They do not certify every member; the contact sheet provides an explicit visual counterexample.',
  'result_scope': 'Numerical historical fits remain reproducible but zero-source-exposure and always-unexposed-sentinel interpretations are unsupported. Extent beyond this pair is unknown.',
  'inputs': {str(p.relative_to(ROOT)): sha(p) for p in [mapping_path, aug_path, raw_path, membership_path, ROOT/'code'/'audit_cold_augmented_lineage.py']},
}
(OUT/'ONION_PAIR24_25_COUNTEREXAMPLE.json').write_text(json.dumps(summary, indent=2)+'\n', encoding='utf-8')
print(json.dumps({'status': summary['status'], 'nominal_zero_seeds_affected':summary['nominal_zero_seeds_affected'], 'event_counts':summary['cross_partition_event_counts']}))
