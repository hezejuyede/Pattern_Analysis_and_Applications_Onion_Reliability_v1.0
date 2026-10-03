"""Final source and PDF consistency checks; no model fitting or file regeneration."""
from pathlib import Path
import csv
import hashlib
import json
import math
import re
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import pdfplumber

D = Path('D:/论文/SCI投稿汇总/Onion_Deep_Revision_20261003_v19')
M = D / 'manuscript_revision'
OLD = D.with_name('Onion_Deep_Revision_20261003_v18') / 'manuscript_revision'
def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()
def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))
def dump(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
def rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

stamp = datetime.now(timezone.utc).isoformat()
build = read_json(M / 'SUPPLEMENT_BUILD_v19.json')
index = rows(M / 'SUPPLEMENT_TABLE_INDEX.csv')
assert len(index) == 53
for r in index:
    p = M / r['file']
    assert sha(p) == r['sha256']
    assert p.stat().st_size == int(r['bytes'])
    assert len(rows(p)) == int(r['rows'])
assert len(build['retained_tables']) == 39
for r in build['retained_tables']:
    assert sha(M / 'supplement_tables' / r['file']) == sha(OLD / 'supplement_tables' / r['file']) == r['sha256']
assert len(build['added_tables']) == 14
for r in build['added_tables']:
    assert sha(M / r['file']) == sha(r['source']) == r['sha256']
freeze = read_json(D / 'additional_analysis/ADDITIONAL_ANALYSIS_FREEZE.json')
assert sha(D / 'code/rebuild_additional_descriptive.py') == freeze['code_sha256']
for r in freeze['inputs']:
    assert sha(r['path']) == r['sha256'], r['path']
audit_path = D / 'portable_additional_verification/ADDITIONAL_PORTABLE_VERIFICATION.json'
audit = read_json(audit_path)
assert audit['status'] == 'PASS_INDEPENDENT_PORTABLE_DESCRIPTIVE_RECONSTRUCTION'
assert len(audit['tables']) == 14 and all(t['status'] == 'PASS' for t in audit['tables'])
assert audit['prediction_rows_scanned'] == 4704230
assert sum(r['rows'] for r in audit['single_raw_vs_saved_target_checks']) == 10200
assert audit['fitted_models'] == audit['p_values'] == audit['population_confidence_intervals'] == 0
assert sha(audit_path) == sha(M / 'supplement_provenance/ADDITIONAL_PORTABLE_VERIFICATION.json')

source_files = ['Online_Resource_1_EN_v1.9.md', 'Online_Resource_1_ZH_v1.9.md']
sources = [(M / f).read_text(encoding='utf-8') for f in source_files]
for name, text in zip(source_files, sources):
    assert sha(M / name) == build['source_hashes'][name]
    for n in range(1,9):
        assert re.search(r'^## S'+str(n)+r'\b', text, re.M), (name,n)
    assert 'tree/v1.9.0/v1.9' in text
    assert 'tree/v1.8.0/v1.8' not in text
    assert all(Path(r['file']).name in text for r in index)
    assert re.search(r'(?:Fig\.|Figure|图)\s*S2', text)
table_lines = [[s.strip() for s in t.splitlines() if s.startswith('|')] for t in sources]
assert table_lines[0] == table_lines[1]
mapping = rows(M / 'V19_RETAINED_RESULT_LOCATIONS.csv')
assert len(mapping) == 28
for r in mapping:
    assert sha(D / 'public_release_staging/v1.9' / r['release_relative_path']) == r['distributed_sha256']

splits = pd.read_csv(D / 'additional_analysis/all_common_loco_split_results.csv')
mc = pd.read_csv(D / 'additional_analysis/conditional_allocation_mcse.csv')
assert len(mc) == 20
errors=[]
for r in mc.itertuples():
    vals = splits[(splits.dataset==r.dataset)&(splits.design==r.design)&(splits.model==r.model)&(splits.weighting=='all_original_classes')].policy_BA_difference.to_numpy()*100
    assert len(vals) == r.split_count == (30 if r.design=='joint' else 10)
    expected = (vals.mean(), vals.std(ddof=1), vals.std(ddof=1)/math.sqrt(len(vals)))
    actual=(r.mean_pp,r.split_sd_pp,r.mcse_pp)
    errors.extend(abs(a-b) for a,b in zip(expected,actual))
assert max(errors)<1e-12
head=pd.read_csv(D / 'additional_analysis/headroom_interpretation_percentage_units.csv')
assert np.allclose(head.corrected_pp-head.harmed_pp,head.net_contrast_pp,atol=1e-12)
assert np.allclose(head.corrected_pp/head.sham_error_headroom_percent*100,head.corrected_over_sham_error_percent,atol=1e-12)
assert np.allclose(head.net_contrast_pp/head.sham_error_headroom_percent*100,head.net_contrast_over_sham_error_percent,atol=1e-12)
table4=pd.read_csv(M / 'Table_4_display.csv')
assert len(table4)==4
for r in table4.to_dict('records'):
    model={'ResNet18 + LR':'resnet_logit','DINOv2 + LR':'dinov2_logit'}[r['model']]
    ref=head[(head.dataset==r['dataset'].lower())&(head.design=='joint')&(head.model==model)].iloc[0]
    for c in table4.columns[2:]:
        assert abs(r[c]-float(f'{ref[c]:.2f}'))<1e-12, (r,c)

compat={
 'status':'PASS_SOURCE_COMPATIBILITY', 'checked_utc':stamp,
 'retained_v18_csvs_byte_identical':39,'added_csvs_source_identical':14,'total_indexed_csvs':53,
 'complete_prior_sections_retained':'S1-S7; current source path map clarified, S8 appended',
 'frozen_original_inputs_sha_verified':len(freeze['inputs']),
 'frozen_analysis_runner_sha256':freeze['code_sha256'],
 'analysis_freeze_utc':freeze['frozen_utc'],
 'en_zh_numeric_table_lines_identical':True,
 'current_release_objects_hash_verified':len(mapping),
 'independent_reconstruction_audit_sha256':sha(audit_path),
 'independent_reconstruction_max_numeric_difference':max(t['maximum_numeric_difference'] for t in audit['tables']),
 'conditional_mcse_recomputed_from_split_values_max_pp_error':max(errors),
 'main_table4_matches_unrounded_joint_source_after_two_decimal_rounding':True,
 'headroom_corrected_and_net_ratio_identities_verified':True,
 'source_hashes':build['source_hashes'],
 'boundaries':'Post-result descriptive analysis, no new fits, no population confidence intervals, no fresh collection or plant-identity claims. Remote release verification is performed separately by the coordinating agent.'
}
dump(M / 'SUPPLEMENT_SOURCE_COMPATIBILITY_QA.json',compat)

pb=read_json(M / 'SUPPLEMENT_PDF_BUILD.json')
pdfs=[]
for j,doc in enumerate(pb['documents']):
    path=M / doc['file']
    assert sha(path)==doc['pdf_sha256']
    assert sha(M / source_files[j])==doc['source_sha256']
    pages=[]
    with pdfplumber.open(path) as p:
        assert len(p.pages)==doc['pages']
        for n,page in enumerate(p.pages,1):
            assert abs(page.width-595.2756)<.05 and abs(page.height-841.8898)<.05
            chars=page.chars
            assert chars
            bad=[c for c in chars if c.get('text','').strip() and (c['x0']<35 or c['x1']>page.width-35 or c['top']<20 or c['bottom']>page.height-18)]
            assert not bad,(doc['file'],n,bad[:3])
            txt=page.extract_text() or ''
            assert '\ufffd' not in txt and '\u25a0' not in txt
            pages.append({'page':n,'characters':len(chars),'minimum_x':min(c['x0'] for c in chars),'maximum_x':max(c['x1'] for c in chars),'minimum_top':min(c['top'] for c in chars),'maximum_bottom':max(c['bottom'] for c in chars)})
        first=p.pages[0].extract_text()
        for s in ['Pattern Analysis and Applications','Xin Li','Bojian Guo','Asel Kartanova','lixin26@kstu.kg','Higher School of Economics and Business']:
            assert s in first,(doc['file'],s)
    pdfs.append({'file':doc['file'],'sha256':sha(path),'pages':pages,'all_characters_within_safe_page_bounds':True,'no_replacement_or_black_square_glyphs':True})
dump(M / 'SUPPLEMENT_PDF_TECHNICAL_QA.json', {'status':'PASS_PDF_TECHNICAL_QA','checked_utc':stamp,'documents':pdfs,'scope':'Automatic geometry/text/file checks supplement, not replace, the separately recorded manual all-page visual inspection.'})
print(json.dumps({'source_status':compat['status'],'pdf_status':'PASS_PDF_TECHNICAL_QA','pdf_pages':[len(p['pages']) for p in pdfs],'mcse_max_error':max(errors)},ensure_ascii=False))
