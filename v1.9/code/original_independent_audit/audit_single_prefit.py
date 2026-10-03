"""Independently reconstruct saved single-source memberships without fitting."""
from pathlib import Path
import argparse, csv, gzip, hashlib, importlib.util, json, sys
import numpy as np
import pandas as pd

D18=Path('D:/论文/SCI投稿汇总/Onion_Deep_Revision_20261003_v18')
D17=D18.parent/'Onion_Deep_Revision_20261003_v17'
CODE=D18/'code'
MODELS=['resnet_logit','colour_logit','dinov2_logit','resnet_cosine_1nn','dinov2_cosine_1nn']

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def audit(dataset):
    out=D18/'single_source'/dataset
    fp=out/'protocol_freeze.json'; f=json.loads(fp.read_text(encoding='utf-8'))
    assert f['status']=='PREPARED_BEFORE_NEW_FITS','HOLD: DINO preparation incomplete'
    assert f['models']==MODELS and f['endpoint']=='multiclass'
    assert f['seeds']==list(range(20261003,20261013))
    assert not (out/'predictions.csv.gz').exists(),'Not prefit: predictions already exist'
    for v in f['inputs'].values(): assert sha(v['path'])==v['sha256'],v['path']
    for name,h in f['prepared_hashes'].items(): assert sha(out/name)==h,name
    assert f['inputs']['shared_numerics']['sha256']=='f7db6dd6b174d03264a0ab0d9311ac4ef4dd8f0aaf9042acc0887b13c535c7f2'
    assert f['inputs']['original_assignment_code']['sha256']=='9200c228e766c190637c695ab6cad8dc0155a098e80364dc08ae1b66c783086d'
    frame=pd.read_csv(f['inputs']['manifest']['path'],dtype=str,keep_default_na=False)
    pd.testing.assert_frame_equal(frame,pd.read_csv(out/'manifest.csv',dtype=str,keep_default_na=False))
    ref=(D17/'matched_replacement_onion' if dataset=='onion' else D17/'independent_task/potato/matched_replacement_experiment')
    pd.testing.assert_frame_equal(frame,pd.read_csv(ref/'manifest.csv',dtype=str,keep_default_na=False))
    rec=pd.read_csv(f['inputs']['membership']['path'],dtype=str,keep_default_na=False)
    rec=rec[rec.repetition.astype(int)<10].reset_index(drop=True)
    pd.testing.assert_frame_equal(rec,pd.read_csv(out/'component_membership_first10.csv',dtype=str,keep_default_na=False))
    base_count=240 if dataset=='onion' else 268
    n_targets=96 if dataset=='onion' else 108
    records={}
    expected={}
    expected_tests={}
    target_keys=set()
    target_rows=[]
    for rep in range(10):
        r={x['component_id']:x for x in rec[rec.repetition.eq(str(rep))].to_dict('records')}
        records[rep]=r
        assert all(int(x['seed'])==20261003+rep for x in r.values())
        zero={k for k,v in r.items() if v['role'] in ('core','remove')}
        assert len(zero)==base_count
        expected[rep,'','zero']=zero
        tests={k for k,v in r.items() if int(v['anchor_position'])>=0}
        expected_tests[rep]=tests
        assert not zero&tests
        targets={k for k,v in r.items() if v['paired']=='1'}
        assert len(targets)==n_targets
        for target in sorted(targets):
            v=r[target];d=v['remove'];a=v['alternate']
            assert d in zero and a not in zero and target not in zero
            assert r[d]['label']==r[a]['label']==v['label']
            expected[rep,target,'exposure']=(zero-{d})|{target}
            expected[rep,target,'sham']=(zero-{d})|{a}
            target_keys.add((rep,target))
            target_rows.append((rep,target,d,a,int(v['anchor_position']),int(v['view1_position']),int(v['view2_position'])))
    actual={}
    with gzip.open(out/'training_membership.csv.gz','rt',encoding='utf-8',newline='') as stream:
        for row in csv.DictReader(stream):
            key=(int(row['repetition']),row['target_component'],row['policy'])
            assert key in expected,key
            cid=row['component_id'];r=records[key[0]][cid]
            assert row['seed']==r['seed']
            assert row['source_id']==r['source_id'] and row['label']==r['label']
            assert row['view1_position']==r['view1_position'] and row['view2_position']==r['view2_position']
            s=actual.setdefault(key,set());assert cid not in s,'Duplicate source membership';s.add(cid)
    assert actual==expected,'Executed training membership differs from independent single-source reconstruction'
    count_rows=0
    for key,sources in actual.items():
        rep,target,policy=key;r=records[rep]
        positions=sorted(int(r[c][v]) for c in sources for v in ('view1_position','view2_position'))
        anchors=sorted(int(r[c]['anchor_position']) for c in expected_tests[rep])
        assert len(positions)==len(set(positions))==2*base_count
        assert not set(positions)&set(anchors)
        train=frame.iloc[positions];test=frame.iloc[anchors]
        assert set(train.sha256).isdisjoint(test.sha256)
        overlap=sources&expected_tests[rep]
        assert overlap==({target} if policy=='exposure' else set())
        source_overlap=set(train.intervention_source_id)&set(test.intervention_source_id)
        assert source_overlap==({r[target]['source_id']} if policy=='exposure' else set())
        zpos=sorted(int(r[c][v]) for c in actual[rep,'','zero'] for v in ('view1_position','view2_position'))
        assert train.label.value_counts().to_dict()==frame.iloc[zpos].label.value_counts().to_dict()
        count_rows+=1
    targets=pd.read_csv(out/'targets.csv',dtype=str,keep_default_na=False)
    assert set(zip(targets.repetition.astype(int),targets.target_component))==target_keys
    actual_targets=[(int(x['repetition']),x['target_component'],x['donor_component'],x['alternate_component'],
                     int(x['anchor_position']),int(x['own_view1_position']),int(x['own_view2_position'])) for x in targets.to_dict('records')]
    assert actual_targets==target_rows
    support=pd.read_csv(out/'support_checks.csv')
    assert len(support)==n_targets*10*2 and support.status.eq('PASS').all()
    assert support.training_sources.eq(base_count).all() and support.training_images.eq(2*base_count).all()
    assert support.groupby(['repetition','target_component']).common_background_sha256.nunique().eq(1).all()
    counts=frame.groupby('component_id').label.first().value_counts()
    common=sorted(counts[counts>=30].index.tolist())
    assert common==f['common_classes']
    with np.load(f['inputs']['dino_features']['path'],allow_pickle=False) as z:
        assert z['features'].shape==(len(frame),384) and np.isfinite(z['features']).all()
        assert (np.linalg.norm(z['features'],axis=1)>0).all()
        assert list(z['paths'].astype(str))==frame.local_path.tolist()
        assert list(z['row_idx'].astype(str))==frame.row_idx.tolist()
        assert list(z['image_sha256'].astype(str))==frame.sha256.tolist()
    # A labelled software unit check, not synthetic study data: unequal class
    # sizes must average to .5, not the target-count-weighted .4.
    script=CODE/'run_single_source_controls.py'
    spec=importlib.util.spec_from_file_location('review_single',script)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    unit_dir=D18/'independent_review'/f'weighting_software_check_{dataset}'
    unit_dir.mkdir(exist_ok=True)
    toy=[]
    for rep in range(10):
        for label,values in [('a',[0.1,0.3]),('b',[0.8])]:
            for i,value in enumerate(values):
                toy.append(dict(repetition=rep,seed=20261003+rep,model='SOFTWARE_TEST_ONLY',endpoint='multiclass',
                    panel='own_target',weighting='all_original_classes',target_label=label,
                    target_component=f'{label}_{i}',n_evaluation=1,n_evaluation_classes=1,
                    policy_BA_difference=value))
    mod.summarize_targets(toy,unit_dir)
    sums=pd.read_csv(unit_dir/'split_summary.csv')
    assert np.allclose(sums['mean'],0.5) and np.allclose(sums.split_min,0.5) and np.allclose(sums.split_max,0.5)
    report={'status':'PASS','reviewer':'independent_v18_literature_priority_agent',
        'protocol_sha256':sha(fp),'code_sha256':sha(script),'dataset':dataset,
        'scope':'Independent prefit code, membership, source identity and weighting audit; no classifier fit.',
        'targets_checked':n_targets*10,'model_independent_training_conditions_checked':count_rows,
        'v17_saved_record_and_anchor_parity':'PASS','one_source_E_H_difference':'PASS',
        'source_and_image_overlap_checks':'PASS','constant_original_class_training_counts':'PASS',
        'dino_feature_384d_identity_alignment':'PASS','class_then_split_weighting_software_check':'PASS',
        'model_routing_review':'Five fixed models; each cosine uses its own features and the explicit reviewed 1NN route.',
        'postfit_checks_still_required':['Full prediction reconstruction','Fit logs and convergence','Independent refits','Manuscript consistency'],
        'scientific_limits':['Post hoc extension after v17','Common-class sensitivity changes weighting and is not a new primary endpoint',
            'Ten splits describe design variation, not independent biological cohorts','Single-source policy contrast is not a pure neural-memory causal mechanism']}
    report_path=D18/'independent_review'/f'{dataset.upper()}_SINGLE_SOURCE_PREFIT_REVIEW.json'
    report_path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(report_path)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['onion','potato'],required=True)
    audit(p.parse_args().dataset)
