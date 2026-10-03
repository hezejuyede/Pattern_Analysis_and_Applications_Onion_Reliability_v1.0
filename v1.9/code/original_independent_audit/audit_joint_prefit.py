"""Independent identity/assignment audit; never fits any classifier."""
from pathlib import Path
import argparse, ast, hashlib, importlib.util, json, sys
import numpy as np
import pandas as pd

D18 = Path('D:/论文/SCI投稿汇总/Onion_Deep_Revision_20261003_v18')
D17 = Path('D:/论文/SCI投稿汇总/Onion_Deep_Revision_20261003_v17')
CODE = D18/'code'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def eqfunc(a, b, name):
    pick=lambda p: next(x for x in ast.parse(p.read_text(encoding='utf-8-sig')).body
                        if isinstance(x,ast.FunctionDef) and x.name==name)
    assert ast.dump(pick(a), include_attributes=False)==ast.dump(pick(b), include_attributes=False), name

def audit(experiment, dataset):
    script=CODE/'run_dinov2_joint_controls.py'
    eqfunc(script,D17/'code/run_matched_replacement_controls.py','plan_one')
    eqfunc(script,D17/'code/run_matched_replacement_controls.py','analyse_pair')
    shared=CODE/'vendor/run_corrected_component_experiment.py'
    assert sha(shared)=='f7db6dd6b174d03264a0ab0d9311ac4ef4dd8f0aaf9042acc0887b13c535c7f2'
    freeze_path=experiment/'protocol_freeze.json'
    freeze=json.loads(freeze_path.read_text(encoding='utf-8-sig'))
    for item in freeze['inputs'].values(): assert sha(item['path'])==item['sha256'], item['path']
    for name, wanted in freeze['prepared_hashes'].items(): assert sha(experiment/name)==wanted, name
    assert not (experiment/'predictions.csv.gz').exists(),'Not prefit: results already exist'
    inp=freeze['inputs']; frame=pd.read_csv(inp['manifest']['path'],dtype=str,keep_default_na=False)
    assert frame.reset_index(drop=True).equals(pd.read_csv(experiment/'manifest.csv',dtype=str,keep_default_na=False))
    with np.load(inp['dinov2']['path'],allow_pickle=False) as f:
        assert {'features','paths','row_idx','image_sha256'}<=set(f.files)
        assert f['features'].shape==(len(frame),384)
        assert np.isfinite(f['features']).all()
        assert (np.linalg.norm(f['features'],axis=1)>0).all()
        assert list(f['paths'].astype(str))==frame.local_path.tolist()
        assert list(f['row_idx'].astype(str))==frame.row_idx.tolist()
        assert list(f['image_sha256'].astype(str))==frame.sha256.tolist()
    ref=(D17/'matched_replacement_onion' if dataset=='onion'
         else D17/'independent_task/potato/matched_replacement_experiment')
    for name in ('manifest.csv','component_membership.csv','support_checks.csv','allocation_membership.csv.gz'):
        a=pd.read_csv(experiment/name,dtype=str,keep_default_na=False)
        b=pd.read_csv(ref/name,dtype=str,keep_default_na=False)
        pd.testing.assert_frame_equal(a,b)
    sys.path.insert(0,str(CODE/'vendor'))
    spec=importlib.util.spec_from_file_location('review_joint',script)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    labels=['healthy']+sorted(set(frame.label)-{'healthy'})
    counts=[]
    old_records=pd.read_csv(ref/'component_membership.csv',dtype=str,keep_default_na=False)
    for rep in range(30):
        p=mod.plan_one(frame,labels,20261003+rep,rep)
        generated=pd.DataFrame(p['records'].values()).fillna('').astype(str)
        expected=old_records[old_records.repetition.eq(str(rep))].reset_index(drop=True)
        pd.testing.assert_frame_equal(generated,expected)
        z=p['conditions']['zero']['train_components'];test=set(p['testcid'])
        assert len(z)==(240 if dataset=='onion' else 268)
        assert not z&test
        for k in range(5):
            for arm in ('A','B'):
                e=p['conditions'][f'exposure_{k}_{arm}'];h=p['conditions'][f'sham_{k}_{arm}']
                selected=e['selected'];donors={p['records'][x]['remove'] for x in selected}
                alternatives={p['records'][x]['alternate'] for x in selected}
                assert e['train_components']==z-donors|selected
                assert h['train_components']==z-donors|alternatives
                assert e['train_components']&h['train_components']==z-donors
                assert e['train_components']&test==selected
                assert not h['train_components']&test
                assert len(e['train'])==len(h['train'])==2*len(z)
                assert set(frame.iloc[e['train']].sha256).isdisjoint(frame.iloc[p['test']].sha256)
                assert set(frame.iloc[h['train']].sha256).isdisjoint(frame.iloc[p['test']].sha256)
        counts.append({'repetition':rep,'seed':20261003+rep,'training_sources':len(z),'conditions':len(p['conditions'])})
    result={'status':'PASS_PREFIT_JOINT_EXTENSION','dataset':dataset,
        'script_sha256':sha(script),'protocol_sha256':sha(freeze_path),
        'scope':'Independent code/identity/assignment audit; no fitting or result inspection.',
        'v17_functions_ast_identical':['plan_one','analyse_pair'],
        'shared_numerics_sha256':sha(shared),'feature_shape':[len(frame),384],
        'feature_row_path_image_sha_alignment':'PASS','v17_assignment_exact_parity':'PASS',
        'reconstructed_splits':counts,'conditions_checked':630,
        'model_routing_review':'DINO logistic uses reviewed Pipeline; DINO cosine explicitly calls reviewed 1NN branch.',
        'scientific_limits':['Post hoc representation sensitivity, not new data or hypothesis preregistration.',
            'No assurance that full DINO pretraining corpus excludes archive images.',
            'Split variability is not a biological confidence interval.']}
    output=D18/'independent_review'/f'{dataset.upper()}_JOINT_PREFIT_REVIEW.json'
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(output)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--experiment',type=Path,required=True)
    p.add_argument('--dataset',choices=['onion','potato'],required=True);a=p.parse_args()
    audit(a.experiment,a.dataset)
