"""Replay the unchanged D16 analysis using explicit relocated inputs and frozen weights."""
from pathlib import Path
from types import SimpleNamespace
import argparse,hashlib,importlib.util,json,shutil
import numpy as np
import pandas as pd

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def compare_csv(actual,expected):
    a=pd.read_csv(actual);e=pd.read_csv(expected)
    assert list(a.columns)==list(e.columns) and a.shape==e.shape,actual
    maximum=0.
    for column in a:
        if pd.api.types.is_numeric_dtype(a[column]) and pd.api.types.is_numeric_dtype(e[column]):
            av=a[column].to_numpy(float);ev=e[column].to_numpy(float)
            assert np.array_equal(np.isnan(av),np.isnan(ev)),column
            error=float(np.max(np.abs(av-ev),initial=0));assert error<1e-12,(column,error)
            maximum=max(maximum,error)
        else:assert a[column].fillna('').astype(str).equals(e[column].fillna('').astype(str)),column
    return {'rows':len(a),'columns':len(a.columns),'maximum_absolute_difference':maximum}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1],help='supplementary_context directory')
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();root=args.root.resolve();out=args.out.resolve()
    if out==root or out.is_relative_to(root/'inputs') or out.is_relative_to(root/'prepared') or out.is_relative_to(root/'expected'):
        raise ValueError('Output must not overwrite frozen release inputs/prepared/expected objects.')
    if out.exists() and any(out.iterdir()):raise FileExistsError('Use a new empty output directory: '+str(out))
    out.mkdir(parents=True,exist_ok=True)
    freeze=json.loads((root/'prepared/PROTOCOL_FREEZE.json').read_text(encoding='utf-8'))
    mapping=json.loads((root/'path_mapping.json').read_text(encoding='utf-8'))
    for name,item in freeze['inputs'].items():
        key=item['path'].replace('\\','/');record=mapping[key];p=(root/record['relative_path']).resolve()
        assert p.is_relative_to(root) and sha(p)==item['sha256']==record['sha256'],name
    for name,expected in freeze['prepared_files_sha256'].items():
        p=root/'prepared'/name;assert sha(p)==expected;shutil.copy2(p,out/name)
    shutil.copy2(root/'prepared/PROTOCOL_FREEZE.json',out/'PROTOCOL_FREEZE.json')
    original=root/'code/original/run_external_dependency_sensitivity.py'
    assert sha(original)==freeze['script_sha256']
    spec=importlib.util.spec_from_file_location('frozen_external_dependency_analysis',original)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    original_sha=module.sha;resolved=[]
    def relocated_sha(path):
        key=str(path).replace('\\','/')
        if key in mapping:
            target=(root/mapping[key]['relative_path']).resolve();assert target.is_relative_to(root)
            resolved.append({'historical_path':key,'used_relative_path':mapping[key]['relative_path']})
            return original_sha(target)
        p=Path(path).resolve()
        assert p.is_relative_to(root) or p.is_relative_to(out),'Unmapped external input: '+str(p)
        return original_sha(p)
    # Only filename resolution for SHA checks changes. No metric/weight/prediction code is replaced.
    module.sha=relocated_sha
    module.run(SimpleNamespace(output=out))
    tables={}
    for expected in sorted((root/'expected').glob('*.csv')):tables[expected.name]=compare_csv(out/expected.name,expected)
    arrays={}
    with np.load(out/'bootstrap_metric_distributions.npz',allow_pickle=False) as a,np.load(root/'expected/bootstrap_metric_distributions.npz',allow_pickle=False) as e:
        assert sorted(a.files)==sorted(e.files)
        for key in a.files:
            assert a[key].shape==e[key].shape and a[key].dtype==e[key].dtype
            error=float(np.max(np.abs(a[key]-e[key]),initial=0));assert error<1e-12,(key,error)
            arrays[key]={'shape':list(a[key].shape),'maximum_absolute_difference':error,'bitwise_equal':bool(np.array_equal(a[key],e[key]))}
    actual_summary=json.loads((out/'SUMMARY.json').read_text(encoding='utf-8'));expected_summary=json.loads((root/'expected/SUMMARY.json').read_text(encoding='utf-8'))
    for key,value in expected_summary.items():
        if key in {'completed_utc','maximum_numerical_error'}:continue
        assert actual_summary[key]==value,key
    assert len(resolved)==6
    report={'status':'PASS_RELOCATED_FIXED_PREDICTION_REPLAY','models':7,'replicates_per_stream':3000,'external_graphs':[547,700],'internal_days':103,
        'external_images':813,'internal_images':1643,'model_fits':0,'protocol_sha256':sha(out/'PROTOCOL_FREEZE.json'),'original_code_sha256':sha(original),
        'tables':tables,'arrays':arrays,'relocated_inputs':resolved,
        'adapter_scope':'Only historical input path resolution in SHA checking is replaced; unchanged original run and metric functions use the unchanged seven prepared files and frozen bootstrap weights.',
        'scientific_scope':module.SCOPE,'exclusions':'No image download, classifier retraining, new acquisition, new pathology review or regional-model refitting.'}
    (out/'PORTABLE_EXTERNAL_CONTEXT_VERIFICATION.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
