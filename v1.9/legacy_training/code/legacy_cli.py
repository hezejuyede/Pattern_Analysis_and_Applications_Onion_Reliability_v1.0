"""Explicit path adapter for the preserved regional and calibration training scripts."""
from pathlib import Path
from types import SimpleNamespace
import argparse,hashlib,importlib.util,json,shutil,sys
import numpy as np
import pandas as pd

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load_module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def verify(root):
    records=json.loads((root/'CACHED_INPUT_MANIFEST.json').read_text(encoding='utf-8'))
    for record in records:
        p=(root/record['path']).resolve();assert p.is_relative_to(root) and p.stat().st_size==record['bytes'] and sha(p)==record['sha256'],p
    shared=json.loads((root/'SHARED_PREDICTION_PATHS.json').read_text(encoding='utf-8'))
    for record in shared.values():
        p=(root.parent/record['release_relative_path']).resolve();assert p.is_relative_to(root.parent) and sha(p)==record['sha256'],p
    return shared

def check(root):
    shared=verify(root)
    region=load_module(root/'code/run_regional_robustness.py','original_regional')
    cal=load_module(root/'code/run_calibration_scale_sensitivity.py','original_calibration')
    metadata=root/'metadata/tom_region_recovery/analysis_image_region_join.csv';features=root/'features'
    frame,matrices=region.load_inputs(SimpleNamespace(metadata=metadata,features_dir=features))
    region.verify_freeze(SimpleNamespace(metadata=metadata,features_dir=features,output_dir=root/'results/regional_robustness_v1'))
    manifest=pd.read_csv(root/'manifests/unified_manifest.csv')
    with np.load(features/'resnet18.npz',allow_pickle=False) as cache:
        ids=cache['sample_ids'].astype(str).tolist();assert ids==manifest.sample_id.astype(str).tolist()
        positions={s:i for i,s in enumerate(ids)};assert len(positions)==len(ids)
        arrays={}
        for name,relative,expected in [('source','results/reliability_benchmark_v1/predictions/tom_nested_oof.csv',1643),('external','results/reliability_benchmark_v1/predictions/cold_locked_external.csv',813)]:
            part=pd.read_csv(root.parent/shared[relative]['release_relative_path']);part=part[part.model.eq(cal.MODEL)].reset_index(drop=True)
            x=cache['features'][[positions[s] for s in part.sample_id.astype(str)]].astype(float)
            assert len(part)==expected and np.isfinite(x).all()
            arrays[name]={'shape':list(x.shape),'images':len(part)}
            if name=='source':source=part
            else:external=part
    folds=pd.read_csv(root/'results/reliability_benchmark_v1/split_manifests/tom_acquisition_day_outer_folds.csv')
    assert source.sample_id.astype(str).equals(folds.sample_id.astype(str)) and np.array_equal(source.tom_outer_fold,folds.outer_fold)
    assert not source.analysis_sha256.duplicated().any() and not external.analysis_sha256.duplicated().any()
    assert not set(source.analysis_sha256)&set(external.analysis_sha256)
    for fold in sorted(source.tom_outer_fold.unique()):assert not set(source.loc[source.tom_outer_fold.eq(fold),'acquisition_day_utc'])&set(source.loc[~source.tom_outer_fold.eq(fold),'acquisition_day_utc'])
    old=json.loads((root/'results/calibration_scale_sensitivity_v1/prespecified_protocol.json').read_text(encoding='utf-8'))
    for key,record in old['inputs'].items():
        relative=record['path'].replace('\\','/')
        p=root.parent/shared[relative]['release_relative_path'] if relative in shared else root/relative
        assert sha(p)==record['sha256'],key
    return {'status':'PASS_CACHED_TRAINING_INPUT_CHECK','training_executed':False,'cached_files':len(json.loads((root/'CACHED_INPUT_MANIFEST.json').read_text(encoding='utf-8'))),
        'regional_rows':len(frame),'regional_matrix_shapes':{k:list(v.shape) for k,v in matrices.items()},'calibration_matrices':arrays,
        'fixed_source_folds':len(source.tom_outer_fold.unique()),'source_external_exact_overlap':0,
        'scope':'Both original modules imported; preserved input hashes and original regional freeze verified; every required feature matrix aligned/read and checked finite. This is not full-training reproduction.'}

def run_regional(root,out,mode,jobs):
    module=load_module(root/'code/run_regional_robustness.py','original_regional_training')
    if mode in {'run','all'}:
        if out.exists() and any(out.iterdir()):raise FileExistsError('Use a new/empty regional output directory for training.')
        out.mkdir(parents=True,exist_ok=True)
        frozen=root/'results/regional_robustness_v1';protocol=json.loads((frozen/'protocol_freeze.json').read_text(encoding='utf-8'))
        for name in ['protocol_freeze.json']+list(protocol['frozen_files']):shutil.copy2(frozen/name,out/name)
    for step in (['run','bootstrap'] if mode=='all' else [mode]):
        sys.argv=[str(root/'code/run_regional_robustness.py'),'--metadata',str(root/'metadata/tom_region_recovery/analysis_image_region_join.csv'),
                  '--features-dir',str(root/'features'),'--output-dir',str(out),'--n-jobs',str(jobs),'--mode',step]
        module.main()

def run_calibration(root,out,bootstrap):
    shared=verify(root);module=load_module(root/'code/run_calibration_scale_sensitivity.py','original_calibration_training')
    original_read_csv=pd.read_csv
    def resolve(path):
        if isinstance(path,(str,Path)):
            p=Path(path)
            try:relative=p.resolve().relative_to(root).as_posix()
            except ValueError:return path
            if relative in shared:return root.parent/shared[relative]['release_relative_path']
        return path
    module.sha=lambda path:sha(resolve(path))
    module.pd.read_csv=lambda path,*args,**kwargs:original_read_csv(resolve(path),*args,**kwargs)
    sys.argv=[str(root/'code/run_calibration_scale_sensitivity.py'),'--output-dir',str(out),'--bootstrap',str(bootstrap)]
    try:module.main()
    finally:pd.read_csv=original_read_csv
    (out/'SHARED_INPUT_ADAPTER.json').write_text(json.dumps({'scope':'Only the two prediction-file paths in sha/read_csv were mapped to their exact shared release objects. Original numerical functions, code bytes and model settings are unchanged.','shared':shared},indent=2)+'\n',encoding='utf-8')

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);parser.add_argument('--analysis',choices=['check','regional','calibration'],default='check')
    parser.add_argument('--out',type=Path);parser.add_argument('--report',type=Path);parser.add_argument('--mode',choices=['run','bootstrap','all'],default='all');parser.add_argument('--n-jobs',type=int,choices=[1,2,3,4],default=1);parser.add_argument('--bootstrap',type=int,default=1000)
    args=parser.parse_args();root=args.root.resolve()
    verify(root)
    if args.analysis=='check':
        report=check(root)
        if args.report:args.report.parent.mkdir(parents=True,exist_ok=True);args.report.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
        print(json.dumps(report,indent=2));return
    if args.out is None:parser.error('training requires --out pointing to a separate output directory')
    out=args.out.resolve()
    if out.is_relative_to(root) or root.is_relative_to(out):parser.error('training output must be separate from the preserved legacy_training tree')
    if args.analysis=='regional':run_regional(root,out,args.mode,args.n_jobs)
    else:run_calibration(root,out,args.bootstrap)

if __name__=='__main__':main()
