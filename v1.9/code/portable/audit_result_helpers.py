"""Independent audit numerics; no import of experimental analysis or prediction code."""
from pathlib import Path
import hashlib, json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

import os, csv, gzip
import pyarrow.parquet as pq
ROOT=Path(os.environ.get('ONION_RELEASE_ROOT',str(Path(__file__).resolve().parents[2]))).resolve()
OUTPUT=Path(os.environ.get('ONION_REBUILD_OUT',str(ROOT/'reproduced'))).resolve()
OUTPUT.mkdir(parents=True,exist_ok=True)
REFIT=os.environ.get('ONION_REFIT','smoke')
PATH_MAPPING=json.loads((ROOT/'provenance/path_mapping.json').read_text(encoding='utf-8'))
CONVERSIONS=json.loads((ROOT/'provenance/parquet_conversion_checks.json').read_text(encoding='utf-8'))
TRANSPORT=json.loads((ROOT/'provenance/gzip_transport_checks.json').read_text(encoding='utf-8')) if (ROOT/'provenance/gzip_transport_checks.json').exists() else {}
def resolve(path):
    p=Path(path)
    key=str(path).replace('\\','/')
    if key in PATH_MAPPING:return ROOT/PATH_MAPPING[key]['relative_path']
    if p.is_file():
        assert p.resolve().is_relative_to(ROOT),'Refusing an unmapped external input: '+str(p)
        return p
    if p.name.endswith('.csv.gz'):
        alternate=p.with_name(p.name.replace('.csv.gz','.parquet'))
        if alternate.is_file():return alternate
    if p.suffix=='.csv' and p.with_suffix('.csv.gz').is_file():return p.with_suffix('.csv.gz')
    raise FileNotFoundError('No explicit release mapping for '+str(path))
def read_table(path,**kwargs):
    p=resolve(path)
    if p.suffix!='.parquet':return pd.read_csv(p,**kwargs)
    dtype=kwargs.get('dtype',{});chunksize=kwargs.get('chunksize')
    def cast(table):
        frame=table.to_pandas()
        for col in frame:
            forced=dtype.get(col) if isinstance(dtype,dict) else dtype
            if forced in (str,'str','string'):frame[col]=frame[col].astype(str)
            elif col in {'repetition','seed','truth','predicted','selected','allocation_pair','in_training','row_position','view_position'} or (col.startswith('p') and col[1:].isdigit()):
                frame[col]=pd.to_numeric(frame[col],errors='raise')
        return frame
    if chunksize:return (cast(b) for b in pq.ParquetFile(p).iter_batches(batch_size=chunksize))
    return cast(pq.read_table(p))
def raw_rows(path):
    p=resolve(path)
    if p.suffix=='.parquet':
        for batch in pq.ParquetFile(p).iter_batches(batch_size=20000):yield from batch.to_pylist()
    else:
        with gzip.open(p,'rt',encoding='utf-8',newline='') as stream:yield from csv.DictReader(stream)
def source_sha(path):
    p=resolve(path)
    rel=p.relative_to(ROOT).as_posix()
    if rel in TRANSPORT:return TRANSPORT[rel]['original_sha256']
    if p.suffix=='.parquet':return CONVERSIONS[p.relative_to(ROOT).as_posix()]['original_sha256']
    return sha(p)
def verify_original(path,expected):
    p=resolve(path)
    rel=p.relative_to(ROOT).as_posix()
    if rel in TRANSPORT:
        record=TRANSPORT[rel]
        assert record['original_sha256']==expected and sha(p)==record['distributed_sha256'],str(path)
    elif p.suffix=='.parquet':
        record=CONVERSIONS[p.relative_to(ROOT).as_posix()]
        assert record['original_sha256']==expected and sha(p)==record['distributed_sha256'],str(path)
    else:assert sha(p)==expected,str(path)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def cm(v,y):return float(np.mean([np.mean(np.asarray(v)[y==c]) for c in np.unique(y)]))

def integrity(experiment,freeze):
    for item in freeze['inputs'].values():verify_original(item['path'],item['sha256'])
    for name,value in freeze['prepared_hashes'].items():verify_original(experiment/name,value)
    for row in read_table(experiment/'artifact_manifest.csv').to_dict('records'):verify_original(experiment/row['path'],row['sha256'])
    return 'PASS_FROZEN_HASHES_WITH_EXPLICIT_PARQUET_CONVERSION_LEDGER'

def metrics(y,z,e,h,mask,prob=None):
    y,z,e,h=[np.asarray(x)[mask] for x in (y,z,e,h)]
    out={'policy_BA_difference':cm((e==y).astype(float)-(h==y),y),'policy_label_disagreement':cm(e!=h,y)}
    for name,p in [('exposure',e),('sham',h)]:
        gain=(z!=y)&(p==y);loss=(z==y)&(p!=y);ww=(z!=y)&(p!=y)&(z!=p)
        out.update({name+'_gain':cm(gain,y),name+'_loss':cm(loss,y),name+'_wrong_to_wrong':cm(ww,y),
            name+'_turnover':cm(gain|loss,y),name+'_label_churn':cm(p!=z,y),
            name+'_BA_delta':cm((p==y).astype(float)-(z==y),y)})
    out['turnover_difference']=out['exposure_turnover']-out['sham_turnover']
    out['churn_difference']=out['exposure_label_churn']-out['sham_label_churn']
    if prob is not None:
        pe,ph=[p[mask] for p in prob]
        out.update(zero_BA=cm(z==y,y),sham_BA=cm(h==y,y),exposure_BA=cm(e==y,y),
            corrected_relative_to_sham=cm((h!=y)&(e==y),y),harmed_relative_to_sham=cm((h==y)&(e!=y),y),
            sham_error_headroom=cm(h!=y,y),p_true_difference=cm(pe[np.arange(len(y)),y]-ph[np.arange(len(y)),y],y),
            total_variation=cm(0.5*np.abs(pe-ph).sum(1),y))
    return out

def compare(actual,expected,keys,values):
    a=actual.sort_values(keys).reset_index(drop=True);b=expected.sort_values(keys).reset_index(drop=True)
    assert len(a)==len(b),(len(a),len(b))
    pd.testing.assert_frame_equal(a[keys].astype(str),b[keys].astype(str),check_dtype=False)
    err=np.abs(a[values].to_numpy(float)-b[values].to_numpy(float))
    assert np.isfinite(err).all() and err.max(initial=0)<1e-11,err.max(initial=0)
    return float(err.max(initial=0))

def load_matrix(path,frame):
    with np.load(resolve(path),allow_pickle=False) as z:
        ix={str(p).replace('\\','/'):i for i,p in enumerate(z['paths'])}
        order=[ix[p.replace('\\','/')] for p in frame.local_path]
        return np.asarray(z['features'][order],dtype=np.float64)

def independent_refit(x,y,train,test,seed,expected,classes):
    clf=Pipeline([('s',StandardScaler()),('c',LogisticRegression(C=.01,solver='lbfgs',class_weight='balanced',max_iter=10000,random_state=seed))])
    with threadpool_limits(limits=1):clf.fit(x[train],y[train]);prob=clf.predict_proba(x[test])
    assert list(clf['c'].classes_)==list(range(classes))
    want=expected[[f'p{i}' for i in range(classes)]].to_numpy(float)
    err=float(np.abs(prob-want).max());diff=int((prob.argmax(1)!=expected.predicted.to_numpy(int)).sum())
    assert err<1e-10 and diff==0,(err,diff)
    return {'max_probability_difference':err,'decision_mismatches':diff,'training_rows':len(train),'test_rows':len(test)}

def independent_nn(x,y,train,test,expected,frame):
    x=x/np.linalg.norm(x,axis=1,keepdims=True)
    with threadpool_limits(limits=1):near=train[np.argmax(x[test]@x[train].T,axis=1)]
    actual=frame.iloc[near].row_idx.to_numpy(str)
    recorded=expected.nearest_train_row_idx.astype(str).to_numpy()
    assert np.array_equal(actual,recorded),(actual[:3],recorded[:3])
    assert np.array_equal(y[near],expected.predicted.to_numpy(int))
    return {'nearest_row_mismatches':0,'decision_mismatches':0,'test_rows':len(test)}
