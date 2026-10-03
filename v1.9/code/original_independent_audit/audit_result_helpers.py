"""Independent audit numerics; no import of experimental analysis or prediction code."""
from pathlib import Path
import hashlib, json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

D18=Path('D:/论文/SCI投稿汇总/Onion_Deep_Revision_20261003_v18')
D17=D18.parent/'Onion_Deep_Revision_20261003_v17'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def cm(v,y):return float(np.mean([np.mean(np.asarray(v)[y==c]) for c in np.unique(y)]))

def integrity(experiment,freeze):
    for item in freeze['inputs'].values():assert sha(item['path'])==item['sha256'],item['path']
    for name,value in freeze['prepared_hashes'].items():assert sha(experiment/name)==value,name
    for row in pd.read_csv(experiment/'artifact_manifest.csv').to_dict('records'):
        assert sha(experiment/row['path'])==row['sha256'],row['path']
    return 'PASS_FROZEN_INPUTS_PREPARED_ASSIGNMENTS_AND_OUTPUT_MANIFEST'

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
    with np.load(path,allow_pickle=False) as z:
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
