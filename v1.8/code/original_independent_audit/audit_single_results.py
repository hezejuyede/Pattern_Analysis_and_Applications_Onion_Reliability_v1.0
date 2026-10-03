"""Stream complete single-source raw outcomes; independently audit weighting/refits."""
import argparse,csv,gzip,itertools,json
import numpy as np
import pandas as pd
from audit_result_helpers import *

SCHEMES=['all_original_classes','common_classes_n_ge_30']
MODELS=['resnet_logit','colour_logit','dinov2_logit','resnet_cosine_1nn','dinov2_cosine_1nn']

def grouped_predictions(path):
    with gzip.open(path,'rt',encoding='utf-8',newline='') as stream:
        reader=csv.DictReader(stream)
        key=lambda r:(int(r['repetition']),r['model'],r['target_component'],r['policy'])
        for k,g in itertools.groupby(reader,key):yield k,list(g)

def audit(dataset):
    p=D18/'single_source'/dataset;o=D18/'independent_review'/f'single_results_{dataset}';o.mkdir(exist_ok=True)
    comp=json.loads((p/'completion.json').read_text(encoding='utf-8'));freeze=json.loads((p/'protocol_freeze.json').read_text(encoding='utf-8'))
    assert comp['protocol_sha256']==sha(p/'protocol_freeze.json') and comp['warnings']==0
    integrity_status=integrity(p,freeze)
    labels=freeze['class_order'];k=len(labels);frame=pd.read_csv(p/'manifest.csv',dtype=str,keep_default_na=False)
    target_table=pd.read_csv(p/'targets.csv',dtype={'target_component':str})
    selected=target_table[target_table.repetition<2].sort_values('target_component').groupby(['repetition','target_label'],as_index=False).first()
    fixed_cases=set(zip(selected.repetition.astype(int),selected.target_component))
    zeros={};exposure=None;rows=[];fixed_predictions={};seen=set();n=0;group_count=0
    for key,g in grouped_predictions(p/'predictions.csv.gz'):
        rep,model,target,policy=key;assert key not in seen,key;seen.add(key)
        n+=len(g);group_count+=1
        prob=np.array([[float(r[f'p{i}']) for i in range(k)] for r in g])
        pred=np.array([int(r['predicted']) for r in g]);truth=np.array([int(r['truth']) for r in g])
        rid=np.array([r['row_idx'] for r in g]);cid=np.array([r['component_id'] for r in g])
        l=np.array([r['label'] for r in g]);panel=np.array([r['panel'] for r in g])
        assert np.array_equal(pred,prob.argmax(1)) and np.allclose(prob.sum(1),1)
        assert np.array_equal(truth,np.array([labels.index(q) for q in l]))
        if policy=='zero':
            assert target=='';zeros[rep,model]=dict(prob=prob,pred=pred,truth=truth,row_idx=rid,component_id=cid,labels=l,panel=panel)
            continue
        z=zeros[rep,model]
        assert np.array_equal(rid,z['row_idx']) and np.array_equal(truth,z['truth']) and np.array_equal(cid,z['component_id'])
        if (rep,target) in fixed_cases:fixed_predictions[key]=pd.DataFrame(g)
        if policy=='exposure':
            assert exposure is None
            exposure=(key,prob,pred);continue
        assert policy=='sham' and exposure[0]==(rep,model,target,'exposure')
        ep=exposure[1];e=exposure[2];exposure=None
        target_label=g[0]['target_label'];own=cid==target;assert own.sum()==1
        assert all(r['target_label']==target_label for r in g)
        for scheme in SCHEMES:
            if scheme==SCHEMES[1] and target_label not in freeze['common_classes']:continue
            supported=np.ones(len(g),dtype=bool) if scheme==SCHEMES[0] else np.isin(l,freeze['common_classes'])
            for name,mask in [('own_target',own),('all_unexposed_probe',(panel=='probe')&~own),('sentinel',panel=='sentinel')]:
                mask=mask&supported;assert mask.any()
                rows.append(dict(repetition=rep,seed=20261003+rep,model=model,endpoint='multiclass',target_component=target,
                    target_label=target_label,panel=name,weighting=scheme,n_evaluation=int(mask.sum()),
                    n_evaluation_classes=len(np.unique(truth[mask])),**metrics(truth,z['pred'],e,pred,mask,(ep,prob))))
    assert exposure is None and n==comp['predictions'] and group_count==comp['model_conditions']
    result=pd.DataFrame(rows);result.to_csv(o/'independent_target_panel_contrasts.csv',index=False)
    keys=['repetition','seed','model','endpoint','panel','weighting','target_label','target_component']
    values=[c for c in result if c not in keys+['n_evaluation','n_evaluation_classes']]
    errs={'target_panel':compare(result,pd.read_csv(p/'target_panel_contrasts.csv'),keys,values+['n_evaluation','n_evaluation_classes'])}
    ck=[q for q in keys if q!='target_component'];byclass=result.groupby(ck,as_index=False)[values].mean()
    byclass.to_csv(o/'independent_class_split_means.csv',index=False)
    errs['class_means']=compare(byclass,pd.read_csv(p/'target_class_split_means.csv'),ck,values)
    sk=[q for q in ck if q!='target_label'];splits=byclass.groupby(sk,as_index=False)[values].mean()
    splits.to_csv(o/'independent_split_means.csv',index=False)
    errs['split_means']=compare(splits,pd.read_csv(p/'split_means.csv'),sk,values)
    gk=['model','endpoint','panel','weighting'];summary=[]
    for key,part in splits.groupby(gk):
        assert len(part)==10
        for metric in values:
            x=part[metric].to_numpy();summary.append({**dict(zip(gk,key)),'metric':metric,'mean':x.mean(),'split_min':x.min(),
                'split_max':x.max(),'split_sd':x.std(ddof=1),'split_count':len(x)})
    summary=pd.DataFrame(summary);summary.to_csv(o/'independent_split_summary.csv',index=False)
    errs['summary']=compare(summary,pd.read_csv(p/'split_summary.csv'),gk+['metric'],['mean','split_min','split_max','split_sd','split_count'])
    dump(o/'RAW_RECONSTRUCTION.json',dict(status='PASS_RAW_RECONSTRUCTION',predictions=n,conditions=group_count,errors=errs))
    print(dataset+': single raw outcomes, class/split weighting all match',flush=True)
    # Match q0 and selected-target joint policies using raw data, including DINO.
    old=(D17/'matched_replacement_onion' if dataset=='onion' else D17/'independent_task/potato/matched_replacement_experiment')
    joint_parts=[];q0_max=0.;q0_rows=0
    for source in [old,D18/'joint_dinov2'/dataset]:
        for chunk in pd.read_csv(source/'predictions.csv.gz',chunksize=150000,dtype={'row_idx':str,'component_id':str}):
            part=chunk[(chunk.repetition<10)&chunk.endpoint.eq('multiclass')]
            for (rep,model),g in part[part.condition.eq('zero')].groupby(['repetition','model']):
                # A chunk can end part-way through q0: match by row identifier.
                z=zeros[int(rep),model];index={r:i for i,r in enumerate(z['row_idx'])};idx=[index[r] for r in g.row_idx]
                err=float(np.abs(z['prob'][idx]-g[[f'p{i}' for i in range(k)]].to_numpy()).max())
                assert err<1e-11 and np.array_equal(z['pred'][idx],g.predicted.to_numpy());q0_max=max(q0_max,err);q0_rows+=len(g)
            selected_rows=part[part.selected.eq(1)].copy()
            if len(selected_rows):joint_parts.append(selected_rows[['repetition','seed','model','condition','component_id','label','truth','predicted']])
    assert q0_rows==sum(len(z['row_idx']) for z in zeros.values())
    j=pd.concat(joint_parts,ignore_index=True);sp=j.condition.str.split('_',expand=True)
    j['policy']=sp[0];j['pair']=sp[1].astype(int);j['arm']=sp[2];j['correct']=j.predicted.eq(j.truth).astype(int)
    jk=['repetition','seed','model','component_id','label','pair','arm']
    jt=j.pivot(index=jk,columns='policy',values='correct').reset_index();assert jt[['exposure','sham']].notna().all().all()
    jt['joint_E_minus_H']=jt.exposure-jt.sham
    tg=['repetition','seed','model','component_id','label'];jm=jt.groupby(tg,as_index=False).agg(joint_E_minus_H=('joint_E_minus_H','mean'),joint_allocations=('joint_E_minus_H','size'))
    assert jm.joint_allocations.eq(5).all()
    jm=jm.rename(columns={'component_id':'target_component','label':'target_label'})
    own=result[result.panel.eq('own_target')&result.weighting.eq(SCHEMES[0])]
    merged=own.merge(jm,on=['repetition','seed','model','target_component','target_label'],validate='one_to_one')
    assert len(merged)==len(own)
    merged['single_minus_joint']=merged.policy_BA_difference-merged.joint_E_minus_H
    merged.to_csv(o/'independent_single_vs_joint_all_models_targets.csv',index=False)
    jvalues=['policy_BA_difference','joint_E_minus_H','single_minus_joint'];jc=['repetition','seed','model','target_label']
    js=merged.groupby(jc,as_index=False)[jvalues].mean().groupby(['repetition','seed','model'],as_index=False)[jvalues].mean()
    js.to_csv(o/'independent_single_vs_joint_all_models_splits.csv',index=False)
    old_expected=pd.read_csv(p/'single_vs_v17_joint_targets.csv');old_expected=old_expected[old_expected.joint_E_minus_H.notna()]
    keep=merged[~merged.model.str.startswith('dinov2')]
    errs['legacy_joint_target_difference']=compare(keep,old_expected,['repetition','seed','model','target_component','target_label'],jvalues)
    errs['legacy_joint_split_difference']=compare(js[~js.model.str.startswith('dinov2')],pd.read_csv(p/'single_vs_v17_joint_split_means.csv'),['repetition','seed','model'],jvalues)
    dump(o/'Q0_AND_JOINT_COMPARISON.json',dict(status='PASS',q0_rows=q0_rows,max_q0_probability_difference=q0_max,complete_joint_target_comparisons=len(merged),errors=errs))
    print(dataset+': q0 parity and joint same-target comparisons match',flush=True)
    # Independently refit the first target by original label for two fixed seeds.
    inputs=freeze['inputs'];matrices={name:load_matrix(inputs[key]['path'],frame) for name,key in [('resnet_logit','resnet_features'),('colour_logit','colour_features'),('dinov2_logit','dino_features')]}
    y=frame.label.map(dict(zip(labels,range(k)))).to_numpy(int);position={r:i for i,r in enumerate(frame.row_idx)}
    records=pd.read_csv(p/'component_membership_first10.csv',dtype=str,keep_default_na=False)
    refits=[]
    for rep,target in sorted(fixed_cases):
        r=records[records.repetition.eq(str(rep))].set_index('component_id')
        zero=set(r.index[r.role.isin(['core','remove'])]);donor=r.loc[target,'remove'];alternative=r.loc[target,'alternate']
        for policy,insertion in [('exposure',target),('sham',alternative)]:
            sources=(zero-{donor})|{insertion}
            train=np.array(sorted(int(r.loc[c,v]) for c in sources for v in ['view1_position','view2_position']))
            for model in MODELS:
                expected=fixed_predictions[rep,model,target,policy].copy()
                expected['predicted']=expected.predicted.astype(int)
                test=np.array([position[x] for x in expected.row_idx])
                feature='dinov2_logit' if model.startswith('dinov2') else 'colour_logit' if model=='colour_logit' else 'resnet_logit'
                if model.endswith('1nn'):check=independent_nn(matrices[feature],y,train,test,expected,frame)
                else:check=independent_refit(matrices[feature],y,train,test,20261003+rep,expected,k)
                refits.append(dict(repetition=rep,model=model,target_component=target,target_label=r.loc[target,'label'],policy=policy,**check))
    pd.DataFrame(refits).to_csv(o/'independent_fixed_refit_checks.csv',index=False)
    report=dict(status='PASS_INDEPENDENT_RESULTS',dataset=dataset,predictions_checked=n,conditions_checked=group_count,
        protocol_sha256=sha(p/'protocol_freeze.json'),predictions_sha256=sha(p/'predictions.csv.gz'),all_reconstruction_errors=errs,
        q0_rows_checked=q0_rows,q0_max_probability_difference=q0_max,
        logistic_refits=sum(not c['model'].endswith('1nn') for c in refits),independent_nn_conditions=sum(c['model'].endswith('1nn') for c in refits),
        maximum_refit_probability_difference=max(c.get('max_probability_difference',0) for c in refits),
        artifact_integrity=integrity_status,
        single_vs_joint_targets_all_models=len(merged),scientific_submission_gate='UNCHANGED_HOLD')
    dump(D18/'independent_review'/f'{dataset.upper()}_SINGLE_SOURCE_RESULT_REVIEW.json',report)
    print(json.dumps(report,ensure_ascii=False),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['onion','potato'],required=True);audit(p.parse_args().dataset)
