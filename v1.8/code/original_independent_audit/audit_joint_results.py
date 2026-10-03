"""Recompute every joint contrast and refit fixed cases independently."""
import argparse,json
import numpy as np
import pandas as pd
from audit_result_helpers import *

def audit(dataset):
    p=D18/'joint_dinov2'/dataset;o=D18/'independent_review'/f'joint_results_{dataset}';o.mkdir(exist_ok=True)
    completed=json.loads((p/'completion.json').read_text(encoding='utf-8'))
    freeze=json.loads((p/'protocol_freeze.json').read_text(encoding='utf-8'))
    assert completed['protocol_sha256']==sha(p/'protocol_freeze.json')
    integrity_status=integrity(p,freeze)
    raw=pd.read_csv(p/'predictions.csv.gz',dtype={'row_idx':str,'nearest_train_row_idx':str,'component_id':str},keep_default_na=False)
    assert len(raw)==completed['predictions'] and completed['warnings']==0
    assert set(raw.model)=={'dinov2_logit','dinov2_cosine_1nn'} and set(raw.endpoint)=={'multiclass'}
    membership=pd.read_csv(p/'component_membership.csv',dtype=str,keep_default_na=False)
    labels=['healthy']+sorted(set(raw.label)-{'healthy'})
    rows=[];baselines=[]
    for (rep,model),part in raw.groupby(['repetition','model'],sort=True):
        g={key:v.reset_index(drop=True) for key,v in part.groupby('condition',sort=False)}
        z=g['zero'];y=z.truth.to_numpy(int);q0=z.predicted.to_numpy(int)
        r=membership[membership.repetition.eq(str(rep))].set_index('component_id')
        paired=z.component_id.map(r.paired).eq('1').to_numpy()
        for condition,v in g.items():
            assert np.array_equal(v.row_idx.to_numpy(),z.row_idx.to_numpy())
            assert np.array_equal(v.truth.to_numpy(),y)
            assert np.array_equal(v.predicted.to_numpy(),v[[f'p{i}' for i in range(len(labels))]].to_numpy().argmax(1))
        for pair in range(5):
            for arm in ['A','B']:
                e=g[f'exposure_{pair}_{arm}'];h=g[f'sham_{pair}_{arm}']
                sel=e.selected.to_numpy(int)==1;assert np.array_equal(sel,h.selected.to_numpy(int)==1)
                common=dict(repetition=rep,seed=20261003+rep,model=model,endpoint='multiclass',allocation_pair=pair,arm=arm)
                base={**common,'zero_BA':cm(q0[sel]==y[sel],y[sel]),'sham_BA':cm(h.predicted.to_numpy()[sel]==y[sel],y[sel]),
                    'exposure_BA':cm(e.predicted.to_numpy()[sel]==y[sel],y[sel])}
                base['E_minus_H']=base['exposure_BA']-base['sham_BA'];base['sham_error_headroom']=1-base['sham_BA'];baselines.append(base)
                panels={'selected_probe':sel,'unselected_paired_probe':paired&~sel,'all_paired_probe':paired,'sentinel':z.panel.eq('sentinel').to_numpy()}
                for panel,mask in panels.items():
                    for stratum in ['all_classes']+labels:
                        m=mask if stratum=='all_classes' else mask&z.label.eq(stratum).to_numpy()
                        rows.append({**common,'panel':panel,'stratum':stratum,'n':int(m.sum()),**metrics(y,q0,e.predicted.to_numpy(),h.predicted.to_numpy(),m)})
    result=pd.DataFrame(rows);result.to_csv(o/'independent_arm_contrasts.csv',index=False)
    keys=['repetition','seed','model','endpoint','allocation_pair','arm','panel','stratum'];values=[x for x in result if x not in keys+['n']]
    errors={'all_arm_contrasts':compare(result,pd.read_csv(p/'arm_contrasts.csv'),keys,values+['n'])}
    sk=['repetition','seed','model','endpoint','panel','stratum'];splits=result.groupby(sk,as_index=False)[values].mean()
    splits.to_csv(o/'independent_split_means.csv',index=False)
    errors['all_split_means']=compare(splits,pd.read_csv(p/'split_means.csv'),sk,values)
    summ=[];gk=['model','endpoint','panel','stratum']
    for key,g in splits.groupby(gk):
        for m in values:
            v=g[m].to_numpy();summ.append({**dict(zip(gk,key)),'metric':m,'mean':v.mean(),'split_sd':v.std(ddof=1),
                'split_p025':np.quantile(v,.025),'split_p975':np.quantile(v,.975),'split_count':len(v)})
    s=pd.DataFrame(summ);s.to_csv(o/'independent_split_summary.csv',index=False)
    errors['all_split_summary']=compare(s,pd.read_csv(p/'split_summary.csv'),gk+['metric'],['mean','split_sd','split_p025','split_p975','split_count'])
    b=pd.DataFrame(baselines);b.to_csv(o/'independent_selected_baseline_arms.csv',index=False)
    bv=['zero_BA','sham_BA','exposure_BA','E_minus_H','sham_error_headroom']
    bs=b.groupby(['repetition','seed','model'],as_index=False)[bv].mean();bs.to_csv(o/'independent_selected_baseline_splits.csv',index=False)
    bs.groupby('model',as_index=False)[bv].mean().to_csv(o/'independent_selected_baseline_summary.csv',index=False)
    dump(o/'STATISTICAL_RECONSTRUCTION.json',dict(status='PASS_RAW_RECONSTRUCTION',dataset=dataset,predictions=len(raw),arm_rows=len(result),max_abs_errors=errors))
    print(dataset+': all raw summaries independently reconstructed',flush=True)
    # Refit only the cases fixed before observing their outputs.
    frame=pd.read_csv(p/'manifest.csv',dtype=str,keep_default_na=False);x=load_matrix(freeze['inputs']['dinov2']['path'],frame)
    y=frame.label.map(dict(zip(labels,range(len(labels))))).to_numpy(int);pos={v:i for i,v in enumerate(frame.row_idx)}
    am=pd.read_csv(p/'allocation_membership.csv.gz',dtype=str,keep_default_na=False)
    checks=[]
    for rep in range(3):
        r=membership[membership.repetition.eq(str(rep))].set_index('component_id')
        for condition in ['exposure_0_A','sham_0_A']:
            m=am[(am.repetition.eq(str(rep)))&am.condition.eq(condition)&am.in_training.eq('1')]
            train=np.array(sorted(int(r.loc[c,col]) for c in m.component_id for col in ['view1_position','view2_position']))
            for model in ['dinov2_logit','dinov2_cosine_1nn']:
                v=raw[(raw.repetition.eq(rep))&raw.condition.eq(condition)&raw.model.eq(model)].reset_index(drop=True)
                test=np.array([pos[q] for q in v.row_idx]);common=dict(repetition=rep,seed=20261003+rep,condition=condition,model=model)
                if model=='dinov2_logit':check=independent_refit(x,y,train,test,20261003+rep,v,len(labels))
                else:check=independent_nn(x,y,train,test,v,frame)
                checks.append({**common,**check})
    pd.DataFrame(checks).to_csv(o/'independent_fixed_refit_checks.csv',index=False)
    report=dict(status='PASS_INDEPENDENT_RESULTS',dataset=dataset,protocol_sha256=sha(p/'protocol_freeze.json'),predictions_sha256=sha(p/'predictions.csv.gz'),
        predictions_checked=len(raw),all_raw_summary_max_errors=errors,logistic_refits=6,independent_nn_conditions=6,
        maximum_refit_probability_difference=max(c.get('max_probability_difference',0) for c in checks),
        artifact_integrity=integrity_status,
        scope='Complete raw primary/secondary contrast reconstruction, same-selected-mask baselines, fixed LR refits and cosine-neighbour checks; no new study.',
        scientific_submission_gate='UNCHANGED_HOLD')
    dump(D18/'independent_review'/f'{dataset.upper()}_JOINT_RESULT_REVIEW.json',report)
    print(json.dumps(report,ensure_ascii=False),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['onion','potato'],required=True);audit(p.parse_args().dataset)
