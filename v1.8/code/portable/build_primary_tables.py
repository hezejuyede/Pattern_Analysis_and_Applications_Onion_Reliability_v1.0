"""Rebuild manuscript-wide primary tables from independently reconstructed outputs."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

SCHEMES=['all_original_classes','common_classes_n_ge_30']

def check(actual,expected,keys,values):
    a=actual.sort_values(keys).reset_index(drop=True);b=expected.sort_values(keys).reset_index(drop=True)
    assert len(a)==len(b),(len(a),len(b))
    pd.testing.assert_frame_equal(a[keys].astype(str),b[keys].astype(str),check_dtype=False)
    error=float(np.max(np.abs(a[values].to_numpy(float)-b[values].to_numpy(float))))
    assert error<1e-10,error
    return error

def summarize(frame,design):
    keys=['dataset','design','panel','metric','weighting','model'];rows=[]
    for key,part in frame.groupby(keys):
        x=part.value.to_numpy(float)*100
        assert len(x)==(30 if design=='joint' else 10)
        lo,hi=np.quantile(x,[.025,.975]) if design=='joint' else (x.min(),x.max())
        rows.append(dict(zip(keys,key))|dict(mean_pp=x.mean(),lower_pp=lo,upper_pp=hi,split_count=len(x)))
    return pd.DataFrame(rows)

def build(out,root):
    out=Path(out);root=Path(root);dest=out/'primary_tables';dest.mkdir(exist_ok=True)
    joint=[];single=[];context=[];baselines=[]
    for dataset in ['onion','potato']:
        protocol=json.loads((root/'experiments/v18_single'/dataset/'protocol_freeze.json').read_text(encoding='utf-8'))
        j=pd.concat([pd.read_csv(out/f'{v}_{dataset}'/'independent_split_means.csv') for v in ['v17','v18_joint']],ignore_index=True)
        j=j[j.endpoint.eq('multiclass')]
        for panel,metric in [('selected_probe','policy_BA_difference'),('sentinel','turnover_difference')]:
            p=j[j.panel.eq(panel)]
            for scheme in SCHEMES:
                if scheme==SCHEMES[0]:a=p[p.stratum.eq('all_classes')][['repetition','seed','model',metric]]
                else:a=p[p.stratum.isin(protocol['common_classes'])].groupby(['repetition','seed','model'],as_index=False)[metric].mean()
                for r in a.to_dict('records'):joint.append(dict(dataset=dataset,design='joint',panel=panel,metric=metric,weighting=scheme,repetition=r['repetition'],seed=r['seed'],model=r['model'],value=r[metric]))
        s=pd.read_csv(out/f'v18_single_{dataset}'/'independent_split_means.csv')
        for panel,metric in [('own_target','policy_BA_difference'),('sentinel','turnover_difference')]:
            for r in s[s.panel.eq(panel)].to_dict('records'):single.append(dict(dataset=dataset,design='single',panel=panel,metric=metric,weighting=r['weighting'],repetition=r['repetition'],seed=r['seed'],model=r['model'],value=r[metric]))
        targets=pd.read_csv(out/f'v18_single_{dataset}'/'independent_single_vs_joint_all_models_targets.csv',dtype={'target_component':str})
        targets=targets.rename(columns={'policy_BA_difference':'single_E_minus_H'})
        values=['single_E_minus_H','joint_E_minus_H','single_minus_joint']
        for scheme in SCHEMES:
            t=targets if scheme==SCHEMES[0] else targets[targets.target_label.isin(protocol['common_classes'])]
            a=t.groupby(['repetition','seed','model','target_label'],as_index=False)[values].mean().groupby(['repetition','seed','model'],as_index=False)[values].mean()
            a['dataset']=dataset;a['weighting']=scheme;context.append(a)
        b=pd.concat([pd.read_csv(out/f'{v}_{dataset}'/'independent_selected_baseline_splits.csv') for v in ['v17','v18_joint']],ignore_index=True)
        b['dataset']=dataset;baselines.append(b[b.endpoint.eq('multiclass')])
    joint=pd.DataFrame(joint);single=pd.DataFrame(single);ctx=pd.concat(context,ignore_index=True)
    base=pd.concat(baselines,ignore_index=True)
    base.to_csv(dest/'selected_same_mask_zero_sham_exposure_split_means.csv',index=False)
    base.groupby(['dataset','model'],as_index=False)[['zero_BA','sham_BA','exposure_BA','E_minus_H','sham_error_headroom']].mean().to_csv(dest/'selected_same_mask_zero_sham_exposure_summary.csv',index=False)
    errors={}
    for name,frame,design in [('Fig_2_Joint_Replacement',joint,'joint'),('Fig_3_Single_Source_Replacement',single,'single')]:
        frame.to_csv(dest/(name+'_split_source.csv'),index=False);summary=summarize(frame,design)
        summary.to_csv(dest/(name+'_plot_source.csv'),index=False)
        expected=root/'expected/figures'/(name+'_plot_source.csv')
        if expected.exists():errors[name]=check(summary,pd.read_csv(expected),['dataset','design','panel','metric','weighting','model'],['mean_pp','lower_pp','upper_pp','split_count'])
    ctx.to_csv(dest/'same_target_context_split_means.csv',index=False)
    rows=[]
    for key,g in ctx.groupby(['dataset','model','weighting']):
        assert len(g)==10
        for metric in values:
            x=g[metric].to_numpy(float)
            rows.append(dict(zip(['dataset','model','weighting'],key))|dict(metric=metric,mean=x.mean(),split_min=x.min(),split_max=x.max(),split_sd=x.std(ddof=1),split_count=len(x)))
    summary=pd.DataFrame(rows);summary.to_csv(dest/'same_target_context_summary.csv',index=False)
    expected=root/'expected/context_summary.csv'
    if expected.exists():errors['same_target_context']=check(summary,pd.read_csv(expected),['dataset','model','weighting','metric'],['mean','split_min','split_max','split_sd','split_count'])
    (dest/'PRIMARY_TABLE_RECONSTRUCTION.json').write_text(json.dumps({'status':'PASS','errors':errors,'units':'Figure tables are percentage points; other tables are fractions.','interpretation':'Class weighting sensitivities are post hoc. Joint 30-split percentiles and single 10-split min/max are design ranges, not confidence intervals.'},indent=2)+'\n',encoding='utf-8')
    print('All primary, same-mask baseline, figure-source and matched-context tables reconstructed.',flush=True)
