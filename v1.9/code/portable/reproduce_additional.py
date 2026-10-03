"""Rebuild all v1.9 descriptive tables from frozen portable inputs, without fitting.

This implementation does not import the original additional-analysis script. Joint
events and single-target events are reconstructed from complete recorded decisions.
Expected tables are opened only after all numerical tables have been calculated.
"""
import argparse
from pathlib import Path
import gzip
import hashlib
import json
from itertools import combinations
import platform
import numpy as np
import pandas as pd
import pyarrow
import pyarrow.parquet as pq

GROUP=['repetition','seed','model']
EVENT=['sham_BA','exposure_BA','policy_BA_difference','corrected_relative_to_sham',
       'harmed_relative_to_sham','sham_error_headroom','unchanged_correctness']
KEYS={
 'source_target_support.csv':['dataset','label'],
 'unique_target_frequency.csv':['dataset','label','target_component'],
 'cross_split_overlap.csv':['dataset','split_a','split_b'],
 'single_per_class_split_recall_headroom.csv':['dataset']+GROUP+['target_label'],
 'single_event_counts_not_independent_samples.csv':['dataset']+GROUP+['label'],
 'joint_per_class_split_recall_headroom.csv':['dataset']+GROUP+['label'],
 'joint_repeated_context_event_counts.csv':['dataset']+GROUP+['label'],
 'joint_repeated_context_confusion_records.csv':['dataset']+GROUP+['policy','label','predicted'],
 'all_common_loco_split_results.csv':['dataset','design']+GROUP+['weighting'],
 'all_common_loco_summary.csv':['dataset','design','model','weighting'],
 'frequency_weighted_split_results.csv':['dataset','design']+GROUP+['weighting'],
 'frequency_weighted_summary.csv':['dataset','design','model','weighting'],
 'headroom_interpretation_percentage_units.csv':['dataset','design','model'],
 'conditional_allocation_mcse.csv':['dataset','design','model','metric','weighting'],
}

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

class Inputs:
    def __init__(self,root):
        self.root=root
        self.bindings=json.loads((root/'additional_analysis/INPUT_BINDINGS.json').read_text(encoding='utf-8'))
        self.freeze=json.loads((root/'additional_analysis/ADDITIONAL_ANALYSIS_FREEZE.json').read_text(encoding='utf-8'))
        self.conversions=json.loads((root/'provenance/parquet_conversion_checks.json').read_text(encoding='utf-8'))
        self.read_rows={}
    def verify(self):
        assert sha(self.root/'additional_analysis/code/original/rebuild_additional_descriptive.py')==self.freeze['code_sha256']
        for item in self.freeze['inputs']:
            key=item['path'].replace('\\','/')
            assert self.bindings[key]['original_sha256']==item['sha256'],key
        for key,b in self.bindings.items():
            p=(self.root/b['relative_path']).resolve()
            assert p.is_relative_to(self.root),p
            assert sha(p)==b['distributed_sha256'],str(p)
            if p.suffix=='.parquet':
                rec=self.conversions[b['relative_path']]
                assert rec['original_sha256']==b['original_sha256']
                assert rec['distributed_sha256']==b['distributed_sha256']
                assert rec['validation']=='PASS_EVERY_COLUMN_EVERY_ROW_EXACT'
            elif p.name.endswith('.csv.gz') and b['original_sha256']!=b['distributed_sha256']:
                assert hashlib.sha256(gzip.decompress(p.read_bytes())).hexdigest()==b['original_sha256']
    def choose(self,fragment):
        matches=[b['relative_path'] for k,b in self.bindings.items() if fragment in k]
        assert len(matches)==1,(fragment,matches)
        return self.root/matches[0]
    def csv(self,fragment):return pd.read_csv(self.choose(fragment))
    def predictions(self,fragment,cols):
        p=self.choose(fragment)
        frame=pq.read_table(p,columns=cols).to_pandas()
        for c in ('repetition','seed','selected','is_own_target','truth','predicted'):
            if c in frame:frame[c]=pd.to_numeric(frame[c],errors='raise')
        self.read_rows[p.relative_to(self.root).as_posix()]=len(frame)
        return frame

def event_columns(frame):
    # E and H hold 0/1 correctness, not probabilities.
    frame['sham_BA']=frame['H']
    frame['exposure_BA']=frame['E']
    frame['policy_BA_difference']=frame.E-frame.H
    frame['corrected_relative_to_sham']=((frame.H==0)&(frame.E==1)).astype(int)
    frame['harmed_relative_to_sham']=((frame.H==1)&(frame.E==0)).astype(int)
    frame['sham_error_headroom']=1-frame.H
    frame['unchanged_correctness']=(frame.E==frame.H).astype(int)
    assert np.array_equal(frame.policy_BA_difference,frame.corrected_relative_to_sham-frame.harmed_relative_to_sham)
    return frame

def group_summary(frame,metric):
    keys=['dataset','design','model','weighting']
    rows=[]
    for key,g in frame.groupby(keys,sort=True):
        x=g[metric].to_numpy(float)
        rows.append(dict(zip(keys,key),mean=float(x.mean()),split_sd=float(x.std(ddof=1)),
                         split_min=float(x.min()),split_max=float(x.max()),split_count=len(x)))
    return pd.DataFrame(rows)

def rebuild(inputs):
    support=[];freq=[];overlap=[];single_classes=[];single_counts=[]
    joint_classes=[];joint_counts=[];confusions=[];split_rows=[];weighted=[]
    raw_single_errors=[]
    for task in ('onion','potato'):
        fragment=f'/single_source/{task}/'
        manifest=inputs.csv(fragment+'manifest.csv')
        target=inputs.csv(fragment+'targets.csv')
        membership=inputs.csv(fragment+'component_membership_first10.csv')
        sizes=manifest[['component_id','label']].drop_duplicates().groupby('label').size()
        for label,n in sizes.items():
            t=target.loc[target.target_label.eq(label)]
            counts=t.target_component.value_counts().sort_index()
            m=membership.loc[membership.label.eq(label)].groupby(['repetition','role']).size().unstack(fill_value=0)
            per=t.groupby('repetition').size()
            support.append(dict(dataset=task,label=label,manifest_source_units=int(n),unique_target_units=len(counts),
                target_within_split_records=len(t),target_occurrences_min=int(counts.min()),target_occurrences_max=int(counts.max()),
                probe_count_each_split_min=int(m.probe.min()),probe_count_each_split_max=int(m.probe.max()),
                sentinel_count_each_split_min=int(m.sentinel.min()),sentinel_count_each_split_max=int(m.sentinel.max()),
                paired_target_each_split_min=int(per.min()),paired_target_each_split_max=int(per.max())))
            freq.extend(dict(dataset=task,label=label,target_component=u,split_occurrences=int(nr)) for u,nr in counts.items())
        for a,b in combinations(range(10),2):
            targets=[set(target.loc[target.repetition.eq(r),'target_component']) for r in (a,b)]
            trains=[set(membership.loc[membership.repetition.eq(r)&membership.role.isin(['core','remove']),'component_id']) for r in (a,b)]
            assert all(len(t)==(240 if task=='onion' else 268) for t in trains)
            overlap.append(dict(dataset=task,split_a=a,split_b=b,target_intersection=len(targets[0]&targets[1]),
                target_jaccard=len(targets[0]&targets[1])/len(targets[0]|targets[1]),q0_train_intersection=len(trains[0]&trains[1]),
                q0_train_jaccard=len(trains[0]&trains[1])/len(trains[0]|trains[1]),a_target_in_b_training=len(targets[0]&trains[1])))

        # All single predictions are scanned; retain only target-owned E/H and q0.
        cols=GROUP+['target_component','target_label','policy','is_own_target','component_id','truth','predicted']
        p=inputs.predictions('raw_single_'+task,cols)
        p['correct']=p.truth.eq(p.predicted).astype(int)
        zero=p.loc[p.policy.eq('zero'),GROUP+['component_id','correct']].rename(columns={'component_id':'target_component','correct':'zero_BA'})
        own=p.loc[p.is_own_target.eq(1)]
        own=own.pivot(index=GROUP+['target_component','target_label'],columns='policy',values='correct').reset_index()
        assert own[['exposure','sham']].notna().all().all()
        own=own.rename(columns={'exposure':'E','sham':'H'}).merge(zero,on=GROUP+['target_component'],validate='many_to_one')
        own=event_columns(own)
        # Confirm derived legacy target table is the same, without using it to calculate.
        old=inputs.csv(fragment+'target_panel_contrasts.csv')
        old=old.loc[old.panel.eq('own_target')&old.weighting.eq('all_original_classes')]
        check=own.merge(old,on=GROUP+['target_component','target_label'],suffixes=('_raw','_saved'),validate='one_to_one')
        delta=max(float(np.abs(check[c+'_raw']-check[c+'_saved']).max()) for c in ['zero_BA']+EVENT[:-1])
        assert delta<1e-12 and len(check)==len(target)*5
        raw_single_errors.append({'dataset':task,'rows':len(check),'maximum_metric_difference':delta})
        metrics=['zero_BA']+EVENT
        sc=own.groupby(GROUP+['target_label'],as_index=False)[metrics].mean();sc['dataset']=task
        single_classes.append(sc)
        for k,g in own.groupby(GROUP+['target_label'],sort=True):
            rep,seed,model,label=k
            single_counts.append(dict(dataset=task,repetition=rep,seed=seed,model=model,label=label,target_records=len(g),
                corrected_events=int(g.corrected_relative_to_sham.sum()),harmed_events=int(g.harmed_relative_to_sham.sum()),
                unchanged_correctness_events=int(g.unchanged_correctness.sum()),sham_error_events=int(g.sham_error_headroom.sum()),
                zero_correct_events=int(g.zero_BA.sum()),sham_correct_events=int(g.H.sum()),exposure_correct_events=int(g.E.sum())))

        jparts=[]
        for legacy in ('v17','v18'):
            key=(f'Onion_Deep_Revision_20261003_v17/'+('matched_replacement_onion/' if task=='onion' else 'independent_task/potato/matched_replacement_experiment/')) if legacy=='v17' else f'/joint_dinov2/{task}/'
            cols=GROUP+['endpoint','condition','selected','component_id','label','truth','predicted']
            q=inputs.predictions(key+'predictions.csv.gz',cols)
            q=q.loc[q.endpoint.eq('multiclass')&q.selected.eq(1)].copy()
            parts=q.condition.str.split('_',expand=True)
            q['policy']=parts[0];q['allocation_pair']=parts[1].astype(int);q['arm']=parts[2]
            q['correct']=q.truth.eq(q.predicted).astype(int);jparts.append(q)
        j=pd.concat(jparts,ignore_index=True)
        idkeys=GROUP+['allocation_pair','arm','component_id','label']
        z=j.pivot(index=idkeys,columns='policy',values='correct').reset_index().rename(columns={'exposure':'E','sham':'H'})
        z=event_columns(z)
        # Class means are averaged over allocations/arms first, then classes.
        arm_class=z.groupby(GROUP+['allocation_pair','arm','label'],as_index=False)[EVENT].mean()
        jc=arm_class.groupby(GROUP+['label'],as_index=False)[EVENT].mean();jc['dataset']=task
        joint_classes.append(jc)
        counts=z.groupby(GROUP+['label']).agg(repeated_target_context_records=('component_id','size'),
            unique_target_units_in_split=('component_id','nunique'),corrected_events=('corrected_relative_to_sham','sum'),
            harmed_events=('harmed_relative_to_sham','sum'),unchanged_correctness_events=('unchanged_correctness','sum'),
            sham_error_events=('sham_error_headroom','sum')).reset_index();counts['dataset']=task;joint_counts.append(counts)
        cf=j.groupby(GROUP+['policy','label','predicted']).size().reset_index(name='repeated_prediction_records');cf['dataset']=task;confusions.append(cf)
        schemes={'all_original_classes':list(sizes.index),'common_classes_n_ge_30':list(sizes[sizes>=30].index)}
        schemes.update({'leave_out_'+x:[y for y in sizes.index if y!=x] for x in sizes.index})
        for design,classes,labelcol,values in [('single',sc,'target_label',metrics),('joint',jc,'label',EVENT)]:
            for k,g in classes.groupby(GROUP,sort=True):
                meta=dict(zip(GROUP,k),dataset=task,design=design)
                for name,labels in schemes.items():
                    select=g.loc[g[labelcol].isin(labels),values].to_numpy(float)
                    split_rows.append(dict(meta,weighting=name,**dict(zip(values,select.mean(axis=0)))))
                weights=sizes.reindex(g[labelcol]).to_numpy(float)
                weighted.append(dict(meta,weighting='manifest_source_frequency',value=float(np.dot(g.policy_BA_difference,weights)/weights.sum())))
        for k,g in own.groupby(GROUP,sort=True):
            weighted.append(dict(zip(GROUP,k),dataset=task,design='single',weighting='actual_target_frequency_accuracy',value=float(g.policy_BA_difference.mean())))
        joint_micro=z.groupby(GROUP+['allocation_pair','arm']).policy_BA_difference.mean().groupby(GROUP).mean()
        weighted.extend(dict(zip(GROUP,k),dataset=task,design='joint',weighting='actual_target_frequency_accuracy',value=float(v)) for k,v in joint_micro.items())
        print(json.dumps({'stage':'raw_reconstruction_complete','dataset':task,'single_target_model_rows':len(own),'joint_selected_policy_rows':len(j)}),flush=True)

    splits=pd.DataFrame(split_rows)
    summary=group_summary(splits,'policy_BA_difference')
    key=['dataset','design','model','weighting']
    for col in [v for v in EVENT if v!='policy_BA_difference']:
        summary=summary.merge(splits.groupby(key,as_index=False)[col].mean(),on=key,validate='one_to_one')
    summary['correction_fraction_of_available_sham_errors']=summary.corrected_relative_to_sham/summary.sham_error_headroom
    summary['net_fraction_of_available_sham_errors']=summary['mean']/summary.sham_error_headroom
    names={'sham_error_headroom':'sham_error_headroom_percent','corrected_relative_to_sham':'corrected_pp','harmed_relative_to_sham':'harmed_pp',
           'mean':'net_contrast_pp','correction_fraction_of_available_sham_errors':'corrected_over_sham_error_percent','net_fraction_of_available_sham_errors':'net_contrast_over_sham_error_percent'}
    head=summary.loc[summary.weighting.eq('all_original_classes')&summary.model.isin(['resnet_logit','dinov2_logit']),['dataset','design','model']+list(names)].copy()
    head[list(names)]=head[list(names)]*100;head=head.rename(columns=names)
    mc=summary.loc[summary.weighting.eq('all_original_classes'),['dataset','design','model','weighting','split_count','mean','split_sd']].copy()
    mc['metric']='policy_BA_difference';mc['mean_pp']=mc['mean']*100;mc['split_sd_pp']=mc.split_sd*100;mc['mcse_pp']=mc.split_sd_pp/np.sqrt(mc.split_count)
    mc=mc.drop(columns=['mean','split_sd'])
    wf=pd.DataFrame(weighted)
    tables={'source_target_support.csv':pd.DataFrame(support),'unique_target_frequency.csv':pd.DataFrame(freq),
        'cross_split_overlap.csv':pd.DataFrame(overlap),'single_per_class_split_recall_headroom.csv':pd.concat(single_classes),
        'single_event_counts_not_independent_samples.csv':pd.DataFrame(single_counts),'joint_per_class_split_recall_headroom.csv':pd.concat(joint_classes),
        'joint_repeated_context_event_counts.csv':pd.concat(joint_counts),'joint_repeated_context_confusion_records.csv':pd.concat(confusions),
        'all_common_loco_split_results.csv':splits,'all_common_loco_summary.csv':summary,
        'frequency_weighted_split_results.csv':wf,'frequency_weighted_summary.csv':group_summary(wf,'value'),
        'headroom_interpretation_percentage_units.csv':head,'conditional_allocation_mcse.csv':mc}
    return tables,raw_single_errors

def check_tables(tables,root,out):
    report=[]
    hashes=json.loads((root/'additional_analysis/EXPECTED_TABLE_HASHES.json').read_text(encoding='utf-8'))
    assert set(hashes)==set(tables)
    for name,a in tables.items():
        p=root/'additional_analysis/expected'/name
        assert sha(p)==hashes[name]['sha256']
        b=pd.read_csv(p);keys=KEYS[name]
        assert set(a.columns)==set(b.columns),(name,set(a.columns)^set(b.columns))
        a=a[b.columns].sort_values(keys).reset_index(drop=True);b=b.sort_values(keys).reset_index(drop=True)
        assert len(a)==len(b),name
        err=0.
        for c in b:
            if pd.api.types.is_numeric_dtype(b[c]):
                x=a[c].to_numpy(float);y=b[c].to_numpy(float)
                assert np.array_equal(np.isnan(x),np.isnan(y)),(name,c,'missing values')
                delta=np.abs(x-y);e=float(np.nanmax(delta)) if np.isfinite(delta).any() else 0.
                err=max(err,e);assert e<=1e-10,(name,c,e)
            else:assert a[c].astype(str).tolist()==b[c].astype(str).tolist(),(name,c)
        a.to_csv(out/name,index=False)
        report.append({'file':name,'rows':len(a),'maximum_numeric_difference':err,'status':'PASS'})
    return report

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2])
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();root=args.root.resolve();out=args.out.resolve()
    assert not out.is_relative_to(root/'additional_analysis/expected'),'Do not overwrite expected tables'
    out.mkdir(parents=True,exist_ok=True)
    data=Inputs(root);data.verify()
    tables,rawsingle=rebuild(data)
    checks=check_tables(tables,root,out)
    data.verify()
    report={'status':'PASS_INDEPENDENT_PORTABLE_DESCRIPTIVE_RECONSTRUCTION','tables':checks,
        'prediction_rows_scanned':sum(data.read_rows.values()),'prediction_files':data.read_rows,
        'single_raw_vs_saved_target_checks':rawsingle,'fitted_models':0,'p_values':0,'population_confidence_intervals':0,
        'input_bindings_sha256':sha(root/'additional_analysis/INPUT_BINDINGS.json'),'freeze_sha256':sha(root/'additional_analysis/ADDITIONAL_ANALYSIS_FREEZE.json'),
        'script_sha256':sha(Path(__file__)),'versions':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'pyarrow':pyarrow.__version__},
        'scope':'Comment-prompted descriptive sensitivity after v18 results were known. MCSE is conditional on the fixed archive and saved allocation design; it is not a plant-population interval or new independent validation.'}
    (out/'ADDITIONAL_PORTABLE_VERIFICATION.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':report['status'],'tables':len(checks),'prediction_rows_scanned':report['prediction_rows_scanned'],
        'max_table_difference':max(r['maximum_numeric_difference'] for r in checks)},indent=2),flush=True)

if __name__=='__main__':main()
