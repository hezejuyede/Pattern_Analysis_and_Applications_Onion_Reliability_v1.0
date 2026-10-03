"""Read-only v1.8 review diagnostics. No fitting, re-selection, CI, or p-values."""
from pathlib import Path
from itertools import combinations
import hashlib
import json
import numpy as np
import pandas as pd

D19=Path(__file__).resolve().parent.parent
OUT=D19/'additional_analysis'
D18=D19.parent/'Onion_Deep_Revision_20261003_v18'
D17=D18.parent/'Onion_Deep_Revision_20261003_v17'
INPUTS={}
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def register(p):
    INPUTS[str(p)]=sha(p)
    return p
def read(p):return pd.read_csv(register(p))
def write(frame,name):frame.to_csv(OUT/name,index=False)
def summarize(frame,keys,metric):
    return frame.groupby(keys)[metric].agg(mean='mean',split_sd='std',split_min='min',split_max='max',split_count='count').reset_index()

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    supports=[]; frequencies=[]; overlap=[]; own_classes=[]; own_splits=[]; own_counts=[]; weighted=[]; joint_classes=[]; joint_counts=[]; confusion=[]
    for task in ('onion','potato'):
        single=D18/'single_source'/task
        manifest=read(single/'manifest.csv')
        sources=manifest[['component_id','label']].drop_duplicates()
        sizes=sources.groupby('label').size()
        target=read(single/'targets.csv')
        membership=read(single/'component_membership_first10.csv')
        for label,n in sizes.items():
            t=target[target.target_label.eq(label)]
            counts=t.groupby('target_component').size()
            m=membership[membership.label.eq(label)]
            byrole=m.groupby(['repetition','role']).size().unstack(fill_value=0)
            supports.append(dict(dataset=task,label=label,manifest_source_units=int(n),unique_target_units=int(t.target_component.nunique()),target_within_split_records=len(t),target_occurrences_min=int(counts.min()),target_occurrences_max=int(counts.max()),probe_count_each_split_min=int(byrole['probe'].min()),probe_count_each_split_max=int(byrole['probe'].max()),sentinel_count_each_split_min=int(byrole['sentinel'].min()),sentinel_count_each_split_max=int(byrole['sentinel'].max()),paired_target_each_split_min=int(t.groupby('repetition').size().min()),paired_target_each_split_max=int(t.groupby('repetition').size().max())))
            for unit,nr in counts.items():frequencies.append(dict(dataset=task,label=label,target_component=unit,split_occurrences=int(nr)))
        for a,b in combinations(range(10),2):
            ta=set(target.loc[target.repetition.eq(a),'target_component']);tb=set(target.loc[target.repetition.eq(b),'target_component'])
            # q0 membership roles are recorded by the frozen assignment code.
            ma=membership[membership.repetition.eq(a)];mb=membership[membership.repetition.eq(b)]
            train_a=set(ma.loc[ma.role.isin(['core','remove']),'component_id']);train_b=set(mb.loc[mb.role.isin(['core','remove']),'component_id'])
            assert len(train_a)==(240 if task=='onion' else 268)
            overlap.append(dict(dataset=task,split_a=a,split_b=b,target_intersection=len(ta&tb),target_jaccard=len(ta&tb)/len(ta|tb),q0_train_intersection=len(train_a&train_b),q0_train_jaccard=len(train_a&train_b)/len(train_a|train_b),a_target_in_b_training=len(ta&train_b)))
        own=read(single/'target_panel_contrasts.csv')
        own=own[own.panel.eq('own_target')&own.weighting.eq('all_original_classes')].copy()
        metrics=['zero_BA','sham_BA','exposure_BA','policy_BA_difference','corrected_relative_to_sham','harmed_relative_to_sham','sham_error_headroom']
        assert np.allclose(own.policy_BA_difference,own.corrected_relative_to_sham-own.harmed_relative_to_sham)
        assert (own.corrected_relative_to_sham<=own.sham_error_headroom+1e-12).all()
        own['unchanged_correctness']=1-own.corrected_relative_to_sham-own.harmed_relative_to_sham
        metrics+=['unchanged_correctness']
        cls=own.groupby(['repetition','seed','model','target_label'],as_index=False)[metrics].mean()
        cls['dataset']=task
        own_classes.append(cls)
        for (rep,seed,model),p in own.groupby(['repetition','seed','model']):
            for label,q in p.groupby('target_label'):
                own_counts.append(dict(dataset=task,repetition=rep,seed=seed,model=model,label=label,target_records=len(q),corrected_events=int(q.corrected_relative_to_sham.sum()),harmed_events=int(q.harmed_relative_to_sham.sum()),unchanged_correctness_events=int(q.unchanged_correctness.sum()),sham_error_events=int(q.sham_error_headroom.sum()),zero_correct_events=int(q.zero_BA.sum()),sham_correct_events=int(q.sham_BA.sum()),exposure_correct_events=int(q.exposure_BA.sum())))
        for (rep,seed,model),p in cls.groupby(['repetition','seed','model']):
            schemes={'all_original_classes':list(sizes.index),'common_classes_n_ge_30':list(sizes[sizes>=30].index)}
            schemes.update({'leave_out_'+label:[c for c in sizes.index if c!=label] for label in sizes.index})
            for name,labels in schemes.items():
                row=p[p.target_label.isin(labels)][metrics].mean().to_dict()
                row.update(dataset=task,repetition=rep,seed=seed,model=model,weighting=name,design='single')
                own_splits.append(row)
            labelweights=sizes.reindex(p.target_label).to_numpy(float)
            weighted.append(dict(dataset=task,repetition=rep,seed=seed,model=model,design='single',weighting='manifest_source_frequency',value=np.average(p.policy_BA_difference,weights=labelweights)))
        micro=own.groupby(['repetition','seed','model'],as_index=False).policy_BA_difference.mean()
        for r in micro.itertuples():weighted.append(dict(dataset=task,repetition=r.repetition,seed=r.seed,model=r.model,design='single',weighting='actual_target_frequency_accuracy',value=r.policy_BA_difference))
        old=D17/('matched_replacement_onion' if task=='onion' else 'independent_task/potato/matched_replacement_experiment')
        joint=[]
        for folder in (old,D18/'joint_dinov2'/task):
            p=register(folder/'predictions.csv.gz')
            cols=['repetition','seed','model','endpoint','condition','selected','component_id','label','truth','predicted']
            for chunk in pd.read_csv(p,usecols=cols,chunksize=200000):
                q=chunk[chunk.endpoint.eq('multiclass')&chunk.selected.eq(1)].copy()
                if not len(q):continue
                names=q.condition.str.split('_',expand=True)
                q['policy']=names[0];q['allocation_pair']=names[1].astype(int);q['arm']=names[2]
                q['correct']=q.predicted.eq(q.truth).astype(int)
                joint.append(q)
        j=pd.concat(joint,ignore_index=True)
        keys=['repetition','seed','model','allocation_pair','arm','component_id','label']
        z=j.pivot(index=keys,columns='policy',values='correct').reset_index()
        assert z[['exposure','sham']].notna().all().all()
        z['corrected_relative_to_sham']=((z.sham==0)&(z.exposure==1)).astype(int)
        z['harmed_relative_to_sham']=((z.sham==1)&(z.exposure==0)).astype(int)
        z['sham_error_headroom']=1-z.sham
        z['policy_BA_difference']=z.exposure-z.sham
        z['sham_BA']=z.sham;z['exposure_BA']=z.exposure
        z['unchanged_correctness']=1-z.corrected_relative_to_sham-z.harmed_relative_to_sham
        jm=[m for m in metrics if m!='zero_BA']
        arms=z.groupby(['repetition','seed','model','allocation_pair','arm','label'],as_index=False)[jm].mean()
        jc=arms.groupby(['repetition','seed','model','label'],as_index=False)[jm].mean()
        jc['dataset']=task;joint_classes.append(jc)
        counts=z.groupby(['repetition','seed','model','label']).agg(repeated_target_context_records=('component_id','size'),unique_target_units_in_split=('component_id','nunique'),corrected_events=('corrected_relative_to_sham','sum'),harmed_events=('harmed_relative_to_sham','sum'),unchanged_correctness_events=('unchanged_correctness','sum'),sham_error_events=('sham_error_headroom','sum')).reset_index()
        counts['dataset']=task;joint_counts.append(counts)
        cf=j.groupby(['repetition','seed','model','policy','label','predicted']).size().reset_index(name='repeated_prediction_records')
        cf['dataset']=task;confusion.append(cf)
        for (rep,seed,model),p in jc.groupby(['repetition','seed','model']):
            schemes={'all_original_classes':list(sizes.index),'common_classes_n_ge_30':list(sizes[sizes>=30].index)}
            schemes.update({'leave_out_'+label:[c for c in sizes.index if c!=label] for label in sizes.index})
            for name,labels in schemes.items():
                row=p[p.label.isin(labels)][jm].mean().to_dict()
                row.update(dataset=task,repetition=rep,seed=seed,model=model,weighting=name,design='joint')
                own_splits.append(row)
            labelweights=sizes.reindex(p.label).to_numpy(float)
            weighted.append(dict(dataset=task,repetition=rep,seed=seed,model=model,design='joint',weighting='manifest_source_frequency',value=np.average(p.policy_BA_difference,weights=labelweights)))
        micro=z.groupby(['repetition','seed','model','allocation_pair','arm'],as_index=False).policy_BA_difference.mean().groupby(['repetition','seed','model'],as_index=False).policy_BA_difference.mean()
        for r in micro.itertuples():weighted.append(dict(dataset=task,repetition=r.repetition,seed=r.seed,model=r.model,design='joint',weighting='actual_target_frequency_accuracy',value=r.policy_BA_difference))
    write(pd.DataFrame(supports),'source_target_support.csv')
    write(pd.DataFrame(frequencies),'unique_target_frequency.csv')
    write(pd.DataFrame(overlap),'cross_split_overlap.csv')
    write(pd.concat(own_classes),'single_per_class_split_recall_headroom.csv')
    write(pd.DataFrame(own_counts),'single_event_counts_not_independent_samples.csv')
    write(pd.concat(joint_classes),'joint_per_class_split_recall_headroom.csv')
    write(pd.concat(joint_counts),'joint_repeated_context_event_counts.csv')
    write(pd.concat(confusion),'joint_repeated_context_confusion_records.csv')
    splits=pd.DataFrame(own_splits)
    write(splits,'all_common_loco_split_results.csv')
    keys=['dataset','design','model','weighting']
    summary=summarize(splits,keys,'policy_BA_difference')
    for name in ('sham_BA','exposure_BA','corrected_relative_to_sham','harmed_relative_to_sham','unchanged_correctness','sham_error_headroom'):
        summary=summary.merge(splits.groupby(keys)[name].mean().reset_index(),on=keys,validate='one_to_one')
    # Ratio of identically weighted aggregate quantities, not the mean of ratios.
    summary['correction_fraction_of_available_sham_errors']=summary.corrected_relative_to_sham/summary.sham_error_headroom
    summary['net_fraction_of_available_sham_errors']=summary['mean']/summary.sham_error_headroom
    write(summary,'all_common_loco_summary.csv')
    mc=summary[summary.weighting.eq('all_original_classes')][['dataset','design','model','weighting','mean','split_sd','split_count']].copy()
    mc['metric']='policy_BA_difference'
    mc['mean_pp']=100*mc['mean'];mc['split_sd_pp']=100*mc['split_sd']
    mc['mcse_pp']=mc.split_sd_pp/np.sqrt(mc.split_count)
    mc=mc[['dataset','design','model','metric','weighting','split_count','mean_pp','split_sd_pp','mcse_pp']]
    write(mc,'conditional_allocation_mcse.csv')
    display_cols={'sham_error_headroom':'sham_error_headroom_percent','corrected_relative_to_sham':'corrected_pp','harmed_relative_to_sham':'harmed_pp','mean':'net_contrast_pp','correction_fraction_of_available_sham_errors':'corrected_over_sham_error_percent','net_fraction_of_available_sham_errors':'net_contrast_over_sham_error_percent'}
    hd=summary[(summary.weighting=='all_original_classes')&summary.model.isin(['resnet_logit','dinov2_logit'])][['dataset','design','model']+list(display_cols)].copy()
    hd[list(display_cols)]*=100
    write(hd.rename(columns=display_cols),'headroom_interpretation_percentage_units.csv')
    table4=hd[hd.design.eq('joint')].rename(columns=display_cols).copy()
    table4=table4[['dataset','model','sham_error_headroom_percent','corrected_pp','harmed_pp','net_contrast_pp','corrected_over_sham_error_percent']]
    table4['dataset']=pd.Categorical(table4.dataset,categories=['onion','potato'],ordered=True)
    table4['model']=pd.Categorical(table4.model,categories=['resnet_logit','dinov2_logit'],ordered=True)
    table4=table4.sort_values(['dataset','model'])
    table4['dataset']=table4.dataset.astype(str).map({'onion':'Onion','potato':'Potato'})
    table4['model']=table4.model.astype(str).map({'resnet_logit':'ResNet18 + LR','dinov2_logit':'DINOv2 + LR'})
    for col in table4.columns[2:]:table4[col]=table4[col].map(lambda v:f'{v:.2f}')
    table4.to_csv(D19/'manuscript_revision/Table_4_display.csv',index=False)
    weighted=pd.DataFrame(weighted)
    write(weighted,'frequency_weighted_split_results.csv')
    write(summarize(weighted,keys,'value'),'frequency_weighted_summary.csv')
    original=read(D18/'figures/Fig_2_Joint_Replacement_plot_source.csv')
    chk=summary[(summary.design=='joint')&summary.weighting.isin(['all_original_classes','common_classes_n_ge_30'])].merge(original[original.panel=='selected_probe'],on=['dataset','model','weighting'])
    assert len(chk)==20 and np.max(np.abs(chk['mean']*100-chk.mean_pp))<1e-10
    ctx=read(D18/'context_sensitivity/target_contrasts.csv')
    assert len(ctx)==10200 and ctx.n_joint_pairs.eq(5).all()
    table3=read(D18/'manuscript_revision/Table_3_display.csv')
    assert len(table3)==10 and table3.groupby('task').size().eq(5).all()
    for p,before in INPUTS.items():assert sha(p)==before,p
    ledger=dict(status='PASS_READ_ONLY_RECONSTRUCTION',inputs=[dict(path=p,sha256=h) for p,h in INPUTS.items()],original_inputs_unchanged=True,new_fits=0,p_values=0,population_confidence_intervals=0,joint_figure_max_pp_difference=float(np.max(np.abs(chk['mean']*100-chk.mean_pp))),table3_rows=10,same_target_context_pairs=10200,scope='Descriptive recalculation from frozen outputs. LOCO and frequency weighting do not change any fit. Event counts contain recurring units and training contexts, and are not independent biological sample sizes.')
    (OUT/'READ_ONLY_SUMMARY_QA.json').write_text(json.dumps(ledger,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=ledger['status'],joint_figure_max_pp_difference=ledger['joint_figure_max_pp_difference'],unique_targets=pd.DataFrame(frequencies).groupby('dataset').size().to_dict()),ensure_ascii=False))

if __name__=='__main__':main()
