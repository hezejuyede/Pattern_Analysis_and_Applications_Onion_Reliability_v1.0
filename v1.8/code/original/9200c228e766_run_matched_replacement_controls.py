"""Matched training-replacement policies; post hoc mechanism experiment.

Preparation writes the complete assignment before fitting. Execution requires
the unchanged prepared hashes. No arm is selected using observed outcomes.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse
import json
import time
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import run_corrected_component_experiment as base


def plan_one(frame, labels, seed, rep):
    rng = np.random.default_rng(seed)
    groups = {c: g.index.to_numpy() for c, g in frame.groupby('component_id', sort=True)}
    cl = frame.groupby('component_id', sort=True).label.first()
    records, pairs = {}, []
    for label in labels:
        ids = cl.index[cl.eq(label)].to_numpy(dtype=str)
        order = rng.permutation(ids)
        np_, ns = round(.2*len(ids)), round(.1*len(ids))
        probes, sentinels, donors = order[:np_], order[np_:np_+ns], order[np_+ns:]
        pairorder = rng.permutation(probes)
        paired = pairorder[:2*(np_//2)]
        base.require(np_ >= 2 and ns >= 1 and len(donors) >= 2*len(paired), 'Insufficient class support')
        remove, alternate = donors[:len(paired)], donors[len(paired):2*len(paired)]
        maps = {p: (r, a) for p, r, a in zip(paired, remove, alternate)}
        for k in range(0, len(paired), 2):
            pairs.append((paired[k], paired[k+1]))
        for cid in ids:
            role = ('probe' if cid in probes else 'sentinel' if cid in sentinels else
                    'remove' if cid in remove else 'alternate' if cid in alternate else 'core')
            selected = rng.choice(groups[cid], 3 if role == 'probe' else 1 if role == 'sentinel' else 2, replace=False)
            anchor = int(selected[0]) if role in ('probe', 'sentinel') else -1
            views = selected[1:] if role == 'probe' else np.array([], dtype=int) if role == 'sentinel' else selected
            records[cid] = dict(repetition=rep, seed=seed, component_id=cid, label=label,
                source_id=frame.iloc[groups[cid][0]].intervention_source_id,
                role=role, paired=int(cid in maps), remove=maps.get(cid, ('',''))[0],
                alternate=maps.get(cid, ('',''))[1], anchor_position=anchor,
                view1_position=int(views[0]) if len(views) else -1,
                view2_position=int(views[1]) if len(views) else -1)
    test = np.array(sorted(r['anchor_position'] for r in records.values() if r['anchor_position'] >= 0))
    testcid = frame.iloc[test].component_id.to_numpy()
    train0 = {c for c, r in records.items() if r['role'] in ('core', 'remove')}
    selected_sets = {}
    conditions = {'zero': dict(policy='zero', pair=-1, arm='zero', selected=set(), train_components=train0)}
    for pairno in range(5):
        coin = rng.integers(0,2,len(pairs))
        for arm, offsets in [('A', coin), ('B', 1-coin)]:
            selected = {pair[int(j)] for pair,j in zip(pairs,offsets)}
            selected_sets[(pairno,arm)] = selected
            removed = {records[c]['remove'] for c in selected}
            for policy in ('exposure','sham'):
                inserted = selected if policy == 'exposure' else {records[c]['alternate'] for c in selected}
                conditions[f'{policy}_{pairno}_{arm}'] = dict(policy=policy, pair=pairno, arm=arm,
                    selected=selected, train_components=(train0-removed)|inserted)
    checks, memberships = [], []
    for name,c in conditions.items():
        tc = c['train_components']
        train = np.array(sorted(j for cid in tc for j in (records[cid]['view1_position'],records[cid]['view2_position'])))
        c['train'] = train
        expected = c['selected'] if c['policy']=='exposure' else set()
        overlap = tc & set(testcid)
        counts = frame.iloc[train].label.value_counts().to_dict()
        expectedcounts = {l: 2*sum(records[cid]['label']==l for cid in train0) for l in labels}
        sourceoverlap = set(frame.iloc[train].intervention_source_id)&set(frame.iloc[test].intervention_source_id)
        expectedsource = {records[cid]['source_id'] for cid in expected}
        base.require(len(train)==2*len(train0) and len(set(train))==len(train), 'Training size changed')
        base.require(counts==expectedcounts and overlap==expected and sourceoverlap==expectedsource,'Support or source mismatch')
        base.require(not set(frame.iloc[train].sha256)&set(frame.iloc[test].sha256),'Exact train/test overlap')
        checks.append(dict(repetition=rep,seed=seed,condition=name,policy=c['policy'],training_components=len(tc),
            training_files=len(train),test_files=len(test),selected_probe_count=len(c['selected']),
            exposed_source_count=len(sourceoverlap),sentinel_overlap=0,class_counts=json.dumps(counts,sort_keys=True),status='PASS'))
        memberships.extend(dict(repetition=rep,condition=name,component_id=cid,in_training=int(cid in tc),selected_probe=int(cid in c['selected'])) for cid in records)
    for k in range(5):
        for arm in ('A','B'):
            e,h = conditions[f'exposure_{k}_{arm}'],conditions[f'sham_{k}_{arm}']
            base.require(train0-e['train_components']==train0-h['train_components'],'Different removed donors')
            base.require(len(e['train_components']-train0)==len(h['train_components']-train0),'Different replacement count')
    return dict(records=records,test=test,testcid=testcid,conditions=conditions,checks=checks,memberships=memberships)


def analyse_pair(truth, predicted0, pe, ph, mask):
    y=truth[mask]; zero=predicted0[mask]; e=pe[mask]; h=ph[mask]
    cm=lambda a: base.class_mean(np.asarray(a,dtype=float),y)
    out={'policy_BA_difference':cm((e==y).astype(int)-(h==y).astype(int)),
         'policy_label_disagreement':cm(e!=h)}
    for name,p in [('exposure',e),('sham',h)]:
        gain=(zero!=y)&(p==y); loss=(zero==y)&(p!=y)
        ww=(zero!=y)&(p!=y)&(zero!=p)
        out.update({name+'_gain':cm(gain),name+'_loss':cm(loss),name+'_wrong_to_wrong':cm(ww),
            name+'_turnover':cm(gain|loss),name+'_label_churn':cm(p!=zero),
            name+'_BA_delta':cm((p==y).astype(int)-(zero==y).astype(int))})
        base.require(abs(out[name+'_BA_delta']-out[name+'_gain']+out[name+'_loss'])<1e-12,'Transition identity failed')
    out['turnover_difference']=out['exposure_turnover']-out['sham_turnover']
    out['churn_difference']=out['exposure_label_churn']-out['sham_label_churn']
    return out


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest',required=True,type=Path)
    ap.add_argument('--resnet-features',required=True,type=Path)
    ap.add_argument('--colour-features',required=True,type=Path)
    ap.add_argument('--output',required=True,type=Path)
    ap.add_argument('--mode',choices=['prepare','run'],required=True)
    args=ap.parse_args()
    args.healthy_label='healthy'
    frame,matrices,labels,has_source,info=base.load_inputs(args)
    base.require(has_source,'Missing common reference identifiers')
    plans=[plan_one(frame,labels,20261003+i,i) for i in range(30)]
    out=args.output
    paths={'manifest':args.manifest,'resnet':args.resnet_features,'colour':args.colour_features,
           'code':Path(__file__),'shared_numerics':Path(base.__file__)}
    hashes={k:{'path':str(v.resolve()),'sha256':base.sha(v)} for k,v in paths.items()}
    if args.mode=='prepare':
        base.require(not out.exists(),'Output must be new')
        out.mkdir(parents=True)
        frame.to_csv(out/'manifest.csv',index=False)
        pd.DataFrame([r for p in plans for r in p['records'].values()]).to_csv(out/'component_membership.csv',index=False)
        pd.DataFrame([r for p in plans for r in p['checks']]).to_csv(out/'support_checks.csv',index=False)
        with base.csv_writer(out/'allocation_membership.csv.gz',list(plans[0]['memberships'][0])) as writer:
            for p in plans: writer.writerows(p['memberships'])
        base.dump(out/'protocol_freeze.json',{'frozen_utc':datetime.now(timezone.utc).isoformat(),'inputs':hashes,
            'status':'PREPARED_BEFORE_NEW_FITS','chronology':'Post hoc after all v16 outcomes; not independent replication or preregistration.',
            'primary_probe':'Class-balanced correctness difference exposure minus matched sham on selected paired probes; average A/B then 5 pairs within split.',
            'primary_sentinel':'Class-balanced correctness turnover relative to same zero baseline, exposure minus sham.',
            'secondary':'Unselected probes, all paired probes, per-class results; label churn, gains/losses, wrong-to-wrong; all models and endpoints.',
            'policy_interpretation':'Same removed donors and class counts; replacement sibling vs unrelated same-class source. Other sources change; not pure direct causal effect.',
            'class_counts':frame.groupby('component_id').label.first().value_counts().to_dict(),
            'training_budget':'240 components and 480 files on onion; smaller than v16 336 and 672. Do not attribute differences only to control policy.',
            'randomization':'30 seeds starting 20261003, 5 complementary pairs; odd probe remains unselected.',
            'uncertainty':'30 split means and percentile variation, no independent-plant CIs or significance tests.',
            'selection':'No feature-distance or outcome matching, exclusions, tuning, reranking, or stopping on results.',
            'model_protocol':'Frozen ResNet18 and colour standardized balanced L2 logistic C=.01; untuned cosine 1NN; multiclass primary, binary secondary.',
            'prepared_hashes':{p.name:base.sha(p) for p in sorted(out.iterdir()) if p.is_file()}})
        print('Prepared 30 splits, 630 conditions; no model fitted.',flush=True)
        return
    freeze=json.loads((out/'protocol_freeze.json').read_text(encoding='utf-8'))
    base.require(hashes==freeze['inputs'],'Code or input changed after freeze')
    for n,h in freeze['prepared_hashes'].items(): base.require(base.sha(out/n)==h,'Prepared assignment changed')
    base.require(not (out/'predictions.csv.gz').exists(),'Cannot overwrite an executed run')
    ys={'multiclass':frame.label.map(dict(zip(labels,range(len(labels))))).to_numpy(),
        'binary_healthy_other':frame.label.ne('healthy').to_numpy(dtype=int)}
    xd=matrices['resnet_logit']; norm=np.linalg.norm(xd,axis=1)
    base.require((norm>0).all(),'Zero cosine norm')
    cosine=xd/norm[:,None]
    fields=['repetition','seed','model','endpoint','condition','panel','component_id','row_idx','label','truth','predicted','selected','nearest_train_row_idx']+[f'p{i}' for i in range(len(labels))]
    rows,metrics,fits,warnings=[],[],[],[]
    t=time.perf_counter(); n_predictions=0
    with base.csv_writer(out/'predictions.csv.gz',fields) as writer,threadpool_limits(limits=1):
        for rep,plan in enumerate(plans):
            seed=20261003+rep; test=plan['test']; records=plan['records']; tc=plan['testcid']
            panel=np.array([records[c]['role'] for c in tc]); paired=np.array([bool(records[c]['paired']) for c in tc])
            for model in ['resnet_logit','colour_logit','resnet_cosine_1nn']:
                x=matrices['colour_logit' if model=='colour_logit' else 'resnet_logit']
                for endpoint,y in ys.items():
                    key=dict(repetition=rep,seed=seed,model=model,endpoint=endpoint); prob={}
                    for condition,c in plan['conditions'].items():
                        p,nearest,caught,n_iter=base.predict(x,y,c['train'],test,seed,model,cosine)
                        prob[condition]=p
                        fits.append({**key,'condition':condition,'n_train':len(c['train']),'n_iter':n_iter,'warnings':len(caught)})
                        warnings.extend({**key,'condition':condition,**w} for w in caught)
                        for j,pos in enumerate(test):
                            item=frame.iloc[pos]
                            writer.writerow({**key,'condition':condition,'panel':panel[j],'component_id':item.component_id,
                                'row_idx':item.row_idx,'label':item.label,'truth':int(y[pos]),'predicted':int(p[j].argmax()),
                                'selected':int(item.component_id in c['selected']),
                                'nearest_train_row_idx':'' if nearest is None else frame.iloc[nearest[j]].row_idx,
                                **{f'p{k}':float(p[j,k]) for k in range(p.shape[1])}})
                            n_predictions+=1
                        for name,mask in [('all_probe',panel=='probe'),('sentinel',panel=='sentinel')]:
                            metrics.append({**key,'condition':condition,'panel':name,**base.metric_values(y[test][mask],p[mask])})
                    truth=y[test]; q0=prob['zero'].argmax(1)
                    for k in range(5):
                        for arm in ('A','B'):
                            selected=np.array([c in plan['conditions'][f'exposure_{k}_{arm}']['selected'] for c in tc])
                            e=prob[f'exposure_{k}_{arm}'].argmax(1); h=prob[f'sham_{k}_{arm}'].argmax(1)
                            panels={'selected_probe':selected,'unselected_paired_probe':paired&~selected,'all_paired_probe':paired,'sentinel':panel=='sentinel'}
                            for name,mask in panels.items():
                                for stratum in ['all_classes']+labels:
                                    m=mask if stratum=='all_classes' else mask & frame.iloc[test].label.eq(stratum).to_numpy()
                                    base.require(m.any(),'Empty required panel')
                                    rows.append({**key,'allocation_pair':k,'arm':arm,'panel':name,'stratum':stratum,'n':int(m.sum()),**analyse_pair(truth,q0,e,h,m)})
            print(f'Finished matched replacement split {rep+1}/30 in {time.perf_counter()-t:.1f}s',flush=True)
    r=pd.DataFrame(rows); r.to_csv(out/'arm_contrasts.csv',index=False)
    values=[c for c in r if c not in ['repetition','seed','model','endpoint','allocation_pair','arm','panel','stratum','n']]
    seedmeans=r.groupby(['repetition','seed','model','endpoint','panel','stratum'],as_index=False)[values].mean()
    seedmeans.to_csv(out/'split_means.csv',index=False)
    base.summarize(seedmeans,['model','endpoint','panel','stratum'],values).to_csv(out/'split_summary.csv',index=False)
    pd.DataFrame(metrics).to_csv(out/'condition_metrics.csv',index=False)
    pd.DataFrame(fits).to_csv(out/'fit_log.csv',index=False)
    base.dump(out/'warnings.json',{'count':len(warnings),'items':warnings})
    for item in hashes.values():base.require(base.sha(item['path'])==item['sha256'],'Input/code mutated during run')
    base.dump(out/'completion.json',dict(status='COMPLETED_POST_HOC_MATCHED_CONTROL',fits=len(fits),logistic_fits=sum(r['model']!='resnet_cosine_1nn' for r in fits),
        predictions=n_predictions,warnings=len(warnings),elapsed_seconds=time.perf_counter()-t,protocol_sha256=base.sha(out/'protocol_freeze.json'),scientific_submission_gate='HOLD_UNTIL_REVIEW'))
    pd.DataFrame([dict(path=p.name,bytes=p.stat().st_size,sha256=base.sha(p)) for p in sorted(out.iterdir()) if p.is_file()]).to_csv(out/'artifact_manifest.csv',index=False)


if __name__=='__main__':main()
