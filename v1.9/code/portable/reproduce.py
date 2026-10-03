"""Portable release CLI; root may be any path, with no original images required."""
import argparse,os,sys,json,hashlib
from pathlib import Path
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2])
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--dataset',choices=['all','onion','potato'],default='all')
    parser.add_argument('--experiment',choices=['all','v17','v18_joint','v18_single'],default='all')
    parser.add_argument('--refit',choices=['none','smoke','full-audit'],default='smoke')
    args=parser.parse_args();root=args.root.resolve();out=args.out.resolve()
    os.environ['ONION_RELEASE_ROOT']=str(root);os.environ['ONION_REBUILD_OUT']=str(out);os.environ['ONION_REFIT']=args.refit
    import audit_joint_results as joint
    import audit_single_results as single
    datasets=['onion','potato'] if args.dataset=='all' else [args.dataset]
    experiments=['v17','v18_joint','v18_single'] if args.experiment=='all' else [args.experiment]
    for dataset in datasets:
        for experiment in experiments:
            if experiment=='v18_single':single.audit(dataset)
            else:joint.audit(dataset,experiment)
    from build_primary_tables import build
    if args.dataset=='all' and args.experiment=='all':build(out,root)
    reports=[]
    for dataset in datasets:
        for experiment in experiments:
            path=out/f'{experiment}_{dataset}'/'RECONSTRUCTION_REPORT.json'
            reports.append(json.loads(path.read_text(encoding='utf-8')))
    record={'status':'PASS_PORTABLE_RECONSTRUCTION','root':str(root),'output':str(out),'refit_mode':args.refit,
        'experiments':len(reports),'prediction_rows':sum(r['predictions_checked'] for r in reports),
        'logistic_refits':sum(r['logistic_refits'] for r in reports),'nearest_neighbour_conditions':sum(r['independent_nn_conditions'] for r in reports),
        'max_refit_probability_difference':max((r['maximum_refit_probability_difference'] for r in reports),default=0),
        'scientific_interpretation':'Reproduction quality only. Not evidence of journal acceptance, prospective registration, independent field cohorts or diagnostic confirmation.'}
    out.mkdir(parents=True,exist_ok=True);(out/'PORTABLE_RECONSTRUCTION_REPORT.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(record,indent=2),flush=True)
if __name__=='__main__':main()
