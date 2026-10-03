"""Separate conservative blocking components from common-source view support."""
from pathlib import Path
from collections import defaultdict,Counter
from datetime import datetime,timezone
import csv,json,hashlib

ROOT=Path(__file__).resolve().parents[2]
FIRST=Path('D:/论文/SCI投稿汇总/Onion_Deep_Revision_20261002_v16/onion_member_lineage')
OUT=FIRST/'dependence_graph'
def readcsv(p):return list(csv.DictReader(p.open(encoding='utf-8-sig')))
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def writecsv(p,rows,fields=None):
 with p.open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields or list(rows[0]));w.writeheader();w.writerows(rows)
def main():
 OUT.mkdir(exist_ok=True)
 if (OUT/'COMPLETION.json').exists():raise RuntimeError('Refuse overwrite completed graph')
 paths=[FIRST/f'{s}_uniform_geometry/unified_pair_support.csv' for s in ['stage1','stage2']]
 for s in ['stage1','stage2']:assert (FIRST/f'{s}_uniform_geometry/COMPLETION.json').exists()
 pairs=sum([readcsv(p) for p in paths],[]);rawrows=readcsv(FIRST/'raw_content_identity.csv');augs=readcsv(ROOT/'audit/cold_augmented_lineage/augmented_family_manifest.csv')
 rawunique={};families=defaultdict(list)
 for r in rawrows:rawunique.setdefault(r['raw_content_id'],r)
 for r in augs:families[r['family_id']].append(r)
 freeze={'frozen_utc':datetime.now(timezone.utc).isoformat(),'input_sha256':{str(p):digest(p) for p in paths},'script_sha256':digest(Path(__file__)),'blocking_rule':'Undirected graph joining filename-prefix nodes to raw-content nodes for every any-orientation high match, plus every any-orientation high raw/raw edge; exact raw duplicate contents already share one node. No raw-file-count capacity.','spatial_sensitivity_graph':'same graph using only any-orientation high plus spatial/residual screen; not selected after classifier results','intervention_rule':'A blocking component is not a single photographic parent. A proposed intervention triple needs shared spatial-high support to one raw content ID and can use only one such unit per blocking component; no unreviewed experiment is run here.','cross_label':'preserve all original labels; mark whole blocking components with >1 archive label for quarantine in disease classification; do not majority-vote labels','uncertainty_scope':'finite retrieval evidence; unresolved links do not establish independence; components can conservatively merge genuinely distinct photographs','model_fits':0}
 (OUT/'GRAPH_FREEZE.json').write_text(json.dumps(freeze,indent=2),encoding='utf-8')
 member_high=defaultdict(set);member_spatial=defaultdict(set)
 for r in pairs:
  if r['pair_kind']=='aug_raw':
   if r['any_orientation_high']=='True':member_high[r['aug_row']].add(r['target_content_id'])
   if r['any_orientation_spatial_high']=='True':member_spatial[r['aug_row']].add(r['target_content_id'])
 support=[]
 for a in augs:support.append({'aug_row':a['row_idx'],'family_id':a['family_id'],'label':a['label'],'local_path':a['local_path'],'high_raw_content_ids':json.dumps(sorted(member_high[a['row_idx']])),'spatial_high_raw_content_ids':json.dumps(sorted(member_spatial[a['row_idx']])),'high_source_count':len(member_high[a['row_idx']]),'spatial_source_count':len(member_spatial[a['row_idx']]),'unresolved_for_spatial_source':not bool(member_spatial[a['row_idx']])})
 writecsv(OUT/'member_source_support.csv',support)
 summaries={}
 for variant,flag in [('all_high','any_orientation_high'),('spatial_high','any_orientation_spatial_high')]:
  node_labels={**{'p:'+f:part[0]['label'] for f,part in families.items()},**{'r:'+c:r['label'] for c,r in rawunique.items()}};parent={n:n for n in node_labels}
  def find(x):
   while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
   return x
  def union(a,b):
   aa,bb=find(a),find(b)
   if aa!=bb:parent[max(aa,bb)]=min(aa,bb)
  edges=[]
  for r in pairs:
   if r[flag]!='True':continue
   a='p:'+r['family_id'] if r['pair_kind']=='aug_raw' else 'r:'+r['source_content_id'];b='r:'+r['target_content_id'];union(a,b)
   edges.append({'source_node':a,'target_node':b,'pair_kind':r['pair_kind'],'aug_row':r['aug_row'],'origin':r['origin'],'selected_orientation':r['selected_orientation'],'inliers':r['selected_inliers'],'fraction':r['selected_fraction'],'source_hull':r['selected_source_hull'],'target_hull':r['selected_target_hull'],'cross_label':r['cross_label']})
  comps=defaultdict(list)
  for n in node_labels:comps[find(n)].append(n)
  node_to_component={};component_rows=[];meta={}
  for i,(_,nodes) in enumerate(sorted(comps.items())):
   cid=f'onion_{variant}_{i:04d}';labels=sorted({node_labels[n] for n in nodes});pf=[n[2:] for n in nodes if n.startswith('p:')];rw=[n[2:] for n in nodes if n.startswith('r:')];naug=sum(len(families[f]) for f in pf)
   record={'component_id':cid,'filename_prefix_count':len(pf),'raw_content_count':len(rw),'augmented_member_count':naug,'archive_label_count':len(labels),'archive_labels':json.dumps(labels),'quarantine_cross_label':len(labels)>1,'filename_prefixes':json.dumps(sorted(pf)),'raw_content_ids':json.dumps(sorted(rw))};component_rows.append(record);meta[cid]=record
   for n in nodes:node_to_component[n]=cid
  raw_components=[]
  for c,r in sorted(rawunique.items()):
   cid=node_to_component['r:'+c];m=meta[cid];raw_components.append({'raw_content_id':c,'component_id':cid,'label':r['label'],'local_path':r['local_path'],'original_filename':r['original_filename'],'component_raw_content_count':m['raw_content_count'],'component_filename_prefix_count':m['filename_prefix_count'],'component_archive_label_count':m['archive_label_count'],'quarantine_cross_label':m['quarantine_cross_label']})
  augmented_components=[]
  for a in augs:
   cid=node_to_component['p:'+a['family_id']];m=meta[cid];augmented_components.append({'aug_row':a['row_idx'],'family_id':a['family_id'],'component_id':cid,'label':a['label'],'local_path':a['local_path'],'component_filename_prefix_count':m['filename_prefix_count'],'component_raw_content_count':m['raw_content_count'],'component_augmented_member_count':m['augmented_member_count'],'component_archive_label_count':m['archive_label_count'],'quarantine_cross_label':m['quarantine_cross_label'],'spatial_high_raw_content_ids':json.dumps(sorted(member_spatial[a['row_idx']]))})
  writecsv(OUT/f'{variant}_raw_content_components.csv',raw_components);writecsv(OUT/f'{variant}_augmented_components.csv',augmented_components);writecsv(OUT/f'{variant}_component_summary.csv',component_rows);writecsv(OUT/f'{variant}_evidence_edges.csv',edges)
  summaries[variant]={'components_all_nodes':len(comps),'raw_components':len({r['component_id'] for r in raw_components}),'augmented_components':len({r['component_id'] for r in augmented_components}),'cross_label_components':sum(r['quarantine_cross_label'] for r in component_rows),'augmented_rows_in_cross_label_components':sum(r['quarantine_cross_label'] for r in augmented_components),'raw_contents_in_cross_label_components':sum(r['quarantine_cross_label'] for r in raw_components),'largest_raw_content_count':max(r['raw_content_count'] for r in component_rows),'largest_augmented_member_count':max(r['augmented_member_count'] for r in component_rows),'evidence_edges':len(edges),'has_no_model_outcome_based_selection':True}
 (OUT/'GRAPH_SUMMARY.json').write_text(json.dumps({'status':'COMPLETE_GRAPH_REQUIRES_VISUAL_REVIEW_AND_PROTOCOL_FREEZE','variants':summaries,'labels_modified':0,'model_fits':0,'scope_limits':freeze['uncertainty_scope']},indent=2),encoding='utf-8');(OUT/'COMPLETION.json').write_text(json.dumps({'status':'COMPLETE','files_sha256':{p.name:digest(p) for p in sorted(OUT.iterdir()) if p.is_file()}},indent=2),encoding='utf-8');print(json.dumps(summaries,indent=2))

if __name__=='__main__':main()
