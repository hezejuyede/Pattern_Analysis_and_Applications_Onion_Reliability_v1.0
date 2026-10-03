"""Read original PlantVillage leaf metadata before any image or model outcome."""
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict, Counter
import requests, json, hashlib, csv

BASE=Path(__file__).resolve().parent
OUT=BASE/'plantvillage_metadata'
REV='9e97599868962bd0079b8db4b7f1efa9185fa1e7'
REPO='https://huggingface.co/datasets/mohanty/PlantVillage/resolve/'+REV+'/'
FILES=['README.md','plant_village.py','leaf_grouping/leaf-map.json','splits/color_train.txt','splits/color_test.txt']
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 OUT.mkdir(exist_ok=True)
 assert not (OUT/'METADATA_SUMMARY.json').exists()
 freeze={'utc':datetime.now(timezone.utc).isoformat(),'revision':REV,'files':FILES,'script_sha256':sha(Path(__file__)),'selection_stage':'metadata only, no image classifier output inspected','image_mode':'raw/color only; grayscale/segmentation never independent views','eligibility':'exactly one class-compatible official leaf mapping; exclude fallback, copy derivatives, and any cross-class group; distinct original image identifiers >=3','scope':'independent pre-existing archive; leaf metadata provenance requires review; not newly collected plants, no new pathology adjudication'}
 (OUT/'METADATA_AUDIT_FREEZE.json').write_text(json.dumps(freeze,indent=2),encoding='utf-8')
 for name in FILES:
  path=OUT/name;path.parent.mkdir(parents=True,exist_ok=True)
  if not path.exists():
   r=requests.get(REPO+name,timeout=90);r.raise_for_status();path.write_bytes(r.content)
 mapping=json.loads((OUT/'leaf_grouping/leaf-map.json').read_text(encoding='utf-8'))
 rows=[]
 for split in ['train','test']:
  for path in (OUT/f'splits/color_{split}.txt').read_text().splitlines():
   path=path.strip()
   if not path: continue
   parts=path.split('/');label=parts[2];name=parts[3]
   identifier=name.replace('_final_masked','').split('___')[-1].split('copy')[0]
   for ext in ['.jpg','.JPG','.png','.PNG']:identifier=identifier.replace(ext,'')
   key=identifier.lower().strip();suggestions=mapping.get(key,[])
   compatible=[s for s in suggestions if s.split(':::')[0]==label]
   leaf=compatible[0] if len(compatible)==1 else ''
   copied='copy' in name.lower()
   rows.append({'original_path':path,'label':label,'crop':label.split('___')[0],'original_filename':name,'original_image_identifier':key,'official_split':split,'leaf_id':leaf,'n_mapping_candidates':len(suggestions),'n_class_compatible_candidates':len(compatible),'copy_derivative':copied,'metadata_eligible':bool(leaf) and not copied})
 groups=defaultdict(list)
 for row in rows:
  if row['metadata_eligible']:groups[row['leaf_id']].append(row)
 group_rows=[]
 for leaf,part in sorted(groups.items()):
  labels=sorted({r['label'] for r in part});n=len({r['original_image_identifier'] for r in part})
  group_rows.append({'leaf_id':leaf,'label':labels[0],'crop':part[0]['crop'],'n_paths':len(part),'n_distinct_original_identifiers':n,'n_labels':len(labels),'official_split_count':len({r['official_split'] for r in part}),'eligible_three_photographs':n>=3 and len(labels)==1})
 for name,part in [('color_metadata.csv',rows),('leaf_group_metadata.csv',group_rows)]:
  with (OUT/name).open('w',encoding='utf-8',newline='') as f:
   w=csv.DictWriter(f,fieldnames=list(part[0]));w.writeheader();w.writerows(part)
 counts=Counter(r['label'] for r in group_rows if r['eligible_three_photographs'])
 summary={'status':'METADATA_COUNTS_ONLY_IMAGE_IDENTITY_AND_MAP_PROVENANCE_PENDING','revision':REV,'color_rows':len(rows),'color_paths_unique':len({r['original_path'] for r in rows}),'eligible_mapped_rows':sum(r['metadata_eligible'] for r in rows),'mapped_leaf_groups':len(groups),'leaf_groups_with_three_distinct_original_identifiers':sum(counts.values()),'eligible_leaf_groups_by_label':dict(sorted(counts.items())),'official_split_crossing_mapped_leaf_groups':sum(r['official_split_count']>1 for r in group_rows),'unmapped_or_ambiguous_rows':sum(not r['leaf_id'] for r in rows),'copy_rows_excluded':sum(r['copy_derivative'] for r in rows),'files_sha256':{str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file()},'model_fits':0,'new_image_downloads':0}
 (OUT/'METADATA_SUMMARY.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
