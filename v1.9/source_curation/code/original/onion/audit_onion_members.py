"""Per-member geometry and content-based family components, without model fits."""
from __future__ import annotations
from collections import defaultdict,Counter
from datetime import datetime,timezone
from pathlib import Path
import csv,hashlib,io,json,re
import cv2,numpy as np,pyarrow.parquet as pq
from PIL import Image

ROOT=Path(__file__).resolve().parents[2]
OLD=ROOT/'audit/cold_augmented_lineage'
OUT=Path('D:/论文/SCI投稿汇总/Onion_Deep_Revision_20261002_v16/onion_member_lineage')

def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()

def readcsv(path):return list(csv.DictReader(path.open(encoding='utf-8-sig')))
def writecsv(path,rows,fields=None):
 with path.open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields or list(rows[0]));w.writeheader();w.writerows(rows)

def main():
 OUT.mkdir(parents=True,exist_ok=True)
 if (OUT/'COMPLETION.json').exists():raise RuntimeError('Refuse overwriting completed member audit')
 sources={name:OLD/name for name in ['augmented_family_manifest.csv','raw_parquet_manifest.csv','top_candidate_scores.csv','family_to_raw_mapping.csv']}
 aug=readcsv(sources['augmented_family_manifest.csv']);raw={int(r['row_idx']):r for r in readcsv(sources['raw_parquet_manifest.csv'])};cand=defaultdict(list)
 for r in readcsv(sources['top_candidate_scores.csv']):cand[r['family_id']].append(r)
 freeze={'frozen_utc':datetime.now(timezone.utc).isoformat(),'input_sha256':{n:digest(p) for n,p in sources.items()},'script_sha256':digest(Path(__file__)),'purpose':'Re-audit every augmented member against all five previously retained raw candidates; original analysis remains untouched','geometry':{'SIFT_nfeatures':1000,'contrastThreshold':0.02,'Lowe_ratio':0.75,'homography_RANSAC_pixels':4.0,'orientations':['original','horizontal_reflection'],'deterministic_cv_seed':20261002},'high_support_rule':'best >=15 inliers and >=.60 fraction, with inlier margin >=max(8, ceil(.25*best)) over every other content-unique retained raw candidate','moderate_support_rule':'best >=8 inliers and >=.50 fraction, with margin >=4','content_identity':'SHA256 of original parquet decoded RGB shape and pixels; duplicate raw file entries count as one identity','scope_limits':['candidate search is limited to the five previously retained raw candidates','positive geometry supports correspondence, not verified plant identity or pathology','multiple raw matches can mean raw near duplicates, not necessarily mixed-source augmentation'],'model_fits':0}
 (OUT/'AUDIT_FREEZE.json').write_text(json.dumps(freeze,indent=2),encoding='utf-8')
 cv2.setNumThreads(1);cv2.setRNGSeed(20261002)
 rawbytes=pq.read_table(ROOT/'raw/COLD_raw.parquet').to_pylist();contentrows=[];content_to_rows=defaultdict(list)
 for i,r in raw.items():
  b=rawbytes[i]['image']['bytes']
  with Image.open(io.BytesIO(b)) as im:rgb=np.asarray(im.convert('RGB'))
  content=hashlib.sha256(str(rgb.shape).encode()+b'\0'+rgb.tobytes()).hexdigest();r['raw_content_id']=content;content_to_rows[content].append(i)
  contentrows.append({'raw_row':i,'raw_content_id':content,'parquet_sha256':hashlib.sha256(b).hexdigest(),'local_sha256':r['sha256'],'label':r['label'],'original_filename':r['original_filename'],'local_path':r['local_path']})
 writecsv(OUT/'raw_content_identity.csv',contentrows)
 detector=cv2.SIFT_create(nfeatures=1000,contrastThreshold=0.02);matcher=cv2.BFMatcher(cv2.NORM_L2);cache={}
 def describe(path,flip=False):
  key=(str(path),flip)
  if key not in cache:
   im=cv2.imdecode(np.fromfile(path,dtype=np.uint8),cv2.IMREAD_GRAYSCALE)
   if flip:im=cv2.flip(im,1)
   kp,desc=detector.detectAndCompute(im,None);cache[key]=(np.float32([x.pt for x in kp]),desc)
  return cache[key]
 def match(a,b,flip):
  ap,ad=describe(a,flip);bp,bd=describe(b)
  if ad is None or bd is None or len(ad)<2 or len(bd)<2:return (0,0,0.,0.,0.)
  pairs=matcher.knnMatch(ad,bd,k=2);good=[x for x,y in pairs if x.distance<.75*y.distance]
  if len(good)<4:return (0,len(good),0.,0.,0.)
  s=np.float32([ap[x.queryIdx] for x in good]);t=np.float32([bp[x.trainIdx] for x in good]);_,mask=cv2.findHomography(s,t,cv2.RANSAC,4.)
  if mask is None:return (0,len(good),0.,0.,0.)
  m=mask.ravel().astype(bool);n=int(m.sum());sa=float(cv2.contourArea(cv2.convexHull(s[m])))/(256*256) if n>=3 else 0.;ta=float(cv2.contourArea(cv2.convexHull(t[m])))/(256*256) if n>=3 else 0.
  return (n,len(good),n/max(1,len(good)),sa,ta)
 scores=[];members=[]
 for ix,a in enumerate(aug):
  family=a['family_id'];candidates={}
  for c in cand[family]:
   rowids=[int(x.strip()) for x in c['candidate_raw_rows'].split('|')]
   for rid in rowids:
    content=raw[rid]['raw_content_id'];candidates.setdefault(content,{'raw_row':rid,'geometric_rank':int(c['geometric_rank']),'raw_cluster':c['candidate_raw_cluster']})
  current=[]
  for content,c in candidates.items():
   rp=ROOT/raw[c['raw_row']]['local_path'];ap=ROOT/a['local_path'];normal=match(ap,rp,False);reflected=match(ap,rp,True)
   best=max([(normal,'normal'),(reflected,'horizontal_reflection')],key=lambda v:(v[0][0],v[0][2]));v,orientation=best
   record={'aug_row':int(a['row_idx']),'family_id':family,'label':a['label'],'aug_path':a['local_path'],'candidate_raw_content_id':content,'candidate_raw_rows':' | '.join(map(str,content_to_rows[content])),'candidate_raw_names':' | '.join(raw[i]['original_filename'] for i in content_to_rows[content]),'original_candidate_rank':c['geometric_rank'],'normal_inliers':normal[0],'normal_good_matches':normal[1],'normal_fraction':normal[2],'reflected_inliers':reflected[0],'reflected_good_matches':reflected[1],'reflected_fraction':reflected[2],'best_orientation':orientation,'best_inliers':v[0],'best_good_matches':v[1],'best_fraction':v[2],'source_inlier_hull_fraction':v[3],'target_inlier_hull_fraction':v[4],'positive_high_geometry':v[0]>=15 and v[2]>=.6,'positive_moderate_geometry':v[0]>=8 and v[2]>=.5}
   scores.append(record);current.append(record)
  current.sort(key=lambda x:(-x['best_inliers'],-x['best_fraction'],x['candidate_raw_content_id']));best=current[0];runner=current[1]['best_inliers'] if len(current)>1 else 0;gap=best['best_inliers']-runner
  if best['positive_high_geometry'] and gap>=max(8,int(np.ceil(.25*best['best_inliers']))):tier='high_unique_within_retained_candidates'
  elif best['positive_moderate_geometry'] and gap>=4:tier='moderate_unique_within_retained_candidates'
  elif any(c['positive_moderate_geometry'] for c in current):tier='multiple_or_weakly_separated_candidates'
  else:tier='unresolved'
  members.append({'aug_row':int(a['row_idx']),'family_id':family,'label':a['label'],'aug_path':a['local_path'],'best_raw_content_id':best['candidate_raw_content_id'],'best_raw_names':best['candidate_raw_names'],'best_inliers':best['best_inliers'],'best_fraction':best['best_fraction'],'runner_up_inliers':runner,'inlier_margin':gap,'confidence':tier,'positive_high_raw_content_count':sum(c['positive_high_geometry'] for c in current),'positive_moderate_raw_content_count':sum(c['positive_moderate_geometry'] for c in current),'candidate_content_count':len(current)})
  # Release descriptors for augmented images; raw descriptors remain cached.
  cache.pop((str(ROOT/a['local_path']),False),None);cache.pop((str(ROOT/a['local_path']),True),None)
  if (ix+1)%500==0:print(json.dumps({'members_checked':ix+1,'total':len(aug)}),flush=True)
 writecsv(OUT/'all_member_candidate_geometry.csv',scores);writecsv(OUT/'per_member_evidence.csv',members)
 byfamily=defaultdict(list)
 for r in members:byfamily[r['family_id']].append(r)
 families=[]
 for f,part in sorted(byfamily.items()):
  supported=[r for r in part if r['confidence'].startswith(('high_unique','moderate_unique'))];ids=sorted({r['best_raw_content_id'] for r in supported});high=[r for r in supported if r['confidence'].startswith('high_unique')]
  families.append({'family_id':f,'label':part[0]['label'],'members':len(part),'high_unique_members':len(high),'moderate_unique_members':len(supported)-len(high),'ambiguous_members':sum(r['confidence']=='multiple_or_weakly_separated_candidates' for r in part),'unresolved_members':sum(r['confidence']=='unresolved' for r in part),'supported_raw_content_count':len(ids),'supported_raw_content_ids':' | '.join(ids),'all_members_unique_same_content':len(supported)==len(part) and len(ids)==1,'interpretation':'candidate-restricted geometry; no biological identity claim'})
 writecsv(OUT/'per_family_evidence.csv',families)
 # Conservative dependence graph uses ALL positive high geometry, not file-count capacity.
 parent={f:f for f in byfamily}
 def find(x):
  while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
  return x
 def union(a,b):
  aa,bb=find(a),find(b)
  if aa!=bb:parent[max(aa,bb)]=min(aa,bb)
 raw_to_families=defaultdict(set)
 for r in scores:
  if r['positive_high_geometry']:raw_to_families[r['candidate_raw_content_id']].add(r['family_id'])
 edges=[]
 for content,fs in sorted(raw_to_families.items()):
  fs=sorted(fs)
  if len(fs)>1:
   for f in fs[1:]:union(fs[0],f)
   edges.append({'raw_content_id':content,'raw_names':' | '.join(raw[i]['original_filename'] for i in content_to_rows[content]),'prefix_count':len(fs),'family_ids':' | '.join(fs),'label_count':len({byfamily[f][0]['label'] for f in fs})})
 components=defaultdict(list)
 for f in byfamily:components[find(f)].append(f)
 comp_rows=[]
 for index,(representative,fs) in enumerate(sorted(components.items())):
  labels=sorted({byfamily[f][0]['label'] for f in fs});n=sum(len(byfamily[f]) for f in fs)
  for f in sorted(fs):comp_rows.append({'family_id':f,'component_id':f'onion_content_component_{index:04d}','component_prefix_count':len(fs),'component_augmented_members':n,'component_label_count':len(labels),'component_labels':' | '.join(labels),'quarantine_cross_label':len(labels)>1,'evidence_rule':'union prefixes sharing any >=15-inlier >=.60-fraction raw-content match; hypothesis-conservative grouping'})
 writecsv(OUT/'shared_raw_content_prefixes.csv',edges,['raw_content_id','raw_names','prefix_count','family_ids','label_count']);writecsv(OUT/'conservative_component_membership.csv',comp_rows)
 summary={'status':'COMPLETE_CANDIDATE_LIMITED_GEOMETRY_AUDIT','completed_utc':datetime.now(timezone.utc).isoformat(),'augmented_members':len(aug),'candidate_comparisons':len(scores),'raw_rows':len(raw),'raw_content_unique':len(content_to_rows),'member_confidence_counts':dict(Counter(r['confidence'] for r in members)),'filename_groups':len(families),'all_members_supported_same_content_families':sum(r['all_members_unique_same_content'] for r in families),'families_with_multiple_uniquely_supported_raw_contents':sum(r['supported_raw_content_count']>1 for r in families),'shared_raw_content_prefix_cases':len(edges),'conservative_components':len(components),'multi_prefix_components':sum(len(fs)>1 for fs in components.values()),'cross_label_components':len({r['component_id'] for r in comp_rows if r['quarantine_cross_label']}),'scientific_interpretation':'The old filename prefix partition does not prove independent source content. Component union addresses observed raw-content collisions, but unresolved/candidate-limited lineage remains. Any new evaluation must retain these limits and cannot relabel split units as plants.','input_hashes':freeze['input_sha256'],'model_fits':0}
 (OUT/'AUDIT_SUMMARY.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
 manifest={p.name:digest(p) for p in sorted(OUT.iterdir()) if p.is_file()}
 (OUT/'COMPLETION.json').write_text(json.dumps({'status':'COMPLETE','files_sha256':manifest},indent=2),encoding='utf-8');print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
