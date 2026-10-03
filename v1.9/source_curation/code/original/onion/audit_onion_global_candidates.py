"""Frozen global retrieval extension; no classifier fitting or model selection."""
from __future__ import annotations
from collections import defaultdict,Counter
from datetime import datetime,timezone
from pathlib import Path
import csv,hashlib,json
import cv2,numpy as np

ROOT=Path(__file__).resolve().parents[2]
OLD=ROOT/'audit/cold_augmented_lineage'
FIRST=Path('D:/论文/SCI投稿汇总/Onion_Deep_Revision_20261002_v16/onion_member_lineage')
OUT=FIRST/'stage2_global'

def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def readcsv(p):return list(csv.DictReader(p.open(encoding='utf-8-sig')))
def writecsv(p,rows,fields=None):
 with p.open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields or list(rows[0]));w.writeheader();w.writerows(rows)

def main():
 OUT.mkdir(exist_ok=True)
 if (OUT/'COMPLETION.json').exists():raise RuntimeError('Refuse overwrite completed global audit')
 raw=readcsv(FIRST/'raw_content_identity.csv');aug=readcsv(OLD/'augmented_family_manifest.csv');previous=readcsv(FIRST/'all_member_candidate_geometry.csv')
 rawunique={}
 for r in raw:rawunique.setdefault(r['raw_content_id'],r)
 raw=[rawunique[k] for k in sorted(rawunique)];rawpos={r['raw_content_id']:i for i,r in enumerate(raw)}
 archive=np.load(OLD/'resnet18_embeddings.npz',allow_pickle=False);positions={str(p).replace('\\','/'):i for i,p in enumerate(archive['paths'])};feat=np.asarray(archive['features'],dtype=np.float64);feat/=np.maximum(np.linalg.norm(feat,axis=1,keepdims=True),1e-12)
 R=feat[[positions[r['local_path']] for r in raw]];A=feat[[positions[r['local_path']] for r in aug]];similarity=A@R.T
 ranks=np.argsort(-similarity,axis=1,kind='stable');reverse_ranks=np.argsort(np.argsort(-similarity,axis=0,kind='stable'),axis=0,kind='stable')+1
 already={(int(r['aug_row']),r['candidate_raw_content_id']) for r in previous};aug_candidates=[]
 for i,a in enumerate(aug):
  for rank,j in enumerate(ranks[i,:5],1):
   r=raw[int(j)]
   if (int(a['row_idx']),r['raw_content_id']) in already:continue
   aug_candidates.append({'aug_index':i,'aug_row':int(a['row_idx']),'family_id':a['family_id'],'aug_label':a['label'],'raw_index':int(j),'raw_content_id':r['raw_content_id'],'raw_label':r['label'],'raw_filename':r['original_filename'],'aug_to_raw_rank':rank,'raw_to_aug_rank':int(reverse_ranks[i,j]),'resnet_cosine':float(similarity[i,j]),'cross_label':a['label']!=r['label']})
 rawsim=R@R.T;np.fill_diagonal(rawsim,-np.inf);raworder=np.argsort(-rawsim,axis=1,kind='stable');rawrank=np.argsort(raworder,axis=1,kind='stable')+1
 pairs=sorted({(min(i,int(j)),max(i,int(j))) for i,row in enumerate(raworder[:,:10]) for j in row});raw_candidates=[]
 for i,j in pairs:
  a,b=raw[i],raw[j];raw_candidates.append({'first_index':i,'second_index':j,'first_raw_content_id':a['raw_content_id'],'second_raw_content_id':b['raw_content_id'],'first_label':a['label'],'second_label':b['label'],'first_filename':a['original_filename'],'second_filename':b['original_filename'],'first_to_second_rank':int(rawrank[i,j]),'second_to_first_rank':int(rawrank[j,i]),'resnet_cosine':float(rawsim[i,j]),'cross_label':a['label']!=b['label']})
 writecsv(OUT/'frozen_augmented_global_candidates.csv',aug_candidates);writecsv(OUT/'frozen_raw_global_candidates.csv',raw_candidates)
 freeze={'frozen_utc':datetime.now(timezone.utc).isoformat(),'stage1_completion_sha256':digest(FIRST/'COMPLETION.json'),'embedding_sha256':digest(OLD/'resnet18_embeddings.npz'),'raw_identity_sha256':digest(FIRST/'raw_content_identity.csv'),'augmented_manifest_sha256':digest(OLD/'augmented_family_manifest.csv'),'script_sha256':digest(Path(__file__)),'candidate_files_sha256':{n:digest(OUT/n) for n in ['frozen_augmented_global_candidates.csv','frozen_raw_global_candidates.csv']},'selection':{'augmented':'per-member cosine top5 over all 813 raw content-unique images, including other labels; compute only not previously checked candidates','raw':'each unique raw cosine top10 other unique raw; undirected union; includes other labels','tie_break':'stable content SHA order'},'pairs':{'augmented':len(aug_candidates),'raw_raw':len(raw_candidates)},'geometry':{'SIFT_nfeatures':1000,'contrastThreshold':.02,'ratio':.75,'RANSAC_pixels':4.,'directions':['normal','horizontal_reflection'],'high_geometric_support':'>=15 inliers and >=.60 inlier fraction','spatially_distributed_high_support':'high geometry plus each inlier convex hull >=.05 of image area, median inlier residual <=2 px and p95 <=4 px; heuristic screen, not certified identity'},'limitations':['finite nearest-neighbor candidates do not establish absence of other related sources','no pathology or plant identity inference','negative SIFT never interpreted as proof of independence'],'model_fits':0}
 (OUT/'AUDIT_FREEZE.json').write_text(json.dumps(freeze,indent=2),encoding='utf-8')
 cv2.setNumThreads(1);cv2.setRNGSeed(20261002);detector=cv2.SIFT_create(nfeatures=1000,contrastThreshold=.02);matcher=cv2.BFMatcher(cv2.NORM_L2);cache={}
 def describe(path,flip):
  key=(str(path),flip)
  if key not in cache:
   im=cv2.imdecode(np.fromfile(path,dtype=np.uint8),cv2.IMREAD_GRAYSCALE)
   if flip:im=cv2.flip(im,1)
   kp,desc=detector.detectAndCompute(im,None);cache[key]=(np.float32([x.pt for x in kp]),desc,im.shape)
  return cache[key]
 def compare(a,b,flip):
  ap,ad,ashape=describe(a,flip);bp,bd,bshape=describe(b,False)
  empty={'inliers':0,'good_matches':0,'fraction':0.,'source_hull_fraction':0.,'target_hull_fraction':0.,'median_inlier_residual':float('nan'),'p95_inlier_residual':float('nan')}
  if ad is None or bd is None or len(ad)<2 or len(bd)<2:return empty
  good=[x for x,y in matcher.knnMatch(ad,bd,k=2) if x.distance<.75*y.distance];empty['good_matches']=len(good)
  if len(good)<4:return empty
  source=np.float32([ap[x.queryIdx] for x in good]);target=np.float32([bp[x.trainIdx] for x in good]);H,mask=cv2.findHomography(source,target,cv2.RANSAC,4.)
  if mask is None or H is None:return empty
  keep=mask.ravel().astype(bool);n=int(keep.sum())
  if n<3:return empty
  projected=cv2.perspectiveTransform(source[keep].reshape(-1,1,2),H).reshape(-1,2);residual=np.linalg.norm(projected-target[keep],axis=1)
  return {'inliers':n,'good_matches':len(good),'fraction':n/max(1,len(good)),'source_hull_fraction':float(cv2.contourArea(cv2.convexHull(source[keep])))/int(np.prod(ashape)),'target_hull_fraction':float(cv2.contourArea(cv2.convexHull(target[keep])))/int(np.prod(bshape)),'median_inlier_residual':float(np.median(residual)),'p95_inlier_residual':float(np.quantile(residual,.95))}
 def geometry(a,b):
  normal=compare(a,b,False);reflected=compare(a,b,True);best,orientation=max([(normal,'normal'),(reflected,'horizontal_reflection')],key=lambda x:(x[0]['inliers'],x[0]['fraction']))
  high=best['inliers']>=15 and best['fraction']>=.6;spatial=high and best['source_hull_fraction']>=.05 and best['target_hull_fraction']>=.05 and best['median_inlier_residual']<=2 and best['p95_inlier_residual']<=4
  return {'normal_inliers':normal['inliers'],'normal_fraction':normal['fraction'],'reflected_inliers':reflected['inliers'],'reflected_fraction':reflected['fraction'],'best_orientation':orientation,**best,'positive_high_geometry':high,'positive_spatially_distributed_high_geometry':spatial}
 aug_scores=[];last_aug=None
 for ix,c in enumerate(aug_candidates):
  a=aug[c['aug_index']];r=raw[c['raw_index']];ap=ROOT/a['local_path'];rp=ROOT/r['local_path']
  if last_aug is not None and last_aug!=str(ap):cache.pop((last_aug,False),None);cache.pop((last_aug,True),None)
  last_aug=str(ap);aug_scores.append({**c,**geometry(ap,rp)})
  if (ix+1)%2500==0:print(json.dumps({'stage2_aug_pairs':ix+1,'total':len(aug_candidates)}),flush=True)
 writecsv(OUT/'augmented_global_candidate_geometry.csv',aug_scores)
 raw_scores=[]
 for ix,c in enumerate(raw_candidates):
  a=raw[c['first_index']];b=raw[c['second_index']];raw_scores.append({**c,**geometry(ROOT/a['local_path'],ROOT/b['local_path'])})
  if (ix+1)%1000==0:print(json.dumps({'stage2_raw_pairs':ix+1,'total':len(raw_candidates)}),flush=True)
 writecsv(OUT/'raw_global_candidate_geometry.csv',raw_scores)
 summary={'status':'COMPLETE_GLOBAL_CANDIDATE_GEOMETRY','completed_utc':datetime.now(timezone.utc).isoformat(),'augmented_new_pairs':len(aug_scores),'augmented_cross_label_tested':sum(r['cross_label'] for r in aug_scores),'augmented_high_support':sum(r['positive_high_geometry'] for r in aug_scores),'augmented_cross_label_high_support':sum(r['cross_label'] and r['positive_high_geometry'] for r in aug_scores),'augmented_spatial_high_support':sum(r['positive_spatially_distributed_high_geometry'] for r in aug_scores),'augmented_cross_label_spatial_high_support':sum(r['cross_label'] and r['positive_spatially_distributed_high_geometry'] for r in aug_scores),'raw_pairs':len(raw_scores),'raw_cross_label_tested':sum(r['cross_label'] for r in raw_scores),'raw_high_support':sum(r['positive_high_geometry'] for r in raw_scores),'raw_cross_label_high_support':sum(r['cross_label'] and r['positive_high_geometry'] for r in raw_scores),'raw_spatial_high_support':sum(r['positive_spatially_distributed_high_geometry'] for r in raw_scores),'raw_cross_label_spatial_high_support':sum(r['cross_label'] and r['positive_spatially_distributed_high_geometry'] for r in raw_scores),'limits':freeze['limitations'],'model_fits':0}
 (OUT/'AUDIT_SUMMARY.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');(OUT/'COMPLETION.json').write_text(json.dumps({'status':'COMPLETE','files_sha256':{p.name:digest(p) for p in sorted(OUT.iterdir()) if p.is_file()}},indent=2),encoding='utf-8');print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
