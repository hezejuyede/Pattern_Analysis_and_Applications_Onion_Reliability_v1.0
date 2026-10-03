"""Append full two-orientation geometry for every previously strong candidate."""
from __future__ import annotations
from pathlib import Path
from datetime import datetime,timezone
import argparse,csv,hashlib,json
import cv2,numpy as np

ROOT=Path(__file__).resolve().parents[2]
FIRST=Path('D:/论文/SCI投稿汇总/Onion_Deep_Revision_20261002_v16/onion_member_lineage')
def readcsv(p):return list(csv.DictReader(p.open(encoding='utf-8-sig')))
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def writecsv(p,rows):
 with p.open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['stage1','stage2'],required=True);args=parser.parse_args();out=FIRST/(args.stage+'_uniform_geometry');out.mkdir(exist_ok=True)
 if (out/'COMPLETION.json').exists():raise RuntimeError('Refuse replacing uniform geometry results')
 raw={}
 for r in readcsv(FIRST/'raw_content_identity.csv'):raw.setdefault(r['raw_content_id'],r)
 aug={r['row_idx']:r for r in readcsv(ROOT/'audit/cold_augmented_lineage/augmented_family_manifest.csv')}
 inputpaths=[FIRST/'all_member_candidate_geometry.csv'] if args.stage=='stage1' else [FIRST/'stage2_global/augmented_global_candidate_geometry.csv',FIRST/'stage2_global/raw_global_candidate_geometry.csv']
 assert (FIRST/'COMPLETION.json').exists()
 if args.stage=='stage2':assert (FIRST/'stage2_global/COMPLETION.json').exists()
 pairs=[];old_argmax_false=0
 for path in inputpaths:
  for r in readcsv(path):
   anyhigh=any(float(r[k+'_inliers'])>=15 and float(r[k+'_fraction'])>=.6 for k in ['normal','reflected'])
   if not anyhigh:continue
   old_argmax_false+=r['positive_high_geometry']!='True'
   if 'aug_row' in r:
    a=aug[r['aug_row']];content=r.get('candidate_raw_content_id',r.get('raw_content_id'));b=raw[content]
    pairs.append({'pair_kind':'aug_raw','aug_row':r['aug_row'],'family_id':a['family_id'],'source_content_id':'','target_content_id':content,'source_path':a['local_path'],'target_path':b['local_path'],'source_label':a['label'],'target_label':b['label'],'source_filename':a['original_filename'],'target_filename':b['original_filename'],'origin':args.stage,'old_best_high':r['positive_high_geometry']})
   else:
    a=raw[r['first_raw_content_id']];b=raw[r['second_raw_content_id']]
    pairs.append({'pair_kind':'raw_raw','aug_row':'','family_id':'','source_content_id':a['raw_content_id'],'target_content_id':b['raw_content_id'],'source_path':a['local_path'],'target_path':b['local_path'],'source_label':a['label'],'target_label':b['label'],'source_filename':a['original_filename'],'target_filename':b['original_filename'],'origin':args.stage,'old_best_high':r['positive_high_geometry']})
 writecsv(out/'frozen_pairs.csv',pairs)
 freeze={'frozen_utc':datetime.now(timezone.utc).isoformat(),'input_sha256':{str(p):digest(p) for p in inputpaths},'script_sha256':digest(Path(__file__)),'pair_file_sha256':digest(out/'frozen_pairs.csv'),'pairs':len(pairs),'old_argmax_hid_any_high_pairs':old_argmax_false,'selection':'either recorded orientation has >=15 inliers and fraction >=.60, irrespective of old chosen-orientation flag or its coverage','support_rule':'any orientation qualifies; representative orientation chosen within qualifying set, not before thresholding','high':{'inliers_min':15,'fraction_min':.60},'spatial_high':{'source_hull_fraction_min':.05,'target_hull_fraction_min':.05,'median_residual_max_px':2.,'p95_residual_max_px':4.},'raw_identity':'original parquet decoded RGB content hash; exact duplicates counted once','model_fits':0,'scope':'append-only geometry eligibility screen; neither proof of independence nor plant/pathology identity'}
 (out/'AUDIT_FREEZE.json').write_text(json.dumps(freeze,indent=2),encoding='utf-8');cv2.setNumThreads(1);cv2.setRNGSeed(20261002);sift=cv2.SIFT_create(nfeatures=1000,contrastThreshold=.02);matcher=cv2.BFMatcher(cv2.NORM_L2);cache={}
 def desc(path,flip):
  k=(path,flip)
  if k not in cache:
   im=cv2.imdecode(np.fromfile(ROOT/path,dtype=np.uint8),cv2.IMREAD_GRAYSCALE)
   if flip:im=cv2.flip(im,1)
   kp,d=sift.detectAndCompute(im,None);cache[k]=(np.float32([x.pt for x in kp]),d,im.shape)
  return cache[k]
 def geometry(a,b,flip):
  x,xd,xshape=desc(a,flip);y,yd,yshape=desc(b,False);empty={'inliers':0,'good_matches':0,'fraction':0.,'source_hull_fraction':0.,'target_hull_fraction':0.,'median_inlier_residual':float('nan'),'p95_inlier_residual':float('nan'),'positive_high':False,'positive_spatial_high':False}
  if xd is None or yd is None or min(len(xd),len(yd))<2:return empty
  good=[u for u,v in matcher.knnMatch(xd,yd,k=2) if u.distance<.75*v.distance];empty['good_matches']=len(good)
  if len(good)<4:return empty
  source=np.float32([x[k.queryIdx] for k in good]);target=np.float32([y[k.trainIdx] for k in good]);H,m=cv2.findHomography(source,target,cv2.RANSAC,4.)
  if m is None or H is None:return empty
  keep=m.ravel().astype(bool);n=int(keep.sum())
  if n<3:return empty
  errors=np.linalg.norm(cv2.perspectiveTransform(source[keep].reshape(-1,1,2),H).reshape(-1,2)-target[keep],axis=1)
  fraction=n/len(good);sx=float(cv2.contourArea(cv2.convexHull(source[keep])))/int(np.prod(xshape));sy=float(cv2.contourArea(cv2.convexHull(target[keep])))/int(np.prod(yshape));median=float(np.median(errors));p95=float(np.quantile(errors,.95));high=n>=15 and fraction>=.60;spatial=high and min(sx,sy)>=.05 and median<=2 and p95<=4
  return {'inliers':n,'good_matches':len(good),'fraction':fraction,'source_hull_fraction':sx,'target_hull_fraction':sy,'median_inlier_residual':median,'p95_inlier_residual':p95,'positive_high':high,'positive_spatial_high':spatial}
 orientations=[];unified=[];last=None
 for i,p in enumerate(pairs):
  if last is not None and last!=p['source_path'] and last.startswith('data/cold_augmented'):
   cache.pop((last,False),None);cache.pop((last,True),None)
  last=p['source_path'];options=[]
  for flip in [False,True]:
   result=geometry(p['source_path'],p['target_path'],flip);r={**p,'orientation':'horizontal_reflection' if flip else 'normal',**result};orientations.append(r);options.append(r)
  spatial=[r for r in options if r['positive_spatial_high']];high=[r for r in options if r['positive_high']];chosen=max(spatial or high or options,key=lambda r:(r['inliers'],r['fraction']))
  unified.append({**p,'any_orientation_high':bool(high),'any_orientation_spatial_high':bool(spatial),'selected_orientation':chosen['orientation'],'selected_inliers':chosen['inliers'],'selected_fraction':chosen['fraction'],'selected_source_hull':chosen['source_hull_fraction'],'selected_target_hull':chosen['target_hull_fraction'],'selected_median_residual':chosen['median_inlier_residual'],'selected_p95_residual':chosen['p95_inlier_residual'],'cross_label':p['source_label']!=p['target_label']})
  if (i+1)%1000==0:print(json.dumps({'uniform_stage':args.stage,'pairs':i+1,'total':len(pairs)}),flush=True)
 writecsv(out/'all_orientation_geometry.csv',orientations);writecsv(out/'unified_pair_support.csv',unified)
 summary={'status':'COMPLETE_UNIFORM_TWO_ORIENTATION_GEOMETRY','pairs':len(unified),'any_high':sum(r['any_orientation_high'] for r in unified),'any_spatial_high':sum(r['any_orientation_spatial_high'] for r in unified),'cross_label_any_high':sum(r['cross_label'] and r['any_orientation_high'] for r in unified),'cross_label_any_spatial_high':sum(r['cross_label'] and r['any_orientation_spatial_high'] for r in unified),'old_argmax_hid_any_high_pairs':old_argmax_false,'previous_any_high_but_recheck_neither':sum(not r['any_orientation_high'] for r in unified),'model_fits':0}
 (out/'AUDIT_SUMMARY.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');(out/'COMPLETION.json').write_text(json.dumps({'status':'COMPLETE','files_sha256':{p.name:digest(p) for p in sorted(out.iterdir()) if p.is_file()}},indent=2),encoding='utf-8');print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
