"""Small, pre-training geometric check of reused chilli filename indices."""
from pathlib import Path
from collections import defaultdict
import csv,json,hashlib
import cv2,numpy as np
BASE=Path('D:/论文/SCI投稿汇总/Onion_Deep_Revision_20261002_v16/data')
OUT=BASE/'audit'

def main():
 rows=list(csv.DictReader((OUT/'all_image_manifest.csv').open(encoding='utf-8-sig')))
 families=['chilli/healthy/0','chilli/healthy/26','chilli/cercospora/0','chilli/mites_trips/0','chilli/nutritional/0','chilli/powdery mildew/0']
 detector=cv2.SIFT_create(nfeatures=1000,contrastThreshold=0.02)
 matcher=cv2.BFMatcher(cv2.NORM_L2);cache={}
 def description(row):
  if row['sample_id'] not in cache:
   im=cv2.imdecode(np.fromfile(BASE/row['path'],dtype=np.uint8),cv2.IMREAD_GRAYSCALE)
   kp,desc=detector.detectAndCompute(im,None);cache[row['sample_id']]=(np.float32([k.pt for k in kp]),desc)
  return cache[row['sample_id']]
 def verify(first,second):
  ak,ad=description(first);bk,bd=description(second)
  if ad is None or bd is None or len(ad)<2 or len(bd)<2:return 0,0,0.
  pairs=matcher.knnMatch(ad,bd,k=2);good=[a for a,b in pairs if a.distance<0.75*b.distance]
  if len(good)<4:return 0,len(good),0.
  source=np.float32([ak[x.queryIdx] for x in good]);target=np.float32([bk[x.trainIdx] for x in good]);_,mask=cv2.findHomography(source,target,cv2.RANSAC,4.)
  n=int(mask.sum()) if mask is not None else 0
  return n,len(good),n/max(1,len(good))
 results=[]
 for family in families:
  part=[r for r in rows if r['family_id']==family]
  selected=[part[i] for i in np.linspace(0,len(part)-1,7,dtype=int)]
  for i,a in enumerate(selected):
   for b in selected[i+1:]:
    n,m,f=verify(a,b);results.append({'family_id':family,'first':a['sample_id'],'second':b['sample_id'],'sift_inliers':n,'good_matches':m,'inlier_fraction':f,'positive_geometric_support':n>=15 and f>=.6})
 with (OUT/'candidate_family_geometry_spotcheck.csv').open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(results[0]));w.writeheader();w.writerows(results)
 report={'status':'FAIL_FILENAME_INDEX_AS_CONFIRMED_SINGLE_PARENT','scope':'42 deterministically spaced images in six illustrative groups, 126 pairs; not exhaustive lineage recovery','visual_review':'Six by seven contact sheet inspected. Healthy 0, cercospora 0, nutritional 0 and powdery mildew 0 contain visibly distinct source photographs under one candidate index. Healthy 26 appears one source. Mites 0 inconclusive from inspection alone.','support_rule':'SIFT >=15 homography inliers with >=0.60 inlier fraction is positive transformation correspondence; a negative match is not proof of distinct source','positive_pair_counts':{f:sum(r['family_id']==f and r['positive_geometric_support'] for r in results) for f in families},'prohibition':'Do not use candidate prefix/index as a confirmed source family without further lineage work. No image/biological diagnosis is inferred from visual inspection.','model_fits':0,'contact_sheet_sha256':hashlib.sha256((OUT/'candidate_family_contact_sheet.png').read_bytes()).hexdigest(),'geometry_csv_sha256':hashlib.sha256((OUT/'candidate_family_geometry_spotcheck.csv').read_bytes()).hexdigest()}
 (OUT/'FILENAME_FAMILY_ADMISSIBILITY.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
