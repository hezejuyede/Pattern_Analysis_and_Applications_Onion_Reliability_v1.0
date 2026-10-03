"""Acquire the preselected complete three-class original-color Potato task."""
from pathlib import Path
from collections import defaultdict,Counter
from datetime import datetime,timezone
from concurrent.futures import ThreadPoolExecutor,as_completed
import csv,json,hashlib,time,io
import requests,numpy as np
from PIL import Image

BASE=Path(__file__).resolve().parent; META=BASE/'plantvillage_metadata'; OUT=BASE/'potato'
REV='7f7ecc7e1eaca78107e3affe7cb5abd9427e139a'
URL='https://raw.githubusercontent.com/spMohanty/PlantVillage-Dataset/'+REV+'/'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return list(csv.DictReader(p.open(encoding='utf-8-sig')))
def write(p,rows):
 with p.open('w',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
 OUT.mkdir(exist_ok=True);(OUT/'images').mkdir(exist_ok=True)
 assert not (OUT/'ACQUISITION_COMPLETE.json').exists()
 groups=read(META/'leaf_group_metadata.csv')
 eligible={r['leaf_id'] for r in groups if r['crop']=='Potato' and r['eligible_three_photographs']=='True'}
 rows=[r for r in read(META/'color_metadata.csv') if r['leaf_id'] in eligible and r['metadata_eligible']=='True']
 rows=sorted(rows,key=lambda r:(r['leaf_id'],r['original_image_identifier'],r['original_path']))
 identifiers=set();selected=[]
 for r in rows:
  key=(r['leaf_id'],r['original_image_identifier'])
  if key not in identifiers:selected.append(r);identifiers.add(key)
 for i,r in enumerate(selected):r['sample_id']=f'pv_potato_{i:05d}'
 write(OUT/'SELECTED_BEFORE_DOWNLOAD.csv',selected)
 freeze={'frozen_utc':datetime.now(timezone.utc).isoformat(),'github_revision':REV,'hf_leaf_map_revision':'9e97599868962bd0079b8db4b7f1efa9185fa1e7','selection':'All three-class Potato original-color records with unique class-compatible author leaf mapping and >=3 distinct original photo identifiers, excluding filename copy variants. No subsampling by model output.','selection_justification':'Compact complete crop task with two disease categories and healthy controls; independent collection project from COLD.','selected_leaves':len(eligible),'selected_images':len(selected),'class_leaf_counts':dict(Counter(r['label'] for r in groups if r['leaf_id'] in eligible)),'class_image_counts':dict(Counter(r['label'] for r in selected)),'same_identifier_deduplication':'stable original-path lexical first; variants do not become independent photographs','post_download_rule':'Verify decoded image; remove duplicate pixel rows stably, union leaves sharing exact RGB content for isolation, exclude cross-label connected components; require >=3 pixel-distinct photographs per chosen leaf. No arbitrary geometric assignment used as leaf identity.','analysis_admissibility':'all three classes >=12 nonconflict isolation components and >=100 total; no performance-based fallback','features':'frozen torchvision ResNet18 ImageNet1K_V1 512-D, weights.transforms; 189-D fixed color histogram/moments identical to onion. No training or augmentation of image feature extractor.','analysis':'fixed-budget source-exposure and matched donor-replacement control; root protocol and independently checked runner required before classifier fitting','max_total_download_bytes':100000000,'metadata_sha256':{n:sha(META/n) for n in ['color_metadata.csv','leaf_group_metadata.csv','leaf_grouping/leaf-map.json']},'selection_sha256':sha(OUT/'SELECTED_BEFORE_DOWNLOAD.csv'),'script_sha256':sha(Path(__file__)),'model_results_inspected':False,'scope_limits':['author-reported leaf identity, not our newly observed plant ID','independent archival task, not newly collected field cohort','laboratory backgrounds and author diagnostic labels, no unified pathology adjudication']}
 (OUT/'ACQUISITION_FREEZE.json').write_text(json.dumps(freeze,indent=2),encoding='utf-8')
 def get(r):
  p=OUT/'images'/(r['sample_id']+'.jpg');u=URL+r['original_path']
  if not p.exists():
   for attempt in range(4):
    try:
     response=requests.get(u,timeout=60);response.raise_for_status();data=response.content
     if len(data)>1000000:raise RuntimeError('Unexpected image >1MB; stop for review')
     with Image.open(io.BytesIO(data)) as im:im.verify()
     p.write_bytes(data);break
    except Exception:
     if attempt==3:raise
     time.sleep(2*(attempt+1))
  data=p.read_bytes()
  with Image.open(io.BytesIO(data)) as im:rgb=np.asarray(im.convert('RGB'));width,height=im.size
  return {**r,'local_path':p.as_posix(),'sha256':hashlib.sha256(data).hexdigest(),'pixel_sha256':hashlib.sha256(str(rgb.shape).encode()+b'\0'+rgb.tobytes()).hexdigest(),'bytes':len(data),'width':width,'height':height,'source_url':u}
 fetched=[];failures=[];total=0
 with ThreadPoolExecutor(max_workers=8) as ex:
  futures={ex.submit(get,r):r for r in selected}
  for f in as_completed(futures):
   try:
    r=f.result();fetched.append(r);total+=r['bytes']
    if total>freeze['max_total_download_bytes']:raise RuntimeError('Download budget exceeded')
   except Exception as e:failures.append({'sample_id':futures[f]['sample_id'],'error':repr(e)})
   if (len(fetched)+len(failures))%100==0:print(json.dumps({'downloaded':len(fetched),'errors':len(failures),'total':len(selected),'bytes':total}),flush=True)
 fetched=sorted(fetched,key=lambda r:r['sample_id']);write(OUT/'downloaded_manifest.csv',fetched)
 (OUT/'download_errors.json').write_text(json.dumps(failures,indent=2),encoding='utf-8')
 assert not failures,failures[:3]
 assert len(fetched)==len(selected)
 result={'status':'COMPLETE_ORIGINAL_IMAGE_DOWNLOAD_NOT_ANALYSIS_APPROVAL','utc':datetime.now(timezone.utc).isoformat(),'images':len(fetched),'bytes':total,'unique_rgb_images':len({r['pixel_sha256'] for r in fetched}),'source_revision':REV,'manifest_sha256':sha(OUT/'downloaded_manifest.csv'),'freeze_sha256':sha(OUT/'ACQUISITION_FREEZE.json'),'image_sizes':dict(Counter(f"{r['width']}x{r['height']}" for r in fetched)),'model_fits':0}
 (OUT/'ACQUISITION_COMPLETE.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
