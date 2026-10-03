"""Exact-content audit and fixed features for independent archival Potato task."""
from pathlib import Path
from collections import defaultdict,Counter
from datetime import datetime,timezone
import json,csv,hashlib,platform
import numpy as np,cv2,torch,torchvision
from PIL import Image,ImageOps,ImageDraw
from torchvision.models import ResNet18_Weights,resnet18

BASE=Path(__file__).resolve().parent; OUT=BASE/'potato'; F=OUT/'features'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return list(csv.DictReader(p.open(encoding='utf-8-sig')))
def write(p,rows):
 with p.open('w',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def color_feature(path):
 im=cv2.imdecode(np.fromfile(path,dtype=np.uint8),cv2.IMREAD_COLOR)
 if im is None:raise RuntimeError(path)
 im=cv2.resize(im,(128,128),interpolation=cv2.INTER_AREA)
 hsv=cv2.cvtColor(im,cv2.COLOR_BGR2HSV);lab=cv2.cvtColor(im,cv2.COLOR_BGR2LAB);parts=[]
 for arr,ranges in [(im,((0,256),)*3),(hsv,((0,180),(0,256),(0,256))),(lab,((0,256),)*3)]:
  for ch,limits in enumerate(ranges):
   hist=cv2.calcHist([arr],[ch],None,[16],list(limits)).ravel();hist/=max(hist.sum(),1.)
   values=arr[:,:,ch].astype(np.float32).ravel()
   parts.extend([hist,np.array([values.mean(),values.std(),*np.quantile(values,[.1,.5,.9])])])
 result=np.concatenate(parts).astype(np.float32);assert result.shape==(189,);return result
def main():
 assert (OUT/'ACQUISITION_COMPLETE.json').exists()
 assert not (OUT/'PREPARATION_COMPLETE.json').exists()
 F.mkdir(exist_ok=True)
 rows=read(OUT/'downloaded_manifest.csv');groups=defaultdict(list);pixel=defaultdict(list)
 for r in rows:groups[r['leaf_id']].append(r);pixel[r['pixel_sha256']].append(r)
 parent={x:x for x in groups}
 def find(a):
  while parent[a]!=a:parent[a]=parent[parent[a]];a=parent[a]
  return a
 def union(a,b):
  x,y=find(a),find(b)
  if x!=y:parent[max(x,y)]=min(x,y)
 for part in pixel.values():
  for r in part[1:]:union(part[0]['leaf_id'],r['leaf_id'])
 comps=defaultdict(list)
 for leaf in groups:comps[find(leaf)].append(leaf)
 selected=[];decisions=[]
 for component,leaves in sorted(comps.items()):
  labels={r['label'] for leaf in leaves for r in groups[leaf]}
  reference=min(leaves,key=lambda x:(-len({r['pixel_sha256'] for r in groups[x]}),x))
  seen=set();keep=[]
  for r in sorted(groups[reference],key=lambda r:r['sample_id']):
   if r['pixel_sha256'] not in seen:keep.append(r);seen.add(r['pixel_sha256'])
  admitted=len(labels)==1 and len(keep)>=3
  decisions.append({'component_id':component,'author_leaf_ids':json.dumps(leaves),'n_leaves':len(leaves),'n_labels':len(labels),'selected_leaf_id':reference,'n_unique_photographs':len(keep),'admitted':admitted})
  if admitted:
   for r in keep:
    selected.append({'row_idx':str(int(r['sample_id'].split('_')[-1])),'label':'healthy' if r['label']=='Potato___healthy' else r['label'],'local_path':r['local_path'],'sha256':r['sha256'],'pixel_sha256':r['pixel_sha256'],'component_id':'pv_author_leaf::'+component,'intervention_source_id':reference,'original_path':r['original_path'],'original_image_identifier':r['original_image_identifier'],'official_split':r['official_split'],'author_leaf_id':r['leaf_id'],'source_type':'original_color_photograph_author_leaf_metadata'})
 selected=sorted(selected,key=lambda r:int(r['row_idx']));write(OUT/'analysis_manifest.csv',selected);write(OUT/'selection_decisions.csv',decisions)
 counts=Counter(r['label'] for r in {r['component_id']:r for r in selected}.values())
 assert len(counts)==3 and min(counts.values())>=12 and sum(counts.values())>=100
 assert len({r['sha256'] for r in selected})==len({r['pixel_sha256'] for r in selected})==len(selected)
 mapbytes=(BASE/'plantvillage_metadata/leaf_grouping/leaf-map.json').read_bytes();mapgit=hashlib.sha1(b'blob '+str(len(mapbytes)).encode()+b'\0'+mapbytes).hexdigest()
 assert mapgit=='cb04e3723d2ce15b0411483d71a60b5d536bc7f6'
 ckpt=Path(torch.hub.get_dir())/'checkpoints/resnet18-f37072fd.pth';assert ckpt.exists()
 freeze={'utc':datetime.now(timezone.utc).isoformat(),'script_sha256':sha(Path(__file__)),'manifest_sha256':sha(OUT/'analysis_manifest.csv'),'download_manifest_sha256':sha(OUT/'downloaded_manifest.csv'),'author_leaf_map_git_blob':mapgit,'author_leaf_map_sha256':hashlib.sha256(mapbytes).hexdigest(),'leaf_map_initial_author_commit':'40789680ba2e6382608dba47b397688fcbd0d04a','class_components':dict(counts),'images':len(selected),'pixel_duplicate_rows':sum(len(x)-1 for x in pixel.values()),'cross_leaf_pixel_components':sum(len(x)>1 for x in comps.values()),'resnet_weights_sha256':sha(ckpt),'resnet_config':'ResNet18_Weights.IMAGENET1K_V1, weights.transforms(), avgpool 512, eval, no augmentation or fine-tuning','color_config':'Original onion BGR/HSV/LAB each 3 channels x (16-bin normalized histogram + mean/std/q10/q50/q90) =189','model_fits':0,'runtime':{'python':platform.python_version(),'torch':torch.__version__,'torchvision':torchvision.__version__,'opencv':cv2.__version__,'numpy':np.__version__}}
 (OUT/'FEATURE_AND_SELECTION_FREEZE.json').write_text(json.dumps(freeze,indent=2),encoding='utf-8')
 torch.set_num_threads(4);cv2.setNumThreads(1)
 weights=ResNet18_Weights.IMAGENET1K_V1;transform=weights.transforms();base=resnet18(weights=weights);model=torch.nn.Sequential(*list(base.children())[:-1],torch.nn.Flatten(1)).eval()
 features=[];colors=[]
 with torch.inference_mode():
  for start in range(0,len(selected),64):
   part=selected[start:start+64];batch=[]
   for r in part:
    with Image.open(r['local_path']) as im:batch.append(transform(im.convert('RGB')))
    colors.append(color_feature(r['local_path']))
   features.append(model(torch.stack(batch)).numpy().astype(np.float32))
   if start%256==0:print(json.dumps({'features_done':min(start+64,len(selected)),'total':len(selected)}),flush=True)
 paths=np.array([r['local_path'] for r in selected],dtype=str)
 np.savez_compressed(F/'resnet18.npz',features=np.concatenate(features),paths=paths)
 np.savez_compressed(F/'colour189.npz',features=np.stack(colors),paths=paths)
 chosen=[]
 for label in sorted(counts):
  ids=sorted({r['author_leaf_id'] for r in selected if r['label']==label})
  for i in [0,1,len(ids)//2,len(ids)//2+1,len(ids)-2,len(ids)-1]:chosen.append(ids[i])
 contacts=[]
 for pg in range(2):
  leafids=chosen[pg*9:(pg+1)*9];canvas=Image.new('RGB',(1120,9*255),'white');draw=ImageDraw.Draw(canvas)
  for rr,leaf in enumerate(leafids):
   part=[r for r in selected if r['author_leaf_id']==leaf]
   for cc,r in enumerate(part[:4]):
    with Image.open(r['local_path']) as im:canvas.paste(ImageOps.contain(im.convert('RGB'),(240,220)),(cc*280,rr*255+25))
    draw.text((cc*280+3,rr*255+3),leaf.replace('Potato___','')+' / '+r['row_idx'],fill='black')
    contacts.append({'page':pg+1,'leaf_id':leaf,'row_idx':r['row_idx'],'path':r['local_path']})
  canvas.save(OUT/f'author_leaf_contacts_{pg+1}.png')
 write(OUT/'contact_sheet_manifest.csv',contacts)
 summary={'status':'PREPARED_REAL_ARCHIVAL_LEAF_TASK_FOR_INDEPENDENT_REVIEW','components':sum(counts.values()),'images':len(selected),'class_components':dict(counts),'all_selected_leaf_photo_counts':dict(Counter(Counter(r['author_leaf_id'] for r in selected).values())),'feature_shapes':{'resnet18':list(np.concatenate(features).shape),'colour189':list(np.stack(colors).shape)},'hashes':{str(p.relative_to(OUT)):sha(p) for p in [OUT/'analysis_manifest.csv',OUT/'selection_decisions.csv',OUT/'FEATURE_AND_SELECTION_FREEZE.json',F/'resnet18.npz',F/'colour189.npz']},'source_provenance':'2016 authors leaf mapping blob matches downloaded HF mapping exactly; original paper states repeated photos of the same leaves','limits':['Independent archive and author-labelled leaf groups, not a new field cohort or verified plant grouping.','No new diagnostic/pathology review.','Selected original-color Potato subset; no claim about all PlantVillage species.'],'classifier_fits':0}
 (OUT/'PREPARATION_COMPLETE.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':main()
