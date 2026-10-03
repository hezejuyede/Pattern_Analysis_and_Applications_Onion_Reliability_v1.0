"""Independently compare every acquired image with author Git blob identifiers."""
from pathlib import Path
from datetime import datetime,timezone
from urllib.parse import quote
import csv,hashlib,json,requests
OUT=Path(__file__).resolve().parent/'potato'
REV='7f7ecc7e1eaca78107e3affe7cb5abd9427e139a'
def main():
 assert not (OUT/'AUTHOR_GIT_BYTE_VERIFICATION.json').exists()
 rows=list(csv.DictReader((OUT/'downloaded_manifest.csv').open(encoding='utf-8')))
 expected={};responses=[]
 for label in sorted({r['label'] for r in rows}):
  u='https://api.github.com/repos/spMohanty/PlantVillage-Dataset/contents/'+quote('raw/color/'+label,safe='/')+'?ref='+REV
  response=requests.get(u,timeout=90);response.raise_for_status();values=response.json();assert isinstance(values,list)
  expected.update({v['path']:v for v in values});responses.append({'url':u,'response_sha256':hashlib.sha256(response.content).hexdigest(),'listed_files':len(values)})
 checks=[]
 for r in rows:
  b=Path(r['local_path']).read_bytes();actual=hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest();e=expected[r['original_path']]
  checks.append({'sample_id':r['sample_id'],'original_path':r['original_path'],'expected_git_blob':e['sha'],'observed_git_blob':actual,'expected_bytes':e['size'],'observed_bytes':len(b),'pass':actual==e['sha'] and len(b)==e['size']})
 with (OUT/'author_git_blob_checks.csv').open('w',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(checks[0]));w.writeheader();w.writerows(checks)
 summary={'status':'PASS' if all(r['pass'] for r in checks) else 'FAIL','utc':datetime.now(timezone.utc).isoformat(),'revision':REV,'checked_images':len(checks),'failures':sum(not r['pass'] for r in checks),'source_responses':responses,'checked_manifest_sha256':hashlib.sha256((OUT/'downloaded_manifest.csv').read_bytes()).hexdigest(),'checks_sha256':hashlib.sha256((OUT/'author_git_blob_checks.csv').read_bytes()).hexdigest(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
 (OUT/'AUTHOR_GIT_BYTE_VERIFICATION.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary,indent=2))
 assert summary['status']=='PASS'
if __name__=='__main__':main()
