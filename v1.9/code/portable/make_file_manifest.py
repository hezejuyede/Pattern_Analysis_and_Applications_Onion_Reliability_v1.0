"""Generate the final distribution inventory after every release contributor is done."""
from pathlib import Path
import argparse,csv,hashlib,json
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2]);args=parser.parse_args();root=args.root.resolve()
    rows=[]
    for p in sorted(root.rglob('*')):
        if not p.is_file() or '__pycache__' in p.parts:continue
        if p.parent==root and p.name in {'FILE_MANIFEST.csv','RELEASE_SIZE.json'}:continue
        rows.append({'path':p.relative_to(root).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)})
    with (root/'FILE_MANIFEST.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['path','bytes','sha256']);w.writeheader();w.writerows(rows)
    size={'files':len(rows),'bytes':sum(r['bytes'] for r in rows),'MiB':sum(r['bytes'] for r in rows)/1048576,
        'scope':'Inventory excludes only the root generated FILE_MANIFEST.csv, root RELEASE_SIZE.json and transient Python bytecode. Historical manifests in provenance are included.'}
    (root/'RELEASE_SIZE.json').write_text(json.dumps(size,indent=2)+'\n',encoding='utf-8');print(json.dumps(size,indent=2))
if __name__=='__main__':main()
