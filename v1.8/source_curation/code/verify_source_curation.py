"""Verify preserved curation objects and optionally restore transport-compressed files."""
from pathlib import Path
import argparse,gzip,hashlib,json
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--restore',type=Path);args=p.parse_args();root=args.root.resolve()
 records=json.loads((root/'PATH_AND_TRANSPORT_MANIFEST.json').read_text(encoding='utf-8'))
 restore=args.restore.resolve() if args.restore else None
 if restore:
  if restore.exists() and any(restore.iterdir()):raise FileExistsError('Use a new or empty restoration directory.')
  restore.mkdir(parents=True,exist_ok=True)
 for record in records:
  path=(root/record['relative_path']).resolve();assert path.is_relative_to(root)
  payload=path.read_bytes();assert hashlib.sha256(payload).hexdigest()==record['distributed_sha256']
  data=gzip.decompress(payload) if record['gzip_transport'] else payload
  assert len(data)==record['original_bytes'] and hashlib.sha256(data).hexdigest()==record['original_sha256']
  if restore:
   target=(restore/record['original_relative_path']).resolve();assert target.is_relative_to(restore);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
 print(json.dumps({'status':'PASS_PRESERVED_CURATION_BYTES','objects':len(records),'restored':str(restore) if restore else None,
  'scope':'Hash and decompression checks, not new image retrieval, geometry estimation, source validation or classifier training.'},indent=2))
if __name__=='__main__':main()
