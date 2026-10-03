"""Verify distribution hashes and exact converted CSV-cell sequence digests."""
from pathlib import Path
import argparse,csv,gzip,hashlib,json
import pyarrow.parquet as pq

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2])
    parser.add_argument('--skip-cell-digests',action='store_true',help='Verify distributed file hashes without decoding all Parquet cells.')
    args=parser.parse_args();root=args.root.resolve()
    with (root/'FILE_MANIFEST.csv').open(encoding='utf-8',newline='') as stream:
        manifest=list(csv.DictReader(stream))
    for row in manifest:
        p=(root/row['path']).resolve();assert p.is_relative_to(root) and p.is_file(),p
        assert p.stat().st_size==int(row['bytes']) and sha(p)==row['sha256'],p
    count=0
    transports=json.loads((root/'provenance/gzip_transport_checks.json').read_text(encoding='utf-8'))
    for relative,record in transports.items():
        decoded=gzip.decompress((root/relative).read_bytes())
        assert len(decoded)==record['original_bytes'] and hashlib.sha256(decoded).hexdigest()==record['original_sha256'],relative
    if not args.skip_cell_digests:
        records=json.loads((root/'provenance/parquet_conversion_checks.json').read_text(encoding='utf-8'))
        for relative,record in records.items():
            p=root/relative;h=hashlib.sha256();names=record['columns'];rows=0
            h.update(json.dumps(names,ensure_ascii=False,separators=(',',':')).encode()+b'\n')
            pf=pq.ParquetFile(p);assert pf.schema_arrow.names==names
            for batch in pf.iter_batches(batch_size=50000):
                columns=[column.to_pylist() for column in batch.columns]
                for cells in zip(*columns):
                    assert all(isinstance(c,str) for c in cells)
                    h.update(json.dumps(list(cells),ensure_ascii=False,separators=(',',':')).encode()+b'\n');rows+=1
            assert rows==record['rows'] and h.hexdigest()==record['cell_sha256'],relative
            count+=rows;print(f'PASS {relative}: {rows} exact string-cell rows',flush=True)
    print(json.dumps({'status':'PASS_DISTRIBUTION_INTEGRITY','files':len(manifest),'converted_rows_verified':count,'cell_digest_check':not args.skip_cell_digests},indent=2))
if __name__=='__main__':main()
