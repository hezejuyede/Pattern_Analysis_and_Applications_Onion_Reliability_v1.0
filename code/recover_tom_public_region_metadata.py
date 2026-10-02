"""Recover public search-index region associations, never biological plant IDs.

Read-only POSTs reproduce the documented search and pagination form. Raw HTML
responses are retained so inferred filename-to-region joins can be audited.
"""
from pathlib import Path
import csv
import concurrent.futures
import hashlib
import http.cookiejar
import json
import re
import time
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
OUT = Path('D:/论文/SCI投稿汇总/Onion_External_Cohort_Feasibility_20261002/tom_metadata_recovery')
URL = 'https://ppedmas.org/search_images.php'
REGIONS = ['Plateau-Central', 'Centre-Ouest', 'Centre-Sud']

def recover(region):
    dest = OUT / region
    dest.mkdir(parents=True, exist_ok=True)
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
    def post(payload):
        # Pagination mutates session cursor: never retry an uncertain response.
        req = urllib.request.Request(URL, data=urllib.parse.urlencode(payload).encode())
        with opener.open(req, timeout=40) as response:
            return response.read()
    form = post({'crop': 'Onion', 'region': region, 'SearchImages': 'Search Images'})
    (dest / 'search_form.html').write_bytes(form)
    raw = post({'commune': '', 'image_type': '', 'labelled_WASCAL': '', 'crop_type': '', 'submitBtn': 'Search'})
    rows, pages, expected = [], [], None
    page = 1
    while True:
        html = raw.decode('utf8', 'replace')
        file = dest / f'page_{page:04d}.html'
        file.write_bytes(raw)
        count = re.search(r'\[(\d+) Image\(s\)\]', html)
        position = re.search(r'>\s*(\d+) of (\d+)\s*<', html)
        if not count or not position:
            raise RuntimeError(f'Missing search count or cursor: {region}, page {page}')
        if int(position[1]) != page:
            raise RuntimeError(f'Unexpected cursor: {region}, wanted {page}, got {position[1]}')
        expected = int(count[1])
        if not 1 <= int(position[2]) <= 1000:
            raise RuntimeError('Unreasonable page count')
        keys = list(dict.fromkeys(re.findall(r"href=['\"]datasets/images/original_images/+(\d+)\.jpg", html)))
        rows.extend({'region': region, 'source_key': key, 'search_page': page,
                     'evidence_file': str(file.relative_to(OUT)), 'evidence_type': 'public_search_index_membership',
                     'plant_identity_verified': False} for key in keys)
        pages.append({'file': str(file.relative_to(OUT)), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest(), 'images': len(keys)})
        if page % 20 == 0 or page == 1 or page == int(position[2]):
            print(f'{region}: page {page}/{position[2]}, recovered {len(rows)}/{expected}', flush=True)
        if page == int(position[2]):
            break
        time.sleep(0.15)
        raw = post({'next': 'Next'})
        page += 1
    result = {'region': region, 'declared_image_count': expected, 'listed_records': len(rows),
              'unique_source_keys': len({r['source_key'] for r in rows}), 'pages': pages}
    (dest / 'audit.json').write_text(json.dumps(result, indent=2), encoding='utf8')
    return rows, result

def write_csv(path, rows, columns=None):
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=columns or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, audits, failures = [], [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(recover, region): region for region in REGIONS}
        for future in concurrent.futures.as_completed(futures):
            region = futures[future]
            try:
                data, audit = future.result()
                rows.extend(data)
                audits.append(audit)
            except Exception as exc:
                failures.append({'region': region, 'error': repr(exc)})
    if rows:
        write_csv(OUT / 'public_region_index.csv', rows)
    index = {}
    for row in rows:
        index.setdefault(row['source_key'], set()).add(row['region'])
    with (ROOT / 'results/reliability_benchmark_v1/split_manifests/tom_acquisition_day_outer_folds.csv').open(encoding='utf-8-sig') as f:
        frame = list(csv.DictReader(f))
    joined = []
    for row in frame:
        regions = sorted(index.get(row['source_group'].split(':')[-1], []))
        joined.append({**row, 'recovered_region': '|'.join(regions),
                       'region_match_status': 'unique_index_match' if len(regions) == 1 else ('not_listed' if not regions else 'ambiguous_multiple_regions'),
                       'plant_identity_verified': False})
    write_csv(OUT / 'analysis_image_region_join.csv', joined)
    summary = {'source_url': URL, 'date': '2026-10-02', 'regions': audits, 'failures': failures,
               'source_records': len(rows), 'unique_source_keys': len(index),
               'source_keys_in_multiple_regions': sum(len(x)>1 for x in index.values()),
               'analysis_images': len(frame), 'analysis_images_region_matched': sum(r['region_match_status']=='unique_index_match' for r in joined),
               'analysis_images_ambiguous': sum(r['region_match_status']=='ambiguous_multiple_regions' for r in joined),
               'inference_limit': 'Region search membership is provenance recovered from a public index. It does not establish plant/plot identity, independent acquisition, or new pathology review. No model outcomes were evaluated.'}
    (OUT / 'RECOVERY_SUMMARY.json').write_text(json.dumps(summary, indent=2), encoding='utf8')
    print(json.dumps({k:v for k,v in summary.items() if k!='regions'}, indent=2), flush=True)

if __name__ == '__main__':
    main()
