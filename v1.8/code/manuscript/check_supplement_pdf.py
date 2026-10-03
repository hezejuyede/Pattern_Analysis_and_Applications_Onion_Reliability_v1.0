"""Bind supplementary PDF text/geometry checks and actual PDF render hashes."""
from pathlib import Path
import hashlib
import json
import re
import pdfplumber
from pypdf import PdfReader

D18 = Path(__file__).resolve().parent.parent
OUT = D18 / 'manuscript_revision'

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    build = json.loads((OUT / 'SUPPLEMENT_PDF_BUILD.json').read_text(encoding='utf-8'))
    previous = json.loads((OUT / 'qa_supp/pre_final_render_hashes.json').read_text(encoding='utf-8'))
    docs = []
    for lang, record in zip(('en', 'zh'), build['documents']):
        path = OUT / record['file']
        assert sha(path) == record['pdf_sha256']
        pages = list((OUT / 'qa_supp' / lang).glob('page-*.png'))
        assert len(pages) == record['pages']
        text = '\n'.join(p.extract_text() for p in PdfReader(path).pages)
        compact = re.sub(r'\s+', '', text)
        for token in ('Xin Li', 'Bojian Guo', 'Asel Kartanova', 'lixin26@kstu.kg', 'v1.8.0', 'code/portable', 'supplementary_context', 'SUPPLEMENT_BUILD_v17_RETAINED.json'):
            assert re.sub(r'\s+', '', token) in compact, token
        for token in ('v1.2.1', '\ufffd', 'final journal upload format has not been certified', '尚未认证'):
            assert token not in text, token
        assert len(re.findall(r'supplement_tables/', text)) >= 39
        bad = []
        with pdfplumber.open(path) as pdf:
            for number, p in enumerate(pdf.pages, 1):
                assert abs(p.width - 595.2756) < .1 and abs(p.height - 841.8898) < .1
                for c in p.chars:
                    if c['x0'] < 40 or c['x1'] > p.width-40 or c['top'] < 25 or c['bottom'] > p.height-20:
                        bad.append({'page': number, 'text': c['text'], 'bbox': [c['x0'], c['top'], c['x1'], c['bottom']]})
        assert not bad, bad[:20]
        renders = {str(p.relative_to(OUT / 'qa_supp')): sha(p) for p in sorted(pages)}
        docs.append(dict(file=path.name, pages=record['pages'], pdf_sha256=sha(path), source_sha256=record['source_sha256'], inline_tables=record['inline_tables'], inline_table_body_rows=record['inline_table_body_rows'], supplementary_figures=2, render_sha256=renders, changed_vs_pre_release_visual_check=[k for k,v in renders.items() if previous.get(k)!=v], text_and_geometry_status='PASS'))
    result = dict(status='PASS_PDF_TEXT_GEOMETRY_AND_RENDER_BINDING', documents=docs, scope='PDF text, expected metadata and file references, table-index completeness, visible-character page bounds, and actual rendered page hashes. This automated report does not itself constitute visual review or certify remote tag publication.')
    (OUT / 'SUPPLEMENT_PDF_TECHNICAL_QA.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps([dict(file=d['file'], changed_pages=d['changed_vs_pre_release_visual_check'], status=d['text_and_geometry_status']) for d in docs], ensure_ascii=False))

if __name__ == '__main__':
    main()
