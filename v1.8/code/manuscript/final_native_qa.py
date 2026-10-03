"""Bind completed visual inspection to final manuscript source and byte copies.

This is local artifact QA, not journal acceptance or independent peer review.
The page inspection ledger reflects visual checks actually performed in this run.
"""
from pathlib import Path
from datetime import datetime, timezone
from zipfile import ZipFile
import hashlib
import json
import re
import shutil
from docx import Document
from lxml import etree
from pypdf import PdfReader

D18 = Path('D:/论文/SCI投稿汇总/Onion_Deep_Revision_20261003_v18')
SOURCE = D18 / 'manuscript_revision'
NATIVE = SOURCE / 'native'
W = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
     'm': 'http://schemas.openxmlformats.org/officeDocument/2006/math'}

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def normalize(text):
    return ' '.join(text.replace('−', '-').split())

def markdown_tables(text):
    groups, current = [], []
    for line in text.splitlines() + ['']:
        if line.startswith('|'):
            row = [x.strip() for x in line.strip().strip('|').split('|')]
            if not all(re.fullmatch(r'[:\s-]+', x) for x in row):
                current.append(row)
        elif current:
            groups.append(current)
            current = []
    return groups

def citations(text):
    return [x.strip() for block in re.findall(r'\\cite\{([^}]+)\}', text)
            for x in block.split(',')]

def numbered_citations(text):
    return [int(x.strip()) for block in re.findall(r'\[([0-9, ]+)\]', text)
            for x in block.split(',')]

def main():
    build = json.loads((NATIVE / 'NATIVE_DOCUMENT_BUILD.json').read_text('utf-8'))
    inventory = json.loads((SOURCE / 'reference_inventory.json').read_text('utf-8'))
    refs = {x['key']: x for x in inventory}
    sources = {k: (SOURCE / n).read_text('utf-8') for k, n in
               [('en', 'full_text.md'), ('zh', 'full_text_zh.md')]}
    assert citations(sources['en']) == citations(sources['zh'])
    assert set(citations(sources['en'])) == set(refs)
    assert len(refs) == 22
    numeric_tables = {lang: [[[re.findall(r'[-+]?\d+(?:\.\d+)?', normalize(cell))
                               for cell in row] for row in table]
                             for table in markdown_tables(source)]
                      for lang, source in sources.items()}
    assert numeric_tables['en'] == numeric_tables['zh']
    equations = {k: re.findall(r'\\\[(.*?)\\\]', v, flags=re.S)
                 for k, v in sources.items()}
    assert equations['en'] == equations['zh'] and len(equations['en']) == 3
    figures = json.loads((SOURCE / 'figure_map.json').read_text('utf-8'))
    figure_hashes = {k: sha(v) for k, v in figures.items()}
    prior = {'en': '7d401b6019b9', 'zh': '52d505b32cd5'}
    initial = {'en': 'f44bf0572cea', 'zh': '814e332a3516'}
    initial_reuse = {'en': {2, 3, 9}, 'zh': {3, 6, 9}}
    final_inspected = {'en': {8}, 'zh': {5, 7, 8, 10, 11}}
    reports = []
    for report in build['reports']:
        lang = report['language']
        assert sha(report['source']) == report['source_sha256']
        assert sha(report['docx']) == report['docx_sha256']
        doc = Document(report['docx'])
        grids = [[[normalize(c.text) for c in row.cells] for row in table.rows]
                 for table in doc.tables]
        expected_grids = [[[normalize(c) for c in row] for row in table]
                          for table in markdown_tables(sources[lang])]
        assert grids == expected_grids
        with ZipFile(report['docx']) as archive:
            xml = etree.fromstring(archive.read('word/document.xml'))
            alltext = '\n'.join(''.join(p.xpath('.//w:t/text()', namespaces=W))
                                for p in xml.xpath('//w:p', namespaces=W))
            expected_citations = [refs[k]['citation_number'] for k in citations(sources[lang])]
            assert numbered_citations(alltext) == expected_citations
            assert len(xml.xpath('//m:oMath', namespaces=W)) == 3
            assert len(xml.xpath('//w:tbl', namespaces=W)) == 3
            assert len(xml.xpath('//w:tblHeader', namespaces=W)) == 3
            assert len(xml.xpath('//w:cantSplit', namespaces=W)) == 30
            media = [x for x in archive.namelist() if x.startswith('word/media/')]
            media_hashes = {hashlib.sha256(archive.read(x)).hexdigest() for x in media}
            assert media_hashes == set(figure_hashes.values()) and len(media) == 3
            footer = '\n'.join(archive.read(x).decode('utf-8') for x in archive.namelist()
                               if re.fullmatch(r'word/footer\d+\.xml', x))
            assert 'PAGE' in footer
        urls = sorted(set(re.findall(r'https?://[^\s]+', sources[lang] + '\n' +
                                    '\n'.join(r['formatted'] for r in inventory))))
        assert all(u in alltext for u in urls)
        assert '\\cite{' not in alltext and '\\tag{' not in alltext
        assert not any(c in alltext for c in ['\ufffd', '\u25a1'])
        reader = PdfReader(report['rendered_pdf'])
        assert len(reader.pages) == report['page_count']
        page_records = []
        for index in range(1, report['page_count'] + 1):
            current_page = Path(report['qa_dir']) / f'page-{index}.png'
            record = {'page': index, 'path': str(current_page), 'sha256': sha(current_page)}
            if index in final_inspected[lang]:
                record['visual_review'] = 'PASS_DIRECTLY_INSPECTED_FINAL_RENDER'
            else:
                reuse_dir = initial[lang] if index in initial_reuse[lang] else prior[lang]
                prior_page = NATIVE / 'qa' / lang / reuse_dir / current_page.name
                assert sha(prior_page) == record['sha256']
                record['visual_review'] = 'PASS_IDENTICAL_TO_PREVIOUSLY_INSPECTED_PAGE'
                record['reviewed_source_page'] = str(prior_page)
                record['reviewed_source_page_sha256'] = sha(prior_page)
            page_records.append(record)
        stem = ('Pattern_Analysis_and_Applications_Research_Manuscript_v1.8'
                if lang == 'en' else 'Pattern_Analysis_and_Applications_中文研究稿_v1.8')
        copies = {}
        for extension, original in [('docx', report['docx']), ('pdf', report['rendered_pdf'])]:
            destination = SOURCE / f'{stem}.{extension}'
            shutil.copy2(original, destination)
            assert sha(destination) == sha(original)
            copies[extension] = {'path': str(destination), 'sha256': sha(destination),
                                 'byte_identical_to_reviewed_original': True}
        reports.append({**report, 'visual_review': 'ALL_PAGES_PASS',
                        'pdf_sha256': sha(report['rendered_pdf']),
                        'table_cell_text_and_numeric_values_match_source': True,
                        'citation_order_and_inventory_check': 'PASS',
                        'literal_urls_preserved_in_docx': len(urls),
                        'equations_native_omml': 3,
                        'figure_embedded_bytes_match_scientific_pngs': True,
                        'page_number_field': 'PASS', 'copies': copies, 'pages': page_records})
    result = {
        'utc': datetime.now(timezone.utc).isoformat(),
        'status': 'LOCAL_FORMAT_PASS_NOT_EDITOR_APPROVAL',
        'review_scope': 'Native structure, source fidelity, English/Chinese numeric/citation/equation parity, and visual inspection of all rendered pages; prior visual review reused only for byte-identical page PNGs.',
        'limitations': [
            'This is artifact QA, not acceptance forecasting or an independent human peer review.',
            'Source scientific evidence and external-verification limits remain as stated in the manuscript.',
            'The prepared v1.8.0 repository declaration requires parent verification of the remote tag before delivery.',
            'Chinese manuscript is a reference translation, not an English journal submission replacement.',
            'Tables spanning pages use repeated headers and protected rows; figure/caption blocks stay on one page.'
        ],
        'builder_sha256': sha(D18 / 'code/build_native_manuscript.py'),
        'qa_script_sha256': sha(__file__),
        'reference_inventory_sha256': sha(SOURCE / 'reference_inventory.json'),
        'figure_map_sha256': sha(SOURCE / 'figure_map.json'),
        'figure_sha256': figure_hashes,
        'bilingual_checks': {'citation_key_sequence': 'PASS', 'three_table_per_cell_numeric_parity': 'PASS',
                             'display_equation_source_parity': 'PASS',
                             'manual_scientific_paragraph_review': str(D18 / 'independent_review/SCIENTIFIC_PARAGRAPH_REVIEW_NATIVE_AGENT.json')},
        'reports': reports,
    }
    outfile = NATIVE / 'FINAL_NATIVE_DOCUMENT_QA.json'
    outfile.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'status': result['status'], 'qa': str(outfile),
                      'languages': [{'language': r['language'], 'pages': r['page_count'],
                                     'copies': r['copies']} for r in reports]}, ensure_ascii=False))

if __name__ == '__main__':
    main()
