"""Complete v1.9 local QA only after a hash-bound all-page visual ledger exists."""
from pathlib import Path
from zipfile import ZipFile
from datetime import datetime, timezone
import argparse, hashlib, json, re, shutil
from lxml import etree
from docx import Document
from pypdf import PdfReader

D19=Path('D:/论文/SCI投稿汇总/Onion_Deep_Revision_20261003_v19')
NS={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
    'm':'http://schemas.openxmlformats.org/officeDocument/2006/math'}
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def norm(s): return ' '.join(s.replace('−','-').split())
def tables(s):
    out=[]; current=[]
    for line in s.splitlines()+['']:
        if line.startswith('|'):
            row=[v.strip() for v in line.strip('| ').split('|')]
            if not all(re.fullmatch(r'[: -]+',v) for v in row): current.append(row)
        elif current: out.append(current);current=[]
    return out
def citations(s):
    return [k.strip() for b in re.findall(r'\\cite\{([^}]+)\}',s) for k in b.split(',')]

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--native-dir',type=Path,default=D19/'manuscript_revision/native')
    ap.add_argument('--visual-ledger',type=Path,required=True)
    a=ap.parse_args()
    build_path=a.native_dir/'NATIVE_DOCUMENT_BUILD.json'
    build=json.loads(build_path.read_text('utf-8'))
    assert build['status']=='NATIVE_STRUCTURE_AND_RENDER_PASS_PENDING_VISUAL_QA'
    ledger=json.loads(a.visual_ledger.read_text('utf-8'))
    assert ledger['status']=='ALL_RENDERED_PAGES_VISUALLY_INSPECTED'
    assert ledger['build_manifest_sha256']==sha(build_path)
    refs_path=D19/'manuscript_revision/reference_inventory.json'
    refs=json.loads(refs_path.read_text('utf-8'))
    refmap={x['key']:x['citation_number'] for x in refs}
    seen_sources={};out=[]
    for r in build['reports']:
        assert sha(r['source'])==r['source_sha256']
        assert sha(r['docx'])==r['docx_sha256']
        pdf=Path(r['rendered_pdf']); reader=PdfReader(pdf)
        assert len(reader.pages)==r['page_count']
        entry=ledger['documents'][r['language']]
        assert entry['docx_sha256']==sha(r['docx']) and entry['pdf_sha256']==sha(pdf)
        assert entry['status']=='PASS' and len(entry['pages'])==r['page_count']
        for i,page in enumerate(entry['pages'],1):
            assert page['page']==i and page['status']=='PASS'
            png=Path(r['qa_dir'])/f'page-{i}.png'
            assert page['sha256']==sha(png)
        source=Path(r['source']).read_text('utf-8')
        with ZipFile(r['docx']) as z:
            xml=etree.fromstring(z.read('word/document.xml'))
            paragraphs=[''.join(p.xpath('.//w:t/text()',namespaces=NS)) for p in xml.xpath('//w:body/w:p',namespaces=NS)]
            native='\n'.join(paragraphs)
            assert '\ufffd' not in native
            for u in re.findall(r'https?://[^\s]+',source): assert u in native
            if r['language']!='cover_en':
                seen_sources[r['language']]=source
                d=Document(r['docx'])
                actual=[[[norm(c.text) for c in row.cells] for row in t.rows] for t in d.tables]
                expected=[[[norm(c) for c in row] for row in t] for t in tables(source)]
                assert actual==expected and len(actual)==4
                assert len(xml.xpath('//m:oMath',namespaces=NS))==3
                assert len(xml.xpath('//w:tblHeader',namespaces=NS))==4
                assert len([n for n in z.namelist() if n.startswith('word/media/')])==3
                actual_citations=[int(k) for b in re.findall(r'\[([\d, ]+)\]',native) for k in b.split(',')]
                assert actual_citations==[refmap[k] for k in citations(source)]
                assert len([p for p in paragraphs if re.match(r'^\d+\. [A-Z]',p)])==len(refs)
                assert all(x in native for x in ['https://orcid.org/0009-0005-0670-5006',
                    'https://orcid.org/0009-0005-8353-2436','https://orcid.org/0000-0003-1479-0747'])
                assert entry['all_four_tables_complete_and_same_page'] is True
                assert entry['formula_1_denominator_visually_1_over_abs_Pc'] is True
        out.append({**r,'visual_review':'ALL_PAGES_PASS'})
    assert citations(seen_sources['en'])==citations(seen_sources['zh'])
    numeric=lambda s:[[[re.findall(r'[-+]?\d+(?:\.\d+)?',norm(c)) for c in row] for row in t] for t in tables(s)]
    assert numeric(seen_sources['en'])==numeric(seen_sources['zh'])
    assert re.findall(r'\\\[(.*?)\\\]',seen_sources['en'],re.S)==re.findall(r'\\\[(.*?)\\\]',seen_sources['zh'],re.S)
    # Do not create delivery-path copies until all three files and bilingual checks pass.
    for r in out:
        target_dir=D19/('submission_files' if r['language']=='cover_en' else 'manuscript_revision')
        copies={}
        for ext,original in [('docx',Path(r['docx'])),('pdf',Path(r['rendered_pdf']))]:
            target=target_dir/(Path(r['docx']).stem+'.'+ext)
            shutil.copy2(original,target); assert sha(target)==sha(original)
            copies[ext]={'path':str(target),'sha256':sha(target),'matches_reviewed_original':True}
        r['copies']=copies
    report={'utc':datetime.now(timezone.utc).isoformat(),'status':'LOCAL_FORMAT_PASS_NOT_EDITOR_APPROVAL',
        'scope':'Native structure, source fidelity, all-page visual review, bilingual table numbers/equations/citation order, and exact delivery copies.',
        'build_manifest_sha256':sha(build_path),'visual_ledger_sha256':sha(a.visual_ledger),
        'reference_inventory_sha256':sha(refs_path),'qa_script_sha256':sha(__file__),'reports':out,
        'limitations':['Not editor approval or a promise of peer review or acceptance.',
            'Scientific constraints and uncertainty remain as explicitly described in the manuscript.']}
    destination=a.native_dir/'FINAL_NATIVE_DOCUMENT_QA.json'
    destination.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'status':report['status'],'qa':str(destination)},ensure_ascii=False))
if __name__=='__main__':main()
