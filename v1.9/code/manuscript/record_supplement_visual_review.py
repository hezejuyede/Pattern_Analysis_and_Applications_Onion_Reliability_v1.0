"""Record the completed manual inspection of every rendered v1.9 supplement page.

This records the actual 31-page visual review performed in the agent conversation;
it does not infer visual quality from successful PDF generation.
"""
from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone

M=Path('D:/论文/SCI投稿汇总/Onion_Deep_Revision_20261003_v19/manuscript_revision')
def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,obj):
    Path(p).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
build=json.loads((M/'SUPPLEMENT_PDF_BUILD.json').read_text())
docs=[]
for language,doc in zip(('en','zh'),build['documents']):
    images=sorted((M/'qa_supp'/language).glob('page-*.png'))
    assert len(images)==doc['pages']
    assert sha(M/doc['file'])==doc['pdf_sha256']
    docs.append({
        'file':doc['file'],'pdf_sha256':doc['pdf_sha256'],
        'source_sha256':doc['source_sha256'],
        'pages_reviewed':list(range(1,len(images)+1)),
        'rendered_pages':[{'page':i,'file':str(p.relative_to(M)).replace('\\','/'),'sha256':sha(p)} for i,p in enumerate(images,1)],
        'observed_layout':'All pages checked visually: complete first-page metadata; readable body and native tables; no observed clipped or overlapping text, missing negative signs, detached table headings, or excessive table width. Deliberate whitespace preserves complete tables and separates the two appendix figures.',
        'inline_tables':9,'inline_table_body_rows':106,'appendix_figures':2,
        'status':'PASS_ALL_PAGES_VISUALLY_REVIEWED'
    })
    doc['status']='PASS_TEXT_GEOMETRY_SOURCE_AND_ALL_PAGE_VISUAL_QA'
report={
    'status':'PASS_ALL_31_PAGES_VISUALLY_REVIEWED',
    'recorded_utc':datetime.now(timezone.utc).isoformat(),
    'renderer':'Poppler pdftoppm, 115 dpi, every PDF page',
    'method':'Actual manual inspection of each individual rendered page in the agent visual viewer, followed by automatic source and geometry checks.',
    'documents':docs,
    'numeric_tables':'53 supplied CSVs; 39 retained byte-identically, 14 added; EN/ZH inline numeric tables match.',
    'technical_qa_sha256':sha(M/'SUPPLEMENT_PDF_TECHNICAL_QA.json'),
    'source_compatibility_qa_sha256':sha(M/'SUPPLEMENT_SOURCE_COMPATIBILITY_QA.json'),
    'limitations':'Layout and source consistency checks do not guarantee journal acceptance. External release availability is verified separately before delivery.'
}
write(M/'SUPPLEMENT_PDF_VISUAL_QA.json',report)
build['visual_qa_file']='SUPPLEMENT_PDF_VISUAL_QA.json'
build['technical_qa_file']='SUPPLEMENT_PDF_TECHNICAL_QA.json'
build['source_compatibility_qa_file']='SUPPLEMENT_SOURCE_COMPATIBILITY_QA.json'
build['status']='PASS_SOURCE_TECHNICAL_AND_ALL_PAGE_VISUAL_QA'
write(M/'SUPPLEMENT_PDF_BUILD.json',build)
print(report['status'])
