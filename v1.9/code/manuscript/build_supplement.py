"""Build complete bilingual supplementary PDFs without manuscript-specific gates."""
from pathlib import Path
import hashlib
import html
import json
import re
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether, PageBreak, Image
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import mm
from pypdf import PdfReader
from PIL import Image as PILImage

D19 = Path(__file__).resolve().parent.parent
OUT = D19 / "manuscript_revision"
TITLE = "Attributing related-view gains in plant image evaluation with single-source and joint replacement controls"
AFFIL = "Higher School of Economics and Business, Department of Information Systems in Economics, I. Razzakov Kyrgyz State Technical University, 66 Ch. Aitmatov Ave., Bishkek 720044, Kyrgyz Republic"
WIDTH = 174*mm
for name,path in (("SupBody","times.ttf"),("SupBold","timesbd.ttf"),("SupItalic","timesi.ttf"),("SupBoldItalic","timesbi.ttf"),("SupCN","simsun.ttc"),("SupCNBold","simhei.ttf")):
    pdfmetrics.registerFont(TTFont(name,"C:/Windows/Fonts/"+path))
pdfmetrics.registerFontFamily("SupBody",normal="SupBody",bold="SupBold",italic="SupItalic",boldItalic="SupBoldItalic")
pdfmetrics.registerFontFamily("SupCN",normal="SupCN",bold="SupCNBold",italic="SupCN",boldItalic="SupCNBold")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inline(value):
    value=value.replace("\u2212","-").replace("\u2011","-").replace("\u2013","-").replace("\u2014","-")
    value=html.escape(value)
    value=re.sub(r"\*\*([^*]+)\*\*",r"<b>\1</b>",value)
    value=re.sub(r"\*([^*]+)\*",r"<i>\1</i>",value)
    return value.replace("`","")


def build(lang):
    zh=lang=="ZH"
    source=OUT/f"Online_Resource_1_{lang}_v1.9.md"
    text=source.read_text(encoding="utf-8")
    body="SupCN" if zh else "SupBody"; bold="SupCNBold" if zh else "SupBold"
    styles={
        "body":ParagraphStyle("body",fontName=body,fontSize=10,leading=14.3,spaceAfter=6,wordWrap="CJK" if zh else None,allowWidows=0,allowOrphans=0),
        "title":ParagraphStyle("title",fontName="SupBold",fontSize=15,leading=19,spaceAfter=9),
        "h1":ParagraphStyle("h1",fontName=bold,fontSize=12,leading=16,spaceBefore=11,spaceAfter=6,keepWithNext=True),
        "h2":ParagraphStyle("h2",fontName=bold,fontSize=10.5,leading=14.5,spaceBefore=8,spaceAfter=5,keepWithNext=True),
        "small":ParagraphStyle("small",fontName=body,fontSize=9,leading=12,spaceAfter=5,wordWrap="CJK" if zh else None),
        "cell":ParagraphStyle("cell",fontName="SupBody",fontSize=7.8,leading=10.2,splitLongWords=1),
        "headcell":ParagraphStyle("headcell",fontName="SupBold",fontSize=7.7,leading=10.1,splitLongWords=1),
        "index":ParagraphStyle("index",fontName="SupBody",fontSize=8.5,leading=11.4,spaceAfter=4,splitLongWords=1),
    }
    def p(value,style="body"):
        value=inline(value)
        plain=html.unescape(re.sub(r"<[^>]*>","",value))
        font=pdfmetrics.getFont(styles[style].fontName)
        missing={c for c in plain if not c.isspace() and ord(c) not in font.face.charToGlyph}
        assert not missing, f"Missing glyphs in {lang}/{style}: {sorted(missing)}"
        return Paragraph(value,styles[style])
    story=[p("Pattern Analysis and Applications","small"),
           p("Online Resource 1" if not zh else "在线资源1（中文参考稿）","h1"),p(TITLE,"title"),
           p("Xin Li, Bojian Guo and Asel Kartanova","small"),p(AFFIL,"small"),
           p("Corresponding author: Xin Li; lixin26@kstu.kg","small"),
           p("ORCID: Xin Li 0009-0005-0670-5006; Bojian Guo 0009-0005-8353-2436; Asel Kartanova 0000-0003-1479-0747","small"),Spacer(1,6)]
    # The first three original blocks repeat this complete, deliberately uniform cover metadata.
    blocks=re.split(r"\n\s*\n",text.strip())
    blocks=blocks[3:]
    table_count=0; headings=[]; table_rows=[]; pending_table_caption=[]
    header_names={"task":"Task","model":"Model","endpoint":"Endpoint","difference":"Difference (pp)",
                  "split_p025":"2.5th split percentile","split_p975":"97.5th split percentile","stratum":"Original class",
                  "mean":"Mean (pp)","graph":"Dependence graph","balanced_accuracy":"Balanced accuracy",
                  "auroc":"AUROC","brier":"Brier score","variant":"Variant","metric":"Metric","estimate":"Estimate",
                  "lower":"Lower","upper":"Upper","region":"Region","train_n":"Training images","n":"Test images",
                  "roc_auc":"AUROC","single_pp":"Single (pp)","joint_pp":"Joint (pp)","single_minus_joint_pp":"Single - joint (pp)",
                  "difference_min_pp":"Minimum difference (pp)","difference_max_pp":"Maximum difference (pp)"}
    for block_index,block in enumerate(blocks):
        b=block.strip()
        if not b:continue
        if b.startswith("### "):
            title=b[4:];headings.append(title)
            if title.startswith("Table ") or title.startswith("表"):
                pending_table_caption=[p(title,"h2")]
            else:
                story.append(p(title,"h2"))
        elif b.startswith("## "):
            title=b[3:];story.append(p(title,"h1"));headings.append(title)
        elif b.startswith("|"):
            rows=[[x.strip() for x in line.strip().strip("|").split("|")] for line in b.splitlines()
                  if not re.fullmatch(r"[|:\-\s]+",line.strip())]
            assert rows and all(len(row)==len(rows[0]) for row in rows)
            n=len(rows[0]); headers=rows[0]
            if headers[:2]==["Task","Original class"]:
                widths=[13,45,19,24,26,22,25]
            elif headers[0]=="Task/design" and n==8:
                widths=[23,30,22,18,18,18,22.5,22.5]
            elif headers[0]=="Task/design" and n==6:
                widths=[25,49,16,28,28,28]
            elif headers[:3]==["variant","region","model"]:
                widths=[23,25,34,15,13,23,20,21]
            elif headers[:2]==["graph","model"]:
                widths=[24,42,36,36,36]
            elif headers[:2]==["task","model"] and n==7:
                widths=[15,39,21,21,25,26.5,26.5]
            elif n==6:
                widths=[16,37,37,27,28.5,28.5]
            elif n==5:
                widths=[35,37,34,34,34]
            elif n==4:
                widths=[29,70,37.5,37.5]
            else:
                widths=[174/n]*n
            assert abs(sum(widths)-174)<1e-8
            cells=[]
            for index,row in enumerate(rows):
                values=[header_names.get(v,v) if index==0 else v.replace("_"," ") for v in row]
                cells.append([p(v,"headcell" if index==0 else "cell") for v in values])
            tab=Table(cells,colWidths=[v*mm for v in widths],repeatRows=1,hAlign="LEFT",splitByRow=1)
            tab.setStyle(TableStyle([("VALIGN",(0,0),(-1,-1),"TOP"),("BACKGROUND",(0,0),(-1,0),colors.HexColor("#EDF1F4")),
                                    ("LINEABOVE",(0,0),(-1,0),.7,colors.black),("LINEBELOW",(0,0),(-1,0),.4,colors.black),
                                    ("LINEBELOW",(0,-1),(-1,-1),.65,colors.black),("LEFTPADDING",(0,0),(-1,-1),4),
                                    ("RIGHTPADDING",(0,0),(-1,-1),4),("TOPPADDING",(0,0),(-1,-1),4),
                                    ("BOTTOMPADDING",(0,0),(-1,-1),4)]))
            # Small tables stay together. Long tables split only at complete rows,
            # repeat the header, and always retain at least two body rows at a split.
            if len(rows)>15:
                tab._rowSplitRange=(3,-2)
            story.append(KeepTogether(pending_table_caption+[tab]))
            pending_table_caption=[]
            story.append(Spacer(1,8));table_count+=1;table_rows.append(len(rows)-1)
        elif b.startswith("- "):
            story.append(p(b[2:],"index"))
        else:
            paragraph=p(b)
            if pending_table_caption:
                pending_table_caption.append(paragraph)
                continue
            if block_index+1<len(blocks) and blocks[block_index+1].lstrip().startswith("|"):
                paragraph.keepWithNext=True
            story.append(paragraph)
    story.append(PageBreak())
    story.append(p("Supplementary figures" if not zh else "补充图件","h1"))
    figures=[("Fig_S1_Matched_Replacement_Binary.png",
              "Figure S1. Retained three-model binary endpoint. Means and 2.5th-97.5th percentiles of 30 split means describe design variation. These are not biological confidence intervals; the binary/multiclass comparison does not identify a causal effect of label granularity.",
              "补图S1。保留原始三个模型的二分类端点。点及横线为30个划分均值的平均与2.5-97.5百分位，描述设计变异，不是生物置信区间；二分类/多分类差不识别标签粒度的因果效应。"),
             ("Fig_S2_External_Context.png",
              "Figure S2. Retained locked onion transfer context. Point estimates are unchanged for the same 813 predictions. The all-positive and spatial graph intervals use 3,000 fixed-prediction component resamples; the image-unit interval is illustrative. Available dependence rules do not establish independent plants or a new field cohort.",
              "补图S2。保留的锁定洋葱迁移情境。相同813条预测的点估计始终不变；全部阳性与空间关系图区间基于3,000次固定预测组件重采样，图像单位区间仅为示例。现有依赖规则不能证明植株独立或新增田间队列。")]
    for index,(file,en_caption,zh_caption) in enumerate(figures):
        if index:story.append(PageBreak())
        path=D19/"figures"/file
        with PILImage.open(path) as im:w,h=im.size
        story.append(KeepTogether([Image(str(path),width=WIDTH,height=WIDTH*h/w),Spacer(1,8),p(zh_caption if zh else en_caption,"small")]))
    name="ESM_1.pdf" if not zh else "Online_Resource_1_中文参考_v1.9.pdf"
    def page(c,doc):
        c.saveState();c.setFont("SupBody",8)
        c.setFillColor(colors.HexColor("#5D6870"));c.drawString(18*mm,12*mm,"Pattern Analysis and Applications | Online Resource 1")
        c.drawRightString(192*mm,12*mm,str(doc.page));c.restoreState()
    doc=SimpleDocTemplate(str(OUT/name),pagesize=(210*mm,297*mm),leftMargin=18*mm,rightMargin=18*mm,topMargin=17*mm,bottomMargin=21*mm,
                         title=TITLE+" - Online Resource 1",author="Xin Li; Bojian Guo; Asel Kartanova")
    doc.build(story,onFirstPage=page,onLaterPages=page)
    pdf=PdfReader(OUT/name)
    extracted="\n".join(page.extract_text() or "" for page in pdf.pages)
    assert "S7.6" in extracted and "S8.5" in extracted and "S1" in extracted and "S53" in extracted
    assert "lixin26@kstu.kg" in extracted and "Asel Kartanova" in extracted
    assert "\x00" not in extracted
    assert table_count==9 and table_rows==[12,7,14,4,24,10,7,8,20]
    (OUT/(name+".extracted.txt")).write_text(extracted,encoding="utf-8")
    return dict(file=name,pages=len(pdf.pages),inline_tables=table_count,inline_table_body_rows=table_rows,supplementary_figures=2,
                source_sha256=sha(source),pdf_sha256=sha(OUT/name),status="BUILT_TEXT_QA_PASS_AWAITING_ALL_PAGE_VISUAL_REVIEW")


if __name__=="__main__":
    reports=[build("EN"),build("ZH")]
    (OUT/"SUPPLEMENT_PDF_BUILD.json").write_text(json.dumps(dict(documents=reports,code_sha256=sha(__file__)),indent=2)+"\n",encoding="utf-8")
    print(json.dumps(reports,ensure_ascii=False))
