"""Confirm native figure dimensions and record completed PDF visual review."""
from pathlib import Path
import hashlib
import json
import pandas as pd
from PIL import Image
from pypdf import PdfReader

root = Path(__file__).resolve().parent
records = []
for name in ("Fig_1_Matched_Replacement_Design", "Fig_2_Joint_Replacement", "Fig_3_Single_Source_Replacement"):
    pdf = root / (name + ".pdf")
    pages = PdfReader(pdf).pages
    assert len(pages) == 1
    width, height = float(pages[0].mediabox.width)*25.4/72, float(pages[0].mediabox.height)*25.4/72
    expected_height = 118 if name.startswith("Fig_1") else 124
    assert abs(width-174) < .001 and abs(height-expected_height) < .001
    with Image.open(root / (name + ".png")) as image:
        assert image.mode == "RGB"
        assert all(abs(d-600) < .02 for d in image.info["dpi"])
        pixels, dpi = image.size, image.info["dpi"]
    if not name.startswith("Fig_1"):
        raw = pd.read_csv(root / (name + "_split_source.csv"))
        plotted = pd.read_csv(root / (name + "_plot_source.csv"))
        assert raw.model.nunique() == 5 and plotted.model.nunique() == 5
        assert len(plotted) == 40
    rendered = root / "qa" / (name + "-1.png")
    assert rendered.is_file()
    records.append(dict(name=name, width_mm=width, height_mm=height, png_mode="RGB", png_pixels=pixels, png_dpi=dpi,
                        pdf_sha256=hashlib.sha256(pdf.read_bytes()).hexdigest(),
                        rendered_pdf_review_path=str(rendered), plot_rows=None if name.startswith("Fig_1") else len(plotted),
                        split_rows=None if name.startswith("Fig_1") else len(raw),
                        visual_review=("Actual PDF rendered with Poppler at 180 dpi and inspected in full. Diagram boxes, branch arrows, q0 budgets, replacement counts, E/H insertions and anchor rules are legible without clipping or overlap."
                                       if name.startswith("Fig_1") else
                                       "Actual PDF rendered with Poppler at 180 dpi and inspected in full. No clipped title, label, tick, legend or interval detected; all five models and both tasks visible.")))
(root / "FIGURE_VISUAL_QA.json").write_text(json.dumps(dict(status="PASS_DEFINED_VISUAL_AND_FILE_CHECKS", figures=records,
    limitations="Visual/file checks do not replace independent statistical reconstruction. Joint ranges are 30-split percentiles; single ranges are 10-split min/max. Captions must accompany figures."), indent=2)+"\n", encoding="utf-8")
print("Figure visual and file checks PASS.")
