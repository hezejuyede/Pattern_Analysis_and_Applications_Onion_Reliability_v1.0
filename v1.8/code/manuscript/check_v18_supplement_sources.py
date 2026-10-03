"""Verify complete evidence preservation and source-level bilingual tables."""
from pathlib import Path
import json
import hashlib
import re
import pandas as pd
import numpy as np

D18 = Path(__file__).resolve().parent.parent
OUT = D18 / "manuscript_revision"
D17 = D18.parent / "Onion_Deep_Revision_20261003_v17" / "manuscript_revision"
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    index = pd.read_csv(OUT / "SUPPLEMENT_TABLE_INDEX.csv")
    assert len(index) == 39
    for r in index.itertuples():
        p = OUT / r.file
        assert p.is_file() and sha(p) == r.sha256 and len(pd.read_csv(p)) == r.rows
    original = list((D17 / "supplement_tables").glob("*.csv"))
    assert len(original) == 18
    for p in original:
        assert sha(p) == sha(OUT / "supplement_tables" / p.name)
    files = pd.read_csv(OUT / "V18_COMPLETE_RESULT_FILE_INDEX.csv")
    for r in files.itertuples():
        assert sha(OUT / r.file) == r.sha256
    en = (OUT / "Online_Resource_1_EN_v1.8.md").read_text(encoding="utf-8")
    zh = (OUT / "Online_Resource_1_ZH_v1.8.md").read_text(encoding="utf-8")
    assert [line for line in en.splitlines() if line.startswith("| ")] == [line for line in zh.splitlines() if line.startswith("| ")]
    assert "{{" not in en+zh
    for text in (en,zh):
        for path in re.findall(r"`([^`]+\.csv)`", text):
            assert (OUT/path).is_file(), path
        assert text.count("S7.1") == 1 and text.count("S7.6") == 1
    display = pd.read_csv(OUT / "supplement_tables/S27_context_display.csv")
    assert len(display) == 10 and np.allclose(display.single_pp-display.joint_pp, display.single_minus_joint_pp)
    assert (display.difference_min_pp < 0).all() and (display.difference_max_pp > 0).all()
    feature = json.loads((D18 / "features/DINOV2_EXTRACTION_COMPLETE.json").read_text(encoding="utf-8"))
    assert feature["unique_manifest_images"] == 5228 and feature["repeated_qa_images"] == 16
    reviews = [json.loads((D18 / "independent_review" / name).read_text(encoding="utf-8")) for name in
               ("ONION_JOINT_RESULT_REVIEW.json", "POTATO_JOINT_RESULT_REVIEW.json", "ONION_SINGLE_SOURCE_RESULT_REVIEW.json", "POTATO_SINGLE_SOURCE_RESULT_REVIEW.json")]
    assert sum(r["logistic_refits"] for r in reviews) == 96
    assert sum(r["independent_nn_conditions"] for r in reviews) == 68
    assert sum(r["predictions_checked"] for r in reviews) == 3543770
    qa = dict(status="PASS_SUPPLEMENT_SOURCE_COMPATIBILITY", original_csvs_preserved_byte_identically=18,
              total_csvs_verified=39, complete_result_files_verified=len(files), bilingual_table_lines_identical=True,
              derived_context_rows_checked=10, new_prediction_count=3543770, independent_new_logistic_refits=96,
              independent_new_nn_checks=68, source_hashes={name:sha(OUT/name) for name in ("Online_Resource_1_EN_v1.8.md","Online_Resource_1_ZH_v1.8.md")},
              scope="File existence, hashes, preservation, indexed row counts, bilingual numerical tables and cited audit counts. Does not certify native document rendering or repeat model fits.")
    (OUT / "SUPPLEMENT_SOURCE_COMPATIBILITY_QA.json").write_text(json.dumps(qa,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(qa))

if __name__ == "__main__":
    main()
