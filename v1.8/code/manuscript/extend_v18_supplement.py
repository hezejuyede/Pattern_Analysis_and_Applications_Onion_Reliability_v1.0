"""Preserve v17 supplementary evidence and append the frozen v18 extension."""
from pathlib import Path
import hashlib
import json
import shutil
import pandas as pd

D18 = Path(__file__).resolve().parent.parent
D17 = D18.parent / "Onion_Deep_Revision_20261003_v17"
OLD = D17 / "manuscript_revision"
OUT = D18 / "manuscript_revision"
TITLE = "Attributing related-view gains in plant image evaluation with single-source and joint replacement controls"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def copy_exact(source, dest, ledger):
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, dest)
    assert sha(source) == sha(dest)
    ledger.append(dict(source=str(source), destination=str(dest.relative_to(OUT)) if dest.is_relative_to(OUT) else str(dest),
                       bytes=dest.stat().st_size, sha256=sha(dest), mode="byte-identical copy"))


def table(frame):
    return "| " + " | ".join(frame.columns) + " |\n| " + " | ".join(["---"]*len(frame.columns)) + " |\n" + "\n".join(
        "| " + " | ".join(str(v) for v in row) + " |" for row in frame.itertuples(index=False, name=None)) + "\n"


def main():
    OUT.mkdir(exist_ok=True)
    tables = OUT / "supplement_tables"
    ledger = []
    old_tables = sorted((OLD / "supplement_tables").glob("*.csv"))
    assert len(old_tables) == 18
    for folder in ("supplement_tables", "supplement_sources", "supplement_provenance"):
        for p in sorted((OLD / folder).rglob("*")):
            if p.is_file():
                copy_exact(p, OUT / folder / p.relative_to(OLD / folder), ledger)
    copy_exact(OLD / "SUPPLEMENT_BUILD.json", OUT / "SUPPLEMENT_BUILD_v17_RETAINED.json", ledger)
    for p in sorted((D17 / "journal_figures").glob("Fig_S1_Matched_Replacement_Binary*")):
        if p.is_file(): copy_exact(p, D18 / "figures" / p.name, ledger)
    for p in sorted((D17 / "journal_figures").glob("Fig_3_External_Context*")):
        if p.is_file(): copy_exact(p, D18 / "figures" / p.name.replace("Fig_3_External_Context", "Fig_S2_External_Context"), ledger)
    new_table_specs = [
        ("joint_dinov2/{task}/split_summary.csv", "S16_{task}_dinov2_joint_summary.csv"),
        ("single_source/{task}/target_panel_contrasts.csv", "S17_{task}_single_target_panel_contrasts.csv"),
        ("single_source/{task}/target_class_split_means.csv", "S18_{task}_single_target_class_split_means.csv"),
        ("single_source/{task}/split_means.csv", "S19_{task}_single_split_means.csv"),
        ("single_source/{task}/split_summary.csv", "S20_{task}_single_summary.csv"),
        ("joint_dinov2/{task}/arm_contrasts.csv", "S28_{task}_dinov2_joint_arm_contrasts.csv"),
        ("joint_dinov2/{task}/split_means.csv", "S29_{task}_dinov2_joint_split_means.csv")]
    for task in ("onion", "potato"):
        for src, dest in new_table_specs:
            copy_exact(D18 / src.format(task=task), tables / dest.format(task=task), ledger)
        for experiment in ("single_source", "joint_dinov2"):
            for name in ("protocol_freeze.json", "completion.json"):
                copy_exact(D18 / experiment / task / name, OUT / "supplement_provenance" / f"v18_{task}_{experiment}_{name}", ledger)
    for src, dest in [("target_contrasts", "S21_context_all_target_contrasts"),
                      ("class_split_means", "S22_context_target_class_split_means"),
                      ("split_means", "S23_context_split_means"), ("summary", "S24_context_summary")]:
        copy_exact(D18 / "context_sensitivity" / (src + ".csv"), tables / (dest + ".csv"), ledger)
    for src, dest in [("Fig_2_Joint_Replacement_plot_source.csv", "S25_joint_all_five_models_plot_source.csv"),
                      ("Fig_3_Single_Source_Replacement_plot_source.csv", "S26_single_all_five_models_plot_source.csv")]:
        copy_exact(D18 / "figures" / src, tables / dest, ledger)
    for name in ("DINOV2_PRE_EXTRACTION_FREEZE.json", "DINOV2_EXTRACTION_COMPLETE.json", "DINOV2_OFFICIAL_SOURCE_BLOB_VERIFICATION.json", "DINOV2_TRANSFORM_EQUIVALENCE_QA.json"):
        copy_exact(D18 / "features" / name, OUT / "supplement_provenance" / name, ledger)
    for name in ("ONION_JOINT_RESULT_REVIEW.json", "POTATO_JOINT_RESULT_REVIEW.json", "ONION_SINGLE_SOURCE_RESULT_REVIEW.json", "POTATO_SINGLE_SOURCE_RESULT_REVIEW.json", "FEATURE_QA_BINDING_REVIEW.json"):
        copy_exact(D18 / "independent_review" / name, OUT / "supplement_provenance" / name, ledger)
    context = pd.read_csv(D18 / "context_sensitivity/summary.csv")
    context = context[context.weighting.eq("all_original_classes")]
    model_labels = {"resnet_logit":"ResNet18 + LR", "colour_logit":"Colour + LR", "dinov2_logit":"DINOv2 + LR",
                    "resnet_cosine_1nn":"ResNet18 cosine 1NN", "dinov2_cosine_1nn":"DINOv2 cosine 1NN"}
    show = []
    for (dataset, model), part in context.groupby(["dataset", "model"]):
        p = part.set_index("metric")
        show.append(dict(task=dataset, model=model_labels[model], single_pp=100*p.loc["single_E_minus_H", "mean"],
                         joint_pp=100*p.loc["joint_E_minus_H", "mean"], single_minus_joint_pp=100*p.loc["single_minus_joint", "mean"],
                         difference_min_pp=100*p.loc["single_minus_joint", "split_min"], difference_max_pp=100*p.loc["single_minus_joint", "split_max"]))
    display = pd.DataFrame(show)
    display.to_csv(tables / "S27_context_display.csv", index=False)
    display_text = display.copy()
    for col in display_text.columns[2:]: display_text[col] = display_text[col].map(lambda x: f"{x:.4f}")
    display_table = table(display_text)
    en = (OLD / "Online_Resource_1_EN_v1.7.md").read_text(encoding="utf-8")
    zh = (OLD / "Online_Resource_1_ZH_v1.7.md").read_text(encoding="utf-8")
    en = en.replace("Distinguishing related-image exposure from ordinary training-sample replacement in plant image evaluation", TITLE)
    zh = zh.replace("植物图像评估中相关图像暴露与普通训练样本替换的区分", "采用单来源与联合替换对照归因植物图像评估中的相关视图收益")
    en = en.replace("research revision v1.7", "research revision v1.8; Sections S1-S6 retain the preceding evidence, with the new representation and single-source extensions specified in S7")
    zh = zh.replace("本文件为v1.7研究阅读稿的可编辑补充源稿", "本文件为v1.8研究阅读稿的可编辑补充源稿；S1-S6保留此前完整证据，新增表示与单来源扩展见S7")
    en = en.replace("Tables S1 retain every saved model, endpoint, panel, stratum and outcome.", "Tables S1 retain every saved original three-model endpoint, panel, stratum and outcome; the new DINOv2 results are added in S7.")
    zh = zh.replace("表S1保留所有模型、端点、面板、分层与指标", "表S1保留原始三个模型的全部端点、面板、分层与指标；新DINOv2结果见S7")
    en = en.replace("The two matched tasks produced 5,040", "The retained original three-model, two-endpoint joint analysis produced 5,040")
    zh = zh.replace("两任务共完成5,040", "保留的原始三模型、两端点联合分析共完成5,040")
    en = en.replace("main Fig. 3", "Fig. S2")
    zh = zh.replace("主图3", "补图S2")
    en = en.replace("../journal_figures/", "../figures/").replace("for every model and both tasks", "for all three original models and both tasks")
    zh = zh.replace("../journal_figures/", "../figures/").replace("它呈现全部模型、两任务", "它呈现原始三个模型、两任务")
    en = en.replace("SUPPLEMENT_BUILD.json", "SUPPLEMENT_BUILD_v17_RETAINED.json")
    zh = zh.replace("SUPPLEMENT_BUILD.json", "SUPPLEMENT_BUILD_v17_RETAINED.json")
    en = en.replace("This is the editable supplementary source for research revision v1.8; Sections S1-S6 retain the preceding evidence, with the new representation and single-source extensions specified in S7. It accompanies the English and Chinese reading manuscripts; the final journal upload format has not been certified.", "Sections S1-S6 document source identity, the original replacement controls and the retained transfer analyses. Section S7 specifies the frozen-representation and single-source extensions.")
    zh = zh.replace("本文件为v1.8研究阅读稿的可编辑补充源稿；S1-S6保留此前完整证据，新增表示与单来源扩展见S7，尚未认证为期刊最终上传文件。", "S1-S6记录来源身份、原始替换对照及保留的迁移分析；S7说明冻结表示与单来源扩展。")
    public_url = "https://github.com/hezejuyede/Pattern_Analysis_and_Applications_Onion_Reliability_v1.0/tree/v1.8.0/v1.8"
    en = en.replace("The top-level REPRODUCIBILITY_README.md supplies actual commands and distinguishes portable cached-input checks from historical-path dependencies and original-image acquisition. Public tag v1.2.1 is an earlier correction notice and does not yet archive this revision's new experiments.", "Release v1.8.0 archives the current replacement experiments, frozen numerical inputs, predictions, code and verification records at " + public_url + ". The release README provides portable commands under code/portable; supplementary_context documents the retained regional, transfer and calibration analysis history. Original photographs must be obtained from their source archives under the recorded upstream terms. The portable smoke check covers replacement and single-source reconstruction with selected refits; it does not repeat all regional or calibration training. These limits are stated separately from the original experiment logs and the larger independent computational audit.")
    zh = zh.replace("顶层REPRODUCIBILITY_README.md给出实际命令，并区分D盘缓存可读取验证、历史路径依赖和原图下载。公开v1.2.1为先前纠错说明，尚不代表本次新实验已存档。", "v1.8.0发布版本存档了当前替换实验、冻结数值输入、预测、代码及核验记录，固定地址为 " + public_url + "。发布版README给出code/portable下的可移植命令；supplementary_context说明保留的区域、迁移与校准分析历史。原始照片需按记录的上游条款从来源档案获取。本轮可移植烟测覆盖替换及单来源结果重建和指定组合的重训，并未重复全部区域或校准训练；这一范围与原始实验日志及较大规模独立计算审计分别说明。")
    en = en.replace("This source document does not certify a final journal upload, public release, author approval or editorial outcome.", "The original frozen scripts preserve the historical computation; the release's portable entry points resolve those paths through a recorded mapping to distributed numerical inputs.")
    zh = zh.replace("本源稿不认证最终上传、公开发布、作者已审阅或编辑结果。", "原始冻结脚本保留历史计算过程；发布版可移植入口通过记录的映射将历史路径解析到随附数值输入。")
    insertion_en = (D18 / "supplement_v18_section_EN.txt").read_text(encoding="utf-8").replace("{{CONTEXT_TABLE}}", display_table)
    insertion_zh = (D18 / "supplement_v18_section_ZH.txt").read_text(encoding="utf-8").replace("{{CONTEXT_TABLE}}", display_table)
    en = en.replace("## Supplementary method references", insertion_en + "\n## Supplementary method references")
    zh = zh.replace("## 补充方法参考文献", insertion_zh + "\n## 补充方法参考文献")
    index_records = []
    for p in sorted(tables.glob("*.csv")):
        index_records.append(dict(file="supplement_tables/"+p.name, rows=len(pd.read_csv(p)), bytes=p.stat().st_size, sha256=sha(p)))
    index = "\n\n".join(f"- `{r['file']}`: {r['rows']} rows" for r in index_records)
    en = en.split("## Complete machine-readable table index")[0] + "## Complete machine-readable table index\n\n" + index + "\n"
    zh = zh.split("## 完整机器可读数表索引")[0] + "## 完整机器可读数表索引\n\n" + index + "\n"
    (OUT / "Online_Resource_1_EN_v1.8.md").write_text(en, encoding="utf-8")
    (OUT / "Online_Resource_1_ZH_v1.8.md").write_text(zh, encoding="utf-8")
    pd.DataFrame(index_records).to_csv(OUT / "SUPPLEMENT_TABLE_INDEX.csv", index=False)
    records = []
    for task in ("onion", "potato"):
        for design in ("single_source", "joint_dinov2"):
            folder = D18 / design / task
            for name in ("predictions.csv.gz", "fit_log.csv", "support_checks.csv", "protocol_freeze.json", "completion.json", "targets.csv", "training_membership.csv.gz", "component_membership.csv", "allocation_membership.csv.gz"):
                p = folder / name
                if p.exists(): records.append(dict(task=task, design=design, file="../"+str(p.relative_to(D18)).replace("\\","/"), bytes=p.stat().st_size, sha256=sha(p)))
    pd.DataFrame(records).to_csv(OUT / "V18_COMPLETE_RESULT_FILE_INDEX.csv", index=False)
    qa = dict(status="PRESERVED_V17_AND_EXTENDED_V18_SOURCE", retained_original_csv_count=18, total_csv_count=len(index_records),
              title=TITLE, copied_files=ledger, tables=index_records, derived_tables=[dict(file="supplement_tables/S27_context_display.csv", sha256=sha(tables / "S27_context_display.csv"), source="context_sensitivity/summary.csv; all classes, same first ten seeds")],
              source_documents={p.name:sha(p) for p in (OUT/"Online_Resource_1_EN_v1.8.md", OUT/"Online_Resource_1_ZH_v1.8.md")},
              scope="Editable supplement sources, byte-preserved legacy evidence, exact new results and file indexes. Does not certify native DOCX/PDF layout, public release, author approval or acceptance.")
    (OUT / "SUPPLEMENT_BUILD_v18.json").write_text(json.dumps(qa, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(dict(status=qa["status"], tables=len(index_records), retained_v17_tables=18)))


if __name__ == "__main__":
    main()
