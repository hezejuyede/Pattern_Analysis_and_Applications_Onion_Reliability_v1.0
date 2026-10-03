"""Preserve all v1.8 supplement evidence and add explicit post-result S8."""
from pathlib import Path
import hashlib
import json
import shutil
import pandas as pd

D19=Path(__file__).resolve().parent.parent
D18=D19.parent/'Onion_Deep_Revision_20261003_v18'
OLD=D18/'manuscript_revision';OUT=D19/'manuscript_revision'
NEW=D19/'additional_analysis'
RELEASE=D19/'public_release_staging/v1.9'
URL='https://github.com/hezejuyede/Pattern_Analysis_and_Applications_Onion_Reliability_v1.0/tree/v1.9.0/v1.9'
NAMES=[(40,'source_target_support'),(41,'unique_target_frequency'),(42,'cross_split_overlap'),(43,'single_per_class_split_recall_headroom'),(44,'joint_per_class_split_recall_headroom'),(45,'single_event_counts_not_independent_samples'),(46,'joint_repeated_context_event_counts'),(47,'joint_repeated_context_confusion_records'),(48,'all_common_loco_split_results'),(49,'all_common_loco_summary'),(50,'frequency_weighted_split_results'),(51,'frequency_weighted_summary'),(52,'headroom_interpretation_percentage_units'),(53,'conditional_allocation_mcse')]
MODELS={'resnet_logit':'ResNet18 + LR','dinov2_logit':'DINOv2 + LR','colour_logit':'Colour + LR','resnet_cosine_1nn':'ResNet18 1NN','dinov2_cosine_1nn':'DINOv2 1NN'}

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def copy(src,dest):
    dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dest);assert sha(src)==sha(dest)
def table(f):
    return '| '+' | '.join(f.columns)+' |\n| '+' | '.join(['---']*len(f.columns))+' |\n'+'\n'.join('| '+' | '.join(map(str,r))+' |' for r in f.itertuples(index=False,name=None))+'\n'

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    for folder in ('supplement_tables','supplement_sources','supplement_provenance'):
        for p in sorted((OLD/folder).rglob('*')):
            if p.is_file():copy(p,OUT/folder/p.relative_to(OLD/folder))
    for src,dest in [('SUPPLEMENT_BUILD_v17_RETAINED.json','SUPPLEMENT_BUILD_v17_RETAINED.json'),('SUPPLEMENT_BUILD_v18.json','SUPPLEMENT_BUILD_v18_RETAINED.json'),('V18_COMPLETE_RESULT_FILE_INDEX.csv','V18_COMPLETE_RESULT_FILE_INDEX.csv')]:copy(OLD/src,OUT/dest)
    assert len(list((OLD/'supplement_tables').glob('*.csv')))==39
    rows=[]
    for number,name in NAMES:
        src=NEW/(name+'.csv');dest=OUT/'supplement_tables'/f'S{number}_{name}.csv'
        copy(src,dest)
        rows.append(dict(table=f'S{number}',source=str(src),file='supplement_tables/'+dest.name,rows=len(pd.read_csv(src)),bytes=dest.stat().st_size,sha256=sha(dest)))
    for name in ('ADDITIONAL_ANALYSIS_FREEZE.json','READ_ONLY_SUMMARY_QA.json'):copy(NEW/name,OUT/'supplement_provenance'/name)
    copy(D19/'portable_additional_verification/ADDITIONAL_PORTABLE_VERIFICATION.json',OUT/'supplement_provenance/ADDITIONAL_PORTABLE_VERIFICATION.json')
    support=pd.read_csv(NEW/'source_target_support.csv')
    labels={'healthy':'Healthy','iris_yellow_virus':'IYSV-labelled','purple_blotch':'Purple blotch','stemphylium_colletotrichum_leaf_blight':'Pooled leaf blight','Potato___Early_blight':'Early blight','Potato___Late_blight':'Late blight'}
    support=support[['dataset','label','manifest_source_units','unique_target_units','target_within_split_records','probe_count_each_split_min','sentinel_count_each_split_min']].copy()
    support.dataset=support.dataset.str.title();support.label=support.label.map(labels)
    support.columns=['Task','Original class','Sources','Unique targets','Target records','Probes','Sentinels']
    hd=pd.read_csv(NEW/'headroom_interpretation_percentage_units.csv')
    hd['Task/design']=hd.dataset.str.title()+' / '+hd.design
    hd['Model']=hd.model.map(MODELS)
    mapping={'sham_error_headroom_percent':'Available error (%)','corrected_pp':'Corrected (pp)','harmed_pp':'Harmed (pp)','net_contrast_pp':'Net (pp)','corrected_over_sham_error_percent':'Corrected/error (%)','net_contrast_over_sham_error_percent':'Net/error (%)'}
    hd=hd[['Task/design','Model']+list(mapping)].rename(columns=mapping)
    for c in hd.columns[2:]:hd[c]=hd[c].map(lambda x:f'{x:.4f}')
    mc=pd.read_csv(NEW/'conditional_allocation_mcse.csv')
    mc['Task/design']=mc.dataset.str.title()+' / '+mc.design;mc['Model']=mc.model.map(MODELS)
    mc=mc[['Task/design','Model','split_count','mean_pp','split_sd_pp','mcse_pp']].rename(columns={'split_count':'Splits','mean_pp':'Mean (pp)','split_sd_pp':'Split SD (pp)','mcse_pp':'MCSE (pp)'})
    for c in mc.columns[3:]:mc[c]=mc[c].map(lambda x:f'{x:.4f}')
    mapping=json.loads((RELEASE/'provenance/path_mapping.json').read_text(encoding='utf-8'))
    locations=[]
    for r in pd.read_csv(OLD/'V18_COMPLETE_RESULT_FILE_INDEX.csv').itertuples(index=False):
        original=(OLD/r.file).resolve();key=str(original).replace('\\','/')
        mapped=mapping[key];assert mapped['original_sha256']==r.sha256
        distributed=RELEASE/mapped['relative_path'];assert distributed.is_file() and sha(distributed)==mapped['distributed_sha256']
        locations.append(dict(task=r.task,design=r.design,historical_local_source=key,original_sha256=r.sha256,release_relative_path=mapped['relative_path'],distributed_sha256=mapped['distributed_sha256'],release_root_url=URL))
    pd.DataFrame(locations).to_csv(OUT/'V19_RETAINED_RESULT_LOCATIONS.csv',index=False)
    refs='\n\n[S3] Owen AB (2013) Monte Carlo theory, methods and examples. Chapters 1-2: Introduction and Simple Monte Carlo. Author-maintained electronic book. https://artowen.su.domains/mc/\n\n[S4] Cameron AC, Gelbach JB, Miller DL (2011) Robust Inference With Multiway Clustering. Journal of Business & Economic Statistics 29:238-249. https://doi.org/10.1198/jbes.2010.07136\n\n[S5] Bengio Y, Grandvalet Y (2004) No Unbiased Estimator of the Variance of K-Fold Cross-Validation. Journal of Machine Learning Research 5:1089-1105. https://www.jmlr.org/papers/v5/grandvalet04a.html\n\n'
    source_hashes={}
    for lang in ('EN','ZH'):
        s=(OLD/f'Online_Resource_1_{lang}_v1.8.md').read_text(encoding='utf-8')
        s=s.replace('tree/v1.8.0/v1.8','tree/v1.9.0/v1.9').replace('Release v1.8.0','Release v1.9.0').replace('v1.8.0发布版本','v1.9.0发布版本')
        s=s.replace('SUPPLEMENT_BUILD_v18.json','SUPPLEMENT_BUILD_v18_RETAINED.json')
        if lang=='EN':
            s=s.replace('Section S7 specifies the frozen-representation and single-source extensions.','Section S7 specifies the frozen-representation and single-source extensions; S8 adds post-result descriptive support, weighting and error-headroom analyses.')
            old="All paths in that index are relative to this supplement."
            s=s.replace(old,"That byte-preserved index records the historical v1.8 layout. V19_RETAINED_RESULT_LOCATIONS.csv maps each original file and hash to its actual object relative to the v1.9 public release root, including lossless transport formats.")
            s=s.replace("The single-source reproduction instructions are `../single_source/REPRODUCTION_AND_ESTIMANDS.txt`. Source scripts are `../code/run_single_source_controls.py`, `../code/run_dinov2_joint_controls.py` and `../code/compare_single_joint_context.py`.","The public release preserves the original experimental scripts and their hashes under code/original. Use code/portable/reproduce.py for the retained replacement analyses and code/portable/reproduce_additional.py for S8; the release README specifies commands and verification scope. Historical scripts are provenance, not a claim of path-independent execution.")
            refheading='## Supplementary method references';indexheading='## Complete machine-readable table index'
        else:
            s=s.replace('S7说明冻结表示与单来源扩展。','S7说明冻结表示与单来源扩展；S8新增结果已知后的来源支持、权重及错误余量描述分析。')
            # Replace the complete original file-location paragraph by its updated provenance equivalent.
            for block in s.split('\n\n'):
                if 'SUPPLEMENT_TABLE_INDEX.csv' in block:
                    replacement='`SUPPLEMENT_TABLE_INDEX.csv`给出全部补充数表数量、行数、大小及哈希。`V18_COMPLETE_RESULT_FILE_INDEX.csv`按原字节保留v1.8历史布局，索引完整预测、拟合日志、分配/来源成员和冻结记录，不在数表中重复大型预测。`V19_RETAINED_RESULT_LOCATIONS.csv`将每个原文件和哈希映射到v1.9公开发布根目录下的实际对象，包含无损运输格式。`SUPPLEMENT_BUILD_v18_RETAINED.json`保留此前18个CSV及扩展证据的构建记录。原始时间线与历史路径仍保留在冻结JSON中，不改写成始终可移植。发布版code/original保留原实验脚本与哈希；先前替换分析使用code/portable/reproduce.py，S8使用code/portable/reproduce_additional.py，README给出命令与核验范围。历史脚本是来源证据，不代表可不依赖原路径直接运行。'
                    s=s.replace(block,replacement)
            refheading='## 补充方法参考文献';indexheading='## 完整机器可读数表索引'
        sec=(D19/f'supplement_v19_section_{lang}.txt').read_text(encoding='utf-8').replace('{{SUPPORT_TABLE}}',table(support)).replace('{{HEADROOM_TABLE}}',table(hd)).replace('{{MCSE_TABLE}}',table(mc))
        s=s.replace(refheading,sec+'\n'+refheading)
        pre=s.split(indexheading)[0].rstrip()+refs
        records=[]
        for p in sorted((OUT/'supplement_tables').glob('*.csv')):records.append(dict(file='supplement_tables/'+p.name,rows=len(pd.read_csv(p)),bytes=p.stat().st_size,sha256=sha(p)))
        assert len(records)==53
        s=pre+indexheading+'\n\n'+'\n\n'.join(f"- `{r['file']}`: {r['rows']} rows" for r in records)+'\n'
        dest=OUT/f'Online_Resource_1_{lang}_v1.9.md';dest.write_text(s,encoding='utf-8');source_hashes[dest.name]=sha(dest)
    pd.DataFrame(records).to_csv(OUT/'SUPPLEMENT_TABLE_INDEX.csv',index=False)
    pd.DataFrame(rows).to_csv(OUT/'V19_ADDITIONAL_TABLE_INDEX.csv',index=False)
    retained=[]
    for old in sorted((OLD/'supplement_tables').glob('*.csv')):
        dest=OUT/'supplement_tables'/old.name;assert sha(old)==sha(dest)
        retained.append(dict(file=old.name,sha256=sha(old)))
    report=dict(status='PASS_PRESERVED_39_TABLES_AND_ADDED_14',retained_tables=retained,added_tables=rows,total_csvs=53,source_hashes=source_hashes,release_result_locations=28,chronology='Post-result descriptive extension; no new fits.',scope='Source assembly and file identity only; independent numerical replay and final PDF rendering are recorded separately.')
    (OUT/'SUPPLEMENT_BUILD_v19.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=report['status'],sources=source_hashes)))

if __name__=='__main__':main()
