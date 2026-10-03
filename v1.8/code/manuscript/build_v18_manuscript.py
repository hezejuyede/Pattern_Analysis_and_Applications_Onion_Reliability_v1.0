"""Integrate reviewed v18 controlled experiments into the immutable v17 manuscript.

All numerical tables are constructed from independently checked output tables.
This builder does not certify editorial priority or submission readiness.
"""
from pathlib import Path
import json, re, hashlib, shutil
import pandas as pd

D = Path('D:/论文/SCI投稿汇总/Onion_Deep_Revision_20261003_v18')
OLD = D.parent/'Onion_Deep_Revision_20261003_v17'/'manuscript_revision'
OUT = D/'manuscript_revision'
OUT.mkdir(exist_ok=True)
MODELS = ['resnet_logit','dinov2_logit','colour_logit','resnet_cosine_1nn','dinov2_cosine_1nn']
NAMES = dict(zip(MODELS,['ResNet18 + LR','DINOv2 + LR','Colour + LR','ResNet18 1NN','DINOv2 1NN']))

def replace_once(text, old, new):
    assert text.count(old)==1, (old[:100],text.count(old))
    return text.replace(old,new,1)

def section(text, heading, new):
    start=text.index(heading)
    m=re.search(r'^##?##? ',text[start+len(heading):],re.M)
    # Explicit level: replace only until the next heading of the same level or higher.
    level=len(heading)-len(heading.lstrip('#'))
    m=re.search(r'^#{1,'+str(level)+r'} ',text[start+len(heading):],re.M)
    end=start+len(heading)+m.start() if m else len(text)
    return text[:start]+new.strip()+'\n\n'+text[end:]

def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,row))+' |' for row in rows])

oldba=pd.read_csv(OLD/'supplement_tables/S3_selected_probe_absolute_BA.csv')
print('old BA columns:',list(oldba))
rows=[]
for task in ['onion','potato']:
    newba=pd.read_csv(D/'independent_review'/f'joint_results_{task}'/'independent_selected_baseline_summary.csv')
    for model in MODELS:
        if model.startswith('dinov2'):
            r=newba.set_index('model').loc[model]
            vals=[r[x] for x in ['zero_BA','sham_BA','exposure_BA','E_minus_H']]
        else:
            r=oldba[(oldba.dataset==task)&(oldba.model==model)&(oldba.endpoint=='multiclass')].iloc[0]
            # v17 independent source uses these raw BA columns, verified below.
            vals=[r[x] for x in ['zero_BA_mean','sham_BA_mean','exposure_BA_mean','exposure_minus_sham_BA_mean']]
        rows.append([task.title(),NAMES[model]]+[f'{100*x:.2f}' for x in vals])
jointtable=table(['Task','Model','Zero BA (%)','Sham BA (%)','Exposure BA (%)','E−H (pp)'],rows)
pd.DataFrame(rows,columns=['task','model','zero_BA_percent','sham_BA_percent','exposure_BA_percent','difference_pp']).to_csv(OUT/'Table_2_display.csv',index=False)

single_rows=[]
context=pd.read_csv(D/'context_sensitivity'/'summary.csv')
print('context columns:',list(context))
for task in ['onion','potato']:
    ss=pd.read_csv(D/'single_source'/task/'split_summary.csv')
    for model in MODELS:
        sub=ss[(ss.model==model)&(ss.panel=='own_target')&(ss.metric=='policy_BA_difference')].set_index('weighting')
        # Values for same-target joint are independently reconstructed below, rather than use 30-seed joint means.
        c=context[(context.dataset==task)&(context.model==model)&(context.weighting=='all_original_classes')].set_index('metric')
        single_rows.append([task.title(),NAMES[model],f"{100*sub.loc['all_original_classes','mean']:.2f}",f"{100*sub.loc['common_classes_n_ge_30','mean']:.2f}",f"{100*c.loc['joint_E_minus_H','mean']:.2f}",f"{100*c.loc['single_minus_joint','mean']:.2f}"])
singletable=table(['Task','Model','Single: all (pp)','Single: ≥30 (pp)','Joint: same 10 (pp)','Single−joint (pp)'],single_rows)
pd.DataFrame(single_rows,columns=['task','model','single_all_pp','single_common_pp','matched_joint_pp','context_difference_pp']).to_csv(OUT/'Table_3_display.csv',index=False)

en=(OLD/'full_text.md').read_text(encoding='utf-8')
zh=(OLD/'full_text_zh.md').read_text(encoding='utf-8')
en=replace_once(en,en.splitlines()[0],'# Attributing related-view gains in plant image evaluation with single-source and joint replacement controls')
zh=replace_once(zh,zh.splitlines()[0],'# 利用单来源与联合替换对照区分植物图像评估中的相关视图增益')
en=section(en,'## Abstract',r'''## Abstract

Related test views can benefit from shared-source training images, but ordinary training-sample replacement also changes predictions. We separate these responses using matched deletions followed by either related views or unrelated same-class sources. The study combines joint replacement with a single-source intervention that holds every other training observation fixed. Two public archives provide 481 audited onion components and 538 author-identified potato leaves. Five fixed classifiers include ResNet18 and DINOv2 representations, colour features and nearest-neighbour controls. Across ten predetermined splits, single-source replacement improved class-balanced target correctness by 25.19 percentage points for ResNet18 logistic regression and 18.49 points for DINOv2 in onion, compared with 2.48 and 2.83 points in potato. Restricting evaluation weight to onion classes with at least 30 sources reduced the corresponding contrasts to 13.59 and 11.32 points; no retraining or target deletion was involved. Joint replacement across 30 splits also showed a larger local advantage in onion. However, never-exposed predictions changed under both policies, and single-versus-joint contrasts on identical targets varied in sign across splits. The evidence supports a local advantage conditional on the fitted representation and archive, while prediction turnover alone does not specifically identify shared-source exposure. Matched single-source controls strengthen attribution without establishing a universal leakage magnitude, whole-network memorization or field diagnostic validity.

**Keywords:** related images; data leakage; training-sample replacement; prediction stability; grouped evaluation; plant image classification''')
zh=section(zh,'## 摘要',r'''## 摘要

测试图像可能因训练集中存在同源视图而获益，但普通训练样本替换也会改变预测。本文在删除相同训练来源后，分别插入相关视图或同类无关联来源，以区分这两种响应；同时结合联合替换与每次仅改变一个来源、保持其他训练观测不变的干预。两个公开档案提供481个经审计的洋葱组件和538片由原作者记录身份的马铃薯叶片。五个固定分类器包括ResNet18与DINOv2表征、颜色特征及最近邻对照。在预先确定的十次划分中，单来源替换使洋葱目标样本的类别平衡正确率相对普通替换提高25.19个百分点（ResNet18逻辑回归）和18.49个百分点（DINOv2），马铃薯相应提高2.48和2.83个百分点。仅对至少含30个来源的洋葱类别赋予评价权重时，相应差值降至13.59和11.32个百分点；该分析没有重新训练或删除目标记录。三十次划分的联合替换同样显示洋葱具有较大的局部优势。然而，两种策略都会改变从未暴露来源上的预测，而且相同目标上的单来源与联合替换差值在不同划分中有正有负。现有证据支持以拟合表征和档案为条件的局部优势，而预测转换本身不能特异性识别同源暴露。匹配单来源对照加强了归因，但不建立普适泄漏幅度、整网记忆机制或田间诊断有效性。

**关键词：** 相关图像；数据泄漏；训练样本替换；预测稳定性；分组评估；植物图像分类''')

old='The policies change multiple sources simultaneously, so their contrast does not identify an isolated individual direct effect.'
en=replace_once(en,old,'The joint policies change several sources together. We therefore add a single-source comparison: only the donor assigned to one probe is replaced, while every other training observation remains identical. This intervention separates the response of that target from changes caused by replacing the rest of the selected sources.')
old='两种策略同时改变多个来源，因此其差值不能识别单个来源孤立的直接效应。'
if old not in zh:
    candidates=[p for p in zh.split('\n\n') if '多个来源' in p and '### 2.2' not in p]
    print('ZH joint paragraph candidates', candidates[:1])
    # Match the precise sentence from the source without changing unrelated methods.
    zh=re.sub(r'这些策略同时改变多个来源[^。]*。','因此，进一步加入单来源比较：每次仅替换一个探针对应的供体，其余训练观测全部保持相同。该干预将目标自身的响应与替换其他受选来源导致的变化区分开。',zh,count=1)
else:
    zh=replace_once(zh,old,'因此，进一步加入单来源比较：每次仅替换一个探针对应的供体，其余训练观测全部保持相同。该干预将目标自身的响应与替换其他受选来源导致的变化区分开。')
en=en.replace('The contribution is a reproducible empirical test of attribution under matched replacement, including its negative and task-dependent results.','The contribution is a controlled reanalysis of attribution: it measures the local response under a fixed training background, tests its sensitivity to joint replacement and class weighting, and compares that response with changes on never-exposed sources. These are evaluation findings, not a new classifier or a claim of improved disease recognition.')
zh=re.sub(r'本文的贡献是[^。]*。','本文贡献是对归因问题的受控再分析：在固定训练背景下测量局部响应，检验其对联合替换及类别权重的敏感性，并与从未暴露来源上的变化比较。这些是评估研究的发现，不是新的分类器，也不声称改进了病害识别。',zh,count=1)

newen=r'''### 2.3 Representations and fitting

The reference representation was a frozen ImageNet-pretrained ResNet18 with 512 features \cite{He2016ResNet,Deng2009ImageNet}. A standardized, class-balanced L2 logistic classifier used fixed regularization C = 0.01, the lbfgs solver and at most 10,000 iterations. Scaling was fitted separately inside each training arm. The modest fitted head avoids training a high-capacity network on a few hundred source units.

Before the new fits, we specified one modern representation sensitivity: the official DINOv2 ViT-S/14 checkpoint \cite{Oquab2024DINOv2}. Its normalized class token supplied 384 features after the official short-side 256-pixel bicubic resize, 224-pixel centre crop and ImageNet normalization. The backbone remained frozen. The identical logistic specification was applied to these features; no checkpoint or hyperparameter was selected on either task's outcomes. Checkpoint and source hashes, image-byte checks for all 5,228 files, and exact preprocessing and repeated-extraction checks accompany the outputs. The membership of these archives in DINOv2 pretraining cannot be verified; the experiment controls our fitted heads and training sets, not unknown foundation-model pretraining overlap.

A colour comparator used 189 histogram and summary-statistic features across BGR, HSV and Lab channels with the same logistic specification. Untuned cosine nearest-neighbour comparators used the ResNet18 and DINOv2 representations; deterministic row order resolved ties. All five models used identical sources, selected images and allocations. The original multiclass endpoint was primary. Earlier healthy-versus-other refits of ResNet18, colour and ResNet18 nearest-neighbour models remain secondary analyses; DINOv2 and the single-source experiment were specified for multiclass labels only. Complete records include all retained models, rather than a best-performing subset. Fitting used scikit-learn \cite{Pedregosa2011ScikitLearn}.

### 2.4 Outcomes and interpretation'''
# Insert the additional intervention before the outcomes section without renumbering original equations.
en=section(en,'### 2.3 Representations and fitting',newen.rsplit('\n\n### 2.4',1)[0])
zhr=r'''### 2.3 表征与模型拟合

参照表征为冻结的ImageNet预训练ResNet18，输出512维特征 \cite{He2016ResNet,Deng2009ImageNet}。逻辑分类器采用标准化、类别平衡和L2正则化，固定C = 0.01，使用lbfgs求解器，最多迭代10,000次。标准化在每个训练组内部单独拟合。使用规模适中的拟合分类头，避免在仅数百个来源单位上训练高容量网络。

在新拟合前，确定一种现代表征敏感性分析：官方DINOv2 ViT-S/14检查点 \cite{Oquab2024DINOv2}。采用官方短边256像素双三次缩放、224像素中心裁剪和ImageNet标准化流程，以归一化类别token产生384维特征，主干保持冻结。使用完全相同的逻辑回归设置，没有依据两任务结果选择检查点或超参数。随结果提供检查点与源码哈希、全部5,228个图像文件的字节校验，以及预处理等价性和重复提取校验。无法核实这些档案是否进入DINOv2预训练；本文控制自行拟合的分类头及训练集，不声称排除了未知的基础模型预训练重叠。

颜色对照使用BGR、HSV和Lab通道的189维直方图与统计特征，逻辑回归设置不变。未经调参的余弦最近邻对照分别使用ResNet18和DINOv2表征，按确定行顺序处理距离并列。五个模型使用完全相同的来源、图像与分配。原始多分类任务为主要终点。此前ResNet18、颜色和ResNet18最近邻模型的健康/其他标签二分类重拟合保留为次要分析；DINOv2和单来源实验在新拟合前已限定使用多分类标签。保留全部模型记录，没有根据表现择优取舍。拟合使用scikit-learn \cite{Pedregosa2011ScikitLearn}。'''
zh=section(zh,'### 2.3 表征与模型拟合',zhr)

single_methods_en='''For the single-source intervention, the first ten recorded seeds (20261003–20261012) and every paired probe within them were fixed before new outcomes were available. Starting from that split's zero-exposure set, we removed only the donor assigned to the current target. Exposure inserted that target's two non-anchor views; sham inserted its already assigned reserved alternative. All other training observations, class counts, image counts and evaluation anchors were identical. This produced 960 onion and 1,080 potato target-within-split interventions; repeated occurrences across overlapping splits are not new biological units. The training budgets remained 240 sources/480 files and 268 leaves/536 photographs. Actual memberships and image hashes were checked before fitting. Figure 1 distinguishes this fixed-background comparison from joint replacement.'''
single_methods_zh='''单来源干预在新结果可见前固定使用前十个已记录种子（20261003–20261012），并纳入其中全部成对探针。每次从该划分的零暴露训练集出发，仅删除当前目标对应的供体；暴露插入该目标的两张非锚点视图，sham插入预先指定的备用来源。其余训练观测、类别数量、图像数量和评估锚点全部相同。由此得到960个洋葱和1,080个马铃薯“目标—划分”干预；同一来源在重叠划分中重复出现，不是新增生物学单位。训练预算保持为240个来源/480张图像，以及268片叶/536张照片。拟合前核查实际成员关系与图像哈希。图1区分了这种固定背景比较与联合替换。'''
en=en.replace('\n### 2.3 Representations and fitting','\n'+single_methods_en+'\n\n### 2.3 Representations and fitting',1)
zh=zh.replace('\n### 2.3 表征与模型拟合','\n'+single_methods_zh+'\n\n### 2.3 表征与模型拟合',1)

out_en='''The single-source estimand uses each target's own exposure/sham fit pair. Its correctness difference is first averaged across targets within their original class, then equally across classes, and finally across the ten splits. It is an average over target-specific fitted models, not BA from one common model. We retain separately the fraction corrected relative to sham and the fraction harmed, whose difference gives the net contrast. Changes on all other probes and sentinels are evaluated with the same class weighting, averaging targets within their original class before averaging intervention classes. Ten-split minima, maxima and standard deviations describe the design variation.

A prespecified sensitivity gives evaluation weight only to original classes with at least 30 source units in the complete audited task. It excludes the 13-component onion purple-blotch class from the averaging weights and retains all potato classes. The full-class fits and all prediction records remain unchanged; this is a weighting sensitivity, not a retrained reduced task or a claim about population prevalence. Finally, each single-source target is matched to its joint-policy predictions in the same split, averaging the five joint allocations in which it was selected. This ten-split comparison estimates sensitivity to the surrounding replacement policy. We do not compare ten-split single-source means with unmatched 30-split joint means as though they were paired.'''
out_zh='''单来源估计量使用每个目标自身对应的暴露/sham拟合对。先在原始类别内部平均目标正确性差值，再对类别等权平均，最后平均十次划分。它是多个目标特定模型上的平均差值，不是一个共同模型的BA。分别保留相对sham由错变对与由对变错的比例，两者相减得到净差值。其他探针与哨兵变化采用相同的类别权重，先在干预目标原类内部平均，再对干预类别平均。十次划分的最小值、最大值和标准差描述设计变动。

预先规定的敏感性分析只对完整审计任务中至少含30个来源的原始类别赋予评价权重：洋葱的13组件紫斑类不参与加权平均，马铃薯所有类别均保留。完整类别拟合和全部预测记录保持不变；这是权重敏感性分析，不是重新训练缩减类别任务，也不代表总体患病率。最后，将每个单来源目标与同一划分中的联合策略预测配对，平均该目标被选中的五次联合分配。这一十划分比较衡量对其他来源替换背景的敏感性，不把十划分单来源均值与未配对的三十划分联合均值当作配对比较。'''
en=en.replace('\n### 2.5 Separate locked', '\n'+out_en+'\n\n### 2.5 Separate locked',1)
zh=zh.replace('\n### 2.5 单独锁定', '\n'+out_zh+'\n\n### 2.5 单独锁定',1)
en=en.replace('This was a post hoc study extension, not a prospectively registered field study.','The DINOv2 and single-source extensions were specified after the earlier joint results were known; their features, assignments, fixed models and computational checks were frozen before their new fits. These were post hoc study extensions, not a prospectively registered field study.')
zh=zh.replace('这是事后的研究扩展，不是前瞻性注册的田间研究。','DINOv2与单来源扩展是在此前联合结果已知后确定的；相应特征、分配、固定模型及计算核查规则在新拟合前冻结。这些均为事后研究扩展，不是前瞻性注册的田间研究。')

en=en.replace('all three models. The primary ResNet18','all five models. The reference ResNet18',1)
en=en.replace('The colour and nearest-neighbour comparators also had much larger mean contrasts in onion than in potato.','DINOv2 logistic regression gave corresponding contrasts of +21.18 and +1.93 points. Colour and both nearest-neighbour comparators also had larger mean contrasts in onion than in potato.',1)
zh=zh.replace('全部三个模型的多分类比较。主要ResNet18','全部五个模型的多分类比较。参照ResNet18',1)
zh=zh.replace('颜色特征和最近邻对照在洋葱上的平均差值也明显大于马铃薯。','DINOv2逻辑回归相应差值为+21.18和+1.93个百分点。颜色特征及两种最近邻对照在洋葱上的平均差值也大于马铃薯。',1)
en=re.sub(r'\| Task \| Model \| Zero BA.*?(?=\n\n)',lambda m:jointtable,en,flags=re.S,count=1)
zh=re.sub(r'\| 任务 \| 模型 \| Zero BA.*?(?=\n\n)',lambda m:jointtable,zh,flags=re.S,count=1)

result_en='''### 3.3 The local advantage persisted with a fixed training background

Single-source replacement yielded an average own-target correctness advantage of 25.19 percentage points for ResNet18 logistic regression and 18.49 points for DINOv2 in onion. The potato contrasts were 2.48 and 2.83 points. Figure 3 displays all five models and the ten individual split values; Table 3 separates class weighting from the matched background comparison. The observed single-source advantages were corrections relative to sham: no own-target correct-to-incorrect event under exposure relative to sham was recorded in these five-model interventions. This is a finite-sample observation, not a monotonicity guarantee for the classifiers.

When only classes with at least 30 sources received evaluation weight, the onion ResNet18 and DINOv2 contrasts fell to 13.59 and 11.32 points. Related-view gains therefore persisted outside the sparsely supported purple-blotch class, but that class materially affected their magnitude. The sensitivity retains the same multiclass fits; it cannot estimate the accuracy of a new three-class training problem.

On exactly the same ten splits and targets, the onion joint-policy contrasts were 23.34 points for ResNet18 and 17.69 for DINOv2, compared with the respective single-source values of 25.19 and 18.49. Across all five models and both tasks, the single-minus-joint split ranges crossed zero. Thus the fixed-background experiment supports a local advantage without requiring the other probes to enter training, but does not establish systematic amplification or attenuation by joint replacement. The colour model's mean onion contrast decreased from joint to single replacement, so a universal direction would also misdescribe the model comparison.

Single-source interventions changed never-exposed predictions as well. Mean onion sentinel exposure-minus-sham turnover was −0.39 points for ResNet18 and −0.33 for DINOv2, with split ranges spanning zero. These observations retain the distinction between a local advantage and more general sensitivity to training data. Full rescue, harm, probability-change, per-class and collateral outcomes are available in Online Resource 1.

**Table 3** Single-source own-target contrasts and their matched joint-policy comparison. All entries are percentage points and average the same first ten splits. “≥30” changes only evaluation weights to classes with at least 30 sources; potato retains all classes. “Joint: same 10” averages the five selected joint allocations for each identical target. The last column is paired within split before averaging; all corresponding ten-split ranges span zero

'''+singletable
result_zh='''### 3.3 固定训练背景后局部优势仍然存在

单来源替换使洋葱自身目标正确率相对普通替换平均提高25.19个百分点（ResNet18逻辑回归）和18.49个百分点（DINOv2），马铃薯相应提高2.48和2.83个百分点。图3展示全部五个模型及十个单独划分值，表3区分类别权重变化与配对背景比较。观测到的单来源优势来自相对sham的错误纠正：这五模型干预中，没有记录到暴露使自身目标从sham下预测正确变为预测错误的事件。这是有限样本观察，不是分类器的单调性保证。

仅对至少含30个来源的类别赋予评价权重后，洋葱ResNet18与DINOv2差值降至13.59和11.32个百分点。因此，相关视图增益在样本稀少的紫斑类之外仍存在，但该类别明显影响效应幅度。敏感性分析保留相同的多分类拟合，不能估计一个新三分类训练任务的准确率。

在完全相同的十次划分和目标上，洋葱联合策略差值为23.34个百分点（ResNet18）和17.69个百分点（DINOv2），对应单来源值为25.19和18.49。五个模型、两个任务的单来源减联合策略划分范围全部跨零。因此，固定背景实验支持无需其他探针进入训练集也能出现局部优势，但没有证实联合替换会系统性放大或减弱这一优势。颜色模型的洋葱平均差值从联合替换到单来源替换反而下降，不能宣称全部模型具有一致变化方向。

单来源干预也改变了从未暴露来源的预测。洋葱哨兵暴露减sham的平均正误转换率差值为−0.39个百分点（ResNet18）和−0.33个百分点（DINOv2），划分范围均跨零。这些观察继续支持将局部优势与训练数据改变引起的更一般敏感性分开。完整纠正、新增错误、概率变化、逐类别和其他样本变化结果见在线资源1。

**表3** 单来源自身目标差值及与配对联合策略的比较。单位均为百分点，平均相同的前十次划分。“≥30”仅将评价权重限制到至少含30个来源的类别，马铃薯保留全部类别。“Joint: same 10”对相同目标被选中的五次联合分配取平均。末列先在划分内部配对相减再取平均，相应十划分范围全部跨零

'''+singletable
en=en.replace('### 3.3 The separate onion transfer result remained a limitation',result_en+'\n\n### 3.4 The separate onion transfer result remained a limitation')
zh=zh.replace('### 3.3 单独的洋葱迁移结果仍构成限制',result_zh+'\n\n### 3.4 单独的洋葱迁移结果仍构成限制')
en=en.replace('Figure 3 reports both dependence sensitivities','Figure S2 in Online Resource 1 reports both dependence sensitivities')
zh=zh.replace('图3报告了两种依赖性敏感性结果','在线资源1的图S2报告了两种依赖性敏感性结果')

en=en.replace('our joint replacement contrast cannot recover such an individual causal quantity. This boundary is important because A and B alter multiple sources at once.','the single-source comparison fixes the other observations and identifies the response to the specified donor-to-related-view versus donor-to-alternative replacement within that training pipeline. It does not recover a background-independent property of the source. Moreover, influence measured through a frozen representation and fitted head is not whole-network memorization; Feldman and Zhang explicitly distinguish last-layer approximations from full-model effects. The paired joint comparison tests one change of background, not all possible training contexts.')
zh=zh.replace('本文的联合替换差值不能恢复此类个体因果量。由于A和B都会同时改变多个来源，这一边界必须保留。','本文单来源比较固定其余观测，识别的是特定训练流程中“供体换为相关视图”相对“供体换为备用来源”的响应，而不是与背景无关的来源固有属性。冻结表征与拟合分类头上的影响也不能称为整网记忆；Feldman和Zhang明确区分了末层近似与完整模型效应。配对联合比较检验的是一种背景变化，而不是所有训练情境。')
lit_en=r'''Recent controlled evaluations in this journal clarify the evidential role of such a study. Božič et al. compared seven unsupervised anomaly-detection methods across four tasks and multiple training-pollution levels \cite{Bozic2025Robustness}; Requena et al. examined label noise across eight multilabel tasks and several neighbour and prototype strategies \cite{Requena2026Noise}. Those studies varied a threat to evaluation under explicit controls rather than relying on a new application dataset alone. Our narrower question concerns shared-source insertion with matched donor deletion and an unchanged test panel. Single-source and joint replacement answer different attribution questions, and the weighting sensitivity exposes an otherwise hidden dependence on class support. The present two-task breadth is smaller than those comparisons; repeated fits cannot compensate for that population limitation. Its useful result is the controlled distinction among local advantage, background sensitivity and changes on never-exposed sources.

Larger exposure contrasts should not be read as better classification models. In onion, nearest neighbours and colour features were often more responsive to related-view insertion than either logistic deep-feature head. Because these models use identical replacements, that result demonstrates that the vulnerability does not require a complex learned head. DINOv2 changes the representation and pretraining regime while preserving the comparison; it supplies a modern sensitivity test, not a comprehensive state-of-the-art contest. All features and penalties remain fixed, so the empirical magnitudes are conditional on those choices.'''
lit_zh=r'''本刊近年的受控评估研究说明了这类研究应承担的证据任务。Božič等在四个任务和多个训练污染水平下比较七种无监督异常检测方法 \cite{Bozic2025Robustness}；Requena等在八个多标签任务中考察标签噪声及多种近邻、原型策略 \cite{Requena2026Noise}。这些研究通过明确对照改变评估威胁，并非仅将既有算法应用于另一个数据集。本文更窄的问题是：在匹配供体删除和固定测试组的条件下插入同源视图。单来源与联合替换回答不同的归因问题，权重敏感性则揭示效应量对类别支持的依赖。本文两个任务的覆盖面小于上述比较，重复拟合次数不能弥补总体覆盖范围有限这一事实。有用的结果是受控地区分局部优势、背景敏感性及从未暴露来源上的预测变化。

较大的暴露差值不应解释为更好的分类模型。在洋葱任务中，最近邻和颜色特征对相关视图插入的响应往往大于深度特征逻辑回归头。由于替换条件相同，该结果表明这种脆弱性不需要复杂的拟合分类头。DINOv2改变表征及预训练方案，同时保持实验比较一致，构成现代表征敏感性检验，而不是全面的最先进方法竞赛。所有特征与惩罚参数固定，因此经验效应量仍以这些选择为条件。'''
en=en.replace('\nThe provenance contrast is also informative.', '\n'+lit_en+'\n\nThe provenance contrast is also informative.',1)
zh=zh.replace('\n来源证据的差异也提供了信息。', '\n'+lit_zh+'\n\n来源证据的差异也提供了信息。',1)
en=en.replace('The complete per-class tables should accompany any interpretation of the overall result.','The common-class weighting sensitivity reduces this influence but does not create more rare-class sources. Ten overlapping splits and 2,040 target-within-split interventions are not independent acquisitions. Unknown overlap with DINOv2 pretraining also prevents treating its feature space as an independently trained replication.')
zh=zh.replace('解释总体结果时，应同时提供完整的逐类别表格。','常见类别权重敏感性减小了这种影响，但没有增加稀有类别来源。十个重叠划分和2,040个“目标—划分”干预不是独立采集。DINOv2预训练重叠未知，也不能将其特征空间视作独立训练的重复验证。')
en=section(en,'## 6 Conclusion','''## 6 Conclusion

Related-view insertion produced a local advantage over matched unrelated replacement even when every other training observation was held fixed. The effect was larger in the onion archive than in potato and smaller after reducing the weight of the sparsely supported onion class. Ordinary replacement also changed never-exposed predictions, while the matched single-versus-joint contrasts did not show a uniform direction across splits. A defensible attribution analysis should therefore report the fixed-background target response, the surrounding replacement policy and class support alongside aggregate scores. These controlled findings support more specific interpretation of related-image experiments; they do not establish a universal leakage magnitude or field diagnostic validity.''')
zh=section(zh,'## 6 结论','''## 6 结论

即使保持其余训练观测全部不变，相关视图插入相对匹配的无关联替换仍产生局部优势。该效应在洋葱档案中大于马铃薯；减小样本支持稀少的洋葱类别权重后，效应幅度下降。普通替换同样改变从未暴露来源上的预测，而配对的单来源与联合策略差值没有在各划分中表现出统一方向。因此，可信的归因分析应同时报告固定背景下的目标响应、其他来源的替换策略以及类别支持，而不能只报告总体分数。这些受控发现支持更具体地解释相关图像实验，不建立普适泄漏幅度或田间诊断有效性。''')

cap_en='''## Figure captions

**Fig. 1** Matched replacement at two scales. Joint policies replace 48 onion or 54 potato sources in one allocation. Single-source policies replace only the donor assigned to the current target and keep every other training observation identical. Both policies remove the same donor sources and insert two same-class images per source. Test anchors remain fixed and sentinels never share an audited or recorded source with training

**Fig. 2** Joint replacement contrasts for all five models in onion and potato. Selected-probe correctness is exposure minus sham; sentinel correctness turnover compares each policy with the same zero-exposure reference before taking exposure minus sham. Markers summarize 30 split means; bars show their 2.5th–97.5th percentiles, not population confidence intervals. Blue uses all original classes; orange changes evaluation weights to classes with at least 30 sources without refitting. Potato weights coincide

**Fig. 3** Single-source replacement with the remaining training background held fixed. Each target has its own pair of fitted models. Target effects are averaged within original class, across classes and then across ten predetermined splits. Individual split points and full minima–maxima are displayed; these are not confidence intervals and differ from the percentile bars in Fig. 2. Blue uses all classes; orange changes only evaluation weights to classes with at least 30 sources. Sentinel contrasts retain all classes. Larger own-target effects indicate greater response to exposure, not superior disease recognition

**Online Resource 1** Complete provenance, frozen allocations, all retained models and endpoints, single-source and matched joint comparisons, transfer sensitivities and reproduction instructions'''
cap_zh='''## 图注

**Fig. 1** 两种尺度的匹配替换。联合策略在一次分配中替换48个洋葱或54个马铃薯来源；单来源策略仅替换当前目标对应的供体，其余训练观测全部相同。配对策略删除相同供体，每个来源插入两张同类图像。测试锚点固定，哨兵始终不与训练集共享经审计或已记录的来源

**Fig. 2** 洋葱与马铃薯全部五模型的联合替换差值。被选探针正确率为暴露减sham；哨兵正误转换率先与同一零暴露参照比较，再计算暴露减sham。标记汇总30个划分均值，误差线为第2.5–97.5百分位数，不是总体置信区间。蓝色使用所有原类；橙色只改变评价权重，限定为至少含30个来源的类别，不重新拟合。马铃薯两种权重相同

**Fig. 3** 保持其他训练背景固定的单来源替换。每个目标对应自身的一对拟合模型，先在原始类别内部平均目标效应，再对类别平均，最后平均十个预定划分。图中显示各划分点和完整最小—最大范围，不是置信区间，与图2百分位误差线不同。蓝色使用所有类别；橙色仅将评价权重限定到至少含30个来源的类别。哨兵差值保留所有类别。自身目标效应越大表示对暴露的响应越强，不代表病害识别模型越好

**Online Resource 1** 完整来源记录、冻结分配、全部模型和终点、单来源与配对联合比较、迁移敏感性及复现说明'''
en=section(en,'## Figure captions',cap_en)
zh=section(zh,'## 图注',cap_zh)
# Cite the published PlantVillage analysis rather than duplicate the unaccepted repository preprint.
en=en.replace(',Hughes2015PlantVillage','')
zh=zh.replace(',Hughes2015PlantVillage','')
en=en.replace('no retraining or target deletion was involved','the original fits and all prediction records were retained')
zh=zh.replace('该分析没有重新训练或删除目标记录','该分析保留原有拟合及全部预测记录')
en=en.replace('Its normalized class token supplied 384 features','Its final LayerNorm CLS token supplied 384 features')
en=en.replace('all corresponding ten-split ranges span zero\n','all corresponding ten-split ranges span zero. Entries are rounded independently\n')
zh=zh.replace('相应十划分范围全部跨零\n','相应十划分范围全部跨零，各数值分别舍入显示\n')
zh=zh.replace('以归一化类别token产生384维特征','以末层LayerNorm后的CLS token产生384维特征')
en=en.replace('Figure 3 displays all five models and the ten individual split values','Figure 3 displays all five models and the ten-split variation')
zh=zh.replace('图3展示全部五个模型及十个单独划分值','图3展示全部五个模型及十划分变化范围')
en=en.replace('Individual split points and full minima–maxima are displayed','Means and full ten-split minima–maxima are displayed')
zh=zh.replace('图中显示各划分点和完整最小—最大范围','图中显示均值及完整的十划分最小—最大范围')
en=en.replace('Blue uses all classes; orange changes only evaluation weights to classes with at least 30 sources. Sentinel contrasts retain all classes.','Blue uses all classes; orange assigns weight only to classes with at least 30 sources, both for intervention targets and evaluated anchors, without refitting.')
zh=zh.replace('蓝色使用所有类别；橙色仅将评价权重限定到至少含30个来源的类别。哨兵差值保留所有类别。','蓝色使用所有类别；橙色只对至少含30个来源的类别赋予权重，同时作用于干预目标和评价锚点，不重新拟合。')
en=en.replace('The full-class fits and all prediction records remain unchanged; this is a weighting sensitivity, not a retrained reduced task or a claim about population prevalence.','For joint replacement, the restriction changes evaluated-anchor weights only. For single-source replacement, it changes both intervention-target weights and evaluated-anchor weights; within each retained intervention class, eligible targets are averaged before equal weighting across retained classes. The full-class fits and all prediction records remain unchanged. This is a weighting sensitivity, not a retrained reduced task or a claim about population prevalence.')
zh=zh.replace('完整类别拟合和全部预测记录保持不变；这是权重敏感性分析，不是重新训练缩减类别任务，也不代表总体患病率。','联合替换只改变评价锚点权重；单来源替换同时改变干预目标和评价锚点的类别权重，先在保留的干预类别内部平均合格目标，再对保留类别等权平均。完整类别拟合和全部预测记录保持不变。这是权重敏感性分析，不是重新训练缩减类别任务，也不代表总体患病率。')
# These publication statements are prepared for the authorized atomic Git release.
# The local delivery gate separately requires a successful remote tag/tree check.
release_url='https://github.com/hezejuyede/Pattern_Analysis_and_Applications_Onion_Reliability_v1.0/tree/v1.8.0/v1.8'
en=re.sub(r'\*\*Data availability\.\*\*[^\n]+',lambda m:'**Data availability.** TOM2024 is available at https://doi.org/10.17632/3d4yg89rtr.1; the COLD dataset descriptor is https://doi.org/10.1016/j.dib.2024.110524 and the verified data record is https://doi.org/10.17632/7nxxn4gj5s.2. PlantVillage original images and historical leaf records are maintained at https://github.com/spMohanty/PlantVillage-Dataset and https://github.com/digitalepidemiologylab/plantvillage_deeplearning_paper_analysis. Version v1.8.0 of the derived research materials is archived at '+release_url+'. It provides fixed source assignments, image checksums, cached numerical features, complete replacement predictions and the corrected transfer-sensitivity inputs. Original images remain subject to their upstream terms. Earlier augmented-prefix estimates are explicitly superseded; the correction history is retained.',en)
en=re.sub(r'\*\*Code availability\.\*\*[^\n]+',lambda m:'**Code availability.** The same versioned repository provides original experimental code and a portable entry point for reconstructing all retained replacement statistics, checking same-target comparisons and repeating specified model refits from cached features. The corrected fixed-prediction transfer sensitivity has a separate portable replay. Source-image acquisition and earlier regional/calibration model training are documented separately; those steps are not implied by the cached-input verification. Reproduction commands, versions, licenses and verification scope are specified in the repository and Online Resource 1.',en)
zh=re.sub(r'\*\*数据可用性。\*\*[^\n]+',lambda m:'**数据可用性。** TOM2024见https://doi.org/10.17632/3d4yg89rtr.1；COLD数据描述论文见https://doi.org/10.1016/j.dib.2024.110524，已核实数据记录见https://doi.org/10.17632/7nxxn4gj5s.2。PlantVillage原图及历史叶片记录由原作者维护于https://github.com/spMohanty/PlantVillage-Dataset和https://github.com/digitalepidemiologylab/plantvillage_deeplearning_paper_analysis。衍生研究材料v1.8.0存档于'+release_url+'，提供固定来源分配、图像校验值、缓存数值特征、完整替换预测及修正后的迁移敏感性输入。原图仍受上游许可约束。早期增强文件前缀估计已明确被替代，同时保留更正历史。',zh)
zh=re.sub(r'\*\*代码可用性。\*\*[^\n]+',lambda m:'**代码可用性。** 同一版本仓库提供原始实验代码和可移植入口，用于从缓存特征重建全部保留的替换统计、核查相同目标比较并重复预定模型拟合。修正后的固定预测迁移敏感性具有单独的可移植重放入口。原图获取及早期区域/校准模型训练另行记录，不包含在缓存输入验证完成的含义之内。复现命令、版本、许可及验证范围详见仓库和在线资源1。',zh)

inventory=json.loads((OLD/'reference_inventory.json').read_text(encoding='utf-8'))
extra=[
dict(key='Oquab2024DINOv2',author='Oquab, Maxime and Darcet, Timothée and Moutakanni, Théo and others',title='DINOv2: Learning Robust Visual Features without Supervision',year='2024',journal='Transactions on Machine Learning Research',url='https://openreview.net/forum?id=a68SUt6zFt',formatted='Oquab M et al (2024) DINOv2: Learning Robust Visual Features without Supervision. Transactions on Machine Learning Research. https://openreview.net/forum?id=a68SUt6zFt'),
dict(key='Bozic2025Robustness',author='Božič, Jakob and Fučka, Matic and Zavrtanik, Vitjan and Skočaj, Danijel',title='Robustness of unsupervised methods for image surface-anomaly detection',year='2025',journal='Pattern Analysis and Applications',volume='28',pages='99',doi='10.1007/s10044-025-01477-y',formatted='Božič J, Fučka M, Zavrtanik V, Skočaj D (2025) Robustness of unsupervised methods for image surface-anomaly detection. Pattern Analysis and Applications 28:99. https://doi.org/10.1007/s10044-025-01477-y'),
dict(key='Requena2026Noise',author='Requena, Antonio and Galan-Cuenca, Alejandro and Gallego, Antonio Javier and Valero-Mas, Jose J.',title='Exploring the impact of label-level noise on multi-label k-Nearest Neighbor classification',year='2026',journal='Pattern Analysis and Applications',volume='29',pages='130',doi='10.1007/s10044-026-01707-x',formatted='Requena A, Galan-Cuenca A, Gallego AJ, Valero-Mas JJ (2026) Exploring the impact of label-level noise on multi-label k-Nearest Neighbor classification. Pattern Analysis and Applications 29:130. https://doi.org/10.1007/s10044-026-01707-x')]
lookup={x['key']:x for x in inventory+extra}
order=[]
for k in re.findall(r'\\cite\{([^}]+)\}',en):
    for key in k.split(','):
        if key not in order: order.append(key)
newinv=[]
for i,k in enumerate(order,1):
    r=lookup[k].copy();r['citation_number']=i;newinv.append(r)
assert set(k for x in re.findall(r'\\cite\{([^}]+)\}',zh) for k in x.split(','))==set(order)
abstract=en.split('## Abstract\n\n')[1].split('\n\n**Keywords')[0]
assert 150<=len(abstract.split())<=250,len(abstract.split())
for lang,text in [('en',en),('zh',zh)]:
    assert len(re.findall(r'\\tag\{\d\}',text))==3
    assert len(re.findall(r'^\*\*(?:Table |表)\d',text,re.M))==3
    (OUT/('full_text.md' if lang=='en' else 'full_text_zh.md')).write_text(text,encoding='utf-8')
(OUT/'reference_inventory.json').write_text(json.dumps(newinv,ensure_ascii=False,indent=2),encoding='utf-8')
bib=[]
for r in newinv:
    typ='inproceedings' if 'booktitle' in r else 'article'
    fields=[f'  {k} = {{{v}}}' for k,v in r.items() if k not in ['key','citation_number','formatted']]
    bib.append('@'+typ+'{'+r['key']+',\n'+',\n'.join(fields)+'\n}')
(OUT/'references_paa.bib').write_text('\n\n'.join(bib)+'\n',encoding='utf-8')
(OUT/'figure_map.json').write_text(json.dumps({str(i):str(D/'figures'/fn) for i,fn in [(1,'Fig_1_Matched_Replacement_Design.png'),(2,'Fig_2_Joint_Replacement.png'),(3,'Fig_3_Single_Source_Replacement.png')]},indent=2),encoding='utf-8')
report={'abstract_words':len(abstract.split()),'main_references':len(newinv),'native_equations_expected':3,'main_tables':3,'main_figures':3,'submission_status':'RESEARCH_REVISION_PENDING_PUBLIC_RELEASE_AUTHOR_REVIEW_AND_FINAL_QA','source_base':'v17 immutable source','all_new_result_audits':'independent_review/*RESULT_REVIEW.json'}
(OUT/'MANUSCRIPT_INTEGRATION.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
shutil.copy2(__file__,D/'code'/Path(__file__).name)
print(json.dumps(report))
