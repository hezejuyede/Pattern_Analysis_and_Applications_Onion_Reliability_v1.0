"""Integrate the post-result review analyses without modifying fitted experiments."""
from pathlib import Path
import csv, hashlib, json, re

BASE = Path('D:/论文/SCI投稿汇总/Onion_Deep_Revision_20261003_v18/manuscript_revision')
OUT = Path('D:/论文/SCI投稿汇总/Onion_Deep_Revision_20261003_v19/manuscript_revision')
REVIEW = Path('D:/论文/SCI投稿汇总/Onion_Review_Response_20261003')

def replace_para(text, prefix, new):
    parts = text.split('\n\n')
    indices = [i for i,p in enumerate(parts) if p.startswith(prefix)]
    assert len(indices) == 1, (prefix, indices)
    parts[indices[0]] = new
    return '\n\n'.join(parts)

en = (BASE/'full_text.md').read_text(encoding='utf-8')
zh = (BASE/'full_text_zh.md').read_text(encoding='utf-8')
draft = (REVIEW/'建议吸收的中英段落_工作草稿.txt').read_text(encoding='utf-8').split('\n\n')
abstract = next(p for p in draft if p.startswith('Evaluations with related views'))
abstract = abstract.replace('Joint replacement across 30 splits retained the larger onion response.', 'Joint replacement across 30 splits retained the larger onion response. However, the ResNet18 policy corrected similar fractions of ordinary-replacement errors in onion and potato (58.33% and 58.41%), despite their different absolute contrasts.')
assert 150 <= len(abstract.split()) <= 250, len(abstract.split())
en = replace_para(en, 'Related test views', abstract)
zh = replace_para(zh, '测试图像可能', '相关视图评估可能同时改变测试来源的熟悉程度和普通训练样本构成。我们比较删除相同训练来源后插入相关视图或未使用同类来源的两种匹配策略。单来源干预固定其余训练观察，联合干预同时替换多个来源。两套公开档案包含481个经审计的洋葱组件和538片原作者标识的马铃薯叶。五个固定分类器采用ResNet18、DINOv2及颜色表征，搭配逻辑回归或最近邻。各模型使用相同的替换分配、来源预算和测试锚点。十个固定划分中，洋葱ResNet18和DINOv2逻辑模型的全类单来源差分别为25.19和18.49个百分点；仅给至少30个来源的类别分配目标及评价权重后，分别为13.59和11.32。马铃薯对应差为2.48和2.83。稀疏类别权重明显影响幅度，而支持较充分的类别仍保留正差。30次划分的联合替换也表现为洋葱响应较大；但ResNet18策略在洋葱和马铃薯中纠正的普通替换错误份额相近（58.33%和58.41%），尽管绝对差值不同。两种策略均改变未暴露预测，同一目标的单来源与联合差值随划分改变符号。匹配替换有助于区分本档案内局部相关视图响应与一般训练集敏感性。权重分析具有事后性质，这些计算差异不估计田间诊断收益。')

en = replace_para(en, 'The two tasks supply', 'The onion task uses visual-dependence components reconstructed from an augmented archive; the potato task uses original photographs and explicit leaf numbers in the original PlantVillage acquisition records. Potato inclusion was fixed from metadata before fitting that task. A separate source-locked onion analysis retains the agricultural context. The study makes three empirical contributions. First, matched donor deletion and single-source replacement quantify the response of an identical test target while the remaining training observations are held fixed. Second, comparison with ordinary same-class replacement distinguishes that local response from prediction changes on sources that never enter training. Third, per-class, weighting and error-headroom analyses show how source support and baseline errors affect the reported magnitude. The two archive tasks delimit the settings supported by this evidence.')
zh = replace_para(zh, '两个任务具有', '洋葱任务使用从增强档案中重建的视觉依赖组件，马铃薯任务使用原始照片及PlantVillage原始采集记录中的明确叶片编号。马铃薯纳入范围依据元数据、在该任务拟合前固定。另行开展的来源锁定洋葱分析保留农业情境。本文提供三项实证贡献：第一，通过匹配供体删除与单来源替换，在其余训练观测固定时量化相同测试目标的响应；第二，通过普通同类替换对照，区分局部响应与从未进入训练集的来源上的预测变化；第三，结合逐类别、权重和错误余量分析，揭示来源支持及基线错误如何影响所报告的幅度。两个档案任务界定了这些证据支持的情境范围。')

en = en.replace('\n\n### 2.3 Representations', '\n\nThe 960 onion and 1,080 potato target-within-split records involved 428 distinct audited components and 480 distinct author-recorded leaves, respectively. Targets recurred across splits, and fitted comparisons shared training sources. These counts describe archive coverage and computational reuse, not independently sampled plants.\n\n### 2.3 Representations')
zh = zh.replace('\n\n### 2.3 表征', '\n\n960条洋葱与1,080条马铃薯目标—划分记录分别涉及428个不同审计组件与480片不同作者标识叶片。目标会在划分间重复，拟合比较也共享训练来源。这些数量描述档案覆盖与计算复用，不是独立采样的植株数。\n\n### 2.3 表征')
en = en.replace('A prespecified sensitivity gives evaluation weight', 'A weighting sensitivity, fixed after earlier joint results were known and before the new DINOv2 and single-source fits, gives evaluation weight')
zh = zh.replace('预先规定的敏感性分析只对', '一项在早期联合结果已知后、但在新的DINOv2及单来源拟合前固定的权重敏感性分析，只对')
extra_en = 'Further descriptive analyses use the saved predictions without refitting. Leave-one-class-out summaries omit each evaluation class in turn. Archive-source-frequency weights use the audited class counts, while target-frequency summaries weight observed targets; these are accuracy contrasts under the stated weights, not balanced accuracy or field-prevalence estimates. For error headroom, we average ordinary-replacement errors, corrections and newly introduced errors under the same class and allocation weights. The corrected fraction divided by the ordinary-replacement error gives the fraction of available errors corrected; net contrast divided by that error also subtracts introduced errors. These are ratios of weighted means, not means of per-split ratios. Online Resource 1 provides all models, designs and denominators.'
extra_zh = '进一步的描述性分析使用已保存预测，不重新拟合。逐类剔除汇总依次省略各评价类别；档案来源频率权重取自已审计的类别计数，目标频率汇总则按实际观察到的目标加权。这些是指定权重下的准确率差值，不是平衡准确率，也不是田间患病率估计。错误余量分析在相同类别和分配权重下平均普通替换错误、纠正及新增错误。平均纠正比例除以普通替换错误比例，表示原有错误中被纠正的份额；净差除以该错误比例还扣除了新增错误。这些是加权均值之比，不是逐划分比率的平均。在线资源1给出全部模型、设计及分母。'
mc_en = 'We additionally report conditional Monte Carlo standard errors for the full-class contrast means: the sample standard deviation of the complete split summaries divided by the square root of the number of splits (30 joint or 10 single-source). Under independently generated seeded allocations, this describes numerical uncertainty from allocation sampling in the fixed archive. Shared archive observations do not turn this quantity into uncertainty for independently sampled plants. It excludes between-archive variation and all unmeasured acquisition dependence; no population confidence interval or significance test is inferred from it.'
mc_zh = '另报告全类差值均值的条件蒙特卡洛标准误：完整划分汇总值的样本标准差除以划分次数的平方根（联合30次、单来源10次）。在各带种子分配独立生成的条件下，该量描述固定档案内分配抽样引起的数值不确定性。共享档案观测不会使它转变为独立植株抽样的不确定性；该量不包括档案间变异及未测量的采集依赖，不据此推断总体置信区间或显著性。'
en = en.replace('\n\n### 2.5 Separate', '\n\n'+extra_en+'\n\n'+mc_en+'\n\n### 2.5 Separate')
zh = zh.replace('\n\n### 2.5 单独', '\n\n'+extra_zh+'\n\n'+mc_zh+'\n\n### 2.5 单独')
en = en.replace('\n\nOpenAI Codex assisted', '\n\nPer-class, leave-one-class-out, frequency-weighted, error-headroom and conditional Monte Carlo summaries were added after the replacement results were known. They retain all fitted models and frozen allocations and are reported as post-result descriptive analyses.\n\nOpenAI Codex assisted')
zh = zh.replace('\n\nOpenAI Codex辅助', '\n\n逐类别、逐类剔除、频率加权、错误余量及条件蒙特卡洛汇总均在替换结果已知后追加，保留全部拟合模型和冻结分配，作为事后描述性分析报告。\n\nOpenAI Codex辅助')

en = en.replace('\n\n**Table 2**', '\n\nWithin the same 30-split joint design, assigning evaluation weight only to classes with at least 30 sources reduced the onion ResNet18 contrast from 23.66 to 14.10 points and the DINOv2 contrast from 21.18 to 12.13 points. For ResNet18, leaving out each class in turn yielded contrasts of 14.10–29.51 points; the minimum occurred when purple blotch was omitted. Weighting by archive source frequency instead gave 11.67 points. Thus a positive descriptive contrast is retained, while its magnitude depends substantially on what the class weights represent.\n\n**Table 2**',1)
zh = zh.replace('\n\n**表2**', '\n\n在相同的30划分联合设计内，只对至少含30个来源的类别赋予评价权重，使洋葱ResNet18差值从23.66降至14.10个百分点，DINOv2从21.18降至12.13个百分点。ResNet18依次省略各类别后的差值为14.10–29.51个百分点；省略紫斑类时最小。改按档案来源频率加权时为11.67个百分点。因此，描述性正差仍然保留，但其幅度明显依赖类别权重的含义。\n\n**表2**',1)
en = en.replace('When only classes with at least 30 sources received evaluation weight,', 'When only classes with at least 30 sources received both intervention-target and evaluation-anchor weight,')
zh = zh.replace('仅对至少含30个来源的类别赋予评价权重后，', '仅对至少含30个来源的类别同时赋予干预目标与评价锚点权重后，')
en = en.replace('The sensitivity retains the same multiclass fits; it cannot estimate the accuracy of a new three-class training problem.', 'The 20 purple-blotch target-within-split records involved only ten distinct components. The sensitivity retains the same multiclass fits; it cannot estimate the accuracy of a new three-class training problem.')
zh = zh.replace('敏感性分析保留相同的多分类拟合，不能估计一个新三分类训练任务的准确率。', '20条紫斑类目标—划分记录仅涉及10个不同组件。敏感性分析保留相同的多分类拟合，不能估计一个新三分类训练任务的准确率。')
en = en.replace('“≥30” changes only evaluation weights to classes with at least 30 sources;', '“≥30” assigns both intervention-target and evaluation-anchor weights only to classes with at least 30 sources;')
zh = zh.replace('“≥30”仅将评价权重限制到至少含30个来源的类别，', '“≥30”将干预目标与评价锚点权重均限制到至少含30个来源的类别，')

rows = list(csv.DictReader((OUT/'Table_4_display.csv').open(encoding='utf-8-sig')))
table = '| Task | Model | Sham error (%) | Corrected (pp) | Harmed (pp) | Net (pp) | Corrected/error (%) |\n| --- | --- | --- | --- | --- | --- | --- |\n'
for r in rows:
    table += '| '+' | '.join([r['dataset'].capitalize(),r['model']]+[f'{float(r[k]):.2f}' for k in ['sham_error_headroom_percent','corrected_pp','harmed_pp','net_contrast_pp','corrected_over_sham_error_percent']])+' |\n'
head_en = ('### 3.4 Absolute contrasts and available errors give different comparisons\n\n'
    'Table 4 decomposes the joint contrast against ordinary-replacement errors. ResNet18 error headroom was 42.92% in onion and 3.26% in potato. Related-view insertion corrected 58.33% and 58.41% of those weighted errors, respectively. After subtracting newly introduced errors, net contrasts represented 55.12% and 52.25% of the corresponding headroom. DINOv2 likewise corrected 72.76% and 71.12% of ordinary-replacement errors. Smaller absolute potato contrasts therefore coexisted with similar observed correction fractions. These ratios do not establish equivalence between tasks: error headroom was not manipulated, and acquisition, class support and view construction also differed.\n\n'
    '**Table 4** Post-result error-headroom decomposition of the full-class joint comparison across 30 splits. All fractions use the same class-balanced weights and allocation averaging as Table 2. Corrected and harmed describe changes from sham to exposure; net equals corrected minus harmed before rounding. The final column is the ratio of the mean corrected fraction to the mean sham error, expressed as a percentage. It is neither an accuracy nor a mean of split-level ratios. All five models and single-source results are retained in Online Resource 1\n\n'+table.strip())
head_zh = ('### 3.4 绝对差值与可供纠正的错误提供不同的比较视角\n\n'
    '表4以普通替换错误为参照分解联合差值。ResNet18的错误余量在洋葱为42.92%，马铃薯为3.26%；相关视图插入分别纠正这些加权错误的58.33%和58.41%。扣除新增错误后，净差分别占对应余量的55.12%和52.25%。DINOv2相应纠正72.76%和71.12%的普通替换错误。因此，马铃薯较小的绝对差与相近的观测纠正份额可以同时存在。这些比率不建立任务等价性：错误余量未经操控，采集条件、类别支持与视图构成也不同。\n\n'
    '**表4** 30划分全类联合比较的事后错误余量分解。所有比例使用与表2相同的类别平衡权重和分配平均方法。纠正与新增错误描述从sham到暴露的变化；舍入前净差等于纠正减新增错误。末列为平均纠正比例除以平均sham错误比例，以百分数表示；既不是准确率，也不是逐划分比率的平均。在线资源1保留全部五模型及单来源结果\n\n'+table.strip())
en = en.replace('### 3.4 The separate onion transfer', head_en+'\n\n### 3.5 The separate onion transfer')
zh = zh.replace('### 3.4 单独的洋葱迁移', head_zh+'\n\n### 3.5 单独的洋葱迁移')

en = replace_para(en, 'The useful empirical distinction', 'The central result is an attribution distinction: related-view insertion improved the exposed target, whereas ordinary same-class replacement also changed predictions on never-exposed sources. A before-and-after comparison without the latter control would conflate these responses. The large onion contrast was sensitive to sparse-class weighting, yet remained positive across the examined class omissions. Conversely, the smaller potato contrast did not imply that its remaining errors were insensitive to related views. The two tasks had very different error headroom and similar observed correction fractions for both logistic deep-feature models. Reporting class support and available errors therefore changes what can be learned from a contrast expressed only in percentage points.')
zh = replace_para(zh, '有用的经验区分在于', '核心结果在于区分两类归因：相关视图插入改善了被暴露目标的预测，而普通同类替换也改变了从未暴露来源上的预测。如果缺少后一对照，简单的前后比较会混淆这两类响应。较大的洋葱差值对稀疏类别权重敏感，但在所考察的逐类剔除分析中均保留正差。相反，较小的马铃薯差值不意味着其剩余错误对相关视图不敏感。两个任务的错误余量很不相同，但两个深度特征逻辑模型观察到的纠正份额相近。因此，同时报告类别支持和可供纠正的错误，会改变仅从百分点差值中所能获得的认识。')
en = en.replace('The large onion effect and small potato effect should therefore be presented together as limits on generalization.', 'The two tasks should be presented together with their source support and error headroom. Their contrast cannot isolate an augmentation mechanism or a crop effect.')
zh = zh.replace('因此，应将较大的洋葱效应和较小的马铃薯效应同时呈现，用以说明推广的边界。', '因此，应同时呈现两个任务及各自来源支持与错误余量；两任务差异不能单独识别增强机制或作物效应。')
en = en.replace('Ten overlapping splits and 2,040 target-within-split interventions are not independent acquisitions.', 'The 2,040 target-within-split records reuse 908 distinct components or leaves across ten splits, rather than representing 2,040 independent acquisitions. Conditional Monte Carlo errors quantify only allocation sampling in these fixed archives.')
zh = zh.replace('十个重叠划分和2,040个“目标—划分”干预不是独立采集。', '十次划分中的2,040条目标—划分记录复用了908个不同组件或叶片，而非2,040次独立采集。条件蒙特卡洛误差只量化固定档案内的分配抽样。')
en = replace_para(en, 'Related-view insertion produced', 'Related-view insertion produced a local advantage over matched unrelated replacement even with every other training observation held fixed. Its magnitude depended on class support and ordinary-replacement errors: reducing sparse-class weight lowered the onion contrast, while the smaller potato contrast coexisted with a similar observed fraction of corrected errors. Ordinary replacement also changed never-exposed predictions, and matched single-versus-joint contrasts varied in direction across splits. Related-image evaluations should therefore report fixed-background target responses, a matched replacement control, class support and error headroom together. These archive-conditional findings strengthen attribution within the evaluated pipelines; they do not establish field diagnostic validity.')
zh = replace_para(zh, '即使保持其余训练观测', '即使保持其余训练观测全部不变，相关视图插入相对匹配的无关联替换仍产生局部优势。幅度依赖类别支持与普通替换错误：减小稀疏类别权重会降低洋葱差值，马铃薯较小的绝对差则与相近的观测错误纠正份额同时存在。普通替换也改变从未暴露来源上的预测，配对的单来源与联合差值在不同划分中方向不一。因此，相关图像评估应同时报告固定背景目标响应、匹配替换对照、类别支持及错误余量。这些以档案为条件的结果加强了所评估流程内的归因，但不建立田间诊断有效性。')
en = en.replace('v1.8.0/v1.8','v1.9.0/v1.9').replace('Version v1.8.0','Version v1.9.0')
zh = zh.replace('v1.8.0/v1.8','v1.9.0/v1.9').replace('衍生研究材料v1.8.0','衍生研究材料v1.9.0')
en = en.replace('**Online Resource 1** Complete provenance, frozen allocations, all retained models and endpoints, single-source and matched joint comparisons, transfer sensitivities and reproduction instructions', '**Online Resource 1** Complete provenance, frozen allocations, all retained models and endpoints, single-source and matched joint comparisons, transfer sensitivities, post-result weighting and error-headroom analyses, conditional Monte Carlo errors and reproduction instructions')
zh = zh.replace('**Online Resource 1** 完整来源记录、冻结分配、全部模型和终点、单来源与配对联合比较、迁移敏感性及复现说明', '**Online Resource 1** 完整来源记录、冻结分配、全部模型和终点、单来源与配对联合比较、迁移敏感性、事后权重与错误余量分析、条件蒙特卡洛误差及复现说明')

for name,t in [('full_text.md',en),('full_text_zh.md',zh)]:
    assert '\ufffd' not in t
    assert len(re.findall(r'\\tag\{[123]\}',t)) == 3
    assert len(re.findall(r'\*\*(?:Table |表)[1-4]\*\*',t)) == 4
    assert 'v1.8.0/v1.8' not in t
    (OUT/name).write_text(t,encoding='utf-8')
def cites(t): return [x for group in re.findall(r'\\cite\{([^}]+)\}',t) for x in group.split(',')]
assert cites(en) == cites(zh)
assert set(cites(en)) == set(cites((BASE/'full_text.md').read_text(encoding='utf-8')))
qa={'abstract_words':len(abstract.split()),'citation_keys':len(set(cites(en))),'tables':4,'equations':3,'new_fits':0,'sources':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [OUT/'full_text.md',OUT/'full_text_zh.md']}}
(OUT/'SOURCE_INTEGRATION_QA.json').write_text(json.dumps(qa,indent=2),encoding='utf-8')
print(json.dumps(qa))
