# 在线资源1——补充方法与完整结果表

采用单来源与联合替换对照归因植物图像评估中的相关视图收益

Xin Li、Bojian Guo、Asel Kartanova；通讯邮箱：lixin26@kstu.kg。

S1-S6记录来源身份、原始替换对照及保留的迁移分析；S7说明冻结表示与单来源扩展。原始数值CSV保留已保存估计的完整精度；S5b是四位小数展示表，完整精度见S5。S2/S4正文表将对应CSV中的比例乘以100，以百分点显示；下列路径相对于本文件。计算划分次数不能当作生物样本量。

## S1 数据来源与纳入规则

洋葱档案包含4,502张增强图与815条原始记录，后者为813份不同RGB内容。已发现两个文件名前缀共享同一原图，因此前缀不能证明来源独立。逐成员审计先比较每张增强图的五个保留候选，共22,510对；随后在全部原始标签间用冻结ResNet余弦相似度补充候选，每增强图检索前五个原始来源，每原图检索前十个其他原图，按内容哈希稳定处理并列。新增检验16,985对增强/原图及6,508对无向原图关系。有限候选检索不能证明不存在其他关系。

SIFT[S1]最多提取1,000个特征，contrastThreshold=0.02；使用L2最近邻匹配及0.75比值检验，单应性RANSAC[S2]阈值为4像素。修正后的支持记录包括查询图的正常及水平镜像方向。几何支持要求至少15个内点且内点比例不低于0.60；更严格的空间筛选要求两图内点凸包均覆盖至少5%图像面积，残差中位数不超过2像素、95百分位不超过4像素。坐标来自未经缩放的解码灰度图。这些是启发式对应判据，不能建立植株身份或完整增强历史。相关可执行规则保存在supplement_provenance中。

所有阳性关系加上文件名前缀关系构成保守隔离组件；跨原始标签组件整体隔离，不按多数票改标签。其余每组件选择获最多不同增强成员空间支持的共同原始内容，按内容哈希字典序处理并列；至少需要三张不同受支持视图，只纳入支持该共同参照的成员，其余保持排除或隔离。最终481个组件含3,076张图像。组件不能直接等同一张原图、一片叶或一株植物，纳入子集也不同于完整档案。规则未读取分类器结果，但设计是在早期档案分析发现来源问题之后完成的。

独立PlantVillage任务保留完整合格的原始彩色三类马铃薯照片：538片原作者编号叶片、2,152张照片，每叶四张；早疫标签250叶、晚疫250叶、健康38叶。HF提交9e97599868962bd0079b8db4b7f1efa9185fa1e7的映射blob为cb04e3723d2ce15b0411483d71a60b5d536bc7f6，与2016年原GitHub提交40789680ba2e6382608dba47b397688fcbd0d04a中的映射一致。全部叶片归属逐一对回2016年File Name和Leaf #字段；全部照片字节与原作者提交7f7ecc7e1eaca78107e3affe7cb5abd9427e139a的Git对象一致，解码RGB均不重复。灰度、分割和增强处理版本不计作新观测。这是叶片标识，仍不是整株身份，也没有新的统一病理复核或前瞻田间采集。

## S2 匹配替换、模型与变异

正文给出被选探针政策差及共同零暴露参照下的哨兵正误转换率。每个被选探针对应不同的可移除供体及同类预留替代来源。两政策删除同一供体，暴露补入探针同来源的两张非锚点视图，sham补入预留来源的两张视图；预留来源不参加零暴露训练。每对政策训练类别数和预算一致，测试锚点相同。多个来源同时变化，因此这是允许相互影响的联合政策差，不能解释为单样本纯直接因果效应。

洋葱每划分97个探针（96个配对）、48个哨兵，每训练臂240个来源/480张图，每分配选48个配对探针。马铃薯每划分108个配对探针、54个哨兵，每训练臂268叶/536图，每分配选54个探针。30个种子20261003–20261032各产生五组互补A/B；先平均A/B，再平均该划分五组。30个划分均值的2.5及97.5百分位表示设计变异，不是独立生物重复或植株总体置信区间。没有显著性检验、等效界值或通用泄漏阈值。

主特征为冻结的torchvision ImageNet1K_V1 ResNet18全局平均池化512维特征，采用标准评估变换。颜色对照为189维：BGR/HSV/Lab九通道各含16箱直方图、均值、标准差及10/50/90百分位。每个逻辑回归臂独立拟合标准化和类别平衡L2逻辑回归，C=0.01、lbfgs、最多10,000次迭代。未调参余弦1NN使用相同ResNet特征，固定行顺序处理并列。没有根据新任务测试结果选择表示、参数或保留模型。原始多分类为主，健康/其他重新拟合为次要端点；二者拟合目标及类别权重不同，不能把差别归为标签粒度的纯因果效应。

表S1保留原始三个模型的全部端点、面板、分层与指标；新DINOv2结果见S7；下面表S2列出两个端点所有模型的差，单位为百分点。表S3保存同一被选掩码上的零暴露、sham、暴露原始BA及错误余量；没有构造归一收益或据此推断跨任务因果机制。

### 表S2 被选探针暴露减sham差（百分点）

| task | model | endpoint | difference | split_p025 | split_p975 |
| --- | --- | --- | --- | --- | --- |
| onion | colour_logit | binary_healthy_other | 12.481739 | 8.194565 | 17.788913 |
| onion | colour_logit | multiclass | 37.714989 | 25.064245 | 49.032838 |
| onion | resnet_cosine_1nn | binary_healthy_other | 26.391304 | 19.420870 | 32.147609 |
| onion | resnet_cosine_1nn | multiclass | 50.250038 | 33.817162 | 62.449027 |
| onion | resnet_logit | binary_healthy_other | 6.575072 | 2.721957 | 10.100870 |
| onion | resnet_logit | multiclass | 23.655187 | 8.336442 | 37.939559 |
| potato | colour_logit | binary_healthy_other | 3.830000 | -0.182500 | 10.632500 |
| potato | colour_logit | multiclass | 2.635556 | -0.731667 | 7.277500 |
| potato | resnet_cosine_1nn | binary_healthy_other | 1.721667 | -6.365000 | 13.300000 |
| potato | resnet_cosine_1nn | multiclass | 2.136667 | -3.350000 | 9.766667 |
| potato | resnet_logit | binary_healthy_other | 1.908333 | -0.327500 | 9.591250 |
| potato | resnet_logit | multiclass | 1.704444 | -0.170000 | 6.049167 |

### 表S4 主多分类模型的原始类别差（百分点）

| task | stratum | mean | split_p025 | split_p975 |
| --- | --- | --- | --- | --- |
| onion | healthy | 6.101449 | 1.619565 | 13.293478 |
| onion | iris_yellow_virus | 12.385965 | 4.539474 | 22.210526 |
| onion | purple_blotch | 52.333333 | 0.000000 | 100.000000 |
| onion | stemphylium_colletotrichum_leaf_blight | 23.800000 | 8.000000 | 45.650000 |
| potato | Potato___Early_blight | 0.213333 | -1.020000 | 1.310000 |
| potato | Potato___Late_blight | 0.733333 | -0.910000 | 2.910000 |
| potato | healthy | 4.166667 | 0.000000 | 18.187500 |

洋葱紫斑标签只有13个来源组件，每臂仅一个被选探针。其较大差值和很宽的划分范围显著影响等权四类平均值；这不能解释为按田间疾病比例得到的收益或某病原机制。原始类别支持量见主文表1，所有分层结果保留在表S1。

## S3 单独锁定的洋葱跨来源迁移

该二分类任务使用1,643张完全去重TOM2024图像、103个文件名推得UTC日，以及813张原始COLD图。端点为健康/档案标记的外观受损叶片，不是病原诊断。五个按日分组外折估计内部性能；每外折四内折从0.0001、0.001、0.01、0.1、1、10、100中选C，标准化、Platt映射和Youden阈值均只用开发数据。另一次源数据五折搜索选择最终C，并在全部TOM拟合最终分类头；最终校准器和阈值用汇总的TOM外折折外决策分数训练，然后不作改变地用于COLD。固定C的校准构造敏感性见S4。

主要ResNet与患病率、颜色和手工特征基线全部保留。EfficientNet-B0、ConvNeXt-Tiny和Swin-T为主要COLD结果已知后的架构敏感性，不依性能删选，也不使用COLD拟合、调参、校准或选阈值；不能称为全新未查看数据的确认，也不能替代匹配替换任务自身的三个同条件对照。

修正后的外部不确定性使用全部阳性几何关系图（547个原始内容组件）与严格空间图（700组件），两种敏感性均保留全部813条记录，每种3,000次固定预测组件自助重采样。模型间比较使用相同配对权重，外部减内部差另独立重采TOM日期。区间以不完整关系图和已拟合模型为条件，不是植株总体区间或完整重训不确定性。旧图像单位区间只作为明确标记的示例留在补图S2，不再作为优先不确定性依据。表S5–S7保留全部模型、指标、配对比较和迁移差。

### 表S5b 两种修正图下全部七个外部模型

各单元为估计值[条件下限, 上限]；没有按结果删除模型。

| graph | model | balanced_accuracy | auroc | brier |
| --- | --- | --- | --- | --- |
| all_high | color_shortcut_logit | 0.5253 [0.4971, 0.5583] | 0.6066 [0.5636, 0.6570] | 0.2752 [0.2517, 0.2958] |
| all_high | convnext_tiny_logit | 0.5829 [0.5579, 0.6051] | 0.7105 [0.6613, 0.7502] | 0.2507 [0.2237, 0.2822] |
| all_high | efficientnet_b0_logit | 0.6502 [0.6143, 0.6791] | 0.7205 [0.6739, 0.7564] | 0.2620 [0.2360, 0.2932] |
| all_high | handcrafted_logit | 0.5707 [0.5363, 0.6126] | 0.6028 [0.5604, 0.6486] | 0.2882 [0.2664, 0.3098] |
| all_high | prior_prevalence | 0.5000 [0.5000, 0.5000] | 0.5000 [0.5000, 0.5000] | 0.3208 [0.2978, 0.3440] |
| all_high | resnet18_logit | 0.6553 [0.6215, 0.6914] | 0.7345 [0.6868, 0.7749] | 0.2403 [0.2142, 0.2720] |
| all_high | swin_t_logit | 0.5621 [0.5450, 0.5790] | 0.7820 [0.7497, 0.8177] | 0.2824 [0.2499, 0.3179] |
| spatial_high | color_shortcut_logit | 0.5253 [0.4973, 0.5544] | 0.6066 [0.5643, 0.6503] | 0.2752 [0.2544, 0.2949] |
| spatial_high | convnext_tiny_logit | 0.5829 [0.5599, 0.6070] | 0.7105 [0.6714, 0.7517] | 0.2507 [0.2252, 0.2757] |
| spatial_high | efficientnet_b0_logit | 0.6502 [0.6177, 0.6829] | 0.7205 [0.6827, 0.7568] | 0.2620 [0.2356, 0.2890] |
| spatial_high | handcrafted_logit | 0.5707 [0.5343, 0.6076] | 0.6028 [0.5596, 0.6462] | 0.2882 [0.2657, 0.3106] |
| spatial_high | prior_prevalence | 0.5000 [0.5000, 0.5000] | 0.5000 [0.5000, 0.5000] | 0.3208 [0.3007, 0.3400] |
| spatial_high | resnet18_logit | 0.6553 [0.6209, 0.6891] | 0.7345 [0.6935, 0.7733] | 0.2403 [0.2155, 0.2668] |
| spatial_high | swin_t_logit | 0.5621 [0.5455, 0.5796] | 0.7820 [0.7463, 0.8157] | 0.2824 [0.2523, 0.3130] |

## S4 仅使用源数据的校准构造

敏感性复用主要表示、五个TOM日分组折及已选定最终C=0.1。原外折选择的C为[0.01,0.01,0.1,0.1,0.1]；一个分数流重建这些折参数，另一个在每折均用最终C=0.1。标准化只在各折训练集拟合，每个源分数流各拟合Platt映射和Youden阈值；两臂共享全部TOM上C=0.1的最终分类头。因此比较改变的是分数构造、校准和阈值，不是表示或最终分类头。用于拟合校准器的源折外分数不能再宣称为无偏验证。

两臂外部样本一致；共同底层分数经过正斜率单调映射，AUROC相同不构成新增独立验证。全源数据拟合仍可能改变分数尺度，固定C并未排除所有尺度偏移。分析是在先前COLD结果已知后设计，没有按COLD结果决定保留哪一臂。表S8–S9采用修正的两图配对组件区间，替代旧图像自助区间。

### 表S9 固定最终C校准减重建的原校准

| variant | metric | estimate | lower | upper |
| --- | --- | --- | --- | --- |
| all_high | balanced_accuracy | -0.009171 | -0.021625 | 0.006271 |
| all_high | brier | -0.009304 | -0.012036 | -0.007194 |
| spatial_high | balanced_accuracy | -0.009171 | -0.022320 | 0.003295 |
| spatial_high | brier | -0.009304 | -0.011604 | -0.007150 |

## S5 TOM2024内部的恢复地区敏感性

公开索引恢复了1,416张图像的唯一地区关联，其余227张缺地区图像在所有地区实验中均不进入训练或测试。地区为Centre-Ouest、Centre-Sud、Plateau-Central。主要敏感性每次留出一个地区，并从训练中清除任何与测试地区共享文件名UTC日的图像；region-only敏感性使用相同测试集但允许训练日期重叠，因此训练数量和组成不同，差不能当作纯泄漏因果效应。

保留患病率、颜色、手工特征和冻结ResNet18四模型。每个开发集内五个日分组折从相同七值C网格选择参数，生成所选C的源折外分数、拟合Platt映射及有限Youden阈值；这些用于校准的折外分数不是无偏内部验证。患病率基线用训练患病率和固定0.5阈值。留出地区标签不选择模型、参数、校准器或阈值。各地区固定预测区间用2,000次UTC日聚类抽样；汇总联合重采所有唯一日期以保留跨地区依赖，并排除任一被表示地区缺失某端点状态的重复。推断以这三个地区与拟合模型为条件。

表S10包含全部24个地区/变体/模型组合。表S11–S15保留等地区均值、最差地区指标、区间、原始标签召回、类别支持及配对差。地区/日期恢复不能当成农场编号、整株身份、独立核实的采集日历或病理复核。这是既有档案中的事后情境压力测试，不是新外部队列。

### 表S10 全部地区结果

| variant | region | model | train_n | n | balanced_accuracy | roc_auc | brier |
| --- | --- | --- | --- | --- | --- | --- | --- |
| region_day_disjoint | Centre-Ouest | prior_prevalence | 384 | 736 | 0.500000 | 0.500000 | 0.303774 |
| region_day_disjoint | Centre-Ouest | color_shortcut_logit | 384 | 736 | 0.767256 | 0.838138 | 0.173628 |
| region_day_disjoint | Centre-Ouest | handcrafted_logit | 384 | 736 | 0.675435 | 0.759347 | 0.191594 |
| region_day_disjoint | Centre-Ouest | resnet18_logit | 384 | 736 | 0.801487 | 0.953616 | 0.061066 |
| region_day_disjoint | Centre-Sud | prior_prevalence | 854 | 313 | 0.500000 | 0.500000 | 0.464941 |
| region_day_disjoint | Centre-Sud | color_shortcut_logit | 854 | 313 | 0.793376 | 0.879924 | 0.243571 |
| region_day_disjoint | Centre-Sud | handcrafted_logit | 854 | 313 | 0.749820 | 0.843092 | 0.290565 |
| region_day_disjoint | Centre-Sud | resnet18_logit | 854 | 313 | 0.874258 | 0.952150 | 0.148902 |
| region_day_disjoint | Plateau-Central | prior_prevalence | 760 | 367 | 0.500000 | 0.500000 | 0.210596 |
| region_day_disjoint | Plateau-Central | color_shortcut_logit | 760 | 367 | 0.794751 | 0.839485 | 0.128990 |
| region_day_disjoint | Plateau-Central | handcrafted_logit | 760 | 367 | 0.801152 | 0.893642 | 0.114369 |
| region_day_disjoint | Plateau-Central | resnet18_logit | 760 | 367 | 0.880165 | 0.960458 | 0.084670 |
| region_only | Centre-Ouest | prior_prevalence | 680 | 736 | 0.500000 | 0.500000 | 0.221366 |
| region_only | Centre-Ouest | color_shortcut_logit | 680 | 736 | 0.804363 | 0.894501 | 0.123817 |
| region_only | Centre-Ouest | handcrafted_logit | 680 | 736 | 0.741653 | 0.803223 | 0.141883 |
| region_only | Centre-Ouest | resnet18_logit | 680 | 736 | 0.855499 | 0.960455 | 0.054004 |
| region_only | Centre-Sud | prior_prevalence | 1103 | 313 | 0.500000 | 0.500000 | 0.450844 |
| region_only | Centre-Sud | color_shortcut_logit | 1103 | 313 | 0.786090 | 0.872639 | 0.231374 |
| region_only | Centre-Sud | handcrafted_logit | 1103 | 313 | 0.731854 | 0.842193 | 0.267883 |
| region_only | Centre-Sud | resnet18_logit | 1103 | 313 | 0.866905 | 0.950306 | 0.147819 |
| region_only | Plateau-Central | prior_prevalence | 1049 | 367 | 0.500000 | 0.500000 | 0.209140 |
| region_only | Plateau-Central | color_shortcut_logit | 1049 | 367 | 0.822079 | 0.859861 | 0.113869 |
| region_only | Plateau-Central | handcrafted_logit | 1049 | 367 | 0.802699 | 0.897340 | 0.107792 |
| region_only | Plateau-Central | resnet18_logit | 1049 | 367 | 0.883845 | 0.965010 | 0.072307 |

主要按日清除ResNet等地区平均BA为0.851970，条件区间[0.820712,0.885926]；相对颜色差为+0.066842[0.019869,0.119359]，但Centre-Ouest的差跨零。该地区健康特异度为57/88，尽管AUROC达0.953616。这区分了排序表现与可迁移的决策阈值，不能据此声称识别了未见病原。原始COLD BA仍为0.655321。

## S6 复算、责任及补图

保留的原始三模型、两端点联合分析共完成5,040次逻辑回归拟合、2,520个最近邻端点评估和1,160,460条预测，无拟合警告。主要模型/端点差从原始预测独立重建，另独立重训24个逻辑回归组合。这指另写计算实现，不是盲态人类或病理专家复核，也没有声称每次拟合都被独立重训。v1.8.0发布版本存档了当前替换实验、冻结数值输入、预测、代码及核验记录，固定地址为 https://github.com/hezejuyede/Pattern_Analysis_and_Applications_Onion_Reliability_v1.0/tree/v1.8.0/v1.8。发布版README给出code/portable下的可移植命令；supplementary_context说明保留的区域、迁移与校准分析历史。原始照片需按记录的上游条款从来源档案获取。本轮可移植烟测覆盖替换及单来源结果重建和指定组合的重训，并未重复全部区域或校准训练；这一范围与原始实验日志及较大规模独立计算审计分别说明。

补图S1位于../figures/Fig_S1_Matched_Replacement_Binary.pdf，并附EPS及600dpi PNG。它呈现原始三个模型、两任务的次要健康/其他端点：被选探针BA差和暴露减sham的哨兵正误转换率差。点为30个划分均值的平均，横线为其2.5–97.5百分位，不是生物总体置信区间。两种哨兵政策使用同一零暴露参照；此图不是二分类与多分类标签粒度的因果比较。

来源JSON保留历史路径及哈希，复制位置不会改变冻结时间。几何脚本副本仍保留历史依赖。SUPPLEMENT_BUILD_v17_RETAINED.json记录各数表确切来源、复制字节哈希和派生表哈希。原始冻结脚本保留历史计算过程；发布版可移植入口通过记录的映射将历史路径解析到随附数值输入。

## S7 冻结表示与单来源扩展

### S7.1 固定DINOv2表示

本扩展在此前结果已经知道后设计。新增特征提取和分类器结果产生前，固定选择一种不同表示家族：无register token、在LVD-142M预训练的DINOv2 ViT-S/14。它是预训练表示敏感性检查，不是穷尽当前最先进架构，也不是新增表示学习方法。官方权重SHA256为`b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9`。随后以Git对象哈希核对全部190份冻结源码，均与官方提交`7764ea0f912e53c92e82eb78a2a1631e92725fc8`逐字节相符，未改变提取代码或权重。记录保留在supplement_provenance。

图像转RGB后，以带抗锯齿的双三次插值将短边缩放到256像素，中心裁剪224×224，转为浮点张量并按ImageNet通道均值(0.485,0.456,0.406)和标准差(0.229,0.224,0.225)标准化。显式实现与官方分类评估变换在固定图像上逐像素一致。模型在评估模式输出最终归一化CLS表示，共384维。没有微调、梯度、随机增强或依据标签排除图像。逻辑回归使用原始输出向量；仅余弦1NN使用单位长度向量。每行均核对原图SHA256、manifest行身份和规范化后的精确路径，保留全部5,228张纳入图像。预先固定的16行重新提取最大绝对差为0。这些检查和公共权重并不能证明研究档案未出现在预训练语料中。

DINOv2联合实验复用全部30个原始来源划分、锚点、插入视图、供体/替代来源配对和互补分配。采用相同逐臂标准化与类别平衡L2逻辑回归(C=0.01)，同时保留未调参余弦1NN。新增端点仅为多分类；S2保留的三个模型二分类结果没有被悄悄扩展。表示维数和预训练家族同时变化，因此相对ResNet18的差异不是架构或预训练单独因素的因果估计。表S16、S28和S29分别保留DINOv2全部模型、分层、面板和指标的汇总、分配臂及划分结果。

### S7.2 固定背景下的单来源替换

新拟合前固定原始前十个种子20261003–20261012。保留每个配对探针：洋葱每划分96个目标，马铃薯108个，共960和1,080个目标/种子条件。每次从原始零暴露训练集开始，只删除该目标既定映射的供体。暴露插入目标固定的两张非锚点视图；sham插入此前预留的同类替代来源两张固定视图。该E/H配对中其余训练来源和视图逐项相同。训练预算仍为洋葱240来源/480图、马铃薯268叶/536图，类别数量保持一致。所有探针和哨兵锚点保持固定；洋葱奇数未配对探针可作为未暴露评价锚点，但不作为干预目标。

保留预先确定的五个模型：ResNet18逻辑回归、颜色逻辑回归、DINOv2逻辑回归、ResNet18余弦1NN和DINOv2余弦1NN。逻辑回归参数完全沿用S2。两个1NN明确使用各自归一化特征和相同的确定性最近邻规则，没有根据结果删除模型。端点只有多分类。文件索引提供实际训练成员、来源交集、预算、图像哈希和完整概率。执行必须通过完整输入/代码冻结及绑定该协议哈希的独立拟合前PASS审查。

自身目标结果是一个固定锚点的暴露减sham正误差；先在原始目标类别内平均，再在该划分内对目标类别等权平均。只有完成该聚合后，它才成为类别平衡准确率差。对其他全部探针和哨兵，先在每个目标内对评价类别等权，再对各目标类别内的目标求均值，最后对目标类别等权。这避免数量较多的健康/IYSV目标自动压过较少的目标类别。表S17–S20保留目标、目标类别、划分和汇总结果。十个划分的均值、最小值、最大值和样本标准差描述分配变异，不是植株总体置信区间，也不报告p值。

### S7.3 类别支持与错误分解敏感性

常见类别规则为原始manifest至少30个来源单位。因此洋葱保留健康、IYSV标记和混合叶枯标记三类，马铃薯三类全部保留。这是在约20%探针/10%哨兵划分下的透明支持检查，不是保证生物样本充分的定理。阈值为本扩展固定，但此前逐类结果已经知道，不能称为盲态确证性亚组研究。

没有训练删类分类器。在联合替换中，敏感性只对既有评价类别重新赋权，全部原始训练及替换来源保持不变。在单来源替换中，所报常见类别敏感性同时限制目标类别和评价类别，全部完整任务拟合及概率保持不变。完整类别和逐类估计均保留。这一区别对哨兵比较尤其重要：改变目标类别构成与改变评价类别构成是不同操作。主图2和3的橙色序列采用各自对应定义。马铃薯常见类与原始全类结果相同。

每条目标/面板记录另给出sham错误而暴露正确，以及sham正确而暴露错误的比例。在相同类别权重下，两者之差等于E-H平衡正误差，前者不能超过sham错误比例。同时保留相同锚点的零暴露/sham/暴露正误、真实类别概率变化和概率总变差。正误转换、收益、损失和错到另一错均相对同一个零暴露模型。这些恒等式和边界在每行结果上核对，但不能证明两任务差异的原因；错误余量、采集条件、标签与来源关系并未被独立操纵。

### S7.4 同目标的替换范围比较

对前十个划分中的每个目标、模型和种子，将单来源E-H与该目标在五次被选联合分配中的平均E-H配对。两任务合计10,200个目标/种子/模型配对。每个目标在每组互补联合分配中恰有一臂被选。两项分析中每个模型和锚点的零暴露预测在数值精度内一致；两臂使用相同目标类别层级聚合。表S21–S24给出每个配对目标、类别、划分和汇总，下面表S27为四舍五入的百分点展示。主表3必须采用这一同十划分比较，不能把三十划分联合均值与十划分单来源均值直接相减。

### 表S27 前十个划分的同目标比较（百分点）

| task | model | single_pp | joint_pp | single_minus_joint_pp | difference_min_pp | difference_max_pp |
| --- | --- | --- | --- | --- | --- | --- |
| onion | Colour + LR | 30.1310 | 34.6841 | -4.5531 | -16.8021 | 10.9943 |
| onion | DINOv2 cosine 1NN | 44.1487 | 43.9506 | 0.1982 | -8.2815 | 12.1076 |
| onion | DINOv2 + LR | 18.4920 | 17.6899 | 0.8021 | -2.5835 | 8.4462 |
| onion | ResNet18 cosine 1NN | 51.1150 | 49.2413 | 1.8737 | -3.4153 | 8.6911 |
| onion | ResNet18 + LR | 25.1928 | 23.3383 | 1.8545 | -5.5858 | 9.0400 |
| potato | Colour + LR | 2.4167 | 1.6767 | 0.7400 | -5.6667 | 5.3333 |
| potato | DINOv2 cosine 1NN | 6.7833 | 7.7367 | -0.9533 | -4.0000 | 2.9000 |
| potato | DINOv2 + LR | 2.8333 | 2.5867 | 0.2467 | -3.4667 | 4.8667 |
| potato | ResNet18 cosine 1NN | 4.9500 | 4.5700 | 0.3800 | -3.5000 | 3.7667 |
| potato | ResNet18 + LR | 2.4833 | 1.7133 | 0.7700 | -2.1000 | 6.5333 |


洋葱ResNet18和DINOv2的单来源减联合差平均为+1.8545和+0.8021个百分点，颜色模型为-4.5531个百分点。马铃薯均值差较小，但方向随划分变化。所有模型/任务差值的最小至最大范围均跨零。因此没有证据支持联合替换普遍放大或减弱自身目标差，也不能证明两种范围等效。联合比较同时改变了其他被选来源的替换政策，故配对差是情境敏感性，不能识别孤立的相互作用机制。单来源差本身以记录的零暴露训练背景为条件，不是生物身份的一般因果效应。

### S7.5 执行、独立重建和文件索引

DINOv2联合扩展新增1,260次逻辑回归拟合、1,260个最近邻模型条件和386,820条预测。单来源扩展新增12,300次逻辑回归拟合、8,200个最近邻模型条件和3,156,950条预测，两者均无拟合警告。这些是计算量，不增加新的植株、叶片、图像或采集地点。

另写实现从原始预测重建了全部新增差值。预定检查独立重训12个DINOv2联合逻辑回归条件和84个单来源逻辑回归条件，并核对12个联合及56个单来源最近邻条件。重建差异处于浮点精度。这是计算交叉核验，不是盲态人类专家审阅；新增96次逻辑回归核查是部分拟合，并非全部13,560次新增拟合。S6中的24次重训属于此前保留的实验。

`SUPPLEMENT_TABLE_INDEX.csv`列出完整补表数量、行数、大小与哈希；`V18_COMPLETE_RESULT_FILE_INDEX.csv`索引完整预测文件、拟合日志、分配/来源成员和冻结记录，不把大型预测档案重复塞入表文件。该索引的路径相对于本补充材料。`SUPPLEMENT_BUILD_v18.json`记录未改变的18份旧CSV与所有新增复制/派生证据。冻结JSON保留原始来源和时间，历史路径不会被改写成好像一直可移植。单来源复算说明为`../single_source/REPRODUCTION_AND_ESTIMANDS.txt`；脚本为`../code/run_single_source_controls.py`、`../code/run_dinov2_joint_controls.py`和`../code/compare_single_joint_context.py`。

### S7.6 保留的迁移图

补图S2为原先单独的洋葱迁移情境，保留为`../figures/Fig_S2_External_Context.pdf`，另有EPS、600dpi PNG和源CSV。全部点估计始终使用相同813条锁定COLD预测。图像单位区间是此前分析的示例；547组件的全部阳性图与700组件的空间图提供依赖敏感性，各使用3,000次固定预测组件自助重采样。这些条件区间不能证明植株独立或新增田间队列。原迁移结果及完整18份旧表均保留；将图移至补充材料不会修复或掩盖外部性能差距。

## 补充方法参考文献

[S1] Lowe DG (2004) Distinctive Image Features from Scale-Invariant Keypoints. International Journal of Computer Vision 60:91-110. https://doi.org/10.1023/b:visi.0000029664.99615.94

[S2] Fischler MA, Bolles RC (1981) Random sample consensus: a paradigm for model fitting with applications to image analysis and automated cartography. Communications of the ACM 24:381-395. https://doi.org/10.1145/358669.358692

## 完整机器可读数表索引

- `supplement_tables/S10_regional_all_24_cases.csv`: 24 rows

- `supplement_tables/S11_regional_aggregates.csv`: 16 rows

- `supplement_tables/S12_regional_intervals.csv`: 100 rows

- `supplement_tables/S13_regional_source_label_recalls.csv`: 88 rows

- `supplement_tables/S14_regional_class_support.csv`: 81 rows

- `supplement_tables/S15_regional_paired_contrasts.csv`: 60 rows

- `supplement_tables/S16_onion_dinov2_joint_summary.csv`: 640 rows

- `supplement_tables/S16_potato_dinov2_joint_summary.csv`: 512 rows

- `supplement_tables/S17_onion_single_target_panel_contrasts.csv`: 28500 rows

- `supplement_tables/S17_potato_single_target_panel_contrasts.csv`: 32400 rows

- `supplement_tables/S18_onion_single_target_class_split_means.csv`: 1050 rows

- `supplement_tables/S18_potato_single_target_class_split_means.csv`: 900 rows

- `supplement_tables/S19_onion_single_split_means.csv`: 300 rows

- `supplement_tables/S19_potato_single_split_means.csv`: 300 rows

- `supplement_tables/S1_onion_all_model_endpoint_panel_class_summaries.csv`: 1920 rows

- `supplement_tables/S1_potato_all_model_endpoint_panel_class_summaries.csv`: 1536 rows

- `supplement_tables/S20_onion_single_summary.csv`: 720 rows

- `supplement_tables/S20_potato_single_summary.csv`: 720 rows

- `supplement_tables/S21_context_all_target_contrasts.csv`: 10200 rows

- `supplement_tables/S22_context_target_class_split_means.csv`: 650 rows

- `supplement_tables/S23_context_split_means.csv`: 200 rows

- `supplement_tables/S24_context_summary.csv`: 60 rows

- `supplement_tables/S25_joint_all_five_models_plot_source.csv`: 40 rows

- `supplement_tables/S26_single_all_five_models_plot_source.csv`: 40 rows

- `supplement_tables/S27_context_display.csv`: 10 rows

- `supplement_tables/S28_onion_dinov2_joint_arm_contrasts.csv`: 12000 rows

- `supplement_tables/S28_potato_dinov2_joint_arm_contrasts.csv`: 9600 rows

- `supplement_tables/S29_onion_dinov2_joint_split_means.csv`: 1200 rows

- `supplement_tables/S29_potato_dinov2_joint_split_means.csv`: 960 rows

- `supplement_tables/S2_selected_probe_policy_contrast.csv`: 12 rows

- `supplement_tables/S3_selected_probe_absolute_BA.csv`: 12 rows

- `supplement_tables/S4_primary_original_class_policy_contrasts.csv`: 7 rows

- `supplement_tables/S5_locked_transfer_metric_intervals.csv`: 63 rows

- `supplement_tables/S5a_locked_transfer_point_estimates.csv`: 42 rows

- `supplement_tables/S5b_locked_external_display_all_models.csv`: 14 rows

- `supplement_tables/S6_locked_transfer_paired_models.csv`: 36 rows

- `supplement_tables/S7_locked_transfer_gaps.csv`: 42 rows

- `supplement_tables/S8_calibration_arms.csv`: 8 rows

- `supplement_tables/S9_calibration_changes.csv`: 4 rows
