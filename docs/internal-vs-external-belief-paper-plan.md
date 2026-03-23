# 内部空间表征 vs 外部空间信念：论文叙事与实验方案

## 1. 文档目的

本文档用于把当前 Ordinary-Bench 的 `scene belief reconstruction` 思路整理成一套可直接写进论文的叙事与实验蓝图。重点不是再解释一遍代码，而是回答下面几个论文层面的核心问题：

1. 这部分在论文中到底要主张什么，不主张什么。
2. 它如何支撑 “内部空间表征” 与 “外部空间信念” 的区分。
3. 它应该如何进入正文结构、图表系统和实验矩阵。
4. 指标如何制定，才能既严谨又不显得人为堆分数。
5. 现有代码资产如何复用，减少额外工程分叉。

默认论文定位遵循现有 NeurIPS D&B 方向：`benchmark + analysis tool`，而不是“以重建方法本身为主角”的方法论文。

---

## 2. 核心立场

### 2.1 一句话版本

我们不声称直接读取 VLM 的内部隐藏状态，而是通过密集空间 probing 和 scene reconstruction，研究模型在黑盒交互中外显出来的 `externalized spatial belief`，并把它作为内部空间表征的可操作近似。

### 2.2 三层对象

论文中统一使用下面三层对象，避免概念混淆：

- `Ground-truth scene`
  - 真实物理场景，来自可控合成 3D 场景与精确坐标。
- `Externalized spatial belief`
  - 从模型回答中提取约束并重建得到的场景级信念。
  - 这是本文真正测量的对象。
- `Internal spatial representation`
  - 模型内部可能存在的隐式空间结构。
  - 本文不直接读取，只通过行为层的外显信念间接逼近。

### 2.3 论文中要坚持的表述边界

建议全文固定使用下面的表述：

- “behaviorally inferred spatial belief”
- “externalized scene belief”
- “operational proxy of internal spatial representation”

应避免的强表述：

- “我们直接测量了内部空间表征”
- “重建结果就是模型内部世界模型”
- “本文揭示了 VLM 的真实隐藏状态”

更稳妥的写法是：

> 本文关注的是由行为外显、可由重建检验的场景信念，而不是对模型内部神经状态的直接解码。

---

## 3. 论文主叙事

### 3.1 论文要回答的不是“会不会做题”，而是“它到底看见了什么”

当前大多数 VLM 空间 benchmark 主要报告单题准确率。但单题准确率只能说明模型在孤立 probe 上答对了多少题，不能说明这些答案是否能够共同支持一个一致、可重建、并且与真实场景对齐的空间信念。

因此，本文的叙事不应停留在：

- 模型 QRR/TRR 准确率是多少
- 模型是否比随机好

而应升级为：

- 模型的局部判断能否联合构成一个 coherent scene belief
- 这个 belief 是否真正由视觉输入支撑
- 这个 belief 与 GT 场景相差多远
- 这种差距到底来自错误、冲突、欠约束，还是语言先验

### 3.2 三条核心 claim

正文建议围绕下面三条 claim 展开：

#### Claim 1: Accuracy is not belief fidelity

局部题目答对，并不代表模型形成了稳定、可重建、接近 GT 的场景信念。

这条 claim 对应论文中的第一个关键转折：从 item-level accuracy 转向 scene-level fidelity。

#### Claim 2: Belief can be self-consistent yet wrong

模型可能外显出一个几何上较自洽、甚至近似单模态的 belief，但该 belief 仍显著偏离真实场景。

这条 claim 很重要，因为它说明：

- “自洽” 不等于 “正确”
- “能重建” 不等于 “接近 GT”
- 仅靠 consistency 分数还不够

#### Claim 3: Visual grounding and language prior can be separated at the belief level

通过正确图像 / 错误图像 / 无图像三条件，不只可以比较 item accuracy，还可以比较 belief reconstruction 的变化，从而更有力地区分视觉贡献与语言先验。

这条 claim 是本文从普通 benchmark 上升为更强评估框架的关键。

---

## 4. 正文章节整合方式

### 4.1 Introduction

Introduction 中建议这样铺陈：

1. 现有 benchmark 主要测单题正确率。
2. 单题正确率不能区分 “看见了” 与 “猜对了”。
3. 更严格的问题是：模型是否形成了一个 coherent, reconstructable, visually grounded scene belief。
4. 本文通过 controlled probing + scene belief reconstruction + A/B/C conditions 来回答这个问题。

必须显式加入一句边界声明：

> We do not claim direct access to the model's internal neural state; instead, we study the externalized scene belief implied by its behavior.

### 4.2 Scene Belief Reconstruction

这一节的定位必须克制：

- 它是 benchmark 的分析工具
- 它服务于评估协议
- 它不是一篇独立方法论文的主角

这一节只需要完成三件事：

1. 说明如何从 QRR/TRR/FDR answers 提取约束。
2. 说明如何通过 2D gauge-fixed optimization 重建 belief。
3. 说明 scene-level metrics 的物理意义。

### 4.3 Results & Analysis

这部分建议拆成三个紧密相连的小节：

#### 6.2 Scene Belief Fidelity

核心问题：

- 局部准确率和场景级 fidelity 的关系是什么？
- 哪些模型在 item level 看起来不错，但在 scene level 失败？

#### 6.3 Visual Grounding vs Language Prior

核心问题：

- belief 是否真的因正确图像而改善？
- 无图像条件下是否仍能出现“看起来会做题”的假象？

#### 6.4 Consistency and Failure Taxonomy

核心问题：

- 失败来自冲突、欠约束、还是伪一致但错误的 belief？
- 不同 failure mode 在几何上如何呈现？

### 4.4 Discussion

Discussion 要收束在下面这个逻辑：

- 本文没有直接证明内部表征长什么样。
- 但本文给出了黑盒条件下最强的行为层证据。
- 如果一个模型在 dense probing 下长期不能外显出 coherent scene belief，那么至少不能说它具备稳定的空间理解。

---

## 5. 实验矩阵

### 5.1 四个研究问题

建议把实验组织成四个明确的研究问题，而不是按脚本或指标堆叠。

#### RQ1. 局部准确率是否能预测场景级信念质量？

目标：

- 检验 accuracy 和 scene-level fidelity 是否一致

数据：

- 全模型
- 主 split
- belief reconstruction

输出：

- per-model accuracy summary
- per-scene accuracy vs NRMS / tau / CSR 散点图
- complexity bucket 下的对比

预期结论：

- 二者相关但显著不等价
- scene complexity 增长后脱钩更明显

#### RQ2. 模型外显空间信念与真实场景相差多远？

目标：

- 直接测量 belief 与 GT 的几何差距

数据：

- belief reconstruction + GT alignment

输出：

- NRMS
- Kendall tau
- reconstructable rate
- single-mode rate

预期结论：

- 局部题目上有非平凡准确率，但 belief fidelity 远弱于表面 impression

#### RQ3. 这种 belief 是否真正由视觉输入支撑？

目标：

- 区分视觉 grounding 与语言先验

数据：

- A = 正确图像
- B = 错误图像
- C = 无图像

输出：

- item-level VIG
- belief-level VIG
- A/B/C qualitative storyboard

预期结论：

- 有些模型可以在 C 条件保持不低的 item accuracy，但 belief fidelity 几乎不提升

#### RQ4. belief 失败是“信息不足”还是“信息错误/矛盾”？

目标：

- 分解失败来源

数据：

- belief reconstruction
- oracle reconstruction
- random baseline reconstruction

输出：

- solver floor
- belief-oracle gap
- random margin
- conflict / contradiction statistics

预期结论：

- 不同模型失败机制不同
- 有些是“信息不够”，有些是“信息错得很一致”，有些是“整体冲突”

### 5.2 三条 reconstruction 轨必须同时保留

#### 1. Belief reconstruction

- 使用模型原始回复
- 这是主口径
- 代表模型的 externalized scene belief

#### 2. Oracle reconstruction

- 使用 GT answers
- 作用不是“替代 belief”
- 而是估计 scene intrinsic difficulty 与 solver error floor

#### 3. Random baseline reconstruction

- 使用随机回答
- 用来回答：
  - 模型是否真的比瞎猜更接近 scene-level structure
  - 看起来有信号的 belief 是否只是统计假象

---

## 6. 指标体系

### 6.1 原则

指标设计遵循三个原则：

1. 尽量少发明复合分数。
2. 每个指标只表达一个清晰概念。
3. 主文使用少量正交指标，更多诊断指标放附录。

### 6.2 主指标

#### Item-level

- `QRR accuracy`
- `TRR accuracy`
- `FDR accuracy`（如纳入主实验）

作用：

- 只表示局部答题能力
- 不表示全局场景信念

#### Constraint-level

- `CSR_QRR`
- `CSR_TRR`
- `symbolic infeasibility rate`
- `contradiction/conflict rate`

作用：

- 表示 belief 是否自洽

#### Scene-level fidelity

- `NRMS`
- `Kendall tau`

作用：

- 表示 belief 与 GT 的几何差距

#### Scene-level ambiguity

- `K_geom`
- `spread`
- `reconstructable rate`
- `single-mode rate`

作用：

- 表示 belief 是唯一、模糊、多解还是欠约束

### 6.3 两个必须新增的 derived metrics

只建议新增两个真正有解释力的派生指标。

#### 1. Belief-Oracle Gap (BOG)

建议定义：

- `BOG_nrms = NRMS_belief - NRMS_oracle`
- `BOG_tau = Tau_oracle - Tau_belief`

解释：

- 如果 BOG 很大，说明退化主要来自模型回答本身，而不是 scene 太难或 solver 太弱。

#### 2. Belief-level Visual Information Gain (VIG-B)

建议定义：

- `VIG_B_nrms = NRMS_C - NRMS_A`
- `VIG_B_tau = Tau_A - Tau_C`

必要时补充：

- `NRMS_B` 和 `Tau_B`

解释：

- 衡量视觉输入是否真正改善了外显空间信念，而不是只改善表面题分。

### 6.4 failure taxonomy

每个 scene 建议最终归入一个 failure label，作为 qualitative analysis 的统一骨架：

- `faithful`
  - 可重建、单模态、且接近 GT
- `coherent_but_wrong`
  - 自洽，但明显偏离 GT
- `multimodal`
  - 可重建，但 K_geom > 1
- `contradictory`
  - 约束存在明显冲突，无法统一解释
- `underconstrained`
  - 约束不足，解空间过大

这套标签比只报平均分更重要，因为它直接决定案例图和 discussion 的说服力。

---

## 7. 可视化方案

### 7.1 正文必须出现的三张核心图

#### Fig A. GT / Oracle / Belief / Overlay

建议选 3 个典型 scene，分别展示：

- faithful
- coherent but wrong
- contradictory 或 multimodal

每个案例最少包含：

- 原图
- GT top-down
- belief reconstruction
- oracle reconstruction
- overlay with displacement arrows

这张图用于直接展示：

- belief 和 GT 的差异
- belief 和 oracle 的差异
- “能重建” 与 “重建得对” 的区别

#### Fig B. Accuracy vs Belief Fidelity

形式：

- 每个点一个 scene
- x 轴为 item accuracy
- y 轴为 NRMS 或 Kendall tau

可按模型着色，或分面展示。

目的：

- 直观证明 `accuracy != belief fidelity`

#### Fig C. A/B/C Belief Storyboard

同一 scene 在三个条件下并排展示：

- A: correct image
- B: wrong image
- C: no image

每个 panel 显示：

- 重建布局
- NRMS / tau / CSR
- 关键变化箭头或 distortion overlay

目的：

- 让“视觉 grounding”和“语言先验”从抽象统计变成可见几何现象

### 7.2 补充图

可放正文次图或附录：

- consistency vs NRMS 校准图
- violin/box plot of NRMS / CSR / K_geom
- conflict graph 示例
- failure taxonomy gallery

### 7.3 图像资产生成建议

优先复用现有资产：

- SVG top-down comparison
- Blender render scripts
- reconstruction summary JSON

现有 `analysis/generate_svg_examples.py`、`analysis/visualize_svg.py`、`analysis/generate_blender.py` 已经具备较强基础，适合作为论文 figure 生产线，而不是另起一套渲染系统。

---

## 8. 表格系统

建议这部分只保留两张主表，防止主文信息过载。

### Table A. Belief fidelity summary

列建议：

- Model
- QRR acc
- TRR acc
- Reconstructable rate
- Single-mode rate
- CSR_QRR
- CSR_TRR
- NRMS
- Kendall tau
- K_geom mean

作用：

- 给出“局部正确”与“场景级 fidelity” 的正面对照

### Table B. Visual grounding and gap analysis

列建议：

- Model
- NRMS_A
- NRMS_B
- NRMS_C
- VIG_B_nrms
- Tau_A
- Tau_C
- VIG_B_tau
- BOG_nrms
- random margin

作用：

- 把视觉贡献、语言先验和 belief-oracle gap 合并成一张真正有论文价值的表

---

## 9. 现有代码资产如何复用

当前仓库已经有不少可以直接进入论文生产线的资产，建议不要重复造轮子。

### 9.1 已有分析能力

- `analysis/run_analysis.py`
  - 已经串起 accuracy / reconstruction / consistency
- `analysis/information_gain.py`
  - 已有 VIG 与 error decomposition 雏形
- `analysis/baselines.py`
  - 已支持 random baseline 与 GT baseline 思路
- `analysis/reconstruct_scenes.py`
  - 已支持 belief reconstruction 批量化

### 9.2 已有可视化能力

- `analysis/generate_svg_examples.py`
- `analysis/visualize_svg.py`
- `analysis/generate_blender.py`

### 9.3 现有论文写作资产

- `papers/neurips_dab_paper_structure.md`
- `papers/neurips_ed_recon_section.tex`
- `papers/neurips_ed_experiments_section.tex`
- `papers/neurips_ed_intro_draft_20260312.tex`

最优路线不是从零写，而是：

1. 先用本文档统一叙事和术语。
2. 再把现有 tex 草稿向这个口径收束。
3. 同时补齐 belief-level figure/table 所需的汇总输出。

---

## 10. 交付顺序

用户要求是“先写论文叙事，再根据论文叙事做实验设计”，因此建议固定为两阶段。

### Phase 1: 论文叙事包

这一步先不追求新增代码，而是锁定 paper story：

- 统一术语
- 统一 claim
- 明确章节落位
- 明确 figure/table message
- 明确哪些话能说，哪些话不能说

本文件本身就是这一阶段的基础。

### Phase 2: 实验规格包

在叙事锁定后，再落地为实验与数据产出规范：

- scene-level diagnostics schema
- BOG / VIG-B 汇总
- failure taxonomy 标签
- figure selection rule
- table generation rule

### Phase 3: 结果生产与写作回填

最终顺序建议是：

1. 跑主模型完整 belief reconstruction
2. 跑 oracle / random baseline
3. 跑 A/B/C 条件
4. 自动汇总表格数据
5. 选出典型案例并生成 SVG / Blender 图
6. 回填 `§4` 与 `§6.2-6.4`

---

## 11. 这部分在论文中真正支撑了什么

如果写法正确，这部分能稳定支撑下面几件事：

1. Ordinary-Bench 不只是一个新的空间 QA 数据集，而是一个更严格的 scene-level evaluation framework。
2. 单题正确率不足以判断模型是否形成 coherent scene belief。
3. 黑盒 VLM 的外显空间信念可以被系统地诊断为：
   - faithful
   - coherent but wrong
   - contradictory
   - multimodal
   - underconstrained
4. 正确图像、错误图像、无图像三条件能够把视觉贡献与语言先验在 belief level 上分开。
5. 对空间能力的评估应该从 “答对多少题” 上升到 “行为所蕴含的场景到底是什么”。

---

## 12. 建议的下一步

如果以此文档为准继续推进，下一步最合理的是：

1. 把现有英文/中文 paper 草稿中所有 “internal representation” 过强表述统一修正。
2. 基于本文档整理一个 scene-level diagnostics 输出 schema。
3. 生成第一版 Table A / Table B 所需的聚合字段。
4. 选 6-10 个 scene，人工筛出 3 个最能说明问题的核心案例图。
5. 再进入具体实现与跑数阶段。

这会比先写代码、再倒推论文说法更稳，也更符合当前论文定位。
