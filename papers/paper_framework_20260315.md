# Ordinary-Bench 论文框架收敛稿

更新时间：2026-03-15

这份文档基于 `papers/` 目录中的现有 outline、章节草稿、文献整理和仓库实现状态，给出一份适合直接推进到 Overleaf 的收敛版论文框架。

## 1. 论文一句话

这篇论文最稳的主张不是“VLM 学会了 3D/4D 世界模型”，而是：

> 我们提出一个受控 3D 渲染 benchmark 与行为学评估框架，用密集序数探测、场景级重建和三条件视觉分离实验，回答 VLM 究竟看见了什么，而不只是它答对了多少题。

## 2. 推荐定位

- 投稿方向：NeurIPS Datasets & Benchmarks / Evaluation 风格
- 论文类型：`benchmark + evaluation methodology`
- 不是：训练论文、内部表征解码论文、full 4D world model 论文

## 3. 主线必须收敛到三件事

### Contribution 1: Ordinary-Bench benchmark

- 受控 3D 渲染环境
- 700 个场景，复杂度 split 为 `n=4..10`
- 两类静态空间任务：QRR 和 TRR
- 支持单视角、多视角，以及后续动态扩展

### Contribution 2: Scene belief reconstruction

- 将 VLM 离散回答映射为空间约束
- 在 2D / 2.5D 层面重建其外显场景信念
- 用 CSR、NRMS、Kendall tau、K_geom、spread 替代单纯 accuracy

### Contribution 3: Separating vision from language

- 正确图像 / 错误图像 / 无图像三条件协议
- 传递一致性、互逆一致性、随机基线
- 用 VIG 衡量视觉输入对重建质量的真实贡献

## 4. 当前最稳的中心论点

论文的核心发现应该围绕下面三句话组织：

1. 单题正确率会高估 VLM 的空间能力。
2. 一旦转到场景级重建，很多“局部正确”的回答会暴露为全局不一致、几何失真或欠约束。
3. 只有把正确图像、错图和无图像放在同一协议里比较，才能分离视觉贡献与语言先验。

## 5. 应该降级到附录或未来工作的内容

- `data-gen-dynamic/` 对应的动态时序实验
- 训练可塑性、SFT、GRPO 等训练叙事
- 过强的 neuroscience / internal decoding claim
- “full 3D physical world” 或 “full 4D world model” 的大叙事

动态模块可以保留，但更适合作为：

- 附录中的扩展实验
- 或正文 Discussion 里的 future direction

## 6. 正文最终结构

### 1. Introduction

目标：

- 提出 gap：accuracy 不能回答模型是否形成 coherent scene belief
- 引出本文方法：controlled probing + reconstruction + visual ablation
- 明确边界：研究的是 externalized scene belief，不是模型内部真实神经状态

现状：

- 已有高质量英文初稿，可直接沿用

### 2. Related Work

保留五组文献：

1. VLM spatial benchmarks
2. controlled synthetic benchmarks
3. language priors / hallucination
4. selective answering / uncertainty
5. ordinal embedding / geometry / rigidity

写法要求：

- 每组 2-3 句
- 不展开大段神经科学历史
- 核心是说明 Ordinary-Bench 补上了“scene-level coherence + visual contribution separation”的空缺

### 3. Ordinary-Bench: Data & Protocol

包含：

- scene generation
- QRR / TRR definitions
- batch JSON protocol + ReAct correction
- three-condition setup
- dataset organization / split / release plan

写作原则：

- 所有数字和流程尽量跟当前代码实现对齐
- 对尚未正式发布的 license / dataset card / Croissant 用 future-facing 表述

### 4. Scene Belief Reconstruction

定位必须写清：

- 这是 benchmark 的 analysis tool
- 不是独立算法论文

包含：

- responses -> constraints
- gauge fixing + optimization
- scene-level metrics
- validation under GT constraints

### 5. Separating Vision from Language

这一节是论文最能打 reviewer 的方法学部分，建议单独成节。

包含：

- three-condition protocol 的科学目的
- VIG 定义
- transitivity / reciprocity consistency
- random baselines 与 belief faithfulness gap
- selective answering 的接口位置

### 6. Experiments

正文里分成四个分析块即可：

1. question-level accuracy
2. scene belief reconstruction
3. visual contribution separation
4. consistency analysis

如果动态实验来不及补齐，不要塞进正文主线。

### 7. Discussion and Limitations

集中讨论：

- VLM 看见了什么，没看见什么
- 局部正确 vs 全局一致性的差距
- synthetic-to-real 的外推边界
- 2D reconstruction、API black-box、TRR 粒度等局限

### 8. Release, Reproducibility, and Ethics

这是 D&B 风格的必需节。

包含：

- release package 组成
- code / data / prompt / split / metadata
- compute and API cost
- synthetic data ethics 和 misuse boundary

### 9. Conclusion

回扣一句话：

> Ordinary-Bench 的价值不只是多一个空间 benchmark，而是把“模型答对了多少题”升级为“模型究竟外显了怎样的场景信念”。

## 7. 当前可直接开写的部分

已经有足够材料直接进入正文的部分：

- Abstract 初稿
- Introduction
- Related Work
- Section 3 benchmark
- Section 4 reconstruction
- Section 5 methodology
- Discussion / Limitations
- Release / Reproducibility / Ethics
- Conclusion

仍然需要数据冻结后再定稿的部分：

- 实验模型 roster
- 所有表格中的最终数值
- 统计显著性检验
- 图 2/3/4/5/6 的最终版本

## 8. 本轮写作策略

本轮应当做的是：

1. 生成一份可直接上传 Overleaf 的 `main.tex`
2. 复用已有 `intro / benchmark / reconstruction` 草稿
3. 补写缺失章节的首版正文
4. 为实验节留下不伪造结果的 skeleton

本轮不该做的是：

1. 编造未跑完的实验数值
2. 把动态模块强行升格为主贡献
3. 在正文里做过大的理论或 neuroscience claim
