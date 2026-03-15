# Ordinary-Bench 投稿版收敛大纲

更新时间：2026-03-12

目标：给出一个**更适合 NeurIPS Evaluation / Datasets track** 的论文版本。  
原则：不追求把所有想法塞进一篇论文，而是围绕当前仓库最成熟、最能打动 reviewer 的资产收敛成一个主问题。

---

## 1. 一句话判断

最适合当前仓库的投稿版本，不是：

- `benchmark + reconstruction + dynamic video + training plasticity` 的四线并行论文

而是：

- `benchmark + scene belief reconstruction + visual contribution separation`

动态时空实验可以保留，但只能作为第二主实验或强补充，不能和训练一起并列为主贡献。

---

## 2. 推荐论文定位

### 推荐 track 定位

`NeurIPS Evaluation / Datasets Track`

### 推荐论文类型

这是一个：

1. **受控评测基准论文**  
2. **行为学评估方法论文**  

而不是：

1. world model 论文  
2. representation learning 理论论文  
3. RL training 论文  

### 推荐主张

本文提出一个受神经科学行为实验启发的 VLM 时空知觉评测框架。它不仅测模型“答对了多少题”，还将模型输出转化为时空约束，进一步重建其外显场景信念，并通过错图/无图等干预实验分离视觉信息与语言先验。

---

## 3. 建议保留的三大贡献

### Contribution 1: Ordinary-Bench 作为受控时空知觉 benchmark

- Blender 生成的受控 3D 渲染场景
- 静态空间关系主线：QRR + TRR
- 多视角设置
- 动态短视频扩展作为时空实验补充

这部分回答：

> VLM 在受控世界里的空间和时间知觉到底表现如何？

### Contribution 2: Scene Belief Reconstruction

- 把模型回答转为约束集
- 从约束集重建 2D / 2.5D 外显场景信念
- 用 `CSR / NRMS / Kendall tau / K_geom / spread` 取代单纯准确率

这部分回答：

> 模型给出的局部正确答案，是否真的能拼成一个一致的场景？

### Contribution 3: Visual Contribution Separation

- 正确图像 / 错误图像 / 无图像 三条件实验
- selective QA / consistency 分析
- `VIG` 量化视觉信号的真实贡献

这部分回答：

> 模型的空间回答是来自视觉感知，还是来自语言先验？

---

## 4. 明确建议删掉或降级的内容

### 不建议作为主贡献写入正文

#### 4.1 训练可塑性整节

原因：

- 这是另一篇论文的材料
- 会把 benchmark 论文变成 benchmark + training 混合稿
- reviewer 会要求更多 ablation、更多训练细节、更多 compute 报告
- 还会稀释 evaluation/dataset 的核心价值

建议处理：

- 正文删除
- 如果已经有初步结果，可以放附录最后一节，作为 future direction

#### 4.2 过强的“内部表征解码”表述

不要写：

- reverse-engineering the internal representation
- decoding the VLM’s internal scene representation
- neurophysiology-equivalent decoding

更稳妥的写法：

- reverse-engineering the model’s **externalized scene belief**
- a **behavioral** reconstruction of VLM spatiotemporal belief
- a neuroscience-inspired **behavioral evaluation protocol**

#### 4.3 过大的 full 4D / full 3D 叙事

当前动态部分更合适的表述是：

- `rendered 3D environments with object-centric 2D/2.5D spatiotemporal annotations`

不建议直接写：

- full 4D world understanding benchmark
- full 3D physical reasoning benchmark

---

## 5. 推荐标题

### 首选标题

**What Do Vision-Language Models Really See? Ordinary-Bench: Evaluating Spatial Belief via Controlled Probing and Scene Reconstruction**

### 备选标题

**Ordinary-Bench: Beyond Accuracy in Evaluating Vision-Language Models’ Spatial Belief**

**Seeing or Guessing? Evaluating VLM Spatial Belief with Controlled Probing, Reconstruction, and Visual Ablation**

### 不建议的标题方向

- `Spatial Mind`
- `Neurophysiology-Inspired ...` 放主标题
- `4D World Model` 放主标题

这些词太大，容易让 reviewer 先提高预期，再因为证据不足扣分。

---

## 6. 推荐摘要结构

摘要建议严格控制成四句半：

1. 现有 VLM 空间与视频评测过于依赖单题正确率，无法判断模型是否形成稳定时空表征。
2. 我们提出 Ordinary-Bench，在受控 3D 渲染环境中构造静态空间和动态时序探测任务。
3. 我们进一步提出 scene belief reconstruction，将 VLM 输出转为时空约束并重建外显场景信念，同时通过正确图像/错误图像/无图像三条件实验估计视觉信息增益。
4. 实验表明，VLM 在局部题目上可取得非平凡准确率，但在场景级重建、一致性和时间知觉上暴露出系统缺陷，且复杂场景下更容易退化为语言先验驱动的伪空间判断。
5. Ordinary-Bench 提供了一个从“答对多少题”转向“模型究竟看见了什么”的评测框架。

---

## 7. 推荐正文结构

## 1. Introduction

目标：

- 提出问题，不讲太多宏大类比
- 让 reviewer 在第一页就明白：这是 benchmark + evaluation methodology

建议结构：

1. 现有 VLM 空间/视频 benchmark 的局限：过度依赖 accuracy
2. benchmark 缺少对 scene-level consistency 和 visual contribution 的分析
3. 我们的方法：controlled probing + scene belief reconstruction + visual ablation
4. 贡献列表

### 引言中建议保留的 claim

- 我们不是在估计“模型内部真实神经状态”
- 我们在研究的是可由行为外显、可由重建检验的 scene belief

## 2. Related Work

只保留 4 组文献：

1. VLM spatial benchmark
2. video / temporal benchmark
3. selective answering / uncertainty / hallucination
4. ordinal embedding / distance geometry / rigidity

不要在 related work 中展开过多神经科学历史，否则像 essay，不像 benchmark 论文。

## 3. Ordinary-Bench

这是 benchmark 论文的核心节。

应包含：

- 数据生成
- 场景组成
- 任务定义
- 静态任务：QRR / TRR
- 动态任务：顺序、速度、轨迹
- 多视角设定
- 数据划分

必须额外补：

- dataset release format
- annotation schema
- licensing / usage
- metadata / Croissant

## 4. Behavioral Reconstruction of Scene Belief

这是方法学核心。

建议结构：

1. 从回答到约束
2. 从约束到 2D scene reconstruction
3. 解空间分析
4. 场景级指标

重点强调：

- 这是 evaluation methodology
- 不是通用 world model reconstruction algorithm paper

## 5. Separating Vision from Language

这一节单独讲：

- 条件 A: correct image
- 条件 B: wrong image
- 条件 C: no image
- VIG 定义
- consistency / abstention / selective QA

这会是这篇论文最容易打 reviewer 的一节，因为它回答了 benchmark 的科学性问题。

## 6. Experiments

实验只分 4 块：

1. 静态空间能力
2. scene belief reconstruction
3. visual contribution separation
4. dynamic temporal extension

不要再单开 training 节。

## 7. Discussion

建议只讨论：

- 模型“看见”和“猜测”的边界
- 局部正确与全局一致性的差距
- 时间知觉为什么更难
- benchmark 的局限与未来扩展

## 8. Release, Reproducibility, Ethics

这是 NeurIPS E&D 版本必须新增的一节。

应明确写：

- data hosting
- code availability
- model access notes
- annotation generation pipeline
- metadata and documentation
- limitations / misuse / compute

---

## 8. 推荐图表精简版

不建议 11 个 figure + 6 个 table。

正文建议控制到：

### Figure

1. 方法总览图  
2. GT vs reconstructed belief 的核心案例图  
3. 三条件 A/B/C 重建对比图  
4. 动态任务结果图  
5. 可选：复杂度曲线或 distortion taxonomy

### Table

1. 主 benchmark 准确率表  
2. reconstruction 指标表  
3. three-condition / VIG 表  
4. dynamic temporal results 表  
5. 可选：human baseline / consistency 表

---

## 9. 实验上必须补强的点

如果不补，投稿风险很高。

### 9.1 Human baseline

至少覆盖：

- 静态空间判断
- 时间顺序判断
- 遮挡补全或未来想象中的一个任务

原因：

- 对“人类直观但 VLM 困难”的 claim 至关重要

### 9.2 非单一家族模型

不建议只用 Qwen 系列。

至少再加：

- GPT-4o / GPT-4.1 类
- Gemini 类
- Claude 类

原因：

- D&B reviewer 会天然怀疑 benchmark 是否只适配单一模型家族

### 9.3 动态部分要收缩

动态实验不需要做成一个完整新 benchmark。

建议只做这 3 类：

1. frame order / temporal order
2. velocity ranking
3. trajectory type / endpoint prediction

这样足够支撑“时空扩展”而不会把论文写散。

### 9.4 selective QA 至少要有一个版本

如果最终没有 selective QA 实验，也要至少保留：

- abstain / unknown 机制
- consistency-based reliability

否则三条件实验会显得不完整。

---

## 10. NeurIPS E&D 需要单独补的 submission checklist

这部分通常不是 reviewer 最爱读的，但缺了会直接掉印象分。

### 必须准备

1. 数据下载地址或 review-time access
2. benchmark code 与 evaluation script
3. 任务和 annotation schema 文档
4. 数据许可、模型调用依赖、API 说明
5. metadata / Croissant
6. dataset card / benchmark card
7. ethics / limitations / responsible use
8. 复现实验脚本

### 建议准备

1. 最小可运行子集
2. 一个 notebook 或 demo
3. 所有 figure/table 生成脚本

---

## 11. 推荐的最终贡献写法

建议最终 contribution 只写三条：

1. 我们提出 Ordinary-Bench，一个面向 VLM 时空知觉的受控 benchmark，覆盖静态空间关系、多视角输入和动态时序扩展。
2. 我们提出 scene belief reconstruction，将 VLM 的离散回答转化为场景级、可重建、可量化的外显空间表征，从而超越单题准确率。
3. 我们提出三条件视觉分离实验与一致性分析框架，量化视觉输入相对于语言先验的真实贡献，并揭示模型在复杂场景和时间任务中的系统退化。

---

## 12. 这个版本不该再写什么

下面这些内容即使有，也不建议写进主文核心：

- GRPO / SFT 训练前后对比
- 大量神经科学名词堆砌
- full 4D / full 3D physical world 的大 claim
- 过多内部表征解码表述
- 还没有跑完的超大动态计划

---

## 13. 最后给作者的执行建议

如果你要真正朝投稿推进，建议按下面顺序写：

1. 先固定标题、摘要、三条贡献。
2. 再固定正文结构：3 / 4 / 5 / 6 四个核心节。
3. 再反推“正文只需要哪 4-5 张图、哪 4 张表”。
4. 最后才决定动态部分保留多少。

一句话版本：

> 先把论文写成一篇锋利的 benchmark/evaluation paper，再决定要不要把它扩成更大的认知科学叙事。

