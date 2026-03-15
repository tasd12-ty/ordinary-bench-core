# NeurIPS 2026 D&B Track — 推荐论文结构

更新时间：2026-03-12

---

## 推荐标题

**首选：**
> What Do Vision-Language Models Really See? Ordinary-Bench: Evaluating Spatial Belief via Controlled Probing and Scene Reconstruction

**备选：**
> Ordinary-Bench: Beyond Accuracy in Evaluating Vision-Language Models' Spatial Belief

**不建议：**
- 不要在主标题中出现 Spatial Mind、Neurophysiology-Inspired、4D World Model 等词

---

## 推荐摘要（四句半结构）

1. 现有 VLM 空间与视频评测过于依赖单题正确率，无法判断模型是否形成稳定时空表征。
2. 我们提出 Ordinary-Bench，在受控 3D 渲染环境中构造静态空间探测任务（QRR 距离排序 + TRR 方位判断），覆盖 700 场景、327K 探测点。
3. 我们进一步提出 scene belief reconstruction，将 VLM 输出转为时空约束并重建外显场景信念，同时通过正确图像/错误图像/无图像三条件实验估计视觉信息增益（VIG）。
4. 实验表明，VLM 在局部题目上可取得非平凡准确率，但在场景级重建、一致性和结构保持上暴露出系统缺陷，且复杂场景下更容易退化为语言先验驱动的伪空间判断。
5. Ordinary-Bench 提供了一个从"答对多少题"转向"模型究竟看见了什么"的评测框架。

---

## 三大贡献（正式写法）

1. 我们提出 **Ordinary-Bench**，一个面向 VLM 空间知觉的受控 benchmark，使用 Blender 渲染 700 个 3D 场景，覆盖 QRR（距离排序）和 TRR（方位判断）两类序数空间关系，支持复杂度梯度（4–10 物体）和多视角输入。
2. 我们提出 **scene belief reconstruction**，将 VLM 的离散回答转化为场景级、可重建、可量化的外显空间表征（CSR、NRMS、Kendall τ、K_geom），从而超越单题准确率揭示 VLM 的真实几何能力。
3. 我们提出 **三条件视觉分离实验**（正确图像/错误图像/无图像）与一致性分析框架，量化视觉输入相对于语言先验的真实贡献（VIG），并揭示模型在复杂场景中的系统退化。

---

## 正文结构详细指导

### § 1. Introduction (1.0 页)

**逻辑流：**

1. **缺口**: 现有 VLM 基准将视觉理解压缩为离散问答分数，"答对 56% 的空间题"并不意味着模型理解了空间结构。
2. **问题**: 我们缺少回答"VLM 是否形成了一致、可重建的场景表征"的评测方法。
3. **方法**: 本文借鉴行为反演思路：用序数空间探测 → 约束提取 → 场景重建 → 三条件消融，全链路分析 VLM 的空间信念。
4. **贡献列表**: 三条（见上）。

**必须包含的声明：**
> 本文关注的不是 VLM 的"内部真实神经状态"，而是可由行为外显、可由重建检验的 scene belief。

**引用重点：** SpatialVLM, BLINK, MMStar, VLInd-Bench, CLEVR

### § 2. Related Work (0.8 页)

5 个子方向，每个 2-3 句：

| 子方向 | 核心引用 | 要点 |
|--------|---------|------|
| VLM 空间基准 | SpatialVLM, VSR, What'sUp, SpatialBench | 现有基准侧重分类准确率，缺少场景级分析 |
| 合成基准方法论 | CLEVR, CLEVRER | 受控合成数据的科学价值（精确 GT + 可控复杂度） |
| VLM 可靠性/幻觉 | POPE, VLInd-Bench, MMStar | 语言先验污染视觉判断 |
| 选择性回答 | Reliable VQA, Selective VQA | 允许 abstain 的评测协议 |
| 序数嵌入/几何理论 | GNMDS, EDM, Bearing Rigidity | 从比较到坐标的理论基础 |

### § 3. Ordinary-Bench: Data & Protocol (1.5 页)

**3.1 场景生成**
- Blender 渲染管线：物体随机放置 → 材质/光照随机化 → 单视角 + 多视角输出
- 700 场景 = 70 配置 × 10 种随机种子，按 n=4,5,...,10 分 7 个 split
- 每场景输出：3D 坐标 JSON + PNG 渲染图

**3.2 任务定义**
- QRR (Quaternary Relative Relation): "A-B 与 C-D 哪对更近？" → <, ~=, >
- TRR (Ternary Relative Relation): "站在 ref1 看向 ref2，target 在几点钟方向？" → 1-12
- 总计 327,957 探测点 (130K QRR + 197K TRR)

**3.3 协议设计**
- Batch 提问 → JSON 格式回答 → ReAct 纠正循环
- 三条件实验：A=正确图像，B=错误图像，C=无图像

**3.4 数据集文档** (D&B 必须)
- 数据格式、标注 schema、许可证
- Croissant metadata / dataset card
- 下载地址与复现脚本

### § 4. Scene Belief Reconstruction (1.5 页)

**重要定位：这是 benchmark 的分析工具，不是独立的方法贡献。**

**4.1 从回答到约束**
- QRR 回答 → 距离序约束 (P_dist DAG)
- TRR 回答 → 方位角约束 (P_ang arcs)

**4.2 约束优化**
- 3-anchor gauge fix: a=(0,0), b=(1,0), y_c≥0
- QRR: log-domain softplus loss
- TRR: sector-tolerance loss
- L-BFGS-B + 多重启

**4.3 场景级指标**
- CSR (Constraint Satisfaction Rate): QRR/TRR 约束满足比
- NRMS (Normalized RMS Error): 与 GT 的 Procrustes 距离
- Kendall τ: 距离序保持度
- K_geom: 几何保真度
- Spread: 解空间多模态性

**写法注意：**
- 强调这是 evaluation methodology 的一部分
- 用 "the benchmark provides..." 而非 "we propose a novel algorithm..."

### § 5. Experimental Setup (0.5 页)

**5.1 模型列表**
- 至少 6 个模型，覆盖 3+ 模型家族
- 推荐：GPT-4o, Gemini 2.0 Flash, Claude Sonnet, Qwen3-VL-235B, Qwen3-VL-30B, LLaVA-OneVision/InternVL2
- 开源 + 闭源混合

**5.2 人类基线**
- 50-100 场景，3+ 标注者
- 同样使用 QRR/TRR 格式

**5.3 评测配置**
- 单视角 (主实验) + 多视角 (消融)
- temperature=0, batch 模式

### § 6. Results & Analysis (2.5 页 — 最重的部分)

**6.1 主准确率结果 (Tab 1, Fig 4)**
- 模型 × split × QRR/TRR 矩阵
- 准确率 vs 物体数衰减曲线
- 与人类基线对比

**6.2 场景信念重建分析 (Tab 2, Fig 2, Fig 5)**
- CSR/NRMS/τ/K_geom 统计表
- GT vs 重建配置对比（3-4 个典型场景）
- 质量分布 violin plot

**6.3 视觉增益分析 (Tab 3, Fig 3)**
- 三条件 A/B/C 准确率对比
- VIG 统计 + 显著性检验
- 同场景在三条件下的重建对比图

**6.4 一致性分析 (Tab 4)**
- 传递性违反率
- 互反性违反率
- 与随机基线对比

**6.5 错误归因分析 (Fig 6)**
- 感知层失败：根本没读到局部几何
- 结构层失败：局部答案互相矛盾
- 典型扭曲模式可视化

### § 7. Discussion & Limitations (0.7 页)

- VLM 看见了什么 vs 没看见什么
- 局部正确 vs 全局一致性的系统性差距
- 合成 vs 真实场景的 generalization 问题
- 当前方法的局限：2D/2.5D 重建、API 黑盒限制
- 未来扩展：动态时间知觉、真实世界场景、开源模型内部分析

### § 8. Conclusion (0.5 页)

回扣一句话结论：
> 即使在受控的 3D 渲染环境和对人类而言高度直观的空间任务中，当前 VLM 也只能给出局部正确、全局不稳的回答；只有将其行为转化为可重建的空间信念，我们才能真正回答"模型究竟看见了什么"。

---

## 附录结构（不限页数）

| 附录 | 内容 |
|------|------|
| A | 数据集详细文档 (Croissant, datasheet) |
| B | 场景生成细节 (Blender 参数、物体列表、材质) |
| C | 重建算法细节 (loss 函数推导、超参数) |
| D | 全模型逐 split 结果表 |
| E | 更多重建可视化 |
| F | 动态时间知觉初步实验 (如有) |
| G | 人类基线实验细节 |

---

## 不该出现在正文中的内容

- GRPO / SFT 训练前后对比
- 大量神经科学名词堆砌
- "full 4D / full 3D physical world" 的大 claim
- 过多内部表征解码表述
- 还没有跑完的实验结果
