# Ordinary-Bench 论文完整 Outline

**目标投稿**: NeurIPS 2026 Datasets & Benchmarks Track
**正文页数**: 9 页 + references（附录不限）
**更新日期**: 2026-03-15

---

## 标题

> **What Do Vision-Language Models Really See?
> Ordinary-Bench: Evaluating Spatial Belief via Controlled Probing and Scene Reconstruction**

---

## 摘要（四句半结构）

1. **缺口**: 现有 VLM 空间评测依赖单题正确率，无法判断模型是否形成稳定、可重建的场景表征。
2. **方法**: 我们提出 Ordinary-Bench，在受控 3D 渲染环境中构造 700 场景、336K+ 序数空间探测（QRR 距离排序 + TRR 方位判断），覆盖 4–10 物体的复杂度梯度。
3. **核心分析**: 提出 scene belief reconstruction，将 VLM 回答转为空间约束并重建外显场景信念（CSR / NRMS / Kendall τ），并通过正确图像/错误图像/无图像三条件实验量化视觉信息增益（VIG）。
4. **发现**: 实验（6+ 模型 + 人类基线）表明，VLM 在局部题目上可取得非平凡准确率，但场景级重建暴露系统性几何缺陷；复杂场景下模型退化为语言先验驱动的伪空间判断。
5. **意义**: Ordinary-Bench 提供了从"答对多少题"转向"模型究竟看见了什么"的评测框架。

**写作状态**: ❌ 未写

---

## 三大贡献

1. **Ordinary-Bench 基准**: 受控 3D 渲染 benchmark，700 场景 × 336K 序数探测，QRR + TRR 两类任务，复杂度梯度 + 多视角支持。
2. **Scene Belief Reconstruction**: 将 VLM 离散回答 → 空间约束 → 场景级可重建表征，用 CSR/NRMS/Kendall τ/K_geom 超越单题准确率。
3. **Visual Contribution Separation**: 三条件实验（正确图像/错误图像/无图像）+ 一致性分析，量化视觉输入 vs 语言先验的真实贡献。

---

## 正文结构

---

### §1 Introduction（~1.0 页）

**逻辑流:**

| 段落 | 内容 | 关键引用 |
|------|------|---------|
| ¶1 开场 | VLM 能答对空间题 ≠ 理解空间结构；答题分数无法区分 seeing vs guessing | SpatialVLM, BLINK, MMStar, VLind-Bench |
| ¶2 缺口 | 现有 benchmark 缺少场景级一致性分析 + 视觉贡献分离；需要把 VLM 当黑盒观察者来逆向分析其行为 | CLEVR, CLEVRER, psychophysics |
| ¶3 方法预览 | Ordinary-Bench: 受控 3D 渲染 → 序数空间探测 → 约束提取 → scene belief reconstruction → 三条件消融 | — |
| ¶4 贡献列表 | 三条贡献（见上） | — |
| ¶5 边界声明 | "我们关注的不是模型内部神经状态，而是可由行为外显、可由重建检验的 scene belief" | — |

**写作状态**: ✅ 已有高质量初稿 (`neurips_ed_intro_draft_20260312.tex`)

---

### §2 Related Work（~0.8 页）

5 个子方向，每个 2–3 句:

| 子方向 | 核心引用 | 要讲的一句话 |
|--------|---------|------------|
| **VLM 空间基准** | SpatialVLM, VSR, What'sUp, SpatialBench | 侧重分类准确率，缺少场景级结构分析 |
| **合成基准方法论** | CLEVR, CLEVRER | 受控合成数据的科学价值（精确 GT + 可控复杂度） |
| **VLM 可靠性 / 幻觉** | POPE, VLInd-Bench, MMStar | 语言先验污染视觉判断，无图像也能答对 |
| **选择性回答 / 不确定性** | Reliable VQA, Selective VQA, Khan et al. | 允许 abstain + 一致性分析的评测协议 |
| **序数嵌入 / 几何理论** | GNMDS, EDM, Bearing Rigidity, Ma et al. | 从序数比较到坐标重建的数学基础 |

**写作状态**: ❌ 未写

---

### §3 Ordinary-Bench: Data & Protocol（~1.5 页）

#### 3.1 Scene Generation
- Blender 渲染管线：物体随机放置 + 材质/光照/相机固定
- 7 个复杂度 split（n=4,...,10），每 split 100 场景 → 共 700 场景
- 每场景输出：PNG 渲染图 + 3D/2D 坐标 JSON + 4 个多视角图像
- 物体属性：3 shapes × 3 sizes × 2 materials × 8 colors

**写作状态**: ✅ 已有完整初稿

#### 3.2 Task Definition

**QRR (Quaternary Relative Relations)**
- 输入：两对不相交物体 (A,B) vs (C,D)
- 输出：d(A,B) < / ≈ / > d(C,D)（3D 欧氏距离）
- 容差代数 τ=0.10，过滤边界模糊样本
- 组合量：C(n,2)×C(n-2,2)/2，n=10 时 630 题/场景

**TRR (Ternary Relative Relations)**
- 输入：target + ref1 + ref2
- 输出：target 在 ref1→ref2 参照系下的钟面方向（1–12 点）
- 评估粒度：hour-exact / quadrant / adjacent（±1 hour）
- 组合量：P(n,3) = n(n-1)(n-2)，n=10 时 720 题/场景

**Probe Count Table**: 总计 ~336K 探测点

**写作状态**: ✅ 已有完整初稿 + Table

#### 3.3 Evaluation Protocol

**Batch JSON prompting**: 每 20 题一批 + ReAct 纠正循环

**三条件实验**:
- Condition A: 正确图像（标准 VQA）
- Condition B: 错误图像（视觉干扰 + 语言先验）
- Condition C: 无图像（纯语言先验基线）
- VIG = NRMS_C − NRMS_A

**Scoring**: QRR 精确匹配 / TRR hour-exact + quadrant + adjacent

**写作状态**: ✅ 已有完整初稿

#### 3.4 Dataset Release & Documentation（D&B 硬性要求）
- 数据格式说明 + annotation schema
- Croissant metadata / dataset card
- 许可证（推荐 CC BY-SA 4.0）
- 下载地址 + 复现脚本
- 最小可运行子集

**写作状态**: ❌ **未写，且尚无 Croissant / LICENSE / dataset card 实体文件**

---

### §4 Scene Belief Reconstruction（~1.5 页）

**定位**: 这是 benchmark 的分析工具，不是独立算法贡献。

#### 4.1 From Responses to Constraints
- QRR 回答 → 距离偏序 DAG（P_dist），环检测 → 逻辑不一致
- TRR 回答 → 方位角弧段系统（P_ang），弧段交集为空 → 方向矛盾
- 两种模式：ground-truth mode（仅正确回答）vs belief mode（全部回答）

**写作状态**: ✅ 已有完整初稿

#### 4.2 Optimization
- Gauge fixing: 3-anchor 方案消除 5 DOF（平移+旋转+缩放+反射）
- Loss = L_QRR（log-domain softplus）+ L_TRR（sector-tolerance）+ L_sep（分离正则）
- L-BFGS-B + K=10 多重启

**写作状态**: ✅ 已有完整初稿 + 公式

#### 4.3 Scene-Level Metrics
| 指标 | 定义 | 含义 |
|------|------|------|
| **CSR** | 约束满足率 | 回答的全局一致性 |
| **NRMS** | Procrustes 对齐后归一化 RMS | 绝对重建精度 |
| **Kendall τ** | 距离序相关 | 序保持程度 |
| **K_geom** | 几何模态数 | 解的唯一性 |
| **Spread** | 解空间大小 | 约束确定性 |

**写作状态**: ✅ 已有完整初稿

#### 4.4 Pipeline Validation
- 在 GT 约束下重建：CSR≈1.0, NRMS≈0.01, K_geom=1
- 证明退化来自模型回答而非重建算法

**写作状态**: ✅ 已有完整初稿

---

### §5 Separating Vision from Language（~0.8 页）

**单独成节，因为这是最能打 reviewer 的贡献。**

#### 5.1 三条件实验框架
- 对同一场景问同一套问题，分别在 A/B/C 条件下跑
- 每个条件做完整 reconstruction → S_A, S_B, S_C
- 定义 VIG = NRMS(C) - NRMS(A)
  - VIG > 0 → 图像确实贡献了空间信息
  - VIG ≈ 0 → 模型主要依赖语言先验
- 统计检验：置换检验 + paired Wilcoxon

#### 5.2 一致性分析
- **传递一致性**: d(A,B) < d(C,D) 且 d(C,D) < d(E,F) ⟹ d(A,B) < d(E,F)?
  - 传递违反率 T_viol → 约束集结构质量的客观测量
- **互逆一致性**: "A 在 B→C 的 3 点钟" ⟹ "A 在 C→B 的 9 点钟"?
  - 互逆违反率 R_viol → 方向感知的一致性
- 与随机基线对比：语言先验不太可能满足传递性

#### 5.3 随机基线与 Belief Faithfulness
- QRR random baseline: 33.3%（三选一）
- TRR random baseline: 8.3%（十二选一）
- BFG (Belief Faithfulness Gap): 模型重建质量 vs random 重建质量

**写作状态**: ❌ **未写。代码已有（consistency.py, information_gain.py, baselines.py），但无实验数据。**

---

### §6 Experiments（~2.5 页 — 最重的部分）

#### 6.1 Experimental Setup（~0.5 页）

**模型列表**:

| 类型 | 模型 | 状态 |
|------|------|------|
| 闭源 | GPT-4o | ❌ 未跑 |
| 闭源 | Gemini 2.0 Flash | ❌ 未跑 |
| 闭源 | Claude Sonnet | ❌ 未跑 |
| 开源 | Qwen3-VL-235B | ⚠️ 58/700 场景 |
| 开源 | Qwen3-VL-30B | ⚠️ 61/700 场景 |
| 开源 | LLaVA-OneVision / InternVL2 | ❌ 未跑 |

**人类基线**: 50–100 场景 × 3+ 标注者

**评测配置**: temperature=0, batch mode, 单视角（主）+ 多视角（消融）

**写作状态**: ❌ **实验严重不足。仅 2 个 Qwen 模型各 ~60 场景。**

---

#### 6.2 主准确率结果（Tab 1, Fig 4）

**Table 1**: 模型 × QRR Acc / TRR Hour / TRR Quad / TRR Adj / Missing Rate

**Figure 4**: 准确率 vs 物体数（n=4→10）衰减曲线

要点:
- QRR accuracy vs random baseline (33.3%) 的对比
- TRR hour accuracy vs random baseline (8.3%) 的对比 — **当前数据 ~10%，需要严肃讨论 discriminative power**
- 人类基线作为天花板
- 跨模型家族的表现差异

**数据状态**: ⚠️ 有初步 accuracy_table.md，但数据量不足以写论文

---

#### 6.3 场景信念重建分析（Tab 2, Fig 2, Fig 5）

**Table 2**: 模型 × CSR_QRR / CSR_TRR / NRMS / Kendall τ / K_geom / Spread

**Figure 2**: GT vs 重建配置对比图（3–4 个典型场景，Procrustes 对齐后）
- 好模型 vs 差模型
- 简单场景 (n=4) vs 复杂场景 (n=10)

**Figure 5**: NRMS / CSR violin plot（跨场景分布）

要点:
- H1 验证：单题准确率尚可，但 reconstruction 暴露全局不一致
- 重建质量随复杂度衰减的速率
- K_geom > 1 的场景比例（约束不足以确定唯一解）
- belief mode vs GT mode 的差距

**数据状态**: ⚠️ 有初步 recon_summary，但场景覆盖不足

---

#### 6.4 视觉增益分析（Tab 3, Fig 3）

**Table 3**: 模型 × NRMS_A / NRMS_B / NRMS_C / VIG / VIG_B / p-value

**Figure 3**: 同一场景在三条件下的重建对比（最直观的图）
- Condition A: 接近 GT
- Condition B: 扭曲但有某种结构
- Condition C: 更严重的扭曲或随机

要点:
- H3 验证：VIG > 0 且显著 → 图像确实贡献了空间信息
- 如果 VIG ≈ 0 → 模型主要依赖语言先验（更强的 finding）
- 跨复杂度的 VIG 变化：简单场景视觉贡献大 vs 复杂场景退化为先验

**数据状态**: ❌ **完全空白，run_three_condition.py 尚未执行**

---

#### 6.5 一致性分析（Tab 4）

**Table 4**: 模型 × T_viol (传递违反率) / R_viol (互逆违反率) / 与 random 对比

要点:
- 低传递违反率 → 约束集内部一致 → 模型可能有统一的空间模型
- 高传递违反率 → 回答来自互不相关的局部猜测
- 与条件 C（纯语言先验）的对比

**数据状态**: ⚠️ 有 consistency.json，但数据量不足

---

#### 6.6 错误归因分析（Fig 6，可选）

3 层错误分类:
- **感知层失败**: 根本没读到局部几何
- **结构层失败**: 局部答案互相矛盾，无法形成统一场景
- **退化模式**: 典型扭曲可视化（旋转偏差、尺度坍缩、镜像翻转）

**写作状态**: ❌ 未写，需手动选取案例

---

### §7 Discussion & Limitations（~0.7 页）

讨论围绕三句话:
1. **VLM 看见了什么**: 局部视觉线索 > 全局几何结构；距离感 > 方向感
2. **局部正确 vs 全局一致性的系统性差距**: 这是 benchmark 最核心的 finding
3. **合成 vs 真实的迁移问题**: 受控环境的优势（精确 GT）与局限

Limitations:
- 合成场景 → 真实世界的 generalization 尚未验证
- 2D/2.5D 重建，非完整 3D
- API 黑盒限制，无法访问内部表征
- 当前三条件设计假设语言先验稳定
- tau 参数的选择对 QRR boundary 的影响

未来方向:
- 动态时间知觉扩展（data-gen-dynamic/ 已有原型）
- 真实世界场景验证
- 开源模型的内部表征分析
- Selective QA / abstention 机制

**写作状态**: ❌ 未写

---

### §8 Release, Reproducibility & Ethics（~0.5 页，D&B 必须）

- Data hosting: HuggingFace / Zenodo
- Code availability: GitHub repo (生成 + 评测 + 分析全链路)
- 许可证: 代码 MIT，数据 CC BY-SA 4.0
- Model access notes: 闭源模型需 API key
- Compute budget: 场景渲染 ~X GPU-hours, API 费用 ~$Y
- Limitations / misuse: 仅评测几何物体场景，不含人/敏感内容
- Annotation pipeline: 全自动（无人工标注，GT 来自 3D 渲染）
- Croissant metadata + dataset card

**写作状态**: ❌ **未写，且实体文件（LICENSE, Croissant, dataset card）均不存在**

---

### §9 Conclusion（~0.3 页）

一句话:
> 即使在受控的 3D 渲染环境和对人类而言高度直观的空间任务中，当前 VLM 也只能给出局部正确、全局不稳的回答；只有将其行为转化为可重建的空间信念，我们才能真正回答"模型究竟看见了什么"。

**写作状态**: ❌ 未写

---

## 附录结构

| 附录 | 内容 | 状态 |
|------|------|------|
| A | 数据集详细文档（Croissant, datasheet for datasets） | ❌ |
| B | 场景生成细节（Blender 参数、物体列表、材质、渲染样例） | ⚠️ 信息散落在代码中 |
| C | 重建算法细节（loss 推导、超参数列表、convergence 分析） | ⚠️ 代码完整但未整理成文 |
| D | 全模型逐 split 结果表 | ❌ 数据不足 |
| E | 更多重建可视化（每模型 5-10 场景） | ❌ |
| F | 动态时间知觉初步实验（如有） | ❌ data-gen-dynamic 已有代码但未闭环 |
| G | 人类基线实验细节 | ❌ |
| H | Prompt 模板完整列表 | ⚠️ prompts.py 已有，需格式化 |

---

## Figure & Table 清单

### Figures（正文 5–6 张）

| # | 内容 | 类型 | 状态 |
|---|------|------|------|
| Fig 1 | 方法总览图：图像→探测→约束→重建→信念分析 | 手绘/TikZ | ❌ |
| Fig 2 | GT vs 重建配置对比（3-4 场景，好/差模型 × 简单/复杂） | Matplotlib/Blender | ⚠️ 脚本有 |
| Fig 3 | 三条件 A/B/C 同一场景重建对比 | Matplotlib | ❌ 无数据 |
| Fig 4 | 准确率 vs 物体数衰减曲线（所有模型 + human + random） | Matplotlib | ❌ 数据不足 |
| Fig 5 | NRMS/CSR violin plot 或 box plot | Matplotlib | ❌ 数据不足 |
| Fig 6 | 错误案例分析（感知层/结构层失败可视化） | 手动选取 | ❌ |

### Tables（正文 4–5 张）

| # | 内容 | 状态 |
|---|------|------|
| Tab 1 | 主准确率表：模型 × split × QRR/TRR metrics | ⚠️ 有框架，数据不足 |
| Tab 2 | 重建指标表：模型 × CSR/NRMS/τ/K_geom/Spread | ⚠️ 有框架，数据不足 |
| Tab 3 | 三条件 VIG 表：模型 × NRMS_A/B/C / VIG / p-value | ❌ 无数据 |
| Tab 4 | 一致性表：模型 × T_viol / R_viol / vs random | ⚠️ 有初步数据 |
| Tab 5 | 探测量统计表（per-split QRR/TRR 计数） | ✅ 已有 |
| Tab 6 | 人类 vs VLM 对比（可选，放正文或附录） | ❌ 无数据 |

---

## 全局完成度评估

### ✅ 已完成（可直接写入论文）
- §1 Introduction 初稿
- §3.1–3.3 Benchmark 数据与协议（LaTeX 完整）
- §4 Scene Belief Reconstruction（LaTeX 完整 + 公式 + validation）
- Tab 5 探测量统计
- 全部分析代码管线（aggregate, reconstruct, consistency, VIG, baselines, visualize）

### ⚠️ 部分完成（有基础，需补全）
- 2 个 Qwen 模型各 ~60 场景结果（需扩展至 700 场景 × 6+ 模型）
- accuracy_table + consistency + recon_summary（数据量不够写论文）
- 可视化脚本（已有，但需要足够的数据来生成定稿 figure）

### ❌ 完全缺失（必须从零做起）
1. **4+ 模型的全量评测数据**（GPT-4o, Gemini, Claude, LLaVA/InternVL）
2. **三条件实验数据**（至少 1 个模型完整 A/B/C）
3. **人类基线**（界面 + 招募 + 数据收集 + 分析）
4. **统计显著性检验代码**（paired tests, bootstrap CI, 置换检验）
5. **Dataset release 材料**（LICENSE, Croissant, dataset card, HF repo）
6. **§2 Related Work** 全文
7. **§5 Separating Vision from Language** 全文
8. **§6 Experiments** 全文（包含所有 Table/Figure 的数据支撑）
9. **§7 Discussion, §8 Release, §9 Conclusion** 全文
10. **Fig 1 方法总览图**
11. **完整 main.tex 框架**（NeurIPS 模板 + 所有 section 集成）

---

## 推荐执行顺序

```
Phase 1: 数据收集（D-28 ~ D-14）
├── [1] 全量 700 场景 × 现有 2 模型补齐
├── [2] GPT-4o + Gemini + Claude 全量评测
├── [3] 三条件实验（至少 GPT-4o 完整 A/B/C）
├── [4] 人类基线设计 + 招募 + 数据收集
└── [5] 多视角消融（最佳模型）

Phase 2: 分析 + 写作（D-14 ~ D-7）
├── [6] python analysis/run_analysis.py（全量数据）
├── [7] 统计显著性检验
├── [8] 所有 Table/Figure 数据冻结
├── [9] §2, §5, §6, §7, §8, §9 写作
├── [10] Fig 1 方法总览图
└── [11] main.tex 集成

Phase 3: 收尾（D-7 ~ D-1）
├── [12] LICENSE + Croissant + dataset card
├── [13] 附录
├── [14] NeurIPS D&B checklist
├── [15] 匿名化 + 内部审稿
└── [16] 最终提交
```

---

## 风险提示

1. **TRR hour accuracy ~10% vs random 8.3%**: 信号微弱，需要严肃讨论——可通过 quadrant 粒度（~29% vs 25% random）和 reconstruction 来 salvage，或调整 TRR 为 quadrant-level task
2. **三条件实验可能发现 VIG ≈ 0**: 如果视觉信息增益不显著，这反而是更强的 finding（"VLM 根本没在用视觉"），但叙事需要调整
3. **人类基线时间紧张**: 如果招募困难，可考虑作者自身 + 实验室同事的小规模 pilot（至少 30 场景 × 3 人），在论文中声明 scale 限制
