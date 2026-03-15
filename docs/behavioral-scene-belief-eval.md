# 基于行为反演的 VLM 场景信念评估范式

## 1. 文档目的

这份文档定义一套完整、可实现、可复现、可泛化的评估范式，用来回答下面这个问题：

> 给定一张图，如何根据 VLM 的回答行为，反推出它外显出的场景信念，并用严格、可解释的方式比较不同模型？

这里的“场景信念”不是模型内部某层隐藏向量的机械解释，而是：

- 模型愿意明确断言的空间关系
- 模型承认不确定或模糊的空间关系
- 模型拒绝回答的空间关系
- 由这些行为观测反推出的场景可行集

这套范式的目标不是单纯比较答题准确率，而是评估：

- 模型到底“看到了什么”
- 模型对哪些关系是确定的，哪些关系是不确定的
- 模型外显出的场景是紧的、松的，还是多模态的

---

## 2. 核心定义

### 2.1 研究对象

研究对象是 **模型行为**，不是模型参数。

这里的行为包括：

- 对空间关系问题给出的确定回答
- 对空间关系问题给出的模糊/多候选回答
- 对空间关系问题的拒答
- 模型直接输出的关系图或场景描述

### 2.2 潜在变量

我们假设模型对每张图都隐含着一个外显场景信念，记为：

`B_m(I)`

其中：

- `m` 是模型
- `I` 是输入图像

`B_m(I)` 不是单个坐标点，而是一个关于对象和空间关系的潜在可行结构。

### 2.3 可观测量

我们真正观测到的是：

- 原始文本回答
- 解析后的约束观测
- 不确定性信号
- 多次采样的一致性

记为：

`O = observe(B_m(I), protocol, prompt, seed)`

### 2.4 评估目标

评估的核心不是“答对了多少道题”，而是四个问题：

1. 模型给出的约束观测是否可靠
2. 这些观测是否足以反推出稳定场景
3. 反推出的场景是否接近 GT
4. 模型是在“真的看见了”，还是在“被迫猜”

---

## 3. 研究假设

本评估范式固定检验以下四个假设。

### H1：强制单选会污染 VLM 的场景行为

在 `forced_qa` 协议下：

- 错误约束显著增多
- 冲突率显著增高
- 重建结果更接近随机或伪可行

### H2：允许拒答和模糊回答后，约束质量会提升

在 `selective_qa` 协议下：

- 单题覆盖率可能下降
- 但有效约束精度应上升
- 重建的 `CSR_all`、`K_geom`、`spread` 应优于 `forced_qa`

### H3：自主提取更接近模型的外显场景信念

在 `direct_extraction` 协议下：

- 约束覆盖方式不同于逐题问答
- 可能获得更高层、更稀疏但更连贯的关系图
- 若解析可靠，应在“场景层指标”上优于强制单题问答

### H4：重建评估比单题准确率更能区分模型

不同模型可能有相近的单题准确率，但：

- 约束冲突率
- 场景可行率
- `K_geom`
- `spread`
- `CSR_all`

会显著不同。

---

## 4. 统一实验协议

所有模型必须在完全相同的协议下比较。  
协议固定为三种，不允许随模型改评估规则。

### 4.1 `forced_qa`

定义：

- 每道题必须给一个单一答案
- 不允许 `unknown`
- 不允许多候选答案

作用：

- 作为 baseline
- 用于验证“强制作答是否确实导致近随机退化”

### 4.2 `selective_qa`

定义：

- 每道题允许返回：
  - 单一答案
  - 多候选答案集合
  - `unknown`
  - `ambiguous`

作用：

- 这是主协议
- 用于提取模型的“外显场景信念”而不是“被迫猜测行为”

### 4.3 `direct_extraction`

定义：

- 不按题逐条问
- 直接让模型输出对象关系集合或场景关系图

作用：

- 评估模型是否能自主外显其内部场景结构
- 对比逐题问答与自主抽取两条路线

这三种协议必须全部保留。  
任何结论都必须说明是在哪种协议下得到的。

---

## 5. 统一观测数据结构

无论模型输出什么格式，最终都必须归一化为统一 schema。

### 5.1 原始响应

```python
RawResponse(
    run_id: str,
    scene_id: str,
    protocol: str,
    model_name: str,
    prompt_version: str,
    sample_id: int,
    response_text: str,
    raw_json: dict | None,
    latency_ms: int | None,
)
```

### 5.2 观测对象

```python
Observation(
    run_id: str,
    scene_id: str,
    protocol: str,
    relation_type: str,          # qrr | trr | direct_relation
    support: dict,               # pair/triple/object relation scope
    answer_set: list[str],       # 允许单答案或多答案
    abstained: bool,
    ambiguous: bool,
    confidence: float | None,
    consistency: float | None,
    raw_response_ref: str,
    parser_version: str,
)
```

### 5.3 约束对象

```python
ConstraintObservation(
    run_id: str,
    scene_id: str,
    constraint_type: str,        # qrr | trr
    payload: dict,               # 规范化后的约束
    state: str,                  # single | set_valued | abstain | conflict
    confidence: float | None,
    consistency: float | None,
    source_protocol: str,
)
```

关键规则固定如下：

- `answer_set` 允许多个值
- `abstained` 是合法输出，不是异常
- `ambiguous` 是合法输出，不是错误
- 没有 logprob 也可以，只要有 `consistency`
- 所有对象都必须有 `run_id` 和 `scene_id`

---

## 6. 观测状态分类

所有 observation 必须被归类为以下五种状态之一：

1. `correct_single`
2. `correct_set`
3. `wrong_single`
4. `abstain`
5. `ambiguous`

原因很简单：

- `wrong_single` 会制造错误约束
- `abstain` 只会造成缺失，不会直接误导
- `ambiguous` 会扩大可行集，但比硬猜更真实
- `correct_set` 表示模型知道范围，但不知道精确值
- `correct_single` 才是最强的信息

如果代码里不显式区分这五种状态，后续分析会失真。

---

## 7. 从观测到约束的映射规则

映射规则必须固定，不能按模型临时修改。

### 7.1 QRR 映射

`forced_qa`

- `<` / `>` / `~=` 直接映射为单值约束

`selective_qa`

- 若回答是单值：映射为单值 QRR
- 若回答是集合：映射为集合式 QRR
- 若 `unknown`：不生成约束
- 若 `ambiguous`：生成低强度集合约束

### 7.2 TRR 映射

`forced_qa`

- 单个 hour 直接映射

`selective_qa`

- 单个 hour：生成 hour 扇区
- 多个 hour：生成多个扇区并
- quadrant：生成 90 度扇区
- `unknown`：不生成约束

### 7.3 `direct_extraction` 映射

要求模型输出半结构化关系：

```json
{
  "objects": [...],
  "relations": [
    {"type": "left_of", "subject": "...", "object": "...", "confidence": 0.8},
    {"type": "nearer_than", "pair1": ["a","b"], "pair2": ["c","d"], "confidence": 0.7}
  ]
}
```

然后统一映射回 QRR / TRR / 辅助方位约束。

第一版不允许自由文本直接进入 solver。  
自由文本必须先被解析为统一结构。

---

## 8. 下游重建器的责任边界

重建器的责任只有一个：

> 接收统一约束观测，输出场景信念的几何近似表示

重建器不负责：

- 解释原始语言输出
- 评估回答层准确率
- 改写 prompt

重建器输入固定为：

- 单值约束
- 集合约束
- 缺失信息

重建器输出固定为：

```python
SceneBeliefResult(
    feasible: bool,
    status: str,                     # infeasible | underconstrained | single_mode | multimodal
    representative_solutions: list,
    K_geom: int,
    spread: float,
    per_object_uncertainty: dict,
    diagnostics: dict,
)
```

这里的 `SceneBeliefResult` 是“外显场景信念”的几何近似，而不是 GT 恢复结果。

---

## 9. 四层指标体系

所有实验结果必须按四层指标报告，不能只报单题准确率。

## 9.1 回答层指标

用于评估回答行为本身。

- `answer_accuracy`
- `abstain_rate`
- `ambiguous_rate`
- `set_answer_rate`
- `risk_coverage_curve`
- `consistency_score`
- `calibrated_selective_risk`

解释：

- 如果模型答得少但答得准，`coverage` 低但 `risk` 低
- 如果模型全都答但大多瞎猜，`coverage` 高但 `risk` 高

这层是 selective VQA 文献的核心指标层。

## 9.2 约束层指标

用于评估解析后的关系观测质量。

- `constraint_precision_qrr`
- `constraint_precision_trr`
- `constraint_recall_qrr`
- `constraint_recall_trr`
- `cycle_rate_qrr`
- `sector_conflict_rate_trr`
- `constraint_redundancy_ratio`
- `hypergraph_connected`
- `hypergraph_num_components`
- `object_coverage_histogram`

这一层是区分“答题问题”与“重建问题”的关键。

## 9.3 重建层指标

用于评估反演出的场景信念结构。

- `feasible_rate`
- `CSR_selected_qrr`
- `CSR_selected_trr`
- `K_geom`
- `spread`
- `per_object_uncertainty`
- `multimodal_rate`
- `underconstrained_rate`

注意：

- `CSR_selected` 只表示是否满足“模型自己给出的约束”
- 它不代表是否接近 GT

## 9.4 GT 对齐层指标

只有在有 GT 时才计算。

- `CSR_all_qrr`
- `CSR_all_trr`
- `NRMS`
- `Kendall_tau`
- `scene_graph_recall`
- `belief_faithfulness_gap`

其中：

`belief_faithfulness_gap = CSR_selected - CSR_all`

解释：

- 若这个 gap 很大，说明模型给出的约束自洽，但和真实场景差距大
- 这正是“假唯一、假稳定”的危险信号

---

## 10. 误差来源分解

这套范式必须能够区分三类失败来源。

### 10.1 信息不足

表现：

- `abstain_rate` 高
- 约束覆盖稀疏
- 超图不连通
- `K_geom = 1` 但 `spread` 大

解释：

- 模型没有给足够多的有效关系

### 10.2 信息错误

表现：

- 约束 precision 低
- QRR 环率或 TRR 冲突率高
- `CSR_selected` 低
- 或 `CSR_selected` 高但 `CSR_all` 低

解释：

- 模型给了错误关系，或者系统性偏差关系

### 10.3 求解器问题

表现：

- 约束层质量还可以
- 但重建层极不稳定
- restart 间差异异常大
- 同一输入下 solver 波动大

解释：

- 不是模型问题，而是反演算法没把观测用好

没有这三类分解，就不能把实验说清楚。

---

## 11. 实验设计

所有模型都必须跑同一套实验矩阵。

### 11.1 轴一：模型

- GPT 系列
- Qwen 系列
- Gemini 系列
- 开源 VLM

### 11.2 轴二：协议

- `forced_qa`
- `selective_qa`
- `direct_extraction`

### 11.3 轴三：数据 regime

- `oracle_clean`
  只用 GT 正确约束，测试重建上限

- `raw_vlm`
  用模型真实输出

- `controlled_corruption`
  从 GT 约束中人为加入缺失、错误、冗余

这三组 regime 缺一不可：

- 没有 `oracle_clean`，就不知道 solver 上限
- 没有 `raw_vlm`，就不知道真实行为
- 没有 `controlled_corruption`，就不知道失败机制

---

## 12. 统计检验与显著性报告

所有主要结论必须带统计区间，不能只给均值。

固定要求如下：

- 主指标报告 `mean ± 95% bootstrap CI`
- 对同一 scene 上两个协议或两个模型的比较，使用 paired bootstrap
- 样本单位固定为 `scene`
- 不以单题为独立样本做显著性检验

主要比较项：

- `forced_qa` vs `selective_qa`
- `selective_qa` vs `direct_extraction`
- 不同模型在同协议下的比较

主要结论必须至少报告：

- 效应方向
- 效应大小
- 置信区间

---

## 13. 复现要求

每个实验 run 都必须写出完整 manifest。

### 13.1 固定字段

```json
{
  "run_id": "...",
  "dataset_version": "...",
  "scene_split": "...",
  "question_generator_version": "...",
  "prompt_version": "...",
  "parser_version": "...",
  "constraint_builder_version": "...",
  "reconstruct_version": "...",
  "metric_version": "...",
  "protocol": "...",
  "model_name": "...",
  "model_revision": "...",
  "api_provider": "...",
  "temperature": 0.0,
  "top_p": 1.0,
  "n_samples": 5,
  "seed": 1234,
  "timestamp": "...",
  "git_commit": "..."
}
```

### 13.2 强制缓存的中间产物

必须落盘保存：

- prompt
- raw response
- parser output
- observations
- normalized constraints
- reconstruction result
- metrics

任何一层都不能只保留最终汇总结果。

---

## 14. 代码目录规范

推荐目录固定如下：

```text
eval_vlm_scene/
├── adapters/
├── protocols/
├── parsers/
├── observations/
├── constraints/
├── reconstruct/
├── metrics/
├── reports/
├── configs/
└── runs/
```

职责固定如下：

- `adapters/`
  只负责调用模型 API

- `protocols/`
  定义三种交互协议

- `parsers/`
  把原始输出解析成统一 observation

- `observations/`
  定义 schema 和序列化逻辑

- `constraints/`
  把 observation 变成统一约束

- `reconstruct/`
  从约束反演外显场景信念

- `metrics/`
  计算四层指标

- `reports/`
  输出表格、图和 bootstrap 结果

- `runs/`
  保存完整 run 产物

---

## 15. 跨模型泛化要求

这套范式必须能泛化到不同模型，因此模型适配层必须极薄。

统一接口固定为：

```python
class VLMAdapter:
    def generate(self, image, prompt, config) -> RawResponse:
        ...
```

任何模型特有信息都只能附加，不能改变主流程。

例如：

- 有 logprob：可写入 `confidence`
- 没有 logprob：可用多次采样一致性补

但无论如何，后面统一都走：

- parser
- observation builder
- constraint builder
- reconstruct
- evaluate

这就是可泛化的关键。

---

## 16. 风险与边界

这套范式必须明确承认自己的边界。

### 16.1 它能回答什么

- 模型外显出了哪些空间关系
- 模型在哪些地方不确定
- 模型外显出的场景是紧的、松的还是多模态的
- 不同协议下模型行为有什么结构性差异

### 16.2 它不能直接回答什么

- 模型内部某层神经元机制
- hidden state 的精确几何
- 模型“主观视觉体验”

所以这是一套：

- **behavioral interpretability**

而不是：

- **mechanistic interpretability**

### 16.3 三个主要风险

- 拒答可能既反映视觉不确定，也反映 prompt 保守
- 自主提取会混入语言先验和幻觉
- 同一行为可能对应多个潜在场景，存在不可辨识性

这些风险必须在报告里单列。

---

## 17. 阶段性交付

这套范式建议分三步实现。

### Phase 1：最小可运行版

包含：

- `forced_qa`
- `selective_qa`
- 统一 observation schema
- 统一约束构造
- 基础重建器
- 四层指标中的最核心子集

### Phase 2：完整科学评估版

新增：

- `direct_extraction`
- controlled corruption
- bootstrap 显著性分析
- 更完整的报告模块

### Phase 3：增强分析版

新增：

- 主动提问
- 鲁棒重建
- 集合约束更精细建模
- 更细致的 uncertainty calibration

---

## 18. 最终判定标准

如果一套实验设计满足以下条件，就可以认为它是“科学的、可解释的、可复现的”：

1. 使用统一协议和统一 schema
2. 区分回答层、约束层、重建层、GT 层
3. 能分解信息不足、信息错误、求解器失败
4. 所有中间产物可回放
5. 主要结论带置信区间
6. 更换模型时不改主评估流程

如果做不到这 6 点，就不能说这是一套可靠的 VLM 场景信念评估范式。

---

## 19. 与当前仓库的关系

这套文档与当前仓库的关系如下：

- [`brainstorm.md`](brainstorm.md)
  负责重建问题本身的理论与工程讨论

- [`method-proposal.md`](method-proposal.md)
  负责第一版 2D 重建器的实现提案

- **本文**
  负责定义完整的科学评估范式，回答“如何用代码做可解释、可复现、可泛化的行为反演评估”

三份文档职责不同，不应混写。
