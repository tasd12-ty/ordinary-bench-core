# ORDINARY-BENCH 静态问题评分标准

本文档详细记录 ORDINARY-BENCH 中三种静态问题类型的评分标准、计算公式和具体示例。

评分实现代码：`VLM-test/API-test/scoring.py`

---

## 一、QRR — 四元相对关系评分

### 1.1 题型回顾

当前静态基准中的 QRR 有两个变体，评分逻辑相同：

1. `disjoint`：从场景中选取 4 个对象组成两个不相交对 `(pair1, pair2)`，比较两对之间的 3D 欧氏距离大小。
2. `shared_anchor`：固定一个锚点 `anchor`，比较 `dist(anchor, obj_b)` 与 `dist(anchor, obj_c)`。

**VLM 输入（disjoint）**：
```
Compare the distance between obj_0 and obj_1
vs the distance between obj_2 and obj_3.
Answer: < / ~= / >
```

**VLM 输入（shared_anchor）**：
```
From anchor obj_0, compare the distance to obj_1
vs the distance to obj_2.
Answer: < / ~= / >
```

**VLM 输出**：字符串 `"<"`、`"~="` 或 `">"`

### 1.2 比较代数

Ground Truth 由基于容差 τ（默认 0.10）的比较代数确定：

| 关系 | 判定条件 | 含义 |
|------|---------|------|
| `a <_τ b` | `a < b` 且 `|a - b| > τ × max(a, b)` | a 显著小于 b |
| `a ≈_τ b` | `|a - b| ≤ τ × max(a, b)` | a 和 b 近似相等 |
| `a >_τ b` | `a > b` 且 `|a - b| > τ × max(a, b)` | a 显著大于 b |

其中 `a = dist(pair1)`，`b = dist(pair2)`。

**边界过滤**：当 `|a - b|` 处于 `[0.8τ × max, 1.2τ × max]` 范围内时，该问题不会被生成（因为 GT 不可靠）。

### 1.3 评分指标

QRR 只有一个评分维度：

| 指标 | 计算方式 | 输出字段 |
|------|---------|---------|
| **精确匹配** | `Comparator.from_string(predicted) == Comparator.from_string(gt)` | `correct: bool` |

评分时会对输入做归一化处理（去除空格、支持 `lt`/`eq`/`gt`/`≈`/`=` 等别名）。

### 1.4 聚合指标

| 聚合字段 | 公式 |
|---------|------|
| `qrr_accuracy` | `qrr_correct / qrr_total` |
| `qrr_disjoint_accuracy` | `qrr_disjoint_correct / qrr_disjoint_total` |
| `qrr_shared_anchor_accuracy` | `qrr_shared_anchor_correct / qrr_shared_anchor_total` |

### 1.5 示例

```
GT: dist(obj_0, obj_1) = 3.55,  dist(obj_2, obj_3) = 6.01
    差值 |3.55 - 6.01| = 2.46,  τ × max = 0.10 × 6.01 = 0.601
    2.46 > 0.601  →  GT = "<"

VLM 预测: "<"  →  correct = True  ✓
VLM 预测: ">"  →  correct = False ✗
VLM 预测: "~=" →  correct = False ✗
```

---

## 二、TRR — 三元钟面关系评分

### 2.1 题型回顾

TRR 问题选取 3 个对象，确定 target 相对于 ref1→ref2 基线的钟面方向。

**VLM 输入**：
```
Standing at obj_0, facing obj_1 (12 o'clock),
what clock hour (1-12) is obj_2 at?
```

**VLM 输出**：整数 1-12

### 2.2 钟面方向计算

1. 以 ref1→ref2 向量作为参考方向（12 点钟 = 正前方）
2. 计算 ref1→target 向量与参考方向的夹角
3. 将角度映射到钟面小时数（每 30° 一个刻度）
4. 小时数映射到象限：
   - Q1 = {12, 1, 2}
   - Q2 = {3, 4, 5}
   - Q3 = {6, 7, 8}
   - Q4 = {9, 10, 11}

### 2.3 评分指标

TRR 有三个评分维度，从严格到宽松：

| 指标 | 计算方式 | 含义 | 输出字段 |
|------|---------|------|---------|
| **hour 精确匹配** | `predicted == gt_hour` | 钟面小时数完全一致 | `hour_correct: bool` |
| **quadrant 象限匹配** | `hour_to_quadrant(predicted) == gt_quadrant` | 预测落在 GT 所在象限 | `quadrant_correct: bool` |
| **adjacent 相邻匹配** | `|predicted - gt_hour| ≤ 1`（12↔1 循环） | 预测与 GT 相差不超过 1 小时 | `adjacent_correct: bool` |

**相邻匹配的循环处理**：`diff = |predicted - gt_hour|`，当 `diff ≤ 1` 或 `diff ≥ 11` 时判定为相邻（处理 12↔1 的环绕）。

### 2.4 聚合指标

| 聚合字段 | 公式 |
|---------|------|
| `trr_hour_accuracy` | `trr_hour_correct / trr_total` |
| `trr_quadrant_accuracy` | `trr_quadrant_correct / trr_total` |
| `trr_adjacent_accuracy` | `trr_adjacent_correct / trr_total` |

三个指标满足包含关系：`hour_correct ⊆ adjacent_correct`（精确匹配一定是相邻的），`hour_correct ⊆ quadrant_correct`（精确匹配一定在同一象限），但 `adjacent_correct` 和 `quadrant_correct` 之间无严格包含关系。

### 2.5 示例

```
GT: hour = 3, quadrant = 2

VLM 预测 = 3:
  hour_correct    = True  ✓  (3 == 3)
  quadrant_correct = True  ✓  (Q2 == Q2)
  adjacent_correct = True  ✓  (|3-3| = 0 ≤ 1)

VLM 预测 = 4:
  hour_correct    = False ✗  (4 ≠ 3)
  quadrant_correct = True  ✓  (hour_to_quadrant(4) = Q2 == Q2)
  adjacent_correct = True  ✓  (|4-3| = 1 ≤ 1)

VLM 预测 = 5:
  hour_correct    = False ✗  (5 ≠ 3)
  quadrant_correct = True  ✓  (hour_to_quadrant(5) = Q2 == Q2)
  adjacent_correct = False ✗  (|5-3| = 2 > 1)

VLM 预测 = 7:
  hour_correct    = False ✗  (7 ≠ 3)
  quadrant_correct = False ✗  (hour_to_quadrant(7) = Q3 ≠ Q2)
  adjacent_correct = False ✗  (|7-3| = 4 > 1)

特殊情况 — GT = 12, VLM 预测 = 1:
  hour_correct    = False ✗  (1 ≠ 12)
  quadrant_correct = True  ✓  (Q1 == Q1)
  adjacent_correct = True  ✓  (|1-12| = 11 ≥ 11，循环相邻)
```

---

## 三、FDR — 全距离排序评分

### 3.1 题型回顾

FDR 问题以某个物体为锚点，要求 VLM 将场景中其余所有物体按与锚点的 3D 距离从近到远排列。

**VLM 输入**：
```
Rank all other objects by distance from obj_0, nearest to farthest.
Objects to rank: obj_1, obj_2, obj_3.
Answer: ordered JSON list of object IDs.
```

**VLM 输出**：有序列表，如 `["obj_2", "obj_1", "obj_3"]`

### 3.2 并列组（Tie Groups）

当两个物体与锚点的距离在 τ=0.10 容差内时（使用与 QRR 相同的比较代数），它们构成一个并列组。并列组内的物体在评分时可以任意排列而不被扣分。

**并列判定**：`|dist(A,B) - dist(A,C)| ≤ τ × max(dist(A,B), dist(A,C))`

**示例**：
```
锚点: obj_0
距离: obj_1=3.55, obj_2=3.01, obj_3=3.88

obj_1 vs obj_3: |3.55-3.88| / 3.88 = 0.085 < τ=0.10  →  并列
obj_2 vs obj_1: |3.01-3.55| / 3.55 = 0.152 > τ=0.10  →  不并列

GT 排序: [obj_2, obj_1, obj_3]
并列组: [[obj_2], [obj_1, obj_3]]

评分时 [obj_2, obj_3, obj_1] 等价于 [obj_2, obj_1, obj_3]（obj_1 和 obj_3 可互换）
```

### 3.3 评分指标

FDR 有四个评分维度：

#### 3.3.1 精确匹配 (Exact Match)

**判定方式**：按并列组逐组检查，预测序列中对应位置的物体集合是否与 GT 并列组一致。

**公式**：对每个并列组 `group_i`（长度 `len_i`），检查 `set(predicted[start:start+len_i]) == set(group_i)`。全部组匹配则为 True。

| 输出字段 | 类型 | 含义 |
|---------|------|------|
| `exact_correct` | `bool` | 完整排序是否正确（尊重并列组） |

**示例**：
```
GT 排序: [A, B, C, D]
并列组: [[A], [B, C], [D]]

预测 [A, C, B, D] → exact = True  ✓  (B,C 是并列组，可互换)
预测 [A, B, D, C] → exact = False ✗  (C,D 不是并列组)
预测 [A, B, C, D] → exact = True  ✓
```

#### 3.3.2 Kendall τ 排序相关系数

**判定方式**：计算预测排序与 GT 排序之间的 Kendall τ 相关系数，并列组内的对跳过（不计入一致对或不一致对）。

**公式**：
```
τ = (concordant - discordant) / (concordant + discordant)
```

其中：
- **concordant（一致对）**：预测中 A 排在 B 前面，GT 中 A 也排在 B 前面
- **discordant（不一致对）**：预测中 A 排在 B 前面，GT 中 B 排在 A 前面
- **并列对**：A 和 B 在同一 tie_group 中 → 不参与计算

| 输出字段 | 类型 | 范围 | 含义 |
|---------|------|------|------|
| `kendall_tau` | `float` | [-1, 1] | 1.0 = 完全一致，-1.0 = 完全相反，0.0 = 随机 |

**示例**：
```
GT 排序: [A, B, C]，并列组: [[A], [B], [C]]（无并列）
预测: [A, C, B]

对 (A,B): 预测 A<B ✓, GT A<B ✓ → concordant
对 (A,C): 预测 A<C ✓, GT A<C ✓ → concordant
对 (B,C): 预测 C<B ✗, GT B<C → discordant

τ = (2 - 1) / (2 + 1) = 0.333
```

#### 3.3.3 成对序关系正确率 (Pairwise Accuracy)

**判定方式**：遍历 GT 排序中所有 (i, j) 对（i < j，即 GT 中 i 更近），检查预测中是否也将 i 排在 j 前面。并列组内的对自动计为正确。

**公式**：
```
pairwise_accuracy = correct_pairs / total_pairs

其中：
  total_pairs = C(n_ranked, 2)  （所有物体对数）
  correct_pairs = 并列对数 + 预测中顺序正确的非并列对数
```

| 输出字段 | 类型 | 范围 | 含义 |
|---------|------|------|------|
| `pairwise_accuracy` | `float` | [0, 1] | 成对比较正确率 |

**与 QRR 的关系**：每个成对比较更准确地说等价于一个 `shared_anchor` QRR 问题。`pairwise_accuracy` 衡量的是 FDR 排序中隐含的 anchor-based 距离序信息的正确率。

**示例**：
```
GT 排序: [A, B, C]，并列组: [[A], [B, C]]
预测: [A, C, B]

对 (A,B): A 更近，预测 A 在 B 前 → correct  ✓
对 (A,C): A 更近，预测 A 在 C 前 → correct  ✓
对 (B,C): 并列组内                → correct  ✓（自动通过）

pairwise = 3/3 = 1.0
```

```
GT 排序: [A, B, C]，并列组: [[A], [B], [C]]（无并列）
预测: [B, A, C]

对 (A,B): A 更近，预测 B 在 A 前 → wrong   ✗
对 (A,C): A 更近，预测 A 在 C 前 → correct  ✓（B,A 都在 C 前）
对 (B,C): B 更近，预测 B 在 C 前 → correct  ✓

pairwise = 2/3 = 0.667
```

#### 3.3.4 Top-1 正确率

**判定方式**：预测排序中排第一的物体是否与 GT 中最近的物体一致。

**公式**：
```
top1 = 1.0 if predicted[0] == gt_ranking[0] else 0.0
```

| 输出字段 | 类型 | 范围 | 含义 |
|---------|------|------|------|
| `top1_correct` | `bool` | — | 最近物体是否正确识别 |

**示例**：
```
GT 排序: [obj_2, obj_1, obj_3]

预测 [obj_2, obj_3, obj_1] → top1 = True  ✓  (obj_2 对了)
预测 [obj_1, obj_2, obj_3] → top1 = False ✗  (应为 obj_2)
```

### 3.4 聚合指标

| 聚合字段 | 公式 | 含义 |
|---------|------|------|
| `fdr_exact_accuracy` | `fdr_exact_correct / fdr_total` | 精确匹配准确率 |
| `fdr_kendall_mean` | `Σ(kendall_tau × fdr_total_i) / Σ(fdr_total_i)` | 加权平均 Kendall τ |
| `fdr_pairwise_mean` | `Σ(pairwise_accuracy × fdr_total_i) / Σ(fdr_total_i)` | 加权平均成对正确率 |
| `fdr_top1_mean` | `Σ(top1 × fdr_total_i) / Σ(fdr_total_i)` | 加权平均 Top-1 正确率 |

跨场景聚合时，使用各场景的 `fdr_total` 作为权重进行加权平均。

### 3.5 无效/缺失预测处理

| 情况 | 处理方式 |
|------|---------|
| VLM 未返回该题答案 (`pred = None`) | 计入 `missing`，对应题型的 total 计数但不计入 correct |
| VLM 返回非列表 | 转为空列表 `[]`，所有指标为 0 或 False |
| VLM 返回不完整列表（缺少部分物体） | Kendall τ 取公共物体计算；pairwise 将缺失物体计为错误 |
| VLM 返回包含无效物体 ID | 该 ID 不在 GT 中，不参与计算 |

---

## 四、三种题型对比总结

| 维度 | QRR | TRR | FDR |
|------|-----|-----|-----|
| **答案类型** | 字符串 `<`/`~=`/`>` | 整数 1-12 | 有序 ID 列表 |
| **评分维度数** | 1 | 3 | 4 |
| **最严格指标** | exact match | hour_correct | exact_correct |
| **最宽松指标** | exact match | quadrant_correct | pairwise_accuracy |
| **容差机制** | GT 生成时 τ 过滤边界 | 象限匹配/相邻匹配 | tie_groups 并列组 |
| **信息密度** | 1 个距离序比较/题 | 1 个方向/题 | C(N-1,2) 个隐含比较/题 |
| **题数/场景 (N=4)** | 15 (= 3 disjoint + 12 shared_anchor) | 24 | 4 |
| **题数/场景 (N=10)** | 990 (= 630 disjoint + 360 shared_anchor) | 720 | 10 |

---

## 五、逐题评分输出格式

### QRR 逐题

```json
{
  "qid": "qrr_0001",
  "type": "qrr",
  "variant": "shared_anchor",
  "anchor": "obj_0",
  "predicted": "<",
  "gt": "<",
  "correct": true
}
```

### TRR 逐题

```json
{
  "qid": "trr_0001",
  "type": "trr",
  "predicted": 4,
  "gt_hour": 3,
  "hour_correct": false,
  "quadrant_correct": true,
  "adjacent_correct": true
}
```

### FDR 逐题

```json
{
  "qid": "fdr_0001",
  "type": "fdr",
  "predicted": ["obj_2", "obj_3", "obj_1"],
  "gt_ranking": ["obj_2", "obj_1", "obj_3"],
  "exact_correct": true,
  "kendall_tau": 1.0,
  "pairwise_accuracy": 1.0,
  "top1_correct": true
}
```

---

## 六、场景级聚合输出格式

```json
{
  "qrr_correct": 2,
  "qrr_total": 3,
  "qrr_disjoint_correct": 1,
  "qrr_disjoint_total": 1,
  "qrr_shared_anchor_correct": 1,
  "qrr_shared_anchor_total": 2,
  "trr_hour_correct": 5,
  "trr_quadrant_correct": 12,
  "trr_adjacent_correct": 8,
  "trr_total": 24,
  "fdr_exact_correct": 1,
  "fdr_total": 4,
  "fdr_kendall_mean": 0.6667,
  "fdr_pairwise_mean": 0.8333,
  "fdr_top1_mean": 0.75,
  "missing": 0,
  "per_question": [...]
}
```
