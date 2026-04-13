# Variable-Size Subset Ablation Experiment

## 实验目的

测试 VLM 空间推理能力是否随场景中**物体数量**变化而变化。

从一个 N=20 的父场景出发，消融 1-11 个物体，生成大小为 9-19 的子场景（每种大小 20 个，共 220 个子场景），在每个子场景上评测 VLM 的 QRR 空间推理能力。

## 组合数规模

| 消融数 k | 子场景大小 m | C(20,m) | 采样数 |
|---------|------------|---------|-------|
| 1       | 19         | 20      | 20 (全取) |
| 2       | 18         | 190     | 20 |
| 3       | 17         | 1,140   | 20 |
| 4       | 16         | 4,845   | 20 |
| 5       | 15         | 15,504  | 20 |
| 6       | 14         | 38,760  | 20 |
| 7       | 13         | 77,520  | 20 |
| 8       | 12         | 125,970 | 20 |
| 9       | 11         | 167,960 | 20 |
| 10      | 10         | 184,756 | 20 |
| 11      | 9          | 167,960 | 20 |
| **总计** |           | **784,625** | **220** |

## QRR 问题数量

每种物体数量下可答的 QRR 题数（Disjoint = C(m,4)×3, Shared = m×C(m-1,2)）：

| 物体数 m | Disjoint | Shared Anchor | 合计 |
|---------|----------|---------------|------|
| 9       | 378      | 252           | 630 |
| 10      | 630      | 360           | 990 |
| 11      | 990      | 495           | 1,485 |
| 12      | 1,485    | 660           | 2,145 |
| 13      | 2,145    | 858           | 3,003 |
| 14      | 3,003    | 1,092         | 4,095 |
| 15      | 4,095    | 1,365         | 5,460 |
| 16      | 5,460    | 1,680         | 7,140 |
| 17      | 7,140    | 2,040         | 9,180 |
| 18      | 9,180    | 2,448         | 11,628 |
| 19      | 11,628   | 2,907         | 14,535 |
| 20      | 14,535   | 3,420         | 17,955 |

每种大小 20 个场景，**总 QA 对约 120 万**。

## 管线概览

```
Step 0: 生成 N=20 场景           → data-gen/
Step 1: 枚举各尺寸子场景         → enumerate_variable_subsets.py     (本实验)
Step 2: 生成 Master 题库          → subset_ablation/generate_master_questions.py (复用)
Step 3: 分配题目 + 可答标记       → subset_ablation/assign_subset_questions.py   (复用)
Step 4: 渲染子场景图像            → subset_ablation/render_subsets.py             (复用)
Step 5: VLM 评测                 → subset_ablation/run_subset_eval.py            (复用)
Step 6: 分析                     → analyze_variable_subsets.py      (本实验)
Step 7: 可视化                   → visualize_variable_subsets.py    (本实验)
```

## 完整运行步骤

### Step 0: 生成 N=20 场景

```bash
cd data-gen
python generate.py --config config_n20.toml
```

输出: `data-gen/output/scenes/n20_000000.json` + 对应图像。

> **注意**: 20 个物体较拥挤，config_n20.toml 已调大 camera_distance (14.0) 并缩小
> min_dist (0.20)。如果放置失败，可进一步调整这些参数。

### Step 1: 枚举各尺寸子场景

```bash
cd experiments/variable_subset_ablation

python enumerate_variable_subsets.py \
    --scene ../../data-gen/output/scenes/n20_000000.json \
    --min-size 9 --max-size 19 \
    --max-subsets 20 --seed 42 \
    --output-dir output
```

输出:
- `output/scenes/*.json` — 220 个子场景 JSON
- `output/manifest.json` — 映射关系（含 `subset_size` 和 `subsets_by_size` 字段）

### Step 2: 生成 Master 题库

```bash
python ../subset_ablation/generate_master_questions.py \
    --scenes-dir ../../data-gen/output/scenes \
    --output-dir output \
    --splits n20
```

输出: `output/master_questions/n20_000000.json` (17,955 题)

### Step 3: 分配题目

```bash
python ../subset_ablation/assign_subset_questions.py \
    --manifest output/manifest.json \
    --master-dir output/master_questions \
    --output-dir output
```

输出: `output/questions/qrr/{subset_id}.json` — 每个子场景的题目，含 answerable 标签

### Step 4: 渲染子场景图像

```bash
python ../subset_ablation/render_subsets.py \
    --manifest output/manifest.json \
    --output-dir output \
    --blender /Applications/Blender.app/Contents/MacOS/Blender \
    --workers 4 --samples 64 --use-gpu
```

输出: `output/images/single_view/{subset_id}.png` (220 张)

### Step 5: VLM 评测

```bash
export VLM_BASE_URL="https://openrouter.ai/api/v1"
export VLM_API_KEY="sk-..."
export VLM_MODEL="openai/gpt-4o"

python ../subset_ablation/run_subset_eval.py \
    --questions-dir output/questions/qrr \
    --images-dir output/images/single_view \
    --output-dir output/results/gpt4o \
    --concurrency 4 --batch-size 20
```

输出: `output/results/gpt4o/scenes/{subset_id}.json`

### Step 6: 分析

```bash
python analyze_variable_subsets.py \
    --manifest output/manifest.json \
    --results-dir output/results/gpt4o/scenes \
    --output-dir output/analysis
```

输出:
- `output/analysis/accuracy_by_size.json` — 精度汇总
- `output/analysis/refusal_by_size.json` — 拒答率
- `output/analysis/consistency.json` — 跨尺寸一致性

### Step 7: 可视化

```bash
python visualize_variable_subsets.py \
    --analysis-dir output/analysis \
    --output-dir output/figures \
    --baseline-acc 0.65   # 可选: N=20 全图基线精度
```

输出 (PDF + PNG):
- `fig_accuracy_vs_size` — QRR 精度 vs 子集大小 (核心结果)
- `fig_refusal_vs_size` — 拒答检测率 + Hallucination Rate
- `fig_answerable_ratio` — 可答题比例 (实测 vs 理论)
- `fig_consistency_distribution` — 跨尺寸答案一致性分布

## 目录结构

```
experiments/variable_subset_ablation/
├── README.md                           # 本文件
├── enumerate_variable_subsets.py       # Step 1: 枚举子场景
├── analyze_variable_subsets.py         # Step 6: 分析
├── visualize_variable_subsets.py       # Step 7: 可视化
└── output/                             # 运行产物 (gitignore)
    ├── scenes/                         # 子场景 JSON
    ├── manifest.json                   # 父→子映射
    ├── master_questions/               # N=20 全量题库
    ├── questions/qrr/                  # 每子场景题目
    ├── images/single_view/             # 渲染图像
    ├── results/{model}/scenes/         # VLM 结果
    ├── analysis/                       # 分析 JSON
    └── figures/                        # 可视化图表
```

## 复用的脚本

本实验**不修改**任何现有脚本，仅通过 import 或命令行调用复用:

| 脚本 | 位置 | 用途 |
|------|------|------|
| `generate_master_questions.py` | `experiments/subset_ablation/` | 生成 N=20 全量 QRR Master Bank |
| `assign_subset_questions.py` | `experiments/subset_ablation/` | 标记每题在每子场景中是否可答 |
| `render_subsets.py` | `experiments/subset_ablation/` | Blender 批量渲染 |
| `run_subset_eval.py` | `experiments/subset_ablation/` | VLM API 调用 + 评分 |
| `build_subset_scene_json` | `experiments/subset_ablation/enumerate_subsets.py` | 构建子集 JSON (Python import) |

## Manifest 格式

`manifest.json` 同时支持按大小分组和扁平列表两种访问方式:

```json
{
  "parent_scenes": {
    "n20_000000": {
      "n_objects": 20,
      "subsets_by_size": {
        "9":  [{"subset_id": "...", "object_ids": [...], "subset_size": 9}, ...],
        "10": [...]
      },
      "subsets": [...]  // 扁平列表，兼容现有 subset_ablation 管线
    }
  },
  "config": {
    "min_size": 9, "max_size": 19, "max_subsets": 20, "seed": 42
  }
}
```

## 分析产出说明

### accuracy_by_size.json
```json
{
  "9": {
    "overall": {"correct": 100, "total": 200, "acc": 0.5},
    "disjoint": {"correct": 80, "total": 160, "acc": 0.5},
    "shared_anchor": {"correct": 20, "total": 40, "acc": 0.5},
    "n_scenes": 20,
    "per_scene_acc": [0.45, 0.52, ...]  // 用于误差条
  }
}
```

### refusal_by_size.json
```json
{
  "9": {
    "refusal_total": 17325,
    "refusal_correct": 15000,
    "refusal_rate": 0.8658,
    "hallucinated": 2325,
    "hallucination_rate": 0.1342,
    "answerable_ratio": 0.035
  }
}
```

### consistency.json
```json
{
  "mean_consistency": 0.85,
  "n_questions_multi_size": 5000,
  "per_question": [
    {
      "qid": "mqrr_0001",
      "answers_by_size": {"9": ["<"], "10": ["<", "<"], ...},
      "consistency": 1.0,
      "majority_answer": "<"
    }
  ]
}
```
