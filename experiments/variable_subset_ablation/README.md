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
Step 0: 生成 N=20 场景           → data-gen/generate.py --config config_n20.toml
Step 1: 枚举各尺寸子场景         → enumerate_variable_subsets.py     (本实验)
Step 2: 渲染 5 视角图像           → render_all_views.py               (本实验)
Step 3: 生成题目 + VLM 评测       → 复用 subset_ablation 管线
Step 4: 分析 + 可视化            → analyze / visualize              (本实验)
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

### Step 2: 渲染 5 视角图像

每个子场景渲染 **5 个视角**：4 侧视角（方位角 45°/135°/225°/315°，仰角 30°）+ 1 俯视角（仰角 90°）。

```bash
python render_all_views.py \
    --manifest output/manifest.json \
    --output-dir output \
    --blender /Applications/Blender.app/Contents/MacOS/Blender \
    --workers 4 --samples 64 --use-gpu
```

输出:
- `output/images/multi_view/{subset_id}/view_0.png ~ view_3.png` — 4 侧视角
- `output/images/top_view/{subset_id}.png` — 1 俯视角

> 共 220 × 5 = **1,100 张图像**。
> 调试时用 `--limit 5` 只渲染前 5 个子场景。

### Step 3: 生成题目 + VLM 评测

题目生成和评测可通过两种方式进行：

**方式 A：Master Bank 模式**（含 N/A 拒答测试）
```bash
# 生成 Master 题库
python ../subset_ablation/generate_master_questions.py \
    --scenes-dir ../../data-gen/output/scenes --output-dir output --splits n20

# 分配题目
python ../subset_ablation/assign_subset_questions.py \
    --manifest output/manifest.json --master-dir output/master_questions --output-dir output

# 评测（3 种图像模式分别跑）
python ../subset_ablation/run_subset_eval_multiview.py \
    --questions-dir output/questions/qrr --images-dir output/images \
    --output-dir output/results/gpt4o_single --mode pick_view --view-index 0

python ../subset_ablation/run_subset_eval_multiview.py \
    --questions-dir output/questions/qrr --images-dir output/images \
    --output-dir output/results/gpt4o_multi --mode multi_view

python ../subset_ablation/run_subset_eval.py \
    --questions-dir output/questions/qrr \
    --images-dir output/images/top_view \
    --output-dir output/results/gpt4o_top
```

**方式 B：直接生成模式**（仅可答题，效率更高）
```bash
# 用 generate_questions_v2.py 直接为每个子场景生成 QRR 题目
cd ../../VLM-test
for f in ../experiments/variable_subset_ablation/output/scenes/*.json; do
    python generate_questions_v2.py --data-dir - --scene "$f" \
        --output ../experiments/variable_subset_ablation/output/questions_direct
done
```

### Step 4: 分析 + 可视化

```bash
python analyze_variable_subsets.py \
    --manifest output/manifest.json \
    --results-dir output/results/gpt4o_multi/scenes \
    --output-dir output/analysis

python visualize_variable_subsets.py \
    --analysis-dir output/analysis \
    --output-dir output/figures
```

## 3 种评测图像模式

| 模式 | 图像来源 | 评测命令 | 研究目的 |
|------|---------|---------|---------|
| 单侧视角 | `multi_view/{id}/view_0.png` | `--mode pick_view --view-index 0` | 基线 |
| 四侧视角 | `multi_view/{id}/view_0~3.png` | `--mode multi_view` | 多视角是否提升推理 |
| 俯视角 | `top_view/{id}.png` | 单视角模式 | 鸟瞰是否更利于距离判断 |

## 目录结构

```
experiments/variable_subset_ablation/
├── README.md                           # 本文件
├── enumerate_variable_subsets.py       # Step 1: 枚举子场景
├── render_all_views.py                 # Step 2: 渲染 5 视角
├── analyze_variable_subsets.py         # Step 4: 分析
├── visualize_variable_subsets.py       # Step 4: 可视化
└── output/                             # 运行产物 (gitignore)
    ├── scenes/                         # 子场景 JSON
    ├── manifest.json                   # 父→子映射
    ├── images/
    │   ├── multi_view/{id}/view_0~3.png  # 4 侧视角
    │   └── top_view/{id}.png             # 俯视角
    ├── master_questions/               # Master 题库 (方式 A)
    ├── questions/qrr/                  # 每子场景题目
    ├── results/{model}_{mode}/scenes/  # VLM 结果
    ├── analysis/                       # 分析 JSON
    └── figures/                        # 可视化图表
```

## 复用的脚本

本实验**不修改**任何现有脚本，仅通过 import 或命令行调用复用:

| 脚本 | 位置 | 用途 |
|------|------|------|
| `render_subset_blender.py` | `experiments/subset_ablation/` | Blender 渲染（被 render_all_views.py 调用） |
| `generate_master_questions.py` | `experiments/subset_ablation/` | 生成 N=20 全量 QRR Master Bank |
| `assign_subset_questions.py` | `experiments/subset_ablation/` | 标记每题在每子场景中是否可答 |
| `run_subset_eval_multiview.py` | `experiments/subset_ablation/` | 多模式 VLM 评测 |
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
      "subsets": [...]
    }
  },
  "config": {
    "min_size": 9, "max_size": 19, "max_subsets": 20, "seed": 42
  }
}
```
