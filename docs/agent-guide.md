# ORDINARY-BENCH — AI Agent 操作指南

本文档为 AI Agent 提供端到端操作指引：从数据配置到 VLM 评测到结果批改。

## 目录

1. [数据配置](#1-数据配置)
2. [数据核对](#2-数据核对)
3. [问题生成](#3-问题生成)
4. [配置 VLM 进行测试](#4-配置-vlm-进行测试)
5. [运行评测](#5-运行评测)
6. [结果批改与评分](#6-结果批改与评分)
7. [结果分析 Pipeline](#7-结果分析-pipeline)
8. [Subset Ablation 分析](#8-subset-ablation-分析)
9. [常见问题](#9-常见问题)

---

## 1. 数据配置

### 方式 A：使用仓库内 test-data（推荐，最快）

仓库已包含 140 个 test 场景的完整数据，clone 后即可使用：

```
datasets/test-data/
├── scenes/           # 场景 JSON（物体位置、属性）
├── images/           # 渲染图像
│   ├── single_view/  # 单视角 PNG (480x320)
│   └── multi_view/   # 多视角 (view_0..view_3)
└── questions/        # 预生成问题
    ├── qrr/          # 距离比较（disjoint + shared_anchor）
    ├── trr/          # 钟面方向
    └── fdr/          # 全距离排序
```

无需额外配置，直接进入 [第4节](#4-配置-vlm-进行测试)。

### 方式 B：从 HuggingFace 下载

```bash
python datasets/hf_to_local.py \
    --repo TYTSTQ/ordinary-bench \
    --split test \
    --output ./hf_data \
    --workers 4
```

输出目录结构与 test-data 一致，可直接用于评测。

### 方式 C：从头生成数据

需要 Blender。步骤：

```bash
# 1. 生成场景和图像
cd data-gen && python generate.py --preset test

# 2. 生成问题
cd ../VLM-test && python generate_questions_v2.py --data ../data-gen/output
```

---

## 2. 数据核对

运行以下检查确保数据完整且格式正确。

### 2.1 文件完整性

```bash
# 进入数据目录
DATA_DIR="datasets/test-data"  # 或 hf_data, 或 data-gen/output

# 检查场景文件
echo "Scenes: $(ls $DATA_DIR/scenes/*.json | wc -l)"

# 检查图像
echo "Single-view: $(ls $DATA_DIR/images/single_view/*.png | wc -l)"
echo "Multi-view dirs: $(ls -d $DATA_DIR/images/multi_view/*/ | wc -l)"

# 检查问题
for t in qrr trr fdr; do
    echo "$t questions: $(ls $DATA_DIR/questions/$t/*.json 2>/dev/null | wc -l)"
done
```

期望输出（test split）：Scenes=140, Single-view=140, QRR=140, TRR=140, FDR=140

### 2.2 QRR 变体检查

QRR 应包含 disjoint 和 shared_anchor 两种变体：

```python
import json, glob
disjoint = shared = 0
for f in glob.glob(f"{DATA_DIR}/questions/qrr/*.json"):
    data = json.load(open(f))
    for batch in data["batches"]:
        for q in batch["questions"]:
            if q["variant"] == "disjoint": disjoint += 1
            elif q["variant"] == "shared_anchor": shared += 1
print(f"disjoint: {disjoint}, shared_anchor: {shared}")
# 期望: disjoint: 26136, shared_anchor: 18583
```

### 2.3 问题格式校验

```python
import json

# 检查单个 QRR 问题文件
with open(f"{DATA_DIR}/questions/qrr/n04_000080.json") as f:
    data = json.load(f)

# 必须包含的字段
assert "scene_id" in data
assert "objects" in data     # [{"id": "obj_0", "desc": "..."}]
assert "batches" in data     # [{batch_id, n_questions, questions}]

# 检查问题字段
q = data["batches"][0]["questions"][0]
assert q["type"] == "qrr"
assert q["variant"] in ("disjoint", "shared_anchor")
assert q["gt_comparator"] in ("<", "~=", ">")
assert "pair1" in q and "pair2" in q
```

### 2.4 场景-图像对齐

```python
import os
scenes = {f[:-5] for f in os.listdir(f"{DATA_DIR}/scenes") if f.endswith(".json")}
images = {f[:-4] for f in os.listdir(f"{DATA_DIR}/images/single_view") if f.endswith(".png")}
questions = {f[:-5] for f in os.listdir(f"{DATA_DIR}/questions/qrr") if f.endswith(".json")}

assert scenes == images == questions, f"Mismatch: {scenes - images}, {images - questions}"
print(f"All {len(scenes)} scenes have matching images and questions")
```

---

## 3. 问题生成

### 3.1 主数据集问题生成

问题生成是**完全确定性的**：相同 scene JSON + tau → 相同问题。

```bash
cd VLM-test
python generate_questions_v2.py --data ../data-gen/output --tau 0.10 --batch-size 20
```

参数说明：
- `--tau 0.10`：容差参数，控制 `~=`（近似相等）的判定阈值
- `--batch-size 20`：每批问题数（影响 VLM 单次请求的问题量）
- `--split n04`：只处理特定 split

### 3.2 Subset ablation 问题生成

```bash
cd datasets/test-data/subset_ablation
python generate_questions.py
# 输出: questions/qrr/ 下 912 个文件
```

### 3.3 问题类型说明

| 类型 | 描述 | 答案格式 | 每场景数量（n=4） | 每场景数量（n=10） |
|------|------|---------|-------------------|-------------------|
| QRR | 比较两对物体间距离 | `<` / `~=` / `>` | ~14 | ~930 |
| TRR | 钟面方向推理 | 整数 1-12 | 24 | 720 |
| FDR | 全距离排序 | `["obj_2", "obj_1", ...]` | 4 | 10 |

### 3.4 问题确定性验证

```python
# 同一场景重新生成，结果应完全一致
import subprocess, json
result1 = subprocess.run(["python", "generate_questions_v2.py", "--data", "../data-gen/output", "--split", "n04"], capture_output=True)
# 对比 output/questions/qrr/n04_000080.json 的内容
```

---

## 4. 配置 VLM 进行测试

### 4.1 Job TOML 配置

所有评测配置通过 TOML 文件控制。创建 `my_job.toml`：

```toml
job_name = "my_eval"

[provider]
adapter = "openai_chat"          # openai_chat | gemini_native | mock_static
model = "openai/gpt-4o"          # 模型标识符
base_url = "https://openrouter.ai/api/v1"
api_key = "env:VLM_API_KEY"      # 从环境变量读取

[provider.options]
temperature = 0.0                # 建议 0 确保可复现
max_tokens = 65536
max_retries = 5
retry_base_delay = 2.0           # 指数退避基础延迟（秒）
timeout = 120                    # 单次请求超时（秒）
max_concurrency = 4              # 并行场景数

[input]
questions_dir = "../../datasets/test-data/questions"
question_layout = "v2"           # v2: 按题型分目录
question_types = ["qrr", "trr", "fdr"]
batch_size = 20                  # 每次请求的问题数

[images]
mode = "single"                  # single | multi_view | none
single_view_root = "../../datasets/test-data/images/single_view"
# 多视角模式：
# mode = "multi_view"
# multi_view_root = "../../datasets/test-data/images/multi_view"
# n_views = 4

[selection]
split = ""                       # 筛选 split（如 "n04"），空=全部
scene = ""                       # 单个场景 ID
max_scenes = 10                  # 限制场景数（调试用）

[prompt]
react_max_rounds = 2             # 响应不完整时的重试轮数
missing_threshold = 0.2          # 缺失答案超过 20% 触发重试

[output]
results_dir = "./results"
run_name = "my_eval"
```

### 4.2 常用模型配置示例

```toml
# GPT-4o (via OpenRouter)
[provider]
adapter = "openai_chat"
model = "openai/gpt-4o"
base_url = "https://openrouter.ai/api/v1"
api_key = "env:OPENROUTER_API_KEY"

# Gemini 2.0 Flash (via Google AI)
[provider]
adapter = "gemini_native"
model = "gemini-2.0-flash"
api_key = "env:GOOGLE_API_KEY"

# Qwen2.5-VL (via DashScope)
[provider]
adapter = "openai_chat"
model = "qwen-vl-max-latest"
base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
api_key = "env:DASHSCOPE_API_KEY"

# 本地模型 (vLLM / Ollama)
[provider]
adapter = "openai_chat"
model = "local-model"
base_url = "http://localhost:8000/v1"
api_key = "none"
```

### 4.3 环境变量

```bash
export VLM_API_KEY="your-api-key"
# 或在 TOML 中使用 api_key = "env:VLM_API_KEY"
```

---

## 5. 运行评测

### 5.1 执行评测

```bash
cd VLM-test/API-test
python run_eval.py --job my_job.toml
```

### 5.2 烟雾测试（不需要 API）

```bash
python run_eval.py --job jobs/mock_smoke.toml
# 使用 mock adapter，不调用真实 API，验证管线完整性
```

### 5.3 评测流程

```
run_eval.py --job my_job.toml
  │
  ├─ 发现场景（discover_scene_ids）
  │     └─ 按 selection 筛选
  │
  ├─ 并行处理每个场景（ThreadPoolExecutor）
  │   ├─ 加载问题（question_loader）
  │   ├─ 加载图像（image_resolver → base64）
  │   ├─ 构建 prompt（system_prompt + 物体描述 + 问题列表）
  │   ├─ 调用 VLM API
  │   ├─ 解析响应（JSON 提取 + 修复）
  │   ├─ 缺失检测 → ReAct 重试
  │   └─ 评分（score_batch_scene）
  │
  └─ 汇总 → summary.json
```

### 5.4 中断恢复

如果评测中断，重新运行时会跳过已有 raw 文件的场景（通过 result_store 检查）。

---

## 6. 结果批改与评分

### 6.1 评分规则

#### QRR（距离比较）
- **指标**：精确匹配（predicted == gt_comparator）
- **答案**：`<`、`~=`、`>`
- **分别统计**：overall / disjoint / shared_anchor 三个准确率

#### TRR（钟面方向）
- **hour_correct**：精确小时匹配（1-12）
- **quadrant_correct**：正确象限（宽松）
- **adjacent_correct**：±1 小时内（最宽松）

#### FDR（距离排序）
- **exact_correct**：完全匹配排序（考虑 tie groups）
- **kendall_tau**：Kendall τ 相关系数 [-1, 1]
- **pairwise_accuracy**：正确的成对排序比例
- **top1_correct**：最近物体正确

### 6.2 结果文件结构

```
results/{run_name}/
├── raw/                        # 原始 VLM 响应
│   ├── n04_000080_qrr_0.json   # scene_id + 题型 + batch_id
│   └── ...
├── scenes/                     # 逐场景评分
│   ├── n04_000080.json
│   └── ...
└── summary.json                # 汇总指标
```

### 6.3 summary.json 格式

```json
{
  "overall": {
    "qrr_accuracy": 0.65,
    "qrr_correct": 1200,
    "qrr_total": 1846,
    "qrr_disjoint_accuracy": 0.68,
    "qrr_shared_anchor_accuracy": 0.61,
    "trr_hour_accuracy": 0.32,
    "trr_quadrant_accuracy": 0.58,
    "trr_adjacent_accuracy": 0.52,
    "trr_total": 2400,
    "fdr_exact_accuracy": 0.12,
    "fdr_kendall_mean": 0.45,
    "fdr_pairwise_mean": 0.72,
    "fdr_top1_mean": 0.58,
    "fdr_total": 140,
    "missing": 5
  },
  "by_split": {
    "n04": { /* 同 overall 结构 */ },
    "n05": { /* ... */ }
  }
}
```

### 6.4 手动批改单个场景

```python
import json

# 加载场景结果
with open("results/my_eval/scenes/n04_000080.json") as f:
    result = json.load(f)

# 查看 per_question 详情
for q in result["scores"]["per_question"]:
    print(f"{q['qid']}: predicted={q['predicted']}, gt={q['gt']}, correct={q['correct']}")
```

### 6.5 自定义评分

评分代码在 `VLM-test/API-test/scoring.py`：

```python
from scoring import score_qrr, score_trr_hour, score_fdr_exact

# QRR
correct = score_qrr("<", "<")       # True
correct = score_qrr(">", "<")       # False

# TRR
correct = score_trr_hour(3, 3)      # True (exact)

# FDR
correct = score_fdr_exact(
    predicted=["obj_1", "obj_2", "obj_3"],
    gt_ranking=["obj_1", "obj_2", "obj_3"],
    tie_groups=[["obj_1"], ["obj_2"], ["obj_3"]]
)
```

---

## 7. 结果分析 Pipeline

评测完成后，`VLM-test/analysis/` 提供完整的分析工具链。所有脚本从 `VLM-test/` 目录运行。

### 7.1 一键全套分析

```bash
cd VLM-test
python analysis/run_analysis.py
```

自动发现 `output/results/` 下的所有模型，依次运行：
1. 精度汇总表（Markdown + JSON）
2. 批量场景重建（CSR / NRMS / Kendall τ）
3. 一致性分析（传递性 / 互反性）
4. 可视化图表

参数：
- `--results-base output/results` — 结果根目录
- `--questions-dir output/questions` — 问题目录
- `--scenes-dir ../data-gen/output/scenes` — 场景 JSON 目录
- `--output-dir output/analysis` — 分析输出目录
- `--max-scenes 10` — 限制场景数（调试用）
- `--restarts 10` — 重建优化器重启次数

### 7.2 精度汇总

```bash
python analysis/aggregate.py
```

自动发现模型，输出 Markdown 精度表（QRR overall / disjoint / shared_anchor, TRR hour / quadrant / adjacent, FDR exact / Kendall / pairwise / top-1），按 split 分组。

### 7.3 Excel 报告导出

```bash
python analysis/generate_insight_excel.py
python analysis/generate_insight_excel.py --output path/to/report.xlsx
```

生成多 sheet Excel（模型排行、视角对比、难度曲线、重建指标、逐场景明细）。

### 7.4 场景重建

从 VLM 预测的空间约束重建 2D 物体位置：

```bash
# Belief 模式（使用 VLM 预测，含错误答案）
python analysis/reconstruct_scenes.py \
    -r output/results/gpt-4o \
    -q output/questions \
    -s ../data-gen/output/scenes \
    --belief --restarts 10

# 输出：逐场景 CSR / NRMS / Kendall τ + 汇总统计
```

### 7.5 合并管线（Excel + 重建 + SVG）

```bash
python analysis/export_and_reconstruct.py
python analysis/export_and_reconstruct.py --excel-only     # 仅 Excel
python analysis/export_and_reconstruct.py --recon-only     # 仅重建 + SVG
python analysis/export_and_reconstruct.py --models gpt-4o,claude-sonnet --max-scenes 5
```

输出：
- Excel 报告（5 sheet）
- `output/analysis/belief_recon/{model}/{scene_id}.json` — 重建结果
- `output/analysis/belief_recon/{model}/{scene_id}.svg` — GT vs Recon 可视化

### 7.6 一致性检查

`analysis/consistency.py` 是库模块（无 CLI），被 `run_analysis.py` 调用：
- **传递性**：QRR `d(A,B) < d(C,D) < d(E,F)` → `d(A,B) < d(E,F)` 是否成立
- **互反性**：TRR 180° 镜像对称检查

### 7.7 约束冲突检测

```bash
# FDR 排序与 QRR 距离比较的矛盾检测
python analysis/fdr_qrr_conflict.py
```

### 7.8 分析输出结构

```
output/analysis/
├── accuracy_table.md          # Markdown 精度表
├── accuracy_table.json        # JSON 精度数据
├── consistency.json           # 一致性分析结果
├── belief_recon/              # 重建结果
│   └── {model}/
│       ├── {scene_id}.json    # 重建位置 + 指标
│       └── {scene_id}.svg     # GT vs Recon 可视化
├── results_summary.xlsx       # Excel 综合报告
└── figures/                   # 可视化图表
```

---

## 8. Subset Ablation 分析

评测完 subset 场景后的分析流程。

### 8.1 全图 vs 子集对比

```bash
cd experiments/subset_ablation
python analyze_results.py \
    --full-results output/results/full \
    --subset-results output/results/subset \
    --mapping output/question_mapping.json \
    --output output/analysis
```

输出：
- `comparison.json` — per-split / per-variant 精度对比
- `self_agreement.json` — 同一问题在多个子集中的自一致性

### 8.2 子集结果聚合为父场景格式

将子集评测结果通过多数投票聚合为父场景格式，可直接接入重建管线：

```bash
python aggregate_to_parent.py \
    --results-dir output/results/gpt4o_multiview/scenes \
    --master-dir output/master_questions \
    --scenes-dir ../../data-gen/output/scenes \
    --output-dir output/aggregated/gpt4o_multiview \
    --model gpt-4o \
    --vote-threshold 0.5
```

输出：
- `{parent_scene_id}.json` — 与 `run_eval.py` 输出格式一致的聚合结果
- `summary.json` — 聚合统计（投票率、覆盖率）

然后接入重建管线：

```bash
cd ../../VLM-test
python analysis/reconstruct_scenes.py \
    -r ../experiments/subset_ablation/output/aggregated/gpt4o_multiview \
    -q ../experiments/subset_ablation/output/master_questions \
    --belief
```

---

## 9. 常见问题

### Q: 评测中断了怎么办？
重新运行同一个 job TOML，已完成的场景会被跳过。

### Q: 如何只测试 QRR？
在 TOML 中设置 `question_types = ["qrr"]`。

### Q: 如何限制测试场景数？
设置 `max_scenes = 5` 或 `split = "n04"` 或 `scene = "n04_000080"`。

### Q: VLM 响应解析失败？
`response_parser.py` 有多级 fallback（去除 think 标签、修复截断 JSON、正则提取）。如果仍失败，raw/ 目录保留了原始响应供人工检查。

### Q: 如何对比多个模型？
每个模型使用不同的 `run_name`，结果存在不同目录。`summary.json` 中的指标可直接对比。

### Q: 问题是确定性生成的吗？
是。相同 scene JSON + tau=0.10 → 完全相同的问题（相同顺序、相同 QID）。无随机性。

---

## 参考文件

| 文件 | 说明 |
|------|------|
| `VLM-test/docs/scoring_criteria.md` | 详细评分公式和示例 |
| `VLM-test/API-test/README.md` | Job TOML 配置参考 |
| `VLM-test/analysis/` | 分析脚本目录（精度汇总、重建、一致性、可视化） |
| `docs/pipeline-overview.md` | 完整管线流程图（Mermaid） |
| `datasets/README.md` | HuggingFace 数据集说明 |
| `datasets/prompts/system_prompts.json` | VLM 系统 prompt 模板 |
| `CLAUDE.md` | 项目架构详细指引 |
