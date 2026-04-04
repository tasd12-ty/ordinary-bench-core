# ORDINARY-BENCH — AI Agent 操作指南

本文档为 AI Agent 提供端到端操作指引：从数据配置到 VLM 评测到结果批改。

## 目录

1. [数据配置](#1-数据配置)
2. [数据核对](#2-数据核对)
3. [问题生成](#3-问题生成)
4. [配置 VLM 进行测试](#4-配置-vlm-进行测试)
5. [运行评测](#5-运行评测)
6. [结果批改与评分](#6-结果批改与评分)
7. [常见问题](#7-常见问题)

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

## 7. 常见问题

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
| `docs/pipeline-overview.md` | 完整管线流程图（Mermaid） |
| `datasets/README.md` | HuggingFace 数据集说明 |
| `datasets/prompts/system_prompts.json` | VLM 系统 prompt 模板 |
| `CLAUDE.md` | 项目架构详细指引 |
