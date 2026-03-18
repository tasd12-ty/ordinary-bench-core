# ORDINARY-BENCH 操作指南

评估视觉语言模型 (VLM) 序数空间关系理解能力的基准测试。

三阶段流水线：
1. **数据生成** — 用 Blender 生成 3D 场景并渲染图片
2. **问题生成** — 从场景元数据生成 QRR/TRR 评测问题
3. **VLM 测试** — 通过 API 测试 VLM 并评分

---

## 环境准备

### 依赖

- Python >= 3.9
- [Blender](https://www.blender.org/)（场景渲染，需手动安装）
- Python 包：`numpy`, `openai`

### 安装

```bash
# 推荐使用 uv
uv sync

# 或 pip
pip install -e .
```

### 目录结构

```
ordinary-bench/
├── data-gen/                      # 第一阶段：场景生成与渲染
│   ├── generate.py                # 生成入口
│   ├── pipeline.py                # Blender 子进程编排
│   ├── config.toml                # 默认配置（70 场景）
│   ├── config_expand.toml         # 扩充配置（增量生成 630 新场景）
│   ├── rebuild_splits.py          # 重建 split 索引
│   └── blender/                   # Blender 脚本与资产
├── data-gen-infinigen/            # Infinigen 真实感室内场景后端
│   ├── generate.py                # Infinigen-Indoors 编排器
│   ├── adapter.py                 # Infinigen → ordinary-bench 转换
│   └── README.md                  # 后端文档
├── VLM-test/                      # 第二、三阶段
│   ├── generate_questions.py      # 问题生成
│   ├── generate_questions_v2.py   # 分题型目录输出（推荐）
│   ├── question_bank.py           # 问题枚举逻辑
│   ├── extraction.py              # Ground truth 提取
│   ├── dsl/                       # 空间关系 DSL
│   ├── reconstruct/               # 场景重建管线
│   │   ├── constraints.py         # 约束预处理与可行性检查
│   │   ├── solver.py              # 基于梯度的 2D 位置优化器
│   │   ├── pipeline.py            # 端到端重建入口
│   │   └── evaluate.py            # 重建质量评估
│   ├── docs/
│   │   └── scoring_criteria.md    # 评分标准文档
│   └── API-test/                  # VLM API 测试
│       ├── run_batch.py           # 单视角评测
│       ├── run_batch_v2.py        # 分题型目录评测（推荐）
│       ├── run_multi_view.py      # 多视角评测
│       ├── config.py              # API 配置（环境变量）
│       ├── vlm_client.py          # OpenAI 兼容客户端
│       ├── prompts.py             # 提示模板
│       ├── response_parser.py     # 响应解析
│       └── scoring.py             # 评分逻辑
└── pyproject.toml
```

---

## 第一阶段：数据生成

### 配置说明

编辑 `data-gen/config.toml`：

```toml
[blender]
executable = "/path/to/blender"   # Blender 可执行文件路径
use_gpu = true                    # 启用 GPU 渲染

[rendering]
width = 480          # 图片宽度
height = 320         # 图片高度
samples = 256        # Cycles 采样数（越低越快，噪点越多）
n_views = 4          # 每个场景的摄像机视角数
camera_distance = 12.0
elevation = 30.0     # 俯仰角（度）
azimuth_start = 45.0 # 起始方位角（度）

[objects]
min_dist = 0.25      # 物体间最小距离
margin = 0.4         # 放置边距

[output]
dir = "./output"     # 输出目录
seed = 42            # 随机种子

# Split 定义 — 每个 split 固定物体数量
[splits.n04]
n_scenes = 10        # 场景数量
min_objects = 4      # 物体数量
max_objects = 4
# ... n05 到 n10 类似
```

### 生成命令

```bash
cd data-gen

# 完整生成（使用 config.toml，70 场景）
python generate.py

# 快速测试（每 split 1 个场景，低采样）
python generate.py --preset test

# 并行渲染（4 个 Blender 进程）
python generate.py --workers 4

# 指定自定义配置
python generate.py --config my_config.toml

# 覆盖 Blender 路径和输出目录
python generate.py --blender /usr/bin/blender --output-dir ./my_output

# 启用 GPU
python generate.py --gpu

# 预览配置（不实际渲染）
python generate.py --dry-run
```

### 输出结构

```
data-gen/output/
├── images/
│   ├── single_view/    # 每场景 1 张图片（n04_000000.png ...）
│   └── multi_view/     # 每场景 4 个视角（n04_000000/view_0.png ...）
├── scenes/             # 场景元数据 JSON（物体位置、属性）
├── splits/             # Split 索引文件
└── dataset_info.json   # 数据集摘要
```

---

## 扩充场景：从 70 个到 700 个

当前数据集包含 70 个场景（7 个 split × 10 个场景）。以下步骤将其扩充到 700 个（7 × 100），**保留现有 70 个场景**。

### 步骤 1：验证配置

```bash
cd data-gen

# 预览增量配置，确认参数正确
python generate.py --config config_expand.toml --start-idx 10 --dry-run
```

确认输出中每个 split 显示 `n_scenes: 90`，`start_idx: 10`。

### 步骤 2：增量生成新场景

```bash
# 生成 630 个新场景（每 split 90 个，编号 10-99）
# 现有场景（编号 0-9）不受影响
python generate.py --config config_expand.toml --start-idx 10 --workers 4
```

参数说明：
- `--config config_expand.toml`：使用增量配置（每 split 90 个场景）
- `--start-idx 10`：场景编号从 10 开始（n04_000010, n04_000011, ...）
- `--workers 4`：4 个 Blender 并行进程（根据 CPU/GPU 调整）

> **提示**：渲染 630 个场景耗时较长。可以先用 `--preset test` 做小规模验证。

### 步骤 3：重建 Split 索引

增量生成后需要重建 split 索引。有两种方式：

**方式 A**（推荐）：增量生成会自动合并 split 索引（`start_idx > 0` 时自动追加而非覆盖）。

**方式 B**：手动重建完整索引：

```bash
python rebuild_splits.py --output-dir ./output
```

验证：
```bash
# 检查场景文件总数，应为 700
ls output/scenes/*.json | wc -l

# 查看 dataset_info.json 确认总数
cat output/dataset_info.json | python -m json.tool | grep total_scenes
```

### 步骤 4：重新生成问题

```bash
cd ../VLM-test

# 为所有 700 个场景生成问题
python generate_questions.py --data ../data-gen/output
```

### 完整流程（一键执行）

```bash
# 从项目根目录开始
cd data-gen

# 1. 增量生成（split 索引会自动合并）
python generate.py --config config_expand.toml --start-idx 10 --workers 4

# 2. 生成问题
cd ../VLM-test
python generate_questions.py --data ../data-gen/output

# 3. 验证
python generate_questions.py --counts
```

### 服务器部署（Linux 无头渲染）

```bash
# 1. 安装 Blender（推荐下载官方包）
wget https://download.blender.org/release/Blender4.2/blender-4.2.0-linux-x64.tar.xz
tar -xf blender-4.2.0-linux-x64.tar.xz
export PATH=$PATH:$(pwd)/blender-4.2.0-linux-x64

# 2. 验证
blender --version

# 3. 生成（用 --blender 覆盖 config.toml 中的路径）
cd data-gen
python generate.py --blender blender --gpu --workers 7
```

支持 CUDA/OptiX/HIP GPU 加速，需安装对应驱动。

### 生成耗时参考

基于实测（macOS M 系列，CPU 渲染，256 samples）：

| 场景数 | 耗时 | 每场景 |
|--------|------|--------|
| 70 | ~11 分钟 | ~9.2 秒 |
| 700 | ~1.8 小时（预估） | ~9.2 秒 |
| 5,000 | ~13 小时（预估） | ~9.2 秒 |

GPU 渲染可加速约 3 倍。并行 worker 可进一步缩短墙钟时间。

### 随机种子与可复现性

`config.toml` 中的 `seed = 42` 与 `--start-idx` 组合确保：
- 同一 seed + start_idx 生成完全相同的场景（可复现）
- 不同 start_idx 生成不同场景（增量不重复）

### 700 场景的问题数量估算

每种物体数量 100 个场景：

| 物体数 | QRR-D | QRR-SA | QRR 合计 | TRR | FDR | 合计 |
|--------|-------|--------|---------|-----|-----|------|
| 4      | 3     | 12     | 15      | 24  | 4   | 43   |
| 5      | 15    | 30     | 45      | 60  | 5   | 110  |
| 6      | 45    | 60     | 105     | 120 | 6   | 231  |
| 7      | 105   | 105    | 210     | 210 | 7   | 427  |
| 8      | 210   | 168    | 378     | 336 | 8   | 722  |
| 9      | 378   | 252    | 630     | 504 | 9   | 1143 |
| 10     | 630   | 360    | 990     | 720 | 10  | 1720 |

> 注：QRR-D = 不相交对配对数，QRR-SA = 共享锚点配对数，TRR = 全排列 P(N,3)，FDR = N（每个物体作为锚点各一题）。

---

## 第二阶段：问题生成

从场景 JSON 中枚举所有 QRR、TRR 和 FDR 问题，计算 Ground Truth。

### 命令

```bash
cd VLM-test

# v1 — 原始格式（所有题型混合存储）
python generate_questions.py --data ../data-gen/output
python generate_questions.py --data ../data-gen/output --split n04
python generate_questions.py --data ../data-gen/output --batch-size 10 --tau 0.10
python generate_questions.py --counts

# v2 — 分题型目录存储（推荐）
python generate_questions_v2.py --data ../data-gen/output
python generate_questions_v2.py --data ../data-gen/output --split n04
python generate_questions_v2.py --counts
```

### 问题类型

**QRR（四元相对关系）** 有两个变体：
- `disjoint`：比较两组不相交对的距离 `dist(A,B)` vs `dist(C,D)`
- `shared_anchor`：固定锚点，比较 `dist(A,B)` vs `dist(A,C)`

格式：`dist(A, B) < / ~= / > dist(C, D)?`
- 答案：`<`（更近）、`~=`（近似相等）、`>`（更远）
- 容差参数 `tau = 0.10`：`|a-b| ≤ tau × max(a,b)` 判定为 `~=`

**TRR（三元钟面关系）**：站在 ref1 面朝 ref2（12 点方向），target 在几点钟？
- 答案：整数 1-12

**FDR（全距离排序）**：以某物体为锚点，将其余物体按距离从近到远排序。
- 答案：有序 ID 列表 `["nearest", ..., "farthest"]`
- 并列组：τ 容差内的物体可任意排列

### 输出

v2 推荐的分题型目录结构：

```
VLM-test/output/
├── questions/
│   ├── qrr/{scene_id}.json    # QRR (disjoint + shared_anchor)
│   ├── trr/{scene_id}.json    # TRR
│   └── fdr/{scene_id}.json    # FDR
├── extraction_tasks/
│   ├── qrr/{scene_id}.json
│   ├── trr/{scene_id}.json
│   └── fdr/{scene_id}.json
└── summary.json
```

---

## 第三阶段：VLM 测试

### 环境变量配置

```bash
# 必填：API 密钥
export VLM_API_KEY="your-api-key"

# API 端点（默认 OpenRouter）
export VLM_BASE_URL="https://openrouter.ai/api/v1"

# 模型选择
export VLM_MODEL="google/gemini-2.0-flash-001"

# 可选：OpenRouter 供应商路由
export VLM_PROVIDER="google"

# 并发与重试
export VLM_CONCURRENCY=4        # 并行场景数（默认 4）
export VLM_TIMEOUT=120          # 请求超时秒数（默认 120）
export VLM_MAX_RETRIES=5        # 最大重试次数（默认 5）
export VLM_RETRY_DELAY=2.0      # 重试基础延迟秒数（默认 2.0）
```

### 运行评测

#### 单视角模式

```bash
cd VLM-test/API-test

# 评测所有场景
python run_batch.py

# 评测指定 split
python run_batch.py --split n04

# 评测单个场景
python run_batch.py --scene n04_000000

# v2 — 从分题型目录加载（推荐）
python run_batch_v2.py --split n04
python run_batch_v2.py --scene n04_000000
```

#### 多视角模式

```bash
# 4 个视角（默认）
python run_multi_view.py

# 2 个视角
python run_multi_view.py --n-views 2

# 指定 split
python run_multi_view.py --split n04
```

### 切换模型

通过环境变量即时切换，无需修改代码：

```bash
# GPT-4o（通过 OpenRouter）
VLM_MODEL="openai/gpt-4o" python run_batch.py

# Qwen2.5-VL-72B
VLM_MODEL="qwen/qwen-2.5-vl-72b-instruct" python run_batch.py

# Qwen3-VL-235B（思考模型）
VLM_MODEL="qwen/qwen3-vl-235b-a22b-thinking" python run_batch.py

# 本地模型（SGLang 部署）
VLM_BASE_URL="http://localhost:8000/v1" VLM_API_KEY="EMPTY" VLM_MODEL="my-model" python run_batch.py
```

### 思考模型（SGLang + Qwen 3.5 等）

使用 SGLang 部署思考模型时，需注意：

```bash
# SGLang 启动参数（需要 --reasoning-parser）
sglang serve --model-path /path/to/qwen3.5 --reasoning-parser qwen3 --tp-size 8

# 评测时增大 max_tokens（思考模型需要大量 token）
VLM_BASE_URL="http://localhost:8000/v1" \
VLM_API_KEY="EMPTY" \
VLM_MODEL="qwen3p5-122a10b" \
VLM_MAX_TOKENS=65536 \
VLM_CONCURRENCY=16 \
VLM_TIMEOUT=60000 \
python run_batch.py
```

`max_tokens` 包含思考 token + 回答 token 的总量。如果思考用完预算导致 content 为空，系统会自动用 `enable_thinking=False` 重试。

### 查看结果

结果按模型名存放：

```
VLM-test/output/results/<model>/
├── raw/           # 每 batch 的原始 VLM 响应
├── scenes/        # 每场景的评分详情
└── summary.json   # 汇总指标
```

其中模型名中的 `/` 替换为 `--`（如 `qwen/qwen3-vl` → `qwen--qwen3-vl`）。

多视角模式结果在 `<model>_multi_view/` 目录下。

### 评测指标

| 指标 | 说明 |
|------|------|
| **QRR Accuracy** | 比较器精确匹配（`<` / `~=` / `>`） |
| **QRR Disjoint Accuracy** | 不相交对 QRR 准确率 |
| **QRR Shared-Anchor Accuracy** | 共享锚点 QRR 准确率 |
| **TRR Hour Accuracy** | 钟面小时精确匹配（1-12） |
| **TRR Quadrant Accuracy** | 象限匹配（4 象限） |
| **TRR Adjacent Accuracy** | ±1 小时容差匹配 |
| **FDR Exact Accuracy** | 完整排序精确匹配（尊重并列组） |
| **FDR Kendall τ** | 排序相关系数 |
| **FDR Pairwise Accuracy** | 成对序关系正确率 |
| **FDR Top-1 Accuracy** | 最近物体正确率 |

`summary.json` 示例：

```json
{
  "model": "qwen/qwen3-vl-235b-a22b-thinking",
  "n_scenes": 700,
  "overall": {
    "qrr_accuracy": 0.45,
    "qrr_disjoint_accuracy": 0.48,
    "qrr_shared_anchor_accuracy": 0.43,
    "trr_hour_accuracy": 0.12,
    "trr_quadrant_accuracy": 0.35,
    "fdr_exact_accuracy": 0.15,
    "fdr_kendall_mean": 0.62,
    "fdr_pairwise_mean": 0.73,
    "fdr_top1_mean": 0.45
  }
}
```

---

## Infinigen 后端

`data-gen-infinigen/` 提供基于 [Infinigen](https://infinigen.org/) 的真实感室内场景生成后端。

### 特性

- Infinigen-Indoors 单房间场景
- adapter 将 Infinigen 元数据转换为 ordinary-bench 场景 JSON
- 坐标系转换：x_bench = x_world, y_bench = z_world, z_bench = -y_world
- 单视角/多视角图像导出
- bootstrap 模式：无需 Blender/Infinigen 即可测试

### 快速验证

```bash
# 验证 adapter 兼容性
python3 data-gen-infinigen/validate.py

# 预览 Infinigen 命令（不实际运行）
python3 data-gen-infinigen/generate.py --dry-run
```

详见 `data-gen-infinigen/README.md`。

---

## 场景重建

`VLM-test/reconstruct/` 从 VLM 预测的空间约束重建 2D 物体位置。

### 管线流程

1. **约束提取**：QRR/TRR/FDR 预测 → 符号约束（FDR 排序分解为 shared_anchor QRR 成对约束）
2. **可行性检查**：QRR 偏序 DAG 循环检测、TRR 角度区间交集、超图连通性
3. **数值优化**：多重启梯度下降，hinge loss 约束
4. **评估**：CSR（约束满足率）、Kendall τ、NRMS、K_geom（几何模态数）

### 评分标准

完整评分标准文档：`VLM-test/docs/scoring_criteria.md`

---

## 快速验证流程

用最少的时间验证整条流水线是否正常工作：

```bash
# 1. 快速生成测试数据（7 个场景，低质量）
cd data-gen
python generate.py --preset test

# 2. 生成问题
cd ../VLM-test
python generate_questions.py --data ../data-gen/output

# 3. 测试单个场景
cd API-test
export VLM_API_KEY="your-key"
export VLM_MODEL="google/gemini-2.0-flash-001"
python run_batch.py --scene n04_000000
```
