# 迭代冲突消解 (Iterative Conflict Resolution)

## 背景

VLM 在回答空间关系问题时，即使整体准确率 80%，偶然错误也会在约束图中累积形成**传递性违反（环）**，导致 n≥6 物体的场景无法重建。

本模块通过**靶向重问冲突题目**来区分两类错误：
- **随机噪声**：模型知道答案但偶然出错 → 重问后翻转为正确
- **系统性错误**：模型真的理解错了 → 重问后仍然错误

## 工作原理

```
已有评测结果
    ↓
提取 QRR 约束图 → 检测环（传递性冲突）
    ↓
计算最小反馈弧集 (FAS) → 定位冲突约束
    ↓
映射回原始问题 → 调 VLM 重问 ←──┐
    ↓                              │
用新答案替换 → 重新检测冲突        │
    ↓                              │
FAS 减小了？ ── 是 ────────────────┘
    │
    └─ 连续 N 轮未减小 → 收敛
    ↓
输出诊断报告：
  - 噪声翻转数 / 系统性冲突数
  - 去噪后的评测结果（可用于重建）
```

## 快速开始

```bash
cd VLM-test

# 1. 仅检测冲突（不调 API，不花钱）
uv run python run_conflict_resolution.py \
  --job API-test/jobs/conflict_resolution_gemini.toml \
  --dry-run

# 2. 对单个场景调试
uv run python run_conflict_resolution.py \
  --job API-test/jobs/conflict_resolution_gemini.toml \
  --scene n06_000080

# 3. 指定 split 和数量
uv run python run_conflict_resolution.py \
  --job API-test/jobs/conflict_resolution_gemini.toml \
  --split n06,n07 --max-scenes 5

# 4. 完整执行
uv run python run_conflict_resolution.py \
  --job API-test/jobs/conflict_resolution_gemini.toml
```

## 配置说明

配置文件为 TOML 格式，放在 `API-test/jobs/` 目录下。

### 最小配置模板

```toml
[provider]
adapter = "openai_chat"
model = "模型ID"
base_url = "API地址"
api_key = "env:环境变量名"

[provider.options]
temperature = 0.0

[images]
mode = "single"
single_view_root = "../data-gen/output/images/single_view"

[input]
results_dir = "output/results"
source_run = "结果目录名"
questions_dir = "output/questions"

[resolution]
splits = ["n06", "n07"]

[output]
output_dir = "output/conflict_resolution"
run_name = "运行名称"
```

### 关键配置项

#### `[provider]` — 重问时调用的 VLM

**应与原始评测使用同一个模型**，这样才能判断"同一模型重答是否稳定"。

| 字段 | 说明 | 示例 |
|------|------|------|
| `adapter` | API 协议 | `"openai_chat"` 或 `"gemini_native"` |
| `model` | 模型标识符 | 取决于 API 平台 |
| `base_url` | API 端点 | |
| `api_key` | 密钥，支持 `env:VAR` 格式 | `"env:OPENROUTER_API_KEY"` |

#### `[input].source_run` — 基于哪个已有结果

必须与 `output/results/` 下的目录名**完全一致**：

| source_run 值 | 对应模型 |
|---------------|---------|
| `gemini-3_1-pro-preview` | Gemini 3.1 Pro (single) |
| `gemini-3_1-pro-preview_multi_view` | Gemini 3.1 Pro (multi) |
| `qwen3p5_397B_single_v2` | Qwen3.5 397B (single) |
| `qwen3p5_397B_multi_view` | Qwen3.5 397B (multi) |
| `claude_opus_single_v2` | Claude Opus 4 (single) |
| `claude_opus_multi_view` | Claude Opus 4 (multi) |
| `claude-sonnet-4-6` | Claude Sonnet 4.6 (single) |
| `claude-sonnet-4-6_multi_view` | Claude Sonnet 4.6 (multi) |
| `gpt_5_4_single_v2` | GPT-5.4 (single) |
| `gpt_5_4_multi_view` | GPT-5.4 (multi) |
| `doubao-seed-2_0-pro` | Doubao Seed 2.0 Pro (single) |
| `doubao-seed-2_0-pro_multi_view` | Doubao Seed 2.0 Pro (multi) |
| `kimi_k2_5_single_v2` | Kimi K2.5 (single) |
| `kimi_k2_5_multi_view` | Kimi K2.5 (multi) |

#### `[resolution]` — 消解参数

| 字段 | 默认 | 说明 |
|------|------|------|
| `max_rounds` | 10 | 最大迭代轮数（通常 3-5 轮收敛） |
| `patience` | 2 | FAS 连续几轮未减小判定为收敛 |
| `splits` | `[]` | 目标物体数量，如 `["n06", "n07"]` |
| `max_scenes_per_split` | 全部 | 每个 split 最多处理场景数 |

### 各模型的完整配置示例

<details>
<summary>Gemini（通过 OpenRouter）</summary>

```toml
[provider]
adapter = "openai_chat"
model = "google/gemini-2.5-pro-preview"
base_url = "https://openrouter.ai/api/v1"
api_key = "env:OPENROUTER_API_KEY"

[provider.options]
temperature = 0.0
max_tokens = 8192

[input]
source_run = "gemini-3_1-pro-preview"
```
</details>

<details>
<summary>Qwen3.5（通过 DashScope）</summary>

```toml
[provider]
adapter = "openai_chat"
model = "qwen3.5-plus-2026-02-15"
base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
api_key = "env:DASHSCOPE_API_KEY"

[provider.options]
temperature = 0.0
max_tokens = 8192

[input]
source_run = "qwen3p5_397B_single_v2"
```
</details>

<details>
<summary>GPT-5.4（通过 OpenRouter）</summary>

```toml
[provider]
adapter = "openai_chat"
model = "openai/gpt-5.4"
base_url = "https://openrouter.ai/api/v1"
api_key = "env:OPENROUTER_API_KEY"

[provider.options]
temperature = 0.0
max_tokens = 8192

[input]
source_run = "gpt_5_4_single_v2"
```
</details>

<details>
<summary>Claude Opus（通过 OpenRouter）</summary>

```toml
[provider]
adapter = "openai_chat"
model = "anthropic/claude-opus-4"
base_url = "https://openrouter.ai/api/v1"
api_key = "env:OPENROUTER_API_KEY"

[provider.options]
temperature = 0.0
max_tokens = 8192

[input]
source_run = "claude_opus_single_v2"
```
</details>

## 输出结构

```
output/conflict_resolution/{run_name}/
├── scenes/
│   ├── n06_000080.json    # 消解摘要：轮次、FAS 变化、诊断
│   ├── n06_000081.json
│   └── ...
├── resolved_scenes/
│   ├── n06_000080.json    # 去噪后的完整评测结果（可喂给重建）
│   └── ...
└── summary.json            # 按 split 汇总统计
```

### summary.json 示例

```json
{
  "by_split": {
    "n06": {
      "n_scenes": 20,
      "avg_initial_fas": 9.3,
      "avg_final_fas": 2.1,
      "avg_rounds": 3.2,
      "avg_noise_ratio": 0.035,
      "avg_systematic_ratio": 0.021
    }
  },
  "overall": {
    "noise_ratio": 0.035,
    "systematic_ratio": 0.021
  }
}
```

解读：3.5% 的问题是随机噪声，2.1% 是系统性错误。

## 模块结构

| 文件 | 职责 |
|------|------|
| `fas.py` | 最小反馈弧集算法（贪心近似） |
| `conflict_detector.py` | 约束提取 → FAS → 问题溯源 |
| `vlm_requester.py` | 调 VLM API 重问冲突题 |
| `resolver.py` | 迭代主循环 + 收敛判定 |
| `report.py` | JSON 报告 + 终端汇总 |

所有模块只 import 已有代码（`API-test/`、`reconstruct/`），不修改任何已有文件。

## 成本估算

| Split | 每场景总题数 | 每轮重问题数 | 预计轮数 | 额外调用/原始 |
|-------|------------|------------|---------|-------------|
| n06 | ~170 | ~8 | 3 | ~14% |
| n07 | ~315 | ~25 | 3-4 | ~25% |
| n10 | ~1300 | ~160 | 4-5 | ~58% |

n06-n07 的消解成本很低，性价比最高。
