# Adaptive Sort — 全局距离对快速排序 VLM 评测

## 概述

对场景中所有 C(N,2) 个物体距离对进行**全局排序**，使用 VLM 作为比较器，采用快速排序策略。

**核心思路**：
1. 用 GT 真值选取全排序中位数作为第 0 层 pivot（保证首次划分均匀）
2. VLM 回答每个距离对与 pivot 的大小关系（`<` / `~=` / `>`）
3. 三路划分后递归，后续层 pivot 随机选取
4. 输出无矛盾的全序，从中可推导任意 QRR 答案

**优势**：
- 比较次数从穷举 QRR 的 O(N⁴) 降到 O(N² log N)
- N=10 时节省 ~80%（197 vs 990 次比较）
- 输出是全序，**不可能产生环/矛盾**
- 同层划分任务并发请求

## 快速开始

```bash
cd adaptive-sort

# 1. Mock 烟雾测试（无需 API，验证算法正确性）
python run_eval.py --job jobs/smoke.toml

# 2. 使用 OpenRouter 测试（需设置 API key）
export OPENROUTER_API_KEY=your-key
python run_eval.py --job jobs/qwen_test.toml

# 3. 使用 OpenAI 测试
export OPENAI_API_KEY=your-key
python run_eval.py --job jobs/example.toml
```

## Job 配置详解

所有评测通过 TOML 文件配置，包含 6 段：

### `[provider]` — API 提供者

| 字段 | 必填 | 说明 |
|---|---|---|
| `adapter` | 是 | 适配器类型：`mock_oracle` / `openai_chat` / `anthropic` |
| `model` | 是 | 模型标识，如 `gpt-4o`、`qwen/qwen3.5-9b`、`claude-sonnet-4-6` |
| `base_url` | 否 | API 端点，OpenRouter 用 `https://openrouter.ai/api/v1` |
| `api_key` | 否 | API 密钥，支持 `env:VAR_NAME` 引用环境变量 |

### `[provider.options]` — 模型参数

| 字段 | 默认值 | 说明 |
|---|---|---|
| `temperature` | `0.0` | 采样温度，推荐 0.0（确定性输出） |
| `max_tokens` | `1024` | 最大输出 token 数 |

### `[input]` — 输入数据

| 字段 | 默认值 | 说明 |
|---|---|---|
| `scenes_dir` | （必填） | 场景 JSON 文件目录 |
| `tau` | `0.10` | 距离比较容差，控制 `~=` 判定阈值 |
| `allow_approx` | `true` | 是否允许 `~=`。设为 `false` 时，VLM 输出 `~=` / 约等于会被视为非法并重试；GT/mock 遇到近似距离会报告失败 |

### `[images]` — 图像输入

| 字段 | 默认值 | 说明 |
|---|---|---|
| `mode` | `single` | 图像模式：`single`（单图）/ `multi_view`（多视角）/ `none`（纯文本） |
| `single_view_root` | | 单图目录，图片命名为 `{scene_id}.png` |
| `multi_view_root` | | 多视角目录 |
| `n_views` | `1` | 多视角图片数量 |

### `[selection]` — 场景筛选

| 字段 | 默认值 | 说明 |
|---|---|---|
| `split` | | 场景前缀过滤：`n04`（4物体）/ `n10`（10物体）/ `n15`（15物体） |
| `scene` | | 指定单个场景 ID，如 `n10_000080` |
| `max_scenes` | 无限制 | 最多处理的场景数 |

### `[sorting]` — 排序算法参数

| 字段 | 默认值 | 说明 |
|---|---|---|
| `pivot_strategy` | `middle` | Pivot 选择策略（第 0 层始终用 GT 中位数） |
| `max_retries` | `3` | VLM 返回不可解析时的重试次数 |
| `retry_base_delay` | `2.0` | 重试基础延迟（秒），指数退避 |
| `max_concurrency` | `4` | 同层划分任务的最大并发数 |

### `[output]` — 输出配置

| 字段 | 默认值 | 说明 |
|---|---|---|
| `results_dir` | （必填） | 结果输出根目录 |
| `run_name` | TOML 文件名 | 本次运行的名称，结果保存在 `{results_dir}/{run_name}/` |

## 常用模型配置示例

### OpenRouter — Qwen 3.5 9B（带图，付费）

```toml
[provider]
adapter = "openai_chat"
model = "qwen/qwen3.5-9b"
base_url = "https://openrouter.ai/api/v1"
api_key = "env:OPENROUTER_API_KEY"

[images]
mode = "single"
single_view_root = "../../datasets/test-data/images/single_view"
```

> **注意**：Qwen 等推理模型默认启用思考模式，引擎会自动通过 `reasoning.effort=none` 关闭，确保 VLM 直接输出 JSON。

### OpenRouter — 免费模型（纯文本）

```toml
[provider]
adapter = "openai_chat"
model = "google/gemma-4-31b-it:free"
base_url = "https://openrouter.ai/api/v1"
api_key = "env:OPENROUTER_API_KEY"

[images]
mode = "none"
```

> 免费模型 rate limit 严格，建议 `max_concurrency = 1`，`retry_base_delay = 3.0`。

### OpenAI — GPT-4o

```toml
[provider]
adapter = "openai_chat"
model = "gpt-4o"
base_url = "https://api.openai.com/v1"
api_key = "env:OPENAI_API_KEY"
```

### Anthropic — Claude（直接 SDK）

```toml
[provider]
adapter = "anthropic"
model = "claude-sonnet-4-6"
api_key = "env:ANTHROPIC_API_KEY"
```

### Mock Oracle（无需 API）

```toml
[provider]
adapter = "mock_oracle"
model = "mock-gt-oracle"

[images]
mode = "none"
```

## 数据配置

仓库自带 test 场景数据（`datasets/test-data/`），按物体数量分 split：

| split | 物体数 | 距离对数 C(N,2) | 穷举 QRR 比较数 | 场景数 |
|---|---|---|---|---|
| `n04` | 4 | 6 | 15 | 20 |
| `n05` | 5 | 10 | 40 | 20 |
| `n06`~`n09` | 6~9 | 15~36 | 90~504 | 各 10 |
| `n10` | 10 | 45 | 990 | 10 |
| `n11`~`n15` | 11~15 | 55~105 | 1485~4095 | 各 10 |

配置示例：
```toml
[selection]
split = "n10"       # 所有 10 物体场景
max_scenes = 3      # 只跑前 3 个

# 或指定单个场景
scene = "n10_000080"
```

## 输出格式

结果保存在 `{results_dir}/{run_name}/`：

```
output/results/{run_name}/
├── scenes/
│   └── {scene_id}.json   # 单场景详细结果
└── summary.json           # 跨场景汇总
```

单场景 JSON 结构：
```json
{
  "scene_id": "n10_000080",
  "model": "qwen/qwen3.5-9b",
  "n_objects": 10,
  "n_pairs": 45,
  "gt_ranking": [["obj_1","obj_2"], ["obj_0","obj_1"], ...],
  "vlm_result": {
    "ranking": [["obj_1","obj_2"], ...],
    "tie_groups": [[["obj_1","obj_2"]], ...],
    "rounds": [
      {
        "level": 0,
        "pivot": ["obj_2", "obj_5"],
        "n_comparisons": 44,
        "prompt_tokens": 1200,
        "completion_tokens": 300,
        "partition_lt": [...],
        "partition_eq": [...],
        "partition_gt": [...]
      }
    ],
    "total_comparisons": 197,
    "total_api_calls": 10
  },
  "scores": {
    "pairwise_accuracy": 0.8192,
    "comparison_savings": 0.801
  }
}
```

## 评分指标

| 指标 | 含义 |
|---|---|
| `pairwise_accuracy` | VLM 全排序中，任取两个距离对，它们的相对顺序与 GT 一致的比例。1.0=完美，0.5=随机 |
| `comparison_savings` | `1 - 实际比较次数/穷举QRR比较次数`。值越大说明快排节省越多 |
| `exact_match` | VLM 排序是否与 GT 完全一致（含 tie group 结构匹配） |
| `total_api_calls` | 总 API 调用次数 |
| `prompt_tokens` / `completion_tokens` | Token 消耗统计 |

## 注意事项

1. **API Key 安全**：TOML 中使用 `env:VAR_NAME` 引用环境变量，不要写明文密钥
2. **推理模型**：Qwen、DeepSeek 等默认启用思考模式，引擎默认通过 OpenRouter `reasoning.effort=none` 关闭；可用 `[provider.options.extra_body]` 显式覆盖
3. **Rate Limit**：免费模型限制严格，建议降低 `max_concurrency` 和增大 `retry_base_delay`
4. **图像路径**：`single_view_root` 中图片必须命名为 `{scene_id}.png`
5. **路径解析**：TOML 中的相对路径基于 TOML 文件所在目录解析
