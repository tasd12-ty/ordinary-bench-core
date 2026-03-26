# API-test

统一的 VLM 评测入口已经收敛为：

```bash
python run_eval.py --job jobs/openai_single_v2.toml
```

旧脚本如 `run_batch_v2.py`、`run_batch_gemini.py`、`run_multi_view.py` 等仍然存在，但现在只是兼容包装层；正式实现都走同一个 `eval_engine.py`。

## 目录

- `run_eval.py`
  统一 CLI 入口，读取 TOML job 后执行评测
- `job_spec.py`
  任务配置加载与校验
- `eval_engine.py`
  唯一编排层：场景发现、问题加载、batch、ReAct、评分、落盘
- `question_loader.py`
  兼容 v1/v2/auto 的问题加载
- `image_resolver.py`
  统一图片模式：`single` / `multi_view` / `wrong_single` / `none`
- `providers/`
  协议适配层，目前包含：
  - `openai_chat`
  - `gemini_native`
  - `mock_static`
- `result_store.py`
  统一 raw / scenes / summary 输出
- `jobs/`
  推荐的任务配置模板

## Job 结构

一个 job TOML 分 6 段：

```toml
[provider]
adapter = "openai_chat"
model = "openai/gpt-4o"
base_url = "https://openrouter.ai/api/v1"
api_key = "env:OPENROUTER_API_KEY"

[provider.options]
temperature = 0.0
max_tokens = 65536
max_retries = 5
retry_base_delay = 2.0
timeout = 120
max_concurrency = 4

[input]
questions_dir = "../../output/questions"
question_layout = "auto"
question_grouping = "by_type"
question_types = ["qrr", "trr", "fdr"]
batch_size = 20

[images]
mode = "single"
single_view_root = "../../../data-gen/output/images/single_view"

[selection]
split = ""
scene = ""
test_only = false
train_only = false
max_scenes = 10

[prompt]
react_max_rounds = 2
missing_threshold = 0.2
react_chunk_size = 50
save_prompt = true

[output]
results_dir = "../../output/results"
run_name = "example_run"
```

## 变化轴

现在所有差异都收敛到 job 参数，而不是新脚本：

- provider 协议：
  `provider.adapter`
- 图像输入方式：
  `images.mode`
- 单视角 / 多视角 / 错图 / 无图：
  `images.mode + images.n_views`
- v1 / v2 问题布局：
  `input.question_layout`
- mixed / by_type 提问：
  `input.question_grouping`
- 题型子集：
  `input.question_types`

## 兼容说明

- `question_layout = "auto"`：优先读 v2，找不到时回退到 v1
- 旧 runner 仍可用，但建议只用 `run_eval.py`
- 输出目录仍保持：
  `output/results/<run_name>/{raw,scenes,summary.json}`

## 本地 smoke test

不依赖网络的最小验证：

```bash
python run_eval.py --job jobs/mock_smoke.toml
```
