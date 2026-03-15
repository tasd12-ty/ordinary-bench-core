# CLAUDE.md

本文件为 Claude Code (claude.ai/code) 在本仓库中工作时提供指引。

## 项目概述

ORDINARY-BENCH 是一个用于评估视觉语言模型 (VLM) 在序数空间关系理解上表现的基准测试，包含三个阶段：
1. **数据生成** (`data-gen/`)：通过 Blender 子进程渲染 3D 场景
2. **问题生成** (`VLM-test/`)：从场景元数据中枚举 QRR 和 TRR 空间推理问题
3. **VLM 评估** (`VLM-test/API-test/`)：通过 OpenAI 兼容 API 测试 VLM 并评分

`data-gen-dynamic/` 是一个并行管线，用于生成带有运动模型的时序/视频场景。

## 环境搭建

```bash
uv sync           # 安装依赖（推荐）
pip install -e .  # 备选方案
```

目标 Python 版本为 3.11（`.python-version`），最低支持 3.9（`pyproject.toml`）。

## 各阶段运行方式

### 阶段一 — 数据生成
```bash
cd data-gen
python generate.py --preset test                     # 快速验证
python generate.py --config config.toml --workers 4  # 完整生成
```
需要安装 Blender，路径在 `config.toml` 的 `[blender].executable` 中配置。

### 阶段二 — 问题生成
```bash
cd VLM-test
python generate_questions.py --data ../data-gen/output --split n04
python generate_questions.py --data ../data-gen/output --counts  # 仅统计数量
```

### 阶段三 — VLM 评估
```bash
cd VLM-test/API-test
VLM_MODEL="openai/gpt-4o" python run_batch.py
VLM_MODEL="qwen/qwen-2.5-vl-72b-instruct" python run_batch.py --split n04
```

所有 VLM 设置通过环境变量控制（详见 `VLM-test/API-test/config.py`）：
- `VLM_BASE_URL` — API 端点（默认：OpenRouter）
- `VLM_API_KEY` — API 密钥
- `VLM_MODEL` — 模型标识符
- `VLM_CONCURRENCY` — 并行工作线程数（默认：4）
- `VLM_TIMEOUT`、`VLM_MAX_RETRIES`、`VLM_RETRY_DELAY` — 重试设置

## 测试

无正式测试框架。验证方式：
- 生成脚本使用 `--preset test` 标志进行快速试运行
- 手动检查生成的 JSON/图像产物

## 架构

### DSL 层 (`VLM-test/dsl/`)
核心空间推理逻辑，独立于 VLM 评估：
- **`predicates.py`** — 定义 `QRRConstraint` 和 `TRRConstraint` 数据类、`MetricType` 枚举（DIST_3D、DIST_2D、DEPTH_GAP、SIZE_RATIO），度量计算函数注册在 `METRIC_FUNCTIONS` 字典中
- **`comparators.py`** — `Comparator` 枚举（`<`、`~=`、`>`），基于容差参数 `tau`（默认 0.10）的比较代数

添加新度量：在 `MetricType` 中添加枚举值，并在 `METRIC_FUNCTIONS` 中注册计算函数。

### 问题生成 (`VLM-test/`)
- **`question_bank.py`** — `enumerate_qrr()` 生成不相交对的比较；`enumerate_trr()` 生成钟面方向问题；`make_batches()` 按可配置大小分批
- **`extraction.py`** — 将场景 JSON 转换为 DSL 兼容的对象字典并提取真值

### VLM 客户端 (`VLM-test/API-test/`)
- **`vlm_client.py`** — OpenAI 兼容客户端，支持指数退避重试、base64 图像编码、多视角消息
- **`response_parser.py`** — 从 VLM 输出中健壮地提取 JSON（去除思考标签、修复截断 JSON、合并拆分数组）
- **`scoring.py`** — QRR 比较符匹配、TRR 小时/象限/相邻评分、逐场景和跨数据集聚合
- **`run_batch.py`** — 编排器，包含针对不完整 VLM 响应的 ReAct 风格纠正循环

### 动态生成 (`data-gen-dynamic/`)
- **`motion/models.py`** — 6 种运动模型类型（静止、线性、圆周、加速、航点、反弹），采用注册表模式（`MOTION_REGISTRY`）
- **`extraction/temporal.py`** — 帧到静态场景的转换、跨帧 QRR/TRR 事件检测

### 配置分层（data-gen）
优先级：默认值 → TOML 文件 → 预设 → 命令行参数。详见 `data-gen/generate.py`。

## 关键约定

- 场景和问题数据全程使用 JSON；输出目录遵循 `output/{images,scenes,splits,questions,results}/` 结构
- QRR 问题比较两组对象对之间的成对度量；TRR 问题使用钟面小时数（1–12）表示三个对象间的方向关系
- `tau` 容差参数控制近似相等（`~=`）的比较代数
- Blender 脚本通过 `pipeline.py` 作为子进程运行，不可作为模块导入
- `data-gen/pipeline.py` 中包含 WSL/Windows Blender 路径的跨平台处理
