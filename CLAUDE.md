# CLAUDE.md

本文件为 Claude Code (claude.ai/code) 在本仓库中工作时提供指引。

## 项目概述

ORDINARY-BENCH 是一个用于评估视觉语言模型 (VLM) 在序数空间关系理解上表现的基准测试，包含三个阶段：
1. **数据生成**：通过多种后端渲染 3D 场景
2. **问题生成** (`VLM-test/`)：从场景元数据中枚举 QRR、TRR、FDR 三类空间推理问题
3. **VLM 评估** (`VLM-test/API-test/`)：通过 OpenAI 兼容 API 测试 VLM 并评分

数据生成后端：
- `data-gen/` — Blender CLEVR 风格场景（原始后端）
- `data-gen-dynamic/` — 带运动模型的时序/视频场景
- `data-gen-infinigen/` — Infinigen-Indoors 写实场景（新增）

问题类型：
- **QRR**（Quantitative Relation Reasoning）— disjoint（不相交对）和 shared_anchor（共享锚点）两个变体
- **TRR**（Ternary Relation Reasoning）— 钟面方向推理
- **FDR**（Full Distance Ranking）— 全距离排序

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
# v1（混合存储）
python generate_questions.py --data ../data-gen/output --split n04
# v2（分题型目录存储，推荐）
python generate_questions_v2.py --data ../data-gen/output --split n04
python generate_questions_v2.py --counts
```

### 阶段三 — VLM 评估
```bash
cd VLM-test/API-test

# 烟雾测试（不需要 API）
python run_eval.py --job jobs/mock_smoke.toml

# 正式评测（通过 TOML job 配置）
python run_eval.py --job jobs/example.toml
```

所有评测配置通过 TOML job 文件控制（详见 `VLM-test/API-test/README.md`），包含 6 段配置：
- `[provider]` — adapter、model、base_url、api_key（支持 `env:VAR` 引用环境变量）
- `[provider.options]` — temperature、max_tokens、max_concurrency、重试设置
- `[input]` — questions_dir、question_layout（v1/v2/auto）、question_types、batch_size
- `[images]` — mode（single/multi_view/none）、图像根目录
- `[selection]` — split、scene、max_scenes 筛选
- `[output]` — results_dir、run_name

## 测试

无正式测试框架。验证方式：
- 生成脚本使用 `--preset test` 标志进行快速试运行
- 手动检查生成的 JSON/图像产物

## 架构

### DSL 层 (`VLM-test/dsl/`)
核心空间推理逻辑，独立于 VLM 评估：
- **`predicates.py`** — 定义 `QRRConstraint`、`TRRConstraint`、`FDRConstraint` 数据类、`MetricType` 枚举（DIST_3D、DIST_2D、DEPTH_GAP、SIZE_RATIO），度量计算函数注册在 `METRIC_FUNCTIONS` 字典中；`compute_fdr()` 计算全距离排序；`extract_all_fdr()` 提取 FDR 真值；`extract_all_qrr_shared_anchor()` 提取共享锚点 QRR 真值
- **`comparators.py`** — `Comparator` 枚举（`<`、`~=`、`>`），基于容差参数 `tau`（默认 0.10）的比较代数

添加新度量：在 `MetricType` 中添加枚举值，并在 `METRIC_FUNCTIONS` 中注册计算函数。

### 问题生成 (`VLM-test/`)
- **`question_bank.py`** — `enumerate_qrr()` 支持 `include_disjoint` 和 `include_shared_anchor` 标志控制变体生成；`enumerate_trr()` 生成钟面方向问题；`enumerate_fdr()` 生成全距离排序问题；`make_batches()` 按可配置大小分批
- **`extraction.py`** — 将场景 JSON 转换为 DSL 兼容的对象字典并提取真值
- **`generate_questions_v2.py`** — 按题型分目录存储的问题生成脚本（推荐）

### VLM 评测引擎 (`VLM-test/API-test/`)
- **`run_eval.py`** — 统一 CLI 入口，读取 TOML job 后执行评测
- **`eval_engine.py`** — 编排层：场景发现 → 问题加载 → batch → ReAct 重试 → 评分 → 落盘
- **`job_spec.py`** — TOML job 配置加载与校验
- **`question_loader.py`** — 兼容 v1/v2/auto 的问题加载
- **`image_resolver.py`** — 统一图片模式：single / multi_view / wrong_single / none
- **`vlm_client.py`** — OpenAI 兼容客户端，支持指数退避重试、base64 图像编码
- **`response_parser.py`** — 从 VLM 输出中健壮地提取 JSON（去除思考标签、修复截断 JSON、合并拆分数组）
- **`scoring.py`** — QRR 比较符匹配（按 disjoint/shared_anchor 变体分别统计）、TRR 小时/象限/相邻评分、FDR 四粒度评分（exact/kendall τ/pairwise/top-1）、逐场景和跨数据集聚合
- **`providers/`** — 协议适配层（openai_chat、gemini_native、mock_static）
- **`jobs/`** — TOML job 配置模板

### 重建管线 (`VLM-test/reconstruct/`)
- **`constraints.py`** — QRREntry、TRREntry、FDREntry 数据类；`build_distance_poset()`（QRR 偏序 DAG）、`build_angular_sectors()`（TRR 角度扇区）、`decompose_fdr_to_qrr()`（FDR→QRR 分解）
- **`solver.py`** — 基于梯度下降的 2D 位置优化器，多重启 + hinge loss
- **`pipeline.py`** — `reconstruct()` 和 `reconstruct_from_scoring()` 端到端入口
- **`evaluate.py`** — CSR（约束满足率）、Kendall τ、NRMS、K_geom（几何模态数）

### 动态生成 (`data-gen-dynamic/`)
- **`motion/models.py`** — 6 种运动模型类型（静止、线性、圆周、加速、航点、反弹），采用注册表模式（`MOTION_REGISTRY`）
- **`extraction/temporal.py`** — 帧到静态场景的转换、跨帧 QRR/TRR 事件检测

### Infinigen 后端 (`data-gen-infinigen/`)
- **`generate.py`** — Infinigen-Indoors 场景生成编排器
- **`adapter.py`** — 将 Infinigen 元数据转换为 ordinary-bench 场景 JSON（坐标系转换：x_bench=x_world, y_bench=z_world, z_bench=-y_world）
- **`bootstrap_from_datagen.py`** — 从现有 data-gen 场景创建伪 Infinigen 包，用于无 Blender 测试
- 详见 `data-gen-infinigen/README.md`

### 配置分层（data-gen）
优先级：默认值 → TOML 文件 → 预设 → 命令行参数。详见 `data-gen/generate.py`。

## 关键约定

- 场景和问题数据全程使用 JSON；输出目录遵循 `output/{images,scenes,splits,questions,results}/` 结构
- 三种问题类型：QRR（disjoint + shared_anchor 两个变体）、TRR、FDR
- v2 脚本按题型分目录存储：`output/questions/{qrr,trr,fdr}/{scene_id}.json`
- `tau` 容差参数控制近似相等（`~=`）的比较代数
- Blender 脚本通过 `pipeline.py` 作为子进程运行，不可作为模块导入
- `data-gen/pipeline.py` 中包含 WSL/Windows Blender 路径的跨平台处理
- 评分标准详见 `VLM-test/docs/scoring_criteria.md`
