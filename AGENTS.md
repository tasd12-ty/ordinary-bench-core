# AGENTS.md

本文件为 Codex (Codex.ai/code) 在本仓库中工作时提供指引。

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
python run_eval.py --job jobs/mock_smoke.toml  # 无网络 smoke test
python run_eval.py --job jobs/example.toml     # 编辑 Job TOML 后运行正式评测
```

当前评测统一通过 `run_eval.py --job <toml>` 驱动，配置入口是 `VLM-test/API-test/job_spec.py`。

Job TOML 主要分 6 段：
- `[provider]` — provider adapter、模型名、base_url、api_key
- `[provider.options]` — `temperature`、`max_tokens`、`max_retries`、`retry_base_delay`、`timeout`、`max_concurrency` 等
- `[input]` — `questions_dir`、`question_layout`（`v1`/`v2`/`auto`）、`question_grouping`（`mixed`/`by_type`）、`question_types`
- `[images]` — `mode`（`single` / `multi_view` / `wrong_single` / `none`）及对应图片根目录
- `[selection]` — `scene`、`split`、`max_scenes`、train/test manifest 过滤
- `[prompt]` / `[output]` — ReAct 纠正阈值、prompt 落盘、结果目录等

环境变量仍可在 Job TOML 中通过 `env:VAR_NAME` 或 `${VAR_NAME}` 引用；常见变量包括：
- `VLM_API_KEY`
- `VLM_BASE_URL`
- `VLM_MODEL`

### 实验 — 子集消融 (`experiments/subset_ablation/`)
该实验检验：同一个 QRR 问题在全图与 4 物体子集图上的表现是否变化。主线只评 QRR，并显式区分：
- `answerable=true`：子集图中相关物体都存在，VLM 应回答 `<` / `~=` / `>`
- `answerable=false`：子集图缺少相关物体，VLM 应回答 `N/A`

如果直接使用仓库中 `experiments/subset_ablation/output/` 下的现成产物，可从问题分配开始：

```bash
cd experiments/subset_ablation

python3 assign_subset_questions.py \
    --manifest output/manifest.json \
    --master-dir output/master_questions \
    --output-dir output

export VLM_BASE_URL="https://openrouter.ai/api/v1"
export VLM_API_KEY="your-key"
export VLM_MODEL="openai/gpt-4o"

uv run python run_subset_eval_multiview.py \
    --questions-dir output/questions/qrr \
    --images-dir output/images \
    --output-dir output/results/gpt4o_multiview \
    --mode multi_view \
    --concurrency 4

uv run python aggregate_to_parent.py \
    --results-dir output/results/gpt4o_multiview/scenes \
    --master-dir output/master_questions \
    --scenes-dir ../../data-gen/output/scenes \
    --output-dir output/aggregated/gpt4o_multiview \
    --model gpt-4o
```

如果从零开始生成子集数据，推荐顺序为：

```bash
cd experiments/subset_ablation

# 1. 枚举父场景的所有 C(N,4) 子集，并生成 subset scene JSON
python3 enumerate_subsets.py \
    --scenes-dir ../../data-gen/output/scenes \
    --output-dir output

# 2. 渲染单视角 / 多视角子集图（按需执行）
python3 render_subsets.py \
    --manifest output/manifest.json \
    --output-dir output \
    --blender blender \
    --workers 4 \
    --samples 64

python3 render_subsets.py \
    --manifest output/manifest.json \
    --output-dir output \
    --blender blender \
    --workers 4 \
    --samples 64 \
    --multi-view

# 3. 为父场景生成 QRR master bank，并把全量问题分配到各子集
python3 generate_master_questions.py \
    --scenes-dir ../../data-gen/output/scenes \
    --output-dir output

python3 assign_subset_questions.py \
    --manifest output/manifest.json \
    --master-dir output/master_questions \
    --output-dir output
```

评测推荐使用 `run_subset_eval_multiview.py`，其 `--mode` 支持：
- `multi_view` — 向 VLM 传 4 张视角图
- `single_view` — 使用 `images/single_view/`
- `pick_view --view-index N` — 从 `images/multi_view/{scene_id}/view_N.png` 选一张做单图评测

评测后可选流程：
- `aggregate_to_parent.py` — 把子集结果聚合回父场景格式，供重建管线使用
- `analyze_results.py` — 对比 full-image vs subset-image 的准确率、自一致性等统计

## 测试

无正式测试框架。验证方式：
- 生成脚本使用 `--preset test` 标志进行快速试运行
- 评测脚本使用 `python run_eval.py --job jobs/mock_smoke.toml` 进行无网络 smoke test
- Infinigen 适配链路使用 `python3 data-gen-infinigen/validate.py` 做端到端校验
- 手动检查生成的 JSON/图像产物

## 架构

### DSL 层 (`VLM-test/dsl/`)
核心空间推理逻辑，独立于 VLM 评估：
- **`predicates.py`** — 定义 `QRRConstraint`、`TRRConstraint`、`FDRConstraint` 数据类、`MetricType` 枚举（DIST_3D、DIST_2D、DEPTH_GAP、SIZE_RATIO），度量计算函数注册在 `METRIC_FUNCTIONS` 字典中；`extract_all_qrr()` / `extract_all_qrr_shared_anchor()` 提取 QRR 真值；`extract_all_trr()` 提取 TRR 真值；`compute_fdr()` / `extract_all_fdr()` 计算并提取 FDR 真值
- **`comparators.py`** — `Comparator` 枚举（`<`、`~=`、`>`），基于容差参数 `tau`（默认 0.10）的比较代数

添加新度量：在 `MetricType` 中添加枚举值，并在 `METRIC_FUNCTIONS` 中注册计算函数。

### 问题生成 (`VLM-test/`)
- **`question_bank.py`** — `enumerate_qrr()` 支持 `include_disjoint` 和 `include_shared_anchor` 标志控制变体生成；`enumerate_trr()` 生成钟面方向问题；`enumerate_fdr()` 生成全距离排序问题；`make_batches()` 按可配置大小分批
- **`extraction.py`** — 将场景 JSON 转换为 DSL 兼容的对象字典并提取真值
- **`generate_questions.py`** — 旧版平铺输出（已废弃，保留兼容）
- **`generate_questions_v2.py`** — 按题型分目录存储的问题生成脚本（推荐）

### VLM 客户端 (`VLM-test/API-test/`)
- **`run_eval.py`** — 当前唯一正式 CLI 入口，读取 Job TOML 并执行评测
- **`job_spec.py`** — Job TOML 加载与校验，支持 `env:VAR` / `${VAR}` 环境变量展开
- **`eval_engine.py`** — 唯一编排层：场景发现、问题加载、batch、provider 调用、ReAct 纠正、评分、结果落盘
- **`question_loader.py`** — 兼容 v1/v2/auto 的问题布局加载，支持 `mixed` / `by_type` 分组
- **`image_resolver.py`** — 统一图片模式：`single` / `multi_view` / `wrong_single` / `none`
- **`providers/`** — 协议适配层；当前包含 `openai_chat`、`gemini_native`、`mock_static`
- **`vlm_client.py`** — OpenAI 兼容底层客户端，支持指数退避重试、base64 图像编码、多图消息；主要由 `providers/openai_chat.py` 调用
- **`response_parser.py`** — 从 VLM 输出中健壮地提取 JSON（去除思考标签、修复截断 JSON、合并拆分数组）
- **`scoring.py`** — QRR 比较符匹配（按 disjoint/shared_anchor 变体分别统计）、TRR 小时/象限/相邻评分、FDR 四粒度评分（exact/kendall τ/pairwise/top-1）、逐场景和跨数据集聚合；在 ablation 模式下还统计 refusal / hallucination 指标
- **`result_store.py`** — 统一输出 `raw/`、`scenes/`、`summary.json`

### 重建管线 (`VLM-test/reconstruct/`)
- **`constraints.py`** — QRREntry、TRREntry、FDREntry 数据类；`build_distance_poset()`（QRR 偏序 DAG）、`build_angular_sectors()`（TRR 角度扇区）、`decompose_fdr_to_qrr()`（FDR→QRR 分解）
- **`solver.py`** — 2D 场景求解器；规范固定 + log 域 QRR 损失 + 扇区容差 TRR 损失 + 多重启 L-BFGS-B
- **`preparation.py`** — `prepare_reconstruction_input_from_scoring()`；把评分输出整理为可审计的逐场景 prepared package
- **`pipeline.py`** — 四个入口：`reconstruct()`、`prepare_reconstruction_input_from_scoring()`、`reconstruct_from_scoring()`、`reconstruct_from_prepared()`；支持 `constraint_mode`
- **`evaluate.py`** — CSR、solver 对齐的 `csr_qrr_aligned`、NRL、显著性检验、Kendall τ、NRMS、K_geom（几何模态数）

### 冲突消解 (`VLM-test/conflict_resolution/`)
- **`run_conflict_resolution.py`** — 基于评测 Job 配置运行冲突检测与迭代消解
- **`conflict_detector.py`** — 基于 prepared reconstruction input 检测 QRR/TRR/FDR 冲突
- **`resolver.py` / `voting_resolver.py`** — 迭代 re-ask 或投票式冲突消解
- **`report.py`** — 保存逐场景结果与汇总报告

### 实验 (`experiments/`)
- **`subset_ablation/`** — 子集消融实验；核心脚本包括 `enumerate_subsets.py`、`render_subsets.py`、`generate_master_questions.py`、`assign_subset_questions.py`、`run_subset_eval_multiview.py`、`aggregate_to_parent.py`、`analyze_results.py`
- **`constraint_analysis/`** — 约束可视化与分析辅助脚本

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

- 主线产物以 JSON 为主；通常按阶段落在 `output/` 下：
  `data-gen` 生成 `images/`、`scenes/`、`splits/`，`VLM-test` 生成 `questions/` 与 `extraction_tasks/`，`API-test` 生成 `results/<run_name>/{raw,scenes,summary.json}`
- 三种问题类型：QRR（disjoint + shared_anchor 两个变体）、TRR、FDR
- v2 脚本按题型分目录存储：`output/questions/{qrr,trr,fdr}/{scene_id}.json`
- `question_loader.py` 支持 `question_layout = auto`：优先读 v2，缺失时回退到 v1
- 评测差异优先通过 Job TOML 表达，而不是继续新增独立 runner 脚本
- `tau` 容差参数控制近似相等（`~=`）的比较代数
- Blender 脚本通过 `pipeline.py` 作为子进程运行，不可作为模块导入
- `data-gen/pipeline.py` 中包含 WSL/Windows Blender 路径的跨平台处理
- 评分标准详见 `VLM-test/docs/scoring_criteria.md`
