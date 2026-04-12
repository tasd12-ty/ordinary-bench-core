# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

ORDINARY-BENCH 是一个用于评估视觉语言模型 (VLM) 在序数空间关系理解上表现的基准测试，包含三个阶段：
1. **数据生成**：通过多种后端渲染 3D 场景
2. **问题生成** (`VLM-test/`)：从场景元数据中枚举 QRR、TRR、FDR 三类空间推理问题
3. **VLM 评估** (`VLM-test/API-test/`)：通过 OpenAI 兼容 API 测试 VLM 并评分

数据生成后端：
- `data-gen/` — Blender CLEVR 风格场景（原始后端）
- `data-gen-dynamic/` — 带运动模型的时序/视频场景
- `data-gen-infinigen/` — Infinigen-Indoors 写实场景

问题类型：
- **QRR**（Quantitative Relation Reasoning）— disjoint（不相交对）和 shared_anchor（共享锚点）两个变体
- **TRR**（Ternary Relation Reasoning）— 钟面方向推理
- **FDR**（Full Distance Ranking）— 全距离排序
- **VRF**（Verification）— 固定题量的复合空间关系验证（独立模块 `VRF-test/`）

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
python generate_questions_v2.py --data ../data-gen/output --split n04  # 推荐
python generate_questions_v2.py --counts                                # 显示问题数量表
```

问题生成是完全确定性的：相同 scene JSON + tau → 相同问题（相同顺序、相同 QID）。

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

### 阶段四 — 结果分析
```bash
cd VLM-test

# 一键全套分析（精度表 + 重建 + 一致性 + 可视化）
python analysis/run_analysis.py

# 单独运行
python analysis/aggregate.py                    # 精度汇总表（Markdown）
python analysis/generate_insight_excel.py       # Excel 报告
python analysis/reconstruct_scenes.py -r output/results/gpt-4o --belief  # 场景重建
python analysis/export_and_reconstruct.py       # Excel + 重建 + SVG
```

## 快速测试（无需数据生成）

仓库包含 140 个 test 场景的完整数据，clone 后即可评测：

```bash
# 直接使用 test-data 跑评测
cd VLM-test/API-test
python run_eval.py --job ../../datasets/jobs/hf_eval_example.toml

# 或从 HuggingFace 下载并转换
python datasets/hf_to_local.py --repo TYTSTQ/ordinary-bench --split test --output ./hf_data
```

数据位置：`datasets/test-data/`（scenes、images、questions 齐全，含 VRF 问题）。

### VRF 验证评测
```bash
cd VRF-test
python generate_questions.py --data ../datasets/test-data            # 生成 VRF 题目
python run_eval.py --job jobs/smoke.toml                              # 烟雾测试（不需要 API）
```

## 测试

无正式测试框架。验证方式：
- 生成脚本使用 `--preset test` 标志进行快速试运行
- `python run_eval.py --job jobs/mock_smoke.toml` 验证评测管线完整性（不需要 API）
- 手动检查生成的 JSON/图像产物

## 架构

### DSL 层 (`VLM-test/dsl/`)
核心空间推理逻辑，独立于 VLM 评估：
- **`predicates.py`** — `QRRConstraint`、`TRRConstraint`、`FDRConstraint` 数据类、`MetricType` 枚举，度量计算函数注册在 `METRIC_FUNCTIONS` 字典中
- **`comparators.py`** — `Comparator` 枚举（`<`、`~=`、`>`），基于容差参数 `tau`（默认 0.10）的比较代数

添加新度量：在 `MetricType` 中添加枚举值，并在 `METRIC_FUNCTIONS` 中注册计算函数。

### 问题生成 (`VLM-test/`)
- **`question_bank.py`** — `enumerate_qrr()` 支持 `include_disjoint` 和 `include_shared_anchor` 标志；`enumerate_trr()` 钟面方向；`enumerate_fdr()` 全距离排序；`make_batches()` 按可配置大小分批
- **`extraction.py`** — 将场景 JSON 转换为 DSL 兼容的对象字典并提取真值
- **`generate_questions_v2.py`** — 按题型分目录存储的问题生成脚本（推荐）

### VLM 评测引擎 (`VLM-test/API-test/`)
- **`run_eval.py`** — 统一 CLI 入口，读取 TOML job 后执行评测
- **`eval_engine.py`** — 编排层：场景发现 → 问题加载 → batch → ReAct 重试 → 评分 → 落盘
- **`job_spec.py`** — TOML job 配置加载与校验
- **`question_loader.py`** — 兼容 v1/v2/auto 的问题加载；支持 `skip_unanswerable` 过滤和 `mixed` grouping
- **`image_resolver.py`** — 统一图片模式：single / multi_view / wrong_single / none
- **`vlm_client.py`** — OpenAI 兼容客户端，支持指数退避重试、base64 图像编码
- **`response_parser.py`** — 从 VLM 输出中健壮地提取 JSON（去除思考标签、修复截断 JSON、合并拆分数组）
- **`scoring.py`** — QRR 比较符匹配（按 disjoint/shared_anchor 变体分别统计）、TRR 小时/象限/相邻评分、FDR 四粒度评分（exact/kendall τ/pairwise/top-1）；ablation 模式支持 answerable/refusal/hallucination 评分
- **`providers/`** — 协议适配层（openai_chat、gemini_native、mock_static）
- **`jobs/`** — TOML job 配置模板

### 分析管线 (`VLM-test/analysis/`)
- **`run_analysis.py`** — 一键编排：精度汇总 → 批量重建 → 一致性分析 → 可视化
- **`aggregate.py`** — 跨模型精度表（Markdown）
- **`generate_insight_excel.py`** — 多 sheet Excel 报告（模型排行、per-split、per-scene、重建）
- **`reconstruct_scenes.py`** — 批量 belief 重建（CSR / NRMS / Kendall τ）
- **`export_and_reconstruct.py`** — 合并管线：Excel + 重建 + SVG 可视化
- **`consistency.py`** — 传递性 & 互反性检验
- **`fdr_qrr_conflict.py`** — FDR↔QRR 矛盾检测

### 重建管线 (`VLM-test/reconstruct/`)
- **`constraints.py`** — QRREntry、TRREntry、FDREntry 数据类；`build_distance_poset()`（QRR 偏序 DAG）、`decompose_fdr_to_qrr()`（FDR→QRR 分解）
- **`solver.py`** — 基于梯度下降的 2D 位置优化器，多重启 + hinge loss
- **`pipeline.py`** — `reconstruct()` 和 `reconstruct_from_scoring()` 端到端入口
- **`evaluate.py`** — CSR（约束满足率）、Kendall τ、NRMS、K_geom（几何模态数）

### Subset Ablation (`experiments/subset_ablation/`)
测试 VLM 是否受图像中物体数量影响。对 N 物体场景枚举所有 C(N,4) 四物体子集，重新渲染，用全量 master QRR 题库评测（含 N/A 不可答题）。
- **`assign_subset_questions.py`** — 核心：`classify_questions()` 标记 answerable/refusal，`build_subset_question_file()` 构建问题文件
- **`aggregate_to_parent.py`** — 子集结果多数投票聚合为父场景格式，可接入重建管线
- **`analyze_results.py`** — 全图 vs 子集精度对比 + 自一致性分析

### VRF 验证评测 (`VRF-test/`)
独立的固定题量空间关系验证模块，每场景 K=20 道复合 TRUE/FALSE 题。
- **`vrf_question_bank.py`** — 核心：fact 池构建（QRR/TRR/FDR）、复合题组装、FALSE 篡改策略
- **`generate_questions.py`** — CLI 脚本，场景 JSON → VRF 题目 JSON
- **`run_eval.py`** — 评测入口，读取 TOML job 配置
- **`eval_engine.py`** — 编排层：场景发现 → 问题加载 → API 调用 → 评分
- **`vrf_scoring.py`** — VRF 评分（bool 匹配 + TRUE/FALSE 分别统计）
- **`vrf_prompts.py`** — System/User prompt 模板

复用 `VLM-test/dsl/` 的 GT 计算函数和 `VLM-test/API-test/providers/` 的 API 适配器（只 import，不修改）。

### HuggingFace 数据集工具 (`datasets/`)
- **`build_dataset.py`** / **`build_dataset_multiview.py`** / **`build_dataset_subset.py`** — 构建 Parquet 数据集并上传 HuggingFace
- **`hf_to_local.py`** — HuggingFace → 本地格式转换（并行 I/O，逐场景日志）
- **`test-data/`** — 140 个 test 场景的完整数据（scenes + images + questions），clone 后直接可用
- **`test-data/subset_ablation/generate_questions.py`** — subset 问题本地生成脚本

### 配置分层（data-gen）
优先级：默认值 → TOML 文件 → 预设 → 命令行参数。详见 `data-gen/generate.py`。

## 关键约定

- 场景和问题数据全程使用 JSON；输出目录遵循 `output/{images,scenes,splits,questions,results}/` 结构
- 四种问题类型：QRR（disjoint + shared_anchor 两个变体）、TRR、FDR、VRF（固定题量验证）
- v2 脚本按题型分目录存储：`output/questions/{qrr,trr,fdr}/{scene_id}.json`
- `tau` 容差参数（默认 0.10）控制近似相等（`~=`）的比较代数
- 问题生成完全确定性：`sorted(objects.keys())` + `combinations()` 迭代，无随机性
- Blender 脚本通过 `pipeline.py` 作为子进程运行，不可作为模块导入
- 评分标准详见 `VLM-test/docs/scoring_criteria.md`
- AI Agent 完整操作指南：`docs/agent-guide.md`
