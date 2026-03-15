# NeurIPS 2026 D&B Track — 实验执行计划

更新时间：2026-03-12

---

## 概览

本文档给出投稿所需的全部实验工作，按优先级和依赖关系排列。

**目标**：从当前状态（2 个 Qwen 模型 × ~60 场景）补齐到可投稿状态（6+ 模型 × 700 场景 + 人类基线 + 三条件实验）。

---

## P0：投稿硬性前置（必须完成）

### P0.1 全量 700 场景评测（现有 2 模型）

**预计工作量：** 2-3 天（API 调用时间）

当前仅评测了 ~60 场景。需要补全剩余 640 场景。

```bash
# 确认问题已生成
ls VLM-test/output/questions/ | wc -l  # 应为 700

# 跑全量（Qwen-235B）
cd VLM-test/API-test
VLM_BASE_URL="http://your-local-endpoint/v1" \
VLM_MODEL="qwen/qwen3-vl-235b-a22b" \
VLM_CONCURRENCY=8 \
python run_batch.py

# 跑全量（Qwen-30B）
VLM_MODEL="qwen/qwen3-vl-30b-a3b" \
python run_batch.py
```

**验证：**
```bash
ls VLM-test/output/results/qwen--qwen3-vl-235b-a22b/scenes/ | wc -l  # 应为 700
```

### P0.2 多模型评测（5+ 额外模型）

**预计工作量：** 3-5 天（API 调用 + 费用）

使用 OpenRouter API 评测多个模型家族：

```bash
# 设置 API key
export VLM_API_KEY="your-openrouter-key"

# 逐模型跑（测试集 140 场景先验证，再全量）
./scripts/run_multi_model_eval.sh --model gpt4o --test-only
./scripts/run_multi_model_eval.sh --model gemini-2.0-flash --test-only
./scripts/run_multi_model_eval.sh --model claude-sonnet --test-only

# 验证无误后全量
./scripts/run_multi_model_eval.sh --all
```

**推荐模型列表（优先级排序）：**

| 优先级 | 模型 | OpenRouter ID | 理由 |
|--------|------|--------------|------|
| 必须 | GPT-4o | openai/gpt-4o-2024-11-20 | 标杆闭源模型 |
| 必须 | Gemini 2.0 Flash | google/gemini-2.0-flash-001 | 另一闭源家族 |
| 必须 | Claude Sonnet | anthropic/claude-sonnet-4-20250514 | 第三闭源家族 |
| 推荐 | Qwen3-VL-72B | qwen/qwen3-vl-72b | 开源对比点 |
| 推荐 | LLaVA-OneVision | llava-hf/llava-onevision-qwen2-72b-ov | 开源多样性 |
| 可选 | InternVL2-72B | opengvlab/internvl2-72b | 开源多样性 |

**费用预估：**
- 每场景 ~5-15 API 调用（按 batch 数），每调用约 0.01-0.05 USD
- 700 场景 × 5 模型 ≈ $200-500 USD（OpenRouter 价格）

### P0.3 三条件实验（至少 1 个模型完整数据）

**预计工作量：** 1-2 天

```bash
# 用 GPT-4o 跑完三条件
export VLM_API_KEY="your-key"
./scripts/run_multi_model_eval.sh --model gpt4o --three-cond --test-only

# 或手动分条件跑
cd VLM-test/API-test
VLM_MODEL="openai/gpt-4o-2024-11-20" \
python run_three_condition.py --condition A --max-scenes 100
python run_three_condition.py --condition B --max-scenes 100
python run_three_condition.py --condition C --max-scenes 100
```

**验证：**
- Condition A 结果应与主实验一致
- Condition C（无图像）结果应显著低于 A
- Condition B（错图像）如果接近 C，说明模型确实依赖视觉

### P0.4 人类基线实验

**预计工作量：** 5-7 天（设计 + 招募 + 收集 + 分析）

**设计方案：**

1. **选取场景**：从 700 场景中分层抽样 50-100 场景
   - 每个 split (n04-n10) 至少 7 场景
   - 优先选择模型表现差异大的场景

2. **任务格式**：与 VLM 相同的 QRR/TRR 问题
   - 展示场景图片 + 物体标注
   - 人类通过网页界面回答

3. **标注者**：3+ 人，计算 inter-annotator agreement

4. **平台选择**：
   - Prolific / MTurk（大规模）
   - 实验室招募（小规模但质量高）

5. **分析**：
   - 人类 accuracy 作为天花板
   - 人类 scene belief reconstruction 作为参照
   - 人类 vs VLM 在 CSR/NRMS 上的差异

**需要新建的代码：**
```
VLM-test/human_baseline/
├── generate_tasks.py       # 从场景生成人类任务
├── web_interface/          # 简单网页界面
├── analyze_responses.py    # 分析人类标注
└── compare_human_vlm.py    # 人类 vs VLM 对比
```

---

## P1：强烈推荐（显著提升论文质量）

### P1.1 多视角 vs 单视角对比

**预计工作量：** 1 天

```bash
cd VLM-test/API-test
# 已有 run_multi_view.py，对最佳模型跑多视角
VLM_MODEL="openai/gpt-4o-2024-11-20" \
python run_multi_view.py --test-only
```

### P1.2 Blender 重建可视化（论文 Figure 生成）

**预计工作量：** 1 天

```bash
cd VLM-test
# 已有 generate_blender.py，生成 Blender 脚本
python analysis/generate_blender.py

# 用 Blender 渲染
# 需要选 3-4 个典型场景，生成 GT vs 重建对比图
```

**Figure 清单：**
- Fig 1: 方法总览图（手动绘制，LaTeX/TikZ）
- Fig 2: GT vs 重建配置对比（Blender + 俯视图，3-4 场景）
- Fig 3: 三条件 A/B/C 重建对比（同场景 3 个条件）
- Fig 4: 准确率 vs 物体数衰减曲线（matplotlib）
- Fig 5: NRMS/CSR violin plot（matplotlib）
- Fig 6: 错误案例分析（手动选取 + 标注）

### P1.3 统计显著性检验

**预计工作量：** 0.5 天

需要添加的统计检验：
- 模型间差异：paired t-test 或 Wilcoxon signed-rank test
- VIG 显著性：paired test on NRMS_C - NRMS_A
- 复杂度衰减：趋势检验

---

## P2：可选（锦上添花）

### P2.1 动态时间知觉小规模实验

**预计工作量：** 1 周+

- 当前 `data-gen-dynamic/` 已有 30 场景
- 需要：生成问题 → 评测 → 分析
- 如果结果好，放正文短段；如果不够成熟，放附录

### P2.2 Selective QA 实验

**预计工作量：** 2-3 天

- 允许模型回答 "unknown" / "uncertain"
- 分析 abstention rate vs accuracy trade-off
- 支撑 H2

### P2.3 重述一致性分析

**预计工作量：** 1 天

- 同一问题问两次，比较回答一致性
- 可在三条件实验中顺带做

---

## 分析管线（P0 实验完成后）

```bash
cd VLM-test

# 1. 运行完整分析
python analysis/run_analysis.py --restarts 20

# 2. 三条件信息增益分析
python analysis/information_gain.py

# 3. 生成 SVG 可视化
python analysis/generate_svg_examples.py

# 4. 生成 Blender 重建脚本
python analysis/generate_blender.py

# 5. 基线计算
python analysis/baselines.py
```

---

## 时间线建议

假设投稿截止日期为 D：

| 时间 | 任务 | 产出 |
|------|------|------|
| D-28 | P0.1 全量场景评测 + P0.2 多模型评测启动 | 700 场景 × 6+ 模型结果 |
| D-21 | P0.3 三条件实验 + P0.4 人类基线设计 | 三条件数据 + 人类任务发布 |
| D-14 | P1.1-P1.3 + 人类数据回收 | 多视角/可视化/统计 |
| D-14 | 完整分析管线 + 数据冻结 | 全部 Table/Figure 数据 |
| D-10 | 论文正文初稿 | 9 页正文 |
| D-7  | 附录 + 数据集文档 | Croissant/datasheet |
| D-3  | 内部审稿 + 修改 | 终稿 |
| D-1  | 最终检查 + 提交 | 投稿 |

---

## 检查清单

### 投稿前必须完成

- [ ] 6+ 模型评测完成
- [ ] 人类基线数据
- [ ] 三条件实验 (A/B/C) 至少 1 模型
- [ ] 全量 700 场景分析
- [ ] 所有 Table/Figure 数据齐全
- [ ] 数据集文档 (Croissant/datasheet)
- [ ] 代码仓库清理（README + LICENSE）
- [ ] 复现脚本验证

### 投稿材料清单

- [ ] 正文 PDF (9 页 + references)
- [ ] 附录 PDF
- [ ] 补充材料 (数据集样本、代码链接)
- [ ] NeurIPS D&B checklist 填写
- [ ] 匿名化检查
