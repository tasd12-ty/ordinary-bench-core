# Human Baseline

`human-baseline/` 独立于 `VLM-test/`，用于生成人类标注任务并把标注结果转换成与当前 VLM 结果兼容的目录结构。

## 设计约束

- 不考虑并发，脚本按顺序处理场景。
- 标注图只画偏移文字标签和引导线，不叠加对象点位。
- 导出的响应文件保留 `raw_response`，并按 batch 组织，方便后续统一评分、聚合和场景重建。
- 最终结果目录与 `VLM-test/output/results/<model>/` 对齐：
  - `raw/`
  - `scenes/`
  - `summary.json`

## 1. 生成任务

```bash
python human-baseline/generate_tasks.py --split n04 --max-scenes 5
```

默认输出到 `human-baseline/output/tasks/`：

- `index.html`：任务索引页
- `pages/<scene_id>.html`：按场景的静态标注页
- `json/<scene_id>.json`：任务 JSON
- `images/<scene_id>_labels.png`：带标签图片

标注页导出的 JSON 已经带有：

- `responses`
- `batches`
- `raw_response`

这些字段会被后续分析脚本直接消费。

## 2. 收集响应

把导出的响应 JSON 放到任意目录下，例如：

```text
human-baseline/output/responses/alice/n04_000000__alice.json
human-baseline/output/responses/alice/n04_000001__alice.json
```

也可以混合多个标注者，分析脚本会按 `annotator_id` 自动分组。

## 3. 转成兼容结果目录

```bash
python human-baseline/analyze_responses.py
```

默认读取 `human-baseline/output/responses/`，输出到 `human-baseline/output/results/`。

每个标注者会得到一个独立结果目录：

```text
human-baseline/output/results/human--alice/
├── raw/
├── scenes/
└── summary.json
```

这个目录可以直接喂给现有分析/重建脚本，例如：

```bash
python VLM-test/analysis/prepare_reconstruction_inputs.py \
  --results-dir human-baseline/output/results/human--alice \
  --output-dir human-baseline/output/prepared/alice
```
