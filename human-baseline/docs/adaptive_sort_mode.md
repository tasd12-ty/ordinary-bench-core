# Adaptive Sort 模式接入说明

`human-baseline/server_v2.py` 现在支持两种人类测试模式：

- `progressive`
- `adaptive_sort`

其中 `adaptive_sort` 面向类似 `adaptive-sort/` 的 quicksort 问法，交互特征是：

- 每个 step 固定一个 `pivot` 距离对
- 人类逐个比较若干 `candidate` 距离对相对于 `pivot` 的远近
- 提交后**不显示标准答案**

这份文档只说明 **UI / API / task bundle 接口**，不要求本机拥有完整场景数据。

## 1. 本地最小联调

仓库内已经提供一个 synthetic bundle：

```text
human-baseline/examples/adaptive_sort_tasks/
├── manifest.json
└── scenes/
    ├── synthetic_adsort_0001.json
    └── synthetic_adsort_0002.json
```

直接启动：

```bash
python human-baseline/server_v2.py
```

默认会把 `--adaptive-sort-tasks-dir` 指向上面的 example bundle，所以首页可以直接看到 `Adaptive Sort` 选项。

如果要显式指定：

```bash
python human-baseline/server_v2.py \
  --adaptive-sort-tasks-dir human-baseline/examples/adaptive_sort_tasks
```

打开 `http://127.0.0.1:8124`，选择 `Adaptive Sort`，即可验证：

1. 首页模式切换
2. 无图片时的工作台布局
3. `allow_approx=true/false` 两种按钮配置
4. `step -> next_step -> scene_complete -> all_done` 推进流程

## 2. Task Bundle 目录结构

服务端只读取文件型 bundle，不在当前机器上即时生成 quicksort step。

约定结构：

```text
<adaptive_sort_tasks_dir>/
├── manifest.json
└── scenes/
    ├── <scene_id>.json
    ├── ...
```

`manifest.json` 最少包含：

```json
{
  "schema_version": 1,
  "test_mode": "adaptive_sort",
  "scenes": [
    {
      "scene_id": "n10_000080",
      "task_file": "scenes/n10_000080.json",
      "title": "n10_000080"
    }
  ]
}
```

每个 scene task 最少包含：

```json
{
  "schema_version": 1,
  "scene_id": "n10_000080",
  "title": "n10_000080",
  "allow_approx": true,
  "objects": [
    { "id": "obj_0", "label": "obj_0", "desc": "红色球体" }
  ],
  "images": {
    "single_view": "images/single_view/n10_000080.png",
    "multi_view": [
      "images/multi_view/n10_000080/view_0.png",
      "images/multi_view/n10_000080/view_1.png",
      "images/multi_view/n10_000080/view_2.png",
      "images/multi_view/n10_000080/view_3.png"
    ]
  },
  "steps": [
    {
      "step_id": "step_001",
      "level": 0,
      "pivot": ["obj_0", "obj_1"],
      "candidates": [
        { "qid": "cmp_001", "pair": ["obj_0", "obj_2"], "gt_answer": "<" }
      ]
    }
  ]
}
```

字段要求：

- `allow_approx=true` 时，前端显示 `< / ~= / >`
- `allow_approx=false` 时，前端只显示 `< / >`
- `objects[*].desc` 用于前端展示
- `images` 可为空；这样仍然可以联调接口和交互
- `gt_answer` 当前不会在前端展示，但建议保留，便于后续离线分析

## 3. 图片路径约定

前端的路径解析规则：

- `tasks/...` 走 `/tasks/...`
- 其它相对路径走 `/data-images/...`

因此，在有数据的机器上，最省事的接法是直接复用现有 ordinary-bench 图片目录：

```json
{
  "images": {
    "single_view": "images/single_view/n10_000080.png",
    "multi_view": [
      "images/multi_view/n10_000080/view_0.png",
      "images/multi_view/n10_000080/view_1.png",
      "images/multi_view/n10_000080/view_2.png",
      "images/multi_view/n10_000080/view_3.png"
    ]
  }
}
```

然后启动 server 时照常传：

```bash
python human-baseline/server_v2.py \
  --images-dir /path/to/data-gen/output/images/single_view \
  --multi-view-images-dir /path/to/data-gen/output/images/multi_view \
  --adaptive-sort-tasks-dir /path/to/adaptive_sort_tasks
```

如果你有自己额外生成的标注图，也可以把 path 写成 `tasks/images/...`，由 `/tasks/` 路由提供。

## 4. 在有数据机器上快速导出 task bundle

推荐做法是让 agent 在有完整数据的机器上，额外写一个离线导出脚本，把 `adaptive-sort` 的 step 记录转成上面的 bundle，而不是改 `server_v2.py` 去在线生成。

最直接的来源有两类：

### 方案 A：从 `adaptive-sort` 现有结果导出

适用于你已经有：

- `adaptive-sort/output/results/<run_name>/scenes/<scene_id>.json`

这类结果文件中已经有：

- `vlm_result.rounds[*].pivot`
- `vlm_result.rounds[*].candidates`

导出脚本要做的事情只是：

1. 读取单场景 result JSON
2. 提取 `rounds[*]`
3. 转成 `steps[*]`
4. 补上 `objects` 和 `images`
5. 写出 `manifest.json` + `scenes/<scene_id>.json`

这种方式最快，因为 step 切分已经有了。

### 方案 B：从 scene JSON + GT / quicksort 逻辑现场导出

适用于你只有：

- `data-gen/output/scenes/<scene_id>.json`
- 图片目录

导出脚本可以在离线阶段复用 `adaptive-sort` 里的逻辑：

- `sorting.py`
- `gt_ranking.py`

但**不要**在导出脚本里调用 VLM。要做的是：

1. 根据 scene 列出所有距离对
2. 选择 pivot 序列
3. 生成每个 step 的 candidate list
4. 写出 task bundle

这一步只是组织人类测试任务，不是评测模型。

## 5. v2 API 约定

前端现在统一走这组接口，并显式传 `test_mode`：

- `GET /api/v2/capabilities`
- `POST /api/v2/session/start`
- `GET /api/v2/scene/current?annotator_id=...&test_mode=...`
- `POST /api/v2/scene/allocate`
- `POST /api/v2/round/submit`

`adaptive_sort` 的 round payload 至少会返回：

- `test_mode`
- `scene_id`
- `step_id`
- `step_index`
- `n_steps_total`
- `level`
- `images`
- `objects`
- `pivot_pair`
- `questions`
- `allow_approx`

`submit` 返回：

- `accepted`
- `next_action`
- `submission_summary`
- `progress`

## 6. 输出文件

`adaptive_sort` 的人类回答会单独落到：

```text
human-baseline/output/responses/<annotator_id>/
├── adaptive_sort_progress_v2.json
└── <scene_id>__<annotator_id>__adaptive_sort.json
```

单场景 response payload 会保留：

- `source_task_file`
- `allow_approx`
- `step_records`
- `responses`

当前仓库**没有**把这类结果接进 `human-baseline/analyze_responses.py`。后续如果要评分，建议另写一个离线转换脚本，直接消费这些 `step_records`。

## 7. Agent 在有数据机器上的推荐执行顺序

1. 准备好 scene JSON 和图片目录
2. 写一个离线导出脚本，把 quicksort step 转成 bundle
3. 生成 `manifest.json` + `scenes/*.json`
4. 启动 `human-baseline/server_v2.py --adaptive-sort-tasks-dir <bundle>`
5. 先用浏览器跑 1 个 scene 验证 step 推进、按钮配置和 response 落盘
6. 再扩大到完整场景集
