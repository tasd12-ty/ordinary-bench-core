# Quick-Sort BT0 重建运行手册

这份手册用于把 quick-sort 结果送入 BT=0 重建，并在结束后生成按模型、run、物体数量分组的透视统计。

默认场景是假设：

- quick-sort 原始结果已经在服务器上
- GT scene JSON 也已经在服务器上
- 不需要上传或同步数据，只需要把路径配对

核心入口：

- 运行脚本：`VLM-test/analysis/run_quicksort_bt0_pipeline.sh`
- 透视统计：`VLM-test/analysis/pivot_quicksort_recon_results.py`
- 自定义 manifest：任意纯文本文件，每行一个 `source_attempt/canonical_run`

## 1. 目录约定

脚本只关心 3 个路径：

1. `QS_ROOT`
   quick-sort source results 根目录。每个 run 目录下都要有 `summary.json` 和 `scenes/`。
2. `GT_SCENES`
   GT scene JSON 目录，比如 `.../scenes/*.json`。
3. `OUT_ROOT`
   重建输出根目录。脚本会在这里写每个 run 的 `prepared/`、`recon/` 和最终 workbook。

一个典型服务器布局示例：

```text
/mnt/bench/quick-sort/output/
  result-397-s/397-s/{summary.json,scenes/...}
  results-27-s/27-s/{summary.json,scenes/...}
  ...

/mnt/bench/scenes/
  n20_000000__sz09_s0000.json
  n20_000000__sz09_s0001.json
  ...

/mnt/bench/reconstruction-qrr-bt0/
  ...
```

对应环境变量：

```bash
export QS_ROOT=/mnt/bench/quick-sort/output
export GT_SCENES=/mnt/bench/scenes
export OUT_ROOT=/mnt/bench/reconstruction-qrr-bt0
```

## 2. 最小启动方式

先做一次 dry run，确认 run 选择和路径都对：

```bash
cd /path/to/ordinary-bench-core

export QS_ROOT=/mnt/bench/quick-sort/output
export GT_SCENES=/mnt/bench/scenes
export OUT_ROOT=/mnt/bench/reconstruction-qrr-bt0
export RUNS_FILE=/path/to/my_runs.txt

DRY_RUN=1 bash VLM-test/analysis/run_quicksort_bt0_pipeline.sh
```

确认无误后正式运行：

```bash
bash VLM-test/analysis/run_quicksort_bt0_pipeline.sh
```

如果不提供 `RUNS_FILE`，脚本会自动扫描 `QS_ROOT` 下所有满足 `summary.json + scenes/` 的 run。

## 3. tmux 挂后台

推荐所有长任务都放在 `tmux` 里：

```bash
tmux new-session -d -s bt0-recon
tmux send-keys -t bt0-recon 'cd /path/to/ordinary-bench-core' C-m
tmux send-keys -t bt0-recon 'export QS_ROOT=/mnt/bench/quick-sort/output' C-m
tmux send-keys -t bt0-recon 'export GT_SCENES=/mnt/bench/scenes' C-m
tmux send-keys -t bt0-recon 'export OUT_ROOT=/mnt/bench/reconstruction-qrr-bt0' C-m
tmux send-keys -t bt0-recon 'export RUNS_FILE=/path/to/my_runs.txt' C-m
tmux send-keys -t bt0-recon 'bash VLM-test/analysis/run_quicksort_bt0_pipeline.sh' C-m
```

常用命令：

```bash
tmux attach -t bt0-recon
tmux capture-pane -pt bt0-recon -S -120
```

detach 用 `Ctrl-b d`。

## 4. 常用配置项

脚本全部通过环境变量配置：

```bash
CONCURRENCY=8 \
RESTARTS=10 \
BT_RATIO_ALPHA=0 \
SKIP_DONE=1 \
FORCE_RERUN=0 \
SKIP_PIVOT=0 \
bash VLM-test/analysis/run_quicksort_bt0_pipeline.sh
```

含义如下：

- `CONCURRENCY`
  并行 run 数，实际对应 `xargs -P`。服务器核数够、内存够时可以调大。
- `RESTARTS`
  每个 scene 的 solver restart 次数。
- `BT_RATIO_ALPHA`
  传给 `reconstruct_quicksort_orders.py --bt-ratio-alpha`。BT0 实验通常设成 `0`。
- `SKIP_DONE=1`
  如果某个 run 已经存在 `recon/summary.json`，自动跳过。
- `FORCE_RERUN=1`
  忽略已完成标记，强制重跑。设为 `1` 时会自动关闭 `SKIP_DONE`。
- `SKIP_PIVOT=1`
  只跑重建，不在结尾生成 workbook。
- `MAX_SCENES`
  每个 run 只跑前 N 个 scene，适合 smoke test。
- `WORKBOOK`
  自定义 Excel 输出路径。默认是 `$OUT_ROOT/pivots/quicksort_bt0_recon_pivot.xlsx`。

## 5. 如何加大并发

最直接的是提高 `CONCURRENCY`：

```bash
CONCURRENCY=16 bash VLM-test/analysis/run_quicksort_bt0_pipeline.sh
```

建议按机器情况逐步加：

- CPU 核数接近或高于并发数
- 内存充足，因为每个 run 都会在 scene 级别持续写 JSON
- 文件系统吞吐足够，否则会变成 I/O 拖慢

比较稳妥的做法：

1. 先用 `CONCURRENCY=4`
2. 看 CPU、内存、磁盘写入
3. 再加到 `8`、`12`、`16`

如果是多机或多 agent，优先分片 manifest，而不是在一台机器上无限加 `CONCURRENCY`。

## 6. 如何配置更多数据

有两种方式：

### 方式 A：换 `QS_ROOT`

如果服务器上有更大的 quick-sort 结果树，直接切到新的根目录：

```bash
export QS_ROOT=/mnt/bench/full-quick-sort/output
bash VLM-test/analysis/run_quicksort_bt0_pipeline.sh
```

不提供 `RUNS_FILE` 时，脚本会自动把这个根目录下所有 run 都纳进来。

### 方式 B：换 `RUNS_FILE`

如果只想跑更大的一个子集，或者补跑某些模型/某些尝试，写 manifest：

```text
results-397-m/397-m
results-397-m-1/397-m
results-9-s/9-s
```

然后：

```bash
export RUNS_FILE=/path/to/my_runs.txt
bash VLM-test/analysis/run_quicksort_bt0_pipeline.sh
```

## 7. 多 agent 协同运行

推荐所有 agent 共用同一个 `OUT_ROOT`，但每个 agent 使用**不重叠**的 manifest 分片。

例如把总 manifest 切成 3 份：

```text
shard_a.txt
shard_b.txt
shard_c.txt
```

每个 agent 各跑一份：

```bash
RUNS_FILE=shard_a.txt bash VLM-test/analysis/run_quicksort_bt0_pipeline.sh
RUNS_FILE=shard_b.txt bash VLM-test/analysis/run_quicksort_bt0_pipeline.sh
RUNS_FILE=shard_c.txt bash VLM-test/analysis/run_quicksort_bt0_pipeline.sh
```

注意：

- 不要让两个 agent 同时处理同一个 run
- 可以共用同一个 `OUT_ROOT`
- `SKIP_DONE=1` 能避免重复处理已经完整完成的 run
- 如果某个 agent 只负责重建，不负责最后统计，可以设 `SKIP_PIVOT=1`

推荐流程：

1. 各 agent 先跑自己的 shard，统一 `SKIP_PIVOT=1`
2. 等所有 shard 完成后，任选一个 agent 单独跑透视统计

## 8. 单独生成透视统计

如果重建已经跑完，或者你只想重新汇总 workbook，不需要再跑 solver：

```bash
uv run --with openpyxl python VLM-test/analysis/pivot_quicksort_recon_results.py \
  --recon-root "$OUT_ROOT" \
  --source-results-root "$QS_ROOT" \
  --output-xlsx "$OUT_ROOT/pivots/quicksort_bt0_recon_pivot.xlsx"
```

workbook 包含 4 个 sheet：

- `scene_long`
  scene 粒度长表，每行一个 scene
- `pivot_model_run_n`
  按模型家族 + run + 物体数聚合
- `pivot_model_run`
  按模型家族 + run 聚合
- `pivot_model_n`
  按模型家族 + 物体数聚合

这些表里会保留：

- `model_family`
- `run_label`
- `n_objects`
- `reconstructed`
- `skipped`
- `feasible`
- `single_mode` / `multimodal` / `infeasible`
- `csr_qrr_mean`
- `csr_qrr_aligned_mean`
- `kendall_tau_mean`
- `nrms_mean`

## 9. 断点续跑

这个脚本的续跑粒度是 run，不是 scene。

含义是：

- 一个 run 完整完成后，会写出 `recon/summary.json`
- 之后再次启动时，`SKIP_DONE=1` 会把这个 run 整体跳过
- 如果一个 run 只跑到一半，还没有写出最终 `recon/summary.json`，再次启动时会把这个 run 重新跑一遍

所以更推荐：

- 长任务尽量用 `tmux`
- 多 agent 协同靠 manifest 分片
- 不靠 scene 级恢复

## 10. 验证建议

第一次上服务器时，先做一轮 smoke test：

```bash
MAX_SCENES=3 DRY_RUN=1 bash VLM-test/analysis/run_quicksort_bt0_pipeline.sh
MAX_SCENES=3 SKIP_PIVOT=1 bash VLM-test/analysis/run_quicksort_bt0_pipeline.sh
```

确认：

- 路径解析正确
- 输出目录结构符合预期
- `recon/summary.json` 能正常生成

然后再去掉 `MAX_SCENES` 跑全量。
