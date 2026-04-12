# VRF-test — Verification Question Evaluation

VRF (Verification) 是一种固定题量的空间关系验证测试。每道题包含多条 sub-claims（距离比较 + 钟面方向 + 距离排序），VLM 判断整组是否全部正确。

## 特点

- **固定题量**：每场景 K=20 道题（可配置），不随物体数 N 变化
- **复合验证**：每题 3 条 sub-claims，混合 QRR/TRR/FDR 来源
- **平衡设计**：TRUE/FALSE 各半；FALSE 题中恰好 1 条被篡改
- **确定性**：相同 scene_id + K + tau 恒产出相同题目

## 快速开始

### 1. 生成 VRF 问题

```bash
cd VRF-test
python generate_questions.py --data ../datasets/test-data
```

### 2. 烟雾测试（无需 API）

```bash
python run_eval.py --job jobs/smoke.toml
```

### 3. 正式评测

```bash
python run_eval.py --job jobs/your_model.toml
```

## CLI 参数

### generate_questions.py

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data` | (必填) | 数据目录（含 `scenes/`） |
| `--output` | `./output` | 输出目录 |
| `--split` | 无 | 筛选 split 前缀（如 `n04`） |
| `--K` | 20 | 每场景题量 |
| `--claims-per-question` | 3 | 每题 sub-claims 数 |
| `--tau` | 0.10 | 容差参数 |

### run_eval.py

| 参数 | 说明 |
|------|------|
| `--job` | TOML job 配置文件路径 |

## TOML Job 配置

```toml
[provider]
adapter = "openai_chat"
model = "gpt-4o"
base_url = "env:OPENAI_BASE_URL"
api_key = "env:OPENAI_API_KEY"

[input]
questions_dir = "../datasets/test-data/questions/vrf"

[images]
images_dir = "../datasets/test-data/images/single_view"
mode = "single"

[selection]
split = "n04"

[output]
results_dir = "output/results"
run_name = "gpt4o_vrf"
```

## 评分指标

| 指标 | 说明 |
|------|------|
| `vrf_accuracy` | 总正确率 |
| `vrf_true_accuracy` | TRUE 题正确率（正确识别全对组） |
| `vrf_false_accuracy` | FALSE 题正确率（正确识别含错组） |
