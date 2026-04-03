---
configs:
  - config_name: all
    default: true
    data_files:
      - split: train
        path: data/all/train*.parquet
      - split: test
        path: data/all/test*.parquet
  - config_name: qrr
    data_files:
      - split: train
        path: data/qrr/train*.parquet
      - split: test
        path: data/qrr/test*.parquet
  - config_name: trr
    data_files:
      - split: train
        path: data/trr/train*.parquet
      - split: test
        path: data/trr/test*.parquet
  - config_name: fdr
    data_files:
      - split: train
        path: data/fdr/train*.parquet
      - split: test
        path: data/fdr/test*.parquet
task_categories:
  - visual-question-answering
language:
  - en
license: mit
tags:
  - spatial-reasoning
  - vlm-benchmark
  - ordinal-relations
  - 3d-scenes
  - multi-view
size_categories:
  - 100K<n<1M
---

# ORDINARY-BENCH Dataset

A benchmark dataset for evaluating Vision-Language Models (VLMs) on **ordinal spatial reasoning** in 3D scenes.

> Source code & evaluation pipeline: [GitHub - tasd12-ty/ordinary-bench-core](https://github.com/tasd12-ty/ordinary-bench-core)

## Overview

| | |
|---|---|
| Scenes | 700 synthetic 3D scenes (Blender, CLEVR-style) |
| Complexity | 7 levels: 4 to 10 objects per scene (100 each) |
| Questions | 425,971 total across 3 reasoning types |
| Images | 480 x 320 PNG, single-view (embedded in dataset) |
| Multi-view | 4 camera angles per scene (available in source repo) |

## Question Types

### QRR (Quantitative Relation Reasoning) -- 223,671 questions

Compare 3D distances between object pairs. Two variants:
- **Disjoint**: Is `dist(A,B)` less than, approximately equal to, or greater than `dist(C,D)`?
- **Shared anchor**: From anchor A, is `dist(A,B)` less/equal/greater than `dist(A,C)`?
- **Answer format**: `<`, `~=`, or `>`

### TRR (Ternary Relation Reasoning) -- 197,400 questions

Clock-face direction reasoning:
- Standing at object `ref1`, facing toward object `ref2` (12 o'clock direction)
- What clock hour (1-12) is the `target` object at?
- **Answer format**: integer 1-12

### FDR (Full Distance Ranking) -- 4,900 questions

Given an anchor object, rank all other objects by 3D distance, nearest to farthest.
- **Answer format**: ordered JSON array of object IDs, e.g., `["obj_2", "obj_1", "obj_3"]`

## Quick Start

```python
from datasets import load_dataset

# Load QRR questions (test split)
ds = load_dataset("TYTSTQ/ordinary-bench", "qrr", split="test")

sample = ds[0]
sample["image"]                # PIL Image (480x320)
sample["question_text"]        # "Compare the distance between obj_0 and obj_1 vs ..."
sample["qrr_gt_comparator"]   # Ground truth: "<", "~=", or ">"

# Load all question types
ds_all = load_dataset("TYTSTQ/ordinary-bench", split="test")

# Load by specific type
ds_trr = load_dataset("TYTSTQ/ordinary-bench", "trr", split="test")
ds_fdr = load_dataset("TYTSTQ/ordinary-bench", "fdr", split="test")
```

## Configs

| Config | Description | Questions |
|--------|-------------|-----------|
| `all` (default) | All 3 question types | 425,971 |
| `qrr` | Distance comparison only | 223,671 |
| `trr` | Clock direction only | 197,400 |
| `fdr` | Distance ranking only | 4,900 |

## Data Splits

| Split | Scenes per complexity | Total scenes | Total questions |
|-------|----------------------|--------------|-----------------|
| train | 80 | 560 | 340,777 |
| test  | 20 | 140 | 85,194 |

## Column Schema

### Common columns (all configs)

| Column | Type | Description |
|--------|------|-------------|
| `scene_id` | string | Scene identifier, e.g., `n04_000080` |
| `n_objects` | int | Number of objects in scene (4-10) |
| `split` | string | Complexity split: `n04` through `n10` |
| `image` | Image | Rendered scene image (480x320 PNG) |
| `objects` | string | JSON array: `[{"id": "obj_0", "desc": "large brown rubber sphere"}, ...]` |
| `question_type` | string | `qrr`, `trr`, or `fdr` |
| `qid` | string | Question ID, e.g., `qrr_0001` |
| `question_text` | string | Natural language question |
| `scene_metadata` | string | Full scene JSON (3D coordinates, camera parameters, etc.) |

### QRR-specific columns

| Column | Type | Description |
|--------|------|-------------|
| `qrr_variant` | string | `disjoint` or `shared_anchor` |
| `qrr_pair1` | string | JSON: `["obj_0", "obj_1"]` |
| `qrr_pair2` | string | JSON: `["obj_2", "obj_3"]` |
| `qrr_metric` | string | Distance metric, e.g., `dist3D` |
| `qrr_gt_comparator` | string | Ground truth: `<`, `~=`, or `>` |

### TRR-specific columns

| Column | Type | Description |
|--------|------|-------------|
| `trr_target` | string | Target object ID |
| `trr_ref1` | string | Standing position object |
| `trr_ref2` | string | 12 o'clock facing direction object |
| `trr_gt_hour` | int | Ground truth clock hour (1-12) |
| `trr_gt_quadrant` | int | Ground truth quadrant (1-4) |
| `trr_gt_angle_deg` | float | Ground truth angle in degrees |

### FDR-specific columns

| Column | Type | Description |
|--------|------|-------------|
| `fdr_anchor` | string | Anchor object ID |
| `fdr_n_ranked` | int | Number of objects to rank |
| `fdr_gt_ranking` | string | JSON: `["obj_2", "obj_1", "obj_3"]` (nearest to farthest) |
| `fdr_gt_distances` | string | JSON: `[3.006, 3.553, 3.882]` |
| `fdr_gt_tie_groups` | string | JSON: `[["obj_2"], ["obj_1", "obj_3"]]` |

## Scoring Criteria

| Type | Metric | Description |
|------|--------|-------------|
| QRR | Accuracy | Exact comparator match (`<`, `~=`, `>`) |
| TRR | Hour accuracy | Exact clock hour match |
| TRR | Quadrant accuracy | Correct quadrant (1/4 of clock face) |
| TRR | Adjacent accuracy | Within +/-1 hour of ground truth |
| FDR | Exact accuracy | Full ranking match (respecting tie groups) |
| FDR | Kendall tau | Rank correlation coefficient [-1, 1] |
| FDR | Pairwise accuracy | Fraction of correct pairwise orderings |
| FDR | Top-1 accuracy | Nearest object correctly identified |

## Prompt Templates

System prompts for VLM evaluation are included in `prompts/system_prompts.json`. They instruct VLMs to respond with a JSON array of `{"qid": "...", "answer": ...}` objects.

## Run Evaluation from HuggingFace Dataset

Use `hf_to_local.py` to convert the HuggingFace dataset to local format, then run evaluation with the standard pipeline.

```bash
# Step 1: Convert HF dataset to local format
python datasets/hf_to_local.py --repo TYTSTQ/ordinary-bench --split test --output ./hf_data

# Step 2: Inspect extracted data
ls hf_data/images/single_view/ | head
cat hf_data/questions/qrr/n04_000080.json | python -m json.tool | head

# Step 3: Run evaluation (edit job TOML with your model/API key first)
cd VLM-test/API-test
python run_eval.py --job ../../datasets/jobs/hf_eval_example.toml
```

Multi-view variant:

```bash
python datasets/hf_to_local.py --repo TYTSTQ/ordinary-bench-multiview --split test --output ./hf_data_mv
```

See `datasets/jobs/hf_eval_example.toml` for a complete job configuration template.

## Source Code

The full evaluation pipeline, scene generation code, and reconstruction tools are available at:

**[github.com/tasd12-ty/ordinary-bench-core](https://github.com/tasd12-ty/ordinary-bench-core)**

## License

MIT
