---
configs:
  - config_name: default
    default: true
    data_files:
      - split: train
        path: data/train*.parquet
      - split: test
        path: data/test*.parquet
task_categories:
  - visual-question-answering
language:
  - en
license: mit
tags:
  - spatial-reasoning
  - vlm-benchmark
  - 3d-grid
  - multi-view
  - orthographic
  - object-localization
size_categories:
  - n<1K
---

# ORDINARY-BENCH Grid3D Dataset

A benchmark dataset for evaluating Vision-Language Models (VLMs) on **3D object localization** in a 4x4x4 grid using 6 orthographic views.

> Source code & evaluation pipeline: [GitHub - tasd12-ty/ordinary-bench-core](https://github.com/tasd12-ty/ordinary-bench-core)
>
> Related dataset (free-placement scenes): [TYTSTQ/ordinary-bench](https://huggingface.co/datasets/TYTSTQ/ordinary-bench)

## Overview

| | |
|---|---|
| Scenes | 140 synthetic 3D scenes (Blender, 4x4x4 grid) |
| Complexity | 7 levels: 4 to 10 objects per scene (20 each) |
| Views | 6 orthographic projections per scene (480x480 PNG) |
| Task | Determine each object's grid cell from the 6 views |

## Task Description

Objects are placed in a **4x4x4 discrete grid** (64 possible positions). The VLM receives 6 labeled orthographic views and must output each object's grid cell position.

### Coordinate System

```
Row:    A, B, C, D    (A = front, D = back)
Column: 1, 2, 3, 4    (1 = left,  4 = right)
Layer:  1, 2, 3, 4    (1 = bottom, 4 = top)

Position format: RowCol-Layer
Example: "B3-4" = Row B, Column 3, Layer 4
```

### View Projections

| View | What it shows | Axes |
|------|---------------|------|
| **Top** (looking down) | Row + Column | Rows A-D top-to-bottom, Cols 1-4 left-to-right |
| **Front** (from front) | Column + Layer | Cols 1-4 left-to-right, Layers 1-4 bottom-to-top |
| **Right** (from right) | Row + Layer | Rows A-D left-to-right, Layers 1-4 bottom-to-top |
| **Back** (from back) | Column + Layer | Cols 4-1 left-to-right (reversed), Layers 1-4 bottom-to-top |
| **Left** (from left) | Row + Layer | Rows D-A left-to-right (reversed), Layers 1-4 bottom-to-top |
| **Bottom** (looking up) | Row + Column | Rows A-D top-to-bottom, Cols 4-1 left-to-right (reversed) |

Any two orthogonal views (e.g., Top + Front) are sufficient to determine all 3 coordinates, but 6 views provide redundancy for verification.

## Quick Start

```python
from datasets import load_dataset

ds = load_dataset("TYTSTQ/ordinary-bench-grid3d", split="test")

sample = ds[0]
sample["image_top"]       # PIL Image (480x480) - top-down view
sample["image_front"]     # PIL Image (480x480) - front view
sample["system_prompt"]   # System prompt with coordinate system
sample["user_prompt"]     # User prompt with view labels + object list
sample["ground_truth"]    # JSON: [{"object": "cyan rubber cylinder", "cell": "B3-4"}, ...]
```

## Data Splits

| Split | Scenes per complexity | Total scenes |
|-------|----------------------|--------------|
| train | 15 | 105 |
| test  | 5  | 35  |

## Column Schema

| Column | Type | Description |
|--------|------|-------------|
| `scene_id` | string | Scene identifier, e.g., `g07_000010` |
| `n_objects` | int | Number of objects (4-10) |
| `split` | string | Complexity split: `g04` through `g10` |
| `image_top` | Image | Top orthographic view (looking down) |
| `image_bottom` | Image | Bottom orthographic view (looking up) |
| `image_front` | Image | Front orthographic view |
| `image_back` | Image | Back orthographic view |
| `image_left` | Image | Left orthographic view |
| `image_right` | Image | Right orthographic view |
| `objects` | string | JSON: object descriptions |
| `system_prompt` | string | System prompt for VLM |
| `user_prompt` | string | User prompt with view annotations |
| `ground_truth` | string | JSON: `[{"object": "...", "cell": "B3-4"}, ...]` |
| `scene_metadata` | string | Full scene JSON (3D coords, grid info) |

## Expected Response Format

```json
[
  {"object": "cyan rubber cylinder", "cell": "B3-4"},
  {"object": "brown metal sphere", "cell": "A1-2"}
]
```

## Scoring Criteria

| Metric | Description |
|--------|-------------|
| **Exact accuracy** | Predicted cell == GT cell |
| **Structural accuracy** | Correct under D4 symmetry (rotation/reflection in row-col plane) |
| **Per-dimension accuracy** | Row, Column, Layer accuracy independently |

The D4 symmetry metric accounts for consistent rotations/reflections of the entire grid, which can happen when VLMs misinterpret view orientations.

## Source Code

Full generation pipeline, VLM evaluation, and scoring tools:

**[github.com/tasd12-ty/ordinary-bench-core/tree/main/data-gen-grid3d](https://github.com/tasd12-ty/ordinary-bench-core/tree/main/data-gen-grid3d)**

## License

MIT
