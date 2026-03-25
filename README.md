# ORDINARY-BENCH

Benchmark for evaluating Vision-Language Models (VLMs) on ordinal spatial relation understanding. The benchmark generates 3D scenes with multiple objects, renders images, and tests VLMs on three types of spatial reasoning questions:

- **QRR (Quantitative Relation Reasoning)**: Compare pairwise spatial metrics across objects.
  - *disjoint*: compare two non-overlapping pairs (A,B) vs (C,D)
  - *shared_anchor*: from anchor A, compare dist(A,B) vs dist(A,C)
- **TRR (Ternary Relation Reasoning)**: Determine clock-face directional relations among three objects.
- **FDR (Full Distance Ranking)**: Rank all objects by distance from an anchor, nearest to farthest.

## Dataset

The benchmark dataset is available on HuggingFace Hub with 700 scenes and 332,857 questions:

**[huggingface.co/datasets/TYTSTQ/ordinary-bench](https://huggingface.co/datasets/TYTSTQ/ordinary-bench)**

```python
from datasets import load_dataset

# Load QRR questions
ds = load_dataset("TYTSTQ/ordinary-bench", "qrr", split="test")
sample = ds[0]
sample["image"]               # PIL Image (480x320)
sample["question_text"]       # Natural language question
sample["qrr_gt_comparator"]  # Ground truth: "<", "~=", ">"

# Available configs: all (default), qrr, trr, fdr
```

## Project Structure

```
ordinary-bench/
├── data-gen/                      # Scene generation & rendering
│   ├── generate.py                # Entry point
│   ├── pipeline.py                # Blender orchestration
│   ├── config.toml                # Generation config
│   └── blender/                   # Blender scripts & assets
│       ├── render_multiview.py
│       └── assets/                # .blend files, shapes, materials
├── datasets/                      # HuggingFace dataset build scripts
│   ├── build_dataset.py           # Build parquet from scene/question data
│   ├── README.md                  # HuggingFace dataset card
│   └── prompts/                   # System prompt templates
├── data-gen-infinigen/            # Infinigen realistic scene backend
│   ├── generate.py                # Infinigen-Indoors orchestrator
│   ├── adapter.py                 # Infinigen → ordinary-bench converter
│   └── README.md                  # Backend documentation
├── VLM-test/                      # VLM evaluation
│   ├── generate_questions.py      # Generate QRR/TRR/FDR questions from scenes
│   ├── generate_questions_v2.py   # Per-type directory output (recommended)
│   ├── question_bank.py           # Question enumeration logic
│   ├── extraction.py              # Ground truth extraction
│   ├── dsl/                       # Domain-specific language
│   │   ├── predicates.py          # QRR/TRR constraint definitions
│   │   └── comparators.py         # Comparator enum (<, ~=, >)
│   ├── reconstruct/               # Scene reconstruction from constraints
│   │   ├── constraints.py         # Constraint preprocessing & feasibility
│   │   ├── solver.py              # Gradient-based 2D position optimizer
│   │   ├── pipeline.py            # End-to-end reconstruction entry point
│   │   └── evaluate.py            # Reconstruction quality metrics
│   ├── docs/
│   │   └── scoring_criteria.md    # Detailed scoring documentation
│   └── API-test/                  # VLM API testing
│       ├── run_batch.py           # Batch evaluation entry point
│       ├── run_batch_v2.py        # Per-type directory batch runner
│       ├── config.py              # API config (env vars)
│       ├── vlm_client.py          # OpenAI-compatible API client
│       ├── prompts.py             # System/user prompts
│       ├── response_parser.py     # Parse VLM responses
│       └── scoring.py             # Score predictions against GT
└── pyproject.toml
```

## Requirements

- Python >= 3.9
- [Blender](https://www.blender.org/) (for scene generation)
- Dependencies: `numpy`, `openai`

```bash
# Install with uv (recommended)
uv sync

# Or with pip
pip install -e .
```

## Phase 1: Data Generation

Generate 3D scenes and render images using Blender.

### Configuration

Edit `data-gen/config.toml`:

```toml
[blender]
executable = "/path/to/blender"   # Blender executable path
use_gpu = true                    # Enable GPU rendering

[rendering]
width = 480
height = 320
samples = 256         # Cycles samples (lower = faster)
n_views = 4           # Camera viewpoints per scene
camera_distance = 12.0
elevation = 30.0
azimuth_start = 45.0

[objects]
min_count = 4         # Min objects per scene
max_count = 10        # Max objects per scene
min_dist = 0.25       # Min distance between object centers
margin = 0.4

[output]
dir = "./output"
seed = 42

# Define splits — each split generates scenes with a fixed object count
[splits.n04]
n_scenes = 10
min_objects = 4
max_objects = 4

[splits.n05]
n_scenes = 10
min_objects = 5
max_objects = 5

# ... add more splits as needed (n06–n10)
```

### Run Generation

```bash
cd data-gen

# Full generation (uses config.toml)
python generate.py

# Quick test (1 scene per split, low quality)
python generate.py --preset test

# Custom config file
python generate.py --config my_config.toml

# Override Blender path and output directory
python generate.py --blender /usr/bin/blender --output-dir ./my_output

# Enable GPU rendering
python generate.py --gpu

# Parallel rendering (multiple Blender processes)
python generate.py --workers 4

# Dry run — print resolved config without rendering
python generate.py --dry-run
```

### Output Structure

```
data-gen/output/
├── images/
│   ├── single_view/    # One image per scene
│   └── multi_view/     # Multiple viewpoints per scene
├── scenes/             # Per-scene JSON metadata (object positions, properties)
├── splits/             # Split index files
└── dataset_info.json   # Dataset summary
```

## Phase 2: Question Generation

Generate QRR, TRR, and FDR evaluation questions from scene data.

```bash
cd VLM-test

# Generate questions from all scenes
python generate_questions.py --data ../data-gen/output

# Specify split
python generate_questions.py --data ../data-gen/output --split n04

# Custom batch size and tolerance
python generate_questions.py --data ../data-gen/output --batch-size 10 --tau 0.10

# Show question count table (no generation)
python generate_questions.py --counts

# v2 — per-type directory output (recommended)
python generate_questions_v2.py --data ../data-gen/output
python generate_questions_v2.py --counts  # Show question count table
```

### v2 Output Structure

```
VLM-test/output/questions/
├── qrr/{scene_id}.json
├── trr/{scene_id}.json
└── fdr/{scene_id}.json
```

Legacy output is saved to `VLM-test/output/questions/` (batch mode) and `VLM-test/output/extraction_tasks/` (extraction mode).

## Phase 3: VLM Evaluation

Test VLMs on the generated questions via an OpenAI-compatible API.

### Configuration

Set environment variables:

```bash
# Required
export VLM_API_KEY="your-api-key"

# API endpoint (default: OpenRouter)
export VLM_BASE_URL="https://openrouter.ai/api/v1"

# Model selection
export VLM_MODEL="google/gemini-2.0-flash-001"

# Optional: OpenRouter provider routing
export VLM_PROVIDER="google"

# Concurrency and retry settings
export VLM_CONCURRENCY=4        # Parallel scene processing (default: 4)
export VLM_TIMEOUT=120          # Request timeout in seconds (default: 120)
export VLM_MAX_RETRIES=5        # Max retries per request (default: 5)
export VLM_RETRY_DELAY=2.0      # Base retry delay in seconds (default: 2.0)
```

### Run Evaluation

```bash
cd VLM-test/API-test

# Run all scenes
python run_batch.py

# Run a specific split
python run_batch.py --split n04

# Run a single scene
python run_batch.py --scene n04_000000

# v2 — per-type directory input (recommended, matches v2 question output)
python run_batch_v2.py
python run_batch_v2.py --split n04
```

### Results

Results are organized by model name under `VLM-test/output/results/<model>/`:

```
VLM-test/output/results/google--gemini-2.0-flash-001/
├── raw/          # Raw VLM responses per batch
├── scenes/       # Per-scene scoring results
└── summary.json  # Aggregated metrics
```

Key metrics reported:

- **QRR Accuracy**: Exact match on comparator prediction
- **QRR Disjoint Accuracy**: Accuracy on disjoint-pair QRR questions
- **QRR Shared-Anchor Accuracy**: Accuracy on anchor-based QRR questions
- **TRR Hour Accuracy**: Exact clock-hour match
- **TRR Quadrant Accuracy**: Correct quadrant (coarser granularity)
- **FDR Exact Accuracy**: Full ranking match (respecting tie groups)
- **FDR Kendall τ**: Rank correlation coefficient
- **FDR Pairwise Accuracy**: Fraction of correct pairwise orderings
- **FDR Top-1 Accuracy**: Nearest object correctly identified

## Testing Multiple Models

Switch models by changing environment variables:

```bash
# Test GPT-4o
VLM_MODEL="openai/gpt-4o" python run_batch.py

# Test Qwen2.5-VL via OpenRouter
VLM_MODEL="qwen/qwen-2.5-vl-72b-instruct" python run_batch.py

# Test a local model
VLM_BASE_URL="http://localhost:8000/v1" VLM_MODEL="local-model" python run_batch.py
```

## Infinigen Backend

`data-gen-infinigen/` provides a prototype backend for generating realistic indoor scenes using [Infinigen](https://infinigen.org/). See [`data-gen-infinigen/README.md`](data-gen-infinigen/README.md) for setup and usage.

Key features:
- Infinigen-Indoors single-room scenes
- Adapter converts Infinigen metadata to ordinary-bench scene JSON
- Coordinate system conversion preserving floor plane for TRR
- Multi-view image export
- Bootstrap mode for testing without Blender/Infinigen

## Scene Reconstruction

The `VLM-test/reconstruct/` module reconstructs 2D object positions from VLM-predicted spatial constraints:

1. **Constraint extraction**: QRR/TRR/FDR predictions → symbolic constraints
2. **Feasibility check**: cycle detection (QRR), arc intersection (TRR), connectivity analysis
3. **Numerical optimization**: gradient-based solver with multi-restart
4. **Evaluation**: CSR (constraint satisfaction rate), Kendall τ, NRMS, K_geom (geometric modality count)

FDR rankings are decomposed into equivalent shared-anchor QRR pairwise constraints for the solver.

## Qwen3-VL-32B Training

`Qwen/Qwen3-VL-32B` training entry point is documented in [training/README_qwen3vl32b.md](/Users/tsyq/code/ordinary-bench/training/README_qwen3vl32b.md).

Recommended workflow:

1. `bash training/setup_uv.sh`
2. `bash training/prepare_data.sh --data-dir ./data-gen/output`
3. `bash training/run_grpo.sh --gpus 8`

## License

MIT
