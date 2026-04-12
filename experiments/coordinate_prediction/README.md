# Coordinate Prediction Experiment

Direct coordinate prediction: ask VLMs to estimate object positions from scene images.

## Motivation

Existing ORDINARY-BENCH evaluates spatial understanding indirectly:
- **Indirect**: Image → QRR/TRR/FDR relational Q&A → gradient-descent reconstruction → compare with GT

This experiment adds a **direct** path:
- **Direct**: Image → VLM predicts 2D coordinates → compare with GT

Same evaluation metrics (Kendall τ, NRMS, CSR) enable direct comparison.

## Three Image Conditions

| Mode | Input | Tests |
|------|-------|-------|
| `single` | 1 side-view image | Spatial layout from single perspective |
| `multi_view` | 4 side-view images | Multi-view spatial integration |
| `top_view` | 1 orthographic top-down | Direct spatial perception |

## Quick Start

```bash
cd experiments/coordinate_prediction

# Smoke test (no API needed)
python run_eval.py --job jobs/smoke.toml

# Real evaluation
python run_eval.py --job jobs/single_view.toml
python run_eval.py --job jobs/multi_view.toml
python run_eval.py --job jobs/top_view.toml
```

## Top-view Rendering

Top-view images need to be rendered separately (not included in test-data):

```bash
# If you have a data-gen output with top_view:
python render_topview.py \
    --data-gen-output ../../data-gen/output \
    --scenes-dir ../../datasets/test-data/scenes \
    --output-dir ../../datasets/test-data/images/top_view \
    --max-per-split 4

# Or render from scratch (requires Blender):
cd ../../data-gen
# Edit config.toml: set render_top_view = true
python generate.py --preset test
```

## Output

Results are saved to `output/results/{run_name}/`:
- `raw/{scene_id}/coord_0.json` — raw VLM response + predicted/GT positions + metrics
- `summary.json` — aggregate metrics (mean/median/std by split)

## Metrics

| Metric | Description |
|--------|-------------|
| Kendall τ | Pairwise distance rank correlation (primary) |
| NRMS | Procrustes-aligned normalized RMS |
| CSR_QRR | Fraction of GT distance comparisons satisfied |
| CSR_TRR | Fraction of GT clock-direction constraints satisfied |
