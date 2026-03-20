# Reconstruction Lab

This directory is intentionally isolated from the main benchmark pipeline.

Purpose:
- prototype reconstruction-oriented visualizations
- compare `ground-truth scene` vs `VLM belief world`
- keep all experimental code disposable

Current entry point:

```bash
python3 reconstruction-lab/storyboard_svg.py \
  --scene-json data-gen/output/scenes/n04_000006.json \
  --scene-result-json VLM-test/output/results/qwen--qwen3-vl-235b-a22b-thinking/scenes/n04_000006.json \
  --recon-json VLM-test/output/analysis/recon_qwen3-vl-235b-a22b.json \
  --questions-dir VLM-test/output/questions \
  --image-path data-gen/output/images/single_view/n04_000006.png \
  --scene-id n04_000006 \
  --output reconstruction-lab/output/qwen235_n04_000006_storyboard.svg
```

Output:
- a single SVG storyboard with
  - input RGB
  - ground-truth top-down map
  - aligned belief map
  - overlay / distortion map
  - compact audit sidebar
