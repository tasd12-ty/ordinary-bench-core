# data-gen-infinigen

This directory is a prototype backend for generating ordinary-bench style scenes
with [Infinigen](https://infinigen.org/), while keeping the current
`VLM-test/` extraction and evaluation pipeline reusable.

## Goal

Use Infinigen-Indoors to create scenes that are:

- more realistic than the current CLEVR-style generator
- still visually clean and object-centric
- simple enough to support QRR / shared-anchor QRR / TRR / FDR probing

The prototype is intentionally conservative:

- single-room indoor scenes
- terrain disabled
- overhead camera by default
- adapter converts Infinigen metadata into the current ordinary-bench scene JSON
- first view is exported as `images/single_view/<scene_id>.png`
- all discovered camera views are exported as `images/multi_view/<scene_id>/view_i.png`

## Repository Layout

- `generate.py`
  - Builds and optionally runs Infinigen commands for a single-room indoor scene.
  - Can adapt the rendered Infinigen outputs into ordinary-bench scene JSON.
- `adapter.py`
  - Converts Infinigen `Objects_*.json` + `camview_*.npz/.json` into the current
    ordinary-bench object schema.
  - Discovers all `camera_*` bundles and emits benchmark-compatible `views[]`.
- `validate.py`
  - Runs a smoke test on a small mock Infinigen frame bundle and then checks that
    `VLM-test` can extract GT constraints from the converted scene.
- `bootstrap_from_datagen.py`
  - Replays an existing `data-gen/output` scene into a pseudo-Infinigen `frames/`
    bundle, so the adapter can be tested against real benchmark images without Blender.
- `inject_and_render_livingroom.py`
  - Opens an existing Infinigen indoor `.blend`, finds a desk/table, injects probe
    objects, and renders a single image.
- `render_center_table_multiview.py`
  - Opens an Infinigen living-room `.blend`, clears a center region, adds a large
    central table, places the 4-object `n04_000000`-style layout on top, and renders
    four benchmark-like views.
- `render_livingroom_hybrid_demo.py`
  - Blender-only fallback demo using local assets, useful when validating visual
    composition without depending on a full Infinigen run.
- `fixtures/mock_frame/`
  - Minimal mock files to validate the adapter without Blender / Infinigen.
  - These are intentionally synthetic and should be treated as smoke-test fixtures,
    not as realistic visual examples.
- `examples/`
  - Small checked-in examples only.
  - `examples/center_table_livingroom_v3/` is the current curated result for
    "Infinigen living room + center table + four benchmark objects + four views".
- `generated/`
  - Local large artifacts from real Infinigen runs.
  - Ignored by git.

## Environment Note

Running the real backend requires:

- Blender
- an Infinigen checkout
- Infinigen Python dependencies

The adapter, bootstrap path, and mock validation can still be exercised without
rendering.

## Recommended first run

Validate the adapter and end-to-end compatibility:

```bash
python3 data-gen-infinigen/validate.py
```

Preview the Infinigen commands that would be executed:

```bash
python3 data-gen-infinigen/generate.py --dry-run
```

Inspect the curated checked-in multi-view example:

```bash
open data-gen-infinigen/examples/center_table_livingroom_v3/view_0.png
```

If Blender is unavailable but you want a scene-like example instead of the
synthetic smoke-test fixture, replay an existing `data-gen` scene:

```bash
python3 data-gen-infinigen/bootstrap_from_datagen.py \
  --scene-id n04_000000 \
  --dest-root data-gen-infinigen/examples/replay-source/n04_000000

python3 data-gen-infinigen/generate.py \
  --skip-run \
  --source-root data-gen-infinigen/examples/replay-source/n04_000000 \
  --output-dir data-gen-infinigen/examples/replay-output \
  --room-type Replay
```

## Intended real run

Assuming Infinigen has been cloned at `/path/to/infinigen` and Blender works:

```bash
python3 data-gen-infinigen/generate.py \
  --infinigen-root /path/to/infinigen \
  --output-dir data-gen-infinigen/output \
  --seed 0 \
  --room-type DiningRoom
```

This will:

1. run Infinigen-Indoors coarse generation
2. render frames and metadata
3. preserve a native Infinigen record bundle
4. adapt the first camera/frame into ordinary-bench scene JSON
5. export any discovered camera views into benchmark single-view and multi-view image folders
6. write:
   - `data-gen-infinigen/output/scenes/<scene_id>.json`
   - `data-gen-infinigen/output/native/<scene_id>/manifest.json`
   - `data-gen-infinigen/output/native/<scene_id>/<camera_id>/Objects_*.json`
   - `data-gen-infinigen/output/native/<scene_id>/<camera_id>/camview_*.npz|json`
   - `data-gen-infinigen/output/images/single_view/<scene_id>.png` if found
   - `data-gen-infinigen/output/images/multi_view/<scene_id>/view_*.png` if found

## Mapping to ordinary-bench

The adapter produces the fields expected by [extraction.py](/Users/tsyq/code/ordinary-bench-core/VLM-test/extraction.py):

- `id`
- `shape`
- `size`
- `material`
- `color`
- `3d_coords`
- `pixel_coords`
- `rotation`

Important coordinate conversion:

- Infinigen camera metadata uses a CV-style world convention documented in
  `GroundTruthAnnotations.md`: `+X right, +Y down, +Z forward`.
- ordinary-bench expects `x/y` to be the horizontal plane and uses `[:2]` for TRR.
- The adapter converts world coordinates to ordinary-bench coordinates as:
  - `x_bench = x_world`
  - `y_bench = z_world`
  - `z_bench = -y_world`

This preserves the floor plane for the current TRR implementation.

## Dual Output Policy

The prototype now keeps both representations:

- **Native Infinigen record**
  - preserves Infinigen-style metadata files in `output/native/<scene_id>/`
  - keeps per-camera `Objects_*.json`, `camview_*.npz/.json`, and a local `manifest.json`
- **ordinary-bench export**
  - writes a simplified scene JSON in `output/scenes/<scene_id>.json`
  - writes `views[]` with per-view camera/object projections
  - keeps `source.native_*` pointers back to the native record

## Living-Room Prototypes

Two scripts are useful for scene-staging experiments beyond plain adaptation:

- `inject_and_render_livingroom.py`
  - best for "keep Infinigen furniture as-is, inject a few tabletop probes"
- `render_center_table_multiview.py`
  - best for "replace the center area with a deterministic benchmark-style table
    and render four controlled views"

Example command:

```bash
python3 data-gen-infinigen/render_center_table_multiview.py \
  --scene-blend data-gen-infinigen/generated/livingroom_seed7/coarse/scene.blend \
  --output-dir data-gen-infinigen/generated/livingroom_seed7/center_table_multiview_v3 \
  --save-blend data-gen-infinigen/generated/livingroom_seed7/scene_center_table_multiview_v3.blend \
  --engine BLENDER_EEVEE_NEXT \
  --samples 20 \
  --resolution-x 960 \
  --resolution-y 640
```

## Caveats

- The adapter currently selects up to `N` visible object instances using
  metadata-driven heuristics. It does not yet use segmentation masks to enforce
  full visibility.
- `shape`, `material`, and `color` are inferred from Infinigen names/tags/materials
  and are therefore approximate.
- The current prototype targets indoor scenes only.
