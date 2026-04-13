"""
渲染 5 视角图像：4 侧视角 + 1 俯视角。

每个子场景调用 render_subset_blender.py 两次:
  1. --multi_view: 4 侧视角 (azimuth 45°/135°/225°/315°, elevation 30°)
  2. 单视角 --elevation 90: 俯视角 (正上方向下看)

输出:
  images/multi_view/{subset_id}/view_0.png ~ view_3.png   (4 侧视角)
  images/top_view/{subset_id}.png                          (1 俯视角)

用法:
    cd experiments/variable_subset_ablation

    python render_all_views.py \
        --manifest output/manifest.json \
        --output-dir output \
        --blender /Applications/Blender.app/Contents/MacOS/Blender \
        --workers 4 --samples 64 --use-gpu

    # 只渲染前 5 个子场景（调试）
    python render_all_views.py \
        --manifest output/manifest.json \
        --output-dir output \
        --blender blender \
        --limit 5

    # 跳过已渲染的（默认开启），强制重渲：
    python render_all_views.py \
        --manifest output/manifest.json \
        --output-dir output \
        --blender blender \
        --no-skip-existing

依赖:
    - Blender (通过 --blender 指定路径)
    - experiments/subset_ablation/render_subset_blender.py (不修改，直接调用)
"""

import argparse
import json
import logging
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# 项目路径
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
BLENDER_DIR = PROJECT_ROOT / "data-gen" / "blender"
ASSETS_DIR = BLENDER_DIR / "assets"
RENDER_SCRIPT = SCRIPT_DIR.parent / "subset_ablation" / "render_subset_blender.py"


def render_one(
    subset_scene_json: str,
    output_path: str,
    blender: str,
    samples: int,
    use_gpu: bool,
    width: int,
    height: int,
    multi_view: bool = False,
    elevation: float = 30.0,
    camera_distance: float = 12.0,
) -> dict:
    """调用 Blender 渲染单个子场景的一组视角。"""
    cmd = [
        blender,
        "--background",
        "--python", str(RENDER_SCRIPT),
        "--",
        "--scene_json", subset_scene_json,
        "--base_scene", str(ASSETS_DIR / "base_scene_v5.blend"),
        "--properties_json", str(ASSETS_DIR / "properties.json"),
        "--shape_dir", str(ASSETS_DIR / "shapes_v5"),
        "--material_dir", str(ASSETS_DIR / "materials_v5"),
        "--blender_utils_dir", str(BLENDER_DIR),
        "--samples", str(samples),
        "--width", str(width),
        "--height", str(height),
        "--elevation", str(elevation),
        "--camera_distance", str(camera_distance),
    ]

    if multi_view:
        cmd.extend(["--multi_view", "--output_dir", output_path])
    else:
        cmd.extend(["--output_image", output_path])

    if use_gpu:
        cmd.extend(["--use_gpu", "1"])

    timeout = 600 if multi_view else 300

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if result.returncode != 0:
            return {
                "status": "error",
                "scene_json": subset_scene_json,
                "stderr": result.stderr[-500:] if result.stderr else "",
            }
        return {"status": "ok", "scene_json": subset_scene_json}
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "scene_json": subset_scene_json}
    except Exception as e:
        return {"status": "exception", "scene_json": subset_scene_json, "error": str(e)}


def render_five_views(
    subset_id: str,
    scene_json_path: str,
    output_dir: Path,
    blender: str,
    samples: int,
    use_gpu: bool,
    width: int,
    height: int,
    skip_existing: bool,
    camera_distance: float,
) -> dict:
    """
    为一个子场景渲染全部 5 视角:
      - 4 侧视角 (multi_view, elevation=30°)
      - 1 俯视角 (top_view, elevation=90°)
    """
    mv_dir = output_dir / "images" / "multi_view" / subset_id
    top_path = output_dir / "images" / "top_view" / f"{subset_id}.png"

    results = {"subset_id": subset_id, "multi_view": None, "top_view": None}

    # --- 4 侧视角 ---
    mv_done = skip_existing and all(
        (mv_dir / f"view_{i}.png").exists() for i in range(4)
    )
    if mv_done:
        results["multi_view"] = "skipped"
    else:
        mv_dir.mkdir(parents=True, exist_ok=True)
        r = render_one(
            scene_json_path, str(mv_dir), blender, samples, use_gpu,
            width, height, multi_view=True, elevation=30.0,
            camera_distance=camera_distance,
        )
        results["multi_view"] = r["status"]

    # --- 1 俯视角 ---
    top_done = skip_existing and top_path.exists()
    if top_done:
        results["top_view"] = "skipped"
    else:
        top_path.parent.mkdir(parents=True, exist_ok=True)
        r = render_one(
            scene_json_path, str(top_path), blender, samples, use_gpu,
            width, height, multi_view=False, elevation=90.0,
            camera_distance=camera_distance,
        )
        results["top_view"] = r["status"]

    return results


def main():
    parser = argparse.ArgumentParser(
        description="渲染 5 视角图像：4 侧视角 + 1 俯视角"
    )
    parser.add_argument("--manifest", required=True, help="manifest.json 路径")
    parser.add_argument("--output-dir", default="output", help="输出根目录")
    parser.add_argument("--blender", default="blender", help="Blender 可执行文件路径")
    parser.add_argument("--workers", type=int, default=1, help="并行 worker 数")
    parser.add_argument("--samples", type=int, default=64, help="Cycles 采样数")
    parser.add_argument("--use-gpu", action="store_true", help="使用 GPU 渲染")
    parser.add_argument("--width", type=int, default=480, help="渲染宽度")
    parser.add_argument("--height", type=int, default=320, help="渲染高度")
    parser.add_argument("--camera-distance", type=float, default=14.0,
                        help="相机距离 (N=20 默认 14.0，比标准 12.0 远)")
    parser.add_argument("--skip-existing", action="store_true", default=True,
                        help="跳过已渲染的图片 (默认开启)")
    parser.add_argument("--no-skip-existing", dest="skip_existing",
                        action="store_false", help="强制重新渲染")
    parser.add_argument("--limit", type=int, default=None,
                        help="只渲染前 N 个子场景 (调试用)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    scenes_dir = output_dir / "scenes"

    with open(args.manifest) as f:
        manifest = json.load(f)

    # 收集所有子场景
    tasks = []
    for parent_id, parent_data in manifest["parent_scenes"].items():
        for subset_info in parent_data["subsets"]:
            subset_id = subset_info["subset_id"]
            scene_json = str(scenes_dir / f"{subset_id}.json")
            tasks.append((subset_id, scene_json))

    if args.limit:
        tasks = tasks[:args.limit]

    total = len(tasks)
    logger.info(f"Total subsets to render: {total} (5 views each)")
    logger.info(f"Blender: {args.blender}, samples={args.samples}, "
                f"GPU={args.use_gpu}, workers={args.workers}")

    if total == 0:
        logger.info("Nothing to render.")
        return

    ok = 0
    errors = 0
    skipped = 0

    def _process(task):
        subset_id, scene_json = task
        return render_five_views(
            subset_id, scene_json, output_dir, args.blender,
            args.samples, args.use_gpu, args.width, args.height,
            args.skip_existing, args.camera_distance,
        )

    if args.workers <= 1:
        for i, task in enumerate(tasks):
            result = _process(task)
            mv = result["multi_view"]
            tv = result["top_view"]
            if mv == "ok" or tv == "ok":
                ok += 1
            elif mv == "skipped" and tv == "skipped":
                skipped += 1
            else:
                errors += 1
                logger.error(f"Failed: {result}")
            if (i + 1) % 10 == 0 or (i + 1) == total:
                logger.info(f"Progress: {i+1}/{total} "
                            f"(ok={ok}, skipped={skipped}, errors={errors})")
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(_process, t): t for t in tasks}
            for i, future in enumerate(as_completed(futures)):
                result = future.result()
                mv = result["multi_view"]
                tv = result["top_view"]
                if mv == "ok" or tv == "ok":
                    ok += 1
                elif mv == "skipped" and tv == "skipped":
                    skipped += 1
                else:
                    errors += 1
                    logger.error(f"Failed: {result}")
                if (i + 1) % 10 == 0 or (i + 1) == total:
                    logger.info(f"Progress: {i+1}/{total} "
                                f"(ok={ok}, skipped={skipped}, errors={errors})")

    logger.info(f"Done: {ok} rendered, {skipped} skipped, {errors} errors")


if __name__ == "__main__":
    main()
