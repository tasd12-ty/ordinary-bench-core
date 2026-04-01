"""
Step 2b: 渲染编排器。

读取 manifest，为每个子集启动 Blender 子进程渲染。
支持并行、增量渲染、成本控制。

用法:
    python render_subsets.py --manifest output/manifest.json --blender blender
    python render_subsets.py --manifest output/manifest.json --blender blender --workers 4 --samples 64
"""

import argparse
import json
import logging
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# 项目根目录定位
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
BLENDER_DIR = PROJECT_ROOT / "data-gen" / "blender"
ASSETS_DIR = BLENDER_DIR / "assets"
RENDER_SCRIPT = SCRIPT_DIR / "render_subset_blender.py"


def render_one_subset(
    subset_scene_json: str,
    output_image: str,
    blender: str,
    samples: int,
    use_gpu: bool,
    width: int = 480,
    height: int = 320,
) -> dict:
    """渲染单个子集，返回状态字典。"""
    cmd = [
        blender,
        "--background",
        "--python", str(RENDER_SCRIPT),
        "--",
        "--scene_json", subset_scene_json,
        "--output_image", output_image,
        "--base_scene", str(ASSETS_DIR / "base_scene_v5.blend"),
        "--properties_json", str(ASSETS_DIR / "properties.json"),
        "--shape_dir", str(ASSETS_DIR / "shapes_v5"),
        "--material_dir", str(ASSETS_DIR / "materials_v5"),
        "--blender_utils_dir", str(BLENDER_DIR),
        "--samples", str(samples),
        "--width", str(width),
        "--height", str(height),
    ]
    if use_gpu:
        cmd.extend(["--use_gpu", "1"])

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 分钟超时
        )
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


def main():
    parser = argparse.ArgumentParser(description="批量渲染子集图片")
    parser.add_argument("--manifest", required=True, help="manifest.json 路径")
    parser.add_argument("--output-dir", default="output", help="输出根目录")
    parser.add_argument("--blender", default="blender", help="Blender 可执行文件路径")
    parser.add_argument("--workers", type=int, default=1, help="并行 worker 数")
    parser.add_argument("--samples", type=int, default=64, help="Cycles 采样数")
    parser.add_argument("--use-gpu", action="store_true", help="使用 GPU 渲染")
    parser.add_argument("--width", type=int, default=480, help="渲染宽度")
    parser.add_argument("--height", type=int, default=320, help="渲染高度")
    parser.add_argument("--skip-existing", action="store_true", default=True,
                        help="跳过已渲染的图片 (默认开启)")
    parser.add_argument("--limit", type=int, default=None,
                        help="只渲染前 N 个子集 (调试用)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    images_dir = output_dir / "images" / "single_view"
    scenes_dir = output_dir / "scenes"
    images_dir.mkdir(parents=True, exist_ok=True)

    with open(args.manifest) as f:
        manifest = json.load(f)

    # 收集所有待渲染任务
    tasks = []
    skipped = 0
    for parent_id, parent_data in manifest["parent_scenes"].items():
        for subset_info in parent_data["subsets"]:
            subset_id = subset_info["subset_id"]
            scene_json = str(scenes_dir / f"{subset_id}.json")
            output_image = str(images_dir / f"{subset_id}.png")

            if args.skip_existing and Path(output_image).exists():
                skipped += 1
                continue

            tasks.append((scene_json, output_image))

    if args.limit:
        tasks = tasks[:args.limit]

    total = len(tasks)
    logger.info(f"Tasks: {total} to render, {skipped} skipped (existing)")

    if total == 0:
        logger.info("Nothing to render.")
        return

    # 渲染
    ok = 0
    errors = 0

    if args.workers <= 1:
        for i, (scene_json, output_image) in enumerate(tasks):
            result = render_one_subset(
                scene_json, output_image, args.blender, args.samples, args.use_gpu,
                args.width, args.height
            )
            if result["status"] == "ok":
                ok += 1
            else:
                errors += 1
                logger.error(f"Failed: {result}")
            if (i + 1) % 10 == 0 or (i + 1) == total:
                logger.info(f"Progress: {i+1}/{total} (ok={ok}, errors={errors})")
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    render_one_subset,
                    scene_json, output_image, args.blender, args.samples, args.use_gpu,
                    args.width, args.height
                ): subset_id
                for scene_json, output_image in tasks
                for subset_id in [Path(scene_json).stem]
            }
            for i, future in enumerate(as_completed(futures)):
                result = future.result()
                if result["status"] == "ok":
                    ok += 1
                else:
                    errors += 1
                    logger.error(f"Failed: {result}")
                if (i + 1) % 10 == 0 or (i + 1) == total:
                    logger.info(f"Progress: {i+1}/{total} (ok={ok}, errors={errors})")

    logger.info(f"Done: {ok} ok, {errors} errors out of {total}")


if __name__ == "__main__":
    main()
