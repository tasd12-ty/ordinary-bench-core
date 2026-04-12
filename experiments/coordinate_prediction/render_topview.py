#!/usr/bin/env python3
"""补渲俯视图（top_view）辅助脚本。

从已有的场景 JSON 中读取物体位置和属性，
通过 data-gen 管线为指定场景补渲正交俯视图。

前提条件：
  - Blender 已安装且可执行
  - data-gen 目录中有 assets（base_scene_v5.blend, shapes_v5, etc.）

用法:
    # 为 test-data 的前 28 个场景补渲 top_view（每 split 4 个）
    python render_topview.py \
        --scenes-dir ../../datasets/test-data/scenes \
        --output-dir ../../datasets/test-data/images/top_view \
        --max-per-split 4

    # 只渲染 n04 split
    python render_topview.py \
        --scenes-dir ../../datasets/test-data/scenes \
        --output-dir ../../datasets/test-data/images/top_view \
        --split n04

注意:
    此脚本需要 Blender 重新生成场景后渲染俯视图。
    场景生成是种子确定性的（seed + start_idx），相同参数会产生相同场景。
    如果你的 data-gen 已经有 render_top_view=True 的输出，
    可以直接���制 data-gen/output/images/top_view/ 中的图片。
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_GEN_DIR = PROJECT_ROOT / "data-gen"
PIPELINE_SCRIPT = DATA_GEN_DIR / "pipeline.py"

# data-gen 默认配置中的渲染参数
DEFAULT_RENDERING = {
    "width": 480,
    "height": 320,
    "samples": 256,
    "n_views": 4,
    "camera_distance": 12.0,
    "elevation": 30.0,
    "azimuth_start": 45.0,
    "render_top_view": True,
    "top_view_padding": 0.35,
}


def _group_by_split(scene_ids: list[str]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for sid in scene_ids:
        split = sid.rsplit("_", 1)[0]
        groups.setdefault(split, []).append(sid)
    return groups


def main():
    parser = argparse.ArgumentParser(description="Render top-view images for existing scenes")
    parser.add_argument("--scenes-dir", required=True, help="Directory with scene JSON files")
    parser.add_argument("--output-dir", required=True, help="Output directory for top_view images")
    parser.add_argument("--split", default=None, help="Filter by split prefix (e.g., n04)")
    parser.add_argument("--max-per-split", type=int, default=4, help="Max scenes per split")
    parser.add_argument("--blender", default="blender", help="Blender executable path")
    parser.add_argument("--data-gen-output", default=None,
                        help="Existing data-gen output dir to copy top_view from (skip re-render)")
    args = parser.parse_args()

    scenes_dir = Path(args.scenes_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover scenes
    scene_files = sorted(scenes_dir.glob("*.json"))
    if args.split:
        scene_files = [f for f in scene_files if f.stem.startswith(args.split)]

    scene_ids = [f.stem for f in scene_files]
    if not scene_ids:
        logger.error("No scene files found in %s", scenes_dir)
        return

    # Limit per split
    groups = _group_by_split(scene_ids)
    selected = []
    for split_name in sorted(groups.keys()):
        split_scenes = groups[split_name][:args.max_per_split]
        selected.extend(split_scenes)

    logger.info("Selected %d scenes across %d splits", len(selected), len(groups))

    # Option 1: Copy from existing data-gen output
    if args.data_gen_output:
        src_dir = Path(args.data_gen_output) / "images" / "top_view"
        if not src_dir.exists():
            logger.error("Source top_view dir not found: %s", src_dir)
            return
        copied = 0
        for sid in selected:
            src = src_dir / f"{sid}.png"
            if src.exists():
                shutil.copy2(src, output_dir / f"{sid}.png")
                copied += 1
            else:
                logger.warning("  Missing: %s", src)
        logger.info("Copied %d/%d top_view images", copied, len(selected))
        return

    # Option 2: Re-render using data-gen pipeline
    # Check prerequisites
    if not DATA_GEN_DIR.exists():
        logger.error("data-gen directory not found: %s", DATA_GEN_DIR)
        logger.info("Please ensure the data-gen directory exists with Blender assets.")
        return

    # Verify Blender
    try:
        subprocess.run([args.blender, "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.error("Blender not found at '%s'. Install Blender or specify --blender path.", args.blender)
        return

    logger.info("Re-rendering requires running the full data-gen pipeline with render_top_view=True.")
    logger.info("For each split, this will regenerate scenes and render all views + top_view.")
    logger.info("")
    logger.info("Recommended approach:")
    logger.info("  cd %s", DATA_GEN_DIR)
    logger.info("  # Edit config.toml: set render_top_view = true")
    logger.info("  python generate.py --preset test")
    logger.info("  # Then copy the top_view images:")
    logger.info("  cp output/images/top_view/*.png %s/", output_dir)
    logger.info("")
    logger.info("Or if you already have a data-gen output with top_view:")
    logger.info("  python %s --data-gen-output /path/to/data-gen/output \\", __file__)
    logger.info("      --scenes-dir %s --output-dir %s", scenes_dir, output_dir)

    # List what's needed
    logger.info("\nScenes needing top_view (%d):", len(selected))
    for split_name, split_scenes in sorted(groups.items()):
        subset = [s for s in split_scenes[:args.max_per_split]]
        logger.info("  %s: %s", split_name, ", ".join(subset))


if __name__ == "__main__":
    main()
