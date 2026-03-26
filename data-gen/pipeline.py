"""
管线：Blender 子进程编排 + 输出整理。

每个 split 分两个阶段：
  1. render_split()   — 调用 Blender 渲染场景
  2. organize_split() — 复制图像、保存场景 JSON、构建 split 索引
"""

import json
import logging
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# Blender 脚本和资产与本文件同目录
BLENDER_DIR = Path(__file__).resolve().parent / "blender"
RENDER_SCRIPT = BLENDER_DIR / "render_multiview.py"
ASSETS_DIR = BLENDER_DIR / "assets"


def _is_windows_blender(blender_path: str) -> bool:
  return blender_path.endswith('.exe')


def _wsl_to_win(path: Path) -> str:
  """将 /mnt/d/... 转换为 D:/...，供 Windows 版 Blender 使用。"""
  s = str(path)
  # /mnt/d/rest -> D:/rest（驱动器字母在索引 5，路径从索引 7 开始）
  if s.startswith('/mnt/') and len(s) > 6 and s[6] == '/':
    drive = s[5].upper()
    return f"{drive}:/{s[7:]}"
  return s


def render_split(split_name: str, split_cfg: dict, cfg: dict) -> Path:
  """
  通过 Blender 子进程渲染一个 split 的所有场景。

  Returns:
      渲染输出目录路径（始终为 WSL 可读路径）。
  """
  blender = cfg["blender"]["executable"]
  rendering = cfg["rendering"]
  objects = cfg["objects"]

  min_obj = split_cfg.get("min_objects", objects["min_count"])
  max_obj = split_cfg.get("max_objects", objects["max_count"])
  n_scenes = split_cfg["n_scenes"]
  start_idx = split_cfg.get("start_idx", 0)

  # 根据 Windows 版或原生 Blender 决定输出路径
  if _is_windows_blender(blender):
    blender_output = f"D:/render_{split_name}"
    render_output = Path(f"/mnt/d/render_{split_name}")
  else:
    output_dir = Path(cfg["output"]["dir"])
    render_output = output_dir / "render_temp" / split_name
    blender_output = str(render_output)

  render_output.mkdir(parents=True, exist_ok=True)

  # 构建资产路径——Windows 版 Blender 需要 Windows 格式路径
  if _is_windows_blender(blender):
    base_scene = _wsl_to_win(ASSETS_DIR / "base_scene_v5.blend")
    properties = _wsl_to_win(ASSETS_DIR / "properties.json")
    shape_dir = _wsl_to_win(ASSETS_DIR / "shapes_v5")
    material_dir = _wsl_to_win(ASSETS_DIR / "materials_v5")
    script = _wsl_to_win(RENDER_SCRIPT)
  else:
    base_scene = str(ASSETS_DIR / "base_scene_v5.blend")
    properties = str(ASSETS_DIR / "properties.json")
    shape_dir = str(ASSETS_DIR / "shapes_v5")
    material_dir = str(ASSETS_DIR / "materials_v5")
    script = str(RENDER_SCRIPT)

  cmd = [
    blender,
    "--background",
    "--python", script,
    "--",
    "--base_scene_blendfile", base_scene,
    "--properties_json", properties,
    "--shape_dir", shape_dir,
    "--material_dir", material_dir,
    "--output_dir", blender_output,
    "--split", split_cfg.get("split_prefix", split_name),
    "--num_images", str(n_scenes),
    "--min_objects", str(min_obj),
    "--max_objects", str(max_obj),
    "--min_dist", str(objects.get("min_dist", 0.25)),
    "--margin", str(objects.get("margin", 0.4)),
    "--n_views", str(rendering["n_views"]),
    "--camera_distance", str(rendering["camera_distance"]),
    "--elevation", str(rendering["elevation"]),
    "--azimuth_start", str(rendering.get("azimuth_start", 45.0)),
    "--render_top_view", "1" if rendering.get("render_top_view", False) else "0",
    "--top_view_padding", str(rendering.get("top_view_padding", 2.5)),
    "--width", str(rendering["width"]),
    "--height", str(rendering["height"]),
    "--render_num_samples", str(rendering["samples"]),
    "--start_idx", str(start_idx),
    "--seed", str(cfg["output"].get("seed", 42)),
  ]

  if cfg["blender"].get("use_gpu", False):
    cmd.extend(["--use_gpu", "1"])

  logger.info(f"Rendering {n_scenes} scenes for split '{split_name}' ...")
  logger.info(f"  Objects: {min_obj}-{max_obj}")
  logger.info(f"  Output: {blender_output}")

  try:
    result = subprocess.run(
      cmd,
      capture_output=True,
      text=True,
      timeout=max(3600, n_scenes * 300),
    )
    if result.returncode != 0:
      logger.error(f"Blender stderr:\n{result.stderr[-2000:]}")
      raise RuntimeError(f"Render failed for split '{split_name}'")
    logger.info(f"Render completed for '{split_name}'")
  except subprocess.TimeoutExpired:
    logger.error(f"Render timed out for split '{split_name}'")
    raise

  return render_output


def organize_split(
  split_name: str,
  render_output: Path,
  output_dir: Path,
  n_views: int,
  render_top_view: bool = False,
  effective_split: str = None,
) -> list:
  """
  将 Blender 输出的图像复制到最终目录结构中。
  保存每个场景的 JSON 并构建 split 索引。

  Returns:
      split 条目字典列表。
  """
  if effective_split is None:
    effective_split = split_name
  scenes_file = render_output / f"{effective_split}_scenes.json"
  if not scenes_file.exists():
    logger.error(f"Scenes file not found: {scenes_file}")
    return []

  with open(scenes_file) as f:
    scenes_data = json.load(f)

  split_entries = []

  for scene in scenes_data.get("scenes", []):
    scene_id = scene.get("scene_id", "")

    # 复制多视角图像
    src_mv = render_output / "multi_view" / scene_id
    dst_mv = output_dir / "images" / "multi_view" / scene_id
    if src_mv.exists():
      if dst_mv.exists():
        shutil.rmtree(dst_mv)
      shutil.copytree(src_mv, dst_mv)

    # 复制单视角图像
    src_sv = render_output / "single_view" / f"{scene_id}.png"
    dst_sv = output_dir / "images" / "single_view" / f"{scene_id}.png"
    if src_sv.exists():
      shutil.copy2(src_sv, dst_sv)

    # 复制俯视角图像
    src_tv = render_output / "top_view" / f"{scene_id}.png"
    dst_tv = output_dir / "images" / "top_view" / f"{scene_id}.png"
    if src_tv.exists():
      shutil.copy2(src_tv, dst_tv)

    # 保存场景元数据（原始场景数据，不含约束）
    scene_file = output_dir / "scenes" / f"{scene_id}.json"
    with open(scene_file, 'w') as f:
      json.dump(scene, f, indent=2)

    # 构建索引条目
    entry = {
      "scene_id": scene_id,
      "single_view_image": f"images/single_view/{scene_id}.png",
      "multi_view_images": [
        f"images/multi_view/{scene_id}/view_{i}.png"
        for i in range(n_views)
      ],
      "top_view_image": f"images/top_view/{scene_id}.png" if src_tv.exists() else None,
      "scene_path": f"scenes/{scene_id}.json",
      "n_objects": scene.get("n_objects", len(scene.get("objects", []))),
      "split": split_name,
    }
    split_entries.append(entry)

  # 清理临时渲染目录
  try:
    shutil.rmtree(render_output)
    logger.info(f"Cleaned up temp dir: {render_output}")
  except Exception as e:
    logger.warning(f"Failed to cleanup {render_output}: {e}")

  return split_entries


def build_split(split_name: str, split_cfg: dict, cfg: dict) -> dict:
  """
  渲染并整理一个 split，返回统计字典。
  """
  output_dir = Path(cfg["output"]["dir"])
  n_views = cfg["rendering"]["n_views"]
  render_top_view = cfg["rendering"].get("render_top_view", False)
  start_idx = split_cfg.get("start_idx", 0)
  effective_split = split_cfg.get("split_prefix", split_name)

  render_output = render_split(split_name, split_cfg, cfg)
  entries = organize_split(split_name, render_output, output_dir, n_views,
                           render_top_view=render_top_view,
                           effective_split=effective_split)

  # 保存 split 索引（增量模式时与已有索引合并）
  split_file = output_dir / "splits" / f"{split_name}.json"
  if start_idx > 0 and split_file.exists():
    with open(split_file) as f:
      existing = json.load(f)
    new_ids = {e["scene_id"] for e in entries}
    merged = [e for e in existing if e["scene_id"] not in new_ids] + entries
    merged.sort(key=lambda e: e["scene_id"])
    entries = merged
  with open(split_file, 'w') as f:
    json.dump(entries, f, indent=2)
  logger.info(f"Saved split index: {split_file} ({len(entries)} scenes)")

  return {
    "n_scenes": len(entries),
    "n_single_view_images": len(entries),
    "n_multi_view_images": len(entries) * n_views,
    "n_top_view_images": len(entries) if render_top_view else 0,
  }


def save_dataset_info(cfg: dict, all_stats: dict):
  """写入 dataset_info.json 汇总文件。"""
  output_dir = Path(cfg["output"]["dir"])
  rendering = cfg["rendering"]

  info = {
    "name": "ORDINARY-BENCH Dataset",
    "version": "1.0",
    "created": datetime.now().isoformat(),
    "config": {
      "n_views": rendering["n_views"],
      "image_size": [rendering["width"], rendering["height"]],
      "camera_distance": rendering["camera_distance"],
      "elevation": rendering["elevation"],
      "render_top_view": rendering.get("render_top_view", False),
    },
    "splits": {
      name: {
        "n_scenes": scfg["n_scenes"],
        "min_objects": scfg.get("min_objects", cfg["objects"]["min_count"]),
        "max_objects": scfg.get("max_objects", cfg["objects"]["max_count"]),
      }
      for name, scfg in cfg["splits"].items()
    },
    "statistics": all_stats,
    "total_scenes": sum(s["n_scenes"] for s in all_stats.values()),
    "total_images": sum(
      s["n_single_view_images"] + s["n_multi_view_images"] + s.get("n_top_view_images", 0)
      for s in all_stats.values()
    ),
  }

  info_file = output_dir / "dataset_info.json"
  with open(info_file, 'w') as f:
    json.dump(info, f, indent=2)
  logger.info(f"Saved dataset info: {info_file}")
