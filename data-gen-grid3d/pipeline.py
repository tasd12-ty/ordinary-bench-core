"""
流水线：3D 网格模式的 Blender 子进程编排与输出整理。

每个 split 分两阶段处理：
  1. render_split()   — 调用 Blender 渲染 3D 网格场景
  2. organize_split() — 复制图像、保存场景 JSON、构建 split 索引
"""

import json
import logging
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

BLENDER_DIR = Path(__file__).resolve().parent / "blender"
RENDER_SCRIPT = BLENDER_DIR / "render_grid3d.py"
ASSETS_DIR = Path(__file__).resolve().parent.parent / "data-gen" / "blender" / "assets"

VIEW_NAMES = ["top", "bottom", "front", "back", "left", "right"]


def _is_windows_blender(blender_path: str) -> bool:
  return blender_path.endswith('.exe')


def _wsl_to_win(path: Path) -> str:
  s = str(path)
  if s.startswith('/mnt/') and len(s) > 6 and s[6] == '/':
    drive = s[5].upper()
    return f"{drive}:/{s[7:]}"
  return s


def render_split(split_name: str, split_cfg: dict, cfg: dict) -> Path:
  """通过 Blender 子进程渲染一个 split 的场景。"""
  blender = cfg["blender"]["executable"]
  rendering = cfg["rendering"]
  objects = cfg["objects"]
  grid = cfg.get("grid", {})

  min_obj = split_cfg.get("min_objects", objects["min_count"])
  max_obj = split_cfg.get("max_objects", objects["max_count"])
  n_scenes = split_cfg["n_scenes"]
  start_idx = split_cfg.get("start_idx", 0)

  if _is_windows_blender(blender):
    blender_output = f"D:/render_grid3d_{split_name}"
    render_output = Path(f"/mnt/d/render_grid3d_{split_name}")
  else:
    output_dir = Path(cfg["output"]["dir"])
    render_output = output_dir / "render_temp" / split_name
    blender_output = str(render_output)

  render_output.mkdir(parents=True, exist_ok=True)

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
    "--width", str(rendering["width"]),
    "--height", str(rendering["height"]),
    "--render_num_samples", str(rendering["samples"]),
    "--start_idx", str(start_idx),
    "--seed", str(cfg["output"].get("seed", 42)),
    # 3D 网格参数
    "--grid_rows", str(grid.get("rows", 4)),
    "--grid_cols", str(grid.get("cols", 4)),
    "--grid_layers", str(grid.get("layers", 4)),
    "--cell_size", str(grid.get("cell_size", 1.5)),
    "--grid_line_width", str(grid.get("line_width", 0.02)),
    "--grid_labels", "1" if grid.get("labels", False) else "0",
    "--ortho_scale", str(rendering.get("ortho_scale", 8.0)),
  ]

  if cfg["blender"].get("use_gpu", False):
    cmd.extend(["--use_gpu", "1"])

  logger.info(f"Rendering {n_scenes} 3D grid scenes for split '{split_name}' ...")
  logger.info(f"  Objects: {min_obj}-{max_obj}, "
              f"Grid: {grid.get('rows', 4)}x{grid.get('cols', 4)}x{grid.get('layers', 4)}")
  logger.info(f"  Output: {blender_output}")

  try:
    result = subprocess.run(
      cmd,
      capture_output=True,
      text=True,
      timeout=max(3600, n_scenes * 600),
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
  effective_split: str = None,
) -> list:
  """将 Blender 输出的图像复制到最终目录结构。"""
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

    # 复制 6 个视角图像
    view_images = {}
    src_scene_dir = render_output / "scenes_render" / scene_id
    for view_name in VIEW_NAMES:
      src_img = src_scene_dir / view_name / f"{scene_id}.png"
      dst_dir = output_dir / "images" / view_name
      dst_dir.mkdir(parents=True, exist_ok=True)
      dst_img = dst_dir / f"{scene_id}.png"
      if src_img.exists():
        shutil.copy2(src_img, dst_img)
        view_images[view_name] = f"images/{view_name}/{scene_id}.png"

    # 保存场景元数据
    scene_file = output_dir / "scenes" / f"{scene_id}.json"
    scene_file.parent.mkdir(parents=True, exist_ok=True)
    with open(scene_file, 'w') as f:
      json.dump(scene, f, indent=2)

    entry = {
      "scene_id": scene_id,
      "view_images": view_images,
      "scene_path": f"scenes/{scene_id}.json",
      "n_objects": scene.get("n_objects", len(scene.get("objects", []))),
      "placement_mode": "grid3d",
      "split": split_name,
    }
    split_entries.append(entry)

  # 清理临时目录
  try:
    shutil.rmtree(render_output)
    logger.info(f"Cleaned up temp dir: {render_output}")
  except Exception as e:
    logger.warning(f"Failed to cleanup {render_output}: {e}")

  return split_entries


def build_split(split_name: str, split_cfg: dict, cfg: dict) -> dict:
  """渲染并整理一个 split。"""
  output_dir = Path(cfg["output"]["dir"])
  start_idx = split_cfg.get("start_idx", 0)
  effective_split = split_cfg.get("split_prefix", split_name)

  render_output = render_split(split_name, split_cfg, cfg)
  entries = organize_split(split_name, render_output, output_dir,
                           effective_split=effective_split)

  split_file = output_dir / "splits" / f"{split_name}.json"
  split_file.parent.mkdir(parents=True, exist_ok=True)
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
    "n_images": len(entries) * 6,  # 每个场景 6 个视角
  }


def save_dataset_info(cfg: dict, all_stats: dict):
  """写入 dataset_info.json 摘要文件。"""
  output_dir = Path(cfg["output"]["dir"])
  rendering = cfg["rendering"]
  grid = cfg.get("grid", {})

  info = {
    "name": "ORDINARY-BENCH 3D Grid Dataset",
    "version": "1.0",
    "created": datetime.now().isoformat(),
    "placement_mode": "grid3d",
    "config": {
      "n_views": 6,
      "view_names": VIEW_NAMES,
      "projection": "orthographic",
      "image_size": [rendering["width"], rendering["height"]],
      "ortho_scale": rendering.get("ortho_scale", 8.0),
      "grid": {
        "rows": grid.get("rows", 4),
        "cols": grid.get("cols", 4),
        "layers": grid.get("layers", 4),
        "cell_size": grid.get("cell_size", 1.5),
        "labels": grid.get("labels", False),
      },
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
    "total_images": sum(s["n_images"] for s in all_stats.values()),
  }

  info_file = output_dir / "dataset_info.json"
  with open(info_file, 'w') as f:
    json.dump(info, f, indent=2)
  logger.info(f"Saved dataset info: {info_file}")
