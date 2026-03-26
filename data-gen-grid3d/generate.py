#!/usr/bin/env python3
"""
ordinary-bench 3D 网格数据生成入口。

在可见的 4x4x4 3D 网格中摆放物体并生成场景，
从 6 个正交视角（top、bottom、front、back、left、right）渲染。

用法：
    python generate.py                            # 使用 config.toml
    python generate.py --preset test              # 每个 split 生成 1 个场景，快速测试
    python generate.py --preset test --labels     # 带网格标签
    python generate.py --preset test --dry-run    # 仅打印配置
"""

import argparse
import json
import logging
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

try:
  import tomllib
except ModuleNotFoundError:
  try:
    import tomli as tomllib
  except ModuleNotFoundError:
    tomllib = None

import pipeline

logging.basicConfig(
  level=logging.INFO,
  format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

PRESETS = {
  "test": {
    "splits": {
      "g04": {"n_scenes": 1, "min_objects": 4, "max_objects": 4},
      "g08": {"n_scenes": 1, "min_objects": 8, "max_objects": 8},
      "g12": {"n_scenes": 1, "min_objects": 12, "max_objects": 12},
    },
    "rendering": {"samples": 64},
  },
}

DEFAULT_CONFIG = {
  "blender": {
    "executable": "blender",
    "use_gpu": False,
  },
  "rendering": {
    "width": 480,
    "height": 480,
    "samples": 256,
    "ortho_scale": 8.0,
  },
  "grid": {
    "rows": 4,
    "cols": 4,
    "layers": 4,
    "cell_size": 1.5,
    "line_width": 0.02,
    "labels": False,
  },
  "objects": {
    "min_count": 4,
    "max_count": 12,
  },
  "output": {
    "dir": "./output",
    "seed": 42,
  },
  "splits": {},
}


def deep_merge(base, override):
  """递归地将 override 合并到 base 中（override 的值优先）。"""
  result = dict(base)
  for k, v in override.items():
    if k in result and isinstance(result[k], dict) and isinstance(v, dict):
      result[k] = deep_merge(result[k], v)
    else:
      result[k] = v
  return result


def load_config(config_path, preset, cli_args):
  """加载配置：TOML 文件 -> 预设 -> 命令行覆盖。"""
  cfg = dict(DEFAULT_CONFIG)

  if config_path:
    p = Path(config_path)
    if not p.exists():
      logger.error(f"Config file not found: {p}")
      sys.exit(1)
    if tomllib is None:
      logger.error("TOML support requires Python 3.11+ or 'pip install tomli'")
      sys.exit(1)
    with open(p, "rb") as f:
      file_cfg = tomllib.load(f)
    cfg = deep_merge(cfg, file_cfg)

  if preset:
    if preset not in PRESETS:
      logger.error(f"Unknown preset: {preset}. Choose from: {list(PRESETS)}")
      sys.exit(1)
    cfg = deep_merge(cfg, PRESETS[preset])

  if cli_args.blender:
    cfg["blender"]["executable"] = cli_args.blender
  if cli_args.output_dir:
    cfg["output"]["dir"] = cli_args.output_dir
  if cli_args.gpu is not None:
    cfg["blender"]["use_gpu"] = cli_args.gpu
  if cli_args.labels:
    cfg["grid"]["labels"] = True

  return cfg


def create_directories(cfg):
  """创建输出目录结构，包含 6 个视角子目录。"""
  output = Path(cfg["output"]["dir"])
  for view in ["top", "bottom", "front", "back", "left", "right"]:
    (output / "images" / view).mkdir(parents=True, exist_ok=True)
  for sub in ["scenes", "splits"]:
    (output / sub).mkdir(parents=True, exist_ok=True)


def _run_split(args_tuple):
  split_name, split_cfg, cfg = args_tuple
  return split_name, pipeline.build_split(split_name, split_cfg, cfg)


def main():
  parser = argparse.ArgumentParser(
    description="Generate ordinary-bench 3D grid dataset (4x4x4 grid, 6 ortho views)"
  )
  parser.add_argument("--config", "-c", default=None)
  parser.add_argument("--preset", "-p", choices=list(PRESETS.keys()), default=None)
  parser.add_argument("--blender", default=None)
  parser.add_argument("--output-dir", "-o", default=None)
  parser.add_argument("--gpu", action="store_true", default=None)
  parser.add_argument("--labels", action="store_true", default=False,
    help="Enable 3D grid cell labels in rendered images")
  parser.add_argument("--workers", "-w", type=int, default=1)
  parser.add_argument("--start-idx", type=int, default=0)
  parser.add_argument("--dry-run", action="store_true")

  args = parser.parse_args()

  config_path = args.config
  if config_path is None:
    default_toml = Path(__file__).resolve().parent / "config.toml"
    if default_toml.exists():
      config_path = str(default_toml)

  cfg = load_config(config_path, args.preset, args)

  if args.start_idx > 0:
    for split_cfg in cfg["splits"].values():
      if "start_idx" not in split_cfg:
        split_cfg["start_idx"] = args.start_idx

  if not cfg["splits"]:
    logger.error("No splits configured. Check config.toml or use --preset.")
    sys.exit(1)

  if args.dry_run:
    print(json.dumps(cfg, indent=2))
    return

  blender = cfg["blender"]["executable"]
  if blender == "blender":
    logger.warning("Using default 'blender' command. "
                   "Set blender.executable in config.toml or use --blender.")

  n_splits = len(cfg["splits"])
  total_scenes = sum(s["n_scenes"] for s in cfg["splits"].values())
  workers = min(args.workers, n_splits)
  grid = cfg.get("grid", {})

  logger.info("=" * 50)
  logger.info("ORDINARY-BENCH 3D Grid Data Generation")
  logger.info("=" * 50)
  logger.info(f"Output:  {cfg['output']['dir']}")
  logger.info(f"Blender: {blender}")
  logger.info(f"Grid:    {grid.get('rows', 4)}x{grid.get('cols', 4)}x{grid.get('layers', 4)}, "
              f"cell_size={grid.get('cell_size', 1.5)}, "
              f"labels={'on' if grid.get('labels') else 'off'}")
  logger.info(f"Views:   6 orthographic (top/bottom/front/back/left/right)")
  logger.info(f"Splits:  {n_splits} ({total_scenes} scenes total)")
  logger.info(f"Workers: {workers}")

  create_directories(cfg)

  all_stats = {}

  if workers <= 1:
    for split_name, split_cfg in cfg["splits"].items():
      logger.info(f"\n--- Split: {split_name} ---")
      stats = pipeline.build_split(split_name, split_cfg, cfg)
      all_stats[split_name] = stats
  else:
    tasks = [(name, scfg, cfg) for name, scfg in cfg["splits"].items()]
    with ProcessPoolExecutor(max_workers=workers) as pool:
      futures = {pool.submit(_run_split, t): t[0] for t in tasks}
      for future in as_completed(futures):
        name = futures[future]
        try:
          _, stats = future.result()
          all_stats[name] = stats
          logger.info(f"Split '{name}' done: {stats['n_scenes']} scenes")
        except Exception as e:
          logger.error(f"Split '{name}' failed: {e}")
          raise

  pipeline.save_dataset_info(cfg, all_stats)

  total_scenes = sum(s["n_scenes"] for s in all_stats.values())
  total_images = sum(s["n_images"] for s in all_stats.values())
  print(f"\nDone! {total_scenes} scenes, {total_images} images (6 views each)")
  print(f"Output: {cfg['output']['dir']}")


if __name__ == "__main__":
  main()
