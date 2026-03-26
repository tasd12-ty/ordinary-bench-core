#!/usr/bin/env python3
"""
ordinary-bench-3d 数据生成入口。

生成物体处于不同高度的 3D 场景。

用法：
    python generate.py                            # 使用 config.toml
    python generate.py --preset test              # 每个 split 生成 1 个场景，快速测试
    python generate.py --config my.toml           # 使用自定义配置
    python generate.py --preset test --dry-run    # 仅打印配置，不执行
    python generate.py --workers 4                # 并行 Blender 进程数
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

import pipeline_3d as pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

PRESETS = {
    "test": {
        "splits": {
            "n04": {"n_scenes": 1, "min_objects": 4, "max_objects": 4},
            "n06": {"n_scenes": 1, "min_objects": 6, "max_objects": 6},
            "n08": {"n_scenes": 1, "min_objects": 8, "max_objects": 8},
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
        "height": 320,
        "samples": 256,
        "n_views": 4,
        "camera_distance": 14.0,
        "elevation": 35.0,
        "azimuth_start": 45.0,
        "render_top_view": True,
        "render_side_view": True,
        "top_view_padding": 0.35,
        "views_3d": {
            "oblique_front": [45.0, 35.0],
            "oblique_back": [225.0, 35.0],
            "side_low": [90.0, 15.0],
            "top": [0.0, 90.0],
        },
    },
    "placement": {
        "x_range": [-3.0, 3.0],
        "y_range": [-3.0, 3.0],
        "z_range": [0.0, 2.5],
        "z_distribution": "uniform",
        "discrete_levels": [0.0, 1.0, 2.0],
        "min_dist_3d": 0.5,
    },
    "objects": {
        "min_count": 4,
        "max_count": 10,
        "min_dist": 0.25,
        "margin": 0.4,
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

    return cfg


def create_directories(cfg):
    """创建输出目录结构。"""
    output = Path(cfg["output"]["dir"])
    for sub in [
        "images/single_view", "images/multi_view",
        "images/top_view", "images/side_view",
        "scenes", "splits",
    ]:
        (output / sub).mkdir(parents=True, exist_ok=True)


def _run_split(args_tuple):
    """并行执行的工作函数。"""
    split_name, split_cfg, cfg = args_tuple
    return split_name, pipeline.build_split(split_name, split_cfg, cfg)


def main():
    parser = argparse.ArgumentParser(
        description="Generate ordinary-bench-3d dataset (3D scenes + rendered images)"
    )
    parser.add_argument("--config", "-c", default=None,
        help="Path to TOML config file")
    parser.add_argument("--preset", "-p", choices=list(PRESETS.keys()), default=None,
        help="Size preset")
    parser.add_argument("--blender", default=None,
        help="Blender executable path")
    parser.add_argument("--output-dir", "-o", default=None,
        help="Output directory")
    parser.add_argument("--gpu", action="store_true", default=None,
        help="Enable GPU rendering")
    parser.add_argument("--workers", "-w", type=int, default=1,
        help="Number of parallel Blender processes")
    parser.add_argument("--start-idx", type=int, default=0,
        help="Scene index offset")
    parser.add_argument("--dry-run", action="store_true",
        help="Print resolved config and exit")

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
        logger.warning(
            "Using default 'blender' command. "
            "Set blender.executable in config.toml or use --blender."
        )

    n_splits = len(cfg["splits"])
    total_scenes = sum(s["n_scenes"] for s in cfg["splits"].values())
    workers = min(args.workers, n_splits)

    logger.info("=" * 50)
    logger.info("ORDINARY-BENCH-3D Data Generation")
    logger.info("=" * 50)
    logger.info(f"Output:  {cfg['output']['dir']}")
    logger.info(f"Blender: {blender}")
    logger.info(f"Splits:  {n_splits} ({total_scenes} scenes total)")
    logger.info(f"Workers: {workers}")
    logger.info(f"Z-range: {cfg['placement']['z_range']}")
    logger.info(f"Z-dist:  {cfg['placement']['z_distribution']}")

    create_directories(cfg)

    all_stats = {}

    if workers <= 1:
        for split_name, split_cfg in cfg["splits"].items():
            logger.info(f"\n--- Split: {split_name} ---")
            stats = pipeline.build_split(split_name, split_cfg, cfg)
            all_stats[split_name] = stats
    else:
        tasks = [
            (name, scfg, cfg) for name, scfg in cfg["splits"].items()
        ]
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
    total_images = sum(
        s["n_single_view_images"] + s["n_multi_view_images"]
        + s.get("n_top_view_images", 0) + s.get("n_side_view_images", 0)
        for s in all_stats.values()
    )
    print(f"\nDone! {total_scenes} scenes, {total_images} images")
    print(f"Output: {cfg['output']['dir']}")


if __name__ == "__main__":
    main()
