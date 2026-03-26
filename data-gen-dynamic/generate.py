#!/usr/bin/env python3
"""
动态场景生成入口。

用法：
    python generate.py                        # 使用 config.toml
    python generate.py --preset test          # 1 个场景，快速测试
    python generate.py --config my.toml       # 自定义配置文件
    python generate.py --dry-run              # 打印配置后退出
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
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
        "animation": {"n_frames": 48, "fps": 24},
        "motion": {
            "speed_min": 0.30, "speed_max": 0.60,
            "omega_min": 0.15, "omega_max": 0.35,
        },
        "rendering": {"samples": 32, "camera_distance": 8.0},
        "video": {"encode": True, "crf": 18, "keep_frames": True},
        "splits": {
            "d03": {"n_scenes": 1, "n_objects": 3},
        },
    },
    # ---- test_video：48 帧短视频，每个难度级别 1 个场景 ----
    "test_video": {
        "animation": {"n_frames": 48, "fps": 24},
        "rendering": {"samples": 32, "camera_distance": 10.0},
        "video": {"encode": True, "crf": 18, "keep_frames": True},
        "splits": {
            "tv_L1": {"n_scenes": 1, "n_objects": 3},
            "tv_L3": {"n_scenes": 1, "n_objects": 5},
            "tv_L5": {"n_scenes": 1, "n_objects": 7},
        },
    },
    # ---- 难度等级：L1–L5 ----
    # ---- L1：基础（1 个运动物体，仅线性，速度慢）----
    # 最大位移：0.008 * 360 = 2.88 BU——bounds=5.0 时安全
    "L1": {
        "animation": {"n_frames": 360, "fps": 24},
        "motion": {
            "n_moving": 1,
            "static": 0.0, "linear": 1.0, "circular": 0.0,
            "speed_min": 0.005, "speed_max": 0.008,
        },
        "objects": {"min_count": 3, "max_count": 4, "bounds": 5.0},
        "rendering": {"samples": 64, "camera_distance": 10.0},
        "video": {"encode": True, "crf": 18, "keep_frames": False},
        "splits": {
            "L1_d03": {"n_scenes": 50, "n_objects": 3},
            "L1_d04": {"n_scenes": 50, "n_objects": 4},
        },
    },
    # ---- L2：追踪（2 个运动物体，线性 + 圆周）----
    "L2": {
        "animation": {"n_frames": 360, "fps": 24},
        "motion": {
            "n_moving": 2,
            "static": 0.0, "linear": 0.6, "circular": 0.4,
            "speed_min": 0.006, "speed_max": 0.012,
            "omega_min": 0.015, "omega_max": 0.04,
        },
        "objects": {"min_count": 4, "max_count": 5, "bounds": 5.0},
        "rendering": {"samples": 64, "camera_distance": 10.0},
        "video": {"encode": True, "crf": 18, "keep_frames": False},
        "splits": {
            "L2_d04": {"n_scenes": 50, "n_objects": 4},
            "L2_d05": {"n_scenes": 50, "n_objects": 5},
        },
    },
    # ---- L3：动态（全部运动，有界 + 线性混合）----
    # 使用圆周（有界）+ 弹跳（有界）+ 慢速线性
    "L3": {
        "animation": {"n_frames": 360, "fps": 24},
        "motion": {
            "static": 0.0, "linear": 0.2, "circular": 0.4, "bounce": 0.4,
            "speed_min": 0.008, "speed_max": 0.015,
            "omega_min": 0.03, "omega_max": 0.08,
            "radius_min": 0.4, "radius_max": 1.0,
        },
        "objects": {"min_count": 4, "max_count": 6, "bounds": 5.0},
        "rendering": {"samples": 64, "camera_distance": 10.0},
        "video": {"encode": True, "crf": 18, "keep_frames": False},
        "splits": {
            "L3_d04": {"n_scenes": 50, "n_objects": 4},
            "L3_d05": {"n_scenes": 50, "n_objects": 5},
            "L3_d06": {"n_scenes": 50, "n_objects": 6},
        },
    },
    # ---- L4：影视级（轨道相机，有界运动，速度更快）----
    "L4": {
        "animation": {"n_frames": 360, "fps": 24},
        "motion": {
            "static": 0.0, "linear": 0.0, "circular": 0.3, "bounce": 0.4, "waypoint": 0.3,
            "speed_min": 0.010, "speed_max": 0.020,
            "omega_min": 0.03, "omega_max": 0.08,
            "radius_min": 0.3, "radius_max": 0.8,
        },
        "camera": {
            "type": "orbit",
            "orbit_speed": 0.5,  # 度/秒 → 共 7.5°
        },
        "objects": {"min_count": 5, "max_count": 7, "bounds": 7.0},
        "rendering": {"samples": 64, "camera_distance": 12.0},
        "video": {"encode": True, "crf": 18, "keep_frames": False},
        "splits": {
            "L4_d05": {"n_scenes": 50, "n_objects": 5},
            "L4_d06": {"n_scenes": 50, "n_objects": 6},
            "L4_d07": {"n_scenes": 50, "n_objects": 7},
        },
    },
    # ---- L5：对抗级（复杂相机，快速有界运动，密集）----
    # 更大边界（8.0）+ 相机距离（14.0）以容纳 7-10 个物体
    "L5": {
        "animation": {"n_frames": 360, "fps": 24},
        "motion": {
            "static": 0.0, "linear": 0.0, "circular": 0.2, "bounce": 0.4, "waypoint": 0.4,
            "speed_min": 0.015, "speed_max": 0.030,
            "omega_min": 0.04, "omega_max": 0.10,
            "radius_min": 0.4, "radius_max": 1.0,
        },
        "camera": {
            "type": "composite",
            "orbit_speed": 2.0,      # 度/秒 → 共 30°
            "pan_range": 1.0,        # look_at 偏移量 ±
            "zoom_range": 2.0,       # 距离偏移量 ±
        },
        "objects": {"min_count": 7, "max_count": 8, "bounds": 8.0, "min_dist": 0.15},
        "rendering": {"samples": 64, "camera_distance": 14.0},
        "video": {"encode": True, "crf": 18, "keep_frames": False},
        "splits": {
            "L5_d07": {"n_scenes": 50, "n_objects": 7},
            "L5_d08": {"n_scenes": 50, "n_objects": 8},
        },
    },
}

DEFAULT_CONFIG = {
    "animation": {"n_frames": 360, "fps": 24},
    "motion": {"static": 0.2, "linear": 0.6, "circular": 0.2},
    "objects": {"min_count": 3, "max_count": 5, "min_dist": 0.25, "bounds": 5.0},
    "rendering": {
        "width": 480, "height": 320, "samples": 64,
        "camera_distance": 10.0, "elevation": 30.0, "azimuth_start": 45.0,
    },
    "video": {"encode": True, "crf": 18, "keep_frames": False},
    "blender": {"executable": "blender", "use_gpu": False},
    "output": {"dir": "./output_dynamic", "seed": 42},
    "splits": {},
}


def deep_merge(base, override):
    """递归将 override 合并到 base 中（override 优先）。"""
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def load_config(config_path, preset, cli_args):
    cfg = dict(DEFAULT_CONFIG)

    if config_path:
        p = Path(config_path)
        if not p.exists():
            logger.error(f"Config not found: {p}")
            sys.exit(1)
        if tomllib is None:
            logger.error("TOML support requires Python 3.11+ or 'pip install tomli'")
            sys.exit(1)
        with open(p, "rb") as f:
            cfg = deep_merge(cfg, tomllib.load(f))

    if preset:
        if preset not in PRESETS:
            logger.error(f"Unknown preset: {preset}")
            sys.exit(1)
        preset_data = PRESETS[preset]
        # 预设完全替换 splits（不与 config.toml 的 splits 合并）
        if "splits" in preset_data:
            cfg["splits"] = {}
        cfg = deep_merge(cfg, preset_data)

    if cli_args.blender:
        cfg["blender"]["executable"] = cli_args.blender
    if cli_args.output_dir:
        cfg["output"]["dir"] = cli_args.output_dir

    return cfg


def resolve_blender(cfg):
    """若默认 'blender' 不在 PATH 中，则自动检测 Blender。"""
    exe = cfg["blender"]["executable"]
    if shutil.which(exe):
        return
    mac_path = "/Applications/Blender.app/Contents/MacOS/Blender"
    if exe == "blender" and os.path.isfile(mac_path):
        cfg["blender"]["executable"] = mac_path
        logger.info(f"Auto-detected Blender: {mac_path}")
        return
    logger.warning(f"Blender '{exe}' not found in PATH")


def validate_blender(cfg):
    """在渲染前验证 Blender 可执行文件可正常运行。"""
    exe = cfg["blender"]["executable"]
    try:
        result = subprocess.run(
            [exe, "--version"], capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            logger.error(f"Blender at '{exe}' returned error")
            sys.exit(1)
        version_line = result.stdout.strip().split("\n")[0]
        logger.info(f"Blender: {version_line}")
    except FileNotFoundError:
        logger.error(f"Blender not found: '{exe}'")
        logger.error("Set blender.executable in config.toml or use --blender")
        sys.exit(1)
    except subprocess.TimeoutExpired:
        logger.error(f"Blender timed out: '{exe}'")
        sys.exit(1)


def validate_ffmpeg():
    """检查 ffmpeg 是否可用。"""
    if shutil.which("ffmpeg"):
        try:
            result = subprocess.run(
                ["ffmpeg", "-version"], capture_output=True, text=True, timeout=5,
            )
            version_line = result.stdout.strip().split("\n")[0]
            logger.info(f"ffmpeg: {version_line}")
            return True
        except Exception:
            pass
    logger.warning("未找到 ffmpeg，将跳过视频编码。")
    return False


def create_directories(cfg):
    output = Path(cfg["output"]["dir"])
    subdirs = ["plans", "scenes", "images", "splits"]
    if cfg.get("video", {}).get("encode", False):
        subdirs.append("videos")
    for sub in subdirs:
        (output / sub).mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(
        description="Generate dynamic scenes for ordinary-bench"
    )
    parser.add_argument("--config", "-c", default=None)
    parser.add_argument("--preset", "-p", choices=sorted(PRESETS.keys()), default=None)
    parser.add_argument("--blender", default=None)
    parser.add_argument("--output-dir", "-o", default=None)
    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config_path = args.config
    if config_path is None:
        default_toml = Path(__file__).resolve().parent / "config.toml"
        if default_toml.exists():
            config_path = str(default_toml)

    cfg = load_config(config_path, args.preset, args)
    resolve_blender(cfg)

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

    n_splits = len(cfg["splits"])
    total_scenes = sum(s["n_scenes"] for s in cfg["splits"].values())

    logger.info("=" * 50)
    logger.info("ORDINARY-BENCH Dynamic Scene Generation")
    logger.info("=" * 50)
    logger.info(f"Output:  {cfg['output']['dir']}")
    logger.info(f"Blender: {cfg['blender']['executable']}")
    logger.info(f"Splits:  {n_splits} ({total_scenes} scenes total)")
    n_frames = cfg['animation']['n_frames']
    fps_val = cfg['animation']['fps']
    logger.info(f"Frames:  {n_frames} @ {fps_val}fps ({n_frames/fps_val:.1f}s)")

    video_cfg = cfg.get("video", {})
    if video_cfg.get("encode", False):
        logger.info(f"Video:   MP4 (crf={video_cfg.get('crf', 18)}, keep_frames={video_cfg.get('keep_frames', True)})")

    validate_blender(cfg)
    if video_cfg.get("encode", False):
        if not validate_ffmpeg():
            video_cfg["encode"] = False
    create_directories(cfg)

    all_stats = {}
    for split_name, split_cfg in cfg["splits"].items():
        logger.info(f"\n--- Split: {split_name} ---")
        stats = pipeline.build_split(split_name, split_cfg, cfg)
        all_stats[split_name] = stats

    # 汇总统计
    total_ok = sum(s["n_ok"] for s in all_stats.values())
    total_failed = sum(s["n_failed"] for s in all_stats.values())
    print(f"\nDone! {total_ok} scenes generated, {total_failed} failed")
    print(f"Output: {cfg['output']['dir']}")


if __name__ == "__main__":
    main()
