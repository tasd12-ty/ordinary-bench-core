#!/usr/bin/env python3
"""Minimal Infinigen-Indoors wrapper for ordinary-bench style scene generation."""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

try:
    import tomllib
except ModuleNotFoundError:
    try:
        import tomli as tomllib
    except ModuleNotFoundError:
        tomllib = None

from adapter import adapt_scene, export_native_record


DEFAULT_CONFIG = {
    "infinigen": {
        "root": "",
        "python": "python3",
        "driver_script": "infinigen_examples.generate_indoors",
    },
    "scene": {
        "seed": 0,
        "room_type": "DiningRoom",
        "simple_layout": True,
        "terrain_enabled": False,
        "single_room": True,
        "overhead_camera": True,
    },
    "adapter": {
        "max_objects": 6,
        "min_depth": 0.2,
        "min_screen_margin": 4,
        "query_terms": [],
    },
    "output": {
        "dir": "./data-gen-infinigen/output",
        "scene_prefix": "ifg",
    },
}


def deep_merge(base: Dict, override: Dict) -> Dict:
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str) -> Dict:
    cfg = dict(DEFAULT_CONFIG)
    if path:
        if tomllib is None:
            raise RuntimeError("TOML support requires Python 3.11+ or tomli")
        with open(path, "rb") as f:
            cfg = deep_merge(cfg, tomllib.load(f))
    return cfg


def build_generate_commands(cfg: Dict, output_dir: Path) -> List[List[str]]:
    scene_cfg = cfg["scene"]
    ig_cfg = cfg["infinigen"]
    python_bin = ig_cfg["python"]
    driver = ig_cfg["driver_script"]
    seed = int(scene_cfg["seed"])

    raw_root = output_dir / "raw" / f"seed_{seed:06d}"
    coarse_dir = raw_root / "coarse"
    frames_dir = raw_root / "scene"

    gin_args = []
    if scene_cfg.get("simple_layout", True):
        gin_args.append("fast_solve.gin")
    if scene_cfg.get("overhead_camera", True):
        gin_args.append("overhead.gin")
    if scene_cfg.get("single_room", True):
        gin_args.append("singleroom.gin")

    overrides = [
        f"compose_indoors.terrain_enabled={str(bool(scene_cfg.get('terrain_enabled', False)))}",
        "restrict_solving.solve_max_rooms=1",
        "compose_indoors.restrict_single_supported_roomtype=True",
        "compose_indoors.invisible_room_ceilings_enabled=True",
    ]
    room_type = scene_cfg.get("room_type")
    if room_type:
        overrides.append(f'restrict_solving.restrict_parent_rooms=["{room_type}"]')

    coarse_cmd = [
        python_bin, "-m", driver,
        "--seed", str(seed),
        "--task", "coarse",
        "--output_folder", str(coarse_dir),
        "-g",
        *gin_args,
        "-p",
        *overrides,
    ]

    render_cmd = [
        python_bin, "-m", driver,
        "--seed", str(seed),
        "--task", "render",
        "--input_folder", str(coarse_dir),
        "--output_folder", str(frames_dir),
    ]
    return [coarse_cmd, render_cmd]


def run_command(cmd: List[str], cwd: Path) -> None:
    result = subprocess.run(cmd, cwd=str(cwd), text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with code {result.returncode}: {' '.join(cmd)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prototype Infinigen backend for ordinary-bench")
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.toml")))
    parser.add_argument("--infinigen-root", default=None, help="Path to an Infinigen checkout")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--room-type", default=None)
    parser.add_argument("--max-objects", type=int, default=None)
    parser.add_argument("--query-term", action="append", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-run", action="store_true", help="Do not run Infinigen; only adapt an existing output")
    parser.add_argument("--source-root", default=None, help="Existing Infinigen scene folder to adapt")
    args = parser.parse_args()

    cfg = load_config(args.config if args.config and Path(args.config).exists() else "")
    if args.infinigen_root:
        cfg["infinigen"]["root"] = args.infinigen_root
    if args.output_dir:
        cfg["output"]["dir"] = args.output_dir
    if args.seed is not None:
        cfg["scene"]["seed"] = args.seed
    if args.room_type:
        cfg["scene"]["room_type"] = args.room_type
    if args.max_objects is not None:
        cfg["adapter"]["max_objects"] = args.max_objects
    if args.query_term is not None:
        cfg["adapter"]["query_terms"] = list(args.query_term)

    output_dir = Path(cfg["output"]["dir"]).resolve()
    scene_prefix = cfg["output"]["scene_prefix"]
    seed = int(cfg["scene"]["seed"])
    room_slug = (cfg["scene"].get("room_type") or "room").lower()
    scene_id = f"{scene_prefix}_{room_slug}_{seed:06d}"

    source_root = Path(args.source_root).resolve() if args.source_root else output_dir / "raw" / f"seed_{seed:06d}" / "scene"
    image_dst = output_dir / "images" / "single_view" / f"{scene_id}.png"
    multi_view_dir = output_dir / "images" / "multi_view" / scene_id
    scene_out = output_dir / "scenes" / f"{scene_id}.json"
    native_record_dir = output_dir / "native" / scene_id

    commands = build_generate_commands(cfg, output_dir)

    if args.dry_run:
        print(json.dumps({
            "config": cfg,
            "source_root": str(source_root),
            "scene_out": str(scene_out),
            "image_dst": str(image_dst),
            "multi_view_dir": str(multi_view_dir),
            "native_record_dir": str(native_record_dir),
            "commands": commands,
        }, indent=2))
        return

    if not args.skip_run:
        infinigen_root = cfg["infinigen"].get("root", "")
        if not infinigen_root:
            raise RuntimeError("--infinigen-root is required unless --skip-run is used")
        cwd = Path(infinigen_root).resolve()
        for cmd in commands:
            run_command(cmd, cwd=cwd)

    scene = adapt_scene(
        source_root=source_root,
        scene_id=scene_id,
        split="ifg",
        max_objects=int(cfg["adapter"]["max_objects"]),
        query_terms=cfg["adapter"].get("query_terms", []),
        min_depth=float(cfg["adapter"]["min_depth"]),
        min_screen_margin=int(cfg["adapter"]["min_screen_margin"]),
        image_dst=image_dst,
        multi_view_dir=multi_view_dir,
    )
    native_manifest = export_native_record(
        source_root=source_root,
        scene_id=scene_id,
        dest_dir=native_record_dir,
    )
    scene["source"]["native_record_dir"] = str(native_record_dir)
    scene["source"]["native_manifest"] = str(native_record_dir / "manifest.json")
    scene["source"]["native_objects_json"] = native_manifest["files"]["objects"]
    scene["source"]["native_camview"] = native_manifest["files"]["camview"]
    scene["source"]["native_views"] = native_manifest["views"]
    scene_out.parent.mkdir(parents=True, exist_ok=True)
    with scene_out.open("w") as f:
        json.dump(scene, f, indent=2)

    print(f"Adapted scene written to {scene_out}")
    print(f"Native Infinigen record written to {native_record_dir}")
    if image_dst.exists():
        print(f"Image copied to {image_dst}")
    if multi_view_dir.exists():
        print(f"Multi-view images copied to {multi_view_dir}")


if __name__ == "__main__":
    main()
