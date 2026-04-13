"""
Variable-Size Subset Ablation: 枚举不同大小的子场景。

从一个 N=20 的父场景出发，对每种目标大小 m (9..19)，
从 C(20, m) 个可能的子集中采样 --max-subsets 个，
构建子集 scene JSON 并输出 manifest。

与 experiments/subset_ablation/enumerate_subsets.py 的区别:
  - 原版固定子集大小 k=4，本脚本支持多种大小
  - manifest 额外含 subset_size 字段，方便下游按大小分组分析
  - 复用原版的 build_subset_scene_json 函数

用法:
    cd experiments/variable_subset_ablation

    # 从单个 N=20 场景枚举，每种大小采样 20 个子场景
    python enumerate_variable_subsets.py \
        --scene ../../data-gen/output/scenes/n20_000000.json \
        --min-size 9 --max-size 19 \
        --max-subsets 20 --seed 42 \
        --output-dir output

    # 也支持指定场景目录 + split 过滤（兼容批量场景）
    python enumerate_variable_subsets.py \
        --scenes-dir ../../data-gen/output/scenes \
        --splits n20 \
        --min-size 9 --max-size 19 \
        --max-subsets 20 \
        --output-dir output

输出:
    output/scenes/{subset_id}.json   — 子集场景 JSON（保留原始坐标）
    output/manifest.json             — 父→子映射，含 subset_size 字段

manifest 结构:
    {
      "parent_scenes": {
        "n20_000000": {
          "n_objects": 20,
          "scene_file": "...",
          "subsets_by_size": {
            "9":  [{"subset_id": "...", "object_ids": [...], "subset_size": 9}, ...],
            "10": [...],
            ...
          }
        }
      },
      "config": { "min_size": 9, "max_size": 19, "max_subsets": 20, "seed": 42 }
    }
"""

import argparse
import json
import random
import sys
from itertools import combinations
from math import comb
from pathlib import Path

# 复用现有 enumerate_subsets.py 中的 build_subset_scene_json
SCRIPT_DIR = Path(__file__).resolve().parent
SUBSET_ABLATION_DIR = SCRIPT_DIR.parent / "subset_ablation"
if str(SUBSET_ABLATION_DIR) not in sys.path:
    sys.path.insert(0, str(SUBSET_ABLATION_DIR))

from enumerate_subsets import build_subset_scene_json


def enumerate_variable_subsets(
    scene: dict,
    min_size: int = 9,
    max_size: int = 19,
    max_subsets: int = 20,
    seed: int = 42,
) -> dict:
    """
    对含 N 个物体的场景，枚举多种大小的子集。

    Args:
        scene: 父场景 JSON dict
        min_size: 最小子集大小
        max_size: 最大子集大小（不超过 N-1）
        max_subsets: 每种大小最多采样多少个子集
        seed: 随机种子

    Returns:
        {size: [subset_dict, ...]} 按大小分组的子集列表
    """
    objects = scene["objects"]
    n = len(objects)
    scene_id = scene["scene_id"]

    # 限制范围
    max_size = min(max_size, n - 1)  # 至少消融 1 个物体
    min_size = max(min_size, 1)

    rng = random.Random(seed)
    subsets_by_size = {}
    global_idx = 0  # 全局子集编号，保证 subset_id 唯一

    for target_size in range(min_size, max_size + 1):
        total_combos = comb(n, target_size)

        if total_combos <= max_subsets:
            # 全枚举
            combos = list(combinations(range(n), target_size))
        else:
            # 随机采样：生成随机索引，避免枚举全部组合
            # 对于 C(20,10)=184756 这种规模，先生成全部再采样仍可行
            all_combos = list(combinations(range(n), target_size))
            combos = rng.sample(all_combos, max_subsets)

        size_subsets = []
        for combo in combos:
            subset_objects = [objects[i] for i in combo]
            object_ids = [obj["id"] for obj in subset_objects]
            subset_id = f"{scene_id}__sz{target_size:02d}_s{global_idx:04d}"

            subset_dict = {
                "subset_id": subset_id,
                "parent_scene_id": scene_id,
                "object_ids": object_ids,
                "object_indices": list(combo),
                "objects": subset_objects,
                "subset_size": target_size,
            }
            size_subsets.append(subset_dict)
            global_idx += 1

        subsets_by_size[target_size] = size_subsets

    return subsets_by_size


def main():
    parser = argparse.ArgumentParser(
        description="Variable-Size Subset Ablation: 枚举不同大小的子场景"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--scene", help="单个父场景 JSON 文件路径")
    group.add_argument("--scenes-dir", help="父场景 JSON 目录")
    parser.add_argument("--splits", default=None,
                        help="逗号分隔的 split 前缀 (如 n20)，仅 --scenes-dir 时有效")
    parser.add_argument("--min-size", type=int, default=9,
                        help="最小子集大小 (默认 9)")
    parser.add_argument("--max-size", type=int, default=19,
                        help="最大子集大小 (默认 19)")
    parser.add_argument("--max-subsets", type=int, default=20,
                        help="每种大小最多采样 N 个子集 (默认 20)")
    parser.add_argument("--seed", type=int, default=42, help="随机种子 (默认 42)")
    parser.add_argument("--output-dir", default="output", help="输出目录 (默认 output)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    scenes_out = output_dir / "scenes"
    scenes_out.mkdir(parents=True, exist_ok=True)

    # 加载场景文件
    if args.scene:
        scene_files = [Path(args.scene)]
    else:
        scenes_dir = Path(args.scenes_dir)
        scene_files = sorted(scenes_dir.glob("*.json"))
        if args.splits:
            prefixes = args.splits.split(",")
            scene_files = [
                f for f in scene_files
                if any(f.stem.startswith(p) for p in prefixes)
            ]

    if not scene_files:
        print("No scene files found.")
        return

    manifest = {
        "parent_scenes": {},
        "config": {
            "min_size": args.min_size,
            "max_size": args.max_size,
            "max_subsets": args.max_subsets,
            "seed": args.seed,
        },
    }

    total_subsets = 0

    for scene_file in scene_files:
        with open(scene_file) as f:
            scene = json.load(f)

        scene_id = scene["scene_id"]
        n = len(scene["objects"])
        print(f"\n{'='*60}")
        print(f"Scene: {scene_id} (N={n})")
        print(f"{'='*60}")

        if n <= args.min_size:
            print(f"  SKIP: N={n} <= min_size={args.min_size}")
            continue

        subsets_by_size = enumerate_variable_subsets(
            scene,
            min_size=args.min_size,
            max_size=args.max_size,
            max_subsets=args.max_subsets,
            seed=args.seed,
        )

        # 写入子集 scene JSON + 构建 manifest
        manifest_subsets_by_size = {}
        for size, subsets in sorted(subsets_by_size.items()):
            manifest_size_list = []
            for subset in subsets:
                # 写 scene JSON
                subset_scene = build_subset_scene_json(scene, subset)
                subset_scene["subset_size"] = size  # 额外标记大小
                out_path = scenes_out / f"{subset['subset_id']}.json"
                with open(out_path, "w") as f:
                    json.dump(subset_scene, f, indent=2)

                manifest_size_list.append({
                    "subset_id": subset["subset_id"],
                    "object_ids": subset["object_ids"],
                    "subset_size": size,
                })

            manifest_subsets_by_size[str(size)] = manifest_size_list
            total_combos = comb(n, size)
            print(f"  Size {size:>2}: {len(subsets):>3} subsets "
                  f"(of C({n},{size})={total_combos:>6})")
            total_subsets += len(subsets)

        manifest["parent_scenes"][scene_id] = {
            "n_objects": n,
            "scene_file": str(scene_file),
            "subsets_by_size": manifest_subsets_by_size,
            # 兼容现有管线的 "subsets" 扁平列表
            "subsets": [
                entry
                for size_list in manifest_subsets_by_size.values()
                for entry in size_list
            ],
        }

    # 写入 manifest
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Total: {len(manifest['parent_scenes'])} scenes, {total_subsets} subsets")
    print(f"Output: {output_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
