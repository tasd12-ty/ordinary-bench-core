"""
Step 1: 子集枚举与场景 JSON 构建。

对每个含 N 个物体的场景，枚举所有 C(N,4) 个 4 物体子集，
为每个子集构建精简版 scene JSON。

用法:
    python enumerate_subsets.py --scenes-dir ../../datasets/test-data/scenes --output-dir output
    python enumerate_subsets.py --scenes-dir ../../datasets/test-data/scenes --output-dir output --split n10
    python enumerate_subsets.py --scenes-dir ../../datasets/test-data/scenes --output-dir output --max-subsets 10
"""

import argparse
import json
import random
from itertools import combinations
from pathlib import Path


def enumerate_subsets(scene: dict, k: int = 4) -> list[dict]:
    """
    对含 N 个物体的场景，枚举所有 C(N,k) 个子集。

    Returns:
        子集列表，每个元素含 subset_id, object_ids, objects。
    """
    objects = scene["objects"]
    n = len(objects)
    scene_id = scene["scene_id"]
    subsets = []

    for idx, combo in enumerate(combinations(range(n), k)):
        subset_objects = [objects[i] for i in combo]
        object_ids = [obj["id"] for obj in subset_objects]
        subset_id = f"{scene_id}__s{idx:04d}"

        subsets.append({
            "subset_id": subset_id,
            "parent_scene_id": scene_id,
            "object_ids": object_ids,
            "object_indices": list(combo),
            "objects": subset_objects,
        })

    return subsets


def build_subset_scene_json(parent_scene: dict, subset: dict) -> dict:
    """
    从父场景构建子集 scene JSON，保留物体原始坐标和属性。
    """
    subset_scene = {
        "scene_id": subset["subset_id"],
        "parent_scene_id": subset["parent_scene_id"],
        "split": parent_scene.get("split", ""),
        "n_objects": len(subset["objects"]),
        "objects": subset["objects"],
    }

    # 复制 views 中的 camera 信息（如果存在）
    if "views" in parent_scene and parent_scene["views"]:
        view0 = parent_scene["views"][0]
        subset_scene["camera"] = view0.get("camera", {})

    return subset_scene


def main():
    parser = argparse.ArgumentParser(description="枚举 C(N,4) 子集并构建子集 scene JSON")
    parser.add_argument("--scenes-dir", required=True, help="父场景 JSON 目录")
    parser.add_argument("--output-dir", default="output", help="输出目录")
    parser.add_argument("--splits", default=None,
                        help="逗号分隔的 split 前缀列表 (如 n05,n06,...,n10)")
    parser.add_argument("--k", type=int, default=4, help="子集大小 (默认 4)")
    parser.add_argument("--max-subsets", type=int, default=None,
                        help="每场景最多采样 N 个子集 (控制成本)")
    parser.add_argument("--max-scenes", type=int, default=None,
                        help="每个 split 最多处理 N 个场景")
    parser.add_argument("--seed", type=int, default=42, help="随机采样种子")
    args = parser.parse_args()

    scenes_dir = Path(args.scenes_dir)
    output_dir = Path(args.output_dir)
    scenes_out = output_dir / "scenes"
    scenes_out.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)

    # 发现场景文件，按 split 分组
    all_files = sorted(scenes_dir.glob("*.json"))
    splits = args.splits.split(",") if args.splits else None

    if splits:
        all_files = [f for f in all_files if any(f.stem.startswith(s) for s in splits)]

    if not all_files:
        print(f"No scene files found in {scenes_dir}")
        return

    # 按 split 分组并采样
    if args.max_scenes:
        from collections import defaultdict
        by_split = defaultdict(list)
        for f in all_files:
            # split prefix = everything before the last _ and digits
            prefix = f.stem.rsplit("_", 1)[0]
            by_split[prefix].append(f)
        scene_files = []
        for prefix in sorted(by_split):
            files = by_split[prefix]
            sampled = random.sample(files, min(args.max_scenes, len(files)))
            scene_files.extend(sampled)
        scene_files.sort()
    else:
        scene_files = all_files

    manifest = {"parent_scenes": {}}
    total_subsets = 0
    skipped_small = 0

    for scene_file in scene_files:
        with open(scene_file) as f:
            scene = json.load(f)

        scene_id = scene["scene_id"]
        n = len(scene["objects"])

        if n < args.k:
            skipped_small += 1
            continue

        # 枚举子集
        subsets = enumerate_subsets(scene, k=args.k)

        # 采样控制（保留原始子集 ID，不重新编号）
        if args.max_subsets and len(subsets) > args.max_subsets:
            subsets = random.sample(subsets, args.max_subsets)

        # 写入子集 scene JSON
        for subset in subsets:
            subset_scene = build_subset_scene_json(scene, subset)
            out_path = scenes_out / f"{subset['subset_id']}.json"
            with open(out_path, "w") as f:
                json.dump(subset_scene, f, indent=2)

        # 记录到 manifest
        manifest["parent_scenes"][scene_id] = {
            "n_objects": n,
            "n_subsets": len(subsets),
            "scene_file": str(scene_file),
            "subsets": [
                {
                    "subset_id": s["subset_id"],
                    "object_ids": s["object_ids"],
                }
                for s in subsets
            ],
        }
        total_subsets += len(subsets)

    # 写入 manifest
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Processed {len(manifest['parent_scenes'])} scenes "
          f"(skipped {skipped_small} with n < {args.k})")
    print(f"Generated {total_subsets} subsets")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
