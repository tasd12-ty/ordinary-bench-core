#!/usr/bin/env python3
"""
构建 ORDINARY-BENCH Grid3D HuggingFace 数据集。

读取 grid3d 场景 JSON、问题文件和 6 个正交视角图像，
生成包含内嵌图像的自包含 Parquet 文件。

用法：
    python build_dataset.py
    python build_dataset.py --dry-run

依赖：
    pip install pandas pyarrow datasets pillow
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

SCRIPT_DIR = Path(__file__).parent
DEFAULT_OUTPUT = SCRIPT_DIR.parent / "output"
VIEWS = ["top", "bottom", "front", "back", "left", "right"]
VALID_SPLITS = {"g04", "g05", "g06", "g07", "g08", "g09", "g10"}
# 训练集：索引 000-014（15 个场景），测试集：015-019（5 个场景）
TEST_START_IDX = 15


def build_rows(output_dir: Path) -> Tuple[List[dict], dict]:
    """从 grid3d 输出构建数据集行。返回 (rows, stats)。"""
    scenes_dir = output_dir / "scenes"
    questions_dir = output_dir / "questions"

    rows: List[dict] = []
    stats = {"scenes": 0, "skipped": 0, "train": 0, "test": 0}

    for scene_path in sorted(scenes_dir.glob("*.json")):
        scene_id = scene_path.stem
        split = scene_id.split("_")[0]

        # 跳过非标准 split（如旧版 g12 测试场景）
        if split not in VALID_SPLITS:
            stats["skipped"] += 1
            continue

        scene = json.loads(scene_path.read_text())
        idx = int(scene_id.split("_")[1])
        hf_split = "test" if idx >= TEST_START_IDX else "train"

        # 加载问题文件
        q_path = questions_dir / f"{scene_id}.json"
        if not q_path.exists():
            stats["skipped"] += 1
            continue
        question = json.loads(q_path.read_text())

        # 将 6 个视角图像加载为字节数据
        view_bytes = {}
        for view in VIEWS:
            img_path = output_dir / "images" / view / f"{scene_id}.png"
            if img_path.exists():
                view_bytes[view] = {"bytes": img_path.read_bytes(), "path": None}
            else:
                view_bytes[view] = None

        # 物体描述
        objects_desc = []
        for obj in scene["objects"]:
            desc = f"{obj.get('color', '')} {obj.get('material', '')} {obj.get('shape', '')}".strip()
            objects_desc.append({"id": obj["id"], "desc": desc})

        row: Dict[str, Any] = {
            "scene_id": scene_id,
            "n_objects": scene["n_objects"],
            "split": split,
            "hf_split": hf_split,
            "image_top": view_bytes.get("top"),
            "image_bottom": view_bytes.get("bottom"),
            "image_front": view_bytes.get("front"),
            "image_back": view_bytes.get("back"),
            "image_left": view_bytes.get("left"),
            "image_right": view_bytes.get("right"),
            "objects": json.dumps(objects_desc, ensure_ascii=False),
            "system_prompt": question["system_prompt"],
            "user_prompt": question["user_prompt"],
            "ground_truth": json.dumps(question["ground_truth"], ensure_ascii=False),
            "scene_metadata": json.dumps(scene, ensure_ascii=False),
        }

        rows.append(row)
        stats["scenes"] += 1
        stats[hf_split] += 1

    return rows, stats


def write_parquet(rows: List[dict], output_dir: Path):
    """使用 datasets 库将数据写入训练/测试 Parquet 文件。"""
    import datasets as ds

    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    image_cols = [f"image_{v}" for v in VIEWS]

    features = ds.Features({
        "scene_id": ds.Value("string"),
        "n_objects": ds.Value("int32"),
        "split": ds.Value("string"),
        "image_top": ds.Image(),
        "image_bottom": ds.Image(),
        "image_front": ds.Image(),
        "image_back": ds.Image(),
        "image_left": ds.Image(),
        "image_right": ds.Image(),
        "objects": ds.Value("string"),
        "system_prompt": ds.Value("string"),
        "user_prompt": ds.Value("string"),
        "ground_truth": ds.Value("string"),
        "scene_metadata": ds.Value("string"),
    })

    for hf_split in ("train", "test"):
        split_rows = [
            {k: v for k, v in r.items() if k != "hf_split"}
            for r in rows if r.get("hf_split") == hf_split
        ]
        if not split_rows:
            continue

        hf_ds = ds.Dataset.from_list(split_rows, features=features)
        out_path = data_dir / f"{hf_split}.parquet"
        hf_ds.to_parquet(str(out_path))
        print(f"  {hf_split}: {len(hf_ds)} rows -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Build Grid3D HuggingFace dataset")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT),
                        help="Path to data-gen-grid3d/output/")
    parser.add_argument("--dataset-dir", default=str(SCRIPT_DIR),
                        help="Where to write parquet files")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    dataset_dir = Path(args.dataset_dir)

    if not (output_dir / "scenes").exists():
        print(f"Error: {output_dir / 'scenes'} not found")
        sys.exit(1)

    print("Building ORDINARY-BENCH Grid3D dataset...")
    print(f"  Source: {output_dir}")
    print(f"  Output: {dataset_dir}")
    print()

    rows, stats = build_rows(output_dir)

    print(f"  Scenes: {stats['scenes']} (skipped: {stats['skipped']})")
    print(f"  Train: {stats['train']}, Test: {stats['test']}")

    if args.dry_run:
        print("\n[Dry run] No files written.")
        return

    # 写入前移除 hf_split 字段（保留副本用于分割）
    print("\nWriting Parquet files...")
    write_parquet(rows, dataset_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
