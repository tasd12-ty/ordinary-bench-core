#!/usr/bin/env python3
"""
从现有输出构建 ORDINARY-BENCH HuggingFace 数据集。

从 data-gen/ 和 VLM-test/ 的输出中读取场景 JSON、问题文件和图像，
生成 Parquet 文件并将图像复制到数据集目录。

用法：
    python build_dataset.py
    python build_dataset.py --data-gen-dir ../data-gen/output --questions-dir ../VLM-test/output/questions
    python build_dataset.py --dry-run          # 仅统计，不写入
    python build_dataset.py --skip-images      # 跳过图像复制（开发阶段用于加速）

依赖：
    pip install pandas pyarrow datasets pillow
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


# ── 路径（默认相对于此脚本）──

SCRIPT_DIR = Path(__file__).parent
DEFAULT_DATA_GEN = SCRIPT_DIR.parent / "data-gen" / "output"
DEFAULT_QUESTIONS = SCRIPT_DIR.parent / "VLM-test" / "output" / "questions"


# ── 问题文本生成（复制自 VLM-test/API-test/prompts.py）──

def make_question_text(q: dict, objects: List[dict]) -> str:
    """为单个问题生成自然语言问题文本。"""
    if q["type"] == "qrr":
        p1a, p1b = q["pair1"]
        p2a, p2b = q["pair2"]
        if q.get("variant") == "shared_anchor" and q.get("anchor"):
            anchor = q["anchor"]
            cand1 = next(obj for obj in q["pair1"] if obj != anchor)
            cand2 = next(obj for obj in q["pair2"] if obj != anchor)
            return (
                f"From anchor {anchor}, compare the distance to {cand1} "
                f"vs the distance to {cand2}. Answer: < / ~= / >"
            )
        return (
            f"Compare the distance between {p1a} and {p1b} "
            f"vs the distance between {p2a} and {p2b}. "
            f"Answer: < / ~= / >"
        )
    elif q["type"] == "trr":
        return (
            f"Standing at {q['ref1']}, facing {q['ref2']} "
            f"(12 o'clock), what clock hour (1-12) is {q['target']} at?"
        )
    elif q["type"] == "fdr":
        others = [o["id"] for o in objects if o["id"] != q["anchor"]]
        return (
            f"Rank all other objects by distance from {q['anchor']}, "
            f"nearest to farthest. Objects to rank: {', '.join(others)}. "
            f"Answer: ordered JSON list of object IDs."
        )
    return ""


# ── 场景加载 ──

def load_scene_json(scenes_dir: Path, scene_id: str) -> Optional[dict]:
    """加载场景元数据 JSON 文件。"""
    p = scenes_dir / f"{scene_id}.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def load_question_file(questions_dir: Path, qtype: str, scene_id: str) -> Optional[dict]:
    """加载按题型分类的问题文件。"""
    p = questions_dir / qtype / f"{scene_id}.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


# ── 训练/测试集划分 ──

def load_split_assignment(data_gen_dir: Path) -> Dict[str, str]:
    """从 dataset_split.json 返回 {scene_id: 'train'|'test'} 映射。"""
    split_file = data_gen_dir / "dataset_split.json"
    if split_file.exists():
        with open(split_file) as f:
            info = json.load(f)
        # test_index_range = "80-max" 表示索引 >= 80 的场景为测试集
        test_start = int(info.get("test_index_range", "80-max").split("-")[0])
    else:
        test_start = 80

    # 通过解析 scene_id（如 "n04_000042"）提取索引 42 来构建映射
    assignment = {}
    scenes_dir = data_gen_dir / "scenes"
    if scenes_dir.exists():
        for f in sorted(scenes_dir.iterdir()):
            if f.suffix == ".json":
                sid = f.stem
                idx = int(sid.split("_")[1])
                assignment[sid] = "test" if idx >= test_start else "train"
    return assignment


# ── 行数据构建 ──

def _build_scene_rows(
    scene_id: str,
    scene: dict,
    hf_split: str,
    questions_dir: Path,
    data_gen_dir: Path,
) -> Tuple[List[dict], dict]:
    """为单个场景构建数据行。返回 (rows, stats_delta)。"""
    stats = {"qrr": 0, "trr": 0, "fdr": 0}
    n_objects = scene["n_objects"]
    obj_split = scene["split"]

    objects_desc = []
    for obj in scene["objects"]:
        desc = f"{obj.get('size', '')} {obj.get('color', '')} {obj.get('material', '')} {obj.get('shape', '')}".strip()
        objects_desc.append({"id": obj["id"], "desc": desc})
    objects_json = json.dumps(objects_desc, ensure_ascii=False)
    scene_metadata = json.dumps(scene, ensure_ascii=False)

    # 读取单视角图像字节以嵌入 Parquet 文件
    sv_path = data_gen_dir / "images" / "single_view" / f"{scene_id}.png"
    single_view_bytes = sv_path.read_bytes() if sv_path.exists() else None

    rows: List[dict] = []
    for qtype in ("qrr", "trr", "fdr"):
        qfile = load_question_file(questions_dir, qtype, scene_id)
        if qfile is None:
            continue
        for batch in qfile.get("batches", []):
            for q in batch.get("questions", []):
                row: Dict[str, Any] = {
                    "scene_id": scene_id,
                    "n_objects": n_objects,
                    "split": obj_split,
                    "hf_split": hf_split,
                    "image": {"bytes": single_view_bytes, "path": None},
                    "image_view0": None,
                    "image_view1": None,
                    "image_view2": None,
                    "image_view3": None,
                    "objects": objects_json,
                    "question_type": q["type"],
                    "qid": q["qid"],
                    "question_text": make_question_text(q, objects_desc),
                    "scene_metadata": scene_metadata,
                    "qrr_variant": None, "qrr_pair1": None, "qrr_pair2": None,
                    "qrr_metric": None, "qrr_gt_comparator": None,
                    "trr_target": None, "trr_ref1": None, "trr_ref2": None,
                    "trr_gt_hour": None, "trr_gt_quadrant": None, "trr_gt_angle_deg": None,
                    "fdr_anchor": None, "fdr_n_ranked": None,
                    "fdr_gt_ranking": None, "fdr_gt_distances": None, "fdr_gt_tie_groups": None,
                }
                if q["type"] == "qrr":
                    row["qrr_variant"] = q.get("variant", "disjoint")
                    row["qrr_pair1"] = json.dumps(q["pair1"])
                    row["qrr_pair2"] = json.dumps(q["pair2"])
                    row["qrr_metric"] = q.get("metric", "dist3D")
                    row["qrr_gt_comparator"] = q["gt_comparator"]
                    stats["qrr"] += 1
                elif q["type"] == "trr":
                    row["trr_target"] = q["target"]
                    row["trr_ref1"] = q["ref1"]
                    row["trr_ref2"] = q["ref2"]
                    row["trr_gt_hour"] = q["gt_hour"]
                    row["trr_gt_quadrant"] = q["gt_quadrant"]
                    row["trr_gt_angle_deg"] = q.get("gt_angle_deg")
                    stats["trr"] += 1
                elif q["type"] == "fdr":
                    row["fdr_anchor"] = q["anchor"]
                    row["fdr_n_ranked"] = q.get("n_ranked", len(q["gt_ranking"]))
                    row["fdr_gt_ranking"] = json.dumps(q["gt_ranking"])
                    row["fdr_gt_distances"] = json.dumps(q.get("gt_distances"))
                    row["fdr_gt_tie_groups"] = json.dumps(q.get("gt_tie_groups"))
                    stats["fdr"] += 1
                rows.append(row)
    return rows, stats


# ── Parquet 写入（逐场景流式处理以避免内存溢出）──

def _build_features(columns: List[str], image_columns: List[str]) -> "datasets.Features":
    """从列名构建 HuggingFace Features 模式。"""
    import datasets as ds

    feature_map = {}
    for col in columns:
        if col in image_columns:
            feature_map[col] = ds.Image()
        elif col in ("trr_gt_hour", "trr_gt_quadrant", "fdr_n_ranked"):
            feature_map[col] = ds.Value("int32")
        elif col in ("n_objects",):
            feature_map[col] = ds.Value("int32")
        elif col in ("trr_gt_angle_deg",):
            feature_map[col] = ds.Value("float32")
        else:
            feature_map[col] = ds.Value("string")
    return ds.Features(feature_map)


# 配置定义
_QRR_COLS = ["qrr_variant", "qrr_pair1", "qrr_pair2", "qrr_metric", "qrr_gt_comparator"]
_TRR_COLS = ["trr_target", "trr_ref1", "trr_ref2", "trr_gt_hour", "trr_gt_quadrant", "trr_gt_angle_deg"]
_FDR_COLS = ["fdr_anchor", "fdr_n_ranked", "fdr_gt_ranking", "fdr_gt_distances", "fdr_gt_tie_groups"]
_MV_COLS = ["image_view0", "image_view1", "image_view2", "image_view3"]

CONFIGS = {
    "all":  {"filter": None,  "drop": _MV_COLS, "image_cols": ["image"]},
    "qrr":  {"filter": "qrr", "drop": _TRR_COLS + _FDR_COLS + _MV_COLS, "image_cols": ["image"]},
    "trr":  {"filter": "trr", "drop": _QRR_COLS + _FDR_COLS + _MV_COLS, "image_cols": ["image"]},
    "fdr":  {"filter": "fdr", "drop": _QRR_COLS + _TRR_COLS + _MV_COLS, "image_cols": ["image"]},
    # 多视角配置已省略：每行嵌入 5 张图像对 Parquet 来说过大。
    # 多视角图像可在 images/multi_view/ 目录中获取。
}


def build_and_write(
    data_gen_dir: Path,
    questions_dir: Path,
    output_dir: Path,
) -> dict:
    """逐场景构建数据行并增量写入 Parquet 文件。"""
    import datasets as ds

    scenes_dir = data_gen_dir / "scenes"
    split_map = load_split_assignment(data_gen_dir)
    scene_ids = sorted(split_map.keys())
    data_dir = output_dir / "data"

    # 累加器：config -> hf_split -> 行列表（不含图像字节）
    # 每次处理一个配置以控制内存占用
    stats = {"scenes": 0, "qrr": 0, "trr": 0, "fdr": 0, "skipped_scenes": 0}

    import gc

    for config_name, config in CONFIGS.items():
        config_dir = data_dir / config_name
        config_dir.mkdir(parents=True, exist_ok=True)

        for hf_split in ("train", "test"):
            split_scenes = [sid for sid in scene_ids if split_map[sid] == hf_split]
            if not split_scenes:
                continue

            # 每个场景写入一个 Parquet 分片以保持低内存占用
            shard_paths = []
            for scene_id in split_scenes:
                scene = load_scene_json(scenes_dir, scene_id)
                if scene is None:
                    if config_name == "all":
                        stats["skipped_scenes"] += 1
                    continue

                rows, s = _build_scene_rows(
                    scene_id, scene, hf_split, questions_dir, data_gen_dir,
                )

                if config["filter"]:
                    rows = [r for r in rows if r["question_type"] == config["filter"]]
                if not rows:
                    continue

                for r in rows:
                    for col in config["drop"]:
                        r.pop(col, None)
                    r.pop("hf_split", None)

                if config_name == "all":
                    stats["qrr"] += s["qrr"]
                    stats["trr"] += s["trr"]
                    stats["fdr"] += s["fdr"]

                columns = list(rows[0].keys())
                features = _build_features(columns, config["image_cols"])

                hf_ds = ds.Dataset.from_list(rows, features=features)
                shard_idx = len(shard_paths)
                shard_path = config_dir / f"{hf_split}-{shard_idx:05d}.parquet"
                hf_ds.to_parquet(str(shard_path))
                shard_paths.append(shard_path)

                del rows, hf_ds
                gc.collect()

            if shard_paths:
                import pyarrow.parquet as pq
                total_rows = sum(
                    pq.read_metadata(str(p)).num_rows for p in shard_paths
                )
                print(f"  {config_name}/{hf_split}: {total_rows:,} rows ({len(shard_paths)} shards)")

            if config_name == "all":
                stats["scenes"] += len(split_scenes)

    return stats


# ── 图像复制 ──

def copy_images(data_gen_dir: Path, output_dir: Path):
    """将单视角和多视角图像复制到数据集目录。"""
    src_single = data_gen_dir / "images" / "single_view"
    dst_single = output_dir / "images" / "single_view"
    src_multi = data_gen_dir / "images" / "multi_view"
    dst_multi = output_dir / "images" / "multi_view"

    # 单视角
    dst_single.mkdir(parents=True, exist_ok=True)
    count = 0
    for f in sorted(src_single.glob("*.png")):
        dst = dst_single / f.name
        if not dst.exists():
            shutil.copy2(f, dst)
        count += 1
    print(f"  Single-view images: {count}")

    # 多视角
    count = 0
    for scene_dir in sorted(src_multi.iterdir()):
        if not scene_dir.is_dir():
            continue
        dst_scene = dst_multi / scene_dir.name
        dst_scene.mkdir(parents=True, exist_ok=True)
        for view_img in sorted(scene_dir.glob("view_*.png")):
            dst = dst_scene / view_img.name
            if not dst.exists():
                shutil.copy2(view_img, dst)
        count += 1
    print(f"  Multi-view scenes: {count}")


# ── 主程序 ──

def main():
    parser = argparse.ArgumentParser(
        description="Build ORDINARY-BENCH HuggingFace dataset"
    )
    parser.add_argument(
        "--data-gen-dir", type=str, default=str(DEFAULT_DATA_GEN),
        help="Path to data-gen/output/ directory",
    )
    parser.add_argument(
        "--questions-dir", type=str, default=str(DEFAULT_QUESTIONS),
        help="Path to VLM-test/output/questions/ directory",
    )
    parser.add_argument(
        "--output-dir", type=str, default=str(SCRIPT_DIR),
        help="Output directory (default: same as this script)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print statistics without writing files",
    )
    parser.add_argument(
        "--skip-images", action="store_true",
        help="Skip image copying (faster for development)",
    )
    args = parser.parse_args()

    data_gen_dir = Path(args.data_gen_dir)
    questions_dir = Path(args.questions_dir)
    output_dir = Path(args.output_dir)

    # 验证路径
    if not (data_gen_dir / "scenes").exists():
        print(f"Error: scenes directory not found at {data_gen_dir / 'scenes'}")
        sys.exit(1)
    if not questions_dir.exists():
        print(f"Error: questions directory not found at {questions_dir}")
        sys.exit(1)

    print("Building ORDINARY-BENCH dataset...")
    print(f"  Data source: {data_gen_dir}")
    print(f"  Questions:   {questions_dir}")
    print(f"  Output:      {output_dir}")
    print()

    if args.dry_run:
        # 试运行：仅统计，不读取图像
        split_map = load_split_assignment(data_gen_dir)
        scenes_dir = data_gen_dir / "scenes"
        stats = {"scenes": 0, "qrr": 0, "trr": 0, "fdr": 0, "skipped_scenes": 0}
        train_count = test_count = 0
        for scene_id in sorted(split_map.keys()):
            scene = load_scene_json(scenes_dir, scene_id)
            if scene is None:
                stats["skipped_scenes"] += 1
                continue
            stats["scenes"] += 1
            for qtype in ("qrr", "trr", "fdr"):
                qfile = load_question_file(questions_dir, qtype, scene_id)
                if qfile is None:
                    continue
                for batch in qfile.get("batches", []):
                    n = len(batch.get("questions", []))
                    stats[qtype] += n
                    if split_map[scene_id] == "train":
                        train_count += n
                    else:
                        test_count += n
        total = stats["qrr"] + stats["trr"] + stats["fdr"]
        print(f"  Scenes: {stats['scenes']} (skipped: {stats['skipped_scenes']})")
        print(f"  Questions: {total:,} total")
        print(f"    QRR: {stats['qrr']:,}")
        print(f"    TRR: {stats['trr']:,}")
        print(f"    FDR: {stats['fdr']:,}")
        print(f"  Train: {train_count:,} rows")
        print(f"  Test:  {test_count:,} rows")
        print("\n[Dry run] No files written.")
        return

    # 构建并写入 Parquet（逐场景流式处理，图像以字节方式嵌入）
    print("Building and writing Parquet files (images embedded)...")
    stats = build_and_write(data_gen_dir, questions_dir, output_dir)

    total = stats["qrr"] + stats["trr"] + stats["fdr"]
    print(f"\n  Scenes: {stats['scenes']} (skipped: {stats['skipped_scenes']})")
    print(f"  Questions: {total:,} total")
    print(f"    QRR: {stats['qrr']:,}")
    print(f"    TRR: {stats['trr']:,}")
    print(f"    FDR: {stats['fdr']:,}")

    print("\nDone.")


if __name__ == "__main__":
    main()
