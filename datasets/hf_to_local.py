#!/usr/bin/env python3
"""
将 HuggingFace 数据集转换为本地目录格式，供 VLM 评估管线使用。

从 HuggingFace 下载数据集，提取图像为 PNG 文件，重建问题 JSON 为 v2 格式，
生成的目录结构可直接被 VLM-test/API-test/run_eval.py 使用。

用法：
    # 单视角 test split（默认）
    python hf_to_local.py --output ./hf_data

    # 指定 repo / config / split
    python hf_to_local.py --repo TYTSTQ/ordinary-bench --config qrr --split test --output ./hf_data

    # 多视角
    python hf_to_local.py --repo TYTSTQ/ordinary-bench-multiview --output ./hf_data_mv

    # 仅列出信息，不下载
    python hf_to_local.py --dry-run

依赖：
    pip install datasets pillow
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional


# ── 问题行转换 ──

def _hf_row_to_question(row: dict) -> dict:
    """将 HF 数据集的一行转换为管线所需的问题字典格式。"""
    qtype = row["question_type"]
    q: Dict[str, Any] = {
        "qid": row["qid"],
        "type": qtype,
    }

    if qtype == "qrr":
        pair1 = json.loads(row["qrr_pair1"])
        pair2 = json.loads(row["qrr_pair2"])
        variant = row.get("qrr_variant", "disjoint")
        q["variant"] = variant
        q["pair1"] = pair1
        q["pair2"] = pair2
        q["metric"] = row.get("qrr_metric", "dist3D")
        q["gt_comparator"] = row["qrr_gt_comparator"]
        # shared_anchor 时恢复 anchor 字段
        if variant == "shared_anchor":
            common = set(pair1) & set(pair2)
            if common:
                q["anchor"] = next(iter(common))

    elif qtype == "trr":
        q["target"] = row["trr_target"]
        q["ref1"] = row["trr_ref1"]
        q["ref2"] = row["trr_ref2"]
        q["gt_hour"] = int(row["trr_gt_hour"])
        q["gt_quadrant"] = int(row["trr_gt_quadrant"])
        if row.get("trr_gt_angle_deg") is not None:
            q["gt_angle_deg"] = round(float(row["trr_gt_angle_deg"]), 2)

    elif qtype == "fdr":
        q["anchor"] = row["fdr_anchor"]
        q["n_ranked"] = int(row["fdr_n_ranked"])
        q["gt_ranking"] = json.loads(row["fdr_gt_ranking"])
        if row.get("fdr_gt_distances"):
            q["gt_distances"] = json.loads(row["fdr_gt_distances"])
        if row.get("fdr_gt_tie_groups"):
            q["gt_tie_groups"] = json.loads(row["fdr_gt_tie_groups"])

    return q


def _make_batches(questions: List[dict], batch_size: int = 20) -> List[dict]:
    """将问题列表按批次分割（与 question_bank.make_batches 一致）。"""
    batches = []
    for i in range(0, len(questions), batch_size):
        chunk = questions[i:i + batch_size]
        batches.append({
            "batch_id": len(batches),
            "n_questions": len(chunk),
            "questions": chunk,
        })
    return batches


def _build_scene_question_file(
    scene_id: str,
    objects: List[dict],
    n_objects: int,
    qtype: str,
    questions: List[dict],
    batch_size: int = 20,
) -> dict:
    """组装标准 v2 问题 JSON 结构。"""
    batches = _make_batches(questions, batch_size)
    result = {
        "scene_id": scene_id,
        "image_path": f"images/single_view/{scene_id}.png",
        "objects": objects,
        "n_objects": n_objects,
        "question_type": qtype,
        "total_questions": len(questions),
        "n_batches": len(batches),
        "batches": batches,
    }
    if qtype == "qrr":
        result["total_qrr_disjoint"] = sum(
            1 for q in questions if q.get("variant") == "disjoint"
        )
        result["total_qrr_shared_anchor"] = sum(
            1 for q in questions if q.get("variant") == "shared_anchor"
        )
    return result


# ── 图像提取 ──

def _save_image(image, path: Path) -> None:
    """保存 PIL Image 为 PNG 文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(str(path), format="PNG")


def _detect_multiview(row: dict) -> bool:
    """检测是否为多视角数据集。"""
    return "view_0" in row and row["view_0"] is not None


def _extract_images(row: dict, scene_id: str, images_dir: Path, is_multiview: bool) -> None:
    """从 HF 行中提取并保存图像。"""
    if is_multiview:
        mv_dir = images_dir / "multi_view" / scene_id
        for i in range(4):
            col = f"view_{i}"
            img = row.get(col)
            if img is not None:
                _save_image(img, mv_dir / f"view_{i}.png")
    else:
        img = row.get("image")
        if img is not None:
            _save_image(img, images_dir / "single_view" / f"{scene_id}.png")


# ── 主转换流程 ──

def convert(
    repo_id: str,
    config_name: str,
    split: str,
    output_dir: Path,
    batch_size: int = 20,
) -> dict:
    """
    下载 HF 数据集并提取为本地目录结构。

    Returns: stats dict
    """
    from datasets import load_dataset

    print(f"Loading dataset: {repo_id} (config={config_name}, split={split})...")
    ds = load_dataset(repo_id, config_name, split=split)
    print(f"  Loaded {len(ds)} rows")

    images_dir = output_dir / "images"
    questions_dir = output_dir / "questions"

    # 按 scene_id 分组
    scene_rows: Dict[str, List[dict]] = defaultdict(list)
    for row in ds:
        scene_rows[row["scene_id"]].append(row)

    print(f"  {len(scene_rows)} scenes found")

    # 检测数据集类型
    first_row = ds[0]
    is_multiview = _detect_multiview(first_row)
    print(f"  Type: {'multi-view' if is_multiview else 'single-view'}")

    stats = {"scenes": 0, "images": 0, "qrr": 0, "trr": 0, "fdr": 0}

    for scene_id in sorted(scene_rows.keys()):
        rows = scene_rows[scene_id]
        first = rows[0]

        # 提取图像（每个场景只保存一次）
        _extract_images(first, scene_id, images_dir, is_multiview)
        stats["images"] += 1

        # 解析 objects
        objects = json.loads(first["objects"])
        n_objects = int(first["n_objects"])

        # 按题型分组
        by_type: Dict[str, List[dict]] = defaultdict(list)
        for row in rows:
            q = _hf_row_to_question(row)
            by_type[row["question_type"]].append(q)

        # 写入问题 JSON
        for qtype, questions in by_type.items():
            qtype_dir = questions_dir / qtype
            qtype_dir.mkdir(parents=True, exist_ok=True)

            qfile = _build_scene_question_file(
                scene_id, objects, n_objects, qtype, questions, batch_size,
            )
            out_path = qtype_dir / f"{scene_id}.json"
            with open(out_path, "w") as f:
                json.dump(qfile, f, indent=2)

            stats[qtype] += len(questions)

        stats["scenes"] += 1

    return stats


# ── CLI ──

def main():
    parser = argparse.ArgumentParser(
        description="Convert HuggingFace dataset to local format for VLM evaluation"
    )
    parser.add_argument(
        "--repo", default="TYTSTQ/ordinary-bench",
        help="HuggingFace repo ID (default: TYTSTQ/ordinary-bench)",
    )
    parser.add_argument(
        "--config", default="all",
        help="Dataset config: all, qrr, trr, fdr (default: all)",
    )
    parser.add_argument(
        "--split", default="test",
        help="Dataset split: train, test (default: test)",
    )
    parser.add_argument(
        "--output", "-o", required=True,
        help="Output directory",
    )
    parser.add_argument(
        "--batch-size", type=int, default=20,
        help="Questions per batch in output JSON (default: 20)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Only print dataset info, do not download or extract",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)

    if args.dry_run:
        from datasets import load_dataset_builder
        builder = load_dataset_builder(args.repo, args.config)
        print(f"Dataset: {args.repo}")
        print(f"Config:  {args.config}")
        print(f"Description: {builder.info.description[:200] if builder.info.description else 'N/A'}")
        if builder.info.splits:
            for split_name, split_info in builder.info.splits.items():
                print(f"  {split_name}: {split_info.num_examples:,} rows")
        print("\n[Dry run] No files written.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Converting HuggingFace dataset to local format...")
    print(f"  Repo:   {args.repo}")
    print(f"  Config: {args.config}")
    print(f"  Split:  {args.split}")
    print(f"  Output: {output_dir}")
    print()

    stats = convert(args.repo, args.config, args.split, output_dir, args.batch_size)

    total_q = stats["qrr"] + stats["trr"] + stats["fdr"]
    print(f"\nDone!")
    print(f"  Scenes: {stats['scenes']}")
    print(f"  Images: {stats['images']}")
    print(f"  Questions: {total_q:,}")
    if stats["qrr"]:
        print(f"    QRR: {stats['qrr']:,}")
    if stats["trr"]:
        print(f"    TRR: {stats['trr']:,}")
    if stats["fdr"]:
        print(f"    FDR: {stats['fdr']:,}")

    print(f"\nOutput directory structure:")
    print(f"  {output_dir}/")
    print(f"    images/")
    # Check what was created
    sv_dir = output_dir / "images" / "single_view"
    mv_dir = output_dir / "images" / "multi_view"
    if sv_dir.exists():
        n_sv = len(list(sv_dir.glob("*.png")))
        print(f"      single_view/  ({n_sv} images)")
    if mv_dir.exists():
        n_mv = len(list(mv_dir.iterdir()))
        print(f"      multi_view/   ({n_mv} scene dirs)")
    print(f"    questions/")
    q_dir = output_dir / "questions"
    for qtype in ("qrr", "trr", "fdr"):
        td = q_dir / qtype
        if td.exists():
            n_files = len(list(td.glob("*.json")))
            print(f"      {qtype}/  ({n_files} files)")

    print(f"\nTo run evaluation, use a job TOML like:")
    print(f'  [input]')
    print(f'  questions_dir = "{output_dir / "questions"}"')
    print(f'  [images]')
    if sv_dir.exists():
        print(f'  single_view_root = "{sv_dir}"')
    if mv_dir.exists():
        print(f'  multi_view_root = "{mv_dir}"')


if __name__ == "__main__":
    main()
