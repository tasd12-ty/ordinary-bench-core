#!/usr/bin/env python3
# 已废弃：请改用 generate_questions_v2.py（按题型分目录存储）。
"""
从 data-gen 场景 JSON 生成 VLM 评估问题。

两种输出模式：
  1. questions/        — 批量模式：枚举所有 QRR/TRR 问题并分批
  2. extraction_tasks/ — 自治模式：输出对象列表与真值，供后续评估使用

用法：
    python generate_questions.py --data ../data-gen/output
    python generate_questions.py --data ../data-gen/output --split n04 --batch-size 10
    python generate_questions.py --data ../data-gen/output --counts  # 仅显示问题数量统计
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from extraction import parse_objects, object_description, extract_gt, load_scene
from question_bank import enumerate_qrr, enumerate_trr, enumerate_fdr, make_batches, question_counts

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def process_scene(scene_path: Path, batch_size: int, tau: float) -> dict:
    """处理单个场景：提取真值、枚举问题、划分批次。"""
    scene = load_scene(str(scene_path))
    scene_id = scene["scene_id"]
    objects = parse_objects(scene)

    # 构建 VLM 提示所需的对象描述列表
    obj_list = []
    for obj_id in sorted(objects.keys()):
        obj = objects[obj_id]
        obj_list.append({
            "id": obj_id,
            "desc": object_description(obj),
        })

    # 枚举所有问题
    qrr_questions = enumerate_qrr(objects, tau=tau)
    trr_questions = enumerate_trr(objects, use_3d=True)
    fdr_questions = enumerate_fdr(objects, tau=tau)
    qrr_disjoint = sum(1 for q in qrr_questions if q.get("variant") == "disjoint")
    qrr_shared_anchor = sum(1 for q in qrr_questions if q.get("variant") == "shared_anchor")

    all_questions = qrr_questions + trr_questions + fdr_questions
    batches = make_batches(all_questions, batch_size)

    # 批量模式输出
    batch_output = {
        "scene_id": scene_id,
        "image_path": f"images/single_view/{scene_id}.png",
        "objects": obj_list,
        "n_objects": len(objects),
        "total_qrr_disjoint": qrr_disjoint,
        "total_qrr_shared_anchor": qrr_shared_anchor,
        "total_qrr": len(qrr_questions),
        "total_trr": len(trr_questions),
        "total_fdr": len(fdr_questions),
        "total_questions": len(all_questions),
        "n_batches": len(batches),
        "tau": tau,
        "batches": batches,
    }

    # 自治模式输出
    gt = extract_gt(scene, tau=tau)
    extraction_output = {
        "scene_id": scene_id,
        "image_path": f"images/single_view/{scene_id}.png",
        "objects": obj_list,
        "n_objects": len(objects),
        "tau": tau,
        "ground_truth": gt,
    }

    return batch_output, extraction_output


def main():
    parser = argparse.ArgumentParser(
        description="Generate VLM evaluation questions from scene JSONs"
    )
    parser.add_argument(
        "--data", "-d", default=None,
        help="Path to data-gen output directory",
    )
    parser.add_argument(
        "--output", "-o", default="./output",
        help="Output directory (default: ./output)",
    )
    parser.add_argument(
        "--split", "-s", default=None,
        help="Only process scenes from this split (e.g. n04)",
    )
    parser.add_argument(
        "--batch-size", "-b", type=int, default=20,
        help="Questions per batch (default: 20)",
    )
    parser.add_argument(
        "--tau", type=float, default=0.10,
        help="Tolerance parameter (default: 0.10)",
    )
    parser.add_argument(
        "--counts", action="store_true",
        help="Just print question count table and exit",
    )

    args = parser.parse_args()

    if args.counts:
        print(
            f"{'Objects':>8} {'QRR-D':>8} {'QRR-SA':>8} {'QRR':>8} "
            f"{'TRR':>8} {'FDR':>8} {'Total':>8}"
        )
        print("-" * 68)
        for n in range(4, 11):
            c = question_counts(n)
            print(
                f"{n:>8} {c['n_qrr_disjoint']:>8} {c['n_qrr_shared_anchor']:>8} "
                f"{c['n_qrr']:>8} {c['n_trr']:>8} {c['n_fdr']:>8} {c['total']:>8}"
            )
        return

    if not args.data:
        logger.error("--data is required (path to data-gen output directory)")
        sys.exit(1)

    data_dir = Path(args.data)
    scenes_dir = data_dir / "scenes"
    if not scenes_dir.exists():
        logger.error(f"Scenes directory not found: {scenes_dir}")
        sys.exit(1)

    output_dir = Path(args.output)
    questions_dir = output_dir / "questions"
    extraction_dir = output_dir / "extraction_tasks"
    questions_dir.mkdir(parents=True, exist_ok=True)
    extraction_dir.mkdir(parents=True, exist_ok=True)

    # 查找场景文件
    scene_files = sorted(scenes_dir.glob("*.json"))
    if args.split:
        scene_files = [f for f in scene_files if f.stem.startswith(args.split)]

    if not scene_files:
        logger.error(f"No scene files found in {scenes_dir}")
        sys.exit(1)

    logger.info(f"Processing {len(scene_files)} scenes (batch_size={args.batch_size}, tau={args.tau})")

    total_qrr = 0
    total_qrr_disjoint = 0
    total_qrr_shared_anchor = 0
    total_trr = 0
    total_fdr = 0
    total_batches = 0

    for scene_path in scene_files:
        batch_out, extract_out = process_scene(
            scene_path, args.batch_size, args.tau
        )
        scene_id = batch_out["scene_id"]

        # 保存批量模式输出
        with open(questions_dir / f"{scene_id}.json", 'w') as f:
            json.dump(batch_out, f, indent=2)

        # 保存自治模式输出
        with open(extraction_dir / f"{scene_id}.json", 'w') as f:
            json.dump(extract_out, f, indent=2)

        total_qrr += batch_out["total_qrr"]
        total_qrr_disjoint += batch_out["total_qrr_disjoint"]
        total_qrr_shared_anchor += batch_out["total_qrr_shared_anchor"]
        total_trr += batch_out["total_trr"]
        total_fdr += batch_out["total_fdr"]
        total_batches += batch_out["n_batches"]

        logger.info(
            f"  {scene_id}: {batch_out['n_objects']} objects, "
            f"{batch_out['total_qrr']} QRR "
            f"(disjoint={batch_out['total_qrr_disjoint']}, "
            f"shared_anchor={batch_out['total_qrr_shared_anchor']}) + "
            f"{batch_out['total_trr']} TRR + "
            f"{batch_out['total_fdr']} FDR = "
            f"{batch_out['total_questions']} questions, "
            f"{batch_out['n_batches']} batches"
        )

    # 汇总统计
    summary = {
        "n_scenes": len(scene_files),
        "batch_size": args.batch_size,
        "tau": args.tau,
        "total_qrr_disjoint": total_qrr_disjoint,
        "total_qrr_shared_anchor": total_qrr_shared_anchor,
        "total_qrr": total_qrr,
        "total_trr": total_trr,
        "total_fdr": total_fdr,
        "total_questions": total_qrr + total_trr + total_fdr,
        "total_batches": total_batches,
    }
    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone! {len(scene_files)} scenes processed")
    print(
        f"  QRR: {total_qrr} "
        f"(disjoint={total_qrr_disjoint}, shared_anchor={total_qrr_shared_anchor}), "
        f"TRR: {total_trr}, FDR: {total_fdr}, "
        f"Total: {total_qrr + total_trr + total_fdr}"
    )
    print(f"  Batches: {total_batches} (size={args.batch_size})")
    print(f"  Output: {output_dir}")


if __name__ == "__main__":
    main()
