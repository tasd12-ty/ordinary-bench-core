#!/usr/bin/env python3
"""
三维 VLM 评估问题生成脚本。

按题型分目录存储 QRR、TRR（3D）和 FDR 问题：
    output/questions/qrr/{scene_id}.json
    output/questions/trr/{scene_id}.json
    output/questions/fdr/{scene_id}.json

用法：
    python generate_questions_3d.py --data ../data-gen-3d/output
    python generate_questions_3d.py --data ../data-gen-3d/output --split n04
    python generate_questions_3d.py --counts
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from extraction import parse_objects, object_description, extract_gt, load_scene
from question_bank import (
    enumerate_qrr, enumerate_trr_3d, enumerate_fdr,
    make_batches, question_counts,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# 题型枚举函数映射
# TRR 使用三维版本（含仰角），QRR 和 FDR 不变
QUESTION_TYPES = {
    "qrr": lambda objects, tau: enumerate_qrr(objects, tau=tau),
    "trr": lambda objects, _tau: enumerate_trr_3d(objects),
    "fdr": lambda objects, tau: enumerate_fdr(objects, tau=tau),
}


def process_scene(scene_path: Path, batch_size: int, tau: float) -> dict:
    """处理单个场景：按题型枚举问题并分批。"""
    scene = load_scene(str(scene_path))
    scene_id = scene["scene_id"]
    objects = parse_objects(scene)

    # 构建物体描述列表
    obj_list = []
    for obj_id in sorted(objects.keys()):
        obj_list.append({
            "id": obj_id,
            "desc": object_description(objects[obj_id]),
        })

    gt = extract_gt(scene, tau=tau)

    results = {}
    for qtype, enum_func in QUESTION_TYPES.items():
        questions = enum_func(objects, tau)
        batches = make_batches(questions, batch_size)
        batch_output = {
            "scene_id": scene_id,
            "image_path": f"images/single_view/{scene_id}.png",
            "objects": obj_list,
            "n_objects": len(objects),
            "question_type": qtype,
            "total_questions": len(questions),
            "n_batches": len(batches),
            "tau": tau,
            "is_3d": True,
            "batches": batches,
        }
        if qtype == "qrr":
            batch_output["total_qrr_disjoint"] = sum(
                1 for q in questions if q.get("variant") == "disjoint"
            )
            batch_output["total_qrr_shared_anchor"] = sum(
                1 for q in questions if q.get("variant") == "shared_anchor"
            )
        results[qtype] = {
            "batch_output": batch_output,
            "extraction_output": {
                "scene_id": scene_id,
                "image_path": f"images/single_view/{scene_id}.png",
                "objects": obj_list,
                "n_objects": len(objects),
                "tau": tau,
                "question_type": qtype,
                "ground_truth": gt.get(qtype, []),
            },
        }

    return results


def main():
    parser = argparse.ArgumentParser(
        description="生成三维 VLM 评估问题（按题型分目录存储）"
    )
    parser.add_argument(
        "--data", "-d", default=None,
        help="data-gen-3d 输出目录路径",
    )
    parser.add_argument(
        "--output", "-o", default="./output",
        help="输出目录（默认：./output）",
    )
    parser.add_argument(
        "--split", "-s", default=None,
        help="仅处理指定 split 的场景（如 n04）",
    )
    parser.add_argument(
        "--batch-size", "-b", type=int, default=20,
        help="每批问题数（默认：20）",
    )
    parser.add_argument(
        "--tau", type=float, default=0.10,
        help="容差参数（默认：0.10）",
    )
    parser.add_argument(
        "--counts", action="store_true",
        help="仅打印问题数量表格并退出",
    )

    args = parser.parse_args()

    if args.counts:
        print(
            f"{'物体数':>8} {'QRR-D':>8} {'QRR-SA':>8} {'QRR':>8} "
            f"{'TRR':>8} {'FDR':>8} {'总计':>8}"
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
        logger.error("需要 --data 参数（data-gen-3d 输出目录路径）")
        sys.exit(1)

    data_dir = Path(args.data)
    scenes_dir = data_dir / "scenes"
    if not scenes_dir.exists():
        logger.error(f"场景目录不存在：{scenes_dir}")
        sys.exit(1)

    output_dir = Path(args.output)

    # 创建按题型分目录的输出结构
    q_dirs = {}
    e_dirs = {}
    for qtype in QUESTION_TYPES:
        q_dirs[qtype] = output_dir / "questions" / qtype
        q_dirs[qtype].mkdir(parents=True, exist_ok=True)
        e_dirs[qtype] = output_dir / "extraction_tasks" / qtype
        e_dirs[qtype].mkdir(parents=True, exist_ok=True)

    # 查找场景文件
    scene_files = sorted(scenes_dir.glob("*.json"))
    if args.split:
        scene_files = [f for f in scene_files if f.stem.startswith(args.split)]

    if not scene_files:
        logger.error(f"未找到场景文件：{scenes_dir}")
        sys.exit(1)

    logger.info(f"处理 {len(scene_files)} 个场景（batch_size={args.batch_size}, tau={args.tau}）")

    totals = {qtype: {"questions": 0, "batches": 0} for qtype in QUESTION_TYPES}

    for scene_path in scene_files:
        results = process_scene(scene_path, args.batch_size, args.tau)
        scene_id = results["qrr"]["batch_output"]["scene_id"]

        parts = []
        for qtype in QUESTION_TYPES:
            batch_out = results[qtype]["batch_output"]
            extract_out = results[qtype]["extraction_output"]

            with open(q_dirs[qtype] / f"{scene_id}.json", "w") as f:
                json.dump(batch_out, f, indent=2)
            with open(e_dirs[qtype] / f"{scene_id}.json", "w") as f:
                json.dump(extract_out, f, indent=2)

            totals[qtype]["questions"] += batch_out["total_questions"]
            totals[qtype]["batches"] += batch_out["n_batches"]
            if qtype == "qrr":
                disjoint = batch_out.get("total_qrr_disjoint", 0)
                shared_anchor = batch_out.get("total_qrr_shared_anchor", 0)
                parts.append(
                    f"{batch_out['total_questions']} QRR "
                    f"(disjoint={disjoint}, shared_anchor={shared_anchor})"
                )
            else:
                parts.append(f"{batch_out['total_questions']} {qtype.upper()}")

        n_obj = results["qrr"]["batch_output"]["n_objects"]
        total_q = sum(results[qt]["batch_output"]["total_questions"] for qt in QUESTION_TYPES)
        logger.info(f"  {scene_id}: {n_obj} 个物体, {' + '.join(parts)} = {total_q} 个问题")

    # 汇总
    grand_total = sum(t["questions"] for t in totals.values())
    summary = {
        "n_scenes": len(scene_files),
        "batch_size": args.batch_size,
        "tau": args.tau,
        "is_3d": True,
        "per_type": {qt: totals[qt] for qt in QUESTION_TYPES},
        "total_questions": grand_total,
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n完成！处理了 {len(scene_files)} 个场景")
    for qtype in QUESTION_TYPES:
        print(f"  {qtype.upper()}: {totals[qtype]['questions']} 个问题, {totals[qtype]['batches']} 个批次")
    print(f"  总计: {grand_total}")
    print(f"  输出: {output_dir}")


if __name__ == "__main__":
    main()
