#!/usr/bin/env python3
"""
生成 QRR shared_anchor 问题 —— 输出到独立 qrr_sa 目录。

仅生成共享锚点（三元比较）的 QRR 问题，不影响现有 qrr/ 目录。

用法：
    python generate_qrr_sa.py --data ../datasets/test-data --output ../datasets/test-data
    python generate_qrr_sa.py --data ../data-gen/output --split n04
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from extraction import parse_objects, object_description, load_scene
from question_bank import enumerate_qrr, make_batches

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def process_scene(scene_path: Path, batch_size: int, tau: float) -> dict:
    scene = load_scene(str(scene_path))
    scene_id = scene["scene_id"]
    objects = parse_objects(scene)

    obj_list = [
        {"id": obj_id, "desc": object_description(objects[obj_id])}
        for obj_id in sorted(objects.keys())
    ]

    questions = enumerate_qrr(
        objects, tau=tau,
        include_disjoint=False,
        include_shared_anchor=True,
    )
    batches = make_batches(questions, batch_size)

    return {
        "scene_id": scene_id,
        "image_path": f"images/single_view/{scene_id}.png",
        "objects": obj_list,
        "n_objects": len(objects),
        "question_type": "qrr",
        "variant": "shared_anchor",
        "total_questions": len(questions),
        "n_batches": len(batches),
        "tau": tau,
        "batches": batches,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate QRR shared_anchor questions to qrr_sa/ directory"
    )
    parser.add_argument("--data", "-d", required=True, help="Path to data directory (with scenes/ subdirectory)")
    parser.add_argument("--output", "-o", default=None, help="Output directory (default: same as --data)")
    parser.add_argument("--split", "-s", default=None, help="Only process scenes from this split (e.g. n04)")
    parser.add_argument("--batch-size", "-b", type=int, default=20, help="Questions per batch (default: 20)")
    parser.add_argument("--tau", type=float, default=0.10, help="Tolerance parameter (default: 0.10)")

    args = parser.parse_args()

    data_dir = Path(args.data)
    scenes_dir = data_dir / "scenes"
    if not scenes_dir.exists():
        logger.error(f"Scenes directory not found: {scenes_dir}")
        sys.exit(1)

    output_dir = Path(args.output) if args.output else data_dir
    qrr_sa_dir = output_dir / "questions" / "qrr_sa"
    qrr_sa_dir.mkdir(parents=True, exist_ok=True)

    scene_files = sorted(scenes_dir.glob("*.json"))
    if args.split:
        scene_files = [f for f in scene_files if f.stem.startswith(args.split)]

    if not scene_files:
        logger.error(f"No scene files found in {scenes_dir}")
        sys.exit(1)

    logger.info(f"Generating QRR shared_anchor for {len(scene_files)} scenes (batch_size={args.batch_size}, tau={args.tau})")

    total_questions = 0
    for scene_path in scene_files:
        result = process_scene(scene_path, args.batch_size, args.tau)
        scene_id = result["scene_id"]

        with open(qrr_sa_dir / f"{scene_id}.json", "w") as f:
            json.dump(result, f, indent=2)

        total_questions += result["total_questions"]
        logger.info(f"  {scene_id}: {result['n_objects']} objects, {result['total_questions']} shared_anchor questions")

    print(f"\nDone! {len(scene_files)} scenes, {total_questions} QRR shared_anchor questions")
    print(f"Output: {qrr_sa_dir}")


if __name__ == "__main__":
    main()
