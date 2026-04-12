#!/usr/bin/env python3
"""生成 VRF 验证问题 — 每场景固定 K 道复合 TRUE/FALSE 题。

用法:
    python generate_questions.py --data ../datasets/test-data
    python generate_questions.py --data ../data-gen/output --split n04 --K 20
"""

import argparse
import json
import logging
import sys
from pathlib import Path

_VLM_DIR = Path(__file__).resolve().parent.parent / "VLM-test"
if str(_VLM_DIR) not in sys.path:
    sys.path.insert(0, str(_VLM_DIR))

from extraction import parse_objects, object_description, load_scene
from question_bank import make_batches
from vrf_question_bank import enumerate_vrf

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def process_scene(
    scene_path: Path, K: int, claims_per_q: int, tau: float, batch_size: int,
) -> dict:
    scene = load_scene(str(scene_path))
    scene_id = scene["scene_id"]
    objects = parse_objects(scene)

    obj_list = [
        {"id": oid, "desc": object_description(objects[oid])}
        for oid in sorted(objects.keys())
    ]

    questions = enumerate_vrf(
        objects, K=K, claims_per_q=claims_per_q, tau=tau, scene_id=scene_id,
    )
    batches = make_batches(questions, batch_size)

    n_true = sum(1 for q in questions if q["gt_answer"])
    n_false = len(questions) - n_true

    return {
        "scene_id": scene_id,
        "image_path": f"images/single_view/{scene_id}.png",
        "objects": obj_list,
        "n_objects": len(objects),
        "question_type": "vrf",
        "total_questions": len(questions),
        "n_batches": len(batches),
        "tau": tau,
        "vrf_config": {
            "target_k": K,
            "actual_k": len(questions),
            "claims_per_question": claims_per_q,
            "n_true": n_true,
            "n_false": n_false,
            "seed": scene_id,
        },
        "batches": batches,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate VRF verification questions")
    parser.add_argument("--data", "-d", required=True, help="Data directory (with scenes/)")
    parser.add_argument("--output", "-o", default="./output", help="Output directory")
    parser.add_argument("--split", "-s", default=None, help="Filter by split prefix (e.g. n04)")
    parser.add_argument("--K", type=int, default=20, help="Questions per scene (default: 20)")
    parser.add_argument("--claims-per-question", type=int, default=3, help="Sub-claims per question")
    parser.add_argument("--batch-size", "-b", type=int, default=20, help="Questions per batch")
    parser.add_argument("--tau", type=float, default=0.10, help="Tolerance parameter")
    args = parser.parse_args()

    data_dir = Path(args.data)
    scenes_dir = data_dir / "scenes"
    if not scenes_dir.exists():
        logger.error("Scenes directory not found: %s", scenes_dir)
        sys.exit(1)

    output_dir = Path(args.output) / "questions" / "vrf"
    output_dir.mkdir(parents=True, exist_ok=True)

    scene_files = sorted(scenes_dir.glob("*.json"))
    if args.split:
        scene_files = [f for f in scene_files if f.stem.startswith(args.split)]

    if not scene_files:
        logger.error("No scene files found in %s", scenes_dir)
        sys.exit(1)

    logger.info("Processing %d scenes (K=%d, claims=%d, tau=%.2f)",
                len(scene_files), args.K, args.claims_per_question, args.tau)

    total_questions = 0
    for scene_path in scene_files:
        result = process_scene(
            scene_path, args.K, args.claims_per_question, args.tau, args.batch_size,
        )
        scene_id = result["scene_id"]
        out_path = output_dir / f"{scene_id}.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)

        n_true = result["vrf_config"]["n_true"]
        n_false = result["vrf_config"]["n_false"]
        total_questions += result["total_questions"]
        logger.info("  %s: %d objects, %d questions (TRUE=%d, FALSE=%d)",
                    scene_id, result["n_objects"], result["total_questions"], n_true, n_false)

    print(f"\nDone! {len(scene_files)} scenes, {total_questions} questions total")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
