#!/usr/bin/env python3
"""
从场景和 master bank 生成 subset ablation QRR 问题。

问题生成是完全确定性的：相同 scene JSON + tau → 相同问题。
无需从 HuggingFace 下载，clone 仓库后直接运行即可。

用法：
    cd datasets/test-data/subset_ablation
    python generate_questions.py

    # 自定义参数
    python generate_questions.py --tau 0.10 --batch-size 20
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

# 添加 VLM-test 到 sys.path 以复用 question_bank
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
VLM_TEST_DIR = PROJECT_ROOT / "VLM-test"
if str(VLM_TEST_DIR) not in sys.path:
    sys.path.insert(0, str(VLM_TEST_DIR))

from question_bank import make_batches


def classify_questions(master_bank: dict, subset_object_ids: set) -> list:
    """对 master bank 中的每个问题，标记是否可答（answerable）。"""
    result = []
    for q in master_bank["questions"]:
        involved = set(q["involved_objects"])
        missing = involved - subset_object_ids

        tagged = {**q}
        if not missing:
            tagged["answerable"] = True
        else:
            tagged["answerable"] = False
            tagged["missing_objects"] = sorted(missing)
            tagged["n_missing"] = len(missing)

        result.append(tagged)
    return result


def build_subset_question_file(
    subset_id: str,
    parent_scene_id: str,
    subset_objects: list,
    all_questions: list,
    master_bank: dict,
    batch_size: int = 20,
) -> dict:
    """构建子集问题文件。"""
    subset_obj_ids = {obj["id"] for obj in subset_objects}
    obj_list = [o for o in master_bank["objects"] if o["id"] in subset_obj_ids]

    n_answerable = sum(1 for q in all_questions if q["answerable"])
    n_refusal = sum(1 for q in all_questions if not q["answerable"])

    batches = make_batches(all_questions, batch_size=batch_size)

    return {
        "scene_id": subset_id,
        "parent_scene_id": parent_scene_id,
        "image_path": f"images/single_view/{subset_id}.png",
        "objects": obj_list,
        "all_objects_in_parent": master_bank["objects"],
        "n_objects": len(obj_list),
        "n_objects_parent": master_bank["n_objects"],
        "question_type": "qrr",
        "total_questions": len(all_questions),
        "n_answerable": n_answerable,
        "n_refusal": n_refusal,
        "batches": batches,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate subset ablation QRR questions from scenes and master banks"
    )
    parser.add_argument("--tau", type=float, default=0.10, help="Tolerance (default: 0.10)")
    parser.add_argument("--batch-size", type=int, default=20, help="Batch size (default: 20)")
    args = parser.parse_args()

    manifest_path = SCRIPT_DIR / "manifest.json"
    master_dir = SCRIPT_DIR / "master_questions"
    scenes_dir = SCRIPT_DIR / "scenes"
    questions_dir = SCRIPT_DIR / "questions" / "qrr"
    questions_dir.mkdir(parents=True, exist_ok=True)

    with open(manifest_path) as f:
        manifest = json.load(f)

    total_answerable = 0
    total_refusal = 0
    total_subsets = 0

    for parent_id, parent_data in manifest["parent_scenes"].items():
        master_path = master_dir / f"{parent_id}.json"
        if not master_path.exists():
            print(f"WARNING: missing master bank {master_path}")
            continue

        with open(master_path) as f:
            master_bank = json.load(f)

        for subset_info in parent_data["subsets"]:
            subset_id = subset_info["subset_id"]
            subset_obj_ids = set(subset_info["object_ids"])

            scene_path = scenes_dir / f"{subset_id}.json"
            if not scene_path.exists():
                print(f"WARNING: missing scene {scene_path}")
                continue

            with open(scene_path) as f:
                subset_scene = json.load(f)

            all_questions = classify_questions(master_bank, subset_obj_ids)

            question_file = build_subset_question_file(
                subset_id, parent_id,
                subset_scene["objects"],
                all_questions, master_bank,
                batch_size=args.batch_size,
            )

            out_path = questions_dir / f"{subset_id}.json"
            with open(out_path, "w") as f:
                json.dump(question_file, f, indent=2)

            total_answerable += question_file["n_answerable"]
            total_refusal += question_file["n_refusal"]
            total_subsets += 1

        print(f"  {parent_id}: {len(parent_data['subsets'])} subsets")

    total = total_answerable + total_refusal
    pct = total_answerable / total * 100 if total else 0
    print(f"\nDone! {total_subsets} subsets, {total:,} questions")
    print(f"  Answerable: {total_answerable:,} ({pct:.1f}%)")
    print(f"  Refusal: {total_refusal:,} ({100 - pct:.1f}%)")
    print(f"  Output: {questions_dir}")


if __name__ == "__main__":
    main()
