"""
Step 3b: 子集问题分配 — 全量模式。

对每个子集，使用父场景的 **全量** master bank QRR 问题。
VLM 对每题回答 < / ~= / > / N/A（物体不存在）。
每题标记 answerable=true/false，用于评分。

用法:
    python assign_subset_questions.py \
        --manifest output/manifest.json \
        --master-dir output/master_questions \
        --output-dir output
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
VLM_TEST_DIR = PROJECT_ROOT / "VLM-test"
if str(VLM_TEST_DIR) not in sys.path:
    sys.path.insert(0, str(VLM_TEST_DIR))

from question_bank import make_batches


def classify_questions(master_bank: dict, subset_object_ids: set) -> list:
    """
    对 master bank 中的每个问题，标记是否可答。

    Returns:
        标记后的问题列表（全量，含 answerable 和 missing_objects 字段）。
    """
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
    """构建子集问题文件。objects 列表只含子集中的 4 个物体。"""
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
    parser = argparse.ArgumentParser(description="全量子集问题分配")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--master-dir", required=True)
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--batch-size", type=int, default=20)
    args = parser.parse_args()

    master_dir = Path(args.master_dir)
    output_dir = Path(args.output_dir)
    questions_dir = output_dir / "questions" / "qrr"
    questions_dir.mkdir(parents=True, exist_ok=True)
    scenes_dir = output_dir / "scenes"

    with open(args.manifest) as f:
        manifest = json.load(f)

    total_answerable = 0
    total_refusal = 0
    total_subsets = 0
    per_split = defaultdict(lambda: {"ans": 0, "ref": 0, "sub": 0, "total_q": 0})

    for parent_id, parent_data in manifest["parent_scenes"].items():
        master_path = master_dir / f"{parent_id}.json"
        if not master_path.exists():
            print(f"WARNING: missing master bank {master_path}")
            continue

        with open(master_path) as f:
            master_bank = json.load(f)

        split = parent_id.rsplit("_", 1)[0]

        for subset_info in parent_data["subsets"]:
            subset_id = subset_info["subset_id"]
            subset_obj_ids = set(subset_info["object_ids"])

            scene_path = scenes_dir / f"{subset_id}.json"
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

            n_a = question_file["n_answerable"]
            n_r = question_file["n_refusal"]
            total_answerable += n_a
            total_refusal += n_r
            total_subsets += 1
            per_split[split]["ans"] += n_a
            per_split[split]["ref"] += n_r
            per_split[split]["sub"] += 1
            per_split[split]["total_q"] += len(all_questions)

    print(f"\n{'Split':<8} {'Subsets':>8} {'Q/subset':>10} {'Answerable':>12} {'Refusal':>10} {'Ans%':>8}")
    print("-" * 60)
    for split in sorted(per_split):
        s = per_split[split]
        q_per = s["total_q"] // s["sub"] if s["sub"] else 0
        pct = s["ans"] / (s["ans"] + s["ref"]) * 100 if (s["ans"] + s["ref"]) else 0
        print(f"{split:<8} {s['sub']:>8} {q_per:>10} {s['ans']:>12} {s['ref']:>10} {pct:>7.1f}%")

    total = total_answerable + total_refusal
    pct = total_answerable / total * 100 if total else 0
    print(f"\nTotal: {total_subsets} subsets, {total} questions "
          f"({total_answerable} answerable {pct:.1f}%, {total_refusal} refusal)")


if __name__ == "__main__":
    main()
