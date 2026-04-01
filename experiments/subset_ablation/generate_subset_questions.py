"""
Step 3: 为每个子集场景生成 QRR 问题。

输出 v2 格式，与现有 evaluator 完全兼容。
同时生成 question_mapping.json 追踪问题跨子集的重叠关系。

用法:
    python generate_subset_questions.py --manifest output/manifest.json \
        --parent-scenes ../../datasets/test-data/scenes --output-dir output
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

# 添加 VLM-test 到 sys.path 以复用现有模块
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
VLM_TEST_DIR = PROJECT_ROOT / "VLM-test"
if str(VLM_TEST_DIR) not in sys.path:
    sys.path.insert(0, str(VLM_TEST_DIR))

from question_bank import enumerate_qrr, make_batches
from extraction import parse_objects, object_description


def generate_questions_for_subset(
    subset_scene: dict,
    tau: float = 0.10,
    batch_size: int = 20,
) -> dict:
    """
    为一个子集场景生成 QRR 问题，输出 v2 格式字典。
    """
    # 将 scene JSON 的 objects 转换为 DSL 格式
    objects = parse_objects(subset_scene)

    # 枚举 QRR 问题
    questions = enumerate_qrr(objects, tau=tau)

    # 构建物体描述列表
    obj_list = []
    for obj_data in subset_scene["objects"]:
        obj_list.append({
            "id": obj_data["id"],
            "desc": object_description(obj_data),
        })

    # 分批
    batches = make_batches(questions, batch_size=batch_size)

    return {
        "scene_id": subset_scene["scene_id"],
        "parent_scene_id": subset_scene.get("parent_scene_id", ""),
        "image_path": f"images/single_view/{subset_scene['scene_id']}.png",
        "objects": obj_list,
        "n_objects": len(obj_list),
        "question_type": "qrr",
        "total_questions": len(questions),
        "n_batches": len(batches),
        "tau": tau,
        "batches": batches,
    }


def build_question_key(q: dict) -> str:
    """
    构建问题的规范化 key，用于跨子集去重和映射。
    key = variant + sorted pairs (或 anchor + pair)
    """
    variant = q.get("variant", "disjoint")
    if variant == "disjoint":
        p1 = tuple(sorted(q["pair1"]))
        p2 = tuple(sorted(q["pair2"]))
        # 确保 pair1 < pair2 的字典序
        if p1 > p2:
            p1, p2 = p2, p1
        return f"disjoint|{p1[0]},{p1[1]}|{p2[0]},{p2[1]}"
    else:
        anchor = q["anchor"]
        others = sorted([q["pair1"][1] if q["pair1"][0] == anchor else q["pair1"][0],
                         q["pair2"][1] if q["pair2"][0] == anchor else q["pair2"][0]])
        return f"shared_anchor|{anchor}|{others[0]},{others[1]}"


def main():
    parser = argparse.ArgumentParser(description="为子集场景生成 QRR 问题")
    parser.add_argument("--manifest", required=True, help="manifest.json 路径")
    parser.add_argument("--parent-scenes", required=True, help="父场景 scenes 目录")
    parser.add_argument("--output-dir", default="output", help="输出目录")
    parser.add_argument("--tau", type=float, default=0.10, help="容差参数")
    parser.add_argument("--batch-size", type=int, default=20, help="每批问题数")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    questions_dir = output_dir / "questions" / "qrr"
    questions_dir.mkdir(parents=True, exist_ok=True)
    scenes_dir = output_dir / "scenes"
    parent_scenes_dir = Path(args.parent_scenes)

    with open(args.manifest) as f:
        manifest = json.load(f)

    # question_mapping: parent_scene_id -> {question_key -> {info + subset_ids}}
    question_mapping = {}
    total_questions = 0
    total_subsets = 0

    for parent_id, parent_data in manifest["parent_scenes"].items():
        scene_mapping = {}

        for subset_info in parent_data["subsets"]:
            subset_id = subset_info["subset_id"]
            scene_path = scenes_dir / f"{subset_id}.json"

            if not scene_path.exists():
                print(f"WARNING: missing scene {scene_path}")
                continue

            with open(scene_path) as f:
                subset_scene = json.load(f)

            # 生成问题
            question_file = generate_questions_for_subset(
                subset_scene, tau=args.tau, batch_size=args.batch_size
            )

            # 写入问题 JSON
            out_path = questions_dir / f"{subset_id}.json"
            with open(out_path, "w") as f:
                json.dump(question_file, f, indent=2)

            total_questions += question_file["total_questions"]
            total_subsets += 1

            # 追踪问题映射
            for batch in question_file["batches"]:
                for q in batch["questions"]:
                    qkey = build_question_key(q)
                    if qkey not in scene_mapping:
                        scene_mapping[qkey] = {
                            "variant": q.get("variant", "disjoint"),
                            "pair1": q["pair1"],
                            "pair2": q["pair2"],
                            "gt_comparator": q["gt_comparator"],
                            "subset_ids": [],
                            "subset_qids": {},
                        }
                        if "anchor" in q:
                            scene_mapping[qkey]["anchor"] = q["anchor"]
                    scene_mapping[qkey]["subset_ids"].append(subset_id)
                    scene_mapping[qkey]["subset_qids"][subset_id] = q["qid"]

        question_mapping[parent_id] = scene_mapping

    # 写入 question_mapping
    mapping_path = output_dir / "question_mapping.json"
    with open(mapping_path, "w") as f:
        json.dump(question_mapping, f, indent=2)

    # 统计
    total_unique = sum(len(v) for v in question_mapping.values())
    overlap_counts = defaultdict(int)
    for parent_id, mapping in question_mapping.items():
        for qkey, qinfo in mapping.items():
            n_subsets = len(qinfo["subset_ids"])
            overlap_counts[n_subsets] += 1

    print(f"Generated questions for {total_subsets} subsets")
    print(f"Total questions (with duplicates): {total_questions}")
    print(f"Unique questions: {total_unique}")
    print(f"Overlap distribution:")
    for n, count in sorted(overlap_counts.items()):
        print(f"  appears in {n} subset(s): {count} questions")


if __name__ == "__main__":
    main()
