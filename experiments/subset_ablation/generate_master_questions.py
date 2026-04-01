"""
Step 3a: 全场景 QRR Master Bank 生成。

对每个父场景生成全量 QRR 问题，包括:
  - disjoint QRR (直接)
  - shared_anchor QRR (直接)
  - FDR 分解为 shared_anchor QRR (去重后补充)

统一 qid 编号 (mqrr_XXXX)，附带 involved_objects 和 source 字段。

用法:
    python generate_master_questions.py \
        --scenes-dir ../../datasets/test-data/scenes \
        --output-dir output \
        --splits n06,n07,n08,n09,n10 --max-scenes 2
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
VLM_TEST_DIR = PROJECT_ROOT / "VLM-test"
if str(VLM_TEST_DIR) not in sys.path:
    sys.path.insert(0, str(VLM_TEST_DIR))

from question_bank import enumerate_qrr, enumerate_fdr
from extraction import parse_objects, object_description


def _shared_anchor_key(anchor: str, pair1: list, pair2: list) -> str:
    """规范化 shared_anchor 问题的 key，用于去重。"""
    obj_a = pair1[1] if pair1[0] == anchor else pair1[0]
    obj_b = pair2[1] if pair2[0] == anchor else pair2[0]
    others = tuple(sorted([obj_a, obj_b]))
    return f"sa|{anchor}|{others[0]}|{others[1]}"


def _disjoint_key(pair1: list, pair2: list) -> str:
    """规范化 disjoint 问题的 key。"""
    p1 = tuple(sorted(pair1))
    p2 = tuple(sorted(pair2))
    if p1 > p2:
        p1, p2 = p2, p1
    return f"dj|{p1[0]}|{p1[1]}|{p2[0]}|{p2[1]}"


def _involved_objects(q: dict) -> list:
    """提取问题涉及的所有物体 ID。"""
    objs = set(q["pair1"]) | set(q["pair2"])
    return sorted(objs)


def decompose_fdr_to_qrr(fdr_questions: list, tau: float = 0.10) -> list:
    """
    将 FDR ranking 分解为 shared_anchor QRR pairwise 比较。

    FDR: anchor=A, ranking=[B, C, D] (由近到远)
    → d(A,B) < d(A,C), d(A,B) < d(A,D), d(A,C) < d(A,D)

    尊重 tie_groups: 同组内物体使用 "~=" 而非 "<"。
    """
    derived = []
    for fdr in fdr_questions:
        anchor = fdr["anchor"]
        ranking = fdr["gt_ranking"]
        tie_groups = fdr.get("gt_tie_groups", [])

        # 构建 object -> tie_group_id 映射
        obj_to_group = {}
        for gid, group in enumerate(tie_groups):
            for obj_id in group:
                obj_to_group[obj_id] = gid

        for i in range(len(ranking)):
            for j in range(i + 1, len(ranking)):
                nearer = ranking[i]
                farther = ranking[j]

                # 如果两者在同一 tie group 中, comparator 为 ~=
                g_i = obj_to_group.get(nearer)
                g_j = obj_to_group.get(farther)
                if g_i is not None and g_i == g_j:
                    comparator = "~="
                else:
                    comparator = "<"

                derived.append({
                    "type": "qrr",
                    "variant": "shared_anchor",
                    "anchor": anchor,
                    "pair1": sorted([anchor, nearer]),
                    "pair2": sorted([anchor, farther]),
                    "gt_comparator": comparator,
                    "source": "fdr_decomposition",
                    "source_fdr_qid": fdr["qid"],
                })
    return derived


def generate_master_bank(scene: dict, tau: float = 0.10) -> dict:
    """生成一个场景的全量 QRR master bank。"""
    objects = parse_objects(scene)
    scene_id = scene["scene_id"]

    # 1. 直接 QRR (disjoint + shared_anchor)
    direct_qrr = enumerate_qrr(objects, tau=tau)

    # 2. FDR → 分解为 shared_anchor QRR
    fdr_questions = enumerate_fdr(objects, tau=tau)
    fdr_derived = decompose_fdr_to_qrr(fdr_questions, tau=tau)

    # 3. 去重合并：用规范化 key 追踪
    seen_keys = {}
    master_questions = []

    # 先加入直接 QRR（优先保留）
    for q in direct_qrr:
        q["source"] = "qrr_direct"
        q["involved_objects"] = _involved_objects(q)

        if q["variant"] == "disjoint":
            key = _disjoint_key(q["pair1"], q["pair2"])
        else:
            key = _shared_anchor_key(q["anchor"], q["pair1"], q["pair2"])

        if key not in seen_keys:
            seen_keys[key] = len(master_questions)
            master_questions.append(q)

    # 再加入 FDR 分解的（仅补充缺失的）
    n_fdr_added = 0
    n_fdr_dup = 0
    for q in fdr_derived:
        key = _shared_anchor_key(q["anchor"], q["pair1"], q["pair2"])
        if key not in seen_keys:
            q["involved_objects"] = _involved_objects(q)
            seen_keys[key] = len(master_questions)
            master_questions.append(q)
            n_fdr_added += 1
        else:
            n_fdr_dup += 1

    # 统一编号（保留 source_fdr_qid 用于审计追踪）
    for i, q in enumerate(master_questions):
        q["qid"] = f"mqrr_{i+1:04d}"

    # 构建物体描述
    obj_list = [
        {"id": obj["id"], "desc": object_description(obj)}
        for obj in scene["objects"]
    ]

    # 统计
    stats = {
        "disjoint": sum(1 for q in master_questions if q["variant"] == "disjoint"),
        "shared_anchor_direct": sum(
            1 for q in master_questions
            if q["variant"] == "shared_anchor" and q["source"] == "qrr_direct"
        ),
        "shared_anchor_from_fdr": n_fdr_added,
        "fdr_duplicates_skipped": n_fdr_dup,
        "total": len(master_questions),
    }

    return {
        "scene_id": scene_id,
        "n_objects": len(scene["objects"]),
        "objects": obj_list,
        "questions": master_questions,
        "stats": stats,
    }


def main():
    parser = argparse.ArgumentParser(description="生成全场景 QRR Master Bank")
    parser.add_argument("--scenes-dir", required=True)
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--splits", default=None, help="逗号分隔 split 前缀")
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--tau", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    scenes_dir = Path(args.scenes_dir)
    output_dir = Path(args.output_dir) / "master_questions"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 发现场景
    all_files = sorted(scenes_dir.glob("*.json"))
    splits = args.splits.split(",") if args.splits else None
    if splits:
        all_files = [f for f in all_files if any(f.stem.startswith(s) for s in splits)]

    # 按 split 采样
    if args.max_scenes:
        from collections import defaultdict as dd
        by_split = dd(list)
        for f in all_files:
            prefix = f.stem.rsplit("_", 1)[0]
            by_split[prefix].append(f)
        scene_files = []
        for prefix in sorted(by_split):
            files = by_split[prefix]
            sampled = random.sample(files, min(args.max_scenes, len(files)))
            scene_files.extend(sampled)
        scene_files.sort()
    else:
        scene_files = all_files

    # 也读取 manifest 来确保只处理已枚举的场景
    manifest_path = Path(args.output_dir) / "manifest.json"
    manifest_scenes = None
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
        manifest_scenes = set(manifest["parent_scenes"].keys())
        scene_files = [f for f in scene_files if f.stem in manifest_scenes]

    total_questions = 0
    for scene_file in scene_files:
        with open(scene_file) as f:
            scene = json.load(f)

        bank = generate_master_bank(scene, tau=args.tau)
        out_path = output_dir / f"{scene['scene_id']}.json"
        with open(out_path, "w") as f:
            json.dump(bank, f, indent=2)

        s = bank["stats"]
        print(f"{scene['scene_id']}: {s['total']} questions "
              f"(dj={s['disjoint']}, sa_direct={s['shared_anchor_direct']}, "
              f"sa_fdr={s['shared_anchor_from_fdr']}, fdr_dup={s['fdr_duplicates_skipped']})")
        total_questions += s["total"]

    print(f"\nTotal: {len(scene_files)} scenes, {total_questions} master questions")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
