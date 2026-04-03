"""
子集消融 → 父场景重建转换器。

将多个子集的 VLM 评估结果聚合为父场景格式，
直接对接现有重建管线（reconstruct_from_scoring），无需修改重建代码。

核心逻辑：
  1. 遍历同一 parent_scene 的所有子集结果
  2. 只取 answerable=True 的问题（跳过 N/A）
  3. 同一 qid 在多个子集中都可回答时，用多数投票聚合
  4. 输出与 run_batch.py 格式一致的 scoring_result + questions

用法：
    cd experiments/subset_ablation

    # 从评估结果生成父场景格式
    uv run python aggregate_to_parent.py \
        --results-dir output/results/gpt4o_multiview/scenes \
        --questions-dir output/questions/qrr \
        --master-dir output/master_questions \
        --scenes-dir ../../data-gen/output/scenes \
        --output-dir output/aggregated/gpt4o_multiview

    # 然后用现有重建管线
    cd ../../VLM-test
    uv run python -m reconstruct.batch_reconstruct \
        --results-dir ../experiments/subset_ablation/output/aggregated/gpt4o_multiview \
        --questions-dir ../experiments/subset_ablation/output/master_questions \
        --scenes-dir ../data-gen/output/scenes \
        --output-dir ../experiments/subset_ablation/output/reconstruction/gpt4o_multiview

输出结构：
    output/aggregated/{run}/
      {parent_scene_id}.json    — 与 run_batch.py 输出格式一致
      summary.json              — 聚合统计
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def load_subset_results(results_dir: Path) -> Dict[str, List[dict]]:
    """加载子集评估结果，按 parent_scene_id 分组。"""
    by_parent = defaultdict(list)
    for f in sorted(results_dir.glob("*.json")):
        with open(f) as fh:
            r = json.load(fh)
        if r.get("status") == "no_image":
            continue
        pid = r.get("parent_scene_id", "")
        if pid:
            by_parent[pid].append(r)
    return dict(by_parent)


def load_master_questions(master_dir: Path, parent_id: str) -> Optional[List[dict]]:
    """加载父场景的 master question bank。"""
    p = master_dir / f"{parent_id}.json"
    if not p.exists():
        return None
    with open(p) as f:
        data = json.load(f)
    questions = []
    for batch in data.get("batches", []):
        questions.extend(batch.get("questions", []))
    return questions


def aggregate_parent(
    parent_id: str,
    subset_results: List[dict],
    vote_threshold: float = 0.5,
) -> Tuple[dict, dict]:
    """聚合一个父场景的所有子集结果。

    Returns:
        (scoring_result, aggregation_stats)

    scoring_result 格式与 run_batch.py 的评分输出一致：
        {"per_question": [...], "qrr_correct": N, "qrr_total": N, ...}
    """
    # qid → list of (predicted, gt_comparator) from answerable questions
    qid_predictions = defaultdict(list)
    qid_gt = {}

    for sr in subset_results:
        for pq in sr.get("per_question", []):
            if not pq.get("answerable", True):
                continue  # 跳过 N/A
            qid = pq["qid"]
            pred = pq.get("predicted")
            if pred is None or pred == "N/A":
                continue  # VLM 在可回答题上也可能没给答案
            qid_predictions[qid].append(pred)
            if qid not in qid_gt and "gt" in pq:
                qid_gt[qid] = pq["gt"]

    # 多数投票
    per_question = []
    qrr_correct = 0
    qrr_total = 0

    for qid in sorted(qid_predictions.keys()):
        preds = qid_predictions[qid]
        gt = qid_gt.get(qid)

        counter = Counter(preds)
        winner, count = counter.most_common(1)[0]
        ratio = count / len(preds)

        correct = (winner == gt) if gt else None

        pq_entry = {
            "qid": qid,
            "type": "qrr",
            "predicted": winner if ratio >= vote_threshold else None,
            "gt": gt,
            "correct": correct if correct is not None else False,
            "vote_counts": dict(counter),
            "n_votes": len(preds),
            "majority_ratio": round(ratio, 4),
        }
        per_question.append(pq_entry)

        if pq_entry["predicted"] is not None:
            qrr_total += 1
            if correct:
                qrr_correct += 1

    scoring_result = {
        "per_question": per_question,
        "qrr_correct": qrr_correct,
        "qrr_total": qrr_total,
        "trr_total": 0,
        "fdr_total": 0,
    }

    stats = {
        "parent_scene_id": parent_id,
        "n_subsets": len(subset_results),
        "n_unique_qids": len(qid_predictions),
        "n_voted": sum(1 for pq in per_question if pq["predicted"] is not None),
        "qrr_correct": qrr_correct,
        "qrr_total": qrr_total,
        "qrr_acc": round(qrr_correct / qrr_total, 4) if qrr_total else 0,
        "avg_votes_per_qid": round(
            sum(len(v) for v in qid_predictions.values()) / len(qid_predictions), 2
        ) if qid_predictions else 0,
    }

    return scoring_result, stats


def build_scene_result(
    parent_id: str,
    scoring_result: dict,
    model: str,
    n_objects: int,
) -> dict:
    """构建与 run_batch.py 输出格式一致的场景结果。"""
    return {
        "scene_id": parent_id,
        "model": model,
        "n_objects": n_objects,
        "source": "subset_ablation_aggregation",
        "scores": scoring_result,
    }


def main():
    parser = argparse.ArgumentParser(
        description="将子集消融评估结果聚合为父场景格式，对接重建管线"
    )
    parser.add_argument("--results-dir", required=True,
                        help="子集评估结果目录 (scenes/)")
    parser.add_argument("--master-dir", required=True,
                        help="master_questions/ 目录")
    parser.add_argument("--scenes-dir", default=None,
                        help="父场景 JSON 目录（用于获取 n_objects）")
    parser.add_argument("--output-dir", required=True,
                        help="聚合结果输出目录")
    parser.add_argument("--model", default="unknown",
                        help="模型名称标记")
    parser.add_argument("--vote-threshold", type=float, default=0.5,
                        help="多数投票阈值 (默认 0.5)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    master_dir = Path(args.master_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    by_parent = load_subset_results(results_dir)

    if not by_parent:
        print("No subset results found.")
        sys.exit(1)

    print(f"Found {sum(len(v) for v in by_parent.values())} subset results "
          f"for {len(by_parent)} parent scenes")

    all_stats = []

    for parent_id in sorted(by_parent):
        subset_results = by_parent[parent_id]

        # 获取 n_objects
        n_objects = subset_results[0].get("n_objects_parent", 0)
        if args.scenes_dir:
            scene_path = Path(args.scenes_dir) / f"{parent_id}.json"
            if scene_path.exists():
                with open(scene_path) as f:
                    scene = json.load(f)
                n_objects = scene.get("n_objects", n_objects)

        scoring_result, stats = aggregate_parent(
            parent_id, subset_results, args.vote_threshold
        )

        scene_result = build_scene_result(
            parent_id, scoring_result, args.model, n_objects
        )

        out_path = output_dir / f"{parent_id}.json"
        with open(out_path, "w") as f:
            json.dump(scene_result, f, indent=2)

        all_stats.append(stats)
        print(f"  {parent_id}: {stats['n_subsets']} subsets, "
              f"{stats['n_unique_qids']} qids, "
              f"acc={stats['qrr_acc']:.4f} ({stats['qrr_correct']}/{stats['qrr_total']})")

    # 汇总
    total_correct = sum(s["qrr_correct"] for s in all_stats)
    total_total = sum(s["qrr_total"] for s in all_stats)
    summary = {
        "model": args.model,
        "vote_threshold": args.vote_threshold,
        "n_parents": len(all_stats),
        "overall_acc": round(total_correct / total_total, 4) if total_total else 0,
        "total_qrr_correct": total_correct,
        "total_qrr_total": total_total,
        "per_parent": all_stats,
    }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nOverall: {total_correct}/{total_total} = "
          f"{total_correct/total_total:.4f}" if total_total else "N/A")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
