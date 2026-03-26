"""使用修正后的 TRR 真值重新评分所有 VLM 结果。

读取经 fix_trr_gt.py 修正的 TRR 问题文件，更新逐场景评分，
并为每个模型重新生成 summary.json。

QRR 和 FDR 分数保持不变。

用法：
    python rescore_results.py [--questions-dir DIR] [--results-dir DIR]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Set

REPO_ROOT = Path(__file__).resolve().parent
API_ROOT = REPO_ROOT / "API-test"

sys.path.insert(0, str(API_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from dsl.predicates import hour_to_quadrant
from scoring import (
    aggregate_batch_results,
    score_trr_adjacent,
    score_trr_hour,
    score_trr_quadrant,
)


def load_corrected_gt(questions_dir: Path) -> Dict[str, Dict[str, dict]]:
    """加载修正后的 TRR 真值：{scene_id: {qid: {gt_hour, gt_quadrant}}}。"""
    trr_dir = questions_dir / "trr"
    gt: Dict[str, Dict[str, dict]] = {}

    if not trr_dir.is_dir():
        print(f"Warning: {trr_dir} not found")
        return gt

    for fpath in sorted(trr_dir.glob("*.json")):
        scene_id = fpath.stem
        with open(fpath) as f:
            doc = json.load(f)

        scene_gt: Dict[str, dict] = {}
        for batch in doc.get("batches", []):
            for q in batch.get("questions", []):
                scene_gt[q["qid"]] = {
                    "gt_hour": q["gt_hour"],
                    "gt_quadrant": q["gt_quadrant"],
                }
        gt[scene_id] = scene_gt

    return gt


def rescore_scene(scene_result: dict, scene_gt: Dict[str, dict]) -> dict:
    """使用修正后的真值重新评分场景结果中的 TRR 条目。

    直接修改 scene_result 并返回。
    """
    scores = scene_result["scores"]
    per_question = scores["per_question"]

    trr_hour_correct = 0
    trr_quad_correct = 0
    trr_adj_correct = 0
    trr_total = 0

    for pq in per_question:
        if pq["type"] != "trr":
            continue

        trr_total += 1
        qid = pq["qid"]
        predicted = pq.get("predicted", -1)
        corrected = scene_gt.get(qid)

        if corrected is None:
            # 无修正真值可用，保留原值不变
            if pq.get("hour_correct"):
                trr_hour_correct += 1
            if pq.get("quadrant_correct"):
                trr_quad_correct += 1
            if pq.get("adjacent_correct"):
                trr_adj_correct += 1
            continue

        new_gt_hour = corrected["gt_hour"]
        new_gt_quadrant = corrected["gt_quadrant"]

        h = score_trr_hour(predicted, new_gt_hour)
        qu = score_trr_quadrant(predicted, new_gt_quadrant)
        a = score_trr_adjacent(predicted, new_gt_hour)

        pq["gt_hour"] = new_gt_hour
        pq["hour_correct"] = h
        pq["quadrant_correct"] = qu
        pq["adjacent_correct"] = a

        if h:
            trr_hour_correct += 1
        if qu:
            trr_quad_correct += 1
        if a:
            trr_adj_correct += 1

    scores["trr_hour_correct"] = trr_hour_correct
    scores["trr_quadrant_correct"] = trr_quad_correct
    scores["trr_adjacent_correct"] = trr_adj_correct
    scores["trr_total"] = trr_total

    return scene_result


def process_model(model_dir: Path, corrected_gt: Dict[str, Dict[str, dict]]) -> dict:
    """对某模型的所有场景重新评分并重新生成汇总文件。"""
    scenes_dir = model_dir / "scenes"
    if not scenes_dir.is_dir():
        return {}

    scene_files = sorted(scenes_dir.glob("*.json"))
    scene_results: List[dict] = []
    updated = 0

    for fpath in scene_files:
        with open(fpath) as f:
            scene_result = json.load(f)

        scene_id = scene_result["scene_id"]
        scene_gt = corrected_gt.get(scene_id, {})

        if scene_gt:
            rescore_scene(scene_result, scene_gt)
            updated += 1

            with open(fpath, "w") as f:
                json.dump(scene_result, f, indent=2, ensure_ascii=False)

        # 确保 aggregate_batch_results 所需的所有键均存在。
        s = scene_result["scores"]
        for key in (
            "qrr_correct", "qrr_total",
            "qrr_disjoint_correct", "qrr_disjoint_total",
            "qrr_shared_anchor_correct", "qrr_shared_anchor_total",
            "trr_hour_correct", "trr_quadrant_correct", "trr_adjacent_correct", "trr_total",
            "fdr_exact_correct", "fdr_total", "missing",
            "fdr_kendall_mean", "fdr_pairwise_mean", "fdr_top1_mean",
        ):
            s.setdefault(key, 0)

        scene_results.append({
            "scene_id": scene_id,
            "scores": s,
        })

    # 重新生成汇总文件。
    summary = aggregate_batch_results(scene_results)

    # 从现有汇总文件中保留模型元数据。
    summary_path = model_dir / "summary.json"
    if summary_path.is_file():
        with open(summary_path) as f:
            old_summary = json.load(f)
        summary["model"] = old_summary.get("model", model_dir.name)
        summary["n_scenes"] = old_summary.get("n_scenes", len(scene_results))
        summary["n_failed"] = old_summary.get("n_failed", 0)
    else:
        summary["model"] = model_dir.name
        summary["n_scenes"] = len(scene_results)
        summary["n_failed"] = 0

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return {
        "model": model_dir.name,
        "scenes_updated": updated,
        "trr_hour_acc": summary["overall"].get("trr_hour_accuracy", 0),
        "trr_quad_acc": summary["overall"].get("trr_quadrant_accuracy", 0),
        "trr_adj_acc": summary["overall"].get("trr_adjacent_accuracy", 0),
        "trr_total": summary["overall"].get("trr_total", 0),
    }


def main():
    parser = argparse.ArgumentParser(description="Re-score VLM results with corrected TRR GT")
    parser.add_argument(
        "--questions-dir",
        default=str(REPO_ROOT / "output" / "questions"),
    )
    parser.add_argument(
        "--results-dir",
        default=str(REPO_ROOT / "output" / "results"),
    )
    args = parser.parse_args()

    questions_dir = Path(args.questions_dir)
    results_dir = Path(args.results_dir)

    print("Loading corrected TRR ground truth...")
    corrected_gt = load_corrected_gt(questions_dir)
    print(f"  Loaded GT for {len(corrected_gt)} scenes")

    model_dirs = sorted(
        d for d in results_dir.iterdir()
        if d.is_dir() and (d / "scenes").is_dir()
    )
    print(f"\nFound {len(model_dirs)} model directories\n")

    print(f"{'Model':<45} {'Scenes':>7} {'TRR_n':>6} {'Hour%':>7} {'Quad%':>7} {'Adj%':>7}")
    print("-" * 85)

    for model_dir in model_dirs:
        result = process_model(model_dir, corrected_gt)
        if result:
            print(
                f"{result['model']:<45} "
                f"{result['scenes_updated']:>7} "
                f"{result['trr_total']:>6} "
                f"{result['trr_hour_acc']:>7.1%} "
                f"{result['trr_quad_acc']:>7.1%} "
                f"{result['trr_adj_acc']:>7.1%}"
            )

    print("\nDone. All scene results and summaries updated.")


if __name__ == "__main__":
    main()
