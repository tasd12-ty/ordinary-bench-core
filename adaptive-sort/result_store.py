"""
全局距离对排序评测的结果存储。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from sorting import SortResult, RoundRecord, pair_key

logger = logging.getLogger(__name__)


def sort_result_to_dict(sr: SortResult) -> dict:
    """将 SortResult 转换为可序列化的字典。"""
    return {
        "ranking": [list(p) for p in sr.ranking],
        "tie_groups": [[list(p) for p in g] for g in sr.tie_groups],
        "total_comparisons": sr.total_comparisons,
        "total_api_calls": sr.total_api_calls,
        "num_levels": sr.num_levels,
        "total_prompt_tokens": sr.total_prompt_tokens,
        "total_completion_tokens": sr.total_completion_tokens,
        "failed": sr.failed,
        "fail_reason": sr.fail_reason,
        "rounds": [
            {
                "level": r.level,
                "pivot": list(r.pivot),
                "candidates": [list(p) for p in r.candidates],
                "results": r.results,
                "partition_lt": [list(p) for p in r.partition_lt],
                "partition_eq": [list(p) for p in r.partition_eq],
                "partition_gt": [list(p) for p in r.partition_gt],
                "n_comparisons": r.n_comparisons,
                "prompt_tokens": r.prompt_tokens,
                "completion_tokens": r.completion_tokens,
            }
            for r in sr.rounds
        ],
    }


def save_scene_result(
    output_dir: Path,
    scene_id: str,
    model: str,
    n_objects: int,
    sort_result_dict: dict,
    scores: dict,
    gt_ranking: list,
    gt_tie_groups: list,
) -> Path:
    """保存单场景结果。"""
    scenes_dir = output_dir / "scenes"
    scenes_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "scene_id": scene_id,
        "model": model,
        "n_objects": n_objects,
        "n_pairs": scores.get("n_pairs", 0),
        "gt_ranking": [list(p) for p in gt_ranking],
        "gt_tie_groups": [[list(p) for p in g] for g in gt_tie_groups],
        "vlm_result": sort_result_dict,
        "scores": scores,
    }

    out_path = scenes_dir / f"{scene_id}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    logger.info("Saved: %s", out_path)
    return out_path


def save_summary(
    output_dir: Path,
    model: str,
    scene_results: list[dict],
) -> Path:
    """保存跨场景汇总。"""
    output_dir.mkdir(parents=True, exist_ok=True)

    if not scene_results:
        summary = {"model": model, "n_scenes": 0, "scenes": []}
    else:
        n = len(scene_results)
        summary = {
            "model": model,
            "n_scenes": n,
            "mean_kendall_tau": round(
                sum(s.get("kendall_tau", 0) for s in scene_results) / n, 4
            ),
            "mean_pairwise_accuracy": round(
                sum(s.get("pairwise_accuracy", 0) for s in scene_results) / n, 4
            ),
            "total_comparisons": sum(s.get("total_comparisons", 0) for s in scene_results),
            "total_exhaustive": sum(s.get("exhaustive_comparisons", 0) for s in scene_results),
            "total_api_calls": sum(s.get("total_api_calls", 0) for s in scene_results),
            "total_prompt_tokens": sum(s.get("prompt_tokens", 0) for s in scene_results),
            "total_completion_tokens": sum(s.get("completion_tokens", 0) for s in scene_results),
            "scenes": scene_results,
        }
        if summary["total_exhaustive"] > 0:
            summary["overall_comparison_savings"] = round(
                1 - summary["total_comparisons"] / summary["total_exhaustive"], 4
            )

    out_path = output_dir / "summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info("Saved summary: %s", out_path)
    return out_path
