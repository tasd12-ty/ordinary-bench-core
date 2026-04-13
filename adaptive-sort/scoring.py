"""
全局距离对排序的评分。

将 VLM 生成的全局排序与 GT 排序进行比较。
"""

from __future__ import annotations

from math import comb
from typing import Tuple

DistPair = Tuple[str, str]


def pair_key(pair: DistPair) -> str:
    return f"{pair[0]}_{pair[1]}"


def exact_match(
    vlm_tie_groups: list[list[DistPair]],
    gt_tie_groups: list[list[DistPair]],
) -> bool:
    """检查 VLM 排序是否与 GT 完全匹配（考虑平局组）。"""
    if len(vlm_tie_groups) != len(gt_tie_groups):
        return False
    for vg, gg in zip(vlm_tie_groups, gt_tie_groups):
        if sorted(vg) != sorted(gg):
            return False
    return True


def kendall_tau(
    vlm_ranking: list[DistPair],
    gt_ranking: list[DistPair],
    gt_tie_groups: list[list[DistPair]],
) -> float:
    """VLM 和 GT 排序之间的 Kendall tau-b，处理 GT 平局。"""
    n = len(vlm_ranking)
    if n < 2:
        return 1.0

    gt_pos = {pair_key(p): i for i, p in enumerate(gt_ranking)}

    gt_group_map = {}
    for gi, group in enumerate(gt_tie_groups):
        for p in group:
            gt_group_map[pair_key(p)] = gi

    concordant = 0
    discordant = 0

    for i in range(n):
        for j in range(i + 1, n):
            ka = pair_key(vlm_ranking[i])
            kb = pair_key(vlm_ranking[j])
            # 跳过 GT 平局
            if gt_group_map.get(ka) == gt_group_map.get(kb):
                continue
            if gt_pos[ka] < gt_pos[kb]:
                concordant += 1
            else:
                discordant += 1

    total = concordant + discordant
    if total == 0:
        return 1.0
    return (concordant - discordant) / total


def pairwise_accuracy(
    vlm_ranking: list[DistPair],
    gt_ranking: list[DistPair],
    gt_tie_groups: list[list[DistPair]],
) -> float:
    """相对顺序正确的距离对比例。"""
    n = len(vlm_ranking)
    if n < 2:
        return 1.0

    gt_pos = {pair_key(p): i for i, p in enumerate(gt_ranking)}
    gt_group_map = {}
    for gi, group in enumerate(gt_tie_groups):
        for p in group:
            gt_group_map[pair_key(p)] = gi

    correct = 0
    total = 0

    for i in range(n):
        for j in range(i + 1, n):
            ka = pair_key(vlm_ranking[i])
            kb = pair_key(vlm_ranking[j])
            total += 1
            if gt_group_map.get(ka) == gt_group_map.get(kb):
                correct += 1
            elif gt_pos[ka] < gt_pos[kb]:
                correct += 1

    return correct / total if total > 0 else 1.0


def score_global(
    vlm_ranking: list[DistPair],
    vlm_tie_groups: list[list[DistPair]],
    gt_ranking: list[DistPair],
    gt_tie_groups: list[list[DistPair]],
    total_comparisons: int,
    total_api_calls: int,
    num_levels: int,
    total_prompt_tokens: int = 0,
    total_completion_tokens: int = 0,
    n_objects: int = 0,
) -> dict:
    """将全局距离对排序与 GT 进行评分。"""
    n_pairs = len(gt_ranking)

    # 穷举基线：QRR disjoint + shared_anchor
    exhaustive_disjoint = 3 * comb(n_objects, 4) if n_objects >= 4 else 0
    exhaustive_shared = n_objects * comb(n_objects - 1, 2) if n_objects >= 3 else 0
    exhaustive = exhaustive_disjoint + exhaustive_shared

    return {
        "n_pairs": n_pairs,
        "exact_match": exact_match(vlm_tie_groups, gt_tie_groups),
        "kendall_tau": round(kendall_tau(vlm_ranking, gt_ranking, gt_tie_groups), 4),
        "pairwise_accuracy": round(pairwise_accuracy(vlm_ranking, gt_ranking, gt_tie_groups), 4),
        "total_comparisons": total_comparisons,
        "exhaustive_comparisons": exhaustive,
        "comparison_savings": round(1 - total_comparisons / exhaustive, 4) if exhaustive > 0 else 0.0,
        "total_api_calls": total_api_calls,
        "num_levels": num_levels,
        "prompt_tokens": total_prompt_tokens,
        "completion_tokens": total_completion_tokens,
    }
