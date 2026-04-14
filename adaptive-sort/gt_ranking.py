"""
全局距离对的真值（GT）排序。

计算场景中所有 C(N,2) 距离对的 GT 排序，
并提供用于模拟 oracle 测试的比较器函数。
"""

from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple

_VLM_TEST = str(Path(__file__).resolve().parent.parent / "VLM-test")
if _VLM_TEST not in sys.path:
    sys.path.append(_VLM_TEST)

from dsl.predicates import MetricType, METRIC_FUNCTIONS
from dsl.comparators import Comparator, compare

DistPair = Tuple[str, str]


class StrictApproxComparisonError(ValueError):
    """Raised when strict binary comparison sees an approximate GT relation."""


def pair_key(pair: DistPair) -> str:
    """距离对的稳定字符串键。假设 pair[0] < pair[1]。"""
    return f"{pair[0]}_{pair[1]}"


def compute_gt_global_ranking(
    objects: dict,
    tau: float = 0.10,
    allow_approx: bool = True,
) -> Tuple[List[DistPair], List[List[DistPair]]]:
    """计算所有 C(N,2) 距离对的 GT 排序，从短到长。

    Args:
        objects: {obj_id: obj_dict} 物体字典。
        tau: 容差参数。
        allow_approx: 是否允许 ~= 平局组。False 时每个距离对独立一组。

    Returns:
        (ranking, tie_groups)
        ranking: 按距离升序排列的 (obj_i, obj_j) 列表
        tie_groups: 在 tau 容差范围内的平局组
    """
    metric_func = METRIC_FUNCTIONS[MetricType.DIST_3D]
    obj_ids = sorted(objects.keys())

    # 计算所有成对距离
    pair_dists: list[tuple[DistPair, float]] = []
    for a, b in combinations(obj_ids, 2):
        d = metric_func(objects[a], objects[b])
        pair_dists.append(((a, b), d))

    # 按距离排序，然后按 pair key 排序以保证稳定性
    pair_dists.sort(key=lambda x: (x[1], x[0]))

    ranking = [p for p, _ in pair_dists]
    distances = [d for _, d in pair_dists]

    # 基于 tau 容差构建平局组
    tie_groups: list[list[DistPair]] = []
    if ranking:
        if allow_approx:
            current_group = [ranking[0]]
            for i in range(1, len(ranking)):
                cmp = compare(distances[i - 1], distances[i], tau)
                if cmp == Comparator.APPROX:
                    current_group.append(ranking[i])
                else:
                    tie_groups.append(current_group)
                    current_group = [ranking[i]]
            tie_groups.append(current_group)
        else:
            # 不允许 ~=：每个距离对独立一组
            tie_groups = [[p] for p in ranking]

    return ranking, tie_groups


def gt_comparator_for_dist_pairs(
    objects: dict,
    candidate: DistPair,
    pivot: DistPair,
    tau: float = 0.10,
    allow_approx: bool = True,
) -> str:
    """GT 比较器：d(candidate) vs d(pivot) → "<" / "~=" / ">"。

    当 allow_approx=False 时，~= 是数据/设置冲突，直接报错。
    """
    metric_func = METRIC_FUNCTIONS[MetricType.DIST_3D]
    d_cand = metric_func(objects[candidate[0]], objects[candidate[1]])
    d_pivot = metric_func(objects[pivot[0]], objects[pivot[1]])
    result = str(compare(d_cand, d_pivot, tau))
    if not allow_approx and result == "~=":
        raise StrictApproxComparisonError(
            "Strict binary comparison encountered approximate GT distances: "
            f"candidate={candidate} d={d_cand:.6g}, pivot={pivot} d={d_pivot:.6g}, tau={tau}"
        )
    return result
