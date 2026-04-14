"""
全局距离对排序的模拟 oracle 提供者。

返回 GT 正确的比较结果，用于测试排序算法。
"""

from __future__ import annotations

from typing import Tuple

from gt_ranking import gt_comparator_for_dist_pairs, pair_key

DistPair = Tuple[str, str]


def make_mock_comparator(objects: dict, tau: float, allow_approx: bool = True):
    """创建返回 GT 答案的模拟比较器。

    Args:
        allow_approx: 是否允许返回 ~=。False 时 ~= 按实际距离差映射为 < 或 >。
    """
    def comparator_fn(pivot: DistPair, candidates: list[DistPair]):
        results = {}
        for cand in candidates:
            results[pair_key(cand)] = gt_comparator_for_dist_pairs(
                objects, cand, pivot, tau, allow_approx=allow_approx,
            )
        return results, {"prompt_tokens": 0, "completion_tokens": 0}
    return comparator_fn
