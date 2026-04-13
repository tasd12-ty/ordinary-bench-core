"""
全局距离对排序的三路快速排序。

使用 VLM 作为比较器 oracle 对所有 C(N,2) 距离对进行排序。
第 0 层 pivot 由 GT 引导（中位数）；第 1 层及以上 pivot 随机选取。
每个划分步骤 = 一次 API 调用。同层划分任务并发执行。
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple

DistPair = Tuple[str, str]  # (obj_i, obj_j)，保证 i < j


def pair_key(pair: DistPair) -> str:
    return f"{pair[0]}_{pair[1]}"


@dataclass
class RoundRecord:
    """一次划分步骤（一次 API 调用）的追踪记录。"""
    level: int
    pivot: DistPair
    candidates: list[DistPair]
    results: dict[str, str]          # pair_key -> "<" / "~=" / ">"
    partition_lt: list[DistPair]
    partition_eq: list[DistPair]
    partition_gt: list[DistPair]
    n_comparisons: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass
class SortResult:
    """全局距离对快速排序的结果。"""
    ranking: list[DistPair] = field(default_factory=list)
    tie_groups: list[list[DistPair]] = field(default_factory=list)
    rounds: list[RoundRecord] = field(default_factory=list)
    total_comparisons: int = 0
    total_api_calls: int = 0
    num_levels: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    failed: bool = False
    fail_reason: str = ""


# 比较器：(pivot_pair, candidate_pairs) -> ({pair_key: cmp}, usage)
ComparatorResult = Tuple[Dict[str, str], Dict[str, int]]
ComparatorFn = Callable[[DistPair, List[DistPair]], ComparatorResult]


def _select_pivot(
    subarray: list[DistPair],
    level: int,
    gt_ranking: list[DistPair] | None,
) -> DistPair:
    """选择 pivot：第 0 层使用 GT 中位数，其余层随机选择。"""
    if level == 0 and gt_ranking is not None:
        return gt_ranking[len(gt_ranking) // 2]
    return random.choice(subarray)


def _partition(
    subarray: list[DistPair],
    pivot: DistPair,
    results: dict[str, str],
) -> Tuple[list[DistPair], list[DistPair], list[DistPair]]:
    """基于比较结果的三路划分。"""
    lt, eq, gt = [], [pivot], []
    for p in subarray:
        if p == pivot:
            continue
        cmp = results.get(pair_key(p), "~=")
        if cmp == "<":
            lt.append(p)
        elif cmp == ">":
            gt.append(p)
        else:
            eq.append(p)
    return lt, eq, gt


def quicksort_global(
    all_pairs: list[DistPair],
    comparator_fn: ComparatorFn,
    gt_ranking: list[DistPair] | None = None,
    executor=None,
) -> SortResult:
    """使用三路快速排序和并发划分对所有距离对进行排序。

    Args:
        all_pairs: 待排序的所有 C(N,2) 距离对。
        comparator_fn: (pivot, candidates) -> (results, usage)。
        gt_ranking: 用于第 0 层 pivot 选择的 GT 排序。
        executor: 用于同层并发划分的 ThreadPoolExecutor。

    Returns:
        包含全局排序、平局组和追踪信息的 SortResult。
    """
    from concurrent.futures import Future

    result = SortResult()

    if len(all_pairs) <= 1:
        result.ranking = list(all_pairs)
        result.tie_groups = [list(all_pairs)] if all_pairs else []
        return result

    # 段：(排序位置, 平局组) — 最后组装为排序结果
    segments: list[Tuple[int, list[DistPair]]] = []

    # 逐层 BFS
    current_level: list[Tuple[list[DistPair], int]] = [(list(all_pairs), 0)]
    level = 0

    while current_level:
        # 收集需要 API 调用的任务（大小 > 1）
        tasks: list[Tuple[list[DistPair], int, DistPair, list[DistPair]]] = []
        next_level: list[Tuple[list[DistPair], int]] = []

        for subarray, order_base in current_level:
            if len(subarray) == 0:
                continue
            if len(subarray) == 1:
                segments.append((order_base, subarray))
                continue

            pivot = _select_pivot(subarray, level, gt_ranking)
            candidates = [p for p in subarray if p != pivot]
            tasks.append((subarray, order_base, pivot, candidates))

        if not tasks:
            break

        # 触发所有划分任务（有 executor 且多任务时并发执行）
        if executor and len(tasks) > 1:
            futures: list[Tuple[Future, list[DistPair], int, DistPair]] = []
            for subarray, order_base, pivot, candidates in tasks:
                fut = executor.submit(comparator_fn, pivot, candidates)
                futures.append((fut, subarray, order_base, pivot))

            for fut, subarray, order_base, pivot in futures:
                try:
                    cmp_results, usage = fut.result()
                except Exception as e:
                    result.failed = True
                    result.fail_reason = f"Level {level}: {e}"
                    segments.append((order_base, subarray))
                    continue
                _process_partition(
                    result, segments, next_level,
                    subarray, order_base, pivot, cmp_results, usage, level,
                )
        else:
            # 顺序执行
            for subarray, order_base, pivot, candidates in tasks:
                try:
                    cmp_results, usage = comparator_fn(pivot, candidates)
                except Exception as e:
                    result.failed = True
                    result.fail_reason = f"Level {level}: {e}"
                    segments.append((order_base, subarray))
                    continue
                _process_partition(
                    result, segments, next_level,
                    subarray, order_base, pivot, cmp_results, usage, level,
                )

        current_level = next_level
        level += 1

    result.num_levels = level

    # 从有序段组装最终排序
    segments.sort(key=lambda x: x[0])
    for _, group in segments:
        result.ranking.extend(group)
        result.tie_groups.append(group)

    return result


def _process_partition(
    result: SortResult,
    segments: list,
    next_level: list,
    subarray: list[DistPair],
    order_base: int,
    pivot: DistPair,
    cmp_results: dict[str, str],
    usage: dict[str, int],
    level: int,
):
    """处理一次划分结果：更新计数器，调度子问题。"""
    candidates = [p for p in subarray if p != pivot]
    lt, eq, gt = _partition(subarray, pivot, cmp_results)

    n_cmp = len(candidates)
    p_tok = usage.get("prompt_tokens", 0)
    c_tok = usage.get("completion_tokens", 0)

    result.total_comparisons += n_cmp
    result.total_api_calls += 1
    result.total_prompt_tokens += p_tok
    result.total_completion_tokens += c_tok

    result.rounds.append(RoundRecord(
        level=level,
        pivot=pivot,
        candidates=candidates,
        results=cmp_results,
        partition_lt=lt, partition_eq=eq, partition_gt=gt,
        n_comparisons=n_cmp,
        prompt_tokens=p_tok, completion_tokens=c_tok,
    ))

    # 调度子问题
    lt_base = order_base
    eq_base = order_base + len(lt)
    gt_base = eq_base + len(eq)

    if len(lt) > 1:
        next_level.append((lt, lt_base))
    elif len(lt) == 1:
        segments.append((lt_base, lt))

    segments.append((eq_base, eq))

    if len(gt) > 1:
        next_level.append((gt, gt_base))
    elif len(gt) == 1:
        segments.append((gt_base, gt))
