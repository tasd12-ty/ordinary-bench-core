"""
最小反馈弧集 (Minimum Feedback Arc Set) 算法
============================================

在 QRR 约束构成的有向图中，「环」意味着距离偏序存在矛盾：
    例如 d(A,B) < d(C,D) < d(E,F) < d(A,B) 不可能同时成立。

本模块找出**最少需要移除的约束边**来消除所有环（NP-hard 问题的贪心近似）。
被移除的边即为「最可疑的错误约束」，对应的原始问题将被重新询问。

图的构建：
    - 节点 = 规范化物体对 tuple(sorted(pair))，如 ("obj_0", "obj_1")
    - 有向边 = 距离偏序方向：若 d(pair_a) < d(pair_b)，则 pair_a → pair_b
    - '~=' 约束不产生边（不参与环的形成）
    - 每条边记录对应的原始约束 dict（含 qid、source_type 等溯源信息）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import networkx as nx


@dataclass
class FASResult:
    """反馈弧集计算结果。

    Attributes:
        edges_removed: 被移除的约束条目列表（含 qid 等溯源信息）
        original_edge_count: 原始有向图总边数
        remaining_edge_count: 移除后剩余边数
        is_acyclic: 处理后图是否已无环
    """
    edges_removed: List[dict] = field(default_factory=list)
    original_edge_count: int = 0
    remaining_edge_count: int = 0
    is_acyclic: bool = True


def _canonical_pair(pair) -> Tuple[str, ...]:
    """将物体对规范化为排序元组，确保 (B, A) 和 (A, B) 映射到同一节点。"""
    return tuple(sorted(pair))


def build_qrr_digraph(qrr_entries: List[dict]) -> Tuple[nx.DiGraph, Dict]:
    """从 QRR 约束列表构建有向图。

    Args:
        qrr_entries: QRR 约束列表，每项含 pair1、pair2、comparator 字段。
                     可以是直接 QRR 约束或 FDR 分解产生的约束。

    Returns:
        (G, edge_to_entries):
            G — networkx 有向图
            edge_to_entries — {(src, dst): [entry, ...]} 边到原始约束的映射
    """
    G = nx.DiGraph()
    edge_to_entries: Dict[Tuple, List[dict]] = {}

    for entry in qrr_entries:
        cmp = entry.get("comparator", "")
        # 仅处理严格不等式；'~=' 不会产生偏序边
        if cmp not in ("<", ">"):
            continue

        pa = _canonical_pair(entry["pair1"])
        pb = _canonical_pair(entry["pair2"])

        # 确定边方向：较小距离 → 较大距离
        if cmp == "<":
            src, dst = pa, pb     # d(pa) < d(pb)
        else:
            src, dst = pb, pa     # d(pb) < d(pa)

        key = (src, dst)
        if not G.has_edge(src, dst):
            G.add_edge(src, dst)
            edge_to_entries[key] = []
        edge_to_entries[key].append(entry)

    return G, edge_to_entries


def compute_fas(qrr_entries: List[dict]) -> FASResult:
    """计算最小反馈弧集（贪心近似）。

    算法：
        1. 构建 QRR 约束有向图
        2. 若已无环，直接返回
        3. 反复执行：
           a. 找到一个环（nx.find_cycle）
           b. 在环中选择「度数得分最高」的边移除
              得分 = 出发节点出度 + 到达节点入度（高度数边更可能参与多个环）
           c. 将对应的约束条目记录到 edges_removed
        4. 直到图无环

    复杂度：O(E × C)，E 为边数，C 为环数。对于实际数据规模完全可接受。

    Returns:
        FASResult，其中 edges_removed 包含所有被移除的原始约束 dict。
    """
    G, edge_to_entries = build_qrr_digraph(qrr_entries)
    original_count = G.number_of_edges()

    # 已经无环，无需移除
    if nx.is_directed_acyclic_graph(G):
        return FASResult(
            original_edge_count=original_count,
            remaining_edge_count=original_count,
            is_acyclic=True,
        )

    removed: List[dict] = []

    while not nx.is_directed_acyclic_graph(G):
        try:
            cycle_edges = nx.find_cycle(G)
        except nx.NetworkXNoCycle:
            break

        # 启发式：选环中度数得分最高的边移除
        # 高度数的边参与更多约束关系，移除它更可能同时打破多个环
        best_edge = None
        best_score = -1
        for u, v, *_ in cycle_edges:
            score = G.in_degree(v) + G.out_degree(u)
            if score > best_score:
                best_score = score
                best_edge = (u, v)

        if best_edge is None:
            best_edge = (cycle_edges[0][0], cycle_edges[0][1])

        u, v = best_edge
        G.remove_edge(u, v)

        # 收集该边对应的所有原始约束条目
        entries = edge_to_entries.get((u, v), [])
        removed.extend(entries)

    return FASResult(
        edges_removed=removed,
        original_edge_count=original_count,
        remaining_edge_count=G.number_of_edges(),
        is_acyclic=True,
    )
