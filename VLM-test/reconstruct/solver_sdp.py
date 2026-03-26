"""
SDP 松弛求解器：基于 Gram 矩阵的凸优化。

理论依据: Jain et al. 2016 — 将距离比较约束转化为 Gram 矩阵上的线性约束，
用半正定规划 (SDP) 求解凸松弛，再投影到 rank-2 提取 2D 坐标。

优势: 凸优化 → 全局最优解，无局部最小值问题。
"""

import math
import numpy as np
from typing import Dict, List, Optional

import cvxpy as cp

from .constraints import QRREntry, TRREntry
from .solver import (
    SolverConfig, SolverSolution,
    select_anchors, compute_qrr_loss, compute_trr_loss, compute_sep_loss,
)
from .utils import pair_key, procrustes_align


def _gram_dist_sq_expr(G, i: int, j: int):
    """从 Gram 矩阵中提取 ||x_i - x_j||² = G_ii + G_jj - 2*G_ij。"""
    return G[i, i] + G[j, j] - 2 * G[i, j]


def solve_sdp(
    object_ids: List[str],
    qrr_entries: List[QRREntry],
    trr_entries: List[TRREntry],
    config: Optional[SolverConfig] = None,
    gt_positions: Optional[Dict[str, np.ndarray]] = None,
) -> List[SolverSolution]:
    """SDP 松弛求解器。

    将 QRR 距离序约束转化为 Gram 矩阵 G 上的线性约束，
    用 SDP 求全局最优，再投影到 rank-2 得 2D 坐标。

    注意: TRR 角度约束是非凸的，无法直接纳入 SDP。
    本 solver 仅处理 QRR 约束，TRR 在后处理阶段通过
    L-BFGS-B 精修（可选）。
    """
    if config is None:
        config = SolverConfig()

    n = len(object_ids)
    if n < 3:
        return []

    oid_to_idx = {oid: i for i, oid in enumerate(object_ids)}

    # ── 构建 SDP ──
    G = cp.Variable((n, n), symmetric=True)
    constraints = [G >> 0]  # G 半正定

    # 居中约束: sum_i x_i = 0 => sum_i G_ij = 0 for all j
    # 等价于 G @ 1 = 0
    ones = np.ones(n)
    constraints.append(G @ ones == 0)

    # QRR 约束: d(p1) < d(p2) => dist_sq(p1) < dist_sq(p2)
    # 带松弛变量，最小化违背
    margin = config.qrr_margin
    slack_vars = []

    for entry in qrr_entries:
        p1 = pair_key(*entry.pair1)
        p2 = pair_key(*entry.pair2)

        i1, j1 = oid_to_idx.get(p1[0]), oid_to_idx.get(p1[1])
        i2, j2 = oid_to_idx.get(p2[0]), oid_to_idx.get(p2[1])

        if any(x is None for x in [i1, j1, i2, j2]):
            continue

        d1_sq = _gram_dist_sq_expr(G, i1, j1)
        d2_sq = _gram_dist_sq_expr(G, i2, j2)

        s = cp.Variable(nonneg=True)
        slack_vars.append(s * entry.weight)

        if entry.comparator == "<":
            # d1 < d2 => d1² + margin ≤ d2² + s
            constraints.append(d1_sq + margin <= d2_sq + s)
        elif entry.comparator == ">":
            # d1 > d2 => d2² + margin ≤ d1² + s
            constraints.append(d2_sq + margin <= d1_sq + s)
        else:  # ~=
            # |d1² - d2²| ≤ delta_eq + s
            delta = config.qrr_delta_eq
            constraints.append(d1_sq - d2_sq <= delta + s)
            constraints.append(d2_sq - d1_sq <= delta + s)

    # 分离约束: 防止物体坍缩
    sep_eps_sq = config.sep_eps ** 2
    for i in range(n):
        for j in range(i + 1, n):
            constraints.append(_gram_dist_sq_expr(G, i, j) >= sep_eps_sq)

    # 目标: 最小化松弛变量之和（满足尽可能多的约束）
    if slack_vars:
        objective = cp.Minimize(cp.sum(slack_vars))
    else:
        objective = cp.Minimize(0)

    prob = cp.Problem(objective, constraints)

    try:
        prob.solve(solver=cp.SCS, verbose=False, max_iters=5000)
    except Exception:
        try:
            prob.solve(solver=cp.CLARABEL, verbose=False)
        except Exception:
            return []

    if prob.status not in ("optimal", "optimal_inaccurate"):
        return []

    G_val = G.value
    if G_val is None:
        return []

    # ── 从 Gram 矩阵提取 2D 坐标 ──
    # 特征分解，取前 2 个特征值/向量
    eigvals, eigvecs = np.linalg.eigh(G_val)
    # eigh 返回升序，取最后 2 个（最大的）
    idx = np.argsort(eigvals)[::-1]
    top_vals = np.maximum(eigvals[idx[:2]], 0)  # 确保非负
    top_vecs = eigvecs[:, idx[:2]]

    coords_2d = top_vecs * np.sqrt(top_vals)[np.newaxis, :]

    # ── 规范化: Procrustes 对齐到锚点形式 ──
    anchor_a, anchor_b, anchor_c = select_anchors(
        object_ids, qrr_entries, trr_entries
    )

    positions = {oid: coords_2d[i] for i, oid in enumerate(object_ids)}

    # 对齐到 anchor_a=(0,0), anchor_b=(1,0)
    positions = _align_to_canonical(positions, anchor_a, anchor_b, anchor_c,
                                     has_trr=bool(trr_entries))

    # 计算 loss（用原始 loss 函数评估，以便与其他 solver 可比）
    l_qrr = compute_qrr_loss(positions, qrr_entries, config)
    l_trr = compute_trr_loss(positions, trr_entries, config)
    l_sep = compute_sep_loss(positions, config)

    solution = SolverSolution(
        positions=positions,
        loss=l_qrr + l_trr + l_sep,
        loss_qrr=l_qrr,
        loss_trr=l_trr,
        loss_sep=l_sep,
        converged=prob.status == "optimal",
        n_iter=0,
    )

    return [solution]


def _align_to_canonical(
    positions: Dict[str, np.ndarray],
    anchor_a: str, anchor_b: str, anchor_c: str,
    has_trr: bool,
) -> Dict[str, np.ndarray]:
    """将 SDP 解对齐到规范形式: anchor_a=(0,0), anchor_b=(1,0)。"""
    pa = positions[anchor_a].copy()
    pb = positions[anchor_b].copy()

    # 平移: anchor_a -> 原点
    for oid in positions:
        positions[oid] = positions[oid] - pa

    pb = positions[anchor_b]
    # 旋转 + 缩放: anchor_b -> (1, 0)
    dist_ab = np.linalg.norm(pb)
    if dist_ab < 1e-12:
        dist_ab = 1.0
    angle = np.arctan2(pb[1], pb[0])
    cos_a, sin_a = np.cos(-angle), np.sin(-angle)
    rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    scale = 1.0 / dist_ab

    for oid in positions:
        positions[oid] = rot @ positions[oid] * scale

    # 镜像反射: 确保 y_c >= 0（仅无 TRR 时）
    if not has_trr and anchor_c in positions:
        if positions[anchor_c][1] < 0:
            for oid in positions:
                positions[oid] = np.array([positions[oid][0], -positions[oid][1]])

    return positions
