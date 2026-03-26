"""
Hinge loss + 差分进化全局求解器。

理论依据: Jain et al. 2016 的 hinge loss 框架。
在坐标空间使用 hinge loss 代替 softplus，
用 scipy differential_evolution 做全局搜索，避免局部最优。
"""

import math
import numpy as np
from scipy.optimize import differential_evolution, minimize
from typing import Dict, List, Optional

from .constraints import QRREntry, TRREntry
from .solver import (
    SolverConfig, SolverSolution,
    select_anchors, pack_free_variables, unpack_free_variables,
    compute_qrr_loss, compute_trr_loss, compute_sep_loss,
)
from .utils import pair_key, hour_to_angle_deg, rotate_vec2


def compute_hinge_qrr_loss(
    positions: Dict[str, np.ndarray],
    qrr_entries: List[QRREntry],
    margin: float = 0.05,
    delta_eq: float = 0.1,
) -> float:
    """Hinge loss 版 QRR 损失。

    对比 softplus 的平滑惩罚，hinge loss 在满足区域为零，
    违背区域线性增长，梯度更尖锐。

    使用距离平方域：
      < : max(0, d1² - d2² + margin)
      > : max(0, d2² - d1² + margin)
      ~= : max(0, |d1² - d2²| - delta_eq)
    """
    total = 0.0
    for entry in qrr_entries:
        p1 = pair_key(*entry.pair1)
        p2 = pair_key(*entry.pair2)

        d1_sq = float(np.sum((positions[p1[0]] - positions[p1[1]]) ** 2))
        d2_sq = float(np.sum((positions[p2[0]] - positions[p2[1]]) ** 2))

        if entry.comparator == "<":
            loss = max(0.0, d1_sq - d2_sq + margin)
        elif entry.comparator == ">":
            loss = max(0.0, d2_sq - d1_sq + margin)
        else:  # ~=
            loss = max(0.0, abs(d1_sq - d2_sq) - delta_eq)

        total += entry.weight * loss

    return total


def compute_hinge_trr_loss(
    positions: Dict[str, np.ndarray],
    trr_entries: List[TRREntry],
    config: SolverConfig,
) -> float:
    """角度 hinge loss（与原 TRR loss 结构相同但用 hinge 代替 softplus）。"""
    total = 0.0
    for entry in trr_entries:
        x_ref1 = positions[entry.ref1]
        x_ref2 = positions[entry.ref2]
        x_target = positions[entry.target]

        ref_vec = x_ref2 - x_ref1
        ref_norm = np.linalg.norm(ref_vec)
        if ref_norm < 1e-10:
            total += entry.weight * 2.0
            continue
        u = ref_vec / ref_norm

        tgt_vec = x_target - x_ref1
        tgt_norm = np.linalg.norm(tgt_vec)
        if tgt_norm < 1e-10:
            total += entry.weight * 2.0
            continue
        v = tgt_vec / tgt_norm

        alpha_rad = math.radians(hour_to_angle_deg(entry.hour))
        u_alpha = rotate_vec2(u, -alpha_rad)

        if entry.level == "hour":
            tol_rad = math.radians(config.trr_hour_tol_deg)
        else:
            tol_rad = math.radians(config.trr_quadrant_tol_deg)

        cos_diff = float(np.dot(u_alpha, v))
        cos_tol = math.cos(tol_rad)

        # hinge: 在扇区外时线性惩罚
        total += entry.weight * max(0.0, cos_tol - cos_diff)

    return total


def _total_hinge_loss(
    x: np.ndarray,
    anchor_a: str, anchor_b: str, anchor_c: str,
    object_ids: List[str],
    qrr_entries: List[QRREntry],
    trr_entries: List[TRREntry],
    config: SolverConfig,
) -> float:
    """差分进化用的总 hinge loss 函数。"""
    positions = unpack_free_variables(x, anchor_a, anchor_b, anchor_c, object_ids)

    l_qrr = compute_hinge_qrr_loss(
        positions, qrr_entries,
        margin=config.qrr_margin,
        delta_eq=config.qrr_delta_eq,
    )
    l_trr = compute_hinge_trr_loss(positions, trr_entries, config)
    l_sep = compute_sep_loss(positions, config)

    return l_qrr + l_trr + l_sep


def solve_hinge(
    object_ids: List[str],
    qrr_entries: List[QRREntry],
    trr_entries: List[TRREntry],
    config: Optional[SolverConfig] = None,
    gt_positions: Optional[Dict[str, np.ndarray]] = None,
) -> List[SolverSolution]:
    """Hinge loss + 差分进化全局求解器。

    两阶段:
      1. differential_evolution 全局搜索（找到好的盆地）
      2. L-BFGS-B 精修（在盆地内收敛到精确最小值）
    """
    if config is None:
        config = SolverConfig()

    n = len(object_ids)
    if n < 3:
        return []

    anchor_a, anchor_b, anchor_c = select_anchors(
        object_ids, qrr_entries, trr_entries
    )
    n_free = 2 * (n - 2)
    free_objs = [oid for oid in object_ids if oid not in (anchor_a, anchor_b)]
    c_idx = free_objs.index(anchor_c) if anchor_c in free_objs else -1

    args = (anchor_a, anchor_b, anchor_c, object_ids,
            qrr_entries, trr_entries, config)

    # 搜索范围
    bounds = [(-3.0, 3.0)] * n_free
    if c_idx >= 0 and not trr_entries:
        y_c_pos = c_idx * 2 + 1
        bounds[y_c_pos] = (0.0, 3.0)

    # 阶段 1: 差分进化全局搜索
    try:
        de_result = differential_evolution(
            _total_hinge_loss,
            bounds=bounds,
            args=args,
            maxiter=300,
            tol=1e-8,
            seed=42,
            polish=False,  # 不用内置 polish，我们自己做 L-BFGS-B
            popsize=15,
            mutation=(0.5, 1.5),
            recombination=0.8,
        )
        x_de = de_result.x
    except Exception:
        # 降级为随机初始化
        x_de = np.random.RandomState(42).randn(n_free) * 1.5

    # 阶段 2: L-BFGS-B 精修（用原始 softplus loss）
    from .solver import compute_total_loss

    lbfgs_bounds = [(None, None)] * n_free
    if c_idx >= 0 and not trr_entries:
        y_c_pos = c_idx * 2 + 1
        lbfgs_bounds[y_c_pos] = (0, None)

    solutions = []

    # 从 DE 结果精修
    for x0 in [x_de]:
        try:
            result = minimize(
                compute_total_loss,
                x0,
                args=(anchor_a, anchor_b, anchor_c, object_ids,
                      qrr_entries, trr_entries, config),
                method="L-BFGS-B",
                bounds=lbfgs_bounds,
                options={"maxiter": config.maxiter, "ftol": config.ftol, "gtol": config.gtol},
            )
            positions = unpack_free_variables(
                result.x, anchor_a, anchor_b, anchor_c, object_ids
            )
            l_qrr = compute_qrr_loss(positions, qrr_entries, config)
            l_trr = compute_trr_loss(positions, trr_entries, config)
            l_sep = compute_sep_loss(positions, config)

            solutions.append(SolverSolution(
                positions=positions,
                loss=result.fun,
                loss_qrr=l_qrr,
                loss_trr=l_trr,
                loss_sep=l_sep,
                converged=result.success,
                n_iter=result.nit,
            ))
        except Exception:
            continue

    # 补充随机重启（与原 solver 类似，但数量减半，因为 DE 已经找到好的起点）
    n_extra = max(1, config.n_restarts // 2)
    for restart in range(n_extra):
        rng = np.random.RandomState(restart * 42 + 7)
        x0 = rng.randn(n_free) * 1.5
        if c_idx >= 0 and not trr_entries:
            x0[c_idx * 2 + 1] = abs(x0[c_idx * 2 + 1])

        try:
            result = minimize(
                compute_total_loss,
                x0,
                args=(anchor_a, anchor_b, anchor_c, object_ids,
                      qrr_entries, trr_entries, config),
                method="L-BFGS-B",
                bounds=lbfgs_bounds,
                options={"maxiter": config.maxiter, "ftol": config.ftol, "gtol": config.gtol},
            )
            positions = unpack_free_variables(
                result.x, anchor_a, anchor_b, anchor_c, object_ids
            )
            l_qrr = compute_qrr_loss(positions, qrr_entries, config)
            l_trr = compute_trr_loss(positions, trr_entries, config)
            l_sep = compute_sep_loss(positions, config)

            solutions.append(SolverSolution(
                positions=positions,
                loss=result.fun,
                loss_qrr=l_qrr,
                loss_trr=l_trr,
                loss_sep=l_sep,
                converged=result.success,
                n_iter=result.nit,
            ))
        except Exception:
            continue

    solutions.sort(key=lambda s: s.loss)
    return solutions
