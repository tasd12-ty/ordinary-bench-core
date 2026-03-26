"""
2D 场景信念求解器：规范固定的 L-BFGS-B 优化。

重建管线的第 2 阶段：
  - 3 锚点规范固定（消除平移/旋转/缩放/镜像反射自由度）
  - 对数域 QRR 损失 + 扇区容差 TRR 损失 + 分离正则化
  - 多重启优化与解收集
"""

import math
import numpy as np
from scipy.optimize import minimize
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

from .constraints import QRREntry, TRREntry
from .utils import hour_to_angle_deg, pair_key, rotate_vec2


# ── 超参数 ──

@dataclass
class SolverConfig:
    # QRR 损失
    qrr_margin: float = 0.1
    qrr_delta_eq: float = 0.1
    qrr_eps: float = 1e-6
    qrr_beta: float = 10.0

    # TRR 损失
    trr_tau: float = 0.1
    trr_hour_tol_deg: float = 15.0
    trr_quadrant_tol_deg: float = 45.0
    trr_beta: float = 10.0

    # 分离正则化
    sep_eps: float = 0.2
    sep_lambda: float = 5.0

    # 求解器
    n_restarts: int = 10
    maxiter: int = 500
    ftol: float = 1e-10
    gtol: float = 1e-7


# ── 解容器 ──

@dataclass
class SolverSolution:
    """单次优化结果。"""
    positions: Dict[str, np.ndarray]
    loss: float
    loss_qrr: float
    loss_trr: float
    loss_sep: float
    converged: bool
    n_iter: int


# ── 规范固定 ──

def select_anchors(
    object_ids: List[str],
    qrr_entries: List[QRREntry],
    trr_entries: List[TRREntry],
) -> Tuple[str, str, str]:
    """按约束参与频率选择 3 个锚点对象。

    返回 (anchor_a, anchor_b, anchor_c)，其中：
      anchor_a -> (0, 0)
      anchor_b -> (1, 0)
      anchor_c -> y >= 0（仅在无 TRR 约束时强制执行）
    """
    freq: Dict[str, int] = {}
    for oid in object_ids:
        freq[oid] = 0

    for entry in qrr_entries:
        for obj in sorted(set(entry.pair1) | set(entry.pair2)):
            freq[obj] = freq.get(obj, 0) + 1
    for entry in trr_entries:
        for obj in [entry.target, entry.ref1, entry.ref2]:
            freq[obj] = freq.get(obj, 0) + 1

    # 按频率降序排列，然后按 id 排列（保证稳定性）
    sorted_objs = sorted(object_ids, key=lambda o: (-freq.get(o, 0), o))

    if len(sorted_objs) < 3:
        # 用剩余对象填充
        while len(sorted_objs) < 3:
            sorted_objs.append(sorted_objs[-1])

    return sorted_objs[0], sorted_objs[1], sorted_objs[2]


def pack_free_variables(
    positions: Dict[str, np.ndarray],
    anchor_a: str,
    anchor_b: str,
    anchor_c: str,
    object_ids: List[str],
) -> np.ndarray:
    """将位置打包为自由变量向量（排除规范固定的自由度）。

    规范固定（消除 5 个自由度）：
      anchor_a = (0, 0)     -> 2 自由度（平移）
      anchor_b = (1, 0)     -> 2 自由度（旋转 + 缩放）
      y_c >= 0              -> 1 自由度（镜像反射）

    自由变量：[x_c, y_c, x_3, y_3, x_4, y_4, ...]
    """
    free = []
    free_objs = [oid for oid in object_ids
                 if oid not in (anchor_a, anchor_b)]
    for oid in free_objs:
        free.append(positions[oid][0])
        free.append(positions[oid][1])

    return np.array(free, dtype=np.float64)


def unpack_free_variables(
    x: np.ndarray,
    anchor_a: str,
    anchor_b: str,
    anchor_c: str,
    object_ids: List[str],
) -> Dict[str, np.ndarray]:
    """将自由变量解包为位置字典。

    anchor_a = (0, 0)、anchor_b = (1, 0) 是固定的。
    y_c >= 0 可选地通过 solve() 中的 L-BFGS-B 边界约束强制执行。
    """
    positions = {}
    positions[anchor_a] = np.array([0.0, 0.0])
    positions[anchor_b] = np.array([1.0, 0.0])

    free_objs = [oid for oid in object_ids
                 if oid not in (anchor_a, anchor_b)]

    idx = 0
    for oid in free_objs:
        xi, yi = x[idx], x[idx + 1]
        positions[oid] = np.array([xi, yi])
        idx += 2

    return positions


# ── 损失函数 ──

def _softplus(x: float, beta: float = 1.0) -> float:
    """数值稳定的 softplus：log(1 + exp(beta*x)) / beta。"""
    bx = beta * x
    if bx > 20:
        return x
    if bx < -20:
        return 0.0
    return math.log1p(math.exp(bx)) / beta


def compute_qrr_loss(
    positions: Dict[str, np.ndarray],
    qrr_entries: List[QRREntry],
    config: SolverConfig,
) -> float:
    """对数域 QRR 排序损失。

    对于每个约束：
      delta = log(d1) - log(d2)
      <:   softplus(delta + margin)
      >:   softplus(-delta + margin)
      ~=:  huber(delta)
    """
    total = 0.0
    eps = config.qrr_eps
    margin = config.qrr_margin
    beta = config.qrr_beta
    delta_eq = config.qrr_delta_eq

    for entry in qrr_entries:
        p1 = pair_key(*entry.pair1)
        p2 = pair_key(*entry.pair2)

        d1 = np.linalg.norm(positions[p1[0]] - positions[p1[1]]) + eps
        d2 = np.linalg.norm(positions[p2[0]] - positions[p2[1]]) + eps
        delta = math.log(d1) - math.log(d2)

        if entry.comparator == "<":
            loss = _softplus(delta + margin, beta)
        elif entry.comparator == ">":
            loss = _softplus(-delta + margin, beta)
        else:  # ~=
            # Huber 损失
            x = delta / delta_eq
            if abs(x) <= 1:
                loss = 0.5 * x * x * delta_eq
            else:
                loss = (abs(x) - 0.5) * delta_eq

        total += entry.weight * loss

    return total


def compute_trr_loss(
    positions: Dict[str, np.ndarray],
    trr_entries: List[TRREntry],
    config: SolverConfig,
) -> float:
    """扇区容差 TRR 损失。

    对于每个约束 (target, ref1, ref2, hour)：
      u = normalize(x_ref2 - x_ref1)  [12 点方向]
      v = normalize(x_target - x_ref1) [目标方向]
      alpha = hour_to_angle_rad(hour)
      u_alpha = rotate(u, -alpha)
      cos_diff = dot(u_alpha, v)
      loss = softplus((cos(tol) - cos_diff) / tau)
    """
    total = 0.0
    beta = config.trr_beta
    tau = config.trr_tau

    for entry in trr_entries:
        x_ref1 = positions[entry.ref1]
        x_ref2 = positions[entry.ref2]
        x_target = positions[entry.target]

        # 参考方向：ref1 -> ref2（12 点方向）
        ref_vec = x_ref2 - x_ref1
        ref_norm = np.linalg.norm(ref_vec)
        if ref_norm < 1e-10:
            # 退化情况：ref1 ≈ ref2，添加惩罚以推开
            total += entry.weight * 2.0
            continue
        u = ref_vec / ref_norm

        # 目标方向：ref1 -> target
        tgt_vec = x_target - x_ref1
        tgt_norm = np.linalg.norm(tgt_vec)
        if tgt_norm < 1e-10:
            # 退化情况：target ≈ ref1，添加惩罚以推开
            total += entry.weight * 2.0
            continue
        v = tgt_vec / tgt_norm

        # 根据钟面小时计算期望方向：从 12 点方向顺时针旋转。
        alpha_rad = math.radians(hour_to_angle_deg(entry.hour))
        u_alpha = rotate_vec2(u, -alpha_rad)

        # 容差半宽
        if entry.level == "hour":
            tol_rad = math.radians(config.trr_hour_tol_deg)
        else:
            tol_rad = math.radians(config.trr_quadrant_tol_deg)

        cos_diff = float(np.dot(u_alpha, v))
        cos_tol = math.cos(tol_rad)

        # 损失：当 cos_diff < cos_tol 时施加惩罚（即在扇区外）
        loss = _softplus((cos_tol - cos_diff) / tau, beta)

        total += entry.weight * loss

    return total


def compute_sep_loss(
    positions: Dict[str, np.ndarray],
    config: SolverConfig,
) -> float:
    """分离正则化：防止对象坍缩。"""
    total = 0.0
    objs = list(positions.keys())
    eps = config.sep_eps

    for i in range(len(objs)):
        for j in range(i + 1, len(objs)):
            dist = np.linalg.norm(positions[objs[i]] - positions[objs[j]])
            if dist < eps:
                total += (eps - dist)

    return config.sep_lambda * total


def compute_total_loss(
    x: np.ndarray,
    anchor_a: str,
    anchor_b: str,
    anchor_c: str,
    object_ids: List[str],
    qrr_entries: List[QRREntry],
    trr_entries: List[TRREntry],
    config: SolverConfig,
) -> float:
    """L-BFGS-B 的总损失函数。"""
    positions = unpack_free_variables(x, anchor_a, anchor_b, anchor_c, object_ids)
    l_qrr = compute_qrr_loss(positions, qrr_entries, config)
    l_trr = compute_trr_loss(positions, trr_entries, config)
    l_sep = compute_sep_loss(positions, config)
    return l_qrr + l_trr + l_sep


# ── 多重启求解器 ──

def solve(
    object_ids: List[str],
    qrr_entries: List[QRREntry],
    trr_entries: List[TRREntry],
    config: Optional[SolverConfig] = None,
    gt_positions: Optional[Dict[str, np.ndarray]] = None,
) -> List[SolverSolution]:
    """运行多重启 L-BFGS-B 优化。

    参数:
        object_ids: 对象 ID 列表
        qrr_entries: QRR 约束
        trr_entries: TRR 约束
        config: 求解器超参数
        gt_positions: 真值位置（用于初始化种子）

    返回:
        SolverSolution 对象列表（每次重启一个，按损失排序）
    """
    if config is None:
        config = SolverConfig()

    n = len(object_ids)
    if n < 3:
        # 对象数不足，无法进行有意义的重建
        return []

    # 选择锚点
    anchor_a, anchor_b, anchor_c = select_anchors(
        object_ids, qrr_entries, trr_entries
    )

    n_free = 2 * (n - 2)  # 每个自由对象 2 个坐标（锚点 a、b 已固定）

    solutions = []
    free_objs = [oid for oid in object_ids
                 if oid not in (anchor_a, anchor_b)]
    c_idx = free_objs.index(anchor_c) if anchor_c in free_objs else -1

    for restart in range(config.n_restarts):
        # 随机初始化：在规范固定锚点周围分散
        rng = np.random.RandomState(restart * 42 + 7)
        x0 = rng.randn(n_free) * 1.5

        # 注意：当存在 TRR 约束时，有意省略 y_c >= 0 约束，
        # 因为 TRR 角度约束自然打破了镜像反射歧义。
        # 强制 y_c >= 0 可能通过反转所有角度而与 TRR 约束冲突。
        bounds = [(None, None)] * n_free
        if c_idx >= 0 and not trr_entries:
            # 仅在无 TRR 约束时强制 y_c >= 0
            y_c_pos = c_idx * 2 + 1
            bounds[y_c_pos] = (0, None)
            x0[y_c_pos] = abs(x0[y_c_pos])

        try:
            result = minimize(
                compute_total_loss,
                x0,
                args=(anchor_a, anchor_b, anchor_c, object_ids,
                      qrr_entries, trr_entries, config),
                method="L-BFGS-B",
                bounds=bounds,
                options={
                    "maxiter": config.maxiter,
                    "ftol": config.ftol,
                    "gtol": config.gtol,
                },
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
        except Exception as e:
            import warnings
            warnings.warn(f"Restart {restart} failed: {e}")
            continue

    # 按总损失排序
    solutions.sort(key=lambda s: s.loss)
    return solutions
