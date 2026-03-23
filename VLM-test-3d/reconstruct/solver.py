"""
3D 场景信念求解器：规范化固定 + L-BFGS-B 多重启优化。

重建管线的第 2 阶段：
  - 三锚点规范化固定（消除平移/旋转/缩放/反射共 7+1 DOF）
  - 对数域 QRR 损失 + 弧段容差 TRR 损失（方位角+仰角）+ 分离正则 + z 正则
  - 多重启优化与解集合
"""

import math
import numpy as np
from scipy.optimize import minimize
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field

from .constraints import QRREntry, TRREntry, TRR3DEntry
from .utils import pair_key, rotate_vec2


# ── 超参数 ──

@dataclass
class SolverConfig:
    """3D 求解器超参数配置。"""

    # QRR 损失
    qrr_margin: float = 0.1       # 排序 hinge 裕量
    qrr_delta_eq: float = 0.1     # 近似相等 Huber 尺度
    qrr_eps: float = 1e-6         # 距离计算中的 epsilon
    qrr_beta: float = 10.0        # softplus 锐度

    # TRR 损失（方位角）
    trr_tau: float = 0.1          # 方位角 softplus 缩放
    trr_hour_tol_deg: float = 15.0    # 小时级方位角容差（度）
    trr_quadrant_tol_deg: float = 45.0  # 象限级方位角容差（度）
    trr_beta: float = 10.0        # 方位角 softplus 锐度

    # TRR 损失（仰角）
    trr_elev_tol_deg: float = 20.0    # 仰角容差（度）
    trr_elev_weight: float = 0.3      # 仰角损失权重

    # 分离正则
    sep_eps: float = 0.2          # 最小间距阈值
    sep_lambda: float = 5.0       # 分离正则系数

    # z 正则
    z_reg_lambda: float = 1e-4    # z 正则项系数

    # 求解器参数
    n_restarts: int = 10          # 多重启次数
    maxiter: int = 500            # L-BFGS-B 最大迭代
    ftol: float = 1e-10           # 函数值收敛容差
    gtol: float = 1e-7            # 梯度收敛容差


# ── 解容器 ──

@dataclass
class SolverSolution:
    """单次优化结果。"""
    positions: Dict[str, np.ndarray]  # 对象 ID -> 3D 坐标
    loss: float             # 总损失
    loss_qrr: float         # QRR 排序损失
    loss_trr: float         # TRR 方位角损失
    loss_elev: float        # TRR 仰角损失
    loss_sep: float         # 分离正则损失
    loss_z_reg: float       # z 正则损失
    converged: bool         # 是否收敛
    n_iter: int             # 迭代次数


# ── 规范化固定（7 DOF + z 反射） ──

def select_anchors(
    object_ids: List[str],
    qrr_entries: List[QRREntry],
    trr_entries: List[Union[TRREntry, TRR3DEntry]],
) -> Tuple[str, str, str]:
    """按约束参与频率选择 3 个锚点对象。

    返回 (anchor_a, anchor_b, anchor_c)：
      anchor_a -> (0, 0, 0)    固定平移（3 DOF）
      anchor_b -> (1, 0, 0)    固定缩放+2个旋转自由度（2 DOF）
      anchor_c -> (cx, cy, 0)  z_c=0 固定剩余旋转+反射（2 DOF）
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

    # 按频率降序排列，频率相同时按 ID 排序（确保稳定性）
    sorted_objs = sorted(object_ids, key=lambda o: (-freq.get(o, 0), o))

    if len(sorted_objs) < 3:
        # 不足 3 个对象时用最后一个填充
        while len(sorted_objs) < 3:
            sorted_objs.append(sorted_objs[-1])

    return sorted_objs[0], sorted_objs[1], sorted_objs[2]


def pack_free_variables_3d(
    positions: Dict[str, np.ndarray],
    anchor_a: str,
    anchor_b: str,
    anchor_c: str,
    object_ids: List[str],
) -> np.ndarray:
    """将 3D 位置打包为自由变量向量（排除规范化固定的自由度）。

    规范化固定（7 DOF + 反射消除）：
      anchor_a = (0, 0, 0)  -> 3 DOF（平移）
      anchor_b = (1, 0, 0)  -> 2 DOF（缩放 + 绕 z 旋转）
      anchor_c = (cx, cy, 0)  -> z_c=0 固定，2 DOF（绕 AB 旋转 + 反射）
      其余对象：每个 3 自由度 (x, y, z)

    自由变量排列：[cx, cy, x3, y3, z3, x4, y4, z4, ...]
    总计：2 + 3*(N-3) 个自由变量
    """
    free = []

    # anchor_c 的 x, y（z_c 固定为 0）
    free.append(positions[anchor_c][0])
    free.append(positions[anchor_c][1])

    # 其余非锚点对象的 x, y, z
    for oid in object_ids:
        if oid in (anchor_a, anchor_b, anchor_c):
            continue
        free.append(positions[oid][0])
        free.append(positions[oid][1])
        free.append(positions[oid][2])

    return np.array(free, dtype=np.float64)


def unpack_free_variables_3d(
    x: np.ndarray,
    anchor_a: str,
    anchor_b: str,
    anchor_c: str,
    object_ids: List[str],
) -> Dict[str, np.ndarray]:
    """将自由变量向量解包为 3D 位置字典。

    anchor_a = (0, 0, 0), anchor_b = (1, 0, 0) 固定。
    anchor_c = (cx, cy, 0)，z_c 固定为 0。
    y_c >= 0 可通过 L-BFGS-B 边界约束实现（无 TRR 时）。
    """
    positions = {}
    positions[anchor_a] = np.array([0.0, 0.0, 0.0])
    positions[anchor_b] = np.array([1.0, 0.0, 0.0])

    # 解包 anchor_c
    cx, cy = x[0], x[1]
    positions[anchor_c] = np.array([cx, cy, 0.0])

    # 解包其余对象
    idx = 2
    for oid in object_ids:
        if oid in (anchor_a, anchor_b, anchor_c):
            continue
        xi, yi, zi = x[idx], x[idx + 1], x[idx + 2]
        positions[oid] = np.array([xi, yi, zi])
        idx += 3

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

    对于每条约束：
      delta = log(d1) - log(d2)
      <:   softplus(delta + margin)
      >:   softplus(-delta + margin)
      ~=:  huber(delta)

    np.linalg.norm 对 2D 和 3D 向量通用，因此逻辑与 2D 版本完全一致。
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
            r = delta / delta_eq
            if abs(r) <= 1:
                loss = 0.5 * r * r * delta_eq
            else:
                loss = (abs(r) - 0.5) * delta_eq

        total += entry.weight * loss

    return total


def compute_trr_loss_3d(
    positions: Dict[str, np.ndarray],
    trr_entries: List[Union[TRREntry, TRR3DEntry]],
    config: SolverConfig,
) -> Tuple[float, float]:
    """3D 弧段容差 TRR 损失（方位角 + 仰角）。

    方位角损失：
      将 ref_vec 和 tgt_vec 投影到 xy 平面，
      然后使用与 2D 相同的弧段容差逻辑（rotate_vec2, cos_diff）。

    仰角损失（仅对 TRR3DEntry 有效）：
      recon_elev = atan2(tgt_vec[2], norm(tgt_vec[:2]))
      loss = softplus((|recon_elev - gt_elev| - tol) / tau)

    返回：
      (方位角损失, 仰角损失)
    """
    total_azimuth = 0.0
    total_elev = 0.0
    beta = config.trr_beta
    tau = config.trr_tau
    elev_tol_rad = math.radians(config.trr_elev_tol_deg)

    for entry in trr_entries:
        x_ref1 = positions[entry.ref1]
        x_ref2 = positions[entry.ref2]
        x_target = positions[entry.target]

        # ── 方位角损失：投影到 xy 平面 ──

        # 参考方向：ref1 -> ref2 的 xy 投影（12 点方向）
        ref_vec_3d = x_ref2 - x_ref1
        ref_vec_xy = ref_vec_3d[:2]  # 取 x, y 分量
        ref_norm = np.linalg.norm(ref_vec_xy)
        if ref_norm < 1e-10:
            # 退化情况：ref1 和 ref2 在 xy 平面重合，施加惩罚
            total_azimuth += entry.weight * 2.0
            continue
        u = ref_vec_xy / ref_norm

        # 目标方向：ref1 -> target 的 xy 投影
        tgt_vec_3d = x_target - x_ref1
        tgt_vec_xy = tgt_vec_3d[:2]
        tgt_norm = np.linalg.norm(tgt_vec_xy)
        if tgt_norm < 1e-10:
            # 退化情况：target 和 ref1 在 xy 平面重合，施加惩罚
            total_azimuth += entry.weight * 2.0
            continue
        v = tgt_vec_xy / tgt_norm

        # 根据小时数计算期望旋转角度
        alpha_rad = math.radians((entry.hour % 12) * 30.0)
        u_alpha = rotate_vec2(u, alpha_rad)

        # 容差半宽度
        if entry.level == "hour":
            tol_rad = math.radians(config.trr_hour_tol_deg)
        else:
            tol_rad = math.radians(config.trr_quadrant_tol_deg)

        cos_diff = float(np.dot(u_alpha, v))
        cos_tol = math.cos(tol_rad)

        # 方位角损失：当 cos_diff < cos_tol 时施加惩罚（即超出弧段范围）
        az_loss = _softplus((cos_tol - cos_diff) / tau, beta)
        total_azimuth += entry.weight * az_loss

        # ── 仰角损失：仅对 TRR3DEntry 有效 ──
        if isinstance(entry, TRR3DEntry):
            # 重建仰角：tgt_vec_3d 的仰角
            tgt_xy_norm = np.linalg.norm(tgt_vec_3d[:2])
            recon_elev = math.atan2(tgt_vec_3d[2], tgt_xy_norm + 1e-12)

            # 真值仰角（从 TRR3DEntry 获取）
            gt_elev = math.radians(entry.elevation_deg)

            # softplus 仰角偏差损失
            elev_diff = abs(recon_elev - gt_elev)
            elev_loss = _softplus((elev_diff - elev_tol_rad) / tau, beta)
            total_elev += entry.weight * elev_loss

    return total_azimuth, total_elev


def compute_sep_loss(
    positions: Dict[str, np.ndarray],
    config: SolverConfig,
) -> float:
    """分离正则：防止对象坍缩到同一位置。

    np.linalg.norm 对 2D 和 3D 向量通用，逻辑与 2D 版本完全一致。
    """
    total = 0.0
    objs = list(positions.keys())
    eps = config.sep_eps

    for i in range(len(objs)):
        for j in range(i + 1, len(objs)):
            dist = np.linalg.norm(positions[objs[i]] - positions[objs[j]])
            if dist < eps:
                total += (eps - dist)

    return config.sep_lambda * total


def compute_z_reg_loss(
    positions: Dict[str, np.ndarray],
    config: SolverConfig,
    anchor_a: str,
    anchor_b: str,
    anchor_c: str,
) -> float:
    """z 正则损失：对非锚点对象的 z 坐标施加 L2 惩罚。

    鼓励解尽量扁平（接近 xy 平面），在约束不充分时防止
    z 值漂移到不合理的范围。

    loss = lambda * sum(z_i^2)，其中 i 遍历所有非锚点对象。
    """
    total = 0.0
    anchors = {anchor_a, anchor_b, anchor_c}

    for oid, pos in positions.items():
        if oid not in anchors:
            total += pos[2] ** 2

    return config.z_reg_lambda * total


def compute_total_loss_3d(
    x: np.ndarray,
    anchor_a: str,
    anchor_b: str,
    anchor_c: str,
    object_ids: List[str],
    qrr_entries: List[QRREntry],
    trr_entries: List[Union[TRREntry, TRR3DEntry]],
    config: SolverConfig,
) -> float:
    """3D L-BFGS-B 总损失函数。"""
    positions = unpack_free_variables_3d(
        x, anchor_a, anchor_b, anchor_c, object_ids
    )
    l_qrr = compute_qrr_loss(positions, qrr_entries, config)
    l_az, l_elev = compute_trr_loss_3d(positions, trr_entries, config)
    l_sep = compute_sep_loss(positions, config)
    l_z_reg = compute_z_reg_loss(positions, config, anchor_a, anchor_b, anchor_c)
    return l_qrr + l_az + config.trr_elev_weight * l_elev + l_sep + l_z_reg


# ── 多重启求解器 ──

def solve_3d(
    object_ids: List[str],
    qrr_entries: List[QRREntry],
    trr_entries: List[Union[TRREntry, TRR3DEntry]],
    config: Optional[SolverConfig] = None,
    gt_positions: Optional[Dict[str, np.ndarray]] = None,
) -> List[SolverSolution]:
    """运行 3D 多重启 L-BFGS-B 优化。

    参数：
        object_ids: 对象 ID 列表
        qrr_entries: QRR 距离排序约束
        trr_entries: TRR 角度约束（支持 TRREntry 和 TRR3DEntry 混合）
        config: 求解器超参数
        gt_positions: 真值位置（可用于初始化种子，暂未使用）

    返回：
        SolverSolution 列表（每次重启一个，按损失升序排列）

    规范化固定策略（7 DOF + z 反射）：
      anchor_a = (0, 0, 0)         消除 3 平移自由度
      anchor_b = (1, 0, 0)         消除 1 缩放 + 1 绕 z 轴旋转
      anchor_c = (cx, cy, 0)       z_c=0 消除 1 绕 AB 轴旋转 + 1 反射
        - 无 TRR 时额外约束 y_c >= 0 消除 xy 平面反射
      anchor_d（第 4 个对象）       z_d >= 0 消除 z 反射
    """
    if config is None:
        config = SolverConfig()

    n = len(object_ids)
    if n < 3:
        # 对象不足，无法进行有意义的重建
        return []

    # 选择锚点
    anchor_a, anchor_b, anchor_c = select_anchors(
        object_ids, qrr_entries, trr_entries
    )

    # 自由变量数：anchor_c 贡献 2 个 (cx, cy)，其余非锚点各贡献 3 个 (x, y, z)
    n_non_anchor = n - 3  # 排除 a, b, c
    n_free = 2 + 3 * n_non_anchor

    # 确定非锚点对象列表（按 object_ids 顺序，排除三个锚点）
    free_objs = [oid for oid in object_ids
                 if oid not in (anchor_a, anchor_b, anchor_c)]

    # 找到 anchor_d（第 4 个对象）在自由变量中的位置索引
    # anchor_d 是 free_objs 中第一个对象（即约束参与频率第 4 高的对象）
    anchor_d = free_objs[0] if free_objs else None
    # anchor_d 的 z 坐标在自由变量中的位置：
    #   自由变量布局 = [cx, cy, x_d, y_d, z_d, ...]
    #   z_d 位于索引 4
    d_z_idx = 4 if anchor_d is not None else -1

    solutions = []

    for restart in range(config.n_restarts):
        # 随机初始化：围绕规范化锚点的 3D 分布
        rng = np.random.RandomState(restart * 42 + 7)
        x0 = rng.randn(n_free) * 1.5

        # ── 构建边界约束 ──
        bounds = [(None, None)] * n_free

        # 无 TRR 约束时：y_c >= 0 消除 xy 平面反射歧义
        # （TRR 角度约束本身可打破反射歧义，强制 y_c >= 0 可能与 TRR 冲突）
        if not trr_entries:
            # y_c 位于索引 1
            bounds[1] = (0, None)
            x0[1] = abs(x0[1])

        # N >= 4 时：z_d >= 0 消除 z 反射歧义
        if n >= 4 and d_z_idx >= 0 and d_z_idx < n_free:
            bounds[d_z_idx] = (0, None)
            x0[d_z_idx] = abs(x0[d_z_idx])

        try:
            result = minimize(
                compute_total_loss_3d,
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

            positions = unpack_free_variables_3d(
                result.x, anchor_a, anchor_b, anchor_c, object_ids
            )
            l_qrr = compute_qrr_loss(positions, qrr_entries, config)
            l_az, l_elev = compute_trr_loss_3d(positions, trr_entries, config)
            l_sep = compute_sep_loss(positions, config)
            l_z_reg = compute_z_reg_loss(
                positions, config, anchor_a, anchor_b, anchor_c
            )

            solutions.append(SolverSolution(
                positions=positions,
                loss=result.fun,
                loss_qrr=l_qrr,
                loss_trr=l_az,
                loss_elev=l_elev,
                loss_sep=l_sep,
                loss_z_reg=l_z_reg,
                converged=result.success,
                n_iter=result.nit,
            ))
        except Exception as e:
            import warnings
            warnings.warn(f"重启 {restart} 失败：{e}")
            continue

    # 按总损失升序排列
    solutions.sort(key=lambda s: s.loss)
    return solutions
