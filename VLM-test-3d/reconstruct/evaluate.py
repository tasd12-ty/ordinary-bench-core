"""
三维场景信念重建的评估指标。

CSR（约束满足率）、K_geom（几何模态数）、spread（离散度）、
Kendall tau（秩相关）、NRMS（归一化均方根误差）+ 多解聚类。

在 2D 版本基础上扩展：
  - 新增 compute_csr_trr_3d() 分别评估方位角和仰角
  - 新增 EvalMetrics3D 数据类（含方位角/仰角/综合 CSR）
  - evaluate_reconstruction_3d() 使用 N 维 Procrustes 对齐
"""

import math
import numpy as np
from itertools import combinations
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from scipy.stats import kendalltau

from .constraints import QRREntry, TRREntry, TRR3DEntry
from .solver import SolverSolution, SolverConfig
from .utils import procrustes_align_nd, compute_nrms, compute_rms, pair_key


# ── QRR 约束满足率 ──

def compute_csr_qrr(
    positions: Dict[str, np.ndarray],
    qrr_entries: List[QRREntry],
    tau: float = 0.10,
) -> float:
    """计算 QRR 约束的满足率。

    使用与原始比较器相同的比值容差机制。
    该函数与维度无关，适用于 2D 和 3D 位置。
    """
    if not qrr_entries:
        return 1.0

    satisfied = 0
    for entry in qrr_entries:
        p1 = pair_key(*entry.pair1)
        p2 = pair_key(*entry.pair2)

        d1 = float(np.linalg.norm(positions[p1[0]] - positions[p1[1]]))
        d2 = float(np.linalg.norm(positions[p2[0]] - positions[p2[1]]))

        # 根据重建位置确定比较符
        max_val = max(d1, d2)
        if max_val < 1e-12:
            recon_cmp = "~="
        else:
            threshold = tau * max_val
            diff = d1 - d2
            if abs(diff) <= threshold:
                recon_cmp = "~="
            elif diff < 0:
                recon_cmp = "<"
            else:
                recon_cmp = ">"

        # 严格匹配：比较符必须完全一致
        if recon_cmp == entry.comparator:
            satisfied += 1

    return satisfied / len(qrr_entries)


# ── 2D TRR 约束满足率（向后兼容） ──

def compute_csr_trr(
    positions: Dict[str, np.ndarray],
    trr_entries: List[TRREntry],
) -> float:
    """计算 TRR 约束的满足率（投影到 xy 平面）。

    向后兼容函数，重建角度在 xy 平面内计算。
    """
    if not trr_entries:
        return 1.0

    satisfied = 0
    evaluated = 0
    for entry in trr_entries:
        x_ref1 = positions[entry.ref1]
        x_ref2 = positions[entry.ref2]
        x_target = positions[entry.target]

        ref_vec = x_ref2 - x_ref1
        if np.linalg.norm(ref_vec) < 1e-10:
            continue
        tgt_vec = x_target - x_ref1
        if np.linalg.norm(tgt_vec) < 1e-10:
            continue

        evaluated += 1

        # 计算重建角度（仅使用 xy 分量）
        ref_angle = math.atan2(ref_vec[1], ref_vec[0])
        tgt_angle = math.atan2(tgt_vec[1], tgt_vec[0])
        rel_angle = math.degrees(tgt_angle - ref_angle) % 360

        # 真值角度由钟面小时确定
        expected_angle = (entry.hour % 12) * 30.0

        # 角距离
        diff = abs(rel_angle - expected_angle)
        diff = min(diff, 360 - diff)

        if entry.level == "hour":
            if diff <= 15.0:
                satisfied += 1
        else:  # quadrant
            if diff <= 45.0:
                satisfied += 1

    return satisfied / evaluated if evaluated > 0 else 1.0


# ── 3D TRR 约束满足率（方位角 + 仰角） ──

# 仰角带到角度范围的映射（度）
_ELEVATION_BAND_RANGES = {
    "above":          (25.0, 90.0),
    "slightly_above": (5.0, 35.0),
    "level":          (-15.0, 15.0),
    "slightly_below": (-35.0, -5.0),
    "below":          (-90.0, -25.0),
}

# 仰角带容差（度）：允许的偏差范围
_ELEVATION_BAND_TOLERANCE = 20.0


def compute_csr_trr_3d(
    positions: Dict[str, np.ndarray],
    trr3d_entries: List[TRR3DEntry],
) -> Tuple[float, float, float]:
    """计算三维 TRR 约束的满足率，分别返回方位角、仰角和综合指标。

    方位角 CSR 的逻辑与 2D compute_csr_trr() 一致（投影到 xy 平面）。
    仰角 CSR 检查重建的仰角是否落在目标仰角带范围内。
    综合 CSR 要求方位角和仰角同时满足。

    参数：
        positions: 重建的 3D 位置字典 {obj_id -> np.ndarray(3,)}
        trr3d_entries: 三维 TRR 约束列表

    返回：
        (csr_azimuth, csr_elevation, csr_full) 三元组
    """
    if not trr3d_entries:
        return 1.0, 1.0, 1.0

    n_azimuth_ok = 0
    n_elevation_ok = 0
    n_both_ok = 0
    evaluated = 0

    for entry in trr3d_entries:
        x_ref1 = positions[entry.ref1]
        x_ref2 = positions[entry.ref2]
        x_target = positions[entry.target]

        ref_vec = x_ref2 - x_ref1
        tgt_vec = x_target - x_ref1

        # 跳过退化情形（向量过短）
        if np.linalg.norm(ref_vec) < 1e-10:
            continue
        if np.linalg.norm(tgt_vec) < 1e-10:
            continue

        evaluated += 1

        # ── 方位角检查（投影到 xy 平面） ──
        ref_angle = math.atan2(ref_vec[1], ref_vec[0])
        tgt_angle = math.atan2(tgt_vec[1], tgt_vec[0])
        rel_angle = math.degrees(tgt_angle - ref_angle) % 360

        expected_angle = (entry.hour % 12) * 30.0
        az_diff = abs(rel_angle - expected_angle)
        az_diff = min(az_diff, 360 - az_diff)

        if entry.level == "hour":
            azimuth_ok = az_diff <= 15.0
        else:  # quadrant
            azimuth_ok = az_diff <= 45.0

        # ── 仰角检查 ──
        # 计算重建的仰角：arctan(dz / dxy)
        tgt_xy_dist = math.sqrt(tgt_vec[0] ** 2 + tgt_vec[1] ** 2)
        tgt_z = tgt_vec[2] if len(tgt_vec) >= 3 else 0.0
        recon_elevation = math.degrees(math.atan2(tgt_z, max(tgt_xy_dist, 1e-12)))

        # 根据仰角带判断是否满足
        band = entry.elevation_band
        if band in _ELEVATION_BAND_RANGES:
            lo, hi = _ELEVATION_BAND_RANGES[band]
            elevation_ok = lo - _ELEVATION_BAND_TOLERANCE <= recon_elevation <= hi + _ELEVATION_BAND_TOLERANCE
        else:
            # 未知仰角带，使用宽松判断
            elevation_ok = abs(recon_elevation - entry.elevation_deg) <= _ELEVATION_BAND_TOLERANCE

        # ── 汇总 ──
        if azimuth_ok:
            n_azimuth_ok += 1
        if elevation_ok:
            n_elevation_ok += 1
        if azimuth_ok and elevation_ok:
            n_both_ok += 1

    if evaluated == 0:
        return 1.0, 1.0, 1.0

    csr_azimuth = n_azimuth_ok / evaluated
    csr_elevation = n_elevation_ok / evaluated
    csr_full = n_both_ok / evaluated
    return csr_azimuth, csr_elevation, csr_full


# ── 多解聚类 ──

@dataclass
class ClusterResult:
    """多解聚类结果。"""
    K_geom: int = 1
    spread: float = 0.0
    cluster_sizes: List[int] = field(default_factory=list)
    representatives: List[Dict[str, np.ndarray]] = field(default_factory=list)
    all_assignments: List[int] = field(default_factory=list)


def cluster_solutions(
    solutions: List[SolverSolution],
    rms_threshold: float = 0.10,
    loss_ratio_cutoff: float = 3.0,
) -> ClusterResult:
    """按 RMS 距离聚类解（在统一标架空间中）。

    所有解共享相同标架（3 锚点固定），无需 Procrustes 对齐。
    贪心聚类：将每个解分配到最近聚类或创建新聚类。

    仅考虑 loss <= loss_ratio_cutoff * best_loss 的解，
    过滤掉未收敛的局部极小值。

    该函数与维度无关，适用于 2D 和 3D。
    """
    if not solutions:
        return ClusterResult()

    # 按损失质量筛选（解已按损失排序）
    best_loss = solutions[0].loss
    cutoff = max(best_loss * loss_ratio_cutoff, best_loss + 0.5)
    good_solutions = [s for s in solutions if s.loss <= cutoff]
    if not good_solutions:
        good_solutions = solutions[:1]

    if len(good_solutions) == 1:
        return ClusterResult(
            K_geom=1,
            spread=0.0,
            cluster_sizes=[1],
            representatives=[solutions[0].positions],
            all_assignments=[0],
        )

    # 提取位置矩阵（仅保留优质解）
    obj_ids = sorted(good_solutions[0].positions.keys())

    def to_matrix(pos: Dict[str, np.ndarray]) -> np.ndarray:
        return np.array([pos[oid] for oid in obj_ids])

    matrices = [to_matrix(s.positions) for s in good_solutions]

    # 贪心聚类
    clusters: List[List[int]] = []
    cluster_reps: List[np.ndarray] = []

    for i, mat in enumerate(matrices):
        assigned = False
        for c_idx, rep in enumerate(cluster_reps):
            rms = compute_rms(mat, rep)
            if rms < rms_threshold:
                clusters[c_idx].append(i)
                assigned = True
                break
        if not assigned:
            clusters.append([i])
            cluster_reps.append(mat)

    # 计算离散度：优质解与最优解之间的平均 RMS
    best_mat = matrices[0]
    rms_values = [compute_rms(m, best_mat) for m in matrices]
    spread = float(np.mean(rms_values))

    # 构建分配关系
    assignments = [0] * len(good_solutions)
    for c_idx, members in enumerate(clusters):
        for m in members:
            assignments[m] = c_idx

    # 每个聚类的代表解：该聚类内损失最低的解
    representatives = []
    for members in clusters:
        best_in_cluster = min(members, key=lambda i: good_solutions[i].loss)
        representatives.append(good_solutions[best_in_cluster].positions)

    return ClusterResult(
        K_geom=len(clusters),
        spread=spread,
        cluster_sizes=[len(c) for c in clusters],
        representatives=representatives,
        all_assignments=assignments,
    )


# ── Kendall Tau ──

def compute_kendall_tau(
    positions: Dict[str, np.ndarray],
    gt_positions: Dict[str, np.ndarray],
) -> float:
    """计算重建与真值之间逐对距离的 Kendall 秩相关系数。

    tau 值越高表示序数结构（距离排序）保持得越好。
    该函数与维度无关。
    """
    obj_ids = sorted(set(positions.keys()) & set(gt_positions.keys()))
    if len(obj_ids) < 2:
        return 0.0

    pairs = list(combinations(obj_ids, 2))
    recon_dists = []
    gt_dists = []

    for a, b in pairs:
        recon_dists.append(float(np.linalg.norm(positions[a] - positions[b])))
        gt_dists.append(float(np.linalg.norm(gt_positions[a] - gt_positions[b])))

    if len(pairs) < 2:
        return 0.0

    tau_val, _ = kendalltau(recon_dists, gt_dists)
    if np.isnan(tau_val):
        return 0.0
    return float(tau_val)


# ── 三维评估指标数据类 ──

@dataclass
class EvalMetrics3D:
    """三维重建的完整评估指标。

    在 2D EvalMetrics 基础上增加：
      - csr_trr_azimuth: 方位角 CSR
      - csr_trr_elevation: 仰角 CSR
      - csr_trr_full: 方位角 + 仰角同时满足的 CSR
    """
    csr_qrr: float = 0.0
    csr_trr: float = 0.0            # 2D 向后兼容（等于 csr_trr_azimuth）
    csr_trr_azimuth: float = 0.0    # 方位角约束满足率
    csr_trr_elevation: float = 0.0  # 仰角约束满足率
    csr_trr_full: float = 0.0       # 方位角 + 仰角综合满足率
    K_geom: int = 1
    spread: float = 0.0
    kendall_tau: Optional[float] = None
    nrms: Optional[float] = None
    best_loss: float = float("inf")
    n_solutions: int = 0
    cluster_sizes: List[int] = field(default_factory=list)


# ── 三维完整评估 ──

def evaluate_reconstruction_3d(
    solutions: List[SolverSolution],
    qrr_entries: List[QRREntry],
    trr3d_entries: List[TRR3DEntry],
    gt_positions: Optional[Dict[str, np.ndarray]] = None,
    rms_threshold: float = 0.10,
    tau: float = 0.10,
) -> EvalMetrics3D:
    """计算三维重建方案的所有评估指标。

    参数：
        solutions: 求解器输出的解列表（按损失排序）
        qrr_entries: QRR 约束列表
        trr3d_entries: 三维 TRR 约束列表
        gt_positions: 可选的真值 3D 位置 {obj_id -> np.ndarray(3,)}
        rms_threshold: 聚类 RMS 阈值
        tau: QRR 比值容差

    返回：
        EvalMetrics3D 包含所有评估指标
    """
    if not solutions:
        return EvalMetrics3D()

    best = solutions[0]  # 按损失排序的最优解

    # QRR 约束满足率（维度无关）
    csr_qrr = compute_csr_qrr(best.positions, qrr_entries, tau=tau)

    # 三维 TRR 约束满足率（方位角 + 仰角）
    csr_azimuth, csr_elevation, csr_full = compute_csr_trr_3d(
        best.positions, trr3d_entries,
    )

    # 多解聚类
    cluster = cluster_solutions(solutions, rms_threshold=rms_threshold)

    metrics = EvalMetrics3D(
        csr_qrr=csr_qrr,
        csr_trr=csr_azimuth,             # 向后兼容：csr_trr 等于方位角 CSR
        csr_trr_azimuth=csr_azimuth,
        csr_trr_elevation=csr_elevation,
        csr_trr_full=csr_full,
        K_geom=cluster.K_geom,
        spread=cluster.spread,
        best_loss=best.loss,
        n_solutions=len(solutions),
        cluster_sizes=cluster.cluster_sizes,
    )

    # 依赖真值的指标
    if gt_positions is not None:
        obj_ids = sorted(set(best.positions.keys()) & set(gt_positions.keys()))
        if len(obj_ids) >= 3:
            # 3D 位置矩阵
            recon_mat = np.array([best.positions[oid] for oid in obj_ids])
            gt_mat = np.array([gt_positions[oid][:3] for oid in obj_ids])

            metrics.kendall_tau = compute_kendall_tau(best.positions, gt_positions)
            metrics.nrms = compute_nrms(recon_mat, gt_mat)

    return metrics
