"""
场景信念重建的评估指标。

CSR、K_geom、spread、Kendall tau、NRMS + 多解聚类。
三层判定准则：solver 对齐 CSR、归一化损失 NRL、二项显著性检验。
"""

import math
import numpy as np
from itertools import combinations
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from scipy.stats import kendalltau, binom

from .constraints import QRREntry, TRREntry
from .solver import SolverSolution, SolverConfig
from .utils import (
    angular_distance,
    compute_nrms,
    compute_rms,
    hour_to_angle_deg,
    pair_key,
    procrustes_align,
    relative_clock_angle_deg,
)


# ── 镜像反射辅助函数 ──

def reflect_positions_y(positions: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """将所有位置沿 x 轴镜像反射：(x, y) -> (x, -y)。

    在规范固定坐标系 (anchor_a=(0,0), anchor_b=(1,0)) 中，
    这是唯一保持规范不变的非平凡镜像反射。
    """
    return {oid: np.array([pos[0], -pos[1]]) for oid, pos in positions.items()}


# ── 约束满足率 ──

def compute_csr_qrr(
    positions: Dict[str, np.ndarray],
    qrr_entries: List[QRREntry],
    tau: float = 0.10,
) -> float:
    """重建位置满足的 QRR 约束比例。

    使用与原始比较器相同的比率容差。
    """
    if not qrr_entries:
        return 1.0

    satisfied = 0
    for entry in qrr_entries:
        p1 = pair_key(*entry.pair1)
        p2 = pair_key(*entry.pair2)

        d1 = float(np.linalg.norm(positions[p1[0]] - positions[p1[1]]))
        d2 = float(np.linalg.norm(positions[p2[0]] - positions[p2[1]]))

        # 确定重建后的比较器
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

        # 严格匹配：比较器必须完全一致
        if recon_cmp == entry.comparator:
            satisfied += 1

    return satisfied / len(qrr_entries)


def compute_csr_trr(
    positions: Dict[str, np.ndarray],
    trr_entries: List[TRREntry],
) -> float:
    """满足的 TRR 约束比例（重建角度在容差范围内）。"""
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

        # 计算重建后的顺时针钟面角度
        rel_angle = relative_clock_angle_deg(ref_vec, tgt_vec)
        expected_angle = hour_to_angle_deg(entry.hour)
        diff = angular_distance(rel_angle, expected_angle)

        if entry.level == "hour":
            if diff <= 15.0:
                satisfied += 1
        else:  # quadrant
            if diff <= 45.0:
                satisfied += 1

    return satisfied / evaluated if evaluated > 0 else 1.0


# ── Solver 对齐的 CSR（log 域） ──

def compute_csr_qrr_aligned(
    positions: Dict[str, np.ndarray],
    qrr_entries: List[QRREntry],
    config: Optional[SolverConfig] = None,
) -> float:
    """在 log 域评估 QRR 约束满足率，与 solver 损失函数容差一致。

    solver 在 log(d1)-log(d2) 空间优化，使用 delta_eq 作为死区宽度。
    本函数使用同一容差判定约束是否满足，消除 ratio 域和 log 域的边界不一致。
    """
    if not qrr_entries:
        return 1.0

    cfg = config or SolverConfig()
    eps = cfg.qrr_eps
    delta_eq = cfg.qrr_delta_eq

    satisfied = 0
    for entry in qrr_entries:
        p1 = pair_key(*entry.pair1)
        p2 = pair_key(*entry.pair2)

        d1 = float(np.linalg.norm(positions[p1[0]] - positions[p1[1]])) + eps
        d2 = float(np.linalg.norm(positions[p2[0]] - positions[p2[1]])) + eps
        delta = math.log(d1) - math.log(d2)

        if entry.comparator == "<":
            if delta < delta_eq:
                satisfied += 1
        elif entry.comparator == ">":
            if -delta < delta_eq:
                satisfied += 1
        else:  # ~=
            if abs(delta) < delta_eq:
                satisfied += 1

    return satisfied / len(qrr_entries)


# ── 归一化重建损失 ──

def compute_nrl(best_loss: float, n_qrr: int, n_trr: int) -> float:
    """归一化重建损失：solver 最优损失除以约束总数。"""
    n_total = n_qrr + n_trr
    if n_total == 0:
        return 0.0
    return best_loss / n_total


def estimate_nrl_random(margin: float = 0.1, beta: float = 10.0) -> float:
    """解析估算随机基线下的期望 NRL。

    随机 QRR 约 1/3 正确（损失≈0），2/3 错误（损失≈softplus(margin, beta)）。
    随机 TRR 约 1/12 正确，11/12 错误。
    取 QRR 占主导的近似值。
    """
    bx = beta * margin
    sp = math.log1p(math.exp(bx)) / beta if bx < 20 else margin
    return (2.0 / 3.0) * sp


# ── 二项显著性检验 ──

def compute_significance(
    csr_qrr: float,
    n_qrr: int,
    csr_trr: float,
    n_trr: int,
    alpha: float = 0.01,
    p_chance_qrr: float = 1.0 / 3.0,
    p_chance_trr: float = 1.0 / 12.0,
) -> Tuple[float, float, bool]:
    """二项检验：CSR 是否显著高于随机猜测。

    返回 (p_value_qrr, p_value_trr, is_significant)。
    """
    if n_qrr > 0:
        k_qrr = round(csr_qrr * n_qrr)
        p_qrr = 1.0 - binom.cdf(k_qrr - 1, n_qrr, p_chance_qrr) if k_qrr > 0 else 1.0
    else:
        p_qrr = 1.0

    if n_trr > 0:
        k_trr = round(csr_trr * n_trr)
        p_trr = 1.0 - binom.cdf(k_trr - 1, n_trr, p_chance_trr) if k_trr > 0 else 1.0
    else:
        p_trr = 1.0

    qrr_sig = p_qrr < alpha if n_qrr > 0 else True
    trr_sig = p_trr < alpha if n_trr > 0 else True
    return float(p_qrr), float(p_trr), qrr_sig and trr_sig


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
    """按 RMS 距离对解进行聚类（在统一规范空间中）。

    由于所有解共享相同的规范（3 锚点），无需 Procrustes 对齐。
    简单贪心聚类：将每个解分配到最近的聚类，或创建新聚类。

    仅考虑 loss <= loss_ratio_cutoff * best_loss 的解，
    以过滤掉非真实几何模态的失败局部最小值。
    """
    if not solutions:
        return ClusterResult()

    # 按损失质量过滤（解已按损失排序）
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

    # 提取位置矩阵（仅从优质解中）
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

    # 计算 spread：优质解与最优解之间的平均 RMS
    best_mat = matrices[0]
    rms_values = [compute_rms(m, best_mat) for m in matrices]
    spread = float(np.mean(rms_values))

    # 构建分配结果
    assignments = [0] * len(good_solutions)
    for c_idx, members in enumerate(clusters):
        for m in members:
            assignments[m] = c_idx

    # 代表解：每个聚类中损失最低的解
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


# ── Kendall Tau 秩相关 ──

def compute_kendall_tau(
    positions: Dict[str, np.ndarray],
    gt_positions: Dict[str, np.ndarray],
) -> float:
    """重建与真值的成对距离之间的 Kendall 秩相关系数。

    tau 越高表示序数结构（距离排序）保持得越好。
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


# ── 完整评估 ──

@dataclass
class EvalMetrics:
    """重建的完整评估指标。"""
    csr_qrr: float = 0.0
    csr_trr: float = 0.0
    csr_qrr_aligned: float = 0.0  # solver 对齐的 log 域 CSR
    nrl: float = float("inf")     # 归一化重建损失
    p_value_qrr: float = 1.0     # QRR 二项检验 p 值
    p_value_trr: float = 1.0     # TRR 二项检验 p 值
    K_geom: int = 1
    spread: float = 0.0
    kendall_tau: Optional[float] = None
    nrms: Optional[float] = None
    best_loss: float = float("inf")
    n_solutions: int = 0
    cluster_sizes: List[int] = field(default_factory=list)
    reflected: bool = False  # 若 y 镜像反射的手性给出了更好的 CSR_TRR 则为 True


def evaluate_reconstruction(
    solutions: List[SolverSolution],
    qrr_entries: List[QRREntry],
    trr_entries: List[TRREntry],
    gt_positions: Optional[Dict[str, np.ndarray]] = None,
    rms_threshold: float = 0.10,
    tau: float = 0.10,
    solver_config: Optional[SolverConfig] = None,
) -> EvalMetrics:
    """为一组重建解计算所有评估指标。

    重建结果仅在相似变换（平移、旋转、缩放和镜像反射）意义下确定。
    对于 CSR_TRR 和 NRMS，我们评估两种手性并保留较优结果。
    CSR_QRR 和 Kendall tau 在构造上具有镜像反射不变性。

    新增三层判定指标：
    - csr_qrr_aligned: solver 对齐的 log 域 CSR
    - nrl: 归一化重建损失
    - p_value_qrr/p_value_trr: 二项检验 p 值
    """
    if not solutions:
        return EvalMetrics()

    best = solutions[0]  # 已按损失排序

    # CSR_QRR：在镜像反射下不变（距离保持）
    csr_qrr = compute_csr_qrr(best.positions, qrr_entries, tau=tau)

    # Solver 对齐的 CSR（log 域）
    csr_qrr_aligned = compute_csr_qrr_aligned(best.positions, qrr_entries, solver_config)

    # CSR_TRR：尝试两种手性，取较优者
    csr_trr_orig = compute_csr_trr(best.positions, trr_entries)
    reflected = reflect_positions_y(best.positions)
    csr_trr_refl = compute_csr_trr(reflected, trr_entries)
    csr_trr = max(csr_trr_orig, csr_trr_refl)
    # 记录哪种手性更优，供下游使用
    used_reflection = csr_trr_refl > csr_trr_orig
    best_positions = reflected if used_reflection else best.positions

    # 归一化重建损失
    nrl = compute_nrl(best.loss, len(qrr_entries), len(trr_entries))

    # 二项显著性检验
    p_qrr, p_trr, _ = compute_significance(
        csr_qrr_aligned, len(qrr_entries),
        csr_trr, len(trr_entries),
    )

    # 聚类
    cluster = cluster_solutions(solutions, rms_threshold=rms_threshold)

    metrics = EvalMetrics(
        csr_qrr=csr_qrr,
        csr_trr=csr_trr,
        csr_qrr_aligned=csr_qrr_aligned,
        nrl=nrl,
        p_value_qrr=p_qrr,
        p_value_trr=p_trr,
        K_geom=cluster.K_geom,
        spread=cluster.spread,
        best_loss=best.loss,
        n_solutions=len(solutions),
        cluster_sizes=cluster.cluster_sizes,
        reflected=used_reflection,
    )

    # 依赖真值的指标（Kendall tau 具有镜像反射不变性，
    # NRMS 使用 allow_reflection=True 的 Procrustes 对齐）
    if gt_positions is not None:
        obj_ids = sorted(set(best_positions.keys()) & set(gt_positions.keys()))
        if len(obj_ids) >= 3:
            recon_mat = np.array([best_positions[oid] for oid in obj_ids])
            gt_mat = np.array([gt_positions[oid][:2] for oid in obj_ids])

            metrics.kendall_tau = compute_kendall_tau(best_positions, gt_positions)
            metrics.nrms = compute_nrms(recon_mat, gt_mat, allow_reflection=True)

    return metrics
