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
    sep_eps: float = 0.4
    sep_lambda: float = 15.0

    # BT 比例损失
    bt_ratio_alpha: float = 1.0
    bt_max_iter: int = 50

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


def compute_bt_scores(
    qrr_entries: List[QRREntry],
    max_iter: int = 50,
) -> Dict[Tuple[str, str], float]:
    """Bradley-Terry MM 算法：从 QRR 约束拟合全局距离分数。

    将每个 object pair 视为一个 "选手"。
    QRR 约束 d(pair1) < d(pair2) 等价于 pair2 "赢了" pair1。
    (分数高 = 距离大)

    返回: {(obj_a, obj_b): score} 分数越大=距离越大。
    """
    # 收集所有 pair 和比较结果
    pairs = set()
    comparisons = []  # (winner_pair, loser_pair)

    for entry in qrr_entries:
        p1 = pair_key(*entry.pair1)
        p2 = pair_key(*entry.pair2)
        pairs.add(p1)
        pairs.add(p2)

        if entry.comparator == "<":
            comparisons.append((p2, p1))  # d(p1) < d(p2) → p2 更大 → p2 赢
        elif entry.comparator == ">":
            comparisons.append((p1, p2))  # d(p1) > d(p2) → p1 更大 → p1 赢
        # ~= → 跳过 (平局不参与 BT)

    if not comparisons:
        return {p: 1.0 for p in pairs}

    pair_list = sorted(pairs)
    pair_idx = {p: i for i, p in enumerate(pair_list)}
    n = len(pair_list)

    # MM 迭代
    s = np.ones(n, dtype=np.float64)

    for _ in range(max_iter):
        s_new = np.zeros(n)

        for i in range(n):
            wins = 0.0
            denom = 0.0

            for w, l in comparisons:
                wi, li = pair_idx[w], pair_idx[l]
                if wi == i:
                    wins += 1.0
                    denom += 1.0 / (s[i] + s[li])
                elif li == i:
                    denom += 1.0 / (s[wi] + s[i])

            if denom > 0:
                s_new[i] = wins / denom
            else:
                s_new[i] = s[i]

        # 归一化
        total = s_new.sum()
        if total > 0:
            s = s_new / total * n
        else:
            break

    return {pair_list[i]: float(s[i]) for i in range(n)}


def compute_ratio_loss(
    positions: Dict[str, np.ndarray],
    qrr_entries: List[QRREntry],
    bt_scores: Dict[Tuple[str, str], float],
    config: SolverConfig,
) -> float:
    """BT 比例保持损失：重建距离比例应接近 BT 全局分数比例。

    L = Σ (log(d1) - log(d2) - log(s1) + log(s2))²
    """
    if not bt_scores or config.bt_ratio_alpha <= 0:
        return 0.0

    total = 0.0
    eps = config.qrr_eps
    score_eps = 1e-6

    for entry in qrr_entries:
        if entry.comparator == "~=":
            continue

        p1 = pair_key(*entry.pair1)
        p2 = pair_key(*entry.pair2)

        s1 = bt_scores.get(p1, 1.0) + score_eps
        s2 = bt_scores.get(p2, 1.0) + score_eps

        d1 = np.linalg.norm(positions[p1[0]] - positions[p1[1]]) + eps
        d2 = np.linalg.norm(positions[p2[0]] - positions[p2[1]]) + eps

        log_ratio_d = math.log(d1) - math.log(d2)
        log_ratio_s = math.log(s1) - math.log(s2)

        total += entry.weight * (log_ratio_d - log_ratio_s) ** 2

    return config.bt_ratio_alpha * total


def compute_sep_loss(
    positions: Dict[str, np.ndarray],
    config: SolverConfig,
) -> float:
    """分离正则化：防止对象坍缩（平方惩罚）。"""
    total = 0.0
    objs = list(positions.keys())
    eps = config.sep_eps

    for i in range(len(objs)):
        for j in range(i + 1, len(objs)):
            dist = np.linalg.norm(positions[objs[i]] - positions[objs[j]])
            if dist < eps:
                total += ((eps - dist) / eps) ** 2

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
    bt_scores: Optional[Dict] = None,
) -> float:
    """L-BFGS-B 的总损失函数。"""
    positions = unpack_free_variables(x, anchor_a, anchor_b, anchor_c, object_ids)
    l_qrr = compute_qrr_loss(positions, qrr_entries, config)
    l_trr = compute_trr_loss(positions, trr_entries, config)
    l_sep = compute_sep_loss(positions, config)
    l_ratio = compute_ratio_loss(positions, qrr_entries, bt_scores, config) if bt_scores else 0.0
    return l_qrr + l_trr + l_sep + l_ratio


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

    # 预计算 BT 全局距离分数（一次性）
    bt_scores = None
    if config.bt_ratio_alpha > 0 and qrr_entries:
        bt_scores = compute_bt_scores(qrr_entries, max_iter=config.bt_max_iter)

    # 选择锚点
    anchor_a, anchor_b, anchor_c = select_anchors(
        object_ids, qrr_entries, trr_entries
    )

    n_free = 2 * (n - 2)  # 每个自由对象 2 个坐标（锚点 a、b 已固定）

    solutions = []
    free_objs = [oid for oid in object_ids
                 if oid not in (anchor_a, anchor_b)]
    c_idx = free_objs.index(anchor_c) if anchor_c in free_objs else -1

    # GT warm-start: 将 GT 坐标对齐到 gauge convention 作为第 0 次 restart 的初始点
    gt_x0 = None
    if gt_positions is not None and anchor_a in gt_positions and anchor_b in gt_positions:
        try:
            pa = gt_positions[anchor_a][:2]
            pb = gt_positions[anchor_b][:2]
            d_ab = np.linalg.norm(pb - pa)
            if d_ab > 1e-10:
                shifted = {oid: pos[:2] - pa for oid, pos in gt_positions.items() if oid in object_ids}
                vb = shifted[anchor_b]
                angle = -np.arctan2(vb[1], vb[0])
                scale = 1.0 / np.linalg.norm(vb)
                cos_a, sin_a = np.cos(angle), np.sin(angle)
                R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
                aligned = {oid: scale * (R @ pos) for oid, pos in shifted.items()}
                gt_x0 = pack_free_variables(aligned, anchor_a, anchor_b, anchor_c, object_ids)
        except Exception:
            gt_x0 = None

    for restart in range(config.n_restarts):
        if restart == 0 and gt_x0 is not None:
            # 第 0 次 restart: 从 GT 坐标出发
            x0 = gt_x0.copy()
        else:
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
                      qrr_entries, trr_entries, config, bt_scores),
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
