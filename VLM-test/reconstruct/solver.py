"""
2D scene belief solver: gauge-fixed L-BFGS-B optimization.

Stage 2 of the reconstruction pipeline:
  - 3-anchor gauge fixing (eliminate translation/rotation/scale/reflection)
  - Log-domain QRR loss + sector-tolerance TRR loss + separation regularization
  - Multi-start optimization with solution collection
"""

import math
import numpy as np
from scipy.optimize import minimize
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

from .constraints import QRREntry, TRREntry
from .utils import pair_key, rotate_vec2


# ── Hyperparameters ──

@dataclass
class SolverConfig:
    # QRR loss
    qrr_margin: float = 0.1
    qrr_delta_eq: float = 0.1
    qrr_eps: float = 1e-6
    qrr_beta: float = 10.0

    # TRR loss
    trr_tau: float = 0.1
    trr_hour_tol_deg: float = 15.0
    trr_quadrant_tol_deg: float = 45.0
    trr_beta: float = 10.0

    # Separation
    sep_eps: float = 0.05
    sep_lambda: float = 1.0

    # Solver
    n_restarts: int = 10
    maxiter: int = 500
    ftol: float = 1e-10
    gtol: float = 1e-7


# ── Solution Container ──

@dataclass
class SolverSolution:
    """A single optimization result."""
    positions: Dict[str, np.ndarray]
    loss: float
    loss_qrr: float
    loss_trr: float
    loss_sep: float
    converged: bool
    n_iter: int


# ── Gauge Fixing ──

def select_anchors(
    object_ids: List[str],
    qrr_entries: List[QRREntry],
    trr_entries: List[TRREntry],
) -> Tuple[str, str, str]:
    """Select 3 anchor objects by constraint participation frequency.

    Returns (anchor_a, anchor_b, anchor_c) where:
      anchor_a -> (0, 0)
      anchor_b -> (1, 0)
      anchor_c -> y >= 0
    """
    freq: Dict[str, int] = {}
    for oid in object_ids:
        freq[oid] = 0

    for entry in qrr_entries:
        for obj in list(entry.pair1) + list(entry.pair2):
            freq[obj] = freq.get(obj, 0) + 1
    for entry in trr_entries:
        for obj in [entry.target, entry.ref1, entry.ref2]:
            freq[obj] = freq.get(obj, 0) + 1

    # Sort by frequency (descending), then by id (for stability)
    sorted_objs = sorted(object_ids, key=lambda o: (-freq.get(o, 0), o))

    if len(sorted_objs) < 3:
        # Pad with remaining objects
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
    """Pack positions into free variable vector (excluding gauge-fixed DOFs).

    Gauge fixing (5 DOF removed):
      anchor_a = (0, 0)     -> 2 DOF (translation)
      anchor_b = (1, 0)     -> 2 DOF (rotation + scale)
      y_c >= 0              -> 1 DOF (reflection)

    Free variables: [x_c, y_c, x_3, y_3, x_4, y_4, ...]
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
    """Unpack free variables to positions dict.

    anchor_a = (0, 0), anchor_b = (1, 0) are fixed.
    y_c >= 0 enforced via L-BFGS-B bounds (not abs()).
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


# ── Loss Functions ──

def _softplus(x: float, beta: float = 1.0) -> float:
    """Numerically stable softplus: log(1 + exp(beta*x)) / beta."""
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
    """Log-domain QRR ranking loss.

    For each constraint:
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
            # Huber loss
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
    """Sector-tolerance TRR loss.

    For each constraint (target, ref1, ref2, hour):
      u = normalize(x_ref2 - x_ref1)  [12 o'clock direction]
      v = normalize(x_target - x_ref1) [target direction]
      alpha = hour_to_angle_rad(hour)
      u_alpha = rotate(u, alpha)
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

        # Reference direction: ref1 -> ref2 (12 o'clock)
        ref_vec = x_ref2 - x_ref1
        ref_norm = np.linalg.norm(ref_vec)
        if ref_norm < 1e-10:
            continue
        u = ref_vec / ref_norm

        # Target direction: ref1 -> target
        tgt_vec = x_target - x_ref1
        tgt_norm = np.linalg.norm(tgt_vec)
        if tgt_norm < 1e-10:
            continue
        v = tgt_vec / tgt_norm

        # Expected angle from hour
        alpha_rad = math.radians((entry.hour % 12) * 30.0)
        u_alpha = rotate_vec2(u, alpha_rad)

        # Tolerance half-width
        if entry.level == "hour":
            tol_rad = math.radians(config.trr_hour_tol_deg)
        else:
            tol_rad = math.radians(config.trr_quadrant_tol_deg)

        cos_diff = float(np.dot(u_alpha, v))
        cos_tol = math.cos(tol_rad)

        # Loss: penalize when cos_diff < cos_tol (i.e., outside sector)
        loss = _softplus((cos_tol - cos_diff) / tau, beta)

        total += entry.weight * loss

    return total


def compute_sep_loss(
    positions: Dict[str, np.ndarray],
    config: SolverConfig,
) -> float:
    """Separation regularization: prevent object collapse."""
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
    """Total loss function for L-BFGS-B."""
    positions = unpack_free_variables(x, anchor_a, anchor_b, anchor_c, object_ids)
    l_qrr = compute_qrr_loss(positions, qrr_entries, config)
    l_trr = compute_trr_loss(positions, trr_entries, config)
    l_sep = compute_sep_loss(positions, config)
    return l_qrr + l_trr + l_sep


# ── Multi-Start Solver ──

def solve(
    object_ids: List[str],
    qrr_entries: List[QRREntry],
    trr_entries: List[TRREntry],
    config: Optional[SolverConfig] = None,
    gt_positions: Optional[Dict[str, np.ndarray]] = None,
) -> List[SolverSolution]:
    """Run multi-start L-BFGS-B optimization.

    Args:
        object_ids: list of object IDs
        qrr_entries: QRR constraints
        trr_entries: TRR constraints
        config: solver hyperparameters
        gt_positions: ground truth positions (for initialization seeding)

    Returns:
        List of SolverSolution objects (one per restart, sorted by loss)
    """
    if config is None:
        config = SolverConfig()

    n = len(object_ids)
    if n < 3:
        # Not enough objects for meaningful reconstruction
        return []

    # Select anchors
    anchor_a, anchor_b, anchor_c = select_anchors(
        object_ids, qrr_entries, trr_entries
    )

    n_free = 2 * (n - 2)  # 2 coords per free object (anchors a,b are fixed)

    solutions = []
    free_objs = [oid for oid in object_ids
                 if oid not in (anchor_a, anchor_b)]
    c_idx = free_objs.index(anchor_c) if anchor_c in free_objs else -1

    for restart in range(config.n_restarts):
        # Random initialization: positions around unit scale
        rng = np.random.RandomState(restart * 42 + 7)
        x0 = rng.randn(n_free) * 0.5

        # Bounds: y_c >= 0 (mirror fixing via box constraint)
        bounds = [(None, None)] * n_free
        if c_idx >= 0:
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

    # Sort by total loss
    solutions.sort(key=lambda s: s.loss)
    return solutions
