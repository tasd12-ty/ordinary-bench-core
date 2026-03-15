"""
Evaluation metrics for scene belief reconstruction.

CSR, K_geom, spread, Kendall tau, NRMS + multi-solution clustering.
"""

import math
import numpy as np
from itertools import combinations
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from scipy.stats import kendalltau

from .constraints import QRREntry, TRREntry
from .solver import SolverSolution, SolverConfig
from .utils import procrustes_align, compute_nrms, compute_rms, pair_key


# ── Constraint Satisfaction Rate ──

def compute_csr_qrr(
    positions: Dict[str, np.ndarray],
    qrr_entries: List[QRREntry],
    tau: float = 0.10,
) -> float:
    """Fraction of QRR constraints satisfied by the reconstructed positions.

    Uses the same ratio-based tolerance as the original comparator.
    """
    if not qrr_entries:
        return 1.0

    satisfied = 0
    for entry in qrr_entries:
        p1 = pair_key(*entry.pair1)
        p2 = pair_key(*entry.pair2)

        d1 = float(np.linalg.norm(positions[p1[0]] - positions[p1[1]]))
        d2 = float(np.linalg.norm(positions[p2[0]] - positions[p2[1]]))

        # Determine what the reconstructed comparator would be
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

        # Strict matching: comparator must match exactly
        if recon_cmp == entry.comparator:
            satisfied += 1

    return satisfied / len(qrr_entries)


def compute_csr_trr(
    positions: Dict[str, np.ndarray],
    trr_entries: List[TRREntry],
) -> float:
    """Fraction of TRR constraints satisfied (reconstructed angle within tolerance)."""
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

        # Compute reconstructed angle
        ref_angle = math.atan2(ref_vec[1], ref_vec[0])
        tgt_angle = math.atan2(tgt_vec[1], tgt_vec[0])
        rel_angle = math.degrees(tgt_angle - ref_angle) % 360

        # Expected angle from hour
        expected_angle = (entry.hour % 12) * 30.0

        # Angular distance
        diff = abs(rel_angle - expected_angle)
        diff = min(diff, 360 - diff)

        if entry.level == "hour":
            if diff <= 15.0:
                satisfied += 1
        else:  # quadrant
            if diff <= 45.0:
                satisfied += 1

    return satisfied / evaluated if evaluated > 0 else 1.0


# ── Multi-Solution Clustering ──

@dataclass
class ClusterResult:
    """Result of clustering multiple solutions."""
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
    """Cluster solutions by RMS distance (in unified gauge space).

    Since all solutions share the same gauge (3-anchor), no Procrustes needed.
    Simple greedy clustering: assign each solution to nearest cluster or create new.

    Only considers solutions with loss <= loss_ratio_cutoff * best_loss to
    filter out failed local minima that aren't true geometric modes.
    """
    if not solutions:
        return ClusterResult()

    # Filter by loss quality (solutions are sorted by loss)
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

    # Extract position matrices (from good solutions only)
    obj_ids = sorted(good_solutions[0].positions.keys())

    def to_matrix(pos: Dict[str, np.ndarray]) -> np.ndarray:
        return np.array([pos[oid] for oid in obj_ids])

    matrices = [to_matrix(s.positions) for s in good_solutions]

    # Greedy clustering
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

    # Compute spread: average RMS of good solutions from best solution
    best_mat = matrices[0]
    rms_values = [compute_rms(m, best_mat) for m in matrices]
    spread = float(np.mean(rms_values))

    # Build assignments
    assignments = [0] * len(good_solutions)
    for c_idx, members in enumerate(clusters):
        for m in members:
            assignments[m] = c_idx

    # Representatives: best (lowest loss) solution in each cluster
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
    """Kendall rank correlation between reconstructed and GT pairwise distances.

    Higher tau means the ordinal structure (distance ranking) is better preserved.
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


# ── Full Evaluation ──

@dataclass
class EvalMetrics:
    """Complete evaluation metrics for a reconstruction."""
    csr_qrr: float = 0.0
    csr_trr: float = 0.0
    K_geom: int = 1
    spread: float = 0.0
    kendall_tau: Optional[float] = None
    nrms: Optional[float] = None
    best_loss: float = float("inf")
    n_solutions: int = 0
    cluster_sizes: List[int] = field(default_factory=list)


def evaluate_reconstruction(
    solutions: List[SolverSolution],
    qrr_entries: List[QRREntry],
    trr_entries: List[TRREntry],
    gt_positions: Optional[Dict[str, np.ndarray]] = None,
    rms_threshold: float = 0.10,
    tau: float = 0.10,
) -> EvalMetrics:
    """Compute all evaluation metrics for a set of reconstruction solutions."""
    if not solutions:
        return EvalMetrics()

    best = solutions[0]  # sorted by loss

    # CSR
    csr_qrr = compute_csr_qrr(best.positions, qrr_entries, tau=tau)
    csr_trr = compute_csr_trr(best.positions, trr_entries)

    # Clustering
    cluster = cluster_solutions(solutions, rms_threshold=rms_threshold)

    metrics = EvalMetrics(
        csr_qrr=csr_qrr,
        csr_trr=csr_trr,
        K_geom=cluster.K_geom,
        spread=cluster.spread,
        best_loss=best.loss,
        n_solutions=len(solutions),
        cluster_sizes=cluster.cluster_sizes,
    )

    # GT-dependent metrics
    if gt_positions is not None:
        obj_ids = sorted(set(best.positions.keys()) & set(gt_positions.keys()))
        if len(obj_ids) >= 3:
            recon_mat = np.array([best.positions[oid] for oid in obj_ids])
            gt_mat = np.array([gt_positions[oid][:2] for oid in obj_ids])

            metrics.kendall_tau = compute_kendall_tau(best.positions, gt_positions)
            metrics.nrms = compute_nrms(recon_mat, gt_mat)

    return metrics
