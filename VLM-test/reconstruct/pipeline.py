"""
End-to-end reconstruction pipeline.

Two entry points:
  1. reconstruct() - from raw constraint lists
  2. prepare_reconstruction_input_from_scoring() - auditable prep bundle
  3. reconstruct_from_scoring() - from scoring results + question metadata
  4. reconstruct_from_prepared() - from prepared bundle
"""

import numpy as np
from typing import Dict, List, Optional, Union
from dataclasses import dataclass, field

from .constraints import (
    QRREntry, TRREntry, FDREntry,
    build_distance_poset, build_angular_sectors,
    analyze_hypergraph, check_feasibility,
    FeasibilityReport,
)
from .preparation import (
    PreparedSceneInput,
    prepare_reconstruction_input_from_scoring,
)
from .solver import SolverConfig, SolverSolution, solve
from .evaluate import EvalMetrics, evaluate_reconstruction, cluster_solutions


@dataclass
class ReconstructResult:
    """Complete reconstruction output."""
    feasible: bool = False
    status: str = "infeasible"  # infeasible | underconstrained | single_mode | multimodal
    positions: Dict[str, np.ndarray] = field(default_factory=dict)
    metrics: EvalMetrics = field(default_factory=EvalMetrics)
    K_geom: int = 0
    all_solutions: List[SolverSolution] = field(default_factory=list)
    feasibility_checks: FeasibilityReport = field(default_factory=FeasibilityReport)

    def to_dict(self) -> dict:
        """Serialize to JSON-compatible dict."""
        pos_dict = {k: v.tolist() for k, v in self.positions.items()}
        return {
            "feasible": self.feasible,
            "status": self.status,
            "positions": pos_dict,
            "metrics": {
                "csr_qrr": self.metrics.csr_qrr,
                "csr_trr": self.metrics.csr_trr,
                "K_geom": self.metrics.K_geom,
                "spread": self.metrics.spread,
                "kendall_tau": self.metrics.kendall_tau,
                "nrms": self.metrics.nrms,
                "best_loss": self.metrics.best_loss,
                "n_solutions": self.metrics.n_solutions,
                "cluster_sizes": self.metrics.cluster_sizes,
            },
            "K_geom": self.K_geom,
            "n_solutions": len(self.all_solutions),
            "feasibility_checks": {
                "qrr_has_cycle": self.feasibility_checks.qrr_has_cycle,
                "qrr_cycle_info": self.feasibility_checks.qrr_cycle_info,
                "trr_n_conflicts": self.feasibility_checks.trr_n_conflicts,
                "hypergraph_connected": self.feasibility_checks.hypergraph_connected,
                "n_components": self.feasibility_checks.n_components,
                "n_qrr": self.feasibility_checks.n_qrr,
                "n_trr": self.feasibility_checks.n_trr,
                "n_objects": self.feasibility_checks.n_objects,
            },
        }


def reconstruct(
    qrr_constraints: List[dict],
    trr_constraints: List[dict],
    object_ids: List[str],
    gt_positions: Optional[Dict[str, np.ndarray]] = None,
    n_restarts: int = 10,
    config: Optional[SolverConfig] = None,
) -> ReconstructResult:
    """Reconstruct 2D scene from QRR + TRR constraints.

    Args:
        qrr_constraints: list of QRR constraint dicts with keys:
            pair1, pair2, comparator, [weight], [variant], [anchor]
        trr_constraints: list of TRR constraint dicts with keys:
            target, ref1, ref2, hour, [weight], [level]
        object_ids: list of object identifiers
        gt_positions: optional ground truth positions for evaluation
        n_restarts: number of optimization restarts
        config: solver configuration

    Returns:
        ReconstructResult with positions, metrics, and diagnostics
    """
    if config is None:
        config = SolverConfig(n_restarts=n_restarts)
    else:
        config.n_restarts = n_restarts

    # Parse constraint dicts to entry objects
    qrr_entries = [
        QRREntry(
            pair1=tuple(c["pair1"]),
            pair2=tuple(c["pair2"]),
            comparator=c["comparator"],
            weight=c.get("weight", 1.0),
            variant=c.get("variant", "disjoint"),
            anchor=c.get("anchor"),
        )
        for c in qrr_constraints
    ]
    trr_entries = [
        TRREntry(
            target=c["target"],
            ref1=c["ref1"],
            ref2=c["ref2"],
            hour=c["hour"],
            weight=c.get("weight", 1.0),
            level=c.get("level", "hour"),
        )
        for c in trr_constraints
    ]

    return _run_pipeline(qrr_entries, trr_entries, object_ids, gt_positions, config)


def reconstruct_from_scoring(
    scoring_result: dict,
    questions: List[dict],
    gt_positions: Optional[Dict[str, np.ndarray]] = None,
    n_restarts: int = 10,
    use_correct_only: bool = True,
    config: Optional[SolverConfig] = None,
) -> ReconstructResult:
    """Reconstruct from scoring output of score_batch_scene().

    Args:
        scoring_result: output of score_batch_scene() with per_question list
        questions: original question list with GT metadata
        gt_positions: ground truth positions (dict of obj_id -> [x, y] or [x, y, z])
        n_restarts: number of optimization restarts
        use_correct_only: if True, only use correctly answered constraints;
                         if False, use all VLM predictions (for belief reconstruction)
        config: solver configuration
    """
    if config is None:
        config = SolverConfig(n_restarts=n_restarts)
    else:
        config.n_restarts = n_restarts

    gt_serialized = None
    if gt_positions is not None:
        gt_serialized = {
            oid: np.asarray(pos, dtype=np.float64).tolist()
            for oid, pos in gt_positions.items()
        }

    prepared = prepare_reconstruction_input_from_scoring(
        scoring_result=scoring_result,
        questions=questions,
        gt_positions=gt_serialized,
        use_correct_only=use_correct_only,
    )
    return reconstruct_from_prepared(prepared, n_restarts=n_restarts, config=config)


def reconstruct_from_prepared(
    prepared_input: Union[PreparedSceneInput, dict],
    n_restarts: int = 10,
    config: Optional[SolverConfig] = None,
) -> ReconstructResult:
    """Reconstruct from a prepared per-scene bundle."""
    if config is None:
        config = SolverConfig(n_restarts=n_restarts)
    else:
        config.n_restarts = n_restarts

    prepared = (
        prepared_input
        if isinstance(prepared_input, PreparedSceneInput)
        else PreparedSceneInput.from_dict(prepared_input)
    )

    gt_positions = None
    if prepared.gt_positions:
        gt_positions = {
            oid: np.asarray(pos, dtype=np.float64)
            for oid, pos in prepared.gt_positions.items()
        }

    return reconstruct(
        qrr_constraints=prepared.qrr_all,
        trr_constraints=prepared.trr_constraints,
        object_ids=prepared.object_ids,
        gt_positions=gt_positions,
        n_restarts=n_restarts,
        config=config,
    )


def _run_pipeline(
    qrr_entries: List[QRREntry],
    trr_entries: List[TRREntry],
    object_ids: List[str],
    gt_positions: Optional[Dict[str, np.ndarray]],
    config: SolverConfig,
) -> ReconstructResult:
    """Internal pipeline: Stage 0-5."""

    result = ReconstructResult()

    # ── Stage 0-1: Symbolic Preprocessing ──
    poset = build_distance_poset(qrr_entries)
    sectors = build_angular_sectors(trr_entries)
    hyper = analyze_hypergraph(qrr_entries, trr_entries, object_ids)
    feasibility = check_feasibility(poset, sectors, hyper)
    result.feasibility_checks = feasibility

    # Not enough objects
    if len(object_ids) < 3:
        result.status = "underconstrained"
        return result

    # No constraints at all
    if not qrr_entries and not trr_entries:
        result.status = "underconstrained"
        return result

    # Fail-fast on symbolic infeasibility
    if feasibility.qrr_has_cycle:
        result.feasible = False
        result.status = "infeasible"
        return result
    if feasibility.trr_n_conflicts > 0:
        result.feasible = False
        result.status = "infeasible"
        return result

    # ── Stage 2-3: Numerical Reconstruction ──
    # Normalize GT positions to 2D
    gt_2d = None
    if gt_positions is not None:
        gt_2d = {}
        for oid, pos in gt_positions.items():
            if oid in object_ids:
                gt_2d[oid] = np.array(pos[:2], dtype=np.float64)

    solutions = solve(
        object_ids=object_ids,
        qrr_entries=qrr_entries,
        trr_entries=trr_entries,
        config=config,
        gt_positions=gt_2d,
    )

    if not solutions:
        result.status = "infeasible"
        return result

    result.all_solutions = solutions
    result.positions = solutions[0].positions  # Best solution

    # ── Stage 4-5: Evaluation ──
    metrics = evaluate_reconstruction(
        solutions=solutions,
        qrr_entries=qrr_entries,
        trr_entries=trr_entries,
        gt_positions=gt_2d,
    )
    result.metrics = metrics
    result.K_geom = metrics.K_geom

    # ── Status Determination ──
    csr_ok = metrics.csr_qrr >= 0.95 and metrics.csr_trr >= 0.95
    if not csr_ok:
        result.feasible = False
        result.status = "infeasible"
    elif metrics.K_geom == 1 and metrics.spread <= 0.10:
        result.feasible = True
        result.status = "single_mode"
    elif metrics.K_geom == 1 and metrics.spread > 0.10:
        result.feasible = True
        result.status = "underconstrained"
    else:
        result.feasible = True
        result.status = "multimodal"

    return result
