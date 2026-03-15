"""
Visual Information Gain (VIG) analysis.

Implements the three-condition experiment analysis from Section 5.2:
  VIG = d(Recon_C, GT) - d(Recon_A, GT)    [basic visual gain]
  VIG_B = d(Recon_C, GT) - d(Recon_B, GT)  [visual interference effect]

And error decomposition from Section 5.3.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class VIGResult:
    """Visual Information Gain results for a single scene."""
    scene_id: str = ""

    # NRMS values per condition
    nrms_a: Optional[float] = None  # correct image
    nrms_b: Optional[float] = None  # wrong image
    nrms_c: Optional[float] = None  # no image

    # Kendall tau per condition
    tau_a: Optional[float] = None
    tau_b: Optional[float] = None
    tau_c: Optional[float] = None

    # CSR per condition
    csr_a: Optional[float] = None
    csr_b: Optional[float] = None
    csr_c: Optional[float] = None

    # Visual Information Gain
    vig_nrms: Optional[float] = None    # nrms_c - nrms_a (positive = vision helps)
    vig_b_nrms: Optional[float] = None  # nrms_c - nrms_b
    vig_tau: Optional[float] = None     # tau_a - tau_c (positive = vision helps)

    @property
    def vision_helps(self) -> Optional[bool]:
        """Does the correct image improve reconstruction?"""
        if self.vig_nrms is not None:
            return self.vig_nrms > 0
        return None


def compute_vig(
    recon_a: dict,
    recon_b: Optional[dict],
    recon_c: Optional[dict],
    scene_id: str = "",
) -> VIGResult:
    """Compute VIG from three-condition reconstruction results.

    Args:
        recon_a: reconstruction result dict (correct image)
        recon_b: reconstruction result dict (wrong image), or None
        recon_c: reconstruction result dict (no image), or None
    """
    result = VIGResult(scene_id=scene_id)

    # Extract metrics
    def get_metrics(recon):
        if recon is None:
            return None, None, None
        m = recon.get("metrics", {})
        nrms = m.get("nrms")
        tau = m.get("kendall_tau")
        csr = (m.get("csr_qrr", 0) + m.get("csr_trr", 0)) / 2
        return nrms, tau, csr

    result.nrms_a, result.tau_a, result.csr_a = get_metrics(recon_a)
    result.nrms_b, result.tau_b, result.csr_b = get_metrics(recon_b)
    result.nrms_c, result.tau_c, result.csr_c = get_metrics(recon_c)

    # Compute VIG
    if result.nrms_a is not None and result.nrms_c is not None:
        result.vig_nrms = result.nrms_c - result.nrms_a

    if result.nrms_b is not None and result.nrms_c is not None:
        result.vig_b_nrms = result.nrms_c - result.nrms_b

    if result.tau_a is not None and result.tau_c is not None:
        result.vig_tau = result.tau_a - result.tau_c

    return result


@dataclass
class ErrorDecomposition:
    """Error decomposition: Insufficiency + Error + Solver Failure."""
    insufficiency: float = 0.0   # from missing/abstained answers
    information_error: float = 0.0  # from wrong answers
    solver_failure: float = 0.0  # from optimization issues

    @property
    def total(self) -> float:
        return self.insufficiency + self.information_error + self.solver_failure


def decompose_errors(
    scene_result: dict,
    recon_result: dict,
    gt_recon_result: Optional[dict] = None,
) -> ErrorDecomposition:
    """Decompose reconstruction error into three sources.

    Args:
        scene_result: VLM evaluation result with per_question
        recon_result: reconstruction from VLM answers
        gt_recon_result: reconstruction from GT answers (for solver failure estimate)
    """
    scores = scene_result.get("scores", scene_result)
    per_q = scores.get("per_question", [])

    n_total = len(per_q)
    if n_total == 0:
        return ErrorDecomposition()

    # Count error sources
    n_missing = sum(1 for q in per_q if q.get("predicted") is None)
    n_wrong = sum(1 for q in per_q
                  if q.get("predicted") is not None and not q.get("correct", False)
                  and not q.get("hour_correct", False))
    n_correct = n_total - n_missing - n_wrong

    # Insufficiency: proportion of missing answers
    insufficiency = n_missing / n_total

    # Information error: proportion of wrong answers
    information_error = n_wrong / n_total

    # Solver failure: estimated from GT reconstruction quality
    solver_failure = 0.0
    if gt_recon_result is not None:
        gt_nrms = gt_recon_result.get("metrics", {}).get("nrms", 0)
        if gt_nrms is not None:
            solver_failure = gt_nrms  # NRMS of GT reconstruction = solver error floor

    return ErrorDecomposition(
        insufficiency=insufficiency,
        information_error=information_error,
        solver_failure=solver_failure,
    )


def aggregate_vig_results(results: List[VIGResult]) -> dict:
    """Aggregate VIG results across scenes with statistical summary."""
    if not results:
        return {}

    def _stats(values):
        values = [v for v in values if v is not None]
        if not values:
            return {"mean": None, "std": None, "n": 0}
        return {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "median": float(np.median(values)),
            "n": len(values),
        }

    return {
        "n_scenes": len(results),
        "vig_nrms": _stats([r.vig_nrms for r in results]),
        "vig_b_nrms": _stats([r.vig_b_nrms for r in results]),
        "vig_tau": _stats([r.vig_tau for r in results]),
        "nrms_a": _stats([r.nrms_a for r in results]),
        "nrms_b": _stats([r.nrms_b for r in results]),
        "nrms_c": _stats([r.nrms_c for r in results]),
        "tau_a": _stats([r.tau_a for r in results]),
        "tau_c": _stats([r.tau_c for r in results]),
        "vision_helps_rate": sum(
            1 for r in results if r.vision_helps
        ) / len(results) if results else 0.0,
    }
