"""Coordinate prediction evaluation: Kendall tau, NRMS, CSR."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

_VLM_DIR = Path(__file__).resolve().parents[2] / "VLM-test"
if str(_VLM_DIR) not in sys.path:
    sys.path.insert(0, str(_VLM_DIR))

from reconstruct.evaluate import (
    compute_csr_qrr,
    compute_csr_trr,
    compute_kendall_tau,
    reflect_positions_y,
)
from reconstruct.utils import compute_nrms
from reconstruct.constraints import QRREntry, TRREntry


def evaluate_predictions(
    predicted: Dict[str, np.ndarray],
    gt_positions: Dict[str, np.ndarray],
    qrr_entries: List[QRREntry],
    trr_entries: List[TRREntry],
    tau: float = 0.10,
) -> dict:
    """Evaluate predicted coordinates against GT.

    Returns dict with kendall_tau, nrms, csr_qrr, csr_trr, etc.
    """
    common_ids = sorted(set(predicted.keys()) & set(gt_positions.keys()))
    n_predicted = len(predicted)
    n_gt = len(gt_positions)
    n_common = len(common_ids)
    n_missing = n_gt - n_common

    if n_common < 2:
        return {
            "kendall_tau": 0.0,
            "nrms": float("inf"),
            "csr_qrr": 0.0,
            "csr_trr": 0.0,
            "n_objects_predicted": n_predicted,
            "n_objects_gt": n_gt,
            "n_common": n_common,
            "n_missing": n_missing,
            "reflected": False,
        }

    # Restrict to common objects
    pred_common = {oid: predicted[oid] for oid in common_ids}
    gt_common = {oid: gt_positions[oid] for oid in common_ids}

    # Kendall tau
    ktau = compute_kendall_tau(pred_common, gt_common)

    # NRMS (Procrustes-aligned, tries both reflections)
    pred_mat = np.array([pred_common[oid] for oid in common_ids])
    gt_mat = np.array([gt_common[oid] for oid in common_ids])
    nrms = compute_nrms(pred_mat, gt_mat, allow_reflection=True)

    # Filter constraints to only involve common objects
    qrr_valid = _filter_qrr(qrr_entries, common_ids)
    trr_valid = _filter_trr(trr_entries, common_ids)

    # CSR QRR
    csr_qrr = compute_csr_qrr(pred_common, qrr_valid, tau) if qrr_valid else float("nan")

    # CSR TRR: try both orientations, pick best
    reflected = False
    if trr_valid:
        csr_trr_orig = compute_csr_trr(pred_common, trr_valid)
        pred_reflected = reflect_positions_y(pred_common)
        csr_trr_ref = compute_csr_trr(pred_reflected, trr_valid)
        if csr_trr_ref > csr_trr_orig:
            csr_trr = csr_trr_ref
            reflected = True
        else:
            csr_trr = csr_trr_orig
    else:
        csr_trr = float("nan")

    return {
        "kendall_tau": round(ktau, 4),
        "nrms": round(nrms, 4),
        "csr_qrr": round(csr_qrr, 4) if not np.isnan(csr_qrr) else None,
        "csr_trr": round(csr_trr, 4) if not np.isnan(csr_trr) else None,
        "n_objects_predicted": n_predicted,
        "n_objects_gt": n_gt,
        "n_common": n_common,
        "n_missing": n_missing,
        "reflected": reflected,
    }


def _filter_qrr(entries: List[QRREntry], valid_ids: List[str]) -> List[QRREntry]:
    valid_set = set(valid_ids)
    return [
        e for e in entries
        if set(e.pair1) <= valid_set and set(e.pair2) <= valid_set
    ]


def _filter_trr(entries: List[TRREntry], valid_ids: List[str]) -> List[TRREntry]:
    valid_set = set(valid_ids)
    return [
        e for e in entries
        if {e.target, e.ref1, e.ref2} <= valid_set
    ]


def aggregate_results(scene_results: List[dict]) -> dict:
    """Aggregate metrics across scenes, overall and by split."""

    def _stats(values):
        arr = np.array([v for v in values if v is not None and np.isfinite(v)])
        if len(arr) == 0:
            return {"mean": None, "median": None, "std": None, "n": 0}
        return {
            "mean": round(float(np.mean(arr)), 4),
            "median": round(float(np.median(arr)), 4),
            "std": round(float(np.std(arr)), 4),
            "n": len(arr),
        }

    def _summarize(results):
        metrics = {}
        for key in ("kendall_tau", "nrms", "csr_qrr", "csr_trr"):
            values = [r["metrics"][key] for r in results]
            metrics[key] = _stats(values)
        metrics["n_missing_total"] = sum(r["metrics"]["n_missing"] for r in results)
        return metrics

    # Overall
    overall = _summarize(scene_results)

    # By split
    by_split: Dict[str, list] = {}
    for r in scene_results:
        split = r["scene_id"].rsplit("_", 1)[0]
        by_split.setdefault(split, []).append(r)

    splits = {split: _summarize(results) for split, results in sorted(by_split.items())}

    return {"overall": overall, "by_split": splits}
