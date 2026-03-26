"""
信念忠实度差距（Belief Faithfulness Gap, BFG）计算所用的随机基线与空模型基线。

BFG = NRMS(random) - NRMS(VLM)

正 BFG 表示 VLM 重建效果优于随机猜测。
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from reconstruct import reconstruct, SolverConfig
from reconstruct.utils import compute_nrms, procrustes_align


def generate_random_answers(
    questions: List[dict],
    seed: int = 42,
) -> dict:
    """为所有问题生成随机的 VLM 风格答案。

    QRR：从 {<, ~=, >} 中均匀随机选取
    TRR：从 {1, 2, ..., 12} 中均匀随机选取

    Returns:
        与 scoring_result 兼容的字典，包含 per_question 字段
    """
    rng = np.random.RandomState(seed)
    qrr_choices = ["<", "~=", ">"]

    per_question = []
    qrr_correct = qrr_total = 0
    trr_hour = trr_quad = trr_adj = trr_total = 0

    for q in questions:
        if q["type"] == "qrr":
            qrr_total += 1
            pred = rng.choice(qrr_choices)
            correct = pred == q["gt_comparator"]
            if correct:
                qrr_correct += 1
            per_question.append({
                "qid": q["qid"], "type": "qrr",
                "predicted": pred, "gt": q["gt_comparator"],
                "correct": correct,
            })
        elif q["type"] == "trr":
            trr_total += 1
            pred = int(rng.randint(1, 13))
            h = pred == q["gt_hour"]
            from dsl.predicates import hour_to_quadrant
            qu = hour_to_quadrant(pred) == q["gt_quadrant"]
            a = abs(pred - q["gt_hour"]) <= 1 or abs(pred - q["gt_hour"]) >= 11
            if h: trr_hour += 1
            if qu: trr_quad += 1
            if a: trr_adj += 1
            per_question.append({
                "qid": q["qid"], "type": "trr",
                "predicted": pred, "gt_hour": q["gt_hour"],
                "hour_correct": h, "quadrant_correct": qu,
                "adjacent_correct": a,
            })

    return {
        "qrr_correct": qrr_correct, "qrr_total": qrr_total,
        "trr_hour_correct": trr_hour,
        "trr_quadrant_correct": trr_quad,
        "trr_adjacent_correct": trr_adj,
        "trr_total": trr_total,
        "missing": 0,
        "per_question": per_question,
    }


def compute_random_baseline_nrms(
    questions: List[dict],
    gt_positions: Dict[str, np.ndarray],
    n_samples: int = 10,
    n_restarts: int = 5,
) -> dict:
    """计算随机基线的 NRMS 分布。

    生成 n_samples 组随机答案集，分别重建后返回统计量。

    Returns:
        {
            "nrms_mean": float,
            "nrms_std": float,
            "nrms_values": [float, ...],
            "kendall_tau_mean": float,
            "kendall_tau_std": float,
        }
    """
    from reconstruct import reconstruct_from_scoring

    nrms_values = []
    tau_values = []

    for i in range(n_samples):
        random_scoring = generate_random_answers(questions, seed=i * 17 + 3)
        result = reconstruct_from_scoring(
            random_scoring, questions,
            gt_positions=gt_positions,
            n_restarts=n_restarts,
            use_correct_only=False,
        )
        m = result.metrics
        if m.nrms is not None:
            nrms_values.append(m.nrms)
        if m.kendall_tau is not None:
            tau_values.append(m.kendall_tau)

    return {
        "nrms_mean": float(np.mean(nrms_values)) if nrms_values else None,
        "nrms_std": float(np.std(nrms_values)) if nrms_values else None,
        "nrms_values": [float(v) for v in nrms_values],
        "kendall_tau_mean": float(np.mean(tau_values)) if tau_values else None,
        "kendall_tau_std": float(np.std(tau_values)) if tau_values else None,
        "n_samples": n_samples,
    }


def compute_belief_faithfulness_gap(
    vlm_nrms: float,
    random_nrms: float,
) -> float:
    """计算信念忠实度差距（Belief Faithfulness Gap，BFG）。

    BFG = NRMS(random) - NRMS(VLM)

    正值表示 VLM 优于随机基线；负值表示 VLM 劣于随机基线。
    """
    return random_nrms - vlm_nrms


def compute_gt_reconstruction_baseline(
    questions: List[dict],
    gt_positions: Dict[str, np.ndarray],
    n_restarts: int = 10,
) -> dict:
    """计算真值基线：基于完美答案进行重建。

    此基线给出求解器误差的下界（solver error floor）。

    Returns:
        包含 NRMS、CSR、Kendall tau 的字典（真值基线）。
    """
    from reconstruct import reconstruct_from_scoring

    # 构造完美评分结果（所有题均答对）
    per_question = []
    for q in questions:
        if q["type"] == "qrr":
            per_question.append({
                "qid": q["qid"], "type": "qrr",
                "predicted": q["gt_comparator"], "gt": q["gt_comparator"],
                "correct": True,
            })
        elif q["type"] == "trr":
            per_question.append({
                "qid": q["qid"], "type": "trr",
                "predicted": q["gt_hour"], "gt_hour": q["gt_hour"],
                "hour_correct": True, "quadrant_correct": True,
                "adjacent_correct": True,
            })

    scoring = {
        "qrr_correct": sum(1 for q in questions if q["type"] == "qrr"),
        "qrr_total": sum(1 for q in questions if q["type"] == "qrr"),
        "trr_hour_correct": sum(1 for q in questions if q["type"] == "trr"),
        "trr_quadrant_correct": sum(1 for q in questions if q["type"] == "trr"),
        "trr_adjacent_correct": sum(1 for q in questions if q["type"] == "trr"),
        "trr_total": sum(1 for q in questions if q["type"] == "trr"),
        "missing": 0,
        "per_question": per_question,
    }

    result = reconstruct_from_scoring(
        scoring, questions,
        gt_positions=gt_positions,
        n_restarts=n_restarts,
        use_correct_only=True,
    )

    return {
        "csr_qrr": result.metrics.csr_qrr,
        "csr_trr": result.metrics.csr_trr,
        "nrms": result.metrics.nrms,
        "kendall_tau": result.metrics.kendall_tau,
        "status": result.status,
        "K_geom": result.K_geom,
    }
