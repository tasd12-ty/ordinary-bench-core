"""
三维问题枚举与批次分割。

从真值约束枚举所有 QRR、TRR（3D）和 FDR 问题，
分配问题 ID，按可配置大小分批。
"""

from itertools import combinations, permutations
from typing import Dict, List

from dsl.predicates import (
    MetricType, compute_qrr, compute_trr_3d, compute_fdr,
    _is_boundary, METRIC_FUNCTIONS,
)


def enumerate_qrr(
    objects: Dict[str, Dict],
    tau: float = 0.10,
    metric: MetricType = MetricType.DIST_3D,
    include_disjoint: bool = True,
    include_shared_anchor: bool = True,
) -> List[dict]:
    """
    Enumerate QRR questions with GT answers.

    Variants:
      - disjoint: compare two disjoint pairs
      - shared_anchor: compare distances from a common anchor

    For N objects:
      - disjoint count = 3 * C(N,4)
      - shared-anchor count = N * C(N-1, 2)

    Returns list of question dicts.
    """
    obj_ids = sorted(objects.keys())
    metric_func = METRIC_FUNCTIONS[metric]
    questions = []
    qid = 0

    if include_disjoint:
        pairs = list(combinations(obj_ids, 2))
        for i, pair1 in enumerate(pairs):
            for pair2 in pairs[i + 1:]:
                if set(pair1) & set(pair2):
                    continue
                m1 = metric_func(objects[pair1[0]], objects[pair1[1]])
                m2 = metric_func(objects[pair2[0]], objects[pair2[1]])
                if _is_boundary(m1, m2, tau):
                    continue

                constraint = compute_qrr(
                    objects, pair1, pair2, metric, tau,
                    variant="disjoint",
                )
                qid += 1
                questions.append({
                    "qid": f"qrr_{qid:04d}",
                    "type": "qrr",
                    "variant": "disjoint",
                    "pair1": list(constraint.pair1),
                    "pair2": list(constraint.pair2),
                    "metric": str(metric),
                    "gt_comparator": str(constraint.comparator),
                })

    if include_shared_anchor:
        for anchor in obj_ids:
            others = [oid for oid in obj_ids if oid != anchor]
            for obj_a, obj_b in combinations(others, 2):
                pair1 = (anchor, obj_a)
                pair2 = (anchor, obj_b)
                m1 = metric_func(objects[pair1[0]], objects[pair1[1]])
                m2 = metric_func(objects[pair2[0]], objects[pair2[1]])
                if _is_boundary(m1, m2, tau):
                    continue

                constraint = compute_qrr(
                    objects, pair1, pair2, metric, tau,
                    variant="shared_anchor", anchor=anchor,
                )
                qid += 1
                questions.append({
                    "qid": f"qrr_{qid:04d}",
                    "type": "qrr",
                    "variant": "shared_anchor",
                    "anchor": anchor,
                    "pair1": list(constraint.pair1),
                    "pair2": list(constraint.pair2),
                    "metric": str(metric),
                    "gt_comparator": str(constraint.comparator),
                })

    return questions


def enumerate_trr_3d(
    objects: Dict[str, Dict],
) -> List[dict]:
    """枚举所有三维 TRR 问题及其真值答案。

    对 N 个物体生成 P(N,3) = N*(N-1)*(N-2) 个问题。
    每个问题包含水平方位角（钟面方向）和垂直仰角信息。

    返回问题字典列表。
    """
    obj_ids = sorted(objects.keys())
    questions = []
    qid = 0

    for triple in permutations(obj_ids, 3):
        target, ref1, ref2 = triple
        # 调用三维 TRR 计算，获取方位角和仰角
        constraint = compute_trr_3d(objects, target, ref1, ref2)
        qid += 1
        questions.append({
            "qid": f"trr_{qid:04d}",
            "type": "trr",
            "target": constraint.target,
            "ref1": constraint.ref1,
            "ref2": constraint.ref2,
            # 水平方位角真值（与 2D TRR 兼容）
            "gt_hour": constraint.hour,
            "gt_quadrant": constraint.quadrant,
            "gt_angle_deg": round(constraint.azimuth_deg, 2),
            # 垂直仰角真值（三维扩展）
            "gt_elevation_deg": round(constraint.elevation_deg, 2),
            "gt_elevation_band": constraint.elevation_band,
        })

    return questions


def enumerate_fdr(
    objects: Dict[str, Dict],
    tau: float = 0.10,
) -> List[dict]:
    """
    Enumerate all FDR (Full Distance Ranking) questions.

    For N objects, FDR count = N (one per anchor object).

    Returns list of question dicts.
    """
    obj_ids = sorted(objects.keys())
    questions = []
    qid = 0

    for anchor in obj_ids:
        constraint = compute_fdr(objects, anchor, tau)
        qid += 1
        questions.append({
            "qid": f"fdr_{qid:04d}",
            "type": "fdr",
            "anchor": constraint.anchor,
            "n_ranked": len(constraint.ranking),
            "gt_ranking": constraint.ranking,
            "gt_distances": [round(d, 6) for d in constraint.distances],
            "gt_tie_groups": constraint.tie_groups,
        })

    return questions


def make_batches(questions: List[dict], batch_size: int) -> List[dict]:
    """
    Split questions into batches.

    Returns list of batch dicts with batch_id and questions.
    """
    batches = []
    for i in range(0, len(questions), batch_size):
        chunk = questions[i:i + batch_size]
        batches.append({
            "batch_id": len(batches),
            "n_questions": len(chunk),
            "questions": chunk,
        })
    return batches


def question_counts(n_objects: int) -> dict:
    """
    Calculate expected question counts for N objects.

    QRR combines disjoint pair-vs-pair and shared-anchor comparisons.
    TRR: P(N,3) = N*(N-1)*(N-2)
    """
    qrr_count = 0
    qrr_disjoint = 0
    ids = list(range(n_objects))
    pairs = list(combinations(ids, 2))
    for i, p1 in enumerate(pairs):
        for p2 in pairs[i + 1:]:
            if not (set(p1) & set(p2)):
                qrr_disjoint += 1

    qrr_shared_anchor = n_objects * (n_objects - 1) * (n_objects - 2) // 2
    qrr_count = qrr_disjoint + qrr_shared_anchor

    trr_count = n_objects * (n_objects - 1) * (n_objects - 2)
    fdr_count = n_objects

    return {
        "n_objects": n_objects,
        "n_qrr_disjoint": qrr_disjoint,
        "n_qrr_shared_anchor": qrr_shared_anchor,
        "n_qrr": qrr_count,
        "n_trr": trr_count,
        "n_fdr": fdr_count,
        "total": qrr_count + trr_count + fdr_count,
    }
