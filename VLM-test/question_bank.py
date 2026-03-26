"""
问题枚举与批次分割。

从 GT 约束中枚举所有 QRR、TRR 和 FDR 问题，
分配问题 ID，并按可配置大小分批。
"""

from itertools import combinations, permutations
from typing import Dict, List

from dsl.predicates import (
    MetricType, compute_qrr, compute_trr, compute_fdr,
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
    枚举带真值答案的 QRR 问题。

    变体：
      - disjoint：比较两对不相交的物体对
      - shared_anchor：比较从公共锚点出发的距离

    对于 N 个物体：
      - disjoint 数量 = 3 * C(N,4)
      - shared_anchor 数量 = N * C(N-1, 2)

    返回问题字典列表。
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


def enumerate_trr(
    objects: Dict[str, Dict],
    use_3d: bool = True,
) -> List[dict]:
    """
    枚举带真值答案的所有 TRR 问题。

    对于 N 个物体，TRR 数量 = P(N,3) = N*(N-1)*(N-2)

    返回问题字典列表。
    """
    obj_ids = sorted(objects.keys())
    questions = []
    qid = 0

    for triple in permutations(obj_ids, 3):
        target, ref1, ref2 = triple
        constraint = compute_trr(objects, target, ref1, ref2, use_3d)
        qid += 1
        questions.append({
            "qid": f"trr_{qid:04d}",
            "type": "trr",
            "target": constraint.target,
            "ref1": constraint.ref1,
            "ref2": constraint.ref2,
            "gt_hour": constraint.hour,
            "gt_quadrant": constraint.quadrant,
            "gt_angle_deg": round(constraint.angle_deg, 2),
        })

    return questions


def enumerate_fdr(
    objects: Dict[str, Dict],
    tau: float = 0.10,
) -> List[dict]:
    """
    枚举所有 FDR（全距离排序）问题。

    对于 N 个物体，FDR 数量 = N（每个锚点物体一个）。

    返回问题字典列表。
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
    将问题列表按批次分割。

    返回包含 batch_id 和 questions 的批次字典列表。
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
    计算 N 个物体的预期问题数量。

    QRR 包含 disjoint 对比对和 shared_anchor 比较两种变体。
    TRR：P(N,3) = N*(N-1)*(N-2)
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
