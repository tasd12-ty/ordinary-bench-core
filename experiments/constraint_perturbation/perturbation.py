"""约束扰动：consistent_flip（无环翻转）。

逐个翻转 QRR 约束的 <↔>，每次检查 DAG 无环，有环则回退。
保证结果无逻辑矛盾，模拟"错但自洽"的模型回答。
"""

import random
import sys
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "VLM-test"))

from dsl.predicates import MetricType, METRIC_FUNCTIONS
from dsl.comparators import compare
from reconstruct.constraints import QRREntry, build_distance_poset


def _check_cycle(constraints: List[dict]) -> bool:
    """快速检查 QRR 约束列表是否存在 DAG 环。"""
    entries = [
        QRREntry(
            pair1=tuple(c["pair1"]),
            pair2=tuple(c["pair2"]),
            comparator=c["comparator"],
            weight=c.get("weight", 1.0),
            variant=c.get("variant", "disjoint"),
            anchor=c.get("anchor"),
        )
        for c in constraints
    ]
    poset = build_distance_poset(entries)
    return poset.has_cycle


def consistent_flip_qrr(
    constraints: List[dict],
    fraction: float,
    rng: random.Random,
) -> Tuple[List[dict], int]:
    """对 QRR strict 约束逐个翻转 <↔>，保证无环。

    Args:
        constraints: QRR 约束 dict 列表
        fraction: 目标翻转比例 (相对于 strict 约束数)
        rng: 随机数生成器

    Returns:
        (perturbed_constraints, n_actually_flipped)
    """
    result = deepcopy(constraints)
    perturbable = [i for i, c in enumerate(result) if c["comparator"] != "~="]
    n_target = int(len(perturbable) * fraction)

    if n_target == 0 or not perturbable:
        return result, 0

    candidates = list(perturbable)
    rng.shuffle(candidates)
    n_flipped = 0

    for idx in candidates:
        if n_flipped >= n_target:
            break
        old_cmp = result[idx]["comparator"]
        new_cmp = ">" if old_cmp == "<" else "<"
        result[idx]["comparator"] = new_cmp

        if _check_cycle(result):
            result[idx]["comparator"] = old_cmp
        else:
            n_flipped += 1

    return result, n_flipped


def compute_gt_satisfaction(
    perturbed_qrr: List[dict],
    objects: Dict,
    tau: float = 0.10,
) -> float:
    """用 GT 3D 坐标检查扰动后约束的满足比例。"""
    if not perturbed_qrr:
        return 1.0
    metric_func = METRIC_FUNCTIONS[MetricType.DIST_3D]
    satisfied = 0
    for c in perturbed_qrr:
        m1 = metric_func(objects[c["pair1"][0]], objects[c["pair1"][1]])
        m2 = metric_func(objects[c["pair2"][0]], objects[c["pair2"][1]])
        gt_cmp = compare(m1, m2, tau)
        if str(gt_cmp) == c["comparator"]:
            satisfied += 1
    return satisfied / len(perturbed_qrr)
