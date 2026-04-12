"""VRF (Verification) 问题生成：固定题量的复合空间关系验证题。

每道题包含多条 sub-claims（距离比较 + 钟面方向 + 距离排序），
VLM 判断整组是否全部正确。TRUE/FALSE 各半。
"""

import hashlib
import random
from itertools import combinations, permutations
from typing import Dict, List

import sys
from pathlib import Path

_VLM_DIR = Path(__file__).resolve().parent.parent / "VLM-test"
if str(_VLM_DIR) not in sys.path:
    sys.path.insert(0, str(_VLM_DIR))

from dsl.predicates import (
    MetricType, compute_qrr, compute_trr, compute_fdr,
    _is_boundary, METRIC_FUNCTIONS,
)

MAX_REUSE = 3


def _scene_seed(scene_id: str) -> int:
    return int(hashlib.sha256(scene_id.encode()).hexdigest()[:16], 16)


# ── Fact pool builders ──────────────────────────────────────────────


def _build_qrr_facts(objects: Dict, tau: float) -> List[dict]:
    obj_ids = sorted(objects.keys())
    metric = MetricType.DIST_3D
    metric_func = METRIC_FUNCTIONS[metric]
    facts = []

    # Disjoint pairs
    pairs = list(combinations(obj_ids, 2))
    for i, pair1 in enumerate(pairs):
        for pair2 in pairs[i + 1 :]:
            if set(pair1) & set(pair2):
                continue
            m1 = metric_func(objects[pair1[0]], objects[pair1[1]])
            m2 = metric_func(objects[pair2[0]], objects[pair2[1]])
            if _is_boundary(m1, m2, tau):
                continue
            c = compute_qrr(objects, pair1, pair2, metric, tau, variant="disjoint")
            facts.append(
                {
                    "source": "qrr",
                    "variant": "disjoint",
                    "pair1": list(c.pair1),
                    "pair2": list(c.pair2),
                    "metric": str(metric),
                    "gt_comparator": str(c.comparator),
                }
            )

    # Shared anchor
    for anchor in obj_ids:
        others = [oid for oid in obj_ids if oid != anchor]
        for obj_a, obj_b in combinations(others, 2):
            pair1, pair2 = (anchor, obj_a), (anchor, obj_b)
            m1 = metric_func(objects[pair1[0]], objects[pair1[1]])
            m2 = metric_func(objects[pair2[0]], objects[pair2[1]])
            if _is_boundary(m1, m2, tau):
                continue
            c = compute_qrr(
                objects, pair1, pair2, metric, tau,
                variant="shared_anchor", anchor=anchor,
            )
            facts.append(
                {
                    "source": "qrr",
                    "variant": "shared_anchor",
                    "anchor": anchor,
                    "pair1": list(c.pair1),
                    "pair2": list(c.pair2),
                    "metric": str(metric),
                    "gt_comparator": str(c.comparator),
                }
            )
    return facts


def _build_trr_facts(objects: Dict) -> List[dict]:
    obj_ids = sorted(objects.keys())
    facts = []
    for target, ref1, ref2 in permutations(obj_ids, 3):
        c = compute_trr(objects, target, ref1, ref2, use_3d=True)
        facts.append(
            {
                "source": "trr",
                "target": c.target,
                "ref1": c.ref1,
                "ref2": c.ref2,
                "gt_hour": c.hour,
                "gt_angle_deg": round(c.angle_deg, 2),
            }
        )
    return facts


def _build_fdr_facts(objects: Dict, tau: float) -> List[dict]:
    """Nearest + farthest per anchor → 2N facts."""
    obj_ids = sorted(objects.keys())
    facts = []
    for anchor in obj_ids:
        c = compute_fdr(objects, anchor, tau)
        if not c.ranking:
            continue
        facts.append(
            {
                "source": "fdr",
                "fdr_type": "nearest",
                "anchor": anchor,
                "gt_object": c.ranking[0],
                "gt_ranking": c.ranking,
                "gt_distances": [round(d, 6) for d in c.distances],
            }
        )
        facts.append(
            {
                "source": "fdr",
                "fdr_type": "farthest",
                "anchor": anchor,
                "gt_object": c.ranking[-1],
                "gt_ranking": c.ranking,
                "gt_distances": [round(d, 6) for d in c.distances],
            }
        )
    return facts


# ── Corruption strategies ───────────────────────────────────────────


def _corrupt_qrr(fact: dict, rng: random.Random) -> str:
    """Return a wrong comparator."""
    gt = fact["gt_comparator"]
    if gt == "<":
        return ">"
    elif gt == ">":
        return "<"
    return rng.choice(["<", ">"])


def _corrupt_trr(fact: dict, rng: random.Random) -> int:
    """Return a wrong hour (offset ≥ 3)."""
    offset = rng.choice([3, 4, 5, 6, 7, 8, 9])
    return ((fact["gt_hour"] - 1 + offset) % 12) + 1


def _corrupt_fdr(fact: dict, rng: random.Random) -> str:
    """Return a wrong object for nearest/farthest."""
    ranking = fact["gt_ranking"]
    gt_obj = fact["gt_object"]
    if fact["fdr_type"] == "nearest":
        candidates = ranking[2:] if len(ranking) > 2 else ranking[1:]
    else:
        candidates = ranking[:-2] if len(ranking) > 2 else ranking[:-1]
    candidates = [c for c in candidates if c != gt_obj]
    if not candidates:
        # Broader fallback: any ranking object that differs from GT
        candidates = [c for c in ranking if c != gt_obj]
    return rng.choice(candidates) if candidates else gt_obj


# ── Claim text formatters ──────────────────────────────────────────


_COMP_TEXT = {"<": "less than", "~=": "approximately equal to", ">": "greater than"}


def _format_qrr_claim(fact: dict, comp: str) -> str:
    p1a, p1b = fact["pair1"]
    p2a, p2b = fact["pair2"]
    ct = _COMP_TEXT.get(comp, comp)
    if fact.get("variant") == "shared_anchor" and fact.get("anchor"):
        anchor = fact["anchor"]
        c1 = p1b if p1a == anchor else p1a
        c2 = p2b if p2a == anchor else p2a
        return (
            f"The distance from {anchor} to {c1} is {ct} "
            f"the distance from {anchor} to {c2}."
        )
    return (
        f"The distance between {p1a} and {p1b} is {ct} "
        f"the distance between {p2a} and {p2b}."
    )


def _format_trr_claim(fact: dict, hour: int) -> str:
    return (
        f"Standing at {fact['ref1']}, facing {fact['ref2']}, "
        f"{fact['target']} is at {hour} o'clock."
    )


def _format_fdr_claim(fact: dict, obj: str) -> str:
    if fact["fdr_type"] == "nearest":
        return f"The nearest object to {fact['anchor']} is {obj}."
    return f"The farthest object from {fact['anchor']} is {obj}."


# ── Claim assembly helpers ─────────────────────────────────────────


def _make_true_claim(source: str, fact: dict) -> dict:
    if source == "qrr":
        comp = fact["gt_comparator"]
        return {
            "source": "qrr",
            "claim_text": _format_qrr_claim(fact, comp),
            "is_true": True,
            "detail": {
                "pair1": fact["pair1"],
                "pair2": fact["pair2"],
                "gt_comparator": comp,
                "claimed_comparator": comp,
            },
        }
    elif source == "trr":
        h = fact["gt_hour"]
        return {
            "source": "trr",
            "claim_text": _format_trr_claim(fact, h),
            "is_true": True,
            "detail": {
                "target": fact["target"],
                "ref1": fact["ref1"],
                "ref2": fact["ref2"],
                "gt_hour": h,
                "claimed_hour": h,
            },
        }
    else:  # fdr
        obj = fact["gt_object"]
        return {
            "source": "fdr",
            "claim_text": _format_fdr_claim(fact, obj),
            "is_true": True,
            "detail": {
                "anchor": fact["anchor"],
                "fdr_type": fact["fdr_type"],
                "gt_object": obj,
                "claimed_object": obj,
            },
        }


def _make_false_claim(source: str, fact: dict, rng: random.Random) -> dict:
    if source == "qrr":
        claimed = _corrupt_qrr(fact, rng)
        return {
            "source": "qrr",
            "claim_text": _format_qrr_claim(fact, claimed),
            "is_true": False,
            "detail": {
                "pair1": fact["pair1"],
                "pair2": fact["pair2"],
                "gt_comparator": fact["gt_comparator"],
                "claimed_comparator": claimed,
            },
        }
    elif source == "trr":
        claimed = _corrupt_trr(fact, rng)
        return {
            "source": "trr",
            "claim_text": _format_trr_claim(fact, claimed),
            "is_true": False,
            "detail": {
                "target": fact["target"],
                "ref1": fact["ref1"],
                "ref2": fact["ref2"],
                "gt_hour": fact["gt_hour"],
                "claimed_hour": claimed,
            },
        }
    else:  # fdr
        claimed = _corrupt_fdr(fact, rng)
        return {
            "source": "fdr",
            "claim_text": _format_fdr_claim(fact, claimed),
            "is_true": False,
            "detail": {
                "anchor": fact["anchor"],
                "fdr_type": fact["fdr_type"],
                "gt_object": fact["gt_object"],
                "claimed_object": claimed,
            },
        }


# ── Main entry ─────────────────────────────────────────────────────


def enumerate_vrf(
    objects: Dict[str, Dict],
    K: int = 20,
    claims_per_q: int = 3,
    tau: float = 0.10,
    scene_id: str = "",
) -> List[dict]:
    """Generate K composite verification questions, TRUE/FALSE balanced."""
    rng = random.Random(_scene_seed(scene_id))
    K = K if K % 2 == 0 else K - 1
    n_true = K // 2

    # Build fact pools
    pools = {
        "qrr": _build_qrr_facts(objects, tau),
        "trr": _build_trr_facts(objects),
        "fdr": _build_fdr_facts(objects, tau),
    }
    use_counts: Dict[str, Dict[int, int]] = {s: {} for s in pools}

    # Preferred source order per claim slot (rotated for diversity)
    preferred = ["qrr", "trr", "fdr"]

    def _pick(source: str) -> dict | None:
        pool = pools[source]
        avail = [i for i in range(len(pool)) if use_counts[source].get(i, 0) < MAX_REUSE]
        if not avail:
            return None
        idx = rng.choice(avail)
        use_counts[source][idx] = use_counts[source].get(idx, 0) + 1
        return pool[idx]

    def _pick_any_available() -> tuple[str, dict] | None:
        for src in rng.sample(list(pools.keys()), len(pools)):
            fact = _pick(src)
            if fact is not None:
                return src, fact
        return None

    questions: List[dict] = []

    for q_idx in range(K):
        is_true = q_idx < n_true

        # Select facts: try one from each preferred source, then fill
        selected: List[tuple[str, dict]] = []
        rng.shuffle(preferred)
        for src in preferred:
            if len(selected) >= claims_per_q:
                break
            fact = _pick(src)
            if fact is not None:
                selected.append((src, fact))

        while len(selected) < claims_per_q:
            result = _pick_any_available()
            if result is None:
                break
            selected.append(result)

        if not selected:
            continue

        # Build claim dicts
        corrupted_index = None
        if not is_true:
            corrupted_index = rng.randrange(len(selected))

        claim_dicts = []
        for ci, (source, fact) in enumerate(selected):
            if not is_true and ci == corrupted_index:
                claim_dicts.append(_make_false_claim(source, fact, rng))
            else:
                claim_dicts.append(_make_true_claim(source, fact))

        questions.append(
            {
                "qid": "",  # assigned after shuffle
                "type": "vrf",
                "gt_answer": is_true,
                "corrupted_index": corrupted_index,
                "claims": claim_dicts,
            }
        )

    # Shuffle so TRUE/FALSE are interleaved
    rng.shuffle(questions)
    for i, q in enumerate(questions):
        q["qid"] = f"vrf_{i + 1:04d}"

    return questions
