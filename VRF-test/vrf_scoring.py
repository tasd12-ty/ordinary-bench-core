"""VRF 评分模块。"""

from typing import Any, Dict, List


def score_vrf(predicted: Any, gt_answer: bool) -> bool:
    """Normalize predicted to bool and compare with GT."""
    if isinstance(predicted, bool):
        return predicted == gt_answer
    if isinstance(predicted, str):
        normalized = predicted.strip().lower()
        if normalized in ("true", "yes", "correct", "1"):
            return gt_answer is True
        if normalized in ("false", "no", "incorrect", "0"):
            return gt_answer is False
    return False


def score_batch_scene(predictions: Dict[str, Any], questions: List[dict]) -> dict:
    """Score all VRF predictions for a single scene."""
    vrf_correct = vrf_total = 0
    vrf_true_correct = vrf_true_total = 0
    vrf_false_correct = vrf_false_total = 0
    missing = 0
    per_question = []

    for q in questions:
        qid = q["qid"]
        pred = predictions.get(qid)
        gt = q["gt_answer"]

        if pred is None:
            missing += 1
            per_question.append({"qid": qid, "predicted": None, "gt": gt, "correct": False})
            vrf_total += 1
            if gt:
                vrf_true_total += 1
            else:
                vrf_false_total += 1
            continue

        correct = score_vrf(pred, gt)
        vrf_total += 1
        if correct:
            vrf_correct += 1
        if gt:
            vrf_true_total += 1
            if correct:
                vrf_true_correct += 1
        else:
            vrf_false_total += 1
            if correct:
                vrf_false_correct += 1

        per_question.append({
            "qid": qid,
            "predicted": pred,
            "gt": gt,
            "correct": correct,
        })

    def _acc(c, t):
        return round(c / t, 4) if t > 0 else 0.0

    return {
        "vrf_correct": vrf_correct,
        "vrf_total": vrf_total,
        "vrf_accuracy": _acc(vrf_correct, vrf_total),
        "vrf_true_correct": vrf_true_correct,
        "vrf_true_total": vrf_true_total,
        "vrf_true_accuracy": _acc(vrf_true_correct, vrf_true_total),
        "vrf_false_correct": vrf_false_correct,
        "vrf_false_total": vrf_false_total,
        "vrf_false_accuracy": _acc(vrf_false_correct, vrf_false_total),
        "missing": missing,
        "per_question": per_question,
    }


_SUM_KEYS = [
    "vrf_correct", "vrf_total",
    "vrf_true_correct", "vrf_true_total",
    "vrf_false_correct", "vrf_false_total",
    "missing",
]


def aggregate_results(scene_results: List[dict]) -> dict:
    """Aggregate VRF scores across all scenes, grouped by split."""
    total = {k: 0 for k in _SUM_KEYS}
    by_split: Dict[str, Dict[str, int]] = {}

    for r in scene_results:
        scene_id = r["scene_id"]
        split = scene_id.rsplit("_", 1)[0]
        s = r["scores"]

        for k in _SUM_KEYS:
            total[k] += s.get(k, 0)

        if split not in by_split:
            by_split[split] = {k: 0 for k in _SUM_KEYS}
        for k in _SUM_KEYS:
            by_split[split][k] += s.get(k, 0)

    def _acc(c, t):
        return round(c / t, 4) if t > 0 else 0.0

    def _summary(counts):
        return {
            "vrf_accuracy": _acc(counts["vrf_correct"], counts["vrf_total"]),
            "vrf_true_accuracy": _acc(counts["vrf_true_correct"], counts["vrf_true_total"]),
            "vrf_false_accuracy": _acc(counts["vrf_false_correct"], counts["vrf_false_total"]),
            **counts,
        }

    return {
        "overall": _summary(total),
        "by_split": {
            split: _summary(counts)
            for split, counts in sorted(by_split.items())
        },
    }
