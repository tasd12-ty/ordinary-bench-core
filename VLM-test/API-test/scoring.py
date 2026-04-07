"""
Batch 评分模块：比较 VLM 预测与 GT。

QRR：比较器精确匹配。
TRR：三粒度评分（hour 精确 / quadrant 象限 / adjacent ±1h）。
FDR：四粒度评分（exact / kendall τ / pairwise / top-k）。
"""

from typing import Dict, List, Any
from dsl.comparators import Comparator
from dsl.predicates import hour_to_quadrant


# ── 单题评分 ──

def score_qrr(predicted: str, gt: str) -> bool:
    """QRR 精确匹配：比较器是否一致。"""
    try:
        return Comparator.from_string(predicted) == Comparator.from_string(gt)
    except ValueError:
        return False


def score_trr_hour(predicted: int, gt_hour: int) -> bool:
    """TRR hour 精确匹配。"""
    return predicted == gt_hour


def score_trr_quadrant(predicted: int, gt_quadrant: int) -> bool:
    """TRR 象限匹配：预测的 hour 是否落在 GT 所在象限。"""
    try:
        predicted = int(predicted)
        if not 1 <= predicted <= 12:
            return False
        return hour_to_quadrant(predicted) == gt_quadrant
    except (ValueError, TypeError):
        return False


def score_trr_adjacent(predicted: int, gt_hour: int) -> bool:
    """TRR 相邻匹配：预测 hour 与 GT hour 差 ≤ 1（12↔1 循环）。"""
    try:
        predicted = int(predicted)
        if not 1 <= predicted <= 12:
            return False
        diff = abs(predicted - gt_hour)
        return diff <= 1 or diff >= 11
    except (ValueError, TypeError):
        return False


# ── FDR 单题评分 ──

def score_fdr_exact(
    predicted: List[str],
    gt_ranking: List[str],
    gt_tie_groups: List[List[str]],
) -> bool:
    """FDR 精确匹配（尊重并列组内的任意排列）。"""
    if len(predicted) != len(gt_ranking):
        return False
    idx = 0
    for group in gt_tie_groups:
        group_set = set(group)
        predicted_slice = set(predicted[idx:idx + len(group)])
        if predicted_slice != group_set:
            return False
        idx += len(group)
    return True


def score_fdr_kendall(
    predicted: List[str],
    gt_ranking: List[str],
    gt_tie_groups: List[List[str]],
) -> float:
    """Kendall τ 排序相关系数（跳过并列对）。返回 [-1, 1]，1.0 = 完全一致。"""
    if not predicted or len(predicted) != len(gt_ranking):
        return 0.0
    gt_rank = {obj: i for i, obj in enumerate(gt_ranking)}
    tie_set = {}
    for gi, group in enumerate(gt_tie_groups):
        for obj in group:
            tie_set[obj] = gi
    common = [obj for obj in predicted if obj in gt_rank]
    if len(common) < 2:
        return 0.0
    concordant = 0
    discordant = 0
    for i in range(len(common)):
        for j in range(i + 1, len(common)):
            a, b = common[i], common[j]
            ga, gb = tie_set.get(a), tie_set.get(b)
            if ga is not None and ga == gb:
                continue  # tied: neutral
            if gt_rank[a] < gt_rank[b]:
                concordant += 1
            elif gt_rank[a] > gt_rank[b]:
                discordant += 1
    scored_pairs = concordant + discordant
    if scored_pairs == 0:
        return 0.0
    return (concordant - discordant) / scored_pairs


def score_fdr_pairwise(
    predicted: List[str],
    gt_ranking: List[str],
    gt_tie_groups: List[List[str]],
) -> float:
    """成对序关系正确率。并列组内任意顺序均算正确。返回 [0, 1]。"""
    if not predicted:
        return 0.0
    pred_rank = {obj: i for i, obj in enumerate(predicted)}
    tie_set = {}
    for gi, group in enumerate(gt_tie_groups):
        for obj in group:
            tie_set[obj] = gi
    correct = 0
    total = 0
    for i in range(len(gt_ranking)):
        for j in range(i + 1, len(gt_ranking)):
            a, b = gt_ranking[i], gt_ranking[j]
            total += 1
            if a not in pred_rank or b not in pred_rank:
                continue
            ga, gb = tie_set.get(a), tie_set.get(b)
            if ga is not None and ga == gb:
                correct += 1
            elif pred_rank[a] < pred_rank[b]:
                correct += 1
    return correct / total if total > 0 else 0.0


def score_fdr_topk(predicted: List[str], gt_ranking: List[str], k: int) -> float:
    """Top-k 最近物体集合正确率。返回 [0, 1]。"""
    if not predicted or k <= 0:
        return 0.0
    k = min(k, len(gt_ranking), len(predicted))
    gt_topk = set(gt_ranking[:k])
    pred_topk = set(predicted[:k])
    return len(gt_topk & pred_topk) / k


# ── 场景评分 ──

def _normalize_na(answer) -> str | None:
    """Normalize N/A variants to canonical 'N/A', return None if not N/A."""
    if isinstance(answer, str) and answer.strip().upper() in ("N/A", "NA"):
        return "N/A"
    return None


def score_batch_scene(predictions: Dict[str, Any], questions: List[dict], *, ablation: bool = False) -> dict:
    """
    对单个场景的所有 batch 预测评分。

    返回 QRR/TRR/FDR 各项统计 + 逐题详情。
    当 ablation=True 时，额外统计 refusal/hallucination 指标。
    """
    qrr_correct = qrr_total = 0
    qrr_disjoint_correct = qrr_disjoint_total = 0
    qrr_shared_anchor_correct = qrr_shared_anchor_total = 0
    trr_hour_correct = trr_quad_correct = trr_adj_correct = trr_total = 0
    fdr_exact_correct = fdr_total = 0
    fdr_kendall_sum = fdr_pairwise_sum = fdr_top1_sum = 0.0
    missing = 0

    # ablation-specific counters
    answerable_correct = answerable_total = 0
    refusal_correct = refusal_hallucinated = refusal_total = 0

    per_question = []

    for q in questions:
        qid = q["qid"]
        pred = predictions.get(qid)
        is_answerable = q.get("answerable", True)

        # In ablation mode, handle answerable/refusal scoring for QRR
        if ablation and q["type"] == "qrr":
            variant = q.get("variant", "disjoint")
            normalized_na = _normalize_na(pred) if pred is not None else None

            if pred is None:
                missing += 1
                if is_answerable:
                    answerable_total += 1
                    qrr_total += 1
                    if variant == "shared_anchor":
                        qrr_shared_anchor_total += 1
                    else:
                        qrr_disjoint_total += 1
                else:
                    refusal_total += 1
                per_question.append({
                    "qid": qid, "type": "qrr", "variant": variant,
                    "anchor": q.get("anchor"),
                    "predicted": None, "answerable": is_answerable,
                    "status": "missing",
                })
            elif is_answerable:
                answerable_total += 1
                qrr_total += 1
                if variant == "shared_anchor":
                    qrr_shared_anchor_total += 1
                else:
                    qrr_disjoint_total += 1
                # If VLM answered N/A for an answerable question, treat as wrong
                if normalized_na:
                    correct = False
                else:
                    correct = score_qrr(str(pred), q["gt_comparator"])
                if correct:
                    answerable_correct += 1
                    qrr_correct += 1
                    if variant == "shared_anchor":
                        qrr_shared_anchor_correct += 1
                    else:
                        qrr_disjoint_correct += 1
                per_question.append({
                    "qid": qid, "type": "qrr", "variant": variant,
                    "anchor": q.get("anchor"),
                    "predicted": str(pred), "gt": q["gt_comparator"],
                    "answerable": True, "correct": correct,
                    "status": "correct" if correct else "wrong",
                })
            else:
                # Not answerable — check if VLM correctly refused
                refusal_total += 1
                if normalized_na:
                    refusal_correct += 1
                    per_question.append({
                        "qid": qid, "type": "qrr", "variant": variant,
                        "predicted": "N/A", "answerable": False,
                        "missing_objects": q.get("missing_objects", []),
                        "status": "correct_refusal",
                    })
                else:
                    refusal_hallucinated += 1
                    per_question.append({
                        "qid": qid, "type": "qrr", "variant": variant,
                        "predicted": str(pred), "answerable": False,
                        "missing_objects": q.get("missing_objects", []),
                        "status": "hallucinated",
                        "hallucinated_answer": str(pred),
                    })
            continue

        # ── Non-ablation path (original logic) ──

        if pred is None:
            missing += 1
            pq = {"qid": qid, "type": q["type"], "predicted": None, "correct": False}
            if q["type"] == "qrr":
                pq["variant"] = q.get("variant", "disjoint")
                if "anchor" in q:
                    pq["anchor"] = q["anchor"]
            per_question.append(pq)
            if q["type"] == "qrr":
                qrr_total += 1
                variant = q.get("variant", "disjoint")
                if variant == "shared_anchor":
                    qrr_shared_anchor_total += 1
                else:
                    qrr_disjoint_total += 1
            elif q["type"] == "trr":
                trr_total += 1
            elif q["type"] == "fdr":
                fdr_total += 1
            continue

        if q["type"] == "qrr":
            qrr_total += 1
            variant = q.get("variant", "disjoint")
            if variant == "shared_anchor":
                qrr_shared_anchor_total += 1
            else:
                qrr_disjoint_total += 1
            correct = score_qrr(str(pred), q["gt_comparator"])
            if correct:
                qrr_correct += 1
                if variant == "shared_anchor":
                    qrr_shared_anchor_correct += 1
                else:
                    qrr_disjoint_correct += 1
            per_question.append({
                "qid": qid, "type": "qrr",
                "variant": variant,
                "anchor": q.get("anchor"),
                "predicted": str(pred), "gt": q["gt_comparator"],
                "correct": correct,
            })
        elif q["type"] == "trr":
            trr_total += 1
            try:
                pred_int = int(pred)
            except (ValueError, TypeError):
                pred_int = -1
            h = score_trr_hour(pred_int, q["gt_hour"])
            qu = score_trr_quadrant(pred_int, q["gt_quadrant"])
            a = score_trr_adjacent(pred_int, q["gt_hour"])
            if h:
                trr_hour_correct += 1
            if qu:
                trr_quad_correct += 1
            if a:
                trr_adj_correct += 1
            per_question.append({
                "qid": qid, "type": "trr",
                "predicted": pred_int, "gt_hour": q["gt_hour"],
                "hour_correct": h, "quadrant_correct": qu, "adjacent_correct": a,
            })
        elif q["type"] == "fdr":
            fdr_total += 1
            pred_list = pred if isinstance(pred, list) else []
            gt_ranking = q["gt_ranking"]
            gt_tie_groups = q.get("gt_tie_groups", [[x] for x in gt_ranking])

            exact = score_fdr_exact(pred_list, gt_ranking, gt_tie_groups)
            kendall = score_fdr_kendall(pred_list, gt_ranking, gt_tie_groups)
            pairwise = score_fdr_pairwise(pred_list, gt_ranking, gt_tie_groups)
            top1 = score_fdr_topk(pred_list, gt_ranking, 1)

            if exact:
                fdr_exact_correct += 1
            fdr_kendall_sum += kendall
            fdr_pairwise_sum += pairwise
            fdr_top1_sum += top1

            per_question.append({
                "qid": qid, "type": "fdr",
                "predicted": pred_list, "gt_ranking": gt_ranking,
                "exact_correct": exact,
                "kendall_tau": round(kendall, 4),
                "pairwise_accuracy": round(pairwise, 4),
                "top1_correct": top1 == 1.0,
            })

    result = {
        "qrr_correct": qrr_correct, "qrr_total": qrr_total,
        "qrr_disjoint_correct": qrr_disjoint_correct,
        "qrr_disjoint_total": qrr_disjoint_total,
        "qrr_shared_anchor_correct": qrr_shared_anchor_correct,
        "qrr_shared_anchor_total": qrr_shared_anchor_total,
        "trr_hour_correct": trr_hour_correct,
        "trr_quadrant_correct": trr_quad_correct,
        "trr_adjacent_correct": trr_adj_correct,
        "trr_total": trr_total,
        "fdr_exact_correct": fdr_exact_correct,
        "fdr_total": fdr_total,
        "fdr_kendall_mean": round(fdr_kendall_sum / fdr_total, 4) if fdr_total > 0 else 0.0,
        "fdr_pairwise_mean": round(fdr_pairwise_sum / fdr_total, 4) if fdr_total > 0 else 0.0,
        "fdr_top1_mean": round(fdr_top1_sum / fdr_total, 4) if fdr_total > 0 else 0.0,
        "missing": missing,
        "per_question": per_question,
    }

    if ablation:
        result.update({
            "answerable_correct": answerable_correct,
            "answerable_total": answerable_total,
            "answerable_acc": round(answerable_correct / answerable_total, 4) if answerable_total else 0.0,
            "refusal_correct": refusal_correct,
            "refusal_hallucinated": refusal_hallucinated,
            "refusal_total": refusal_total,
            "refusal_rate": round(refusal_correct / refusal_total, 4) if refusal_total else 0.0,
            "hallucination_rate": round(refusal_hallucinated / refusal_total, 4) if refusal_total else 0.0,
        })

    return result

# ── 结果聚合 ──

# ── 结果聚合 ──

# Keys that are summed across scenes (integer counters)
_SUM_KEYS = [
    "qrr_correct", "qrr_total",
    "qrr_disjoint_correct", "qrr_disjoint_total",
    "qrr_shared_anchor_correct", "qrr_shared_anchor_total",
    "trr_hour_correct", "trr_quadrant_correct", "trr_adjacent_correct", "trr_total",
    "fdr_exact_correct", "fdr_total",
    "missing",
]

# Additional keys summed in ablation mode
_ABLATION_SUM_KEYS = [
    "answerable_correct", "answerable_total",
    "refusal_correct", "refusal_hallucinated", "refusal_total",
]

# Keys that are averaged (weighted by fdr_total)
_FDR_MEAN_KEYS = ["fdr_kendall_mean", "fdr_pairwise_mean", "fdr_top1_mean"]


def aggregate_batch_results(scene_results: List[dict]) -> dict:
    """汇总所有场景的 batch 评分，按 split 分组统计。"""
    # Detect ablation mode from the first scene's scores
    has_ablation = any("answerable_total" in r["scores"] for r in scene_results)
    sum_keys = _SUM_KEYS + (_ABLATION_SUM_KEYS if has_ablation else [])

    total = {k: 0 for k in sum_keys}
    fdr_weighted = {k: 0.0 for k in _FDR_MEAN_KEYS}
    by_split = {}
    by_split_fdr = {}

    for r in scene_results:
        scene_id = r["scene_id"]
        split = scene_id.rsplit("_", 1)[0]
        s = r["scores"]

        for k in sum_keys:
            total[k] += s.get(k, 0)
        fdr_n = s["fdr_total"]
        for k in _FDR_MEAN_KEYS:
            fdr_weighted[k] += s[k] * fdr_n

        if split not in by_split:
            by_split[split] = {k: 0 for k in sum_keys}
            by_split_fdr[split] = {k: 0.0 for k in _FDR_MEAN_KEYS}
        for k in sum_keys:
            by_split[split][k] += s.get(k, 0)
        for k in _FDR_MEAN_KEYS:
            by_split_fdr[split][k] += s[k] * fdr_n

    def _acc(correct, total_count):
        return round(correct / total_count, 4) if total_count > 0 else 0.0

    def _fdr_means(weighted, fdr_total):
        return {k: round(weighted[k] / fdr_total, 4) if fdr_total > 0 else 0.0 for k in _FDR_MEAN_KEYS}

    def _ablation_rates(counts):
        return {
            "answerable_acc": _acc(counts["answerable_correct"], counts["answerable_total"]),
            "refusal_rate": _acc(counts["refusal_correct"], counts["refusal_total"]),
            "hallucination_rate": _acc(counts["refusal_hallucinated"], counts["refusal_total"]),
        }

    summary = {
        "overall": {
            "qrr_accuracy": _acc(total["qrr_correct"], total["qrr_total"]),
            "qrr_disjoint_accuracy": _acc(total["qrr_disjoint_correct"], total["qrr_disjoint_total"]),
            "qrr_shared_anchor_accuracy": _acc(total["qrr_shared_anchor_correct"], total["qrr_shared_anchor_total"]),
            "trr_hour_accuracy": _acc(total["trr_hour_correct"], total["trr_total"]),
            "trr_quadrant_accuracy": _acc(total["trr_quadrant_correct"], total["trr_total"]),
            "trr_adjacent_accuracy": _acc(total["trr_adjacent_correct"], total["trr_total"]),
            "fdr_exact_accuracy": _acc(total["fdr_exact_correct"], total["fdr_total"]),
            **_fdr_means(fdr_weighted, total["fdr_total"]),
            **((_ablation_rates(total)) if has_ablation else {}),
            **total,
        },
        "by_split": {},
    }
    for split, counts in sorted(by_split.items()):
        summary["by_split"][split] = {
            "qrr_accuracy": _acc(counts["qrr_correct"], counts["qrr_total"]),
            "qrr_disjoint_accuracy": _acc(counts["qrr_disjoint_correct"], counts["qrr_disjoint_total"]),
            "qrr_shared_anchor_accuracy": _acc(counts["qrr_shared_anchor_correct"], counts["qrr_shared_anchor_total"]),
            "trr_hour_accuracy": _acc(counts["trr_hour_correct"], counts["trr_total"]),
            "trr_quadrant_accuracy": _acc(counts["trr_quadrant_correct"], counts["trr_total"]),
            "trr_adjacent_accuracy": _acc(counts["trr_adjacent_correct"], counts["trr_total"]),
            "fdr_exact_accuracy": _acc(counts["fdr_exact_correct"], counts["fdr_total"]),
            **_fdr_means(by_split_fdr[split], counts["fdr_total"]),
            **((_ablation_rates(counts)) if has_ablation else {}),
            **counts,
        }

    return summary
