"""
三维 Batch 评分模块：比较 VLM 预测与 GT。

在 2D 版本基础上新增三维 TRR 评分（仰角分级匹配）。

QRR：比较器精确匹配。
TRR：五粒度评分（hour 精确 / quadrant 象限 / adjacent ±1h / elevation 精确 / elevation_adjacent ±1 级）。
FDR：四粒度评分（exact / kendall τ / pairwise / top-k）。
"""

from typing import Dict, List, Any
from dsl.comparators import Comparator
from dsl.predicates import hour_to_quadrant


# ── 仰角分级定义（从高到低排列） ──

ELEVATION_BANDS = ["above", "slightly_above", "level", "slightly_below", "below"]


# ── QRR 单题评分 ──

def score_qrr(predicted: str, gt: str) -> bool:
    """QRR 精确匹配：比较器是否一致。"""
    try:
        return Comparator.from_string(predicted) == Comparator.from_string(gt)
    except ValueError:
        return False


# ── TRR 单题评分（2D 维度） ──

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
    """TRR 相邻匹配：预测 hour 与 GT hour 差 <= 1（12<->1 循环）。"""
    try:
        predicted = int(predicted)
        if not 1 <= predicted <= 12:
            return False
        diff = abs(predicted - gt_hour)
        return diff <= 1 or diff >= 11
    except (ValueError, TypeError):
        return False


# ── TRR 三维仰角评分 ──

def score_trr_elevation(predicted_elevation: str, gt_elevation_band: str) -> bool:
    """TRR 仰角精确匹配：预测的仰角分级与 GT 分级是否一致。"""
    if predicted_elevation not in ELEVATION_BANDS:
        return False
    return predicted_elevation == gt_elevation_band


def score_trr_elevation_adjacent(predicted_elevation: str, gt_elevation_band: str) -> bool:
    """TRR 仰角相邻匹配：预测分级与 GT 分级相差 <= 1 级。"""
    if predicted_elevation not in ELEVATION_BANDS:
        return False
    if gt_elevation_band not in ELEVATION_BANDS:
        return False
    pred_idx = ELEVATION_BANDS.index(predicted_elevation)
    gt_idx = ELEVATION_BANDS.index(gt_elevation_band)
    return abs(pred_idx - gt_idx) <= 1


def score_trr_3d(
    predicted: Any,
    gt_hour: int,
    gt_quadrant: int,
    gt_elevation_band: str,
) -> dict:
    """三维 TRR 综合评分，返回所有子维度的评分结果。

    predicted 可以是：
      - dict: {"hour": int, "elevation": str}（三维完整回答）
      - int: 仅水平方向（仰角视为缺失）

    返回 dict 包含：
      hour_correct, quadrant_correct, adjacent_correct,
      elevation_correct, elevation_adjacent, full_3d_correct
    """
    # 解析预测值
    if isinstance(predicted, dict):
        pred_hour = predicted.get("hour", -1)
        pred_elev = predicted.get("elevation", "")
    else:
        # 向后兼容：仅返回整数 hour 的情况
        pred_hour = predicted
        pred_elev = ""

    try:
        pred_hour = int(pred_hour)
    except (ValueError, TypeError):
        pred_hour = -1

    # 水平方向评分
    h = score_trr_hour(pred_hour, gt_hour)
    qu = score_trr_quadrant(pred_hour, gt_quadrant)
    a = score_trr_adjacent(pred_hour, gt_hour)

    # 垂直仰角评分
    elev_exact = score_trr_elevation(pred_elev, gt_elevation_band)
    elev_adj = score_trr_elevation_adjacent(pred_elev, gt_elevation_band)

    # 完全三维正确：水平 hour 精确 + 仰角精确
    full_3d = h and elev_exact

    return {
        "hour_correct": h,
        "quadrant_correct": qu,
        "adjacent_correct": a,
        "elevation_correct": elev_exact,
        "elevation_adjacent": elev_adj,
        "full_3d_correct": full_3d,
    }


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
    """Kendall tau 排序相关系数（跳过并列对）。返回 [-1, 1]，1.0 = 完全一致。"""
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
                continue  # 并列：中立
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

def score_batch_scene(predictions: Dict[str, Any], questions: List[dict]) -> dict:
    """
    对单个场景的所有 batch 预测评分。

    返回 QRR/TRR/FDR 各项统计 + 逐题详情。
    TRR 问题若包含 gt_elevation_band 字段，使用三维评分。
    """
    qrr_correct = qrr_total = 0
    qrr_disjoint_correct = qrr_disjoint_total = 0
    qrr_shared_anchor_correct = qrr_shared_anchor_total = 0
    trr_hour_correct = trr_quad_correct = trr_adj_correct = trr_total = 0
    # 三维 TRR 仰角统计
    trr_elev_correct = trr_elev_adj_correct = trr_full_3d_correct = 0
    trr_elev_total = 0  # 含仰角真值的 TRR 题数
    fdr_exact_correct = fdr_total = 0
    fdr_kendall_sum = fdr_pairwise_sum = fdr_top1_sum = 0.0
    missing = 0
    per_question = []

    for q in questions:
        qid = q["qid"]
        pred = predictions.get(qid)

        if pred is None:
            missing += 1
            pq = {"qid": qid, "type": q["type"], "predicted": None, "correct": False}
            if q["type"] == "qrr":
                pq["variant"] = q.get("variant", "disjoint")
                if "anchor" in q:
                    pq["anchor"] = q["anchor"]
            per_question.append(pq)
            # 更新各题型总数（缺失也计入分母）
            if q["type"] == "qrr":
                qrr_total += 1
                variant = q.get("variant", "disjoint")
                if variant == "shared_anchor":
                    qrr_shared_anchor_total += 1
                else:
                    qrr_disjoint_total += 1
            elif q["type"] == "trr":
                trr_total += 1
                if q.get("gt_elevation_band"):
                    trr_elev_total += 1
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
            gt_elev_band = q.get("gt_elevation_band")

            if gt_elev_band:
                # 三维 TRR 评分（含仰角）
                trr_elev_total += 1
                scores_3d = score_trr_3d(pred, q["gt_hour"], q["gt_quadrant"], gt_elev_band)
                if scores_3d["hour_correct"]:
                    trr_hour_correct += 1
                if scores_3d["quadrant_correct"]:
                    trr_quad_correct += 1
                if scores_3d["adjacent_correct"]:
                    trr_adj_correct += 1
                if scores_3d["elevation_correct"]:
                    trr_elev_correct += 1
                if scores_3d["elevation_adjacent"]:
                    trr_elev_adj_correct += 1
                if scores_3d["full_3d_correct"]:
                    trr_full_3d_correct += 1

                # 提取预测值用于记录
                if isinstance(pred, dict):
                    pred_hour = pred.get("hour", -1)
                    pred_elev = pred.get("elevation", "")
                else:
                    pred_hour = pred
                    pred_elev = ""
                try:
                    pred_hour = int(pred_hour)
                except (ValueError, TypeError):
                    pred_hour = -1

                per_question.append({
                    "qid": qid, "type": "trr",
                    "predicted_hour": pred_hour,
                    "predicted_elevation": pred_elev,
                    "gt_hour": q["gt_hour"],
                    "gt_elevation_band": gt_elev_band,
                    **scores_3d,
                })
            else:
                # 纯 2D TRR 评分（无仰角真值）
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

    return {
        "qrr_correct": qrr_correct, "qrr_total": qrr_total,
        "qrr_disjoint_correct": qrr_disjoint_correct,
        "qrr_disjoint_total": qrr_disjoint_total,
        "qrr_shared_anchor_correct": qrr_shared_anchor_correct,
        "qrr_shared_anchor_total": qrr_shared_anchor_total,
        # TRR 水平方向
        "trr_hour_correct": trr_hour_correct,
        "trr_quadrant_correct": trr_quad_correct,
        "trr_adjacent_correct": trr_adj_correct,
        "trr_total": trr_total,
        # TRR 三维仰角
        "trr_elevation_correct": trr_elev_correct,
        "trr_elevation_adjacent_correct": trr_elev_adj_correct,
        "trr_full_3d_correct": trr_full_3d_correct,
        "trr_elevation_total": trr_elev_total,
        # FDR
        "fdr_exact_correct": fdr_exact_correct,
        "fdr_total": fdr_total,
        "fdr_kendall_mean": round(fdr_kendall_sum / fdr_total, 4) if fdr_total > 0 else 0.0,
        "fdr_pairwise_mean": round(fdr_pairwise_sum / fdr_total, 4) if fdr_total > 0 else 0.0,
        "fdr_top1_mean": round(fdr_top1_sum / fdr_total, 4) if fdr_total > 0 else 0.0,
        "missing": missing,
        "per_question": per_question,
    }


# ── 结果聚合 ──

# 跨场景求和的整数计数器键
_SUM_KEYS = [
    "qrr_correct", "qrr_total",
    "qrr_disjoint_correct", "qrr_disjoint_total",
    "qrr_shared_anchor_correct", "qrr_shared_anchor_total",
    "trr_hour_correct", "trr_quadrant_correct", "trr_adjacent_correct", "trr_total",
    # 三维仰角计数器
    "trr_elevation_correct", "trr_elevation_adjacent_correct",
    "trr_full_3d_correct", "trr_elevation_total",
    "fdr_exact_correct", "fdr_total",
    "missing",
]

# 需按 fdr_total 加权平均的键
_FDR_MEAN_KEYS = ["fdr_kendall_mean", "fdr_pairwise_mean", "fdr_top1_mean"]


def aggregate_batch_results(scene_results: List[dict]) -> dict:
    """汇总所有场景的 batch 评分，按 split 分组统计。包含三维仰角指标。"""
    total = {k: 0 for k in _SUM_KEYS}
    fdr_weighted = {k: 0.0 for k in _FDR_MEAN_KEYS}
    by_split = {}
    by_split_fdr = {}

    for r in scene_results:
        scene_id = r["scene_id"]
        split = scene_id.rsplit("_", 1)[0]
        s = r["scores"]

        for k in _SUM_KEYS:
            total[k] += s[k]
        fdr_n = s["fdr_total"]
        for k in _FDR_MEAN_KEYS:
            fdr_weighted[k] += s[k] * fdr_n

        if split not in by_split:
            by_split[split] = {k: 0 for k in _SUM_KEYS}
            by_split_fdr[split] = {k: 0.0 for k in _FDR_MEAN_KEYS}
        for k in _SUM_KEYS:
            by_split[split][k] += s[k]
        for k in _FDR_MEAN_KEYS:
            by_split_fdr[split][k] += s[k] * fdr_n

    def _acc(correct, total_count):
        return round(correct / total_count, 4) if total_count > 0 else 0.0

    def _fdr_means(weighted, fdr_total):
        return {k: round(weighted[k] / fdr_total, 4) if fdr_total > 0 else 0.0 for k in _FDR_MEAN_KEYS}

    def _build_summary(counts, fdr_w, fdr_n):
        """从计数器构建汇总字典，包含 2D 和 3D 指标。"""
        result = {
            # QRR
            "qrr_accuracy": _acc(counts["qrr_correct"], counts["qrr_total"]),
            "qrr_disjoint_accuracy": _acc(counts["qrr_disjoint_correct"], counts["qrr_disjoint_total"]),
            "qrr_shared_anchor_accuracy": _acc(counts["qrr_shared_anchor_correct"], counts["qrr_shared_anchor_total"]),
            # TRR 水平方向
            "trr_hour_accuracy": _acc(counts["trr_hour_correct"], counts["trr_total"]),
            "trr_quadrant_accuracy": _acc(counts["trr_quadrant_correct"], counts["trr_total"]),
            "trr_adjacent_accuracy": _acc(counts["trr_adjacent_correct"], counts["trr_total"]),
            # TRR 三维仰角
            "trr_elevation_accuracy": _acc(counts["trr_elevation_correct"], counts["trr_elevation_total"]),
            "trr_elevation_adjacent_accuracy": _acc(counts["trr_elevation_adjacent_correct"], counts["trr_elevation_total"]),
            "trr_full_3d_accuracy": _acc(counts["trr_full_3d_correct"], counts["trr_elevation_total"]),
            # FDR
            "fdr_exact_accuracy": _acc(counts["fdr_exact_correct"], counts["fdr_total"]),
            **_fdr_means(fdr_w, fdr_n),
            **counts,
        }
        return result

    summary = {
        "overall": _build_summary(total, fdr_weighted, total["fdr_total"]),
        "by_split": {},
    }
    for split, counts in sorted(by_split.items()):
        summary["by_split"][split] = _build_summary(
            counts, by_split_fdr[split], counts["fdr_total"],
        )

    return summary
