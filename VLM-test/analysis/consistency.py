"""
一致性分析：传递性、互反性与结构连贯性检验。

实现 experiment-spec.md（第 3.3、8.5 节）中定义的可靠性度量。
"""

import math
from collections import defaultdict
from typing import Dict, List, Tuple, Optional


def check_transitivity(per_question: List[dict], questions: List[dict]) -> dict:
    """检验 QRR 传递性：若 d(A,B) < d(C,D) 且 d(C,D) < d(E,F)，
    则 d(A,B) < d(E,F) 是否成立？

    Returns:
        {
            "n_chains": int,       # 被测试的传递链数量
            "n_satisfied": int,    # 传递性成立的链数
            "n_violated": int,     # 传递性违反的链数
            "violation_rate": float,
        }
    """
    q_lookup = {q["qid"]: q for q in questions}

    # 构建预测排序：对 -> 对比较关系（仅使用有预测的 QRR 题）
    pred_qrr = {}
    for pq in per_question:
        if pq["type"] != "qrr" or pq.get("predicted") is None:
            continue
        q = q_lookup.get(pq["qid"])
        if q is None:
            continue
        p1 = tuple(sorted(q["pair1"]))
        p2 = tuple(sorted(q["pair2"]))
        pred = str(pq["predicted"]).strip()
        if pred in ("<", ">", "~="):
            pred_qrr[(p1, p2)] = pred

    # 查找传递链：A<B 且 B<C => A<C
    # 以"对"作为节点，以 < 作为有向边
    lt_edges = defaultdict(set)  # pair -> 该 pair 小于的所有 pair 集合
    eq_groups = defaultdict(set)  # pair -> 与该 pair 近似相等的 pair 集合

    for (p1, p2), cmp in pred_qrr.items():
        if cmp == "<":
            lt_edges[p1].add(p2)
        elif cmp == ">":
            lt_edges[p2].add(p1)
        elif cmp == "~=":
            eq_groups[p1].add(p2)
            eq_groups[p2].add(p1)

    # 检验所有长度为 2 的传递链
    chains_tested = 0
    chains_satisfied = 0
    chains_violated = 0

    all_pairs = set()
    for (p1, p2) in pred_qrr:
        all_pairs.add(p1)
        all_pairs.add(p2)

    for a in all_pairs:
        for b in lt_edges.get(a, set()):
            for c in lt_edges.get(b, set()):
                if a == c:
                    continue
                chains_tested += 1
                # a < b < c => 应有 a < c
                key_ac = (a, c) if (a, c) in pred_qrr else None
                key_ca = (c, a) if (c, a) in pred_qrr else None

                if key_ac:
                    if pred_qrr[key_ac] == "<":
                        chains_satisfied += 1
                    else:
                        chains_violated += 1
                elif key_ca:
                    if pred_qrr[key_ca] == ">":
                        chains_satisfied += 1
                    else:
                        chains_violated += 1
                # 若两者均不存在，则无法检验

    return {
        "n_chains": chains_tested,
        "n_satisfied": chains_satisfied,
        "n_violated": chains_violated,
        "violation_rate": chains_violated / chains_tested if chains_tested > 0 else 0.0,
    }


def check_reciprocity(per_question: List[dict], questions: List[dict]) -> dict:
    """检验 TRR 互反性：若以 B->C 方向看 A 位于 3 点钟，
    则以 C->B 方向看 A 应位于约 9 点钟（反方向）。

    小时 h 的互反值为：(h + 6 - 1) % 12 + 1

    Returns:
        {
            "n_pairs": int,
            "n_exact_reciprocal": int,    # 恰好互反
            "n_quadrant_reciprocal": int, # 与互反值同象限
            "n_violated": int,
            "violation_rate": float,
        }
    """
    q_lookup = {q["qid"]: q for q in questions}

    # 构建预测 TRR 映射：(target, ref1, ref2) -> 预测小时
    pred_trr = {}
    for pq in per_question:
        if pq["type"] != "trr" or pq.get("predicted") is None:
            continue
        q = q_lookup.get(pq["qid"])
        if q is None:
            continue
        try:
            hour = int(pq["predicted"])
            if 1 <= hour <= 12:
                pred_trr[(q["target"], q["ref1"], q["ref2"])] = hour
        except (ValueError, TypeError):
            pass

    # 检验互反对
    # 若 (T, R1, R2) = h，则交换 R1<->R2 后应得到互反值
    # 互反小时：旋转 180 度 = (h + 6 - 1) % 12 + 1
    n_pairs = 0
    n_exact = 0
    n_quadrant = 0
    n_violated = 0

    checked = set()
    for (target, ref1, ref2), hour1 in pred_trr.items():
        # Check if the swapped version exists
        key2 = (target, ref2, ref1)
        if key2 in pred_trr and (target, ref1, ref2) not in checked:
            checked.add((target, ref1, ref2))
            checked.add(key2)

            hour2 = pred_trr[key2]
            expected_reciprocal = (hour1 + 6 - 1) % 12 + 1

            n_pairs += 1

            if hour2 == expected_reciprocal:
                n_exact += 1
            else:
                # 检查是否与期望互反值同象限
                h2_q = _hour_to_quadrant(hour2)
                exp_q = _hour_to_quadrant(expected_reciprocal)
                if h2_q == exp_q:
                    n_quadrant += 1
                else:
                    n_violated += 1

    return {
        "n_pairs": n_pairs,
        "n_exact_reciprocal": n_exact,
        "n_quadrant_reciprocal": n_quadrant,
        "n_violated": n_violated,
        "exact_rate": n_exact / n_pairs if n_pairs > 0 else 0.0,
        "violation_rate": n_violated / n_pairs if n_pairs > 0 else 0.0,
    }


def _hour_to_quadrant(hour: int) -> int:
    if hour in (12, 1, 2):
        return 1
    elif hour in (3, 4, 5):
        return 2
    elif hour in (6, 7, 8):
        return 3
    else:
        return 4


def analyze_scene_consistency(
    scene_result: dict,
    questions: List[dict],
) -> dict:
    """对单个场景执行完整一致性分析。"""
    per_question = scene_result.get("scores", scene_result).get("per_question", [])

    transitivity = check_transitivity(per_question, questions)
    reciprocity = check_reciprocity(per_question, questions)

    return {
        "scene_id": scene_result.get("scene_id", "unknown"),
        "transitivity": transitivity,
        "reciprocity": reciprocity,
    }


def format_consistency_table(results: List[dict]) -> str:
    """将一致性分析结果格式化为 Markdown 表格。"""
    lines = ["## Consistency Analysis", ""]
    lines.append("| Scene | Trans. Chains | Trans. Violated | Recip. Pairs | Recip. Exact | Recip. Violated |")
    lines.append("|" + "---|" * 6)

    total_chains = total_trans_viol = 0
    total_recip = total_recip_exact = total_recip_viol = 0

    for r in results:
        t = r["transitivity"]
        rec = r["reciprocity"]
        total_chains += t["n_chains"]
        total_trans_viol += t["n_violated"]
        total_recip += rec["n_pairs"]
        total_recip_exact += rec["n_exact_reciprocal"]
        total_recip_viol += rec["n_violated"]

        lines.append(
            f"| {r['scene_id']} "
            f"| {t['n_chains']} | {t['n_violated']} ({t['violation_rate']:.1%}) "
            f"| {rec['n_pairs']} | {rec['n_exact_reciprocal']} ({rec['exact_rate']:.1%}) "
            f"| {rec['n_violated']} ({rec['violation_rate']:.1%}) |"
        )

    # 汇总行
    trans_rate = total_trans_viol / total_chains if total_chains else 0
    recip_exact_rate = total_recip_exact / total_recip if total_recip else 0
    lines.append(
        f"| **Total** "
        f"| {total_chains} | {total_trans_viol} ({trans_rate:.1%}) "
        f"| {total_recip} | {total_recip_exact} ({recip_exact_rate:.1%}) "
        f"| {total_recip_viol} |"
    )

    return "\n".join(lines)
