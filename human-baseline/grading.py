"""人类基准渐进式测试的逐题评分模块。

封装 VLM-test/API-test/scoring.py 中的评分函数，
为渐进式测试前端提供逐题反馈。
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
API_ROOT = REPO_ROOT / "VLM-test" / "API-test"
VLM_ROOT = REPO_ROOT / "VLM-test"

for p in (str(API_ROOT), str(VLM_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from scoring import (  # noqa: E402
    score_fdr_exact,
    score_fdr_pairwise,
    score_qrr,
    score_trr_adjacent,
    score_trr_hour,
)

from common import normalize_human_answer  # noqa: E402


def grade_question(question: dict, answer: Any) -> dict:
    """对单个问题进行评分并返回详细反馈。

    参数
    ----------
    question:
        至少包含 ``qid``、``type`` 和真值字段的问题字典。
    answer:
        用户的原始回答。

    返回值
    -------
    包含以下键的字典：qid、type、correct、user_answer、correct_answer、
    correct_answer_display、detail。
    """
    qid = question["qid"]
    qtype = question["type"]
    normalized = normalize_human_answer(answer, qtype)

    result = {
        "qid": qid,
        "type": qtype,
        "correct": False,
        "user_answer": answer,
        "user_answer_normalized": normalized,
        "correct_answer": None,
        "correct_answer_display": "",
        "detail": {},
    }

    if qtype == "qrr":
        gt = question["gt_comparator"]
        result["correct_answer"] = gt
        label_map = {"<": "前者更近", "~=": "大致相等", ">": "后者更近"}
        result["correct_answer_display"] = f"{gt} ({label_map.get(gt, gt)})"
        if normalized is not None:
            result["correct"] = score_qrr(str(normalized), gt)

    elif qtype == "trr":
        gt_hour = question["gt_hour"]
        result["correct_answer"] = gt_hour
        result["correct_answer_display"] = f"{gt_hour} 点钟方向"
        if normalized is not None:
            exact = score_trr_hour(int(normalized), gt_hour)
            adjacent = score_trr_adjacent(int(normalized), gt_hour)
            result["correct"] = adjacent  # 晋级阈值
            result["detail"] = {"exact_match": exact, "adjacent_match": adjacent}

    elif qtype == "fdr":
        gt_ranking = question["gt_ranking"]
        gt_tie_groups = question.get("gt_tie_groups", [[x] for x in gt_ranking])
        result["correct_answer"] = gt_ranking
        result["correct_answer_display"] = " > ".join(gt_ranking)
        if normalized is not None and isinstance(normalized, list):
            exact = score_fdr_exact(normalized, gt_ranking, gt_tie_groups)
            pairwise = score_fdr_pairwise(normalized, gt_ranking, gt_tie_groups)
            result["correct"] = pairwise >= 1.0  # 晋级阈值
            result["detail"] = {
                "exact_match": exact,
                "pairwise_accuracy": round(pairwise, 4),
            }

    return result


def grade_round(questions: List[dict], answers: Dict[str, Any]) -> dict:
    """对一轮中的所有问题进行评分。

    参数
    ----------
    questions:
        包含真值的问题字典列表。
    answers:
        qid -> 用户答案的映射。

    返回值
    -------
    包含以下键的字典：all_correct、n_correct、n_total、results、wrong_qids。
    """
    results = []
    wrong_qids = []

    for q in questions:
        answer = answers.get(q["qid"])
        r = grade_question(q, answer)
        results.append(r)
        if not r["correct"]:
            wrong_qids.append(q["qid"])

    n_total = len(results)
    n_correct = n_total - len(wrong_qids)

    return {
        "all_correct": len(wrong_qids) == 0,
        "n_correct": n_correct,
        "n_total": n_total,
        "results": results,
        "wrong_qids": wrong_qids,
    }
