"""
基于多数投票的冲突消解
========================

统计学方法：对冲突问题重复问询 K 次，通过多数投票估计模型的真实信念。

核心假设:
    模型对每道题有一个"真实信念"(true belief)。
    每次回答是该信念 + 噪声。噪声率约 1%。
    通过多次独立采样，多数投票可以还原真实信念。

与旧 resolver.py 的区别:
    旧: 迭代式，每轮重问 → 覆盖旧答案 → 重新检测 FAS → 循环
        问题: "最后答案胜出"，模型猜对也被接受，无法区分噪声和系统错误
    新: 一次性重问 K 次 → 收集所有回答 → 多数投票 → 诊断
        优势: 统计学上可靠，明确区分噪声/系统错误/不确定

流程:
    1. 加载初始评测结果
    2. 检测 FAS 冲突 → conflict_qids
    3. 对 conflict_qids 重问 K 次 (独立调用，不加任何额外 prompt)
    4. 收集 1+K 个回答 (1=原始, K=重问)
    5. 多数投票 → voted_answer
    6. 诊断:
       - noise_corrected: 投票结果 ≠ 原始答案 (噪声被纠正)
       - systematic_wrong: 投票后仍 ≠ GT (系统性错误)
       - uncertain: 投票无明显多数 (模型不确定)
    7. 用投票结果替换预测，重新检测 FAS → remaining = 系统性错误
"""

from __future__ import annotations

import copy
import sys
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

sys.path.insert(0, str(Path(__file__).parent.parent / "API-test"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from scoring import (
    score_fdr_exact,
    score_qrr,
    score_trr_adjacent,
    score_trr_hour,
    score_trr_quadrant,
)

from .conflict_detector import detect_conflicts
from .vlm_requester import reask_questions


# ── 数据结构 ──


@dataclass
class VoteRecord:
    """单个问题的投票记录。"""
    qid: str
    qtype: str
    gt: Any                          # 真值
    original_answer: Any             # Round 0 原始回答
    reask_answers: List[Any]         # K 次重问的回答
    voted_answer: Any                # 多数投票结果
    vote_counts: Dict[str, int]      # {答案: 出现次数}
    n_votes: int                     # 总投票数 (1+K)
    majority_ratio: float            # 多数答案的占比
    original_correct: bool           # 原始回答是否正确
    voted_correct: bool              # 投票后是否正确
    status: str                      # "noise_corrected" | "systematic_wrong" | "uncertain" | "confirmed_correct"


@dataclass
class VotingDiagnosis:
    """投票诊断结果。"""
    total_conflict_questions: int    # FAS 冲突涉及的问题数
    reask_rounds: int                # 重问轮数 K
    noise_corrected: int             # 投票纠正了原始错误答案
    systematic_wrong: int            # 投票后仍然错误 (模型真不会)
    uncertain: int                   # 无明显多数 (模型不确定)
    confirmed_correct: int           # 原始就对，投票确认
    initial_fas_size: int            # 初始 FAS 环边数
    final_fas_size: int              # 投票后 FAS 环边数


@dataclass
class VotingResult:
    """完整的投票消解结果。"""
    scene_id: str
    vote_records: List[VoteRecord]
    diagnosis: VotingDiagnosis
    final_scene_result: Optional[dict] = None  # 更新后的场景结果


# ── 投票逻辑 ──


def majority_vote(answers: List[Any]) -> tuple[Any, Dict[str, int], float]:
    """多数投票。

    冲突问题都是 QRR 形式（含 FDR 分解来的），答案为 "<" / "~=" / ">"。

    Returns:
        (voted_answer, vote_counts, majority_ratio)
    """
    # 过滤 None (VLM 调用失败)
    valid = [str(a) for a in answers if a is not None]
    if not valid:
        return None, {}, 0.0

    counter = Counter(valid)
    winner, count = counter.most_common(1)[0]
    ratio = count / len(valid)

    return winner, dict(counter), ratio


def _answers_match(answer: Any, gt: Any) -> bool:
    """比较答案与真值是否匹配，支持列表和标量类型。"""
    if answer is None or gt is None:
        return False
    return _normalize_answer(answer) == _normalize_answer(gt)


def _classify_vote(
    original_answer: Any,
    voted_answer: Any,
    gt: Any,
    majority_ratio: float,
    uncertainty_threshold: float = 0.6,
) -> str:
    """诊断投票结果。

    Args:
        uncertainty_threshold: 多数比例低于此值视为"不确定"
    """
    if voted_answer is None:
        return "uncertain"

    voted_correct = (str(voted_answer) == str(gt))
    original_correct = (str(original_answer) == str(gt))

    if majority_ratio < uncertainty_threshold:
        return "uncertain"
    elif original_correct and voted_correct:
        return "confirmed_correct"
    elif not original_correct and voted_correct:
        return "noise_corrected"
    else:
        return "systematic_wrong"


# ── 主流程 ──


def voting_resolve_scene(
    scene_id: str,
    scene_result: dict,
    questions: List[dict],
    scene_objects: List[dict],
    image_inputs: List[dict],
    provider_spec,
    metadata: Optional[dict] = None,
    reask_rounds: int = 4,
    uncertainty_threshold: float = 0.6,
) -> VotingResult:
    """对单场景执行投票式冲突消解。

    Args:
        scene_id: 场景 ID
        scene_result: 现有评测结果 (不会被修改)
        questions: 该场景的问题列表 (含 GT)
        scene_objects: 物体描述列表 (构建 VLM prompt 用)
        image_inputs: 图片输入 (传给 VLM adapter)
        provider_spec: VLM 服务配置
        metadata: 问题元数据
        reask_rounds: 重问轮数 K (总投票数 = 1+K)
        uncertainty_threshold: 多数比例低于此值视为不确定

    Returns:
        VotingResult 包含每题投票记录、诊断和更新后的场景结果。
    """
    sr = copy.deepcopy(scene_result)
    q_lookup = {q["qid"]: q for q in questions}

    # ── Step 1: 检测初始冲突 ──
    report = detect_conflicts(sr, questions, metadata)
    initial_fas = len(report.fas_result.edges_removed) if report.fas_result else 0

    if initial_fas == 0:
        # 无冲突，无需投票
        return VotingResult(
            scene_id=scene_id,
            vote_records=[],
            diagnosis=VotingDiagnosis(
                total_conflict_questions=0,
                reask_rounds=0,
                noise_corrected=0,
                systematic_wrong=0,
                uncertain=0,
                confirmed_correct=0,
                initial_fas_size=0,
                final_fas_size=0,
            ),
            final_scene_result=sr,
        )

    conflict_qids = report.conflict_qids
    conflict_questions = report.conflict_questions

    print(f"  Scene {scene_id}: FAS={initial_fas}, "
          f"conflict questions={len(conflict_qids)}, "
          f"reask_rounds={reask_rounds}")

    # ── Step 2: 收集原始回答 (Round 0) ──
    pq_lookup = {pq["qid"]: pq for pq in sr["scores"]["per_question"]}
    original_answers = {}
    for qid in conflict_qids:
        pq = pq_lookup.get(qid)
        if pq:
            original_answers[qid] = pq.get("predicted")

    # ── Step 3: 重问 K 轮，收集所有回答 ──
    all_round_answers: List[Dict[str, Any]] = []

    for k in range(reask_rounds):
        print(f"    Reask round {k+1}/{reask_rounds} "
              f"({len(conflict_questions)} questions)...")
        new_preds = reask_questions(
            conflict_questions,
            scene_objects,
            image_inputs,
            provider_spec,
        )
        all_round_answers.append(new_preds)

    # ── Step 4: 多数投票 ──
    vote_records: List[VoteRecord] = []
    voted_predictions: Dict[str, Any] = {}

    for qid in sorted(conflict_qids):
        q = q_lookup.get(qid, {})
        qtype = q.get("type", "qrr")

        # 收集所有回答: 原始 + K 次重问
        all_answers = [original_answers.get(qid)]
        for round_preds in all_round_answers:
            all_answers.append(round_preds.get(qid))

        # 提取真值
        gt = q.get("gt_comparator") or q.get("gt_hour") or q.get("gt_ranking")

        # 多数投票
        voted, counts, ratio = majority_vote(all_answers)

        # 诊断
        status = _classify_vote(
            original_answers.get(qid), voted, gt, ratio,
            uncertainty_threshold=uncertainty_threshold,
        )

        record = VoteRecord(
            qid=qid,
            qtype=qtype,
            gt=gt,
            original_answer=original_answers.get(qid),
            reask_answers=[r.get(qid) for r in all_round_answers],
            voted_answer=voted,
            vote_counts=counts,
            n_votes=len([a for a in all_answers if a is not None]),
            majority_ratio=ratio,
            original_correct=_answers_match(original_answers.get(qid), gt),
            voted_correct=_answers_match(voted, gt),
            status=status,
        )
        vote_records.append(record)
        voted_predictions[qid] = voted

    # ── Step 5: 用投票结果更新 per_question 并重新评分 ──
    for pq in sr["scores"]["per_question"]:
        qid = pq["qid"]
        if qid not in voted_predictions:
            continue

        voted_val = voted_predictions[qid]
        if voted_val is None:
            continue

        pq["predicted"] = voted_val
        pq["vote_resolved"] = True

        # 冲突问题都是 QRR 形式，重新评分
        q = q_lookup.get(qid, {})
        gt_cmp = q.get("gt_comparator", "")
        pq["correct"] = score_qrr(str(voted_val), gt_cmp) if gt_cmp else False

    # ── Step 6: 重新检测 FAS ──
    final_report = detect_conflicts(sr, questions, metadata)
    final_fas = len(final_report.fas_result.edges_removed) if final_report.fas_result else 0

    # ── 汇总诊断 ──
    counts_by_status = Counter(r.status for r in vote_records)

    diagnosis = VotingDiagnosis(
        total_conflict_questions=len(conflict_qids),
        reask_rounds=reask_rounds,
        noise_corrected=counts_by_status.get("noise_corrected", 0),
        systematic_wrong=counts_by_status.get("systematic_wrong", 0),
        uncertain=counts_by_status.get("uncertain", 0),
        confirmed_correct=counts_by_status.get("confirmed_correct", 0),
        initial_fas_size=initial_fas,
        final_fas_size=final_fas,
    )

    print(f"  Result: FAS {initial_fas} → {final_fas}")
    print(f"    noise_corrected={diagnosis.noise_corrected}, "
          f"systematic_wrong={diagnosis.systematic_wrong}, "
          f"uncertain={diagnosis.uncertain}, "
          f"confirmed_correct={diagnosis.confirmed_correct}")

    return VotingResult(
        scene_id=scene_id,
        vote_records=vote_records,
        diagnosis=diagnosis,
        final_scene_result=sr,
    )


def voting_result_to_dict(result: VotingResult) -> dict:
    """将投票结果序列化为 JSON 可存储的 dict。"""
    return {
        "scene_id": result.scene_id,
        "diagnosis": {
            "total_conflict_questions": result.diagnosis.total_conflict_questions,
            "reask_rounds": result.diagnosis.reask_rounds,
            "noise_corrected": result.diagnosis.noise_corrected,
            "systematic_wrong": result.diagnosis.systematic_wrong,
            "uncertain": result.diagnosis.uncertain,
            "confirmed_correct": result.diagnosis.confirmed_correct,
            "initial_fas_size": result.diagnosis.initial_fas_size,
            "final_fas_size": result.diagnosis.final_fas_size,
        },
        "vote_records": [
            {
                "qid": r.qid,
                "qtype": r.qtype,
                "gt": r.gt,
                "original_answer": r.original_answer,
                "reask_answers": r.reask_answers,
                "voted_answer": r.voted_answer,
                "vote_counts": r.vote_counts,
                "n_votes": r.n_votes,
                "majority_ratio": round(r.majority_ratio, 4),
                "original_correct": r.original_correct,
                "voted_correct": r.voted_correct,
                "status": r.status,
            }
            for r in result.vote_records
        ],
    }
