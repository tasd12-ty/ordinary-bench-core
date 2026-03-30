"""
迭代冲突消解主循环
==================

核心算法（不动点迭代）：

    Round 0: 加载现有评测结果，检测初始冲突（FAS）
    Round k (k ≥ 1):
        1. 检测当前约束图中的冲突 → FAS 边集
        2. 若 FAS = 0（无冲突）→ 完美收敛，退出
        3. 将 FAS 映射回原始问题，调 VLM 重问
        4. 用新答案替换旧答案，重新评分
        5. 检查收敛：FAS 是否减小？
           - 减小了 → 继续下一轮（噪声正在被消除）
           - 连续 patience 轮未减小 → 收敛，退出

    收敛后的诊断：
        noise_flips      = 迭代中答案被翻转的题数 → 随机噪声量
        systematic_conflicts = 收敛后残留的 FAS 大小 → 系统性错误量
        convergence_rounds   = 迭代轮数 → 模型稳定性

收敛判定示例：
    Round 1: FAS=6 → 重问 5 题 → FAS=3 (改善)
    Round 2: FAS=3 → 重问 3 题 → FAS=2 (改善)
    Round 3: FAS=2 → 重问 2 题 → FAS=2 (未改善, stale=1)
    Round 4: FAS=2 → 重问 2 题 → FAS=2 (未改善, stale=2 ≥ patience)
    → 停止。残留 2 条冲突 = 系统性错误。
"""

from __future__ import annotations

import copy
import sys
from dataclasses import dataclass, field
from pathlib import Path
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
class RoundLog:
    """单轮消解的详细记录。"""
    round_idx: int                    # 轮次编号（从 1 开始）
    fas_size: int                     # 本轮开始时的 FAS 大小
    n_conflict_questions: int         # 需重问的原始问题数
    n_flipped: int                    # VLM 给出了不同答案的题数
    conflict_qids: List[str] = field(default_factory=list)
    flipped_qids: List[str] = field(default_factory=list)


@dataclass
class Diagnosis:
    """噪声 vs 系统性错误分解诊断。"""
    total_questions: int              # 场景总问题数
    noise_flips: int                  # 迭代中被翻转的答案总数（= 随机噪声）
    systematic_conflicts: int         # 收敛后残留的 FAS 大小（= 系统性错误）
    noise_ratio: float = 0.0         # noise_flips / total_questions
    systematic_ratio: float = 0.0    # systematic_conflicts / n_qrr_constraints
    convergence_rounds: int = 0      # 迭代轮数


@dataclass
class ResolutionResult:
    """单场景的完整消解结果。"""
    scene_id: str
    converged: bool                   # 是否有改善（FAS 减小或降至 0）
    n_rounds: int                     # 实际迭代轮数
    initial_fas_size: int             # 初始 FAS 大小
    final_fas_size: int               # 最终 FAS 大小
    history: List[RoundLog] = field(default_factory=list)
    final_scene_result: Optional[dict] = None     # 更新后的场景评测结果
    diagnosis: Optional[Diagnosis] = None


# ── 内部函数 ──


def _update_per_question(
    per_question: List[dict],
    new_preds: Dict[str, Any],
    questions: List[dict],
) -> tuple:
    """用新预测更新 per_question 中的 predicted 字段，并重新评分。

    Args:
        per_question: 场景评测结果中的 scores.per_question 列表（就地修改）
        new_preds: {qid: new_prediction} VLM 新回答
        questions: 原始问题列表（含真值用于重新评分）

    Returns:
        (n_flipped, flipped_qids)：答案发生变化的题数和 qid 列表
    """
    q_lookup = {q["qid"]: q for q in questions}
    pq_lookup = {pq["qid"]: pq for pq in per_question}
    n_flipped = 0
    flipped_qids = []

    for qid, new_val in new_preds.items():
        if new_val is None:
            continue
        pq = pq_lookup.get(qid)
        if pq is None:
            continue

        old_val = pq.get("predicted")

        # 检测答案是否翻转
        if str(new_val) != str(old_val):
            n_flipped += 1
            flipped_qids.append(qid)

        # 更新预测值
        pq["predicted"] = new_val

        # 根据题型重新评分
        q = q_lookup.get(qid, {})
        qtype = pq.get("type", q.get("type", ""))

        if qtype == "qrr":
            gt = q.get("gt_comparator", "")
            pq["correct"] = score_qrr(str(new_val), gt) if gt else False

        elif qtype == "trr":
            gt_hour = q.get("gt_hour", pq.get("gt_hour"))
            gt_quad = q.get("gt_quadrant", pq.get("gt_quadrant"))
            try:
                h = int(new_val)
                pq["hour_correct"] = score_trr_hour(h, gt_hour) if gt_hour else False
                pq["quadrant_correct"] = score_trr_quadrant(h, gt_quad) if gt_quad else False
                pq["adjacent_correct"] = score_trr_adjacent(h, gt_hour) if gt_hour else False
            except (ValueError, TypeError):
                pass

        elif qtype == "fdr":
            gt_ranking = q.get("gt_ranking", [])
            gt_ties = q.get("gt_tie_groups", [])
            if isinstance(new_val, list) and gt_ranking:
                pq["correct"] = score_fdr_exact(new_val, gt_ranking, gt_ties)

    return n_flipped, flipped_qids


# ── 主循环 ──


def resolve_scene(
    scene_id: str,
    scene_result: dict,
    questions: List[dict],
    scene_objects: List[dict],
    image_inputs: List[dict],
    provider_spec,
    metadata: Optional[dict] = None,
    max_rounds: int = 10,
    patience: int = 2,
) -> ResolutionResult:
    """对单场景执行迭代冲突消解。

    Args:
        scene_id: 场景 ID（如 "n06_000080"）
        scene_result: 现有评测结果（深拷贝后操作，不修改原始数据）
        questions: 该场景的问题列表
        scene_objects: 物体描述列表（构建 VLM prompt 用）
        image_inputs: 图片输入列表（传给 VLM adapter）
        provider_spec: VLM 服务配置
        metadata: 问题元数据
        max_rounds: 最大迭代轮数
        patience: FAS 连续未减小的容忍轮数

    Returns:
        ResolutionResult，包含收敛状态、诊断信息和更新后的场景结果。
    """
    # 深拷贝，避免污染原始数据
    sr = copy.deepcopy(scene_result)
    history: List[RoundLog] = []
    all_flipped_qids: Set[str] = set()

    # ── Round 0: 初始冲突检测 ──
    report = detect_conflicts(sr, questions, metadata)
    initial_fas = len(report.fas_result.edges_removed) if report.fas_result else 0

    # 无冲突，直接返回
    if initial_fas == 0:
        return ResolutionResult(
            scene_id=scene_id,
            converged=True,
            n_rounds=0,
            initial_fas_size=0,
            final_fas_size=0,
            final_scene_result=sr,
            diagnosis=Diagnosis(
                total_questions=len(questions),
                noise_flips=0,
                systematic_conflicts=0,
                convergence_rounds=0,
            ),
        )

    # ── 迭代消解 ──
    prev_fas = initial_fas
    stale_count = 0

    for round_idx in range(1, max_rounds + 1):
        # 检测当前冲突
        report = detect_conflicts(sr, questions, metadata)
        fas_size = len(report.fas_result.edges_removed) if report.fas_result else 0
        conflict_qids_list = sorted(report.conflict_qids)

        # 冲突已全部消除
        if fas_size == 0:
            history.append(RoundLog(round_idx, 0, 0, 0))
            break

        # 调 VLM 重问冲突题
        new_preds = reask_questions(
            report.conflict_questions,
            scene_objects,
            image_inputs,
            provider_spec,
        )

        # 用新答案更新 per_question 并重新评分
        per_question = sr["scores"]["per_question"]
        n_flipped, flipped = _update_per_question(per_question, new_preds, questions)
        all_flipped_qids.update(flipped)

        # 记录本轮
        history.append(RoundLog(
            round_idx=round_idx,
            fas_size=fas_size,
            n_conflict_questions=len(conflict_qids_list),
            n_flipped=n_flipped,
            conflict_qids=conflict_qids_list,
            flipped_qids=flipped,
        ))

        print(f"    Round {round_idx}: FAS={fas_size} "
              f"重问={len(conflict_qids_list)} 翻转={n_flipped}")

        # ── 收敛检查 ──
        if fas_size >= prev_fas:
            stale_count += 1
        else:
            stale_count = 0

        if stale_count >= patience:
            break

        prev_fas = fas_size

    # ── 最终状态 ──
    final_report = detect_conflicts(sr, questions, metadata)
    final_fas = len(final_report.fas_result.edges_removed) if final_report.fas_result else 0

    diagnosis = Diagnosis(
        total_questions=len(questions),
        noise_flips=len(all_flipped_qids),
        systematic_conflicts=final_fas,
        noise_ratio=len(all_flipped_qids) / len(questions) if questions else 0,
        systematic_ratio=(
            final_fas / report.n_qrr_constraints
            if report.n_qrr_constraints else 0
        ),
        convergence_rounds=len(history),
    )

    return ResolutionResult(
        scene_id=scene_id,
        converged=final_fas < initial_fas or final_fas == 0,
        n_rounds=len(history),
        initial_fas_size=initial_fas,
        final_fas_size=final_fas,
        history=history,
        final_scene_result=sr,
        diagnosis=diagnosis,
    )
