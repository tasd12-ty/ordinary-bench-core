"""
冲突检测与问题溯源
==================

从场景的 VLM 评测结果中提取 QRR 约束图，检测传递性冲突（环），
并将冲突约束**映射回可重问的原始问题**。

溯源逻辑：
    - 直接 QRR 约束 → qid 就是原始问题 ID，可直接重问
    - FDR 分解产生的 QRR 约束 → qid 格式为 "fdr_0006__pair_0_1"，
      需解析出原始 FDR 问题 ID "fdr_0006" 并重问整个 FDR 排序题

输出的 ConflictReport 包含：
    - conflict_qids: 需重问的原始问题 ID 集合（已去重）
    - conflict_questions: 对应的完整问题对象（可直接用于构建 prompt）
    - fas_result: 反馈弧集详情（用于诊断）
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set

sys.path.insert(0, str(Path(__file__).parent.parent))

from reconstruct.preparation import prepare_reconstruction_input_from_scoring

from .fas import FASResult, compute_fas


@dataclass
class ConflictReport:
    """冲突检测结果。

    Attributes:
        conflict_qids: 需要重问的原始问题 qid 集合
        conflict_questions: 对应的完整问题 dict 列表（含 type、pair1 等字段）
        fas_result: 反馈弧集计算详情
        n_total_questions: 该场景的总问题数
        n_qrr_constraints: QRR 约束总数（含 FDR 分解）
    """
    conflict_qids: Set[str] = field(default_factory=set)
    conflict_questions: List[dict] = field(default_factory=list)
    fas_result: Optional[FASResult] = None
    n_total_questions: int = 0
    n_qrr_constraints: int = 0


def _extract_original_qid(qid: str, source_type: str, entry: dict) -> str:
    """从约束条目中提取可重问的原始问题 qid。

    对于 FDR 分解产生的约束（source_type="fdr_decomposition"），
    其 qid 格式为 "fdr_0006__pair_0_1"，需要提取出原始 FDR 问题 ID "fdr_0006"，
    因为我们只能重问完整的 FDR 排序题，不能重问分解后的单个比较。
    """
    if source_type == "fdr_decomposition":
        # 优先使用显式记录的 source_qid 字段
        source_qid = entry.get("source_qid")
        if source_qid:
            return source_qid
        # 回退：从 qid 格式中解析
        m = re.match(r"(.+)__pair_\d+_\d+$", qid)
        return m.group(1) if m else qid
    # 直接 QRR 或 TRR 约束，qid 即为原始问题 ID
    return qid


def detect_conflicts(
    scene_result: dict,
    questions: List[dict],
    metadata: Optional[dict] = None,
) -> ConflictReport:
    """从场景评测结果中检测 QRR 约束冲突，映射回可重问的原始问题。

    流程：
        1. 调用重建准备模块，将 VLM 预测转换为 QRR 约束
           （belief 模式：使用 VLM 实际预测，不筛选正确答案）
        2. 对 qrr_all（直接 QRR + FDR 分解 QRR）计算最小反馈弧集
        3. 将 FAS 中被移除的约束映射回原始问题 qid
        4. 去重（多条 FDR 分解约束可能指向同一个 FDR 问题）

    Args:
        scene_result: 完整场景结果 dict（需包含 scores.per_question）
        questions: 该场景的问题列表（从 questions 目录加载）
        metadata: 可选的问题元数据

    Returns:
        ConflictReport，其中 conflict_questions 可直接传给 vlm_requester
    """
    # 从场景结果中提取评分数据
    scoring_input = scene_result.get("scores", scene_result)

    # 将 VLM 预测转换为约束（belief 模式）
    prep = prepare_reconstruction_input_from_scoring(
        scoring_input,
        questions,
        use_correct_only=False,
        metadata=metadata,
    )

    qrr_all = prep.qrr_all
    if not qrr_all:
        return ConflictReport(n_total_questions=len(questions))

    # 计算最小反馈弧集
    fas = compute_fas(qrr_all)

    # 将 FAS 边直接作为 QRR 问题重问（不追溯到 FDR 原始问题）
    # 原因: FDR 答案是排序列表，无法直接做多数投票；
    #        而 FAS 边本身就是 QRR pairwise 比较，可以直接重问。
    conflict_qids: Set[str] = set()
    conflict_questions: List[dict] = []
    seen_keys: Set[str] = set()

    for entry in fas.edges_removed:
        qid = entry.get("qid", "")
        source_type = entry.get("source_type", "qrr")

        if source_type == "fdr_decomposition":
            # FDR 分解来的约束 → 构造等价的 QRR 问题直接重问
            reask_q = {
                "qid": qid,  # 保留分解后的 qid (如 fdr_0006__pair_0_1)
                "type": "qrr",
                "variant": "shared_anchor",
                "anchor": entry.get("anchor", ""),
                "pair1": entry.get("pair1", []),
                "pair2": entry.get("pair2", []),
                "gt_comparator": entry.get("comparator", ""),
                "source_type": "fdr_decomposition",
                "source_fdr_qid": _extract_original_qid(qid, source_type, entry),
            }
            key = f"{reask_q['anchor']}|{'|'.join(sorted(reask_q['pair1'] + reask_q['pair2']))}"
        else:
            # 直接 QRR → 从原始问题列表中查找
            q_lookup_local = {q["qid"]: q for q in questions}
            reask_q = q_lookup_local.get(qid)
            if reask_q is None:
                continue
            key = qid

        if key not in seen_keys:
            seen_keys.add(key)
            conflict_qids.add(reask_q["qid"])
            conflict_questions.append(reask_q)

    return ConflictReport(
        conflict_qids=conflict_qids,
        conflict_questions=conflict_questions,
        fas_result=fas,
        n_total_questions=len(questions),
        n_qrr_constraints=len(qrr_all),
    )
