"""
端到端重建管线。

四个入口：
  1. reconstruct() — 从原始约束列表
  2. prepare_reconstruction_input_from_scoring() — 可审计的准备包
  3. reconstruct_from_scoring() — 从评分结果 + 问题元数据
  4. reconstruct_from_prepared() — 从准备好的包
"""

import numpy as np
from typing import Dict, List, Optional, Union
from dataclasses import dataclass, field

from .constraints import (
    QRREntry, TRREntry, FDREntry,
    build_distance_poset, build_angular_sectors,
    analyze_hypergraph, check_feasibility,
    FeasibilityReport,
)
from .preparation import (
    PreparedSceneInput,
    prepare_reconstruction_input_from_scoring,
)
from .solver import SolverConfig, SolverSolution, solve
from .evaluate import (
    EvalMetrics, evaluate_reconstruction, cluster_solutions, reflect_positions_y,
    compute_nrl, estimate_nrl_random, compute_significance,
)


CONSTRAINT_MODES = ("all", "fdr_only", "qrr_only", "fdr_qrr", "qrr_trr", "fdr_trr")


@dataclass
class ReconstructResult:
    """完整的重建输出。"""
    feasible: bool = False
    status: str = "infeasible"  # infeasible | underconstrained | single_mode | multimodal（不可行 | 约束不足 | 单模态 | 多模态）
    positions: Dict[str, np.ndarray] = field(default_factory=dict)
    metrics: EvalMetrics = field(default_factory=EvalMetrics)
    K_geom: int = 0
    all_solutions: List[SolverSolution] = field(default_factory=list)
    feasibility_checks: FeasibilityReport = field(default_factory=FeasibilityReport)
    constraint_mode: str = "all"
    constraint_counts: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """序列化为 JSON 兼容的字典。"""
        pos_dict = {k: v.tolist() for k, v in self.positions.items()}
        return {
            "feasible": self.feasible,
            "status": self.status,
            "positions": pos_dict,
            "metrics": {
                "csr_qrr": self.metrics.csr_qrr,
                "csr_trr": self.metrics.csr_trr,
                "csr_qrr_aligned": self.metrics.csr_qrr_aligned,
                "nrl": self.metrics.nrl,
                "p_value_qrr": self.metrics.p_value_qrr,
                "p_value_trr": self.metrics.p_value_trr,
                "K_geom": self.metrics.K_geom,
                "spread": self.metrics.spread,
                "kendall_tau": self.metrics.kendall_tau,
                "nrms": self.metrics.nrms,
                "best_loss": self.metrics.best_loss,
                "n_solutions": self.metrics.n_solutions,
                "cluster_sizes": self.metrics.cluster_sizes,
                "reflected": self.metrics.reflected,
            },
            "K_geom": self.K_geom,
            "n_solutions": len(self.all_solutions),
            "constraint_mode": self.constraint_mode,
            "constraint_counts": self.constraint_counts,
            "feasibility_checks": {
                "qrr_has_cycle": self.feasibility_checks.qrr_has_cycle,
                "qrr_cycle_info": self.feasibility_checks.qrr_cycle_info,
                "trr_n_conflicts": self.feasibility_checks.trr_n_conflicts,
                "hypergraph_connected": self.feasibility_checks.hypergraph_connected,
                "n_components": self.feasibility_checks.n_components,
                "n_qrr": self.feasibility_checks.n_qrr,
                "n_trr": self.feasibility_checks.n_trr,
                "n_objects": self.feasibility_checks.n_objects,
            },
        }


def reconstruct(
    qrr_constraints: List[dict],
    trr_constraints: List[dict],
    object_ids: List[str],
    gt_positions: Optional[Dict[str, np.ndarray]] = None,
    n_restarts: int = 10,
    config: Optional[SolverConfig] = None,
) -> ReconstructResult:
    """从 QRR + TRR 约束重建 2D 场景。

    参数:
        qrr_constraints: QRR 约束字典列表，键包括：
            pair1, pair2, comparator, [weight], [variant], [anchor]
        trr_constraints: TRR 约束字典列表，键包括：
            target, ref1, ref2, hour, [weight], [level]
        object_ids: 对象标识符列表
        gt_positions: 可选的真值位置，用于评估
        n_restarts: 优化重启次数
        config: 求解器配置

    返回:
        包含位置、指标和诊断信息的 ReconstructResult
    """
    if config is None:
        config = SolverConfig(n_restarts=n_restarts)
    else:
        config.n_restarts = n_restarts

    # 将约束字典解析为约束对象
    qrr_entries = [
        QRREntry(
            pair1=tuple(c["pair1"]),
            pair2=tuple(c["pair2"]),
            comparator=c["comparator"],
            weight=c.get("weight", 1.0),
            variant=c.get("variant", "disjoint"),
            anchor=c.get("anchor"),
        )
        for c in qrr_constraints
    ]
    trr_entries = [
        TRREntry(
            target=c["target"],
            ref1=c["ref1"],
            ref2=c["ref2"],
            hour=c["hour"],
            weight=c.get("weight", 1.0),
            level=c.get("level", "hour"),
        )
        for c in trr_constraints
    ]

    return _run_pipeline(qrr_entries, trr_entries, object_ids, gt_positions, config)


def reconstruct_from_scoring(
    scoring_result: dict,
    questions: List[dict],
    gt_positions: Optional[Dict[str, np.ndarray]] = None,
    n_restarts: int = 10,
    use_correct_only: bool = True,
    config: Optional[SolverConfig] = None,
) -> ReconstructResult:
    """从 score_batch_scene() 的评分输出进行重建。

    参数:
        scoring_result: score_batch_scene() 的输出，包含 per_question 列表
        questions: 带有真值元数据的原始问题列表
        gt_positions: 真值位置（obj_id -> [x, y] 或 [x, y, z] 的字典）
        n_restarts: 优化重启次数
        use_correct_only: 若为 True，仅使用正确回答的约束；
                         若为 False，使用所有 VLM 预测（用于信念重建）
        config: 求解器配置
    """
    if config is None:
        config = SolverConfig(n_restarts=n_restarts)
    else:
        config.n_restarts = n_restarts

    gt_serialized = None
    if gt_positions is not None:
        gt_serialized = {
            oid: np.asarray(pos, dtype=np.float64).tolist()
            for oid, pos in gt_positions.items()
        }

    prepared = prepare_reconstruction_input_from_scoring(
        scoring_result=scoring_result,
        questions=questions,
        gt_positions=gt_serialized,
        use_correct_only=use_correct_only,
    )
    return reconstruct_from_prepared(prepared, n_restarts=n_restarts, config=config)


def reconstruct_from_prepared(
    prepared_input: Union[PreparedSceneInput, dict],
    n_restarts: int = 10,
    config: Optional[SolverConfig] = None,
    constraint_mode: str = "all",
) -> ReconstructResult:
    """从准备好的逐场景包进行重建。

    参数:
        constraint_mode: 使用的约束子集：
            "all"       — qrr_direct + qrr_from_fdr + trr（默认）
            "fdr_only"  — 仅由 FDR 分解得到的 QRR
            "qrr_only"  — 仅直接 QRR（disjoint + shared_anchor）
            "fdr_qrr"   — qrr_direct + qrr_from_fdr，无 TRR
            "qrr_trr"   — qrr_direct + trr，无 FDR
            "fdr_trr"   — qrr_from_fdr + trr，无直接 QRR
    """
    if constraint_mode not in CONSTRAINT_MODES:
        raise ValueError(f"Unknown constraint_mode={constraint_mode!r}, "
                         f"expected one of {CONSTRAINT_MODES}")

    if config is None:
        config = SolverConfig(n_restarts=n_restarts)
    else:
        config.n_restarts = n_restarts

    prepared = (
        prepared_input
        if isinstance(prepared_input, PreparedSceneInput)
        else PreparedSceneInput.from_dict(prepared_input)
    )

    gt_positions = None
    if prepared.gt_positions:
        gt_positions = {
            oid: np.asarray(pos, dtype=np.float64)
            for oid, pos in prepared.gt_positions.items()
        }

    # 根据模式选择约束子集
    if constraint_mode == "fdr_only":
        qrr_sel = prepared.qrr_from_fdr
        trr_sel = []
    elif constraint_mode == "qrr_only":
        qrr_sel = prepared.qrr_constraints
        trr_sel = []
    elif constraint_mode == "fdr_qrr":
        qrr_sel = prepared.qrr_all
        trr_sel = []
    elif constraint_mode == "qrr_trr":
        qrr_sel = prepared.qrr_constraints
        trr_sel = prepared.trr_constraints
    elif constraint_mode == "fdr_trr":
        qrr_sel = prepared.qrr_from_fdr
        trr_sel = prepared.trr_constraints
    else:  # "all"
        qrr_sel = prepared.qrr_all
        trr_sel = prepared.trr_constraints

    counts = {
        "n_qrr_direct": len(prepared.qrr_constraints),
        "n_qrr_from_fdr": len(prepared.qrr_from_fdr),
        "n_trr": len(prepared.trr_constraints),
        "n_qrr_used": len(qrr_sel),
        "n_trr_used": len(trr_sel),
    }

    result = reconstruct(
        qrr_constraints=qrr_sel,
        trr_constraints=trr_sel,
        object_ids=prepared.object_ids,
        gt_positions=gt_positions,
        n_restarts=n_restarts,
        config=config,
    )
    result.constraint_mode = constraint_mode
    result.constraint_counts = counts
    return result


def _run_pipeline(
    qrr_entries: List[QRREntry],
    trr_entries: List[TRREntry],
    object_ids: List[str],
    gt_positions: Optional[Dict[str, np.ndarray]],
    config: SolverConfig,
) -> ReconstructResult:
    """内部管线：阶段 0-5。"""

    result = ReconstructResult()

    # ── 阶段 0-1：符号预处理 ──
    poset = build_distance_poset(qrr_entries)
    sectors = build_angular_sectors(trr_entries)
    hyper = analyze_hypergraph(qrr_entries, trr_entries, object_ids)
    feasibility = check_feasibility(poset, sectors, hyper)
    result.feasibility_checks = feasibility

    # 对象数不足
    if len(object_ids) < 3:
        result.status = "underconstrained"
        return result

    # 完全没有约束
    if not qrr_entries and not trr_entries:
        result.status = "underconstrained"
        return result

    # 符号不可行性快速失败
    if feasibility.qrr_has_cycle:
        result.feasible = False
        result.status = "infeasible"
        return result
    if feasibility.trr_n_conflicts > 0:
        result.feasible = False
        result.status = "infeasible"
        return result

    # ── 阶段 2-3：数值重建 ──
    # 将真值位置归一化为 2D
    gt_2d = None
    if gt_positions is not None:
        gt_2d = {}
        for oid, pos in gt_positions.items():
            if oid in object_ids:
                gt_2d[oid] = np.array(pos[:2], dtype=np.float64)

    solutions = solve(
        object_ids=object_ids,
        qrr_entries=qrr_entries,
        trr_entries=trr_entries,
        config=config,
        gt_positions=gt_2d,
    )

    if not solutions:
        result.status = "infeasible"
        return result

    result.all_solutions = solutions

    # ── 阶段 4-5：评估（考虑两种手性） ──
    metrics = evaluate_reconstruction(
        solutions=solutions,
        qrr_entries=qrr_entries,
        trr_entries=trr_entries,
        gt_positions=gt_2d,
        solver_config=config,
    )
    result.metrics = metrics
    result.K_geom = metrics.K_geom

    # 使用最优手性的位置
    if metrics.reflected:
        result.positions = reflect_positions_y(solutions[0].positions)
    else:
        result.positions = solutions[0].positions

    # ── 状态判定（三层准则） ──
    # Layer 2: 归一化损失优于随机基线 2 倍以上
    nrl_random = estimate_nrl_random(config.qrr_margin, config.qrr_beta)
    nrl_ok = metrics.nrl < nrl_random * 0.5

    # Layer 3: 二项显著性检验（CSR 显著高于随机猜测）
    _, _, sig_ok = compute_significance(
        metrics.csr_qrr_aligned, len(qrr_entries),
        metrics.csr_trr, len(trr_entries),
    )

    feasible = nrl_ok and sig_ok

    if not feasible:
        result.feasible = False
        result.status = "infeasible"
    elif metrics.K_geom == 1 and metrics.spread <= 0.10:
        result.feasible = True
        result.status = "single_mode"
    elif metrics.K_geom == 1 and metrics.spread > 0.10:
        result.feasible = True
        result.status = "underconstrained"
    else:
        result.feasible = True
        result.status = "multimodal"

    return result
