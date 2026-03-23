"""
三维端到端重建管线。

入口点：
  1. reconstruct() — 从原始约束列表重建
  2. prepare_reconstruction_input_from_scoring() — 可审计的预处理
  3. reconstruct_from_scoring() — 从评分结果 + 问题元数据重建
  4. reconstruct_from_prepared() — 从预处理包重建

在 2D 版本基础上的改动：
  - 支持 TRR3DEntry 三维角度约束
  - 使用 solve_3d() 替代 solve() 进行三维求解
  - 使用 evaluate_reconstruction_3d() 进行三维评估
  - 真值位置提取 3 个坐标分量（x, y, z）
"""

import numpy as np
from typing import Dict, List, Optional, Union
from dataclasses import dataclass, field

from .constraints import (
    QRREntry, TRREntry, TRR3DEntry, FDREntry,
    build_distance_poset, build_angular_sectors, build_angular_sectors_3d,
    analyze_hypergraph, check_feasibility,
    FeasibilityReport,
)
from .preparation import (
    PreparedSceneInput,
    prepare_reconstruction_input_from_scoring,
)
from .solver import SolverConfig, SolverSolution, solve_3d
from .evaluate import EvalMetrics3D, evaluate_reconstruction_3d, cluster_solutions


# 支持的约束子集模式
CONSTRAINT_MODES = ("all", "fdr_only", "qrr_only", "fdr_qrr", "qrr_trr", "fdr_trr")


@dataclass
class ReconstructResult:
    """三维重建的完整输出。"""
    feasible: bool = False
    status: str = "infeasible"  # infeasible | underconstrained | single_mode | multimodal
    positions: Dict[str, np.ndarray] = field(default_factory=dict)
    metrics: EvalMetrics3D = field(default_factory=EvalMetrics3D)
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
                "csr_trr_azimuth": self.metrics.csr_trr_azimuth,
                "csr_trr_elevation": self.metrics.csr_trr_elevation,
                "csr_trr_full": self.metrics.csr_trr_full,
                "K_geom": self.metrics.K_geom,
                "spread": self.metrics.spread,
                "kendall_tau": self.metrics.kendall_tau,
                "nrms": self.metrics.nrms,
                "best_loss": self.metrics.best_loss,
                "n_solutions": self.metrics.n_solutions,
                "cluster_sizes": self.metrics.cluster_sizes,
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
    """从 QRR + TRR 约束重建三维场景。

    同时支持 TRREntry（2D）和 TRR3DEntry（3D）格式的约束字典。
    运行时自动检测：如果约束字典包含 elevation_deg 或 elevation_band 字段，
    则解析为 TRR3DEntry；否则解析为 TRREntry 并提升为 TRR3DEntry。

    参数：
        qrr_constraints: QRR 约束字典列表，键包括
            pair1, pair2, comparator, [weight], [variant], [anchor]
        trr_constraints: TRR 约束字典列表，键包括
            target, ref1, ref2, hour, [weight], [level],
            以及可选的 3D 字段 [elevation_deg], [elevation_band], [elevation_level]
        object_ids: 对象标识符列表
        gt_positions: 可选的真值位置用于评估
        n_restarts: 优化重启次数
        config: 求解器配置

    返回：
        ReconstructResult 包含位置、指标和诊断信息
    """
    if config is None:
        config = SolverConfig(n_restarts=n_restarts)
    else:
        config.n_restarts = n_restarts

    # 解析 QRR 约束字典为 QRREntry 对象
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

    # 解析 TRR 约束字典：运行时检测 2D 或 3D 格式
    trr3d_entries = []
    for c in trr_constraints:
        # 判断是否为 3D 约束（包含仰角字段）
        is_3d = "elevation_deg" in c or "elevation_band" in c
        if is_3d:
            trr3d_entries.append(TRR3DEntry(
                target=c["target"],
                ref1=c["ref1"],
                ref2=c["ref2"],
                hour=c["hour"],
                elevation_deg=c.get("elevation_deg", 0.0),
                elevation_band=c.get("elevation_band", "level"),
                weight=c.get("weight", 1.0),
                level=c.get("level", "hour"),
                elevation_level=c.get("elevation_level", "band"),
            ))
        else:
            # 2D TRR 约束提升为 3D（仰角默认 level/0度）
            trr3d_entries.append(TRR3DEntry(
                target=c["target"],
                ref1=c["ref1"],
                ref2=c["ref2"],
                hour=c["hour"],
                elevation_deg=0.0,
                elevation_band="level",
                weight=c.get("weight", 1.0),
                level=c.get("level", "hour"),
                elevation_level="band",
            ))

    return _run_pipeline(qrr_entries, trr3d_entries, object_ids, gt_positions, config)


def reconstruct_from_scoring(
    scoring_result: dict,
    questions: List[dict],
    gt_positions: Optional[Dict[str, np.ndarray]] = None,
    n_restarts: int = 10,
    use_correct_only: bool = True,
    config: Optional[SolverConfig] = None,
) -> ReconstructResult:
    """从评分输出重建三维场景。

    参数：
        scoring_result: score_batch_scene() 输出，含 per_question 列表
        questions: 含真值元数据的原始问题列表
        gt_positions: 真值位置 {obj_id -> [x, y, z]}
        n_restarts: 优化重启次数
        use_correct_only: 若 True，仅使用正确回答的约束；
                         若 False，使用所有 VLM 预测（信念重建）
        config: 求解器配置
    """
    if config is None:
        config = SolverConfig(n_restarts=n_restarts)
    else:
        config.n_restarts = n_restarts

    # 序列化真值位置为列表格式
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
    """从预处理的场景包重建三维场景。

    参数：
        prepared_input: 预处理的场景输入（对象或字典）
        n_restarts: 优化重启次数
        config: 求解器配置
        constraint_mode: 使用的约束子集：
            "all"       — qrr_direct + qrr_from_fdr + trr（默认）
            "fdr_only"  — 仅 FDR 分解的 QRR
            "qrr_only"  — 仅直接 QRR
            "fdr_qrr"   — qrr_direct + qrr_from_fdr，无 TRR
            "qrr_trr"   — qrr_direct + trr，无 FDR
            "fdr_trr"   — qrr_from_fdr + trr，无直接 QRR
    """
    if constraint_mode not in CONSTRAINT_MODES:
        raise ValueError(f"未知 constraint_mode={constraint_mode!r}，"
                         f"应为 {CONSTRAINT_MODES} 之一")

    if config is None:
        config = SolverConfig(n_restarts=n_restarts)
    else:
        config.n_restarts = n_restarts

    prepared = (
        prepared_input
        if isinstance(prepared_input, PreparedSceneInput)
        else PreparedSceneInput.from_dict(prepared_input)
    )

    # 加载真值位置
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
    trr3d_entries: List[TRR3DEntry],
    object_ids: List[str],
    gt_positions: Optional[Dict[str, np.ndarray]],
    config: SolverConfig,
) -> ReconstructResult:
    """内部管线：阶段 0-5。

    阶段 0-1：符号预处理（偏序 DAG + 角度弧段 + 超图分析）
    阶段 2-3：数值求解（3D L-BFGS-B 优化）
    阶段 4-5：评估（CSR + 聚类 + Kendall tau + NRMS）
    """

    result = ReconstructResult()

    # ── 阶段 0-1：符号预处理 ──
    poset = build_distance_poset(qrr_entries)

    # 对 TRR3DEntry 使用 3D 弧段构建（方位角部分）
    sectors = build_angular_sectors_3d(trr3d_entries)

    # 超图分析需要 TRREntry 接口，将 TRR3DEntry 转换
    trr_for_hyper = [
        TRREntry(
            target=e.target, ref1=e.ref1, ref2=e.ref2,
            hour=e.hour, weight=e.weight, level=e.level,
        )
        for e in trr3d_entries
    ]
    hyper = analyze_hypergraph(qrr_entries, trr_for_hyper, object_ids)
    feasibility = check_feasibility(poset, sectors, hyper)
    result.feasibility_checks = feasibility

    # 对象数不足
    if len(object_ids) < 3:
        result.status = "underconstrained"
        return result

    # 完全无约束
    if not qrr_entries and not trr3d_entries:
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
    # 将真值位置归一化为 3D
    gt_3d = None
    if gt_positions is not None:
        gt_3d = {}
        for oid, pos in gt_positions.items():
            if oid in object_ids:
                gt_3d[oid] = np.array(pos[:3], dtype=np.float64)

    solutions = solve_3d(
        object_ids=object_ids,
        qrr_entries=qrr_entries,
        trr3d_entries=trr3d_entries,
        config=config,
        gt_positions=gt_3d,
    )

    if not solutions:
        result.status = "infeasible"
        return result

    result.all_solutions = solutions
    result.positions = solutions[0].positions  # 最优解

    # ── 阶段 4-5：评估 ──
    metrics = evaluate_reconstruction_3d(
        solutions=solutions,
        qrr_entries=qrr_entries,
        trr3d_entries=trr3d_entries,
        gt_positions=gt_3d,
    )
    result.metrics = metrics
    result.K_geom = metrics.K_geom

    # ── 状态判定 ──
    # 使用综合 CSR（方位角 + 仰角同时满足）作为判据
    csr_ok = metrics.csr_qrr >= 0.95 and metrics.csr_trr_full >= 0.95
    if not csr_ok:
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
