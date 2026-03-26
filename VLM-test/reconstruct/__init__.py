"""
场景信念重建模块。

根据 VLM 序数空间判断重建 2D 空间配置。
"""

from .pipeline import (
    reconstruct,
    reconstruct_from_prepared,
    reconstruct_from_scoring,
    ReconstructResult,
    CONSTRAINT_MODES,
)
from .preparation import (
    PreparedSceneInput,
    prepare_reconstruction_input_from_scoring,
    load_questions_auto,
    load_scene_gt_positions,
)
from .solver import SolverConfig, SolverSolution
from .evaluate import EvalMetrics
from .constraints import (
    QRREntry,
    TRREntry,
    FeasibilityReport,
    detect_fdr_qrr_conflicts,
)

__all__ = [
    "reconstruct",
    "reconstruct_from_prepared",
    "reconstruct_from_scoring",
    "ReconstructResult",
    "PreparedSceneInput",
    "prepare_reconstruction_input_from_scoring",
    "load_questions_auto",
    "load_scene_gt_positions",
    "SolverConfig",
    "SolverSolution",
    "EvalMetrics",
    "QRREntry",
    "TRREntry",
    "FeasibilityReport",
    "CONSTRAINT_MODES",
    "detect_fdr_qrr_conflicts",
]
