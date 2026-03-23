"""三维场景信念重建模块。

公开 API：
  - reconstruct() — 从原始约束列表重建
  - reconstruct_from_prepared() — 从预处理包重建
  - reconstruct_from_scoring() — 从 VLM 评分结果重建
"""

from .pipeline import (
    reconstruct,
    reconstruct_from_prepared,
    reconstruct_from_scoring,
    ReconstructResult,
    CONSTRAINT_MODES,
)
from .preparation import PreparedSceneInput
from .solver import SolverConfig, SolverSolution, solve_3d
from .evaluate import EvalMetrics3D, evaluate_reconstruction_3d
from .constraints import (
    QRREntry, TRREntry, TRR3DEntry, FDREntry,
    FeasibilityReport,
    detect_fdr_qrr_conflicts,
)
