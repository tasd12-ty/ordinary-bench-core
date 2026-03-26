"""
统一求解器调度接口。

根据 method 参数选择不同的求解器实现：
  - "lbfgsb" (默认): 原始 L-BFGS-B + softplus loss
  - "sdp": SDP 松弛 (Gram 矩阵凸优化)
  - "hinge": Hinge loss + 差分进化全局搜索
"""

from typing import Dict, List, Optional

import numpy as np

from .constraints import QRREntry, TRREntry
from .solver import SolverConfig, SolverSolution, solve as solve_lbfgsb


METHODS = ("lbfgsb", "sdp", "hinge")


def solve(
    object_ids: List[str],
    qrr_entries: List[QRREntry],
    trr_entries: List[TRREntry],
    config: Optional[SolverConfig] = None,
    gt_positions: Optional[Dict[str, np.ndarray]] = None,
    method: str = "lbfgsb",
) -> List[SolverSolution]:
    """统一求解器入口。

    Args:
        method: "lbfgsb" | "sdp" | "hinge"
    """
    if method not in METHODS:
        raise ValueError(f"Unknown method={method!r}, expected one of {METHODS}")

    if method == "sdp":
        from .solver_sdp import solve_sdp
        return solve_sdp(object_ids, qrr_entries, trr_entries, config, gt_positions)
    elif method == "hinge":
        from .solver_hinge import solve_hinge
        return solve_hinge(object_ids, qrr_entries, trr_entries, config, gt_positions)
    else:
        return solve_lbfgsb(object_ids, qrr_entries, trr_entries, config, gt_positions)
