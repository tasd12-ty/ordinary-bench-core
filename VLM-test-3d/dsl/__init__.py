"""三维空间推理 DSL 模块。

扩展原有 QRR/TRR/FDR 约束体系，新增三维 TRR 约束（方位角 + 仰角）。
"""

from .comparators import Comparator, compare
from .predicates import (
    # 度量类型与计算
    MetricType, METRIC_FUNCTIONS,
    compute_dist_3d, compute_dist_2d, compute_depth_gap, compute_size_ratio,
    # QRR 约束
    QRRConstraint, compute_qrr,
    extract_all_qrr, extract_all_qrr_shared_anchor,
    # TRR 2D 约束（保留兼容）
    TRRConstraint, compute_trr, extract_all_trr,
    compute_angle_2d, angle_to_hour, hour_to_quadrant,
    # TRR 3D 约束（新增）
    TRR3DConstraint, compute_trr_3d, extract_all_trr_3d,
    classify_elevation, elevation_bands_adjacent,
    ELEVATION_BANDS, ELEVATION_BAND_ORDER,
    # FDR 约束
    FDRConstraint, compute_fdr, extract_all_fdr,
    # 工具函数
    _is_boundary,
)
