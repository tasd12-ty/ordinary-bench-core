"""序数空间关系 DSL 模块。"""

from .comparators import Comparator, compare
from .predicates import (
  MetricType, QRRConstraint, TRRConstraint,
  compute_qrr, compute_trr,
  extract_all_qrr, extract_all_qrr_shared_anchor, extract_all_trr,
)
