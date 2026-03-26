"""
QRR（量化相对关系）和 TRR（三元钟面关系）。

QRR：跨四个物体比较成对度量。
TRR：使用钟面方向的方向关系。
"""

from dataclasses import dataclass
from enum import Enum
from itertools import combinations, permutations
from typing import Tuple, List, Dict, Any, Optional
import logging
import math
import numpy as np

logger = logging.getLogger(__name__)

from .comparators import Comparator, compare


class MetricType(Enum):
    DIST_3D = "dist3D"
    DIST_2D = "dist2D"
    DEPTH_GAP = "depthGap"
    SIZE_RATIO = "sizeRatio"

    def __str__(self) -> str:
        return self.value

    @classmethod
    def from_string(cls, s: str) -> "MetricType":
        mapping = {
            "dist3d": cls.DIST_3D, "dist3D": cls.DIST_3D, "dist_3d": cls.DIST_3D,
            "dist2d": cls.DIST_2D, "dist2D": cls.DIST_2D, "dist_2d": cls.DIST_2D,
            "depthgap": cls.DEPTH_GAP, "depthGap": cls.DEPTH_GAP, "depth_gap": cls.DEPTH_GAP,
            "sizeratio": cls.SIZE_RATIO, "sizeRatio": cls.SIZE_RATIO, "size_ratio": cls.SIZE_RATIO,
        }
        if s in mapping:
            return mapping[s]
        raise ValueError(f"Unknown metric type: {s}")


@dataclass
class QRRConstraint:
    pair1: Tuple[str, str]
    pair2: Tuple[str, str]
    metric: MetricType
    comparator: Comparator
    variant: str = "disjoint"
    anchor: Optional[str] = None

    def __post_init__(self):
        self.pair1 = tuple(sorted(self.pair1))
        self.pair2 = tuple(sorted(self.pair2))
        if self.variant == "shared_anchor" and self.anchor is None:
            shared = set(self.pair1) & set(self.pair2)
            if len(shared) == 1:
                object.__setattr__(self, "anchor", next(iter(shared)))

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "pair1": list(self.pair1),
            "pair2": list(self.pair2),
            "metric": str(self.metric),
            "comparator": str(self.comparator),
            "variant": self.variant,
        }
        if self.anchor is not None:
            data["anchor"] = self.anchor
        return data

    def canonical_key(self) -> Tuple:
        if self.pair1 < self.pair2:
            return (self.pair1, self.pair2, self.metric, self.variant)
        else:
            return (self.pair2, self.pair1, self.metric, self.variant)


@dataclass
class TRRConstraint:
    target: str
    ref1: str
    ref2: str
    hour: int
    quadrant: int = 0
    angle_deg: float = 0.0

    def __post_init__(self):
        if not 1 <= self.hour <= 12:
            raise ValueError(f"Hour must be 1-12, got {self.hour}")
        if self.quadrant == 0:
            object.__setattr__(self, 'quadrant', hour_to_quadrant(self.hour))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target": self.target,
            "ref1": self.ref1,
            "ref2": self.ref2,
            "hour": self.hour,
            "quadrant": self.quadrant,
            "angle_deg": self.angle_deg,
        }


@dataclass
class FDRConstraint:
    """全距离排序：所有物体按距锚点的距离排序。"""
    anchor: str
    ranking: List[str]          # object IDs, nearest to farthest
    distances: List[float]      # corresponding distances
    tie_groups: List[List[str]] # groups of objects within tau tolerance

    def to_dict(self) -> Dict[str, Any]:
        return {
            "anchor": self.anchor,
            "ranking": self.ranking,
            "distances": [round(d, 6) for d in self.distances],
            "tie_groups": self.tie_groups,
        }


def hour_to_quadrant(hour: int) -> int:
    """将小时数（1-12）转换为象限（1-4）。"""
    if hour in (12, 1, 2):
        return 1
    elif hour in (3, 4, 5):
        return 2
    elif hour in (6, 7, 8):
        return 3
    else:
        return 4


def angle_to_hour(angle_deg: float) -> int:
    """将角度（度）转换为钟面小时数（1-12）。"""
    angle_deg = angle_deg % 360
    shifted = (angle_deg + 15) % 360
    hour_idx = int(shifted // 30)
    hour = (hour_idx % 12)
    if hour == 0:
        hour = 12
    return hour


def compute_angle_2d(
    target_pos: np.ndarray,
    ref1_pos: np.ndarray,
    ref2_pos: np.ndarray
) -> float:
    """计算从 ref1 看向 ref2 的方向为参考轴，target 的顺时针钟面角度。"""
    ref_vec = ref2_pos - ref1_pos
    ref_angle = math.atan2(ref_vec[1], ref_vec[0])
    target_vec = target_pos - ref1_pos
    if np.linalg.norm(target_vec) < 1e-10:
        return 0.0
    target_angle = math.atan2(target_vec[1], target_vec[0])
    rel_angle = target_angle - ref_angle
    angle_deg = (-math.degrees(rel_angle)) % 360  # clockwise, matching real clock faces
    return angle_deg


# ── 度量计算函数 ──

def compute_dist_3d(obj_a: Dict, obj_b: Dict) -> float:
    """计算两个物体的三维欧氏距离。"""
    pos_a = np.array(obj_a.get("position_3d", obj_a.get("3d_coords", [0, 0, 0])))
    pos_b = np.array(obj_b.get("position_3d", obj_b.get("3d_coords", [0, 0, 0])))
    return float(np.linalg.norm(pos_a - pos_b))


def compute_dist_2d(obj_a: Dict, obj_b: Dict) -> float:
    """计算两个物体的二维像素距离。"""
    pos_a = np.array(obj_a.get("position_2d", obj_a.get("pixel_coords", [0, 0])[:2]))
    pos_b = np.array(obj_b.get("position_2d", obj_b.get("pixel_coords", [0, 0])[:2]))
    return float(np.linalg.norm(pos_a - pos_b))


def compute_depth_gap(obj_a: Dict, obj_b: Dict) -> float:
    """计算两个物体的深度差（绝对值）。"""
    depth_a = obj_a.get("depth", obj_a.get("pixel_coords", [0, 0, 0])[2] if len(obj_a.get("pixel_coords", [])) > 2 else 0)
    depth_b = obj_b.get("depth", obj_b.get("pixel_coords", [0, 0, 0])[2] if len(obj_b.get("pixel_coords", [])) > 2 else 0)
    return abs(float(depth_a) - float(depth_b))


def compute_size_ratio(obj_a: Dict, obj_b: Dict) -> float:
    """计算两个物体的尺寸比（size_a / size_b）。"""
    size_a = obj_a.get("size", 1.0)
    size_b = obj_b.get("size", 1.0)
    size_map = {"large": 0.7, "medium": 0.5, "small": 0.35}
    if isinstance(size_a, str):
        size_a = size_map.get(size_a.lower(), 0.5)
    if isinstance(size_b, str):
        size_b = size_map.get(size_b.lower(), 0.5)
    if size_b <= 0:
        logger.warning("SIZE_RATIO: size_b=%s is non-positive, returning 1.0", size_b)
        return 1.0
    return float(size_a) / float(size_b)


METRIC_FUNCTIONS = {
    MetricType.DIST_3D: compute_dist_3d,
    MetricType.DIST_2D: compute_dist_2d,
    MetricType.DEPTH_GAP: compute_depth_gap,
    MetricType.SIZE_RATIO: compute_size_ratio,
}


def _is_boundary(m1: float, m2: float, tau: float) -> bool:
    """检查比较是否落在容差边界附近（不稳定的真值）。"""
    max_val = max(m1, m2)
    if max_val == 0:
        return False
    threshold = tau * max_val
    diff = abs(m1 - m2)
    return diff > 0.8 * threshold and diff < 1.2 * threshold


def compute_qrr(
    objects: Dict[str, Dict],
    pair1: Tuple[str, str],
    pair2: Tuple[str, str],
    metric: MetricType,
    tau: float = 0.10,
    variant: str = "disjoint",
    anchor: Optional[str] = None,
) -> QRRConstraint:
    """计算单个 QRR 约束（两对物体的度量比较）。"""
    metric_func = METRIC_FUNCTIONS[metric]
    m1 = metric_func(objects[pair1[0]], objects[pair1[1]])
    m2 = metric_func(objects[pair2[0]], objects[pair2[1]])
    comparator = compare(m1, m2, tau)

    return QRRConstraint(
        pair1=pair1, pair2=pair2, metric=metric,
        comparator=comparator,
        variant=variant, anchor=anchor,
    )


def compute_trr(
    objects: Dict[str, Dict],
    target: str, ref1: str, ref2: str,
    use_3d: bool = False
) -> TRRConstraint:
    """计算单个 TRR 约束（目标物体相对于参考轴的钟面方向）。"""
    if use_3d:
        target_pos = np.array(objects[target].get("position_3d", [0, 0, 0])[:2])
        ref1_pos = np.array(objects[ref1].get("position_3d", [0, 0, 0])[:2])
        ref2_pos = np.array(objects[ref2].get("position_3d", [0, 0, 0])[:2])
    else:
        target_pos = np.array(objects[target].get("position_2d", [0, 0])[:2])
        ref1_pos = np.array(objects[ref1].get("position_2d", [0, 0])[:2])
        ref2_pos = np.array(objects[ref2].get("position_2d", [0, 0])[:2])

    angle_deg = compute_angle_2d(target_pos, ref1_pos, ref2_pos)
    hour = angle_to_hour(angle_deg)
    quadrant = hour_to_quadrant(hour)

    return TRRConstraint(
        target=target, ref1=ref1, ref2=ref2,
        hour=hour, quadrant=quadrant, angle_deg=angle_deg,
    )


def extract_all_qrr(
    objects: Dict[str, Dict],
    metric: MetricType,
    tau: float = 0.10,
    disjoint_only: bool = True
) -> List[QRRConstraint]:
    """提取所有 QRR 约束（disjoint 不相交对变体）。"""
    obj_ids = list(objects.keys())
    pairs = list(combinations(obj_ids, 2))
    metric_func = METRIC_FUNCTIONS[metric]
    constraints = []
    boundary_skipped = 0

    for i, pair1 in enumerate(pairs):
        for pair2 in pairs[i + 1:]:
            if disjoint_only:
                if set(pair1) & set(pair2):
                    continue
            # 跳过边界情况（真值不稳定）
            m1 = metric_func(objects[pair1[0]], objects[pair1[1]])
            m2 = metric_func(objects[pair2[0]], objects[pair2[1]])
            if _is_boundary(m1, m2, tau):
                boundary_skipped += 1
                continue
            constraint = compute_qrr(
                objects, pair1, pair2, metric, tau,
                variant="disjoint" if disjoint_only else "general",
            )
            constraints.append(constraint)

    if boundary_skipped:
        logger.debug("extract_all_qrr: skipped %d boundary questions (tau=%.2f)", boundary_skipped, tau)
    return constraints


def extract_all_qrr_shared_anchor(
    objects: Dict[str, Dict],
    metric: MetricType,
    tau: float = 0.10,
) -> List[QRRConstraint]:
    """提取所有共享锚点变体的 QRR 约束。"""
    obj_ids = sorted(objects.keys())
    metric_func = METRIC_FUNCTIONS[metric]
    constraints = []
    boundary_skipped = 0

    for anchor in obj_ids:
        others = [oid for oid in obj_ids if oid != anchor]
        for obj_a, obj_b in combinations(others, 2):
            pair1 = (anchor, obj_a)
            pair2 = (anchor, obj_b)
            m1 = metric_func(objects[pair1[0]], objects[pair1[1]])
            m2 = metric_func(objects[pair2[0]], objects[pair2[1]])
            if _is_boundary(m1, m2, tau):
                boundary_skipped += 1
                continue
            constraint = compute_qrr(
                objects, pair1, pair2, metric, tau,
                variant="shared_anchor", anchor=anchor,
            )
            constraints.append(constraint)

    if boundary_skipped:
        logger.debug("extract_all_qrr_shared_anchor: skipped %d boundary questions (tau=%.2f)", boundary_skipped, tau)
    return constraints


def extract_all_trr(
    objects: Dict[str, Dict],
    use_3d: bool = False
) -> List[TRRConstraint]:
    """提取所有 TRR 约束（N 个物体的全排列三元组）。"""
    obj_ids = list(objects.keys())
    constraints = []
    for triple in permutations(obj_ids, 3):
        target, ref1, ref2 = triple
        constraint = compute_trr(objects, target, ref1, ref2, use_3d)
        constraints.append(constraint)
    return constraints


def compute_fdr(
    objects: Dict[str, Dict],
    anchor: str,
    tau: float = 0.10,
) -> FDRConstraint:
    """计算从锚点到所有其他物体的全距离排序。"""
    metric_func = METRIC_FUNCTIONS[MetricType.DIST_3D]
    other_ids = [oid for oid in sorted(objects.keys()) if oid != anchor]

    dist_pairs = []
    for oid in other_ids:
        d = metric_func(objects[anchor], objects[oid])
        dist_pairs.append((oid, d))

    # 按距离升序排列，距离相同则按 ID 排序
    dist_pairs.sort(key=lambda x: (x[1], x[0]))

    ranking = [oid for oid, _ in dist_pairs]
    distances = [d for _, d in dist_pairs]

    # 基于容差计算并列组
    tie_groups: List[List[str]] = []
    if ranking:
        current_group = [ranking[0]]
        for i in range(1, len(ranking)):
            cmp = compare(distances[i - 1], distances[i], tau)
            if cmp == Comparator.APPROX:
                current_group.append(ranking[i])
            else:
                tie_groups.append(current_group)
                current_group = [ranking[i]]
        tie_groups.append(current_group)

    return FDRConstraint(
        anchor=anchor,
        ranking=ranking,
        distances=distances,
        tie_groups=tie_groups,
    )


def extract_all_fdr(
    objects: Dict[str, Dict],
    tau: float = 0.10,
) -> List[FDRConstraint]:
    """为每个物体提取一个 FDR 约束（以该物体为锚点），共 N 个约束。"""
    obj_ids = sorted(objects.keys())
    return [compute_fdr(objects, anchor, tau) for anchor in obj_ids]
