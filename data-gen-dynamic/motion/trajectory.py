"""
轨迹规划：分配运动、检查碰撞、plan_scene()。
"""

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .models import (
    MotionModel,
    StaticMotion,
    LinearMotion,
    CircularMotion,
    AcceleratedLinearMotion,
    WaypointMotion,
    BounceMotion,
    motion_from_dict,
)


@dataclass
class ObjectMotionPlan:
    obj_id: str
    shape: str
    size: str
    size_radius: float
    material: str
    color: str
    rotation: float
    motion: MotionModel
    positions: Optional[List[Tuple[float, float]]] = None
    velocities: Optional[List[Tuple[float, float]]] = None

    def precompute(self, n_frames: int) -> None:
        """预计算所有帧的位置和速度。"""
        self.positions = [self.motion.position(t) for t in range(n_frames)]
        self.velocities = [self.motion.velocity(t) for t in range(n_frames)]

    def to_dict(self) -> dict:
        d = {
            "obj_id": self.obj_id,
            "shape": self.shape,
            "size": self.size,
            "size_radius": self.size_radius,
            "material": self.material,
            "color": self.color,
            "rotation": self.rotation,
            "motion": self.motion.to_dict(),
        }
        if self.positions is not None:
            d["positions"] = [list(p) for p in self.positions]
        if self.velocities is not None:
            d["velocities"] = [list(v) for v in self.velocities]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ObjectMotionPlan":
        plan = cls(
            obj_id=d["obj_id"],
            shape=d["shape"],
            size=d["size"],
            size_radius=d["size_radius"],
            material=d["material"],
            color=d["color"],
            rotation=d["rotation"],
            motion=motion_from_dict(d["motion"]),
        )
        if "positions" in d:
            plan.positions = [tuple(p) for p in d["positions"]]
        if "velocities" in d:
            plan.velocities = [tuple(v) for v in d["velocities"]]
        return plan


def check_trajectory_collisions(
    plans: List[ObjectMotionPlan],
    n_frames: int,
    min_dist: float = 0.25,
    bounds: float = 3.5,
) -> bool:
    """
    若所有轨迹均无碰撞且在边界内则返回 True。

    逐帧检查：
    - 两两距离 > min_dist + 半径之和
    - 所有位置在 [-bounds, bounds] 内
    """
    n = len(plans)
    for t in range(n_frames):
        positions = []
        for p in plans:
            x, y = p.motion.position(t)
            # 边界检查
            if abs(x) > bounds or abs(y) > bounds:
                return False
            positions.append((x, y, p.size_radius))

        # 两两距离检查
        for i in range(n):
            for j in range(i + 1, n):
                dx = positions[i][0] - positions[j][0]
                dy = positions[i][1] - positions[j][1]
                dist = math.sqrt(dx * dx + dy * dy)
                if dist < min_dist + positions[i][2] + positions[j][2]:
                    return False
    return True


def generate_random_motion(
    x0: float,
    y0: float,
    motion_type: str,
    n_frames: int,
    bounds: float,
    rng: random.Random,
    motion_params: Optional[Dict[str, float]] = None,
) -> MotionModel:
    """为指定运动类型生成具有合理参数的运动模型。"""
    if motion_params is None:
        motion_params = {}

    if motion_type == "static":
        return StaticMotion(x0, y0)

    elif motion_type == "linear":
        speed = rng.uniform(
            motion_params.get("speed_min", 0.02),
            motion_params.get("speed_max", 0.08),
        )
        angle = rng.uniform(0, 2 * math.pi)
        vx = speed * math.cos(angle)
        vy = speed * math.sin(angle)
        return LinearMotion(x0, y0, vx, vy)

    elif motion_type == "circular":
        radius = rng.uniform(
            motion_params.get("radius_min", 0.5),
            motion_params.get("radius_max", 1.5),
        )
        omega = rng.choice([-1, 1]) * rng.uniform(
            motion_params.get("omega_min", 0.03),
            motion_params.get("omega_max", 0.10),
        )
        phase0 = rng.uniform(0, 2 * math.pi)
        # 圆心偏移使起始位置恰好在 (x0, y0)
        cx = x0 - radius * math.cos(phase0)
        cy = y0 - radius * math.sin(phase0)
        return CircularMotion(cx, cy, radius, omega, phase0)

    elif motion_type == "accelerated_linear":
        speed = rng.uniform(
            motion_params.get("speed_min", 0.02),
            motion_params.get("speed_max", 0.08),
        )
        angle = rng.uniform(0, 2 * math.pi)
        vx = speed * math.cos(angle)
        vy = speed * math.sin(angle)
        accel = rng.uniform(
            motion_params.get("accel_min", 0.0001),
            motion_params.get("accel_max", 0.0005),
        )
        accel_angle = rng.uniform(0, 2 * math.pi)
        ax = accel * math.cos(accel_angle)
        ay = accel * math.sin(accel_angle)
        return AcceleratedLinearMotion(x0, y0, vx, vy, ax, ay)

    elif motion_type == "waypoint":
        n_waypoints = rng.randint(3, 6)
        waypoints = [(x0, y0)]
        for _ in range(n_waypoints - 1):
            wx = rng.uniform(-bounds + 0.5, bounds - 0.5)
            wy = rng.uniform(-bounds + 0.5, bounds - 0.5)
            waypoints.append((wx, wy))
        return WaypointMotion(waypoints, n_frames)

    elif motion_type == "bounce":
        speed = rng.uniform(
            motion_params.get("speed_min", 0.02),
            motion_params.get("speed_max", 0.08),
        )
        angle = rng.uniform(0, 2 * math.pi)
        vx = speed * math.cos(angle)
        vy = speed * math.sin(angle)
        return BounceMotion(x0, y0, vx, vy, bounds, n_frames)

    raise ValueError(f"Unknown motion type: {motion_type}")


def _choose_motion_type(
    motion_mix: Dict[str, float], rng: random.Random,
) -> str:
    """从混合字典中按权重随机选择运动类型。"""
    types = list(motion_mix.keys())
    weights = [motion_mix[t] for t in types]
    return rng.choices(types, weights=weights, k=1)[0]


def count_qrr_reversals(
    plans: List[ObjectMotionPlan],
    n_frames: int,
    tau: float = 0.10,
) -> int:
    """统计整条轨迹上所有锚点/配对组合的 QRR 反转总次数。"""
    n = len(plans)
    if n < 3:
        return 0

    # 预计算所有位置
    all_pos = []
    for t in range(n_frames):
        frame_pos = {}
        for p in plans:
            if p.positions is not None:
                frame_pos[p.obj_id] = p.positions[t]
            else:
                frame_pos[p.obj_id] = p.motion.position(t)
        all_pos.append(frame_pos)

    ids = [p.obj_id for p in plans]
    reversals = 0

    for a_idx in range(n):
        anchor = ids[a_idx]
        others = [oid for oid in ids if oid != anchor]
        for i in range(len(others)):
            for j in range(i + 1, len(others)):
                b, c = others[i], others[j]
                prev_diff = None
                for t in range(n_frames):
                    pa = all_pos[t][anchor]
                    pb = all_pos[t][b]
                    pc = all_pos[t][c]
                    d_ab = math.sqrt((pa[0]-pb[0])**2 + (pa[1]-pb[1])**2)
                    d_ac = math.sqrt((pa[0]-pc[0])**2 + (pa[1]-pc[1])**2)
                    diff = d_ab - d_ac
                    if prev_diff is not None:
                        if (abs(prev_diff) > tau and abs(diff) > tau
                                and prev_diff * diff < 0):
                            reversals += 1
                    prev_diff = diff

    return reversals


def plan_scene(
    n_objects: int,
    n_frames: int,
    properties: dict,
    motion_mix: Optional[Dict[str, float]] = None,
    bounds: float = 3.0,
    min_dist: float = 0.25,
    seed: int = 0,
    max_retries: int = 500,
    motion_params: Optional[Dict[str, float]] = None,
    n_moving: Optional[int] = None,
    min_reversals: Optional[int] = None,
) -> List[ObjectMotionPlan]:
    """
    为动态场景生成完整运动规划。

    Args:
        n_objects: 放置的物体数量。
        n_frames: 动画总帧数。
        properties: 来自 properties.json 的字典（形状、颜色、材质、尺寸）。
        motion_mix: 概率权重，如 {"static": 0.2, "linear": 0.6, "circular": 0.2}。
        bounds: 放置区域的半宽度。
        min_dist: 物体表面间的最小间距。
        seed: 随机种子。
        max_retries: 抛出异常前的最大尝试次数。
        motion_params: 运动生成的速度/角速度参数。
        n_moving: 若设置，则恰好该数量的物体运动，其余静止。
                  运动物体从排除 'static' 的 motion_mix 中采样。
        min_reversals: 若设置，则拒绝 QRR 反转次数不足的场景。

    Returns:
        每个物体的 ObjectMotionPlan 列表（含预计算位置）。
    """
    if motion_mix is None:
        motion_mix = {"static": 0.2, "linear": 0.6, "circular": 0.2}

    rng = random.Random(seed)

    shapes = list(properties["shapes"].items())  # [(名称, blend文件名), ...]
    colors = list(properties["colors"].keys())
    materials = list(properties["materials"].keys())
    sizes = list(properties["sizes"].items())  # [(名称, 半径), ...]

    # 构建仅含运动类型的混合比例（排除 static），用于 n_moving 模式
    if n_moving is not None:
        moving_mix = {k: v for k, v in motion_mix.items() if k != "static"}
        if not moving_mix:
            moving_mix = {"linear": 1.0}
    else:
        moving_mix = None

    for attempt in range(max_retries):
        plans: List[ObjectMotionPlan] = []

        # 当 n_moving 已设置时，确定哪些物体运动
        if n_moving is not None:
            n_mov = min(n_moving, n_objects)
            moving_indices = set(rng.sample(range(n_objects), n_mov))
        else:
            moving_indices = None

        # 在随机初始位置放置物体
        for i in range(n_objects):
            shape_name, _ = rng.choice(shapes)
            color = rng.choice(colors)
            material = rng.choice(materials)
            size_name, size_radius = rng.choice(sizes)
            rotation = rng.uniform(0, 360)

            # 调整立方体的半径（与 render_multiview.py 保持一致）
            effective_radius = size_radius
            if shape_name == "cube":
                effective_radius = size_radius / math.sqrt(2)

            # 使物体远离边缘；边距取决于帧数
            margin = min(bounds * 0.5, max(0.5, n_frames * 0.003))
            x0 = rng.uniform(-bounds + margin, bounds - margin)
            y0 = rng.uniform(-bounds + margin, bounds - margin)

            # 选择运动类型
            if moving_indices is not None:
                if i in moving_indices:
                    mtype = _choose_motion_type(moving_mix, rng)
                else:
                    mtype = "static"
            else:
                mtype = _choose_motion_type(motion_mix, rng)

            motion = generate_random_motion(x0, y0, mtype, n_frames, bounds, rng, motion_params)

            plans.append(ObjectMotionPlan(
                obj_id=f"obj_{i}",
                shape=shape_name,
                size=size_name,
                size_radius=effective_radius,
                material=material,
                color=color,
                rotation=rotation,
                motion=motion,
            ))

        if not check_trajectory_collisions(plans, n_frames, min_dist, bounds):
            continue

        # 预计算位置/速度
        for p in plans:
            p.precompute(n_frames)

        # 若需要则检查 min_reversals
        if min_reversals is not None:
            rev_count = count_qrr_reversals(plans, n_frames)
            if rev_count < min_reversals:
                continue

        return plans

    raise RuntimeError(
        f"Failed to generate valid scene after {max_retries} attempts "
        f"(n_objects={n_objects}, n_frames={n_frames})"
    )
