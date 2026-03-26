"""
动态场景生成的运动模型。

纯 Python + math，无 Blender 依赖。
`t` 为整数帧索引；物理时间 = t / fps。
"""

import math
from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np


class MotionModel(ABC):
    """所有运动模型的抽象基类。"""

    @abstractmethod
    def position(self, t: int) -> Tuple[float, float]:
        """返回第 t 帧的 (x, y)。"""

    @abstractmethod
    def velocity(self, t: int) -> Tuple[float, float]:
        """返回第 t 帧的 (vx, vy)。"""

    def trajectory(self, n_frames: int) -> np.ndarray:
        """返回完整轨迹，形状为 (n_frames, 2) 的数组。"""
        return np.array([self.position(t) for t in range(n_frames)])

    @abstractmethod
    def to_dict(self) -> dict:
        """序列化为 JSON 兼容字典。"""

    @classmethod
    def from_dict(cls, d: dict) -> "MotionModel":
        """从字典反序列化。通过 MOTION_REGISTRY 分发。"""
        return motion_from_dict(d)


class StaticMotion(MotionModel):
    """物体保持在固定位置。"""

    def __init__(self, x0: float, y0: float):
        self.x0 = x0
        self.y0 = y0

    def position(self, t: int) -> Tuple[float, float]:
        return (self.x0, self.y0)

    def velocity(self, t: int) -> Tuple[float, float]:
        return (0.0, 0.0)

    def to_dict(self) -> dict:
        return {"type": "static", "x0": self.x0, "y0": self.y0}


class LinearMotion(MotionModel):
    """匀速线性运动：pos(t) = (x0 + vx*t, y0 + vy*t)。"""

    def __init__(self, x0: float, y0: float, vx: float, vy: float):
        self.x0 = x0
        self.y0 = y0
        self.vx = vx
        self.vy = vy

    def position(self, t: int) -> Tuple[float, float]:
        return (self.x0 + self.vx * t, self.y0 + self.vy * t)

    def velocity(self, t: int) -> Tuple[float, float]:
        return (self.vx, self.vy)

    def to_dict(self) -> dict:
        return {
            "type": "linear",
            "x0": self.x0, "y0": self.y0,
            "vx": self.vx, "vy": self.vy,
        }


class CircularMotion(MotionModel):
    """绕 (cx, cy) 的匀速圆周运动。"""

    def __init__(
        self, cx: float, cy: float, radius: float,
        omega: float, phase0: float = 0.0,
    ):
        self.cx = cx
        self.cy = cy
        self.radius = radius
        self.omega = omega  # 每帧弧度数
        self.phase0 = phase0  # 初始相位（弧度）

    def position(self, t: int) -> Tuple[float, float]:
        angle = self.phase0 + self.omega * t
        x = self.cx + self.radius * math.cos(angle)
        y = self.cy + self.radius * math.sin(angle)
        return (x, y)

    def velocity(self, t: int) -> Tuple[float, float]:
        angle = self.phase0 + self.omega * t
        vx = -self.radius * self.omega * math.sin(angle)
        vy = self.radius * self.omega * math.cos(angle)
        return (vx, vy)

    def to_dict(self) -> dict:
        return {
            "type": "circular",
            "cx": self.cx, "cy": self.cy,
            "radius": self.radius,
            "omega": self.omega, "phase0": self.phase0,
        }


class AcceleratedLinearMotion(MotionModel):
    """匀加速线性运动：pos(t) = p0 + v0*t + 0.5*a*t²。"""

    def __init__(self, x0: float, y0: float, vx: float, vy: float, ax: float, ay: float):
        self.x0 = x0
        self.y0 = y0
        self.vx = vx
        self.vy = vy
        self.ax = ax
        self.ay = ay

    def position(self, t: int) -> Tuple[float, float]:
        return (
            self.x0 + self.vx * t + 0.5 * self.ax * t * t,
            self.y0 + self.vy * t + 0.5 * self.ay * t * t,
        )

    def velocity(self, t: int) -> Tuple[float, float]:
        return (self.vx + self.ax * t, self.vy + self.ay * t)

    def to_dict(self) -> dict:
        return {
            "type": "accelerated_linear",
            "x0": self.x0, "y0": self.y0,
            "vx": self.vx, "vy": self.vy,
            "ax": self.ax, "ay": self.ay,
        }


class WaypointMotion(MotionModel):
    """
    经过一系列路点的分段线性运动。

    物体在相邻路点之间以恒定速度运动，
    在均匀间隔的帧时刻到达每个路点。
    """

    def __init__(self, waypoints: list, n_frames: int):
        """
        Args:
            waypoints: (x, y) 元组列表，至少需要 2 个点。
            n_frames: 动画总帧数，用于均匀分配路点间隔。
        """
        self.waypoints = [(float(x), float(y)) for x, y in waypoints]
        self.n_frames = n_frames
        n_segs = len(self.waypoints) - 1
        self._frames_per_seg = n_frames / n_segs if n_segs > 0 else n_frames

    def _segment_interp(self, t: int):
        """返回第 t 帧的插值位置和速度。"""
        n_segs = len(self.waypoints) - 1
        if n_segs <= 0:
            wp = self.waypoints[0]
            return wp, (0.0, 0.0)

        seg_f = t / self._frames_per_seg
        seg_idx = min(int(seg_f), n_segs - 1)
        frac = seg_f - seg_idx

        p0 = self.waypoints[seg_idx]
        p1 = self.waypoints[seg_idx + 1]

        x = p0[0] + (p1[0] - p0[0]) * frac
        y = p0[1] + (p1[1] - p0[1]) * frac

        # 速度 = 该段每帧的位移量
        vx = (p1[0] - p0[0]) / self._frames_per_seg
        vy = (p1[1] - p0[1]) / self._frames_per_seg
        return (x, y), (vx, vy)

    def position(self, t: int) -> Tuple[float, float]:
        pos, _ = self._segment_interp(t)
        return pos

    def velocity(self, t: int) -> Tuple[float, float]:
        _, vel = self._segment_interp(t)
        return vel

    def to_dict(self) -> dict:
        return {
            "type": "waypoint",
            "waypoints": [list(wp) for wp in self.waypoints],
            "n_frames": self.n_frames,
        }


class BounceMotion(MotionModel):
    """
    在矩形边界上弹跳的线性运动。

    物体从 (x0, y0) 以速度 (vx, vy) 出发，碰到边界时弹性反射。
    位置已预计算。
    """

    def __init__(self, x0: float, y0: float, vx: float, vy: float,
                 bounds: float, n_frames: int):
        self.x0 = x0
        self.y0 = y0
        self.vx = vx
        self.vy = vy
        self.bounds = bounds
        self.n_frames = n_frames
        self._positions, self._velocities = self._simulate()

    def _simulate(self):
        positions = []
        velocities = []
        x, y = self.x0, self.y0
        vx, vy = self.vx, self.vy
        b = self.bounds
        for _ in range(self.n_frames):
            positions.append((x, y))
            velocities.append((vx, vy))
            x += vx
            y += vy
            if x > b:
                x = 2 * b - x
                vx = -vx
            elif x < -b:
                x = -2 * b - x
                vx = -vx
            if y > b:
                y = 2 * b - y
                vy = -vy
            elif y < -b:
                y = -2 * b - y
                vy = -vy
        return positions, velocities

    def position(self, t: int) -> Tuple[float, float]:
        if t < len(self._positions):
            return self._positions[t]
        return self._positions[-1]

    def velocity(self, t: int) -> Tuple[float, float]:
        if t < len(self._velocities):
            return self._velocities[t]
        return (0.0, 0.0)

    def to_dict(self) -> dict:
        return {
            "type": "bounce",
            "x0": self.x0, "y0": self.y0,
            "vx": self.vx, "vy": self.vy,
            "bounds": self.bounds,
            "n_frames": self.n_frames,
        }


# ---------------------------------------------------------------------------
# 注册表 / 工厂
# ---------------------------------------------------------------------------

MOTION_REGISTRY: dict = {
    "static": lambda d: StaticMotion(d["x0"], d["y0"]),
    "linear": lambda d: LinearMotion(d["x0"], d["y0"], d["vx"], d["vy"]),
    "circular": lambda d: CircularMotion(
        d["cx"], d["cy"], d["radius"], d["omega"], d.get("phase0", 0.0),
    ),
    "accelerated_linear": lambda d: AcceleratedLinearMotion(
        d["x0"], d["y0"], d["vx"], d["vy"], d["ax"], d["ay"],
    ),
    "waypoint": lambda d: WaypointMotion(
        [tuple(wp) for wp in d["waypoints"]], d["n_frames"],
    ),
    "bounce": lambda d: BounceMotion(
        d["x0"], d["y0"], d["vx"], d["vy"], d["bounds"], d["n_frames"],
    ),
}


def motion_from_dict(d: dict) -> MotionModel:
    """从序列化字典重建 MotionModel。"""
    mtype = d["type"]
    if mtype not in MOTION_REGISTRY:
        raise ValueError(f"Unknown motion type: {mtype}")
    return MOTION_REGISTRY[mtype](d)
