"""
Motion models for dynamic scene generation.

Pure Python + math, no Blender dependency.
`t` is an integer frame index; physical time = t / fps.
"""

import math
from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np


class MotionModel(ABC):
    """Abstract base class for all motion models."""

    @abstractmethod
    def position(self, t: int) -> Tuple[float, float]:
        """Return (x, y) at frame t."""

    @abstractmethod
    def velocity(self, t: int) -> Tuple[float, float]:
        """Return (vx, vy) at frame t."""

    def trajectory(self, n_frames: int) -> np.ndarray:
        """Return full trajectory as (n_frames, 2) array."""
        return np.array([self.position(t) for t in range(n_frames)])

    @abstractmethod
    def to_dict(self) -> dict:
        """Serialize to JSON-compatible dict."""

    @classmethod
    def from_dict(cls, d: dict) -> "MotionModel":
        """Deserialize from dict. Dispatches via MOTION_REGISTRY."""
        return motion_from_dict(d)


class StaticMotion(MotionModel):
    """Object stays at a fixed position."""

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
    """Uniform linear motion: pos(t) = (x0 + vx*t, y0 + vy*t)."""

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
    """Uniform circular motion around (cx, cy)."""

    def __init__(
        self, cx: float, cy: float, radius: float,
        omega: float, phase0: float = 0.0,
    ):
        self.cx = cx
        self.cy = cy
        self.radius = radius
        self.omega = omega  # radians per frame
        self.phase0 = phase0  # initial phase in radians

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
    """Linear motion with constant acceleration: pos(t) = p0 + v0*t + 0.5*a*t²."""

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
    Piecewise-linear motion through a sequence of waypoints.

    The object moves at constant speed between consecutive waypoints,
    reaching each one at evenly-spaced frame intervals.
    """

    def __init__(self, waypoints: list, n_frames: int):
        """
        Args:
            waypoints: List of (x, y) tuples. Must have at least 2 points.
            n_frames: Total animation frames. Used to space waypoints evenly.
        """
        self.waypoints = [(float(x), float(y)) for x, y in waypoints]
        self.n_frames = n_frames
        n_segs = len(self.waypoints) - 1
        self._frames_per_seg = n_frames / n_segs if n_segs > 0 else n_frames

    def _segment_interp(self, t: int):
        """Return interpolated position and velocity at frame t."""
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

        # Velocity = displacement per frame along this segment
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
    Linear motion that bounces off rectangular bounds.

    The object starts at (x0, y0) with velocity (vx, vy) and reflects
    elastically when hitting the boundary.  Positions are precomputed.
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
# Registry / factory
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
    """Reconstruct a MotionModel from its serialized dict."""
    mtype = d["type"]
    if mtype not in MOTION_REGISTRY:
        raise ValueError(f"Unknown motion type: {mtype}")
    return MOTION_REGISTRY[mtype](d)
