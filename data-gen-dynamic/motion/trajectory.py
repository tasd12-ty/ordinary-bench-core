"""
Trajectory planning: assign motions, check collisions, plan_scene().
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
        """Precompute positions and velocities for all frames."""
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
    Return True if all trajectories are collision-free and within bounds.

    Checks every frame for:
    - pairwise distances > min_dist + sum of radii
    - all positions within [-bounds, bounds]
    """
    n = len(plans)
    for t in range(n_frames):
        positions = []
        for p in plans:
            x, y = p.motion.position(t)
            # Bounds check
            if abs(x) > bounds or abs(y) > bounds:
                return False
            positions.append((x, y, p.size_radius))

        # Pairwise distance check
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
    """Generate a motion model with reasonable parameters for the given type."""
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
        # Center is offset so starting position = (x0, y0)
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
    """Weighted random choice of motion type from mix dict."""
    types = list(motion_mix.keys())
    weights = [motion_mix[t] for t in types]
    return rng.choices(types, weights=weights, k=1)[0]


def count_qrr_reversals(
    plans: List[ObjectMotionPlan],
    n_frames: int,
    tau: float = 0.10,
) -> int:
    """Count total QRR reversals across all anchor/pair combos over the trajectory."""
    n = len(plans)
    if n < 3:
        return 0

    # Precompute all positions
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
    Generate a complete motion plan for a dynamic scene.

    Args:
        n_objects: Number of objects to place.
        n_frames: Total number of animation frames.
        properties: Dict from properties.json (shapes, colors, materials, sizes).
        motion_mix: Probability weights, e.g. {"static": 0.2, "linear": 0.6, "circular": 0.2}.
        bounds: Half-width of the placement area.
        min_dist: Minimum gap between object surfaces.
        seed: Random seed.
        max_retries: Max attempts before raising.
        motion_params: Speed/omega parameters for motion generation.
        n_moving: If set, exactly this many objects will move; rest are static.
                  Moving objects sample from motion_mix excluding 'static'.
        min_reversals: If set, reject scenes with fewer QRR reversals.

    Returns:
        List of ObjectMotionPlan for each object (with precomputed positions).
    """
    if motion_mix is None:
        motion_mix = {"static": 0.2, "linear": 0.6, "circular": 0.2}

    rng = random.Random(seed)

    shapes = list(properties["shapes"].items())  # [(name, blend_name), ...]
    colors = list(properties["colors"].keys())
    materials = list(properties["materials"].keys())
    sizes = list(properties["sizes"].items())  # [(name, radius), ...]

    # Build moving-only mix (exclude static) for n_moving mode
    if n_moving is not None:
        moving_mix = {k: v for k, v in motion_mix.items() if k != "static"}
        if not moving_mix:
            moving_mix = {"linear": 1.0}
    else:
        moving_mix = None

    for attempt in range(max_retries):
        plans: List[ObjectMotionPlan] = []

        # Determine which objects move when n_moving is set
        if n_moving is not None:
            n_mov = min(n_moving, n_objects)
            moving_indices = set(rng.sample(range(n_objects), n_mov))
        else:
            moving_indices = None

        # Place objects at random initial positions
        for i in range(n_objects):
            shape_name, _ = rng.choice(shapes)
            color = rng.choice(colors)
            material = rng.choice(materials)
            size_name, size_radius = rng.choice(sizes)
            rotation = rng.uniform(0, 360)

            # Adjust radius for cubes (same as render_multiview.py)
            effective_radius = size_radius
            if shape_name == "cube":
                effective_radius = size_radius / math.sqrt(2)

            # Keep objects away from edges; margin depends on how many frames
            margin = min(bounds * 0.5, max(0.5, n_frames * 0.003))
            x0 = rng.uniform(-bounds + margin, bounds - margin)
            y0 = rng.uniform(-bounds + margin, bounds - margin)

            # Choose motion type
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

        # Precompute positions/velocities
        for p in plans:
            p.precompute(n_frames)

        # Check min_reversals if required
        if min_reversals is not None:
            rev_count = count_qrr_reversals(plans, n_frames)
            if rev_count < min_reversals:
                continue

        return plans

    raise RuntimeError(
        f"Failed to generate valid scene after {max_retries} attempts "
        f"(n_objects={n_objects}, n_frames={n_frames})"
    )
