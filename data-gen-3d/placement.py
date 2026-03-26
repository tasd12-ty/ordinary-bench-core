"""
ordinary-bench-3d 的 3D 物体摆放模块。

在 2D (x, y) 摆放基础上，扩展了可配置的 z 坐标采样。
支持 uniform、discrete_levels 和 gaussian 三种高度分布。
"""

import math
import random
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any


@dataclass
class PlacementConfig:
    """3D 物体摆放配置。"""
    x_range: Tuple[float, float] = (-3.0, 3.0)
    y_range: Tuple[float, float] = (-3.0, 3.0)
    z_range: Tuple[float, float] = (0.0, 2.5)
    z_distribution: str = "uniform"  # 高度分布类型："uniform" | "discrete_levels" | "gaussian"
    discrete_levels: List[float] = field(default_factory=lambda: [0.0, 1.0, 2.0])
    gaussian_mean: float = 1.0
    gaussian_std: float = 0.5
    min_dist_3d: float = 0.5
    max_retries: int = 200

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PlacementConfig":
        x_range = tuple(d.get("x_range", [-3.0, 3.0]))
        y_range = tuple(d.get("y_range", [-3.0, 3.0]))
        z_range = tuple(d.get("z_range", [0.0, 2.5]))
        return cls(
            x_range=x_range,
            y_range=y_range,
            z_range=z_range,
            z_distribution=d.get("z_distribution", "uniform"),
            discrete_levels=d.get("discrete_levels", [0.0, 1.0, 2.0]),
            gaussian_mean=d.get("gaussian_mean", 1.0),
            gaussian_std=d.get("gaussian_std", 0.5),
            min_dist_3d=d.get("min_dist_3d", 0.5),
            max_retries=d.get("max_retries", 200),
        )


def sample_z(config: PlacementConfig, rng: random.Random) -> float:
    """根据配置的分布采样 z 坐标。"""
    z_min, z_max = config.z_range

    if config.z_distribution == "uniform":
        return rng.uniform(z_min, z_max)

    elif config.z_distribution == "discrete_levels":
        # 过滤 z_range 范围内的离散层级
        valid_levels = [
            z for z in config.discrete_levels
            if z_min <= z <= z_max
        ]
        if not valid_levels:
            return rng.uniform(z_min, z_max)
        return rng.choice(valid_levels)

    elif config.z_distribution == "gaussian":
        z = rng.gauss(config.gaussian_mean, config.gaussian_std)
        return max(z_min, min(z_max, z))

    else:
        raise ValueError(f"Unknown z_distribution: {config.z_distribution}")


def distance_3d(
    pos1: Tuple[float, float, float],
    pos2: Tuple[float, float, float],
) -> float:
    """计算两个位置之间的 3D 欧氏距离。"""
    dx = pos1[0] - pos2[0]
    dy = pos1[1] - pos2[1]
    dz = pos1[2] - pos2[2]
    return math.sqrt(dx * dx + dy * dy + dz * dz)


@dataclass
class PlacedObject:
    """已摆放物体，包含 3D 位置和尺寸半径。"""
    x: float
    y: float
    z: float
    radius: float  # 用于碰撞检测的物体尺寸半径

    @property
    def position(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.z)


def place_objects_3d(
    num_objects: int,
    config: PlacementConfig,
    rng: Optional[random.Random] = None,
    size_radii: Optional[List[float]] = None,
) -> List[PlacedObject]:
    """
    在 3D 空间中摆放物体，满足最小距离约束。

    Args:
        num_objects: 要摆放的物体数量
        config: 摆放配置
        rng: 随机数生成器（为 None 时自动创建）
        size_radii: 各物体的碰撞半径（默认每个 0.35）

    Returns:
        包含 3D 位置的 PlacedObject 列表

    Raises:
        RuntimeError: 超过最大重试次数后仍无法摆放时抛出
    """
    if rng is None:
        rng = random.Random()

    if size_radii is None:
        size_radii = [0.35] * num_objects

    placed: List[PlacedObject] = []

    for i in range(num_objects):
        r_i = size_radii[i]
        success = False

        for _ in range(config.max_retries):
            x = rng.uniform(*config.x_range)
            y = rng.uniform(*config.y_range)
            z = sample_z(config, rng)

            # 检查与所有已摆放物体的 3D 最小距离
            too_close = False
            for obj in placed:
                dist = distance_3d((x, y, z), obj.position)
                if dist - r_i - obj.radius < config.min_dist_3d:
                    too_close = True
                    break

            if not too_close:
                placed.append(PlacedObject(x=x, y=y, z=z, radius=r_i))
                success = True
                break

        if not success:
            raise RuntimeError(
                f"Failed to place object {i}/{num_objects} after "
                f"{config.max_retries} retries. Consider increasing "
                f"placement volume or reducing min_dist_3d."
            )

    return placed


def compute_scene_bounds(placed: List[PlacedObject]) -> Dict[str, Tuple[float, float]]:
    """计算已摆放物体的包围盒。"""
    if not placed:
        return {"x": (0, 0), "y": (0, 0), "z": (0, 0)}

    xs = [o.x for o in placed]
    ys = [o.y for o in placed]
    zs = [o.z for o in placed]

    return {
        "x": (min(xs), max(xs)),
        "y": (min(ys), max(ys)),
        "z": (min(zs), max(zs)),
    }


def compute_scene_center(placed: List[PlacedObject]) -> Tuple[float, float, float]:
    """计算已摆放物体的质心。"""
    if not placed:
        return (0.0, 0.0, 0.0)

    cx = sum(o.x for o in placed) / len(placed)
    cy = sum(o.y for o in placed) / len(placed)
    cz = sum(o.z for o in placed) / len(placed)
    return (cx, cy, cz)
