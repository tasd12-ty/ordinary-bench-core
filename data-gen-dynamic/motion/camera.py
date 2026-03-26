"""
动态场景渲染的相机运动模型。

生成逐帧相机参数（distance、elevation、azimuth、look_at），
这些参数存储在规划 JSON 中并由 Blender 渲染器读取。
"""

import math
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple


class CameraMotionModel(ABC):
    """相机运动的抽象基类。"""

    @abstractmethod
    def camera_params(self, t: int) -> dict:
        """返回第 t 帧的相机参数。

        返回字典包含以下键：distance、elevation、azimuth、look_at。
        """

    def generate_frames(self, n_frames: int) -> List[dict]:
        """生成逐帧相机参数。"""
        return [self.camera_params(t) for t in range(n_frames)]


class StaticCamera(CameraMotionModel):
    """固定相机位置。"""

    def __init__(self, distance: float = 10.0, elevation: float = 30.0,
                 azimuth: float = 45.0, look_at: Tuple = (0, 0, 0)):
        self.distance = distance
        self.elevation = elevation
        self.azimuth = azimuth
        self.look_at = list(look_at)

    def camera_params(self, t: int) -> dict:
        return {
            "distance": self.distance,
            "elevation": self.elevation,
            "azimuth": self.azimuth,
            "look_at": self.look_at,
        }


class OrbitCamera(CameraMotionModel):
    """相机以恒定速度绕场景轨道运动。

    Args:
        orbit_speed: 方位角旋转速度（度/秒）。
        fps: 帧率。
        base_distance, base_elevation, base_azimuth: 起始相机参数。
    """

    def __init__(self, orbit_speed: float = 0.5, fps: int = 24,
                 base_distance: float = 10.0, base_elevation: float = 30.0,
                 base_azimuth: float = 45.0, look_at: Tuple = (0, 0, 0)):
        self.orbit_speed = orbit_speed
        self.fps = fps
        self.base_distance = base_distance
        self.base_elevation = base_elevation
        self.base_azimuth = base_azimuth
        self.look_at = list(look_at)
        # 每帧度数
        self._dpf = orbit_speed / fps

    def camera_params(self, t: int) -> dict:
        return {
            "distance": self.base_distance,
            "elevation": self.base_elevation,
            "azimuth": self.base_azimuth + self._dpf * t,
            "look_at": self.look_at,
        }


class CompositeCameraMotion(CameraMotionModel):
    """复合相机运动：轨道旋转 + 平移（look_at 偏移）+ 缩放（距离变化）。

    所有运动均为正弦曲线，以产生平滑自然的相机运动。

    Args:
        orbit_speed: 旋转速度（度/秒）。
        pan_range: 最大 look_at 偏移量（正弦）。
        zoom_range: 最大距离偏移量（正弦）。
        fps: 帧率。
    """

    def __init__(self, orbit_speed: float = 2.0, pan_range: float = 1.0,
                 zoom_range: float = 2.0, fps: int = 24,
                 base_distance: float = 10.0, base_elevation: float = 30.0,
                 base_azimuth: float = 45.0, look_at: Tuple = (0, 0, 0)):
        self.orbit_speed = orbit_speed
        self.pan_range = pan_range
        self.zoom_range = zoom_range
        self.fps = fps
        self.base_distance = base_distance
        self.base_elevation = base_elevation
        self.base_azimuth = base_azimuth
        self.look_at = list(look_at)
        self._dpf = orbit_speed / fps

    def camera_params(self, t: int) -> dict:
        time_s = t / self.fps

        # 轨道旋转：线性方位角变化
        azimuth = self.base_azimuth + self._dpf * t

        # 缩放：正弦距离变化（周期约 10 秒）
        distance = self.base_distance + self.zoom_range * math.sin(2 * math.pi * time_s / 10.0)

        # 平移：正弦 look_at 偏移（x 轴周期约 8 秒，y 轴约 12 秒）
        look_at = [
            self.look_at[0] + self.pan_range * math.sin(2 * math.pi * time_s / 8.0),
            self.look_at[1] + self.pan_range * math.sin(2 * math.pi * time_s / 12.0),
            self.look_at[2],
        ]

        return {
            "distance": distance,
            "elevation": self.base_elevation,
            "azimuth": azimuth,
            "look_at": look_at,
        }


def build_camera_plan(camera_cfg: dict, n_frames: int, fps: int = 24) -> dict:
    """从配置构建相机规划字典。返回包含 'type' 和 'frames' 的字典。"""
    cam_type = camera_cfg.get("type", "static")
    base_dist = camera_cfg.get("base_distance", 10.0)
    base_elev = camera_cfg.get("base_elevation", 30.0)
    base_az = camera_cfg.get("base_azimuth", 45.0)

    if cam_type == "static":
        model = StaticCamera(base_dist, base_elev, base_az)
    elif cam_type == "orbit":
        model = OrbitCamera(
            orbit_speed=camera_cfg.get("orbit_speed", 0.5),
            fps=fps,
            base_distance=base_dist,
            base_elevation=base_elev,
            base_azimuth=base_az,
        )
    elif cam_type == "composite":
        model = CompositeCameraMotion(
            orbit_speed=camera_cfg.get("orbit_speed", 2.0),
            pan_range=camera_cfg.get("pan_range", 1.0),
            zoom_range=camera_cfg.get("zoom_range", 2.0),
            fps=fps,
            base_distance=base_dist,
            base_elevation=base_elev,
            base_azimuth=base_az,
        )
    else:
        raise ValueError(f"Unknown camera type: {cam_type}")

    return {
        "type": cam_type,
        "config": camera_cfg,
        "frames": model.generate_frames(n_frames),
    }
