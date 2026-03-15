"""
Camera motion models for dynamic scene rendering.

Generates per-frame camera parameters (distance, elevation, azimuth, look_at)
that are stored in the plan JSON and read by the Blender renderer.
"""

import math
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple


class CameraMotionModel(ABC):
    """Abstract base for camera motion."""

    @abstractmethod
    def camera_params(self, t: int) -> dict:
        """Return camera parameters at frame t.

        Returns dict with keys: distance, elevation, azimuth, look_at.
        """

    def generate_frames(self, n_frames: int) -> List[dict]:
        """Generate per-frame camera parameters."""
        return [self.camera_params(t) for t in range(n_frames)]


class StaticCamera(CameraMotionModel):
    """Fixed camera position."""

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
    """Camera orbits around the scene at constant speed.

    Args:
        orbit_speed: Degrees per second of azimuth rotation.
        fps: Frames per second.
        base_distance, base_elevation, base_azimuth: Starting camera params.
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
        # Degrees per frame
        self._dpf = orbit_speed / fps

    def camera_params(self, t: int) -> dict:
        return {
            "distance": self.base_distance,
            "elevation": self.base_elevation,
            "azimuth": self.base_azimuth + self._dpf * t,
            "look_at": self.look_at,
        }


class CompositeCameraMotion(CameraMotionModel):
    """Complex camera: orbit + pan (look_at shift) + zoom (distance change).

    All motions are sinusoidal to create smooth, natural camera movement.

    Args:
        orbit_speed: Degrees per second.
        pan_range: Max look_at offset (sinusoidal).
        zoom_range: Max distance offset (sinusoidal).
        fps: Frames per second.
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

        # Orbit: linear azimuth change
        azimuth = self.base_azimuth + self._dpf * t

        # Zoom: sinusoidal distance variation (period ~10s)
        distance = self.base_distance + self.zoom_range * math.sin(2 * math.pi * time_s / 10.0)

        # Pan: sinusoidal look_at shift (period ~8s for x, ~12s for y)
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
    """Build camera plan dict from config. Returns dict with 'type' and 'frames'."""
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
