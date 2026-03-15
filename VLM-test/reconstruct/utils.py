"""
Utility functions for scene belief reconstruction.

Union-find, angle helpers, Procrustes alignment, RMS computation.
"""

import math
import numpy as np
from typing import Dict, List, Tuple, Optional


# ── Union-Find ──

class UnionFind:
    """Weighted union-find with path compression."""

    def __init__(self):
        self.parent: Dict[str, str] = {}
        self.rank: Dict[str, int] = {}

    def find(self, x: str) -> str:
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: str, y: str) -> None:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1

    def groups(self) -> Dict[str, List[str]]:
        result: Dict[str, List[str]] = {}
        for x in self.parent:
            root = self.find(x)
            result.setdefault(root, []).append(x)
        return result


# ── Angle Helpers ──

def normalize_angle(angle_deg: float) -> float:
    """Normalize angle to [0, 360)."""
    return angle_deg % 360


def angular_distance(a_deg: float, b_deg: float) -> float:
    """Minimum angular distance between two angles in degrees."""
    diff = abs(normalize_angle(a_deg) - normalize_angle(b_deg))
    return min(diff, 360 - diff)


def hour_to_angle_deg(hour: int) -> float:
    """Convert clock hour (1-12) to angle in degrees [0, 360).

    Clock convention: 12 o'clock = 0 deg, increases clockwise.
    But our angle system uses math convention (CCW from x-axis).
    The hour-to-angle mapping: hour h -> (h % 12) * 30 degrees.
    """
    return (hour % 12) * 30.0


def rotate_vec2(v: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rotate 2D vector by angle (radians, CCW)."""
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


# ── Procrustes Alignment ──

def procrustes_align(
    X: np.ndarray,
    Y: np.ndarray,
    allow_reflection: bool = False,
) -> Tuple[np.ndarray, float]:
    """Align X to Y using Procrustes analysis (translation + rotation + scale).

    Args:
        X: (n, 2) array to be aligned
        Y: (n, 2) target array
        allow_reflection: if True, allow reflections

    Returns:
        X_aligned: (n, 2) aligned version of X
        rms: RMS distance after alignment
    """
    n = X.shape[0]

    # Center both
    mu_x = X.mean(axis=0)
    mu_y = Y.mean(axis=0)
    Xc = X - mu_x
    Yc = Y - mu_y

    # Scale
    sx = np.sqrt(np.sum(Xc ** 2))
    sy = np.sqrt(np.sum(Yc ** 2))
    if sx < 1e-12 or sy < 1e-12:
        return np.tile(mu_y, (n, 1)), float(np.sqrt(np.mean(np.sum(Yc ** 2, axis=1))))

    Xc /= sx
    Yc /= sy

    # Rotation via SVD
    M = Xc.T @ Yc  # (2, 2)
    U, S, Vt = np.linalg.svd(M)

    if not allow_reflection:
        d = np.linalg.det(U) * np.linalg.det(Vt)
        D = np.diag([1.0, np.sign(d)])
        R = U @ D @ Vt
    else:
        R = U @ Vt

    # Apply
    X_aligned = (Xc @ R) * sy + mu_y
    residuals = X_aligned - Y
    rms = float(np.sqrt(np.mean(np.sum(residuals ** 2, axis=1))))
    return X_aligned, rms


def compute_rms(X: np.ndarray, Y: np.ndarray) -> float:
    """RMS distance between two (n, 2) arrays."""
    residuals = X - Y
    return float(np.sqrt(np.mean(np.sum(residuals ** 2, axis=1))))


def compute_nrms(X: np.ndarray, Y: np.ndarray) -> float:
    """Normalized RMS: Procrustes-aligned RMS divided by spatial extent of Y."""
    _, rms = procrustes_align(X, Y)
    extent = np.max(np.ptp(Y, axis=0))
    if extent < 1e-12:
        return rms
    return rms / extent


# ── Pair Key Helpers ──

def pair_key(a: str, b: str) -> Tuple[str, str]:
    """Canonical pair key (sorted)."""
    return tuple(sorted((a, b)))
