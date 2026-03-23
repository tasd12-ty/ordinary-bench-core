"""
场景信念重建工具函数。

并查集、角度辅助、N 维 Procrustes 对齐、RMS 计算。
支持 2D 和 3D 场景重建。
"""

import math
import numpy as np
from typing import Dict, List, Tuple, Optional


# ── 并查集 ──

class UnionFind:
    """带路径压缩的加权并查集。"""

    def __init__(self):
        self.parent: Dict[str, str] = {}
        self.rank: Dict[str, int] = {}

    def find(self, x: str) -> str:
        """查找 x 的根节点，并执行路径压缩。"""
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: str, y: str) -> None:
        """按秩合并 x 和 y 所在的集合。"""
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1

    def groups(self) -> Dict[str, List[str]]:
        """返回所有连通分量，键为根节点，值为成员列表。"""
        result: Dict[str, List[str]] = {}
        for x in self.parent:
            root = self.find(x)
            result.setdefault(root, []).append(x)
        return result


# ── 角度辅助函数 ──

def normalize_angle(angle_deg: float) -> float:
    """将角度归一化到 [0, 360) 范围。"""
    return angle_deg % 360


def angular_distance(a_deg: float, b_deg: float) -> float:
    """计算两个角度之间的最小角距离（度）。"""
    diff = abs(normalize_angle(a_deg) - normalize_angle(b_deg))
    return min(diff, 360 - diff)


def hour_to_angle_deg(hour: int) -> float:
    """将钟面小时（1-12）转换为角度 [0, 360)。

    钟面约定：12 点 = 0 度，顺时针递增。
    映射关系：小时 h -> (h % 12) * 30 度。
    """
    return (hour % 12) * 30.0


def rotate_vec2(v: np.ndarray, angle_rad: float) -> np.ndarray:
    """将 2D 向量按角度旋转（弧度，逆时针）。"""
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


# ── N 维 Procrustes 对齐 ──

def procrustes_align_nd(
    X: np.ndarray,
    Y: np.ndarray,
    allow_reflection: bool = False,
) -> Tuple[np.ndarray, float]:
    """N 维 Procrustes 对齐（支持 2D 和 3D）。

    通过平移、旋转、缩放将 X 对齐到 Y。
    SVD 旋转矩阵为 (d, d)，反射校正使用 Umeyama 方法。

    参数：
        X: (n, d) 待对齐数组
        Y: (n, d) 目标数组
        allow_reflection: 是否允许反射变换

    返回：
        X_aligned: (n, d) 对齐后的数组
        rms: 对齐后的 RMS 距离
    """
    n, d = X.shape
    mu_x = X.mean(axis=0)
    mu_y = Y.mean(axis=0)
    Xc = X - mu_x
    Yc = Y - mu_y
    sx = np.sqrt(np.sum(Xc ** 2))
    sy = np.sqrt(np.sum(Yc ** 2))
    if sx < 1e-12 or sy < 1e-12:
        return np.tile(mu_y, (n, 1)), float(np.sqrt(np.mean(np.sum(Yc ** 2, axis=1))))
    Xc /= sx
    Yc /= sy
    M = Xc.T @ Yc  # (d, d)
    U, S, Vt = np.linalg.svd(M)
    if not allow_reflection:
        det_sign = np.linalg.det(U) * np.linalg.det(Vt)
        D = np.eye(d)
        D[-1, -1] = np.sign(det_sign)
        R = U @ D @ Vt
    else:
        R = U @ Vt
    X_aligned = (Xc @ R) * sy + mu_y
    residuals = X_aligned - Y
    rms = float(np.sqrt(np.mean(np.sum(residuals ** 2, axis=1))))
    return X_aligned, rms


def procrustes_align(
    X: np.ndarray,
    Y: np.ndarray,
    allow_reflection: bool = False,
) -> Tuple[np.ndarray, float]:
    """2D Procrustes 对齐（向后兼容封装）。

    调用 procrustes_align_nd() 实现，保留原有接口。

    参数：
        X: (n, 2) 待对齐数组
        Y: (n, 2) 目标数组
        allow_reflection: 是否允许反射变换

    返回：
        X_aligned: (n, 2) 对齐后的数组
        rms: 对齐后的 RMS 距离
    """
    return procrustes_align_nd(X, Y, allow_reflection=allow_reflection)


def compute_rms(X: np.ndarray, Y: np.ndarray) -> float:
    """计算两个 (n, d) 数组之间的 RMS 距离。"""
    residuals = X - Y
    return float(np.sqrt(np.mean(np.sum(residuals ** 2, axis=1))))


def compute_nrms(X: np.ndarray, Y: np.ndarray) -> float:
    """归一化 RMS：Procrustes 对齐后的 RMS 除以 Y 的空间范围。

    使用 procrustes_align_nd 支持任意维度。
    """
    _, rms = procrustes_align_nd(X, Y)
    extent = np.max(np.ptp(Y, axis=0))
    if extent < 1e-12:
        return rms
    return rms / extent


# ── 对象对键辅助 ──

def pair_key(a: str, b: str) -> Tuple[str, str]:
    """生成规范化的对象对键（排序后）。"""
    return tuple(sorted((a, b)))
