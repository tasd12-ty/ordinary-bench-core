"""
场景信念重建的工具函数。

并查集、角度辅助函数、Procrustes 对齐、RMS 计算。
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


# ── 角度辅助函数 ──

def normalize_angle(angle_deg: float) -> float:
    """将角度归一化到 [0, 360)。"""
    return angle_deg % 360


def angular_distance(a_deg: float, b_deg: float) -> float:
    """两个角度之间的最小角距（度）。"""
    diff = abs(normalize_angle(a_deg) - normalize_angle(b_deg))
    return min(diff, 360 - diff)


def hour_to_angle_deg(hour: int) -> float:
    """将钟面小时 (1-12) 转换为角度（度）[0, 360)。

    这里的角度是 TRR 使用的 clock-angle：
      - 12 点 = 0 度
      - 顺时针递增
      - 3 点 = 90 度, 6 点 = 180 度, 9 点 = 270 度
    """
    return (hour % 12) * 30.0


def relative_clock_angle_deg(ref_vec: np.ndarray, tgt_vec: np.ndarray) -> float:
    """计算从 ref_vec（12 点方向）到 tgt_vec 的顺时针钟面角度。"""
    ref_angle = math.atan2(ref_vec[1], ref_vec[0])
    tgt_angle = math.atan2(tgt_vec[1], tgt_vec[0])
    return normalize_angle(-math.degrees(tgt_angle - ref_angle))


def rotate_vec2(v: np.ndarray, angle_rad: float) -> np.ndarray:
    """将 2D 向量旋转指定角度（弧度，逆时针）。"""
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


# ── Procrustes 对齐 ──

def procrustes_align(
    X: np.ndarray,
    Y: np.ndarray,
    allow_reflection: bool = False,
) -> Tuple[np.ndarray, float]:
    """使用 Procrustes 分析将 X 对齐到 Y（平移 + 旋转 + 缩放）。

    参数:
        X: (n, 2) 待对齐数组
        Y: (n, 2) 目标数组
        allow_reflection: 若为 True，允许镜像反射

    返回:
        X_aligned: (n, 2) X 的对齐结果
        rms: 对齐后的 RMS 距离
    """
    n = X.shape[0]

    # 中心化
    mu_x = X.mean(axis=0)
    mu_y = Y.mean(axis=0)
    Xc = X - mu_x
    Yc = Y - mu_y

    # 缩放
    sx = np.sqrt(np.sum(Xc ** 2))
    sy = np.sqrt(np.sum(Yc ** 2))
    if sx < 1e-12 or sy < 1e-12:
        return np.tile(mu_y, (n, 1)), float(np.sqrt(np.mean(np.sum(Yc ** 2, axis=1))))

    Xc /= sx
    Yc /= sy

    # 通过 SVD 计算旋转
    M = Xc.T @ Yc  # (2, 2)
    U, S, Vt = np.linalg.svd(M)

    if not allow_reflection:
        d = np.linalg.det(U) * np.linalg.det(Vt)
        D = np.diag([1.0, np.sign(d)])
        R = U @ D @ Vt
    else:
        R = U @ Vt

    # 应用变换
    X_aligned = (Xc @ R) * sy + mu_y
    residuals = X_aligned - Y
    rms = float(np.sqrt(np.mean(np.sum(residuals ** 2, axis=1))))
    return X_aligned, rms


def compute_rms(X: np.ndarray, Y: np.ndarray) -> float:
    """两个 (n, 2) 数组之间的 RMS 距离。"""
    residuals = X - Y
    return float(np.sqrt(np.mean(np.sum(residuals ** 2, axis=1))))


def compute_nrms(X: np.ndarray, Y: np.ndarray, allow_reflection: bool = True) -> float:
    """归一化 RMS：Procrustes 对齐后的 RMS 除以 Y 的空间范围。

    同时尝试有镜像和无镜像两种对齐，返回较优（较低）的 NRMS。
    这是正确的，因为重建结果仅在相似变换（含镜像反射）意义下确定。
    """
    extent = np.max(np.ptp(Y, axis=0))
    if extent < 1e-12:
        _, rms = procrustes_align(X, Y, allow_reflection=False)
        return rms

    _, rms_no_ref = procrustes_align(X, Y, allow_reflection=False)
    nrms = rms_no_ref / extent

    if allow_reflection:
        _, rms_ref = procrustes_align(X, Y, allow_reflection=True)
        nrms_ref = rms_ref / extent
        nrms = min(nrms, nrms_ref)

    return nrms


# ── 对象对键辅助函数 ──

def pair_key(a: str, b: str) -> Tuple[str, str]:
    """规范化的对象对键（排序后）。"""
    return tuple(sorted((a, b)))
