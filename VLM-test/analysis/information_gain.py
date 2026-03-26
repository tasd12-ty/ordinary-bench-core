"""
视觉信息增益（Visual Information Gain, VIG）分析。

实现第 5.2 节三条件实验分析：
  VIG = d(Recon_C, GT) - d(Recon_A, GT)    [基础视觉增益]
  VIG_B = d(Recon_C, GT) - d(Recon_B, GT)  [视觉干扰效应]

以及第 5.3 节的误差分解。
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class VIGResult:
    """单个场景的视觉信息增益结果。"""
    scene_id: str = ""

    # 各条件下的 NRMS 值
    nrms_a: Optional[float] = None  # 正确图片
    nrms_b: Optional[float] = None  # 错误图片
    nrms_c: Optional[float] = None  # 无图片

    # 各条件下的 Kendall tau
    tau_a: Optional[float] = None
    tau_b: Optional[float] = None
    tau_c: Optional[float] = None

    # 各条件下的 CSR
    csr_a: Optional[float] = None
    csr_b: Optional[float] = None
    csr_c: Optional[float] = None

    # 视觉信息增益
    vig_nrms: Optional[float] = None    # nrms_c - nrms_a（正值表示视觉有帮助）
    vig_b_nrms: Optional[float] = None  # nrms_c - nrms_b
    vig_tau: Optional[float] = None     # tau_a - tau_c（正值表示视觉有帮助）

    @property
    def vision_helps(self) -> Optional[bool]:
        """正确图片是否改善了重建效果？"""
        if self.vig_nrms is not None:
            return self.vig_nrms > 0
        return None


def compute_vig(
    recon_a: dict,
    recon_b: Optional[dict],
    recon_c: Optional[dict],
    scene_id: str = "",
) -> VIGResult:
    """根据三条件重建结果计算 VIG。

    Args:
        recon_a: 重建结果字典（正确图片）
        recon_b: 重建结果字典（错误图片），或 None
        recon_c: 重建结果字典（无图片），或 None
    """
    result = VIGResult(scene_id=scene_id)

    # 提取各条件的度量值
    def get_metrics(recon):
        if recon is None:
            return None, None, None
        m = recon.get("metrics", {})
        nrms = m.get("nrms")
        tau = m.get("kendall_tau")
        csr = (m.get("csr_qrr", 0) + m.get("csr_trr", 0)) / 2
        return nrms, tau, csr

    result.nrms_a, result.tau_a, result.csr_a = get_metrics(recon_a)
    result.nrms_b, result.tau_b, result.csr_b = get_metrics(recon_b)
    result.nrms_c, result.tau_c, result.csr_c = get_metrics(recon_c)

    # 计算 VIG
    if result.nrms_a is not None and result.nrms_c is not None:
        result.vig_nrms = result.nrms_c - result.nrms_a

    if result.nrms_b is not None and result.nrms_c is not None:
        result.vig_b_nrms = result.nrms_c - result.nrms_b

    if result.tau_a is not None and result.tau_c is not None:
        result.vig_tau = result.tau_a - result.tau_c

    return result


@dataclass
class ErrorDecomposition:
    """误差分解：信息缺失 + 信息错误 + 求解器失败。"""
    insufficiency: float = 0.0      # 来自缺失/弃答的影响
    information_error: float = 0.0  # 来自错误答案的影响
    solver_failure: float = 0.0     # 来自优化器失败的影响

    @property
    def total(self) -> float:
        return self.insufficiency + self.information_error + self.solver_failure


def decompose_errors(
    scene_result: dict,
    recon_result: dict,
    gt_recon_result: Optional[dict] = None,
) -> ErrorDecomposition:
    """将重建误差分解为三个来源。

    Args:
        scene_result: 包含 per_question 的 VLM 评估结果
        recon_result: 基于 VLM 答案的重建结果
        gt_recon_result: 基于真值答案的重建结果（用于估计求解器失败误差）
    """
    scores = scene_result.get("scores", scene_result)
    per_q = scores.get("per_question", [])

    n_total = len(per_q)
    if n_total == 0:
        return ErrorDecomposition()

    # 统计各误差来源数量
    n_missing = sum(1 for q in per_q if q.get("predicted") is None)
    n_wrong = sum(1 for q in per_q
                  if q.get("predicted") is not None and not q.get("correct", False)
                  and not q.get("hour_correct", False))
    n_correct = n_total - n_missing - n_wrong

    # 信息缺失率：缺失答案占比
    insufficiency = n_missing / n_total

    # 信息错误率：错误答案占比
    information_error = n_wrong / n_total

    # 求解器失败误差：由真值重建质量估计
    solver_failure = 0.0
    if gt_recon_result is not None:
        gt_nrms = gt_recon_result.get("metrics", {}).get("nrms", 0)
        if gt_nrms is not None:
            solver_failure = gt_nrms  # 真值重建的 NRMS = 求解器误差下界

    return ErrorDecomposition(
        insufficiency=insufficiency,
        information_error=information_error,
        solver_failure=solver_failure,
    )


def aggregate_vig_results(results: List[VIGResult]) -> dict:
    """跨场景汇总 VIG 结果并输出统计摘要。"""
    if not results:
        return {}

    def _stats(values):
        values = [v for v in values if v is not None]
        if not values:
            return {"mean": None, "std": None, "n": 0}
        return {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "median": float(np.median(values)),
            "n": len(values),
        }

    return {
        "n_scenes": len(results),
        "vig_nrms": _stats([r.vig_nrms for r in results]),
        "vig_b_nrms": _stats([r.vig_b_nrms for r in results]),
        "vig_tau": _stats([r.vig_tau for r in results]),
        "nrms_a": _stats([r.nrms_a for r in results]),
        "nrms_b": _stats([r.nrms_b for r in results]),
        "nrms_c": _stats([r.nrms_c for r in results]),
        "tau_a": _stats([r.tau_a for r in results]),
        "tau_c": _stats([r.tau_c for r in results]),
        "vision_helps_rate": sum(
            1 for r in results if r.vision_helps
        ) / len(results) if results else 0.0,
    }
