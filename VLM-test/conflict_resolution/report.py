"""
收敛报告生成
============

输出三类文件：
    1. scenes/{scene_id}.json     — 逐场景消解摘要（轮次、FAS 变化、诊断）
    2. resolved_scenes/{id}.json  — 更新后的完整场景评测结果（可直接用于重建）
    3. summary.json               — 按 split 汇总的噪声/系统性错误统计
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict

from .resolver import ResolutionResult


def save_scene_result(result: ResolutionResult, output_dir: Path) -> None:
    """保存单场景的消解结果和更新后的评测数据。

    Args:
        result: 场景消解结果
        output_dir: 输出根目录（如 output/conflict_resolution/gemini_cr/）
    """
    # ── 保存消解摘要 ──
    scenes_dir = output_dir / "scenes"
    scenes_dir.mkdir(parents=True, exist_ok=True)

    record = {
        "scene_id": result.scene_id,
        "converged": result.converged,
        "n_rounds": result.n_rounds,
        "initial_fas_size": result.initial_fas_size,
        "final_fas_size": result.final_fas_size,
        "diagnosis": {
            "total_questions": result.diagnosis.total_questions,
            "noise_flips": result.diagnosis.noise_flips,
            "systematic_conflicts": result.diagnosis.systematic_conflicts,
            "noise_ratio": round(result.diagnosis.noise_ratio, 4),
            "systematic_ratio": round(result.diagnosis.systematic_ratio, 4),
            "convergence_rounds": result.diagnosis.convergence_rounds,
        } if result.diagnosis else None,
        "history": [
            {
                "round": h.round_idx,
                "fas_size": h.fas_size,
                "n_conflict_questions": h.n_conflict_questions,
                "n_flipped": h.n_flipped,
            }
            for h in result.history
        ],
    }

    with open(scenes_dir / f"{result.scene_id}.json", "w") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)

    # ── 保存去噪后的完整评测结果（可直接喂给重建 pipeline）──
    resolved_dir = output_dir / "resolved_scenes"
    resolved_dir.mkdir(parents=True, exist_ok=True)
    if result.final_scene_result:
        with open(resolved_dir / f"{result.scene_id}.json", "w") as f:
            json.dump(result.final_scene_result, f, indent=2, ensure_ascii=False)


def generate_summary(
    results: Dict[str, ResolutionResult],
    output_dir: Path,
) -> dict:
    """生成按 split 汇总的收敛报告，同时打印到终端。

    Args:
        results: {scene_id: ResolutionResult} 所有场景的消解结果
        output_dir: 输出根目录

    Returns:
        summary dict（同时保存为 summary.json）
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 按 split 分组
    by_split = defaultdict(list)
    for sid, r in results.items():
        sp = sid.rsplit("_", 1)[0]
        by_split[sp].append(r)

    summary: dict = {"by_split": {}, "overall": {}}

    # 累计统计
    total_scenes = 0
    total_initial = 0
    total_final = 0
    total_noise = 0
    total_systematic = 0
    total_questions = 0

    # 打印表头
    print(f"\n{'Split':>5} | {'场景数':>6} | {'初始FAS':>8} | {'最终FAS':>8} | "
          f"{'平均轮数':>8} | {'噪声%':>7} | {'系统性%':>8}")
    print("-" * 75)

    for sp in sorted(by_split):
        rs = by_split[sp]
        n = len(rs)
        avg_init = sum(r.initial_fas_size for r in rs) / n
        avg_final = sum(r.final_fas_size for r in rs) / n
        avg_rounds = sum(r.n_rounds for r in rs) / n
        avg_noise = sum(r.diagnosis.noise_ratio for r in rs if r.diagnosis) / n
        avg_sys = sum(r.diagnosis.systematic_ratio for r in rs if r.diagnosis) / n

        print(f"{sp:>5} | {n:>6} | {avg_init:>8.1f} | {avg_final:>8.1f} | "
              f"{avg_rounds:>8.1f} | {avg_noise:>6.1%} | {avg_sys:>7.1%}")

        summary["by_split"][sp] = {
            "n_scenes": n,
            "avg_initial_fas": round(avg_init, 2),
            "avg_final_fas": round(avg_final, 2),
            "avg_rounds": round(avg_rounds, 2),
            "avg_noise_ratio": round(avg_noise, 4),
            "avg_systematic_ratio": round(avg_sys, 4),
        }

        total_scenes += n
        total_initial += sum(r.initial_fas_size for r in rs)
        total_final += sum(r.final_fas_size for r in rs)
        total_noise += sum(r.diagnosis.noise_flips for r in rs if r.diagnosis)
        total_systematic += sum(r.diagnosis.systematic_conflicts for r in rs if r.diagnosis)
        total_questions += sum(r.diagnosis.total_questions for r in rs if r.diagnosis)

    # 全局汇总
    if total_scenes:
        summary["overall"] = {
            "n_scenes": total_scenes,
            "avg_initial_fas": round(total_initial / total_scenes, 2),
            "avg_final_fas": round(total_final / total_scenes, 2),
            "total_noise_flips": total_noise,
            "total_systematic_conflicts": total_systematic,
            "noise_ratio": round(total_noise / total_questions, 4) if total_questions else 0,
            "systematic_ratio": round(total_systematic / total_initial, 4) if total_initial else 0,
        }
        print("-" * 75)
        o = summary["overall"]
        print(f"{'合计':>5} | {total_scenes:>6} | {o['avg_initial_fas']:>8.1f} | "
              f"{o['avg_final_fas']:>8.1f} | {'':>8} | "
              f"{o['noise_ratio']:>6.1%} | {o['systematic_ratio']:>7.1%}")

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n结果已保存至: {output_dir}")
    return summary
