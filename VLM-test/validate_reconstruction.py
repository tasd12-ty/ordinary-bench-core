"""
使用真值约束验证重建流程。

加载场景数据，从 3D 坐标提取 GT QRR/TRR 约束，
将其输入重建求解器，并评估算法能否正确恢复原始 2D 布局。

完美真值输入下的预期结果：
  - CSR_QRR ≈ 1.0, CSR_TRR ≈ 1.0
  - NRMS < 0.05
  - K_geom = 1（唯一解）
  - status = "single_mode"
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# 确保 VLM-test 在 sys.path 中
sys.path.insert(0, str(Path(__file__).parent))

from dsl.predicates import (
    MetricType, extract_all_qrr, extract_all_qrr_shared_anchor, extract_all_trr,
)
from extraction import parse_objects, load_scene
from reconstruct import reconstruct, SolverConfig


# ── 格式转换 ──

def qrr_to_dict(c) -> dict:
    """将 DSL 中的 QRRConstraint 转换为 reconstruct() 所需的字典格式。"""
    data = {
        "pair1": c.pair1,
        "pair2": c.pair2,
        "comparator": str(c.comparator),
        "weight": 1.0,
        "variant": c.variant,
    }
    if c.anchor is not None:
        data["anchor"] = c.anchor
    return data


def trr_to_dict(c) -> dict:
    """将 DSL 中的 TRRConstraint 转换为 reconstruct() 所需的字典格式。"""
    return {
        "target": c.target,
        "ref1": c.ref1,
        "ref2": c.ref2,
        "hour": c.hour,
        "weight": 1.0,
        "level": "hour",
    }


def extract_gt_positions(scene: dict) -> Dict[str, np.ndarray]:
    """从场景 JSON 中提取 2D 真值坐标。"""
    positions = {}
    for obj in scene.get("objects", []):
        coords = obj.get("3d_coords", [0, 0, 0])
        positions[obj["id"]] = np.array(coords[:2], dtype=np.float64)
    return positions


# ── 单场景验证 ──

def validate_single_scene(
    scene_path: str,
    tau: float = 0.10,
    n_restarts: int = 10,
) -> Optional[dict]:
    """使用真值约束验证单个场景的重建结果。

    返回包含 scene_id、约束数量和重建指标的字典，
    若场景无法加载则返回 None。
    """
    scene = load_scene(scene_path)
    scene_id = scene.get("scene_id", Path(scene_path).stem)
    objects = parse_objects(scene)

    if len(objects) < 3:
        return None

    # 提取真值约束
    qrr_constraints = extract_all_qrr(
        objects, MetricType.DIST_3D, tau=tau, disjoint_only=True
    )
    qrr_constraints += extract_all_qrr_shared_anchor(
        objects, MetricType.DIST_3D, tau=tau,
    )
    trr_constraints = extract_all_trr(objects, use_3d=True)
    n_qrr_disjoint = sum(1 for c in qrr_constraints if c.variant == "disjoint")
    n_qrr_shared_anchor = sum(1 for c in qrr_constraints if c.variant == "shared_anchor")

    # 转换为 reconstruct() 的输入格式
    qrr_dicts = [qrr_to_dict(c) for c in qrr_constraints]
    trr_dicts = [trr_to_dict(c) for c in trr_constraints]
    object_ids = sorted(objects.keys())
    gt_positions = extract_gt_positions(scene)

    # 执行重建
    result = reconstruct(
        qrr_constraints=qrr_dicts,
        trr_constraints=trr_dicts,
        object_ids=object_ids,
        gt_positions=gt_positions,
        n_restarts=n_restarts,
    )

    rd = result.to_dict()
    return {
        "scene_id": scene_id,
        "n_objects": len(objects),
        "n_qrr_disjoint": n_qrr_disjoint,
        "n_qrr_shared_anchor": n_qrr_shared_anchor,
        "n_qrr": len(qrr_dicts),
        "n_trr": len(trr_dicts),
        "feasible": rd["feasible"],
        "status": rd["status"],
        "metrics": rd["metrics"],
        "feasibility_checks": rd["feasibility_checks"],
    }


# ── 批量验证 ──

def validate_all(
    scenes_dir: str,
    split: Optional[str] = None,
    max_scenes: Optional[int] = None,
    tau: float = 0.10,
    n_restarts: int = 10,
) -> List[dict]:
    """在所有场景（或指定分组）上验证重建结果。"""
    scenes_path = Path(scenes_dir)
    scene_files = sorted(scenes_path.glob("*.json"))

    if split:
        scene_files = [f for f in scene_files if f.stem.startswith(split)]

    if max_scenes:
        scene_files = scene_files[:max_scenes]

    if not scene_files:
        print(f"No scene files found in {scenes_dir}" +
              (f" for split {split}" if split else ""))
        return []

    print(f"Validating reconstruction on {len(scene_files)} scenes "
          f"(restarts={n_restarts}, tau={tau})")
    print("-" * 80)

    results = []
    t0 = time.time()

    for i, scene_file in enumerate(scene_files):
        try:
            output = validate_single_scene(
                str(scene_file), tau=tau, n_restarts=n_restarts
            )
            if output is None:
                print(f"  [{i+1}/{len(scene_files)}] {scene_file.stem}: "
                      f"skipped (< 3 objects)")
                continue

            m = output["metrics"]
            nrms_str = f"{m['nrms']:.4f}" if m.get("nrms") is not None else "N/A"
            tau_str = f"{m['kendall_tau']:.3f}" if m.get("kendall_tau") is not None else "N/A"

            print(f"  [{i+1}/{len(scene_files)}] {output['scene_id']}: "
                  f"n_qrr={output['n_qrr']} "
                  f"(disjoint={output['n_qrr_disjoint']}, "
                  f"shared_anchor={output['n_qrr_shared_anchor']}) "
                  f"n_trr={output['n_trr']} -> "
                  f"status={output['status']} "
                  f"csr_qrr={m['csr_qrr']:.3f} csr_trr={m['csr_trr']:.3f} "
                  f"nrms={nrms_str} tau={tau_str} K={m['K_geom']}")

            results.append(output)

        except Exception as e:
            print(f"  [{i+1}/{len(scene_files)}] {scene_file.stem}: ERROR {e}")

    elapsed = time.time() - t0
    print(f"\nCompleted in {elapsed:.1f}s ({elapsed/max(len(results),1):.2f}s/scene)")
    return results


def summarize(results: List[dict]) -> dict:
    """计算所有已验证场景的汇总统计指标。"""
    if not results:
        return {}

    summary = {
        "n_scenes": len(results),
        "status_counts": {},
        "feasible_rate": sum(1 for r in results if r["feasible"]) / len(results),
    }

    for r in results:
        s = r["status"]
        summary["status_counts"][s] = summary["status_counts"].get(s, 0) + 1

    metric_keys = ["csr_qrr", "csr_trr", "K_geom", "spread", "best_loss",
                   "kendall_tau", "nrms"]

    for key in metric_keys:
        values = [r["metrics"][key] for r in results
                  if r["metrics"].get(key) is not None]
        if values:
            summary[f"{key}_mean"] = float(np.mean(values))
            summary[f"{key}_std"] = float(np.std(values))
            summary[f"{key}_min"] = float(np.min(values))
            summary[f"{key}_max"] = float(np.max(values))

    # 按分组统计
    by_split = {}
    for r in results:
        sp = r["scene_id"].rsplit("_", 1)[0]
        by_split.setdefault(sp, []).append(r)

    summary["by_split"] = {}
    for sp, sp_results in sorted(by_split.items()):
        sp_summary = {"n_scenes": len(sp_results)}
        for key in metric_keys:
            values = [r["metrics"][key] for r in sp_results
                      if r["metrics"].get(key) is not None]
            if values:
                sp_summary[f"{key}_mean"] = float(np.mean(values))
        summary["by_split"][sp] = sp_summary

    # 标记失败场景
    failures = [r for r in results if r["status"] != "single_mode"]
    summary["n_failures"] = len(failures)
    if failures:
        summary["failure_scenes"] = [
            {"scene_id": r["scene_id"], "status": r["status"],
             "csr_qrr": r["metrics"]["csr_qrr"],
             "csr_trr": r["metrics"]["csr_trr"]}
            for r in failures
        ]

    return summary


def print_summary(summary: dict):
    """格式化打印汇总统计信息。"""
    print("\n" + "=" * 60)
    print(f"  GT Reconstruction Validation Summary ({summary['n_scenes']} scenes)")
    print("=" * 60)
    print(f"  Feasible rate:  {summary['feasible_rate']:.1%}")
    print(f"  Status:         {summary['status_counts']}")
    print()

    for key in ["csr_qrr", "csr_trr", "nrms", "kendall_tau", "K_geom", "spread"]:
        if f"{key}_mean" in summary:
            mean = summary[f"{key}_mean"]
            std = summary[f"{key}_std"]
            lo = summary[f"{key}_min"]
            hi = summary[f"{key}_max"]
            print(f"  {key:14s}: {mean:.4f} +/- {std:.4f}  [{lo:.4f}, {hi:.4f}]")

    if summary.get("n_failures", 0) > 0:
        print(f"\n  WARNING: {summary['n_failures']} scenes did NOT reach single_mode:")
        for f in summary.get("failure_scenes", []):
            print(f"    {f['scene_id']}: status={f['status']} "
                  f"csr_qrr={f['csr_qrr']:.3f} csr_trr={f['csr_trr']:.3f}")

    if summary.get("by_split"):
        print(f"\n  Per-split breakdown:")
        print(f"  {'Split':8s} {'N':>4s} {'CSR_QRR':>8s} {'CSR_TRR':>8s} "
              f"{'NRMS':>8s} {'Kendall':>8s} {'K_geom':>7s}")
        for sp, ss in sorted(summary["by_split"].items()):
            def g(k): return f"{ss[k]:.4f}" if k in ss else "N/A"
            print(f"  {sp:8s} {ss['n_scenes']:4d} "
                  f"{g('csr_qrr_mean'):>8s} {g('csr_trr_mean'):>8s} "
                  f"{g('nrms_mean'):>8s} {g('kendall_tau_mean'):>8s} "
                  f"{g('K_geom_mean'):>7s}")

    print("=" * 60)


# ── 命令行入口 ──

def main():
    parser = argparse.ArgumentParser(
        description="Validate reconstruction pipeline using GT constraints"
    )
    parser.add_argument(
        "--scenes-dir", "-d",
        default=str(Path(__file__).parent.parent / "data-gen" / "output" / "scenes"),
        help="Path to scene JSON directory (default: data-gen/output/scenes)",
    )
    parser.add_argument("--split", "-s", default=None,
                        help="Only validate a specific split (e.g., n04, n05)")
    parser.add_argument("--max-scenes", "-n", type=int, default=None,
                        help="Maximum number of scenes to validate")
    parser.add_argument("--restarts", "-r", type=int, default=10,
                        help="Number of optimization restarts (default: 10)")
    parser.add_argument("--tau", type=float, default=0.10,
                        help="QRR tolerance parameter (default: 0.10)")
    parser.add_argument("--output", "-o", default=None,
                        help="Output path for detailed results JSON")

    args = parser.parse_args()

    results = validate_all(
        scenes_dir=args.scenes_dir,
        split=args.split,
        max_scenes=args.max_scenes,
        tau=args.tau,
        n_restarts=args.restarts,
    )

    if not results:
        print("No results to summarize.")
        return

    summary = summarize(results)
    print_summary(summary)

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        output_data = {"summary": summary, "scenes": results}
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2, default=str)
        print(f"\nDetailed results saved to {args.output}")


if __name__ == "__main__":
    main()
