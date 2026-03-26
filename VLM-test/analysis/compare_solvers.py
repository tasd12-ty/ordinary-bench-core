"""
三种求解器对比分析。

对指定场景分别用 lbfgsb / sdp / hinge 求解，比较硬违背数、CSR、NRMS 等指标。
"""

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from reconstruct.preparation import (
    load_questions_auto,
    prepare_reconstruction_input_from_scoring,
    load_scene_gt_positions,
)
from reconstruct.constraints import QRREntry, TRREntry
from reconstruct.solver import SolverConfig, SolverSolution, compute_qrr_loss, compute_trr_loss
from reconstruct.solver_dispatch import solve, METHODS
from reconstruct.evaluate import (
    compute_csr_qrr, compute_csr_trr, compute_kendall_tau, compute_nrms,
    reflect_positions_y,
)
from reconstruct.utils import procrustes_align
from analysis.constraint_violations import check_constraints_against_positions


def run_single_scene(
    result_path: str,
    questions_dir: str,
    scenes_dir: str,
    methods: List[str] = None,
    constraint_mode: str = "fdr_qrr",
    n_restarts: int = 10,
) -> Optional[dict]:
    """对单个场景跑多种 solver 并对比。"""
    if methods is None:
        methods = list(METHODS)

    with open(result_path) as f:
        result = json.load(f)

    scene_id = result.get("scene_id", "unknown")
    questions, _ = load_questions_auto(questions_dir, scene_id)
    if not questions:
        return None

    scoring = result.get("scores", result)
    prepared = prepare_reconstruction_input_from_scoring(
        scoring_result=scoring, questions=questions, use_correct_only=False)

    # 选约束子集
    if constraint_mode == "fdr_qrr":
        qrr_sel = prepared.qrr_all
        trr_sel = []
    elif constraint_mode == "all":
        qrr_sel = prepared.qrr_all
        trr_sel = prepared.trr_constraints
    else:
        qrr_sel = prepared.qrr_all
        trr_sel = []

    # 构建 QRREntry / TRREntry
    qrr_entries = [
        QRREntry(
            pair1=tuple(c["pair1"]), pair2=tuple(c["pair2"]),
            comparator=c["comparator"], weight=c.get("weight", 1.0),
            variant=c.get("variant", "disjoint"), anchor=c.get("anchor"),
        ) for c in qrr_sel
    ]
    trr_entries = [
        TRREntry(
            target=c["target"], ref1=c["ref1"], ref2=c["ref2"],
            hour=c["hour"], weight=c.get("weight", 1.0),
            level=c.get("level", "hour"),
        ) for c in trr_sel
    ]

    # 加载 GT
    gt_path = Path(scenes_dir) / f"{scene_id}.json"
    gt_positions = None
    if gt_path.exists():
        gt_raw = load_scene_gt_positions(str(gt_path))
        if gt_raw:
            gt_positions = {oid: np.array(pos, dtype=np.float64) for oid, pos in gt_raw.items()}

    config = SolverConfig(n_restarts=n_restarts)

    results = {"scene_id": scene_id, "n_objects": len(prepared.object_ids),
               "n_qrr": len(qrr_entries), "n_trr": len(trr_entries),
               "constraint_mode": constraint_mode, "methods": {}}

    for method in methods:
        t0 = time.time()
        try:
            solutions = solve(
                object_ids=prepared.object_ids,
                qrr_entries=qrr_entries,
                trr_entries=trr_entries,
                config=config,
                gt_positions=gt_positions,
                method=method,
            )
        except Exception as e:
            results["methods"][method] = {"error": str(e)}
            continue
        elapsed = time.time() - t0

        if not solutions:
            results["methods"][method] = {"error": "no_solutions", "time": elapsed}
            continue

        best = solutions[0]
        pos = best.positions

        # CSR
        csr_qrr = compute_csr_qrr(pos, qrr_entries)
        csr_trr = compute_csr_trr(pos, trr_entries) if trr_entries else None

        # 硬违背
        _, agg = check_constraints_against_positions(pos, qrr_sel)
        cm = agg["confusion_matrix"]
        hard_viol = cm.get("<_to_>", 0) + cm.get(">_to_<", 0)

        # GT 指标
        kendall = None
        nrms = None
        if gt_positions:
            common = sorted(set(pos.keys()) & set(gt_positions.keys()))
            if len(common) >= 3:
                kendall = compute_kendall_tau(pos, gt_positions)
                recon_mat = np.array([pos[oid] for oid in common])
                gt_mat = np.array([gt_positions[oid][:2] for oid in common])
                nrms = compute_nrms(recon_mat, gt_mat, allow_reflection=True)

        results["methods"][method] = {
            "loss": round(best.loss, 4),
            "loss_qrr": round(best.loss_qrr, 4),
            "csr_qrr": round(csr_qrr, 4),
            "csr_trr": round(csr_trr, 4) if csr_trr is not None else None,
            "hard_violations": hard_viol,
            "n_total": agg["n_total"],
            "kendall_tau": round(kendall, 4) if kendall is not None else None,
            "nrms": round(nrms, 4) if nrms is not None else None,
            "n_solutions": len(solutions),
            "converged": best.converged,
            "time": round(elapsed, 3),
        }

    return results


def print_comparison(results: dict):
    """打印单场景对比表。"""
    sid = results["scene_id"]
    print(f"\n=== {sid} (n_obj={results['n_objects']}, "
          f"n_qrr={results['n_qrr']}, n_trr={results['n_trr']}, "
          f"mode={results['constraint_mode']}) ===\n")

    header = f"{'Method':<10} {'Loss':>8} {'CSR_QRR':>8} {'Hard':>6} {'Kendall':>8} {'NRMS':>8} {'Time':>7}"
    print(header)
    print("-" * len(header))

    for method, m in results["methods"].items():
        if "error" in m:
            print(f"{method:<10} ERROR: {m['error']}")
            continue
        kt = f"{m['kendall_tau']:.3f}" if m["kendall_tau"] is not None else "N/A"
        nrms = f"{m['nrms']:.4f}" if m["nrms"] is not None else "N/A"
        print(f"{method:<10} {m['loss']:>8.2f} {m['csr_qrr']:>8.3f} "
              f"{m['hard_violations']:>3}/{m['n_total']:<2} "
              f"{kt:>8} {nrms:>8} {m['time']:>6.2f}s")


def run_all_svg_scenes(
    recon_base: str = "output/analysis/belief_recon",
    results_base: str = "output/results",
    questions_dir: str = "output/questions",
    scenes_dir: str = "../data-gen/output/scenes",
    constraint_mode: str = "fdr_qrr",
    max_scenes: int = None,
):
    """对所有有 SVG 的场景跑对比。"""
    recon_root = Path(recon_base)
    results_root = Path(results_base)

    all_results = []
    # 汇总: method -> {hard_total, n_total, csr_sum, count, ...}
    from collections import defaultdict
    summary = defaultdict(lambda: {
        "hard": 0, "n_total": 0, "csr_sum": 0.0, "count": 0,
        "nrms_sum": 0.0, "nrms_count": 0, "time_sum": 0.0,
    })

    scene_count = 0
    for model_dir in sorted(recon_root.iterdir()):
        if not model_dir.is_dir():
            continue
        model = model_dir.name
        results_dir = results_root / model / "scenes"
        if not results_dir.exists():
            continue

        svgs = sorted(f for f in model_dir.glob("*.svg")
                      if not f.stem.endswith("_no_trr"))
        if not svgs:
            continue

        for svg in svgs:
            if max_scenes and scene_count >= max_scenes:
                break
            scene_id = svg.stem
            rf = results_dir / f"{scene_id}.json"
            if not rf.exists():
                continue

            r = run_single_scene(
                result_path=str(rf),
                questions_dir=questions_dir,
                scenes_dir=scenes_dir,
                constraint_mode=constraint_mode,
            )
            if r is None:
                continue

            print_comparison(r)
            all_results.append(r)
            scene_count += 1

            for method, m in r["methods"].items():
                if "error" in m:
                    continue
                s = summary[method]
                s["hard"] += m["hard_violations"]
                s["n_total"] += m["n_total"]
                s["csr_sum"] += m["csr_qrr"]
                s["count"] += 1
                s["time_sum"] += m["time"]
                if m.get("nrms") is not None:
                    s["nrms_sum"] += m["nrms"]
                    s["nrms_count"] += 1

        if max_scenes and scene_count >= max_scenes:
            break

    # 打印汇总
    print(f"\n\n{'='*70}")
    print(f"SUMMARY ({scene_count} scenes, mode={constraint_mode})")
    print(f"{'='*70}\n")

    header = f"{'Method':<10} {'Scenes':>6} {'Hard_total':>10} {'CSR_mean':>9} {'NRMS_mean':>10} {'Time_total':>10}"
    print(header)
    print("-" * len(header))
    for method in METHODS:
        s = summary[method]
        if s["count"] == 0:
            print(f"{method:<10} 0 scenes")
            continue
        csr_mean = s["csr_sum"] / s["count"]
        nrms_mean = s["nrms_sum"] / s["nrms_count"] if s["nrms_count"] > 0 else 0
        print(f"{method:<10} {s['count']:>6} {s['hard']:>5}/{s['n_total']:<4} "
              f"{csr_mean:>9.4f} {nrms_mean:>10.4f} {s['time_sum']:>9.1f}s")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compare solver methods")
    parser.add_argument("--scene", type=str, default=None, help="Single scene ID")
    parser.add_argument("--model", type=str, default="claude-sonnet-4-6", help="Model name")
    parser.add_argument("--all", action="store_true", help="Run on all SVG scenes")
    parser.add_argument("--mode", choices=["fdr_qrr", "all"], default="fdr_qrr",
                        help="Constraint mode: fdr_qrr (QRR only) or all (QRR+TRR)")
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--scenes-dir", default="../data-gen/output/scenes")

    args = parser.parse_args()

    if args.scene:
        rf = f"output/results/{args.model}/scenes/{args.scene}.json"
        r = run_single_scene(rf, "output/questions", args.scenes_dir,
                             constraint_mode=args.mode)
        if r:
            print_comparison(r)
    elif args.all:
        run_all_svg_scenes(
            constraint_mode=args.mode,
            max_scenes=args.max_scenes,
            scenes_dir=args.scenes_dir,
        )
    else:
        print("Use --scene SCENE_ID or --all")
