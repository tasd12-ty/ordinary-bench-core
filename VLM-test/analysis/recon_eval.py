"""
多级重建评估系统。

与 VLM 测试解耦——只读取已有评分结果，进行重建并评估。

Level 1: QRR+FDR 零硬违背（距离序在 2D 中几何可实现）
Level 2: Level 1 + TRR adjacent 零违背（距离序 + 角度方向均一致，±1h/45° 容差）
"""

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from reconstruct.preparation import (
    load_questions_auto,
    prepare_reconstruction_input_from_scoring,
    load_scene_gt_positions,
    infer_object_ids_from_questions,
)
from reconstruct.pipeline import _run_pipeline
from reconstruct.constraints import QRREntry, TRREntry
from reconstruct.solver import SolverConfig
from reconstruct.evaluate import (
    cluster_solutions, reflect_positions_y,
    compute_csr_trr, compute_kendall_tau, compute_nrms,
)
from reconstruct.utils import (
    relative_clock_angle_deg, hour_to_angle_deg, angular_distance,
)
from analysis.constraint_violations import check_constraints_against_positions


# ── 单场景评估 ──

def evaluate_scene(
    result_path: str,
    questions_dir: str,
    scenes_dir: str = "../data-gen/output/scenes",
    n_restarts: int = 10,
    trr_adj_tol_deg: float = 45.0,
) -> Optional[dict]:
    """评估单个场景的多级重建质量。"""
    with open(result_path) as f:
        result = json.load(f)

    scene_id = result.get("scene_id", "unknown")
    model = result.get("model", "unknown")

    questions, _ = load_questions_auto(questions_dir, scene_id)
    if not questions:
        return None

    scoring = result.get("scores", result)
    prepared = prepare_reconstruction_input_from_scoring(
        scoring_result=scoring, questions=questions, use_correct_only=False,
    )

    if not prepared.qrr_all:
        return None

    # 构建约束对象
    qrr_entries = [
        QRREntry(
            pair1=tuple(c["pair1"]), pair2=tuple(c["pair2"]),
            comparator=c["comparator"], weight=c.get("weight", 1.0),
            variant=c.get("variant", "disjoint"), anchor=c.get("anchor"),
        )
        for c in prepared.qrr_all
    ]
    trr_entries = [
        TRREntry(
            target=c["target"], ref1=c["ref1"], ref2=c["ref2"],
            hour=c["hour"], weight=c.get("weight", 1.0),
            level=c.get("level", "hour"),
        )
        for c in prepared.trr_constraints
    ]

    config = SolverConfig(n_restarts=n_restarts)

    # ── QRR-only 重建 ──
    recon = _run_pipeline(qrr_entries, [], prepared.object_ids, None, config)
    if not recon.positions:
        return {
            "scene_id": scene_id, "model": model,
            "n_objects": len(prepared.object_ids),
            "level1": {"pass": False, "reason": "no_solution"},
            "level2": {"pass": False, "reason": "no_solution"},
        }

    pos = {k: np.array(v) for k, v in recon.positions.items()}

    # ── Level 1: QRR 硬违背检查 ──
    _, agg = check_constraints_against_positions(pos, prepared.qrr_all)
    cm = agg["confusion_matrix"]
    hard_viol = cm.get("<_to_>", 0) + cm.get(">_to_<", 0)

    cluster = cluster_solutions(recon.all_solutions)

    level1 = {
        "pass": hard_viol == 0,
        "n_qrr": agg["n_total"],
        "hard_violations": hard_viol,
        "soft_violations": agg["n_violated"] - hard_viol,
        "csr_qrr": agg["csr"],
        "k_geom": cluster.K_geom,
    }

    # ── Level 2: TRR adjacent 检查 ──
    if not trr_entries:
        level2 = {"pass": None, "reason": "no_trr_constraints", "n_trr": 0}
    elif not level1["pass"]:
        level2 = {"pass": False, "reason": "level1_failed", "n_trr": len(trr_entries)}
    else:
        # 尝试两种手性，取 TRR 更优的
        csr_orig = compute_csr_trr(pos, trr_entries)
        pos_refl = reflect_positions_y(pos)
        csr_refl = compute_csr_trr(pos_refl, trr_entries)
        best_pos = pos_refl if csr_refl > csr_orig else pos

        # 逐条评估 TRR（adjacent 容差）
        adj_sat = 0
        adj_eval = 0
        adj_violations = []
        for entry in trr_entries:
            x_ref1 = best_pos[entry.ref1]
            x_ref2 = best_pos[entry.ref2]
            x_target = best_pos[entry.target]

            ref_vec = x_ref2 - x_ref1
            if np.linalg.norm(ref_vec) < 1e-10:
                continue
            tgt_vec = x_target - x_ref1
            if np.linalg.norm(tgt_vec) < 1e-10:
                continue

            adj_eval += 1
            rel_angle = relative_clock_angle_deg(ref_vec, tgt_vec)
            expected = hour_to_angle_deg(entry.hour)
            diff = angular_distance(rel_angle, expected)

            if diff <= trr_adj_tol_deg:
                adj_sat += 1
            else:
                adj_violations.append({
                    "target": entry.target, "ref1": entry.ref1, "ref2": entry.ref2,
                    "expected_hour": entry.hour, "angle_diff": round(diff, 1),
                })

        trr_adj_csr = adj_sat / adj_eval if adj_eval > 0 else 0.0
        level2 = {
            "pass": len(adj_violations) == 0 and adj_eval > 0,
            "n_trr": len(trr_entries),
            "trr_adj_satisfied": adj_sat,
            "trr_adj_total": adj_eval,
            "trr_adj_csr": round(trr_adj_csr, 4),
            "trr_adj_violations": len(adj_violations),
        }

    # ── GT 指标（可选） ──
    gt_metrics = {}
    gt_path = Path(scenes_dir) / f"{scene_id}.json"
    if gt_path.exists():
        gt_raw = load_scene_gt_positions(str(gt_path))
        if gt_raw:
            gt_positions = {oid: np.array(p, dtype=np.float64) for oid, p in gt_raw.items()}
            common = sorted(set(pos.keys()) & set(gt_positions.keys()))
            if len(common) >= 3:
                gt_metrics["kendall_tau"] = round(compute_kendall_tau(pos, gt_positions), 4)
                recon_mat = np.array([pos[oid] for oid in common])
                gt_mat = np.array([gt_positions[oid][:2] for oid in common])
                gt_metrics["nrms"] = round(float(compute_nrms(recon_mat, gt_mat, allow_reflection=True)), 4)

    return {
        "scene_id": scene_id,
        "model": model,
        "n_objects": len(prepared.object_ids),
        "level1": level1,
        "level2": level2,
        "gt_metrics": gt_metrics,
    }


# ── 批量评估 ──

def evaluate_model(
    model: str,
    results_base: str = "output/results",
    questions_dir: str = "output/questions",
    scenes_dir: str = "../data-gen/output/scenes",
    output_dir: Optional[str] = None,
) -> List[dict]:
    """评估单个模型的所有场景。"""
    scenes_path = Path(results_base) / model / "scenes"
    if not scenes_path.exists():
        return []

    results = []
    scene_files = sorted(scenes_path.glob("*.json"))

    for i, sf in enumerate(scene_files):
        scene_id = sf.stem
        try:
            r = evaluate_scene(str(sf), questions_dir, scenes_dir)
            if r is None:
                continue
            results.append(r)

            l1 = "PASS" if r["level1"]["pass"] else "FAIL"
            l2_raw = r["level2"]
            if l2_raw["pass"] is None:
                l2 = "N/A"
            elif l2_raw["pass"]:
                l2 = "PASS"
            else:
                l2 = "FAIL"

            if (i + 1) % 20 == 0 or i == len(scene_files) - 1:
                print(f"  [{i+1}/{len(scene_files)}] {scene_id}: L1={l1} L2={l2}")

        except Exception as e:
            print(f"  {scene_id}: ERROR {e}")

    if output_dir:
        out = Path(output_dir) / model
        out.mkdir(parents=True, exist_ok=True)
        for r in results:
            with open(out / f"{r['scene_id']}.json", "w") as f:
                json.dump(r, f, indent=2, ensure_ascii=False)

    return results


def evaluate_all(
    results_base: str = "output/results",
    questions_dir: str = "output/questions",
    scenes_dir: str = "../data-gen/output/scenes",
    output_dir: Optional[str] = None,
) -> dict:
    """评估所有模型并生成汇总。"""
    models_path = Path(results_base)
    model_dirs = sorted(d.name for d in models_path.iterdir()
                        if d.is_dir() and (d / "scenes").exists())

    all_model_results = {}
    for model in model_dirs:
        print(f"\n=== {model} ===")
        results = evaluate_model(model, results_base, questions_dir, scenes_dir, output_dir)
        if results:
            all_model_results[model] = results

    summary = build_summary(all_model_results)

    if output_dir:
        with open(Path(output_dir) / "summary.json", "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    print_summary_table(summary)
    return summary


# ── 汇总 ──

def build_summary(all_model_results: Dict[str, List[dict]]) -> dict:
    """构建跨模型汇总。"""
    models = {}

    for model, results in all_model_results.items():
        total = len(results)
        l1_pass = sum(1 for r in results if r["level1"]["pass"])
        l2_pass = sum(1 for r in results
                      if r["level2"]["pass"] is True)
        l2_na = sum(1 for r in results
                    if r["level2"]["pass"] is None)
        l2_eligible = total - l2_na

        # 按 split 分组
        by_split = defaultdict(lambda: {"total": 0, "level1": 0, "level2": 0, "level2_na": 0})
        for r in results:
            split = r["scene_id"].rsplit("_", 1)[0]
            by_split[split]["total"] += 1
            if r["level1"]["pass"]:
                by_split[split]["level1"] += 1
            if r["level2"]["pass"] is True:
                by_split[split]["level2"] += 1
            if r["level2"]["pass"] is None:
                by_split[split]["level2_na"] += 1

        # K_geom 分布（仅 Level 1 通过的场景）
        k_values = [r["level1"]["k_geom"] for r in results
                    if r["level1"]["pass"] and "k_geom" in r["level1"]]
        k_unique = sum(1 for k in k_values if k == 1)

        # GT 指标均值（仅 Level 1 通过的）
        nrms_vals = [r["gt_metrics"]["nrms"] for r in results
                     if r["level1"]["pass"] and r.get("gt_metrics", {}).get("nrms") is not None]
        tau_vals = [r["gt_metrics"]["kendall_tau"] for r in results
                    if r["level1"]["pass"] and r.get("gt_metrics", {}).get("kendall_tau") is not None]

        models[model] = {
            "total_scenes": total,
            "level1_pass": l1_pass,
            "level1_rate": round(l1_pass / total, 4) if total > 0 else 0,
            "level2_pass": l2_pass,
            "level2_eligible": l2_eligible,
            "level2_rate": round(l2_pass / l2_eligible, 4) if l2_eligible > 0 else 0,
            "k_unique": k_unique,
            "k_unique_rate": round(k_unique / l1_pass, 4) if l1_pass > 0 else 0,
            "nrms_mean": round(float(np.mean(nrms_vals)), 4) if nrms_vals else None,
            "kendall_tau_mean": round(float(np.mean(tau_vals)), 4) if tau_vals else None,
            "by_split": {s: dict(v) for s, v in sorted(by_split.items())},
        }

    # 排名
    ranking = sorted(
        models.items(),
        key=lambda x: (-x[1]["level2_rate"], -x[1]["level1_rate"]),
    )

    return {
        "models": models,
        "ranking": [
            {"model": m, "level1_rate": v["level1_rate"], "level2_rate": v["level2_rate"]}
            for m, v in ranking
        ],
    }


def print_summary_table(summary: dict):
    """打印跨模型汇总表。"""
    models = summary["models"]

    print(f"\n{'='*110}")
    print(f"{'Model':<45} {'Total':>5} {'L1 Pass':>8} {'L1 %':>6} {'L2 Pass':>8} {'L2 %':>6} {'K=1':>5} {'NRMS':>7} {'tau':>6}")
    print(f"{'='*110}")

    for entry in summary["ranking"]:
        m = entry["model"]
        v = models[m]
        nrms = f"{v['nrms_mean']:.3f}" if v["nrms_mean"] is not None else "N/A"
        tau = f"{v['kendall_tau_mean']:.3f}" if v["kendall_tau_mean"] is not None else "N/A"
        l2_str = f"{v['level2_pass']:>3}/{v['level2_eligible']}" if v["level2_eligible"] > 0 else "  N/A"
        l2_pct = f"{v['level2_rate']:.0%}" if v["level2_eligible"] > 0 else "N/A"
        print(f"{m:<45} {v['total_scenes']:>5} "
              f"{v['level1_pass']:>4}/{v['total_scenes']:<3} {v['level1_rate']:>5.0%} "
              f"{l2_str:>8} {l2_pct:>6} "
              f"{v['k_unique']:>5} {nrms:>7} {tau:>6}")

    print(f"{'='*110}")

    # 按 split 的明细
    all_splits = sorted(set(
        s for v in models.values() for s in v["by_split"]
    ))
    if all_splits:
        print(f"\n--- By split ---\n")
        for split in all_splits:
            print(f"  {split}:")
            for entry in summary["ranking"]:
                m = entry["model"]
                sv = models[m]["by_split"].get(split, {})
                if not sv or sv.get("total", 0) == 0:
                    continue
                t = sv["total"]
                l1 = sv["level1"]
                l2 = sv["level2"]
                l2na = sv.get("level2_na", 0)
                l2e = t - l2na
                l2_str = f"{l2}/{l2e}" if l2e > 0 else "N/A"
                print(f"    {m:<42} L1={l1}/{t}  L2={l2_str}")
            print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Multi-level reconstruction evaluation")
    parser.add_argument("--model", type=str, default=None, help="Single model to evaluate")
    parser.add_argument("--all", action="store_true", help="Evaluate all models")
    parser.add_argument("--results-base", default="output/results")
    parser.add_argument("--questions-dir", default="output/questions")
    parser.add_argument("--scenes-dir", default="../data-gen/output/scenes")
    parser.add_argument("--output-dir", "-o", default=None)

    args = parser.parse_args()

    if args.model:
        print(f"=== {args.model} ===")
        results = evaluate_model(
            args.model, args.results_base, args.questions_dir,
            args.scenes_dir, args.output_dir,
        )
        summary = build_summary({args.model: results})
        print_summary_table(summary)
    elif args.all:
        evaluate_all(
            args.results_base, args.questions_dir,
            args.scenes_dir, args.output_dir,
        )
    else:
        print("Use --model MODEL or --all")
