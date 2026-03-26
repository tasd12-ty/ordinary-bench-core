"""
重建坐标的 QRR 约束违背分析。

从重建位置重新计算成对距离，与 Belief（VLM 预测）和 GT（真值）约束逐条对比，
输出逐条明细、混淆矩阵、按 variant 分组统计。
"""

import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from reconstruct.preparation import (
    load_questions_auto,
    prepare_reconstruction_input_from_scoring,
    _decompose_fdr_record,
)
from analysis.aggregate import load_scene_results


# ── 核心：逐条约束检查 ──

def check_constraints_against_positions(
    positions: Dict[str, np.ndarray],
    qrr_constraints: List[dict],
    tau: float = 0.10,
) -> Tuple[List[dict], dict]:
    """用重建位置检查每条 QRR 约束，返回 (逐条明细, 聚合统计)。

    逐条明细包含：qid, pair1, pair2, variant, source, constraint_cmp,
    recon_cmp, d1, d2, delta_log, satisfied。

    聚合统计包含：n_total, n_satisfied, n_violated, csr,
    confusion_matrix, by_variant, by_source。
    """
    details = []
    confusion = defaultdict(int)  # "{constraint_cmp}_to_{recon_cmp}" -> count
    by_variant = defaultdict(lambda: {"n_total": 0, "n_satisfied": 0})
    by_source = defaultdict(lambda: {"n_total": 0, "n_satisfied": 0})

    eps = 1e-12

    for c in qrr_constraints:
        p1 = tuple(sorted(c["pair1"]))
        p2 = tuple(sorted(c["pair2"]))

        # 计算重建距离
        if p1[0] not in positions or p1[1] not in positions:
            continue
        if p2[0] not in positions or p2[1] not in positions:
            continue

        d1 = float(np.linalg.norm(positions[p1[0]] - positions[p1[1]]))
        d2 = float(np.linalg.norm(positions[p2[0]] - positions[p2[1]]))

        # 推导重建比较符（与 evaluate.py compute_csr_qrr 一致）
        max_val = max(d1, d2)
        if max_val < eps:
            recon_cmp = "~="
        else:
            threshold = tau * max_val
            diff = d1 - d2
            if abs(diff) <= threshold:
                recon_cmp = "~="
            elif diff < 0:
                recon_cmp = "<"
            else:
                recon_cmp = ">"

        # log 域差值
        delta_log = math.log(d1 + eps) - math.log(d2 + eps)

        constraint_cmp = c["comparator"]
        satisfied = recon_cmp == constraint_cmp

        variant = c.get("variant", "disjoint")
        source = c.get("source_type", "qrr")

        detail = {
            "qid": c.get("qid", ""),
            "pair1": list(p1),
            "pair2": list(p2),
            "variant": variant,
            "source": source,
            "constraint_cmp": constraint_cmp,
            "recon_cmp": recon_cmp,
            "d1": round(d1, 6),
            "d2": round(d2, 6),
            "delta_log": round(delta_log, 6),
            "satisfied": satisfied,
        }
        if c.get("anchor"):
            detail["anchor"] = c["anchor"]

        details.append(detail)

        # 聚合
        confusion[f"{constraint_cmp}_to_{recon_cmp}"] += 1
        by_variant[variant]["n_total"] += 1
        by_source[source]["n_total"] += 1
        if satisfied:
            by_variant[variant]["n_satisfied"] += 1
            by_source[source]["n_satisfied"] += 1

    n_total = len(details)
    n_satisfied = sum(1 for d in details if d["satisfied"])

    # 构建完整混淆矩阵
    cmps = ["<", "~=", ">"]
    full_confusion = {}
    for c_from in cmps:
        for c_to in cmps:
            key = f"{c_from}_to_{c_to}"
            full_confusion[key] = confusion.get(key, 0)

    # 各分组加上 csr
    by_variant_out = {}
    for v, stats in by_variant.items():
        stats["csr"] = round(stats["n_satisfied"] / stats["n_total"], 4) if stats["n_total"] > 0 else 0.0
        by_variant_out[v] = dict(stats)

    by_source_out = {}
    for s, stats in by_source.items():
        stats["csr"] = round(stats["n_satisfied"] / stats["n_total"], 4) if stats["n_total"] > 0 else 0.0
        by_source_out[s] = dict(stats)

    aggregate = {
        "n_total": n_total,
        "n_satisfied": n_satisfied,
        "n_violated": n_total - n_satisfied,
        "csr": round(n_satisfied / n_total, 4) if n_total > 0 else 0.0,
        "confusion_matrix": full_confusion,
        "by_variant": by_variant_out,
        "by_source": by_source_out,
    }

    return details, aggregate


# ── GT 约束提取（直接从问题元数据，不依赖评分正确性） ──

def extract_gt_qrr_constraints(questions: List[dict]) -> List[dict]:
    """从问题元数据提取完整 GT QRR 约束列表。

    包括：
    - 所有 type=="qrr" 问题的 gt_comparator
    - 所有 type=="fdr" 问题的 gt_ranking 分解为 shared_anchor QRR
    """
    constraints = []

    for q in questions:
        if q["type"] == "qrr":
            gt_cmp = q.get("gt_comparator")
            if gt_cmp not in ("<", "~=", ">"):
                continue
            constraints.append({
                "qid": q["qid"],
                "source_type": "qrr",
                "pair1": list(q["pair1"]),
                "pair2": list(q["pair2"]),
                "comparator": gt_cmp,
                "weight": 1.0,
                "variant": q.get("variant", "disjoint"),
                "anchor": q.get("anchor"),
            })
        elif q["type"] == "fdr":
            gt_ranking = q.get("gt_ranking")
            anchor = q.get("anchor")
            if not gt_ranking or not anchor:
                continue
            fdr_record = {
                "qid": q["qid"],
                "anchor": anchor,
                "ranking": list(gt_ranking),
                "weight": 1.0,
            }
            derived = _decompose_fdr_record(fdr_record)
            constraints.extend(derived)

    return constraints


# ── Belief 约束提取（从评分结果） ──

def extract_belief_qrr_constraints(
    scoring_result: dict,
    questions: List[dict],
) -> List[dict]:
    """从 VLM 评分结果提取 Belief QRR 约束。

    复用 prepare_reconstruction_input_from_scoring 的逻辑，
    但只返回 qrr_all 约束列表。
    """
    prepared = prepare_reconstruction_input_from_scoring(
        scoring_result=scoring_result,
        questions=questions,
        use_correct_only=False,
    )
    return prepared.qrr_all


# ── 单场景分析 ──

def analyze_single_scene(
    recon_path: str,
    result_path: str,
    questions_dir: str,
    tau: float = 0.10,
) -> Optional[dict]:
    """分析单个场景的 QRR 约束违背。

    Args:
        recon_path: 重建结果 JSON 路径（含 positions）
        result_path: 评分结果 JSON 路径（含 per_question）
        questions_dir: 问题目录
        tau: 比较容差

    Returns:
        包含 belief_analysis 和 gt_analysis 的字典
    """
    # 加载重建位置
    with open(recon_path) as f:
        recon = json.load(f)

    positions_raw = recon.get("positions", {})
    if not positions_raw:
        return None
    positions = {oid: np.array(pos, dtype=np.float64) for oid, pos in positions_raw.items()}

    # 加载评分结果
    with open(result_path) as f:
        result = json.load(f)

    scene_id = result.get("scene_id", recon.get("scene_id", "unknown"))

    # 加载问题
    questions, _ = load_questions_auto(questions_dir, scene_id)
    if not questions:
        return None

    scoring = result.get("scores", result)

    # Belief 约束
    belief_constraints = extract_belief_qrr_constraints(scoring, questions)
    belief_details, belief_agg = check_constraints_against_positions(
        positions, belief_constraints, tau=tau
    )
    belief_agg["details"] = belief_details

    # GT 约束
    gt_constraints = extract_gt_qrr_constraints(questions)
    gt_details, gt_agg = check_constraints_against_positions(
        positions, gt_constraints, tau=tau
    )
    gt_agg["details"] = gt_details

    return {
        "scene_id": scene_id,
        "model": result.get("model", recon.get("model", "unknown")),
        "n_objects": result.get("n_objects", len(positions)),
        "positions": positions_raw,
        "recon_status": recon.get("status"),
        "recon_feasible": recon.get("feasible"),
        "belief_analysis": belief_agg,
        "gt_analysis": gt_agg,
    }


# ── 批量分析 ──

def analyze_all_scenes(
    recon_dir: str,
    results_dir: str,
    questions_dir: str,
    output_dir: Optional[str] = None,
    tau: float = 0.10,
    max_scenes: Optional[int] = None,
) -> List[dict]:
    """批量分析所有场景的约束违背。"""
    recon_path = Path(recon_dir)
    results_path = Path(results_dir) / "scenes"

    if not recon_path.exists():
        print(f"Reconstruction directory not found: {recon_dir}")
        return []
    if not results_path.exists():
        print(f"Results directory not found: {results_path}")
        return []

    # 枚举重建结果
    recon_files = sorted(recon_path.glob("*.json"))
    if max_scenes:
        recon_files = recon_files[:max_scenes]

    all_outputs = []
    for i, rf in enumerate(recon_files):
        scene_id = rf.stem
        result_file = results_path / f"{scene_id}.json"

        if not result_file.exists():
            print(f"  [{i+1}/{len(recon_files)}] {scene_id}: no scoring result, skipping")
            continue

        try:
            output = analyze_single_scene(
                recon_path=str(rf),
                result_path=str(result_file),
                questions_dir=questions_dir,
                tau=tau,
            )
            if output is None:
                print(f"  [{i+1}/{len(recon_files)}] {scene_id}: no positions or questions, skipping")
                continue

            ba = output["belief_analysis"]
            ga = output["gt_analysis"]
            print(
                f"  [{i+1}/{len(recon_files)}] {scene_id}: "
                f"belief CSR={ba['csr']:.3f} ({ba['n_satisfied']}/{ba['n_total']})  "
                f"GT CSR={ga['csr']:.3f} ({ga['n_satisfied']}/{ga['n_total']})"
            )
            all_outputs.append(output)

            # 逐场景保存
            if output_dir:
                out_path = Path(output_dir)
                out_path.mkdir(parents=True, exist_ok=True)
                with open(out_path / f"{scene_id}.json", "w") as f:
                    json.dump(output, f, indent=2, ensure_ascii=False)

        except Exception as e:
            print(f"  [{i+1}/{len(recon_files)}] {scene_id}: ERROR {e}")
            continue

    # 跨场景汇总
    if all_outputs:
        summary = summarize_violations(all_outputs)
        print(f"\n=== Summary ({len(all_outputs)} scenes) ===")
        for mode in ("belief", "gt"):
            s = summary[mode]
            print(f"\n  {mode.upper()} constraints:")
            print(f"    CSR mean: {s['csr_mean']:.4f} ± {s['csr_std']:.4f}")
            print(f"    Total: {s['n_total_sum']} constraints, {s['n_violated_sum']} violated")
            if s.get("confusion_matrix_sum"):
                print(f"    Confusion matrix (sum):")
                for key, val in sorted(s["confusion_matrix_sum"].items()):
                    if val > 0:
                        print(f"      {key}: {val}")

        if output_dir:
            with open(Path(output_dir) / "summary.json", "w") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            print(f"\nSaved to {output_dir}")

    return all_outputs


def summarize_violations(results: List[dict]) -> dict:
    """跨场景汇总约束违背统计。"""
    summary = {}

    for mode in ("belief", "gt"):
        key = f"{mode}_analysis"
        csrs = [r[key]["csr"] for r in results if key in r]
        n_totals = [r[key]["n_total"] for r in results if key in r]
        n_viols = [r[key]["n_violated"] for r in results if key in r]

        # 合并混淆矩阵
        cm_sum = defaultdict(int)
        for r in results:
            if key not in r:
                continue
            for k, v in r[key].get("confusion_matrix", {}).items():
                cm_sum[k] += v

        # 按 variant 汇总
        variant_agg = defaultdict(lambda: {"n_total": 0, "n_satisfied": 0})
        for r in results:
            if key not in r:
                continue
            for v, stats in r[key].get("by_variant", {}).items():
                variant_agg[v]["n_total"] += stats["n_total"]
                variant_agg[v]["n_satisfied"] += stats["n_satisfied"]
        variant_out = {}
        for v, stats in variant_agg.items():
            stats["csr"] = round(stats["n_satisfied"] / stats["n_total"], 4) if stats["n_total"] > 0 else 0.0
            variant_out[v] = dict(stats)

        # 按 split 汇总
        by_split = defaultdict(list)
        for r in results:
            if key not in r:
                continue
            split = r["scene_id"].rsplit("_", 1)[0]
            by_split[split].append(r[key]["csr"])
        split_out = {}
        for s, vals in sorted(by_split.items()):
            split_out[s] = {
                "n_scenes": len(vals),
                "csr_mean": round(float(np.mean(vals)), 4),
            }

        summary[mode] = {
            "n_scenes": len(csrs),
            "csr_mean": round(float(np.mean(csrs)), 4) if csrs else 0.0,
            "csr_std": round(float(np.std(csrs)), 4) if csrs else 0.0,
            "csr_median": round(float(np.median(csrs)), 4) if csrs else 0.0,
            "n_total_sum": sum(n_totals),
            "n_violated_sum": sum(n_viols),
            "confusion_matrix_sum": dict(cm_sum),
            "by_variant": variant_out,
            "by_split": split_out,
        }

    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze QRR constraint violations from reconstructed positions"
    )
    parser.add_argument("--recon-dir", "-r", required=True,
                        help="Path to belief_recon output directory (e.g. output/analysis/belief_recon/model)")
    parser.add_argument("--results-dir", "-R", required=True,
                        help="Path to model results directory (e.g. output/results/model)")
    parser.add_argument("--questions-dir", "-q",
                        default="output/questions",
                        help="Path to questions directory")
    parser.add_argument("--output-dir", "-o", default=None,
                        help="Output directory for violation analysis")
    parser.add_argument("--tau", type=float, default=0.10,
                        help="Comparator tolerance (default: 0.10)")
    parser.add_argument("--max-scenes", type=int, default=None)

    args = parser.parse_args()

    analyze_all_scenes(
        recon_dir=args.recon_dir,
        results_dir=args.results_dir,
        questions_dir=args.questions_dir,
        output_dir=args.output_dir,
        tau=args.tau,
        max_scenes=args.max_scenes,
    )
