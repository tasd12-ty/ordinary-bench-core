"""
FDR 与 QRR 约束冲突分析。

对所有有 SVG 输出（成功重建）的场景，检测 FDR 分解产生的 shared_anchor QRR
与直接 QRR 问题在同一对物体对上的矛盾，按场景和模型分类统计。

冲突定义：FDR 分解出的某条 QRR 约束和直接 QRR 约束覆盖了相同的两对物体对，
但比较符方向相反（< vs >）。~= 与 </>不算冲突。
"""

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from reconstruct.preparation import (
    load_questions_auto,
    prepare_reconstruction_input_from_scoring,
)


def _canonical_pair_key(pair1, pair2) -> Tuple[Tuple[str, str], Tuple[str, str]]:
    """将两个 pair 规范化为可比较的 key。"""
    p1 = tuple(sorted(pair1))
    p2 = tuple(sorted(pair2))
    # 保证 pair1 <= pair2 的字典序
    if p1 > p2:
        return p2, p1, True  # swapped
    return p1, p2, False


def find_fdr_qrr_conflicts(
    qrr_direct: List[dict],
    qrr_from_fdr: List[dict],
) -> List[dict]:
    """查找 FDR 分解 QRR 与直接 QRR 之间的冲突。

    Returns:
        冲突列表，每条包含 direct 和 fdr 两侧的约束详情。
    """
    # 索引直接 QRR：canonical key -> list of constraints
    direct_index: Dict[tuple, List[dict]] = defaultdict(list)
    for c in qrr_direct:
        p1, p2, swapped = _canonical_pair_key(c["pair1"], c["pair2"])
        cmp = c["comparator"]
        if swapped:
            cmp = {"<": ">", ">": "<", "~=": "~="}.get(cmp, cmp)
        direct_index[(p1, p2)].append({**c, "_canon_cmp": cmp})

    conflicts = []
    for c in qrr_from_fdr:
        p1, p2, swapped = _canonical_pair_key(c["pair1"], c["pair2"])
        fdr_cmp = c["comparator"]
        if swapped:
            fdr_cmp = {"<": ">", ">": "<", "~=": "~="}.get(fdr_cmp, fdr_cmp)

        for dc in direct_index.get((p1, p2), []):
            direct_cmp = dc["_canon_cmp"]
            # 冲突：方向相反 (< vs >)
            is_conflict = (
                (direct_cmp == "<" and fdr_cmp == ">") or
                (direct_cmp == ">" and fdr_cmp == "<")
            )
            if is_conflict:
                conflicts.append({
                    "pair1": list(p1),
                    "pair2": list(p2),
                    "direct_qid": dc.get("qid", ""),
                    "direct_cmp": dc["comparator"],
                    "direct_variant": dc.get("variant", "disjoint"),
                    "fdr_qid": c.get("qid", ""),
                    "fdr_source_qid": c.get("source_qid", ""),
                    "fdr_cmp": c["comparator"],
                    "fdr_anchor": c.get("anchor", ""),
                })

    return conflicts


def analyze_scene_conflicts(
    result_path: str,
    questions_dir: str,
    mode: str = "belief",
) -> dict:
    """分析单个场景的 FDR-QRR 冲突。

    Args:
        mode: "belief" 用 VLM 预测, "gt" 用真值
    """
    with open(result_path) as f:
        result = json.load(f)

    scene_id = result.get("scene_id", "unknown")
    questions, _ = load_questions_auto(questions_dir, scene_id)
    if not questions:
        return {"scene_id": scene_id, "error": "no_questions"}

    scoring = result.get("scores", result)

    if mode == "belief":
        prepared = prepare_reconstruction_input_from_scoring(
            scoring_result=scoring,
            questions=questions,
            use_correct_only=False,
        )
    else:
        # GT 模式：直接从问题元数据提取
        prepared = prepare_reconstruction_input_from_scoring(
            scoring_result=scoring,
            questions=questions,
            use_correct_only=True,
        )

    conflicts = find_fdr_qrr_conflicts(
        qrr_direct=prepared.qrr_constraints,
        qrr_from_fdr=prepared.qrr_from_fdr,
    )

    return {
        "scene_id": scene_id,
        "n_qrr_direct": len(prepared.qrr_constraints),
        "n_qrr_from_fdr": len(prepared.qrr_from_fdr),
        "n_conflicts": len(conflicts),
        "conflicts": conflicts,
    }


def analyze_all_models(
    recon_base: str,
    results_base: str,
    questions_dir: str,
    mode: str = "belief",
) -> dict:
    """对所有有 SVG 的场景分析 FDR-QRR 冲突。"""
    recon_root = Path(recon_base)
    results_root = Path(results_base)

    all_results = {}

    for model_dir in sorted(recon_root.iterdir()):
        if not model_dir.is_dir():
            continue
        model = model_dir.name

        # 找有 SVG 的场景
        svg_scenes = sorted(s.stem for s in model_dir.glob("*.svg"))
        if not svg_scenes:
            continue

        results_dir = results_root / model / "scenes"
        if not results_dir.exists():
            continue

        model_results = []
        for scene_id in svg_scenes:
            result_file = results_dir / f"{scene_id}.json"
            if not result_file.exists():
                continue
            try:
                r = analyze_scene_conflicts(
                    result_path=str(result_file),
                    questions_dir=questions_dir,
                    mode=mode,
                )
                model_results.append(r)
            except Exception as e:
                print(f"  {model}/{scene_id}: ERROR {e}")

        all_results[model] = model_results

    return all_results


def print_report(all_results: dict):
    """打印汇总报告。"""
    print("=" * 90)
    print(f"{'Model':<45} {'Scenes':>6} {'w/Conflict':>10} {'Conflicts':>9} {'Rate':>8}")
    print("=" * 90)

    grand_scenes = 0
    grand_conflict_scenes = 0
    grand_conflicts = 0

    for model in sorted(all_results.keys()):
        results = all_results[model]
        n_scenes = len(results)
        n_with_conflict = sum(1 for r in results if r["n_conflicts"] > 0)
        n_conflicts = sum(r["n_conflicts"] for r in results)
        rate = n_with_conflict / n_scenes if n_scenes > 0 else 0

        grand_scenes += n_scenes
        grand_conflict_scenes += n_with_conflict
        grand_conflicts += n_conflicts

        print(f"{model:<45} {n_scenes:>6} {n_with_conflict:>10} {n_conflicts:>9} {rate:>7.1%}")

    print("-" * 90)
    grand_rate = grand_conflict_scenes / grand_scenes if grand_scenes > 0 else 0
    print(f"{'TOTAL':<45} {grand_scenes:>6} {grand_conflict_scenes:>10} {grand_conflicts:>9} {grand_rate:>7.1%}")
    print()

    # 逐场景明细（只打印有冲突的）
    print("\n=== Conflict Details ===\n")
    for model in sorted(all_results.keys()):
        has_any = False
        for r in all_results[model]:
            if r["n_conflicts"] == 0:
                continue
            if not has_any:
                print(f"--- {model} ---")
                has_any = True
            print(f"  {r['scene_id']}: {r['n_conflicts']} conflicts "
                  f"(direct={r['n_qrr_direct']}, fdr={r['n_qrr_from_fdr']})")
            for c in r["conflicts"]:
                print(f"    {c['pair1']} vs {c['pair2']}: "
                      f"direct={c['direct_cmp']} (qid={c['direct_qid']})  "
                      f"fdr={c['fdr_cmp']} (anchor={c['fdr_anchor']}, src={c['fdr_source_qid']})")
        if has_any:
            print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze FDR vs QRR constraint conflicts")
    parser.add_argument("--recon-base", default="output/analysis/belief_recon",
                        help="Base directory for reconstruction outputs")
    parser.add_argument("--results-base", default="output/results",
                        help="Base directory for scoring results")
    parser.add_argument("--questions-dir", "-q", default="output/questions",
                        help="Path to questions directory")
    parser.add_argument("--mode", choices=["belief", "gt"], default="belief",
                        help="Constraint source: belief (VLM predictions) or gt")

    args = parser.parse_args()

    all_results = analyze_all_models(
        recon_base=args.recon_base,
        results_base=args.results_base,
        questions_dir=args.questions_dir,
        mode=args.mode,
    )

    print_report(all_results)
