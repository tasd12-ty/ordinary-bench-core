"""
分析流程入口。

对已评估的模型依次运行以下分析：
  1. 准确率汇总（Table 1）
  2. 批量场景重建（Table 2，Figure 2-4）
  3. 一致性分析（Table 4）
  4. 可视化生成

用法：
    python analysis/run_analysis.py [--models MODEL_DIR ...] [--max-scenes N]
"""

import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "API-test"))

from analysis.aggregate import (
    compute_accuracy_table,
    format_accuracy_table_markdown,
    load_scene_results,
    load_questions,
    per_scene_accuracy,
)
from analysis.consistency import analyze_scene_consistency, check_transitivity, check_reciprocity
from analysis.reconstruct_scenes import (
    reconstruct_single_scene,
    load_scene_gt,
    summarize_reconstructions,
)


def discover_models(results_base: str) -> dict:
    """自动发现模型评估结果目录。"""
    base = Path(results_base)
    models = {}
    for d in sorted(base.iterdir()):
        if d.is_dir() and (d / "scenes").exists():
            short = d.name.replace("qwen--", "").replace("-thinking", "")
            models[short] = str(d)
    return models


def run_full_analysis(
    results_base: str = "output/results",
    questions_dir: str = "output/questions",
    scenes_dir: str = "../data-gen/output/scenes",
    output_dir: str = "output/analysis",
    max_scenes: int = None,
    n_restarts: int = 10,
):
    """运行完整分析流程。"""
    os.makedirs(output_dir, exist_ok=True)

    models = discover_models(results_base)
    if not models:
        print(f"No model directories found in {results_base}")
        return

    print(f"Found {len(models)} models: {list(models.keys())}")

    # ════════════════════════════════════════════
    # 1. ACCURACY TABLE
    # ════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("1. ACCURACY ANALYSIS")
    print("=" * 60)

    accuracy_table = compute_accuracy_table(models)
    md = format_accuracy_table_markdown(accuracy_table)
    print(md)

    with open(os.path.join(output_dir, "accuracy_table.json"), "w") as f:
        json.dump(accuracy_table, f, indent=2)
    with open(os.path.join(output_dir, "accuracy_table.md"), "w") as f:
        f.write(md)

    # ════════════════════════════════════════════
    # 2. RECONSTRUCTION ANALYSIS
    # ════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("2. SCENE BELIEF RECONSTRUCTION")
    print("=" * 60)

    all_recon = {}
    for model_name, model_dir in models.items():
        print(f"\n--- Model: {model_name} ---")
        scenes = load_scene_results(model_dir)
        if max_scenes:
            scenes = scenes[:max_scenes]

        recon_results = []
        for i, scene in enumerate(scenes):
            sid = scene["scene_id"]
            questions = load_questions(questions_dir, sid)
            if not questions:
                continue

            gt_path = os.path.join(scenes_dir, f"{sid}.json")
            gt = load_scene_gt(gt_path) if os.path.exists(gt_path) else None

            try:
                # 信念重建（使用所有 VLM 预测）
                result = reconstruct_single_scene(
                    scene, questions, gt,
                    use_correct_only=False,
                    n_restarts=n_restarts,
                )

                status = result["status"]
                m = result["metrics"]
                nrms_str = f'{m["nrms"]:.4f}' if m.get("nrms") is not None else "N/A"
                tau_str = f'{m["kendall_tau"]:.3f}' if m.get("kendall_tau") is not None else "N/A"
                print(f"  [{i+1}/{len(scenes)}] {sid}: "
                      f"status={status} csr={m['csr_qrr']:.3f}/{m['csr_trr']:.3f} "
                      f"nrms={nrms_str} tau={tau_str}")

                recon_results.append(result)
            except Exception as e:
                print(f"  [{i+1}/{len(scenes)}] {sid}: ERROR {e}")

        all_recon[model_name] = recon_results

        # 输出摘要
        if recon_results:
            summary = summarize_reconstructions(recon_results)
            print(f"\n  Summary ({model_name}):")
            print(f"    Feasible: {summary['feasible_rate']:.1%}")
            print(f"    Status: {summary['status_counts']}")
            for k in ["csr_qrr", "csr_trr", "kendall_tau", "nrms"]:
                if f"{k}_mean" in summary:
                    print(f"    {k}: {summary[f'{k}_mean']:.4f} "
                          f"± {summary.get(f'{k}_std', 0):.4f}")

            # 保存结果
            model_out = os.path.join(output_dir, f"recon_{model_name}.json")
            with open(model_out, "w") as f:
                json.dump(recon_results, f, indent=2, default=str)

            summary_out = os.path.join(output_dir, f"recon_summary_{model_name}.json")
            with open(summary_out, "w") as f:
                json.dump(summary, f, indent=2, default=str)

    # ════════════════════════════════════════════
    # 3. CONSISTENCY ANALYSIS
    # ════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("3. CONSISTENCY ANALYSIS")
    print("=" * 60)

    all_consistency = {}
    for model_name, model_dir in models.items():
        print(f"\n--- Model: {model_name} ---")
        scenes = load_scene_results(model_dir)
        if max_scenes:
            scenes = scenes[:max_scenes]

        total_trans_chains = total_trans_viol = 0
        total_recip_pairs = total_recip_exact = total_recip_viol = 0

        for scene in scenes:
            sid = scene["scene_id"]
            questions = load_questions(questions_dir, sid)
            if not questions:
                continue

            result = analyze_scene_consistency(scene, questions)
            t = result["transitivity"]
            r = result["reciprocity"]

            total_trans_chains += t["n_chains"]
            total_trans_viol += t["n_violated"]
            total_recip_pairs += r["n_pairs"]
            total_recip_exact += r["n_exact_reciprocal"]
            total_recip_viol += r["n_violated"]

        trans_rate = total_trans_viol / total_trans_chains if total_trans_chains > 0 else 0
        recip_exact = total_recip_exact / total_recip_pairs if total_recip_pairs > 0 else 0
        recip_viol = total_recip_viol / total_recip_pairs if total_recip_pairs > 0 else 0

        print(f"  Transitivity: {total_trans_chains} chains, "
              f"{total_trans_viol} violations ({trans_rate:.1%})")
        print(f"  Reciprocity:  {total_recip_pairs} pairs, "
              f"{total_recip_exact} exact ({recip_exact:.1%}), "
              f"{total_recip_viol} violated ({recip_viol:.1%})")

        all_consistency[model_name] = {
            "transitivity": {
                "n_chains": total_trans_chains,
                "n_violated": total_trans_viol,
                "violation_rate": trans_rate,
            },
            "reciprocity": {
                "n_pairs": total_recip_pairs,
                "n_exact": total_recip_exact,
                "exact_rate": recip_exact,
                "n_violated": total_recip_viol,
                "violation_rate": recip_viol,
            },
        }

    with open(os.path.join(output_dir, "consistency.json"), "w") as f:
        json.dump(all_consistency, f, indent=2)

    # ════════════════════════════════════════════
    # 4. VISUALIZATION
    # ════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("4. VISUALIZATION")
    print("=" * 60)

    try:
        from analysis.visualize import (
            plot_configuration_comparison,
            plot_accuracy_vs_complexity,
            plot_reconstruction_quality_distribution,
        )

        viz_dir = os.path.join(output_dir, "figures")
        os.makedirs(viz_dir, exist_ok=True)

        # Figure 1：准确率 vs 复杂度
        for model_name, model_dir in models.items():
            psa = per_scene_accuracy(model_dir)
            if psa:
                plot_accuracy_vs_complexity(
                    psa, model_name,
                    save_path=os.path.join(viz_dir, f"accuracy_vs_n_{model_name}.png")
                )
                print(f"  Saved accuracy plot for {model_name}")

        # Figure 2：配置对比图（每个模型取前 3 个场景）
        for model_name, recon_results in all_recon.items():
            for result in recon_results[:3]:
                sid = result["scene_id"]
                gt_path = os.path.join(scenes_dir, f"{sid}.json")
                if not os.path.exists(gt_path):
                    continue
                gt = load_scene_gt(gt_path)
                if gt is None:
                    continue

                recon_pos = {k: np.array(v) for k, v in result["positions"].items()}
                if not recon_pos:
                    continue

                plot_configuration_comparison(
                    gt, recon_pos, sid,
                    metrics=result["metrics"],
                    save_path=os.path.join(viz_dir, f"config_{model_name}_{sid}.png"),
                )
                print(f"  Saved config comparison: {model_name}/{sid}")

        # Figure 3：重建质量分布图
        for model_name, recon_results in all_recon.items():
            if recon_results:
                plot_reconstruction_quality_distribution(
                    recon_results, model_name,
                    save_path=os.path.join(viz_dir, f"recon_quality_{model_name}.png"),
                )
                print(f"  Saved quality distribution for {model_name}")

    except ImportError:
        print("  matplotlib not available, skipping visualizations")

    print("\n" + "=" * 60)
    print(f"Analysis complete. Results saved to {output_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="运行完整 Ordinary-Bench 分析流程")
    parser.add_argument("--results-base", default="output/results")
    parser.add_argument("--questions-dir", default="output/questions")
    parser.add_argument("--scenes-dir", default="../data-gen/output/scenes")
    parser.add_argument("--output-dir", default="output/analysis")
    parser.add_argument("--max-scenes", type=int, default=None,
                        help="限制每个模型处理的场景数量（用于测试）")
    parser.add_argument("--restarts", type=int, default=10)
    args = parser.parse_args()

    run_full_analysis(
        results_base=args.results_base,
        questions_dir=args.questions_dir,
        scenes_dir=args.scenes_dir,
        output_dir=args.output_dir,
        max_scenes=args.max_scenes,
        n_restarts=args.restarts,
    )
