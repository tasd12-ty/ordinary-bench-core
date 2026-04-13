"""
Variable-Size Subset Ablation: 可视化分析结果。

读取 analyze_variable_subsets.py 的输出，生成 4 张图:
  Fig 1: QRR 精度 vs 子集大小 (带误差条)
  Fig 2: 拒答检测率 + Hallucination Rate vs 子集大小
  Fig 3: 可答题比例曲线 (解析 + 实测)
  Fig 4: 跨尺寸答案一致性分布

用法:
    cd experiments/variable_subset_ablation

    python visualize_variable_subsets.py \
        --analysis-dir output/analysis \
        --output-dir output/figures

    # 也可指定 N=20 基线精度 (来自全图评测)
    python visualize_variable_subsets.py \
        --analysis-dir output/analysis \
        --baseline-acc 0.65 \
        --output-dir output/figures

依赖:
    matplotlib, numpy
"""

import argparse
import json
from math import comb
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── 配色方案 (与 constraint_perturbation 实验一致) ──
COLOR_OVERALL = "#2196F3"    # 蓝
COLOR_DISJOINT = "#4CAF50"   # 绿
COLOR_SHARED = "#FF9800"     # 橙
COLOR_REFUSAL = "#4CAF50"    # 绿
COLOR_HALLUC = "#F44336"     # 红
COLOR_BASELINE = "#9E9E9E"   # 灰
COLOR_THEORY = "#9C27B0"     # 紫


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def fig_accuracy_vs_size(accuracy: dict, baseline_acc: float, out_dir: Path):
    """
    Fig 1: QRR 精度 vs 子集大小。
    三条线: overall, disjoint, shared_anchor。
    误差条: per_scene_acc 的 95% CI (mean ± 1.96 * std / sqrt(n))。
    """
    sizes = sorted(int(s) for s in accuracy.keys())
    overall_acc = []
    disjoint_acc = []
    shared_acc = []
    overall_ci = []

    for s in sizes:
        entry = accuracy[str(s)]
        overall_acc.append(entry["overall"]["acc"])
        disjoint_acc.append(entry["disjoint"]["acc"])
        shared_acc.append(entry["shared_anchor"]["acc"])

        # 95% CI from per-scene accuracy
        scene_accs = entry.get("per_scene_acc", [])
        if len(scene_accs) >= 2:
            std = np.std(scene_accs, ddof=1)
            ci = 1.96 * std / np.sqrt(len(scene_accs))
        else:
            ci = 0
        overall_ci.append(ci)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.errorbar(sizes, overall_acc, yerr=overall_ci, fmt="o-",
                color=COLOR_OVERALL, label="Overall", linewidth=2,
                markersize=6, capsize=4)
    ax.plot(sizes, disjoint_acc, "s--", color=COLOR_DISJOINT,
            label="Disjoint", linewidth=1.5, markersize=5)
    ax.plot(sizes, shared_acc, "^--", color=COLOR_SHARED,
            label="Shared Anchor", linewidth=1.5, markersize=5)

    if baseline_acc > 0:
        ax.axhline(y=baseline_acc, color=COLOR_BASELINE, linestyle=":",
                   linewidth=1.5, label=f"N=20 Baseline ({baseline_acc:.2f})")

    ax.set_xlabel("Subset Size (number of objects)", fontsize=12)
    ax.set_ylabel("QRR Accuracy", fontsize=12)
    ax.set_title("QRR Accuracy vs Subset Size", fontsize=14, fontweight="bold")
    ax.set_xticks(sizes)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(out_dir / "fig_accuracy_vs_size.pdf", dpi=150)
    plt.savefig(out_dir / "fig_accuracy_vs_size.png", dpi=150)
    plt.close()
    print("  Saved fig_accuracy_vs_size.{pdf,png}")


def fig_refusal_vs_size(refusal: dict, out_dir: Path):
    """
    Fig 2: 拒答检测率 + Hallucination Rate vs 子集大小。
    双 Y 轴或同轴。
    """
    sizes = sorted(int(s) for s in refusal.keys())
    refusal_rates = [refusal[str(s)]["refusal_rate"] for s in sizes]
    halluc_rates = [refusal[str(s)]["hallucination_rate"] for s in sizes]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(sizes, refusal_rates, "o-", color=COLOR_REFUSAL,
            label="Correct Refusal Rate", linewidth=2, markersize=6)
    ax.plot(sizes, halluc_rates, "s-", color=COLOR_HALLUC,
            label="Hallucination Rate", linewidth=2, markersize=6)

    # 填充区域
    ax.fill_between(sizes, refusal_rates, alpha=0.1, color=COLOR_REFUSAL)
    ax.fill_between(sizes, halluc_rates, alpha=0.1, color=COLOR_HALLUC)

    ax.set_xlabel("Subset Size (number of objects)", fontsize=12)
    ax.set_ylabel("Rate", fontsize=12)
    ax.set_title("Refusal Detection & Hallucination Rate vs Subset Size",
                 fontsize=14, fontweight="bold")
    ax.set_xticks(sizes)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(out_dir / "fig_refusal_vs_size.pdf", dpi=150)
    plt.savefig(out_dir / "fig_refusal_vs_size.png", dpi=150)
    plt.close()
    print("  Saved fig_refusal_vs_size.{pdf,png}")


def fig_answerable_ratio(refusal: dict, n_parent: int, out_dir: Path):
    """
    Fig 3: 可答题比例曲线。
    实测值 (来自 refusal 数据) vs 理论值 (组合数公式)。

    理论可答比例:
      disjoint (4物体): C(m,4) / C(N,4)
      shared_anchor (3物体): C(m,3) / C(N,3)
      加权平均 ≈ (C(m,4)*3 + m*C(m-1,2)) / (C(N,4)*3 + N*C(N-1,2))
    """
    sizes = sorted(int(s) for s in refusal.keys())

    # 实测可答比例
    actual_ratio = [refusal[str(s)]["answerable_ratio"] for s in sizes]

    # 理论可答比例 (加权)
    n = n_parent
    total_parent = comb(n, 4) * 3 + n * comb(n - 1, 2)
    theory_ratio = []
    for m in sizes:
        total_m = comb(m, 4) * 3 + m * comb(m - 1, 2)
        theory_ratio.append(total_m / total_parent if total_parent > 0 else 0)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(sizes, actual_ratio, "o-", color=COLOR_OVERALL,
            label="Actual (from evaluation)", linewidth=2, markersize=6)
    ax.plot(sizes, theory_ratio, "s--", color=COLOR_THEORY,
            label="Theoretical (combinatorial)", linewidth=1.5, markersize=5)

    ax.set_xlabel("Subset Size (number of objects)", fontsize=12)
    ax.set_ylabel("Answerable Ratio", fontsize=12)
    ax.set_title(f"Answerable Question Ratio vs Subset Size (Parent N={n})",
                 fontsize=14, fontweight="bold")
    ax.set_xticks(sizes)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(out_dir / "fig_answerable_ratio.pdf", dpi=150)
    plt.savefig(out_dir / "fig_answerable_ratio.png", dpi=150)
    plt.close()
    print("  Saved fig_answerable_ratio.{pdf,png}")


def fig_consistency_distribution(consistency: dict, out_dir: Path):
    """
    Fig 4: 跨尺寸一致性分布直方图。
    每个 question 有一个 consistency 值 (0~1)，画分布。
    """
    per_q = consistency.get("per_question", [])
    if not per_q:
        print("  SKIP fig_consistency: no multi-size questions")
        return

    values = [q["consistency"] for q in per_q]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(values, bins=20, range=(0, 1), color=COLOR_OVERALL,
            alpha=0.7, edgecolor="white", linewidth=0.5)

    mean_val = np.mean(values)
    ax.axvline(x=mean_val, color=COLOR_HALLUC, linestyle="--",
               linewidth=2, label=f"Mean = {mean_val:.3f}")

    ax.set_xlabel("Consistency (fraction of majority answer)", fontsize=12)
    ax.set_ylabel("Number of Questions", fontsize=12)
    ax.set_title(f"Cross-Size Answer Consistency Distribution "
                 f"(N={len(values)} questions)",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(out_dir / "fig_consistency_distribution.pdf", dpi=150)
    plt.savefig(out_dir / "fig_consistency_distribution.png", dpi=150)
    plt.close()
    print("  Saved fig_consistency_distribution.{pdf,png}")


def main():
    parser = argparse.ArgumentParser(
        description="Variable-Size Subset Ablation: 可视化分析结果"
    )
    parser.add_argument("--analysis-dir", required=True,
                        help="分析结果目录 (analyze_variable_subsets.py 输出)")
    parser.add_argument("--output-dir", default=None,
                        help="图表输出目录 (默认与 analysis-dir 相同)")
    parser.add_argument("--baseline-acc", type=float, default=0.0,
                        help="N=20 全场景基线精度 (可选, 画水平参考线)")
    parser.add_argument("--n-parent", type=int, default=20,
                        help="父场景物体数 (默认 20, 用于理论曲线)")
    args = parser.parse_args()

    analysis_dir = Path(args.analysis_dir)
    out_dir = Path(args.output_dir) if args.output_dir else analysis_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # 加载分析结果
    accuracy = load_json(analysis_dir / "accuracy_by_size.json")
    refusal = load_json(analysis_dir / "refusal_by_size.json")
    consistency = load_json(analysis_dir / "consistency.json")

    print(f"Generating figures to {out_dir} ...")

    fig_accuracy_vs_size(accuracy, args.baseline_acc, out_dir)
    fig_refusal_vs_size(refusal, out_dir)
    fig_answerable_ratio(refusal, args.n_parent, out_dir)
    fig_consistency_distribution(consistency, out_dir)

    print(f"\nAll figures saved to {out_dir}")


if __name__ == "__main__":
    main()
