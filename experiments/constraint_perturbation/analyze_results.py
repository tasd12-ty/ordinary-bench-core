#!/usr/bin/env python3
"""分析约束扰动实验结果，生成相变曲线和 Null Model 对比。

用法:
    python analyze_results.py
    python analyze_results.py --results results/perturbation_results.jsonl
"""

import argparse
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"
DEFAULT_RESULTS = RESULTS_DIR / "perturbation_results.jsonl"


def load_results(path: Path) -> list:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ── Fig A: Phase Transition Curve ────────────────────────────────


def plot_phase_transition(records, out_dir):
    """feasibility vs p, grouped by N."""
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, 7))

    groups = defaultdict(list)
    for r in records:
        groups[(r["n_objects"], r["fraction"])].append(r["feasible"])

    n_values = sorted(set(r["n_objects"] for r in records))
    fractions = sorted(set(r["fraction"] for r in records))

    for i, n in enumerate(n_values):
        rates = []
        for f in fractions:
            vals = groups.get((n, f), [])
            rates.append(np.mean(vals) if vals else 0)
        ax.plot(fractions, rates, "o-", label=f"N={n}",
                color=colors[i % len(colors)], linewidth=2, markersize=4)

    ax.set_xlabel("Perturbation fraction (p)", fontsize=12)
    ax.set_ylabel("Feasible scene fraction", fontsize=12)
    ax.set_title("Phase Transition: Feasibility vs Perturbation Level\n(consistent-flip, QRR only)", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.02, 0.52)
    ax.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    plt.savefig(out_dir / "fig_phase_transition.pdf", dpi=150)
    plt.savefig(out_dir / "fig_phase_transition.png", dpi=150)
    plt.close()
    print("  Saved fig_phase_transition.pdf")


# ── Fig B: Quality Degradation ───────────────────────────────────


def plot_quality_degradation(records, out_dir):
    """CSR, Kendall tau, NRMS vs p."""
    metrics = [
        ("csr_qrr", "CSR (QRR)"),
        ("kendall_tau", "Kendall τ"),
        ("nrms", "NRMS"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, 7))

    fractions = sorted(set(r["fraction"] for r in records))
    n_values = sorted(set(r["n_objects"] for r in records))

    for ax, (key, label) in zip(axes, metrics):
        groups = defaultdict(list)
        for r in records:
            val = r.get(key)
            if val is not None:
                groups[(r["n_objects"], r["fraction"])].append(val)

        for i, n in enumerate(n_values):
            means = []
            for f in fractions:
                vals = groups.get((n, f), [])
                means.append(np.mean(vals) if vals else float("nan"))
            ax.plot(fractions, means, "o-", label=f"N={n}",
                    color=colors[i % len(colors)], linewidth=1.5, markersize=3)

        ax.set_xlabel("Perturbation fraction (p)")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Reconstruction Quality vs Perturbation Level", fontsize=13)
    plt.tight_layout()
    plt.savefig(out_dir / "fig_quality_degradation.pdf", dpi=150)
    plt.savefig(out_dir / "fig_quality_degradation.png", dpi=150)
    plt.close()
    print("  Saved fig_quality_degradation.pdf")


# ── Fig C: Saturation Curve ──────────────────────────────────────


def plot_saturation(records, out_dir):
    """consistent_flip saturation (actual/target) vs p, by N."""
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, 7))

    groups = defaultdict(list)
    for r in records:
        if r["fraction"] > 0:
            groups[(r["n_objects"], r["fraction"])].append(r.get("saturation", 1.0))

    n_values = sorted(set(r["n_objects"] for r in records))
    fractions = sorted(set(r["fraction"] for r in records if r["fraction"] > 0))

    for i, n in enumerate(n_values):
        means = []
        for f in fractions:
            vals = groups.get((n, f), [])
            means.append(np.mean(vals) if vals else 1.0)
        ax.plot(fractions, means, "o-", label=f"N={n}",
                color=colors[i % len(colors)], linewidth=2, markersize=4)

    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Target perturbation fraction (p)", fontsize=12)
    ax.set_ylabel("Saturation (actual / target flips)", fontsize=12)
    ax.set_title("Consistent-Flip Saturation: Structural Rigidity", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.10)

    plt.tight_layout()
    plt.savefig(out_dir / "fig_saturation.pdf", dpi=150)
    plt.savefig(out_dir / "fig_saturation.png", dpi=150)
    plt.close()
    print("  Saved fig_saturation.pdf")


# ── Summary Table ────────────────────────────────────────────────


def print_summary_table(records):
    """按 (N, p) 聚合的汇总表。"""
    groups = defaultdict(list)
    for r in records:
        groups[(r["n_objects"], r["fraction"])].append(r)

    header = f"{'N':>3} | {'p':>5} | {'trials':>6} | {'feas%':>6} | {'sat':>5} | {'CSR':>6} | {'tau':>6} | {'NRMS':>6} | {'gt_sat':>6}"
    print(f"\n{header}")
    print("-" * len(header))

    prev_n = None
    for key in sorted(groups.keys()):
        n, p = key
        if prev_n is not None and n != prev_n:
            print("-" * len(header))
        prev_n = n

        vals = groups[key]
        feas = np.mean([v["feasible"] for v in vals])
        sat_vals = [v.get("saturation", 1.0) for v in vals if v["fraction"] > 0]
        sat = np.mean(sat_vals) if sat_vals else 1.0
        csr_vals = [v["csr_qrr"] for v in vals if v.get("csr_qrr") is not None]
        tau_vals = [v["kendall_tau"] for v in vals if v.get("kendall_tau") is not None]
        nrms_vals = [v["nrms"] for v in vals if v.get("nrms") is not None]
        gt_sat_vals = [v.get("gt_satisfaction", 1.0) for v in vals]

        csr = np.mean(csr_vals) if csr_vals else float("nan")
        tau = np.mean(tau_vals) if tau_vals else float("nan")
        nrms = np.mean(nrms_vals) if nrms_vals else float("nan")
        gt_sat = np.mean(gt_sat_vals)

        print(f"{n:>3} | {p:>5.2f} | {len(vals):>6} | {feas:>5.1%} | {sat:>5.2f} | {csr:>6.3f} | {tau:>6.3f} | {nrms:>6.3f} | {gt_sat:>6.3f}")


# ── Main ─────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Analyze perturbation experiment results")
    parser.add_argument("--results", default=str(DEFAULT_RESULTS), help="JSONL results file")
    args = parser.parse_args()

    results_path = Path(args.results)
    if not results_path.exists():
        print(f"Results file not found: {results_path}")
        return

    records = load_results(results_path)
    print(f"Loaded {len(records)} records from {results_path}")

    out_dir = results_path.parent
    print_summary_table(records)
    print("\nGenerating figures...")
    plot_phase_transition(records, out_dir)
    plot_quality_degradation(records, out_dir)
    plot_saturation(records, out_dir)
    print(f"\nAll figures saved to {out_dir}")


if __name__ == "__main__":
    main()
