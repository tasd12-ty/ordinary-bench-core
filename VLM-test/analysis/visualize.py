"""
Visualization tools for scene belief reconstruction.

Generates GT vs reconstructed configuration plots (Figure 2 in paper).
"""

import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def plot_configuration_comparison(
    gt_positions: Dict[str, np.ndarray],
    recon_positions: Dict[str, np.ndarray],
    scene_id: str = "",
    metrics: Optional[dict] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (12, 5),
) -> Optional[object]:
    """Plot GT vs reconstructed configuration side by side.

    Args:
        gt_positions: ground truth {obj_id: [x, y]}
        recon_positions: reconstructed {obj_id: [x, y]}
        scene_id: scene identifier for title
        metrics: optional metrics dict (CSR, NRMS, etc.)
        save_path: optional file path to save figure
        figsize: figure size

    Returns:
        matplotlib Figure object, or None if matplotlib unavailable
    """
    if not HAS_MPL:
        print("matplotlib not installed, skipping visualization")
        return None

    # Procrustes-align recon to GT for visual comparison
    from reconstruct.utils import procrustes_align

    obj_ids = sorted(set(gt_positions.keys()) & set(recon_positions.keys()))
    if len(obj_ids) < 3:
        return None

    gt_mat = np.array([gt_positions[oid][:2] for oid in obj_ids])
    recon_mat = np.array([recon_positions[oid][:2] for oid in obj_ids])

    recon_aligned, rms = procrustes_align(recon_mat, gt_mat)

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    colors = plt.cm.Set2(np.linspace(0, 1, len(obj_ids)))

    # Panel 1: Ground Truth
    ax = axes[0]
    for i, oid in enumerate(obj_ids):
        ax.scatter(*gt_mat[i], c=[colors[i]], s=120, zorder=5, edgecolors='black')
        ax.annotate(oid, gt_mat[i], fontsize=8, ha='center', va='bottom',
                    xytext=(0, 8), textcoords='offset points')
    ax.set_title("Ground Truth", fontsize=12, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # Panel 2: Reconstructed (aligned)
    ax = axes[1]
    for i, oid in enumerate(obj_ids):
        ax.scatter(*recon_aligned[i], c=[colors[i]], s=120, zorder=5,
                   edgecolors='black', marker='D')
        ax.annotate(oid, recon_aligned[i], fontsize=8, ha='center', va='bottom',
                    xytext=(0, 8), textcoords='offset points')
    ax.set_title("Reconstructed (Procrustes-aligned)", fontsize=12, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # Panel 3: Overlay with displacement arrows
    ax = axes[2]
    for i, oid in enumerate(obj_ids):
        ax.scatter(*gt_mat[i], c=[colors[i]], s=100, zorder=5,
                   edgecolors='black', label=f'{oid} (GT)' if i == 0 else None)
        ax.scatter(*recon_aligned[i], c=[colors[i]], s=100, zorder=5,
                   edgecolors='black', marker='D',
                   label=f'{oid} (Recon)' if i == 0 else None)
        # Displacement arrow
        ax.annotate('', xy=recon_aligned[i], xytext=gt_mat[i],
                    arrowprops=dict(arrowstyle='->', color=colors[i],
                                   lw=1.5, alpha=0.7))
        ax.annotate(oid, gt_mat[i], fontsize=7, ha='center', va='bottom',
                    xytext=(0, 8), textcoords='offset points')
    ax.set_title("Overlay (arrows = distortion)", fontsize=12, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # Add legend for overlay
    circle = mpatches.Patch(facecolor='gray', edgecolor='black', label='GT')
    diamond = mpatches.Patch(facecolor='gray', edgecolor='black', label='Recon')
    ax.legend(handles=[circle, diamond], loc='upper right', fontsize=8)

    # Add metrics text
    title = f"Scene: {scene_id}" if scene_id else "Scene Reconstruction"
    if metrics:
        metric_str = "  |  ".join([
            f"CSR_QRR={metrics.get('csr_qrr', 'N/A'):.3f}"
            if isinstance(metrics.get('csr_qrr'), (int, float)) else "",
            f"CSR_TRR={metrics.get('csr_trr', 'N/A'):.3f}"
            if isinstance(metrics.get('csr_trr'), (int, float)) else "",
            f"NRMS={metrics.get('nrms', 'N/A'):.4f}"
            if isinstance(metrics.get('nrms'), (int, float)) else "",
            f"Kendall_\u03C4={metrics.get('kendall_tau', 'N/A'):.3f}"
            if isinstance(metrics.get('kendall_tau'), (int, float)) else "",
            f"K={metrics.get('K_geom', 'N/A')}",
        ])
        title += f"\n{metric_str}"

    fig.suptitle(title, fontsize=11, y=1.02)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    return fig


def plot_accuracy_vs_complexity(
    per_scene_data: List[dict],
    model_name: str = "",
    save_path: Optional[str] = None,
) -> Optional[object]:
    """Plot accuracy vs number of objects (Figure 1 in paper).

    Args:
        per_scene_data: from aggregate.per_scene_accuracy()
        model_name: model name for title
        save_path: optional save path
    """
    if not HAS_MPL:
        return None

    # Group by n_objects
    by_n = {}
    for d in per_scene_data:
        n = d["n_objects"]
        by_n.setdefault(n, []).append(d)

    ns = sorted(by_n.keys())
    qrr_means = [np.mean([d["qrr_accuracy"] for d in by_n[n]]) for n in ns]
    trr_h_means = [np.mean([d["trr_hour_accuracy"] for d in by_n[n]]) for n in ns]
    trr_q_means = [np.mean([d["trr_quadrant_accuracy"] for d in by_n[n]]) for n in ns]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ns, qrr_means, 'o-', label='QRR Accuracy', linewidth=2, markersize=8)
    ax.plot(ns, trr_h_means, 's-', label='TRR Hour Accuracy', linewidth=2, markersize=8)
    ax.plot(ns, trr_q_means, '^-', label='TRR Quadrant Accuracy', linewidth=2, markersize=8)
    ax.axhline(y=1/3, color='gray', linestyle='--', alpha=0.5, label='Random (QRR)')
    ax.axhline(y=1/12, color='lightgray', linestyle='--', alpha=0.5, label='Random (TRR Hour)')

    ax.set_xlabel("Number of Objects", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title(f"Accuracy vs Scene Complexity{' - ' + model_name if model_name else ''}",
                 fontsize=13)
    ax.legend(fontsize=10)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    return fig


def plot_reconstruction_quality_distribution(
    recon_results: List[dict],
    model_name: str = "",
    save_path: Optional[str] = None,
) -> Optional[object]:
    """Plot distribution of reconstruction quality metrics (Figure 3 in paper).

    Args:
        recon_results: list of reconstruction result dicts
        model_name: model name for title
        save_path: optional save path
    """
    if not HAS_MPL:
        return None

    csr_qrr = [r["metrics"]["csr_qrr"] for r in recon_results]
    csr_trr = [r["metrics"]["csr_trr"] for r in recon_results]
    nrms = [r["metrics"]["nrms"] for r in recon_results
            if r["metrics"].get("nrms") is not None]
    ktau = [r["metrics"]["kendall_tau"] for r in recon_results
            if r["metrics"].get("kendall_tau") is not None]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # CSR QRR
    axes[0, 0].hist(csr_qrr, bins=20, edgecolor='black', alpha=0.7, color='steelblue')
    axes[0, 0].set_xlabel("CSR (QRR)")
    axes[0, 0].set_title(f"CSR QRR (mean={np.mean(csr_qrr):.3f})")
    axes[0, 0].axvline(0.95, color='red', linestyle='--', label='Threshold')
    axes[0, 0].legend()

    # CSR TRR
    axes[0, 1].hist(csr_trr, bins=20, edgecolor='black', alpha=0.7, color='coral')
    axes[0, 1].set_xlabel("CSR (TRR)")
    axes[0, 1].set_title(f"CSR TRR (mean={np.mean(csr_trr):.3f})")
    axes[0, 1].axvline(0.95, color='red', linestyle='--', label='Threshold')
    axes[0, 1].legend()

    # NRMS
    if nrms:
        axes[1, 0].hist(nrms, bins=20, edgecolor='black', alpha=0.7, color='seagreen')
        axes[1, 0].set_xlabel("NRMS")
        axes[1, 0].set_title(f"NRMS (mean={np.mean(nrms):.4f})")

    # Kendall tau
    if ktau:
        axes[1, 1].hist(ktau, bins=20, edgecolor='black', alpha=0.7, color='orchid')
        axes[1, 1].set_xlabel("Kendall \u03C4")
        axes[1, 1].set_title(f"Kendall \u03C4 (mean={np.mean(ktau):.3f})")

    fig.suptitle(
        f"Reconstruction Quality Distribution{' - ' + model_name if model_name else ''}",
        fontsize=13
    )
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    return fig


def plot_three_condition_comparison(
    gt_positions: Dict[str, np.ndarray],
    recon_a: Dict[str, np.ndarray],
    recon_b: Optional[Dict[str, np.ndarray]],
    recon_c: Optional[Dict[str, np.ndarray]],
    scene_id: str = "",
    save_path: Optional[str] = None,
) -> Optional[object]:
    """Plot reconstruction under three conditions: correct/wrong/no image (Figure 5).

    Args:
        gt_positions: ground truth positions
        recon_a: reconstruction from correct image
        recon_b: reconstruction from wrong image (or None)
        recon_c: reconstruction from no image (or None)
        scene_id: scene identifier
        save_path: optional save path
    """
    if not HAS_MPL:
        return None

    from reconstruct.utils import procrustes_align

    obj_ids = sorted(gt_positions.keys())
    gt_mat = np.array([gt_positions[oid][:2] for oid in obj_ids])

    n_panels = 1 + (recon_b is not None) + (recon_c is not None) + 1  # +1 for GT
    fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 4))
    if n_panels == 1:
        axes = [axes]

    colors = plt.cm.Set2(np.linspace(0, 1, len(obj_ids)))

    def plot_panel(ax, positions, title, marker='o'):
        if positions is None:
            ax.text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=14,
                    transform=ax.transAxes)
            ax.set_title(title)
            return

        common = sorted(set(obj_ids) & set(positions.keys()))
        if len(common) < 3:
            ax.text(0.5, 0.5, 'Too few objects', ha='center', va='center',
                    fontsize=10, transform=ax.transAxes)
            ax.set_title(title)
            return

        pos_mat = np.array([positions[oid][:2] for oid in common])
        gt_sub = np.array([gt_positions[oid][:2] for oid in common])
        aligned, rms = procrustes_align(pos_mat, gt_sub)

        for i, oid in enumerate(common):
            idx = obj_ids.index(oid)
            ax.scatter(*aligned[i], c=[colors[idx]], s=100, zorder=5,
                       edgecolors='black', marker=marker)
            ax.annotate(oid, aligned[i], fontsize=7, ha='center', va='bottom',
                        xytext=(0, 6), textcoords='offset points')

        ax.set_title(f"{title}\nRMS={rms:.4f}", fontsize=10)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

    panel_idx = 0

    # GT
    plot_panel(axes[panel_idx], gt_positions, "GT")
    panel_idx += 1

    # Condition A
    plot_panel(axes[panel_idx], recon_a, "A: Correct Image", marker='D')
    panel_idx += 1

    # Condition B
    if recon_b is not None:
        plot_panel(axes[panel_idx], recon_b, "B: Wrong Image", marker='s')
        panel_idx += 1

    # Condition C
    if recon_c is not None:
        plot_panel(axes[panel_idx], recon_c, "C: No Image", marker='^')

    fig.suptitle(f"Three-Condition Comparison: {scene_id}", fontsize=12)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    return fig
