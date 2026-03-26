"""Helpers for legacy CLI wrappers around the unified runner."""

from __future__ import annotations

from pathlib import Path
import sys

API_ROOT = Path(__file__).resolve().parent
VLM_TEST_ROOT = API_ROOT.parent
REPO_ROOT = VLM_TEST_ROOT.parent

sys.path.insert(0, str(API_ROOT))
sys.path.insert(0, str(VLM_TEST_ROOT))

from eval_engine import run_job
from job_spec import JobSpec


def sanitize_run_name(model: str, *, replace_dot: bool = False, suffix: str = "") -> str:
    name = model.replace("/", "--").replace(":", "--")
    if replace_dot:
        name = name.replace(".", "_")
    return name + suffix


def derive_multi_view_root(single_view_root: str) -> str:
    if not single_view_root:
        return single_view_root
    if "single_view" in single_view_root:
        return single_view_root.replace("single_view", "multi_view")
    return single_view_root.rstrip("/") + "_multi_view"


def build_job(job_dict: dict, *, base_dir: str | Path = API_ROOT) -> JobSpec:
    return JobSpec.from_dict(job_dict, base_dir=base_dir)


def run_and_print(job: JobSpec, *, title: str) -> dict:
    _results, summary = run_job(job)
    overall = summary["overall"]
    print(f"\n=== {title} ({summary['n_scenes']} scenes) ===")
    print(f"模型: {job.provider.model}")
    print(
        f"QRR 准确率: {overall['qrr_accuracy']:.2%} "
        f"({overall['qrr_correct']}/{overall['qrr_total']})"
    )
    print(
        f"  disjoint: {overall['qrr_disjoint_accuracy']:.2%} "
        f"({overall['qrr_disjoint_correct']}/{overall['qrr_disjoint_total']})"
    )
    print(
        f"  shared_anchor: {overall['qrr_shared_anchor_accuracy']:.2%} "
        f"({overall['qrr_shared_anchor_correct']}/{overall['qrr_shared_anchor_total']})"
    )
    print(
        f"TRR hour 准确率: {overall['trr_hour_accuracy']:.2%} "
        f"({overall['trr_hour_correct']}/{overall['trr_total']})"
    )
    print(
        f"TRR quadrant 准确率: {overall['trr_quadrant_accuracy']:.2%} "
        f"({overall['trr_quadrant_correct']}/{overall['trr_total']})"
    )
    if overall["fdr_total"] > 0:
        print(
            f"FDR exact 准确率: {overall['fdr_exact_accuracy']:.2%} "
            f"({overall['fdr_exact_correct']}/{overall['fdr_total']})"
        )
        print(f"FDR Kendall τ 均值: {overall['fdr_kendall_mean']:.4f}")
        print(f"FDR pairwise 均值: {overall['fdr_pairwise_mean']:.4f}")
        print(f"FDR top-1 均值: {overall['fdr_top1_mean']:.4f}")
    print(f"缺失: {overall['missing']}")
    if summary.get("n_failed", 0):
        print(f"失败场景: {summary['n_failed']} 个")
    return summary
