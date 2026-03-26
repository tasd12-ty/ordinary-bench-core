#!/usr/bin/env python3
"""Unified evaluation entrypoint."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

API_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(API_ROOT))
sys.path.insert(0, str(API_ROOT.parent))

from eval_engine import run_job
from job_spec import JobSpec


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified VLM evaluation runner")
    parser.add_argument("--job", required=True, help="Path to a TOML job spec")
    args = parser.parse_args()

    job = JobSpec.from_toml(args.job)
    _results, summary = run_job(job)

    overall = summary["overall"]
    print(f"\n=== Job {job.job_name or job.run_name} ({summary['n_scenes']} scenes) ===")
    print(f"Model: {job.provider.model}")
    print(f"Adapter: {job.provider.adapter}")
    print(f"QRR accuracy: {overall['qrr_accuracy']:.2%} ({overall['qrr_correct']}/{overall['qrr_total']})")
    print(
        f"TRR hour accuracy: {overall['trr_hour_accuracy']:.2%} "
        f"({overall['trr_hour_correct']}/{overall['trr_total']})"
    )
    if overall["fdr_total"] > 0:
        print(
            f"FDR exact accuracy: {overall['fdr_exact_accuracy']:.2%} "
            f"({overall['fdr_exact_correct']}/{overall['fdr_total']})"
        )
        print(f"FDR Kendall tau mean: {overall['fdr_kendall_mean']:.4f}")
    print(f"Missing: {overall['missing']}")
    if summary.get("n_failed", 0):
        print(f"Failed scenes: {summary['n_failed']}")


if __name__ == "__main__":
    main()
