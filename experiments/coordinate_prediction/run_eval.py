#!/usr/bin/env python3
"""Coordinate prediction evaluation entry point.

Usage:
    python run_eval.py --job jobs/smoke.toml
    python run_eval.py --job jobs/single_view.toml
"""

import argparse
import logging
from pathlib import Path

from job_spec import JobSpec
from eval_engine import run_job

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


def main():
    parser = argparse.ArgumentParser(description="Run coordinate prediction evaluation")
    parser.add_argument("--job", "-j", required=True, help="Path to TOML job file")
    args = parser.parse_args()

    job = JobSpec.from_toml(args.job)
    results, summary = run_job(job)

    overall = summary.get("overall", {})
    print(f"\n{'='*55}")
    print(f"Coordinate Prediction — {summary.get('image_mode', '?')} mode")
    print(f"{'='*55}")
    print(f"  Model:       {summary.get('model', '?')}")
    print(f"  Scenes:      {summary.get('n_scenes', 0)}")

    for key in ("kendall_tau", "nrms", "csr_qrr", "csr_trr"):
        stats = overall.get(key, {})
        mean = stats.get("mean")
        if mean is not None:
            label = key.replace("_", " ").upper()
            print(f"  {label:12s}  mean={mean:.4f}  median={stats.get('median', 0):.4f}")

    missing = overall.get("n_missing_total", 0)
    if missing:
        print(f"  Missing:     {missing} objects total")

    if summary.get("n_failed"):
        print(f"  FAILED:      {summary['n_failed']} scenes")

    print(f"\nResults: {Path(job.results_dir) / job.run_name}")


if __name__ == "__main__":
    main()
