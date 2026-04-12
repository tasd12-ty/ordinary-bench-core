#!/usr/bin/env python3
"""VRF 评测入口 — 读取 TOML job 配置并执行。

用法:
    python run_eval.py --job jobs/smoke.toml
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from vrf_job_spec import JobSpec
from eval_engine import run_job

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


def main():
    parser = argparse.ArgumentParser(description="Run VRF evaluation")
    parser.add_argument("--job", "-j", required=True, help="Path to TOML job file")
    args = parser.parse_args()

    job = JobSpec.from_toml(args.job)
    results, summary = run_job(job)

    overall = summary.get("overall", {})
    print(f"\n{'='*50}")
    print(f"VRF Evaluation Complete")
    print(f"{'='*50}")
    print(f"  Scenes:     {summary.get('n_scenes', 0)}")
    print(f"  Accuracy:   {overall.get('vrf_accuracy', 0):.1%}")
    print(f"  TRUE acc:   {overall.get('vrf_true_accuracy', 0):.1%}")
    print(f"  FALSE acc:  {overall.get('vrf_false_accuracy', 0):.1%}")
    print(f"  Missing:    {overall.get('missing', 0)}")

    if summary.get("n_failed"):
        print(f"  FAILED:     {summary['n_failed']} scenes")

    print(f"\nResults: {Path(job.results_dir) / job.run_name}")


if __name__ == "__main__":
    main()
