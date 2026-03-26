#!/usr/bin/env python3
"""Compatibility wrapper for the unified runner: three-condition experiments."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from config import CONFIG
from legacy_runner_common import API_ROOT, build_job, run_and_print, sanitize_run_name


def main() -> None:
    parser = argparse.ArgumentParser(description="Three-condition experiment")
    parser.add_argument("--condition", "-c", required=True, choices=["A", "B", "C"], help="Experiment condition")
    parser.add_argument("--questions-dir", "-q", default="output/questions")
    parser.add_argument("--images-dir", "-i", default="../data-gen/output/images/single_view")
    parser.add_argument("--output-dir", "-o", default=None)
    parser.add_argument("--split", "-s", default=None)
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--wrong-image-seed", type=int, default=42)
    args = parser.parse_args()

    config = CONFIG.copy()
    if not config["api_key"]:
        print("请设置环境变量 VLM_API_KEY", file=sys.stderr)
        raise SystemExit(1)

    condition = args.condition.upper()
    if args.output_dir:
        output_dir = Path(args.output_dir)
        results_dir = str(output_dir.parent)
        run_name = output_dir.name
    else:
        results_dir = config["results_dir"]
        run_name = sanitize_run_name(config["model"], suffix=f"_condition_{condition}")

    image_mode = {
        "A": "single",
        "B": "wrong_single",
        "C": "none",
    }[condition]

    job = build_job({
        "job_name": f"condition_{condition}",
        "provider": {
            "adapter": "openai_chat",
            "model": config["model"],
            "base_url": config["base_url"],
            "api_key": config["api_key"],
            "options": {
                "provider": config.get("provider", ""),
                "temperature": config["temperature"],
                "max_tokens": config["max_tokens"],
                "max_retries": config["max_retries"],
                "retry_base_delay": config["retry_base_delay"],
                "timeout": config["timeout"],
                "max_concurrency": config["max_concurrency"],
            },
        },
        "input": {
            "questions_dir": args.questions_dir,
            "question_layout": "v1",
            "question_grouping": "mixed",
            "question_types": ["qrr", "trr", "fdr"],
            "batch_size": 20,
        },
        "images": {
            "mode": image_mode,
            "single_view_root": args.images_dir,
            "wrong_image_seed": args.wrong_image_seed,
        },
        "selection": {
            "scene": "",
            "split": args.split or "",
            "max_scenes": args.max_scenes,
        },
        "prompt": {
            "react_max_rounds": 2,
            "missing_threshold": 0.2,
            "react_chunk_size": 50,
            "save_prompt": False,
        },
        "output": {
            "results_dir": results_dir,
            "run_name": run_name,
        },
    }, base_dir=API_ROOT)
    run_and_print(job, title=f"Condition {condition} complete")


if __name__ == "__main__":
    main()
