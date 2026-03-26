#!/usr/bin/env python3
"""Compatibility wrapper for the unified runner: multi-view batch."""

from __future__ import annotations

import argparse
import sys

from config import CONFIG
from legacy_runner_common import build_job, run_and_print, sanitize_run_name


def main() -> None:
    parser = argparse.ArgumentParser(description="多视角 Batch 提问模式")
    parser.add_argument("--split", default=None, help="只跑指定 split（如 n04）")
    parser.add_argument("--scene", default=None, help="只跑单个场景（如 n04_000000）")
    parser.add_argument("--n-views", type=int, default=4, choices=[1, 2, 3, 4], help="发送的视角数量")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--test-only", action="store_true", help="只跑测试集场景")
    group.add_argument("--train-only", action="store_true", help="只跑训练集场景")
    args = parser.parse_args()

    config = CONFIG.copy()
    if not config["api_key"]:
        print("请设置环境变量 VLM_API_KEY", file=sys.stderr)
        raise SystemExit(1)

    job = build_job({
        "job_name": "run_multi_view",
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
            "questions_dir": config["questions_dir"],
            "question_layout": "v1",
            "question_grouping": "mixed",
            "question_types": ["qrr", "trr", "fdr"],
            "batch_size": 20,
        },
        "images": {
            "mode": "multi_view",
            "multi_view_root": config["multi_view_images_dir"],
            "n_views": args.n_views,
        },
        "selection": {
            "scene": args.scene or "",
            "split": args.split or "",
            "test_only": args.test_only,
            "train_only": args.train_only,
        },
        "prompt": {
            "react_max_rounds": 2,
            "missing_threshold": 0.2,
            "react_chunk_size": 50,
            "save_prompt": False,
        },
        "output": {
            "results_dir": config["results_dir"],
            "run_name": sanitize_run_name(config["model"], suffix="_multi_view"),
        },
    })
    run_and_print(job, title=f"Multi-view 模式结果 ({args.n_views} 视角)")


if __name__ == "__main__":
    main()
