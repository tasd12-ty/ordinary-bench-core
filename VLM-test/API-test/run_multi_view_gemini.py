#!/usr/bin/env python3
"""Compatibility wrapper for the unified runner: Gemini multi-view batch."""

from __future__ import annotations

import argparse

from gemini_config import GEMINI_CONFIG
from legacy_runner_common import build_job, run_and_print, sanitize_run_name


QUESTION_TYPES = ["qrr", "trr", "fdr"]


def _multi_view_root(config: dict) -> str:
    return f"{config['oss_base'].rstrip('/')}/data-gen/output/images/multi_view"


def main() -> None:
    parser = argparse.ArgumentParser(description="Gemini 系列模型多视角批量测试（兼容入口）")
    parser.add_argument("--split", default=None, help="只跑指定 split（如 n04）")
    parser.add_argument("--scene", default=None, help="只跑单个场景")
    parser.add_argument("--n-views", type=int, default=4, choices=[1, 2, 3, 4], help="发送的视角数量")
    parser.add_argument("--v2", action="store_true", help="使用 v2 分题型目录格式")
    parser.add_argument("--batch-size", type=int, default=20, help="Batch 大小（v2 模式）")
    parser.add_argument("--types", nargs="+", choices=QUESTION_TYPES, default=None, help="只测指定题型（v2 模式）")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--test-only", action="store_true", help="只跑测试集")
    group.add_argument("--train-only", action="store_true", help="只跑训练集")
    args = parser.parse_args()

    config = GEMINI_CONFIG.copy()
    max_concurrency = min(int(config["max_concurrency"]), 10)

    job = build_job({
        "job_name": "run_multi_view_gemini",
        "provider": {
            "adapter": "gemini_native",
            "model": config["model"],
            "base_url": config["api_url"],
            "options": {
                "access_key": config["access_key"],
                "quota_id": config["quota_id"],
                "user_id": config["user_id"],
                "app": config["app"],
                "temperature": config["temperature"],
                "max_output_tokens": config["max_output_tokens"],
                "include_thoughts": config["include_thoughts"],
                "thinking_budget": config["thinking_budget"],
                "max_retries": config["max_retries"],
                "retry_base_delay": config["retry_base_delay"],
                "timeout": config["timeout"],
                "max_concurrency": max_concurrency,
            },
        },
        "input": {
            "questions_dir": config["questions_dir"],
            "question_layout": "v2" if args.v2 else "v1",
            "question_grouping": "by_type" if args.v2 else "mixed",
            "question_types": args.types or QUESTION_TYPES,
            "batch_size": args.batch_size,
        },
        "images": {
            "mode": "multi_view",
            "multi_view_root": _multi_view_root(config),
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
            "save_prompt": True,
        },
        "output": {
            "results_dir": config["results_dir"],
            "run_name": sanitize_run_name(config["model"], replace_dot=True, suffix="_multi_view"),
        },
    })
    mode_label = "v2（分题型）" if args.v2 else "v1（混合）"
    run_and_print(job, title=f"Gemini 多视角测试 {mode_label} ({args.n_views} 视角)")


if __name__ == "__main__":
    main()
