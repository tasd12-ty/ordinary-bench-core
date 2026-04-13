#!/usr/bin/env python3
"""自适应排序 VLM 评测的 CLI 入口。"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# 确保本地目录优先，然后添加复用的模块
_SELF_DIR = str(Path(__file__).resolve().parent)
_ROOT = Path(__file__).resolve().parent.parent
if _SELF_DIR not in sys.path:
    sys.path.insert(0, _SELF_DIR)
sys.path.insert(1, str(_ROOT / "VLM-test"))
sys.path.insert(2, str(_ROOT / "VLM-test" / "API-test"))

from job_spec import AdaptiveSortJobSpec
from adaptive_engine import run_evaluation


def _create_adapter(provider_spec):
    """创建对应的 provider 适配器。"""
    adapter_name = provider_spec.adapter

    if adapter_name == "mock_oracle":
        return None  # mock 模式在 engine 内部处理，不需要 adapter

    if adapter_name == "anthropic":
        from anthropic_provider import AnthropicAdapter
        return AnthropicAdapter(provider_spec)

    # 复用 VLM-test 的 OpenAI 兼容适配器
    from providers import create_provider_adapter
    return create_provider_adapter(provider_spec)


def main():
    parser = argparse.ArgumentParser(
        description="Adaptive quicksort-based VLM distance ranking evaluation",
    )
    parser.add_argument(
        "--job", required=True,
        help="Path to TOML job configuration file",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-5s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    job = AdaptiveSortJobSpec.from_toml(args.job)
    adapter = _create_adapter(job.provider)

    logging.getLogger(__name__).info(
        "Starting adaptive sort evaluation: model=%s scenes=%s",
        job.provider.model, job.input.scenes_dir,
    )

    run_evaluation(job, adapter)


if __name__ == "__main__":
    main()
