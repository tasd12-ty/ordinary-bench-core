"""
VLM 测试配置 — 多端点预置，注释切换。

切换方式：取消注释你要用的那一行 ACTIVE_ENDPOINT，注释掉其他的即可。
也可通过环境变量 ALI_VLM_ENDPOINT 覆盖（值为 dashscope / openrouter / ...）。

支持的模型（示例）：
    DashScope:   qwen3.5-plus-2026-02-15, qwen3.5-397b-a17b, kimi-k2.5
    OpenRouter:  google/gemini-2.0-flash-001, anthropic/claude-3.5-sonnet, ...
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# 自动加载同目录下的 .env 文件（不覆盖已有环境变量）
load_dotenv(Path(__file__).resolve().parent / ".env")

# ╔══════════════════════════════════════════════════════════════════╗
# ║  预置端点 — 取消注释你要用的那一行，注释掉其他的                    ║
# ╚══════════════════════════════════════════════════════════════════╝

ACTIVE_ENDPOINT = "dashscope"           # ← 阿里 DashScope（默认，OpenAI 兼容）
# ACTIVE_ENDPOINT = "openrouter"        # ← OpenRouter（OpenAI 兼容）
# ACTIVE_ENDPOINT = "alibaba_internal"  # ← 阿里内部 API（需配合 gemini_config + gemini_vlm_client 使用）

# 也可通过环境变量覆盖
ACTIVE_ENDPOINT = os.environ.get("ALI_VLM_ENDPOINT", ACTIVE_ENDPOINT)

# ── 端点配置表 ──────────────────────────────────────────────────────

ENDPOINTS = {
    "dashscope": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key": os.environ.get("DASHSCOPE_API_KEY", ""),
        "default_model": "qwen3.5-plus-2026-02-15",
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": os.environ.get("OPENROUTER_API_KEY", ""),
        "default_model": "google/gemini-2.0-flash-001",
    },
    "alibaba_internal": {
        "base_url": "https://llm-chat-api.alibaba-inc.com/v1/api/chat",
        "api_key": "",
        "default_model": "gemini-3-pro-preview",
        # 阿里内部 API 专用认证字段
        "access_key": os.environ.get("ALI_INTERNAL_ACCESS_KEY", ""),
        "quota_id": os.environ.get("ALI_INTERNAL_QUOTA_ID", ""),
        "user_id": os.environ.get("ALI_INTERNAL_USER_ID", ""),
        "app": "llm_application",
        "tag": "image_gen_pipeline",
    },
}

# ── 自动组装配置 ────────────────────────────────────────────────────

_endpoint = ENDPOINTS[ACTIVE_ENDPOINT]

ALI_CONFIG = {
    # API 连接（由 ACTIVE_ENDPOINT 决定）
    "base_url": os.environ.get("ALI_VLM_BASE_URL", _endpoint["base_url"]),
    "api_key": os.environ.get("ALI_VLM_API_KEY", _endpoint["api_key"]),
    "model": os.environ.get("ALI_VLM_MODEL", _endpoint["default_model"]),

    # 并发与重试
    "max_concurrency": int(os.environ.get("ALI_VLM_CONCURRENCY", "5")),
    "timeout": int(os.environ.get("ALI_VLM_TIMEOUT", "300")),
    "max_retries": int(os.environ.get("ALI_VLM_MAX_RETRIES", "5")),
    "retry_base_delay": float(os.environ.get("ALI_VLM_RETRY_DELAY", "3.0")),

    # 生成参数
    "temperature": float(os.environ.get("ALI_VLM_TEMPERATURE", "0.0")),
    "max_tokens": int(os.environ.get("ALI_VLM_MAX_TOKENS", "65536")),

    # 路径（相对于 API-test/ 目录）
    "questions_dir": "../output/questions",
    "images_dir": "../../data-gen/output/images/single_view",
    "results_dir": "../output/results",
}
