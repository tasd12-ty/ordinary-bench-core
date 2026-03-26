"""
Gemini 系列模型测试配置。

通过阿里内部平台代理调用 Gemini API，使用相同的 API 端点和认证信息，
但 prompt 协议遵循 Gemini 原生格式（parts 而非 content）。

所有设置通过环境变量覆盖，切换模型只需修改 GEMINI_VLM_MODEL。
"""

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    def load_dotenv(*_args, **_kwargs):
        return False

# 自动加载同目录下的 .env 文件（不覆盖已有环境变量）
load_dotenv(Path(__file__).resolve().parent / ".env")

GEMINI_CONFIG = {
    # API 连接（与阿里内部 API 共用同一端点）
    "api_url": os.environ.get(
        "GEMINI_VLM_API_URL",
        "https://llm-chat-api.alibaba-inc.com/v1/api/chat",
    ),
    "model": os.environ.get("GEMINI_VLM_MODEL", "gemini-3.1-pro-preview"),

    # 认证（与阿里内部 API 共用，从环境变量或 .env 文件读取）
    "access_key": os.environ.get("GEMINI_VLM_ACCESS_KEY", ""),
    "quota_id": os.environ.get("GEMINI_VLM_QUOTA_ID", ""),
    "user_id": os.environ.get("GEMINI_VLM_USER_ID", ""),
    "app": os.environ.get("GEMINI_VLM_APP", "llm_application"),

    # OSS 图片基础路径（与阿里内部 API 共用）
    "oss_base": os.environ.get(
        "GEMINI_VLM_OSS_BASE",
        "oss://quark-llm/datasets/tuyongsiqi.tysq/ordinary-bench/ordinary-bench-core-main",
    ),

    # 并发与重试
    "max_concurrency": int(os.environ.get("GEMINI_VLM_CONCURRENCY", "8")),
    "timeout": int(os.environ.get("GEMINI_VLM_TIMEOUT", "6000")),
    "max_retries": int(os.environ.get("GEMINI_VLM_MAX_RETRIES", "5")),
    "retry_base_delay": float(os.environ.get("GEMINI_VLM_RETRY_DELAY", "2.0")),

    # Gemini 生成参数
    "temperature": float(os.environ.get("GEMINI_VLM_TEMPERATURE", "0.0")),
    "max_output_tokens": int(os.environ.get("GEMINI_VLM_MAX_TOKENS", "65536")),
    "include_thoughts": os.environ.get("GEMINI_VLM_INCLUDE_THOUGHTS", "true"),
    "thinking_budget": int(os.environ.get("GEMINI_VLM_THINKING_BUDGET", "2000")),

    # 路径（相对于 API-test/ 目录）
    "questions_dir": "../output/questions",
    "images_dir": "../../data-gen/output/images/single_view",
    "results_dir": "../output/results",
}
