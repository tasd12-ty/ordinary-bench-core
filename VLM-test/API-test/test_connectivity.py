#!/usr/bin/env python3
"""
API 连通性测试脚本。

对指定的多个模型发送简单的纯文本请求（无图片），
验证 API 密钥、网络连通性和模型可用性。

用法：
    python test_connectivity.py
"""

import json
import os
import sys
import time
from pathlib import Path


def load_dotenv(env_path: str) -> None:
    """简易 .env 加载器，不依赖第三方库。"""
    path = Path(env_path)
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if not os.environ.get(key):
            os.environ[key] = value


# 加载当前目录下的 .env
load_dotenv(str(Path(__file__).resolve().parent / ".env"))

# ── 测试模型配置 ──────────────────────────────────────────────
# GPT / Claude 走阿里内部 API (llm-chat-api)，使用 OpenAI 兼容 prompt 格式
# Gemini 走阿里内部 API，使用 Gemini 原生 prompt 格式（parts）
# Qwen 走 DashScope (OpenAI 兼容)

MODELS_TO_TEST = [
    {
        "name": "gpt-5.4-0305-global",
        "adapter": "ali_internal",
        "model": "gpt-5.4-0305-global",
        "prompt_format": "openai",
        "api_url": "https://llm-chat-api.alibaba-inc.com/v1/api/chat",
        "access_key": os.environ.get("ALI_INTERNAL_ACCESS_KEY", ""),
        "quota_id": os.environ.get("ALI_INTERNAL_QUOTA_ID", ""),
        "user_id": os.environ.get("ALI_INTERNAL_USER_ID", ""),
    },
    {
        "name": "gemini-3.1-pro-preview",
        "adapter": "ali_internal",
        "model": "gemini-3.1-pro-preview",
        "prompt_format": "gemini",
        "api_url": "https://llm-chat-api.alibaba-inc.com/v1/api/chat",
        "access_key": os.environ.get("GEMINI_VLM_ACCESS_KEY", ""),
        "quota_id": os.environ.get("GEMINI_VLM_QUOTA_ID", ""),
        "user_id": os.environ.get("GEMINI_VLM_USER_ID", ""),
    },
    {
        "name": "qwen3.5-plus",
        "adapter": "dashscope",
        "model": "qwen3.5-plus-2026-02-15",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key": os.environ.get("DASHSCOPE_API_KEY", ""),
    },
    {
        "name": "claude-opus-4-6",
        "adapter": "ali_internal",
        "model": "claude-opus-4-6",
        "prompt_format": "openai",
        "api_url": "https://llm-chat-api.alibaba-inc.com/v1/api/chat",
        "access_key": os.environ.get("ALI_INTERNAL_ACCESS_KEY", ""),
        "quota_id": os.environ.get("ALI_INTERNAL_QUOTA_ID", ""),
        "user_id": os.environ.get("ALI_INTERNAL_USER_ID", ""),
    },
]

TEST_PROMPT = "请用一句话回答：1+1等于几？"


def test_dashscope(config: dict) -> tuple:
    """测试 DashScope（阿里云百炼）OpenAI 兼容 API 的连通性。"""
    from openai import OpenAI

    client = OpenAI(base_url=config["base_url"], api_key=config["api_key"])
    messages = [{"role": "user", "content": TEST_PROMPT}]

    start = time.time()
    resp = client.chat.completions.create(
        model=config["model"],
        messages=messages,
        temperature=0.0,
        max_tokens=256,
        timeout=60,
    )
    elapsed = time.time() - start

    content = resp.choices[0].message.content
    if not content:
        reasoning = getattr(resp.choices[0].message, "reasoning_content", None)
        if reasoning:
            return True, f"(reasoning) {reasoning[:120]}", elapsed
        return False, "返回内容为空", elapsed

    return True, content.strip()[:120], elapsed


def test_ali_internal(config: dict) -> tuple:
    """
    测试阿里内部 API（llm-chat-api）的连通性。

    GPT/Claude 使用 OpenAI 兼容 prompt 格式（role + content）。
    Gemini 使用原生 prompt 格式（role + parts）。
    """
    import requests

    prompt_format = config.get("prompt_format", "openai")

    if prompt_format == "gemini":
        prompt_value = [
            {"role": "user", "parts": [{"text": TEST_PROMPT}]}
        ]
    else:
        # OpenAI 兼容格式（GPT / Claude 等）
        prompt_value = [
            {"role": "user", "content": TEST_PROMPT}
        ]

    payload = {
        "model": config["model"],
        "prompt": prompt_value,
        "params": {
            "max_tokens": 256,
        },
        "passparams": {},
        "app": "llm_application",
        "quota_id": config["quota_id"],
        "access_key": config["access_key"],
        "user_id": config["user_id"],
    }

    headers = {"Content-Type": "application/json"}

    start = time.time()
    response = requests.post(
        config["api_url"],
        json=payload,
        headers=headers,
        timeout=120,
    )
    elapsed = time.time() - start

    if response.status_code != 200:
        return False, f"HTTP {response.status_code}: {response.text[:300]}", elapsed

    data = response.json()

    # 检查顶层 code
    if data.get("code") != 0:
        return False, f"code={data.get('code')}, message={data.get('message', '')}", elapsed

    inner = data.get("data")
    if not isinstance(inner, dict):
        return False, f"响应格式异常: {json.dumps(data, ensure_ascii=False)[:200]}", elapsed

    # 方式 1：data.message（平台提取的答案文本）
    message = inner.get("message")
    if message and isinstance(message, str):
        return True, message.strip()[:120], elapsed

    # 方式 2：completion.content
    completion = inner.get("completion")
    if isinstance(completion, dict):
        content = completion.get("content")
        if content and isinstance(content, str):
            return True, content.strip()[:120], elapsed

        # 方式 3：completion.choices（OpenAI 格式）
        choices = completion.get("choices")
        if choices and isinstance(choices, list) and len(choices) > 0:
            choice = choices[0]
            # chat completion 格式
            msg = choice.get("message", {})
            if msg.get("content"):
                return True, msg["content"].strip()[:120], elapsed
            # text completion 格式
            text = choice.get("text")
            if text:
                return True, text.strip()[:120], elapsed

        # 方式 4：candidates（Gemini 格式）
        candidates = completion.get("candidates")
        if candidates and isinstance(candidates, list) and len(candidates) > 0:
            parts = candidates[0].get("content", {}).get("parts", [])
            text_parts = [
                part["text"]
                for part in parts
                if isinstance(part, dict) and "text" in part
                and not part.get("thought", False)
            ]
            if text_parts:
                return True, "\n".join(text_parts).strip()[:120], elapsed

    return False, f"无法提取内容: {json.dumps(data, ensure_ascii=False)[:300]}", elapsed


def run_tests() -> None:
    """逐个测试所有模型的连通性。"""
    print("=" * 70)
    print("  API 连通性测试")
    print(f"  测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  测试 Prompt: {TEST_PROMPT}")
    print("=" * 70)
    print()

    results = []

    for config in MODELS_TO_TEST:
        model_name = config["name"]
        adapter = config["adapter"]
        print(f"🔄 测试 [{model_name}] (adapter: {adapter}) ...")

        # 检查密钥是否配置
        if adapter == "dashscope" and not config.get("api_key"):
            print("   ⚠️  跳过: API Key 未配置")
            results.append((model_name, "SKIP", "API Key 未配置", 0))
            print()
            continue
        if adapter == "ali_internal" and not config.get("access_key"):
            print("   ⚠️  跳过: Access Key 未配置")
            results.append((model_name, "SKIP", "Access Key 未配置", 0))
            print()
            continue

        try:
            if adapter == "dashscope":
                success, response_text, elapsed = test_dashscope(config)
            elif adapter == "ali_internal":
                success, response_text, elapsed = test_ali_internal(config)
            else:
                results.append((model_name, "FAIL", f"未知 adapter: {adapter}", 0))
                continue

            if success:
                print(f"   ✅ 成功 ({elapsed:.2f}s)")
                print(f"   📝 回复: {response_text}")
                results.append((model_name, "PASS", response_text, elapsed))
            else:
                print(f"   ❌ 失败 ({elapsed:.2f}s)")
                print(f"   📝 原因: {response_text}")
                results.append((model_name, "FAIL", response_text, elapsed))

        except Exception as exc:
            print(f"   ❌ 异常: {type(exc).__name__}: {str(exc)[:200]}")
            results.append((model_name, "ERROR", f"{type(exc).__name__}: {str(exc)[:200]}", 0))

        print()

    # ── 汇总 ──
    print("=" * 70)
    print("  测试结果汇总")
    print("=" * 70)
    print(f"  {'模型':<30} {'状态':<8} {'耗时':>8}")
    print("-" * 70)
    for model_name, status, detail, elapsed in results:
        status_icon = {
            "PASS": "✅", "FAIL": "❌", "ERROR": "💥", "SKIP": "⚠️"
        }.get(status, "?")
        elapsed_str = f"{elapsed:.2f}s" if elapsed > 0 else "-"
        print(f"  {status_icon} {model_name:<28} {status:<8} {elapsed_str:>8}")
    print("-" * 70)

    passed = sum(1 for _, s, _, _ in results if s == "PASS")
    total = len(results)
    print(f"  通过: {passed}/{total}")
    print()

    if passed < total:
        sys.exit(1)


if __name__ == "__main__":
    run_tests()
