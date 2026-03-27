"""Ali-internal LLM platform adapter for GPT / Claude models.

Calls https://llm-chat-api.alibaba-inc.com/v1/api/chat with OpenAI-compatible
prompt format (role/content messages).  Authentication uses access_key, quota_id
and user_id placed in the request body.

For Gemini models use the existing ``gemini_native`` adapter instead, which
speaks the Gemini-native parts/inlineData format.
"""

from __future__ import annotations

import json
import logging
import time
from copy import deepcopy
from typing import Any, List

import requests

from .base import ProviderAdapter, ProviderRequest
from vlm_client import load_image_base64

logger = logging.getLogger(__name__)

_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


def _image_content_items(image_inputs: List[dict[str, str]]) -> list[dict[str, Any]]:
    """Convert image inputs to OpenAI vision content items."""
    content: list[dict[str, Any]] = []
    for image in image_inputs:
        if image["kind"] == "file":
            image_b64 = load_image_base64(image["value"])
            url = f"data:image/png;base64,{image_b64}"
        elif image["kind"] == "url":
            url = image["value"]
        else:
            continue
        content.append({
            "type": "image_url",
            "image_url": {"url": url},
        })
    return content


class AliInternalAdapter(ProviderAdapter):
    """Provider adapter for the Ali-internal LLM chat API."""

    _NON_API_OPTION_KEYS = frozenset({
        "max_concurrency",
    })

    def prepare_request(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        image_inputs: List[dict[str, str]],
    ) -> ProviderRequest:
        content = _image_content_items(image_inputs)
        content.append({"type": "text", "text": user_prompt})
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]
        return ProviderRequest(payload=messages, prompt_snapshot=deepcopy(messages))

    def append_correction(
        self,
        request: ProviderRequest,
        *,
        assistant_text: str,
        correction_text: str,
    ) -> ProviderRequest:
        messages = deepcopy(request.payload)
        messages.append({"role": "assistant", "content": assistant_text or ""})
        messages.append({"role": "user", "content": correction_text})
        return ProviderRequest(payload=messages, prompt_snapshot=deepcopy(messages))

    def call(self, request: ProviderRequest) -> str:
        options = {
            key: value
            for key, value in self.spec.options.items()
            if key not in self._NON_API_OPTION_KEYS
        }

        access_key = options.pop("access_key", "")
        quota_id = options.pop("quota_id", "")
        user_id = options.pop("user_id", "345245")
        app = options.pop("app", "llm_application")
        temperature = float(options.pop("temperature", 0.0))
        max_tokens = int(options.pop("max_tokens", 4096))
        max_retries = int(options.pop("max_retries", 5))
        retry_base_delay = float(options.pop("retry_base_delay", 2.0))
        timeout = int(options.pop("timeout", 180))

        api_url = self.spec.base_url

        payload: dict[str, Any] = {
            "model": self.spec.model,
            "prompt": request.payload,
            "access_key": access_key,
            "quota_id": quota_id,
            "user_id": user_id,
            "app": app,
            "params": {
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
        }

        headers = {"Content-Type": "application/json"}

        for attempt in range(max_retries):
            try:
                response = requests.post(
                    api_url,
                    json=payload,
                    headers=headers,
                    timeout=timeout,
                )

                if response.status_code in _RETRYABLE_STATUS_CODES:
                    delay = min(retry_base_delay * (2 ** attempt), 60.0)
                    logger.warning(
                        "重试 %s/%s，HTTP %s，等待 %.1fs",
                        attempt + 1, max_retries, response.status_code, delay,
                    )
                    time.sleep(delay)
                    continue

                response.raise_for_status()
                data = response.json()

                if data.get("code") != 0:
                    error_message = data.get("message", "unknown error")
                    raise RuntimeError(
                        f"Ali-internal API error: code={data.get('code')}, "
                        f"message={error_message}"
                    )

                message = data.get("data", {}).get("message", "")
                if not message:
                    delay = min(retry_base_delay * (2 ** attempt), 60.0)
                    logger.warning(
                        "重试 %s/%s，响应 message 为空，等待 %.1fs",
                        attempt + 1, max_retries, delay,
                    )
                    time.sleep(delay)
                    continue

                if isinstance(message, dict):
                    return json.dumps(message, ensure_ascii=False)
                return str(message)

            except requests.exceptions.Timeout:
                delay = min(retry_base_delay * (2 ** attempt), 60.0)
                logger.warning(
                    "重试 %s/%s，请求超时，等待 %.1fs",
                    attempt + 1, max_retries, delay,
                )
                time.sleep(delay)

            except requests.exceptions.ConnectionError as exc:
                delay = min(retry_base_delay * (2 ** attempt), 60.0)
                logger.warning(
                    "重试 %s/%s，连接错误: %s，等待 %.1fs",
                    attempt + 1, max_retries, exc, delay,
                )
                time.sleep(delay)

        raise RuntimeError(f"Ali-internal API 调用失败，已重试 {max_retries} 次")
