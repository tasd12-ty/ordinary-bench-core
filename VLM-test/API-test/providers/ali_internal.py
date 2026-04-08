"""Ali-internal LLM platform adapter.

This adapter calls the Ali-internal LLM chat API (llm-chat-api.alibaba-inc.com)
which is NOT OpenAI-compatible.  Key differences:
- The endpoint is a single URL (no /chat/completions suffix).
- The request body uses ``prompt`` (list of message dicts) instead of ``messages``.
- ``access_key``, ``quota_id``, ``user_id``, ``app`` are top-level required fields.
- The response wraps the OpenAI-style completion inside ``data.completion``.
"""

from __future__ import annotations

import json
import logging
import time
from copy import deepcopy
from typing import Any, List, Optional

import requests as http_requests

from .base import ProviderAdapter, ProviderRequest

logger = logging.getLogger(__name__)

_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


def _image_content_items(image_inputs: List[dict[str, str]]) -> list[dict[str, Any]]:
    """Build content items for images (supports file base64 and URL)."""
    import base64
    content: list[dict[str, Any]] = []
    for image in image_inputs:
        if image["kind"] == "file":
            with open(image["value"], "rb") as fh:
                encoded = base64.b64encode(fh.read()).decode("utf-8")
            url = f"data:image/png;base64,{encoded}"
        elif image["kind"] == "url":
            url = image["value"]
        else:
            continue
        content.append({"type": "image_url", "image_url": {"url": url}})
    return content


def _extract_content(data: dict) -> Optional[str]:
    """Extract generated text from the Ali-internal API response.

    Response hierarchy (try in order):
    1. data.data.completion.choices[0].message.content
    2. data.data.message
    """
    inner = data.get("data")
    if not isinstance(inner, dict):
        return None

    completion = inner.get("completion")
    if isinstance(completion, dict):
        choices = completion.get("choices")
        if choices and isinstance(choices, list):
            message = choices[0].get("message", {})
            content = message.get("content")
            if content and isinstance(content, str):
                return content

    message = inner.get("message")
    if message and isinstance(message, str) and message != "success":
        return message

    return None


def call_ali_internal(
    api_url: str,
    prompt: list,
    model: str,
    access_key: str,
    quota_id: str,
    user_id: str = "345245",
    app: str = "llm_application",
    temperature: float = 1.0,
    max_tokens: int = 65536,
    max_retries: int = 5,
    retry_base_delay: float = 2.0,
    timeout: int = 600,
) -> str:
    """Call the Ali-internal LLM chat API with exponential-backoff retry."""
    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "access_key": access_key,
        "quota_id": quota_id,
        "user_id": user_id,
        "app": app,
    }
    if temperature > 0:
        payload["temperature"] = temperature
    if max_tokens:
        payload["max_tokens"] = max_tokens

    headers = {"Content-Type": "application/json"}

    for attempt in range(max_retries):
        try:
            response = http_requests.post(
                api_url,
                json=payload,
                headers=headers,
                timeout=timeout,
            )

            if response.status_code in _RETRYABLE_STATUS_CODES:
                delay = min(retry_base_delay * (2 ** attempt), 60.0)
                logger.warning(
                    "重试 %d/%d，HTTP %d，等待 %.1fs",
                    attempt + 1, max_retries, response.status_code, delay,
                )
                time.sleep(delay)
                continue

            response.raise_for_status()

            data = response.json()

            # Check API-level error code
            api_code = data.get("code")
            if api_code is not None and api_code != 0:
                error_msg = data.get("message", "unknown error")
                delay = min(retry_base_delay * (2 ** attempt), 60.0)
                logger.warning(
                    "重试 %d/%d，API error code=%s msg=%s，等待 %.1fs",
                    attempt + 1, max_retries, api_code, error_msg, delay,
                )
                time.sleep(delay)
                continue

            content = _extract_content(data)
            if content:
                return content

            logger.warning(
                "重试 %d/%d，响应内容为空，响应体: %s",
                attempt + 1, max_retries, response.text[:500],
            )
            delay = min(retry_base_delay * (2 ** attempt), 60.0)
            time.sleep(delay)

        except http_requests.exceptions.Timeout:
            delay = min(retry_base_delay * (2 ** attempt), 60.0)
            logger.warning(
                "重试 %d/%d，请求超时，等待 %.1fs",
                attempt + 1, max_retries, delay,
            )
            time.sleep(delay)

        except http_requests.exceptions.ConnectionError as exc:
            delay = min(retry_base_delay * (2 ** attempt), 60.0)
            logger.warning(
                "重试 %d/%d，连接错误: %s，等待 %.1fs",
                attempt + 1, max_retries, exc, delay,
            )
            time.sleep(delay)

        except http_requests.exceptions.HTTPError as exc:
            raise RuntimeError(
                f"API 请求失败: HTTP {response.status_code}, "
                f"响应: {response.text[:500]}"
            ) from exc

    raise RuntimeError(f"Ali-internal API 调用失败，已重试 {max_retries} 次")


class AliInternalAdapter(ProviderAdapter):
    """Provider adapter for the Ali-internal LLM chat API."""

    def prepare_request(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        image_inputs: List[dict[str, str]],
    ) -> ProviderRequest:
        content = _image_content_items(image_inputs)
        content.append({"type": "text", "text": user_prompt})

        prompt = []
        if system_prompt:
            prompt.append({"role": "system", "content": system_prompt})
        prompt.append({"role": "user", "content": content})

        snapshot = deepcopy(prompt)
        return ProviderRequest(payload=prompt, prompt_snapshot=snapshot)

    def append_correction(
        self,
        request: ProviderRequest,
        *,
        assistant_text: str,
        correction_text: str,
    ) -> ProviderRequest:
        prompt = deepcopy(request.payload)
        prompt.append({"role": "assistant", "content": assistant_text or ""})
        prompt.append({"role": "user", "content": correction_text})
        return ProviderRequest(payload=prompt, prompt_snapshot=deepcopy(prompt))

    def call(self, request: ProviderRequest) -> str:
        options = dict(self.spec.options)
        # Pop fields that map to call_ali_internal named params
        access_key = options.pop("access_key")
        quota_id = options.pop("quota_id")
        user_id = options.pop("user_id", "345245")
        app = options.pop("app", "llm_application")
        options.pop("max_concurrency", None)

        return call_ali_internal(
            api_url=self.spec.base_url,
            prompt=request.payload,
            model=self.spec.model,
            access_key=access_key,
            quota_id=quota_id,
            user_id=user_id,
            app=app,
            **options,
        )
