"""OpenAI-compatible chat adapter."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, List

from .base import ProviderAdapter, ProviderRequest
from vlm_client import call_vlm, load_image_base64, make_client


def _image_content_items(image_inputs: List[dict[str, str]]) -> list[dict[str, Any]]:
    content = []
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


class OpenAIChatAdapter(ProviderAdapter):
    def __init__(self, provider_spec):
        super().__init__(provider_spec)
        self.client = make_client(self.spec.base_url, self.spec.api_key)

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
        options = dict(self.spec.options)
        provider = options.pop("provider", "")
        extra_body = dict(options.pop("extra_body", None) or {})
        options.pop("max_concurrency", None)

        # call_vlm 接受的标准参数白名单
        call_vlm_known_keys = {
            "temperature", "max_tokens", "max_retries",
            "retry_base_delay", "timeout",
        }
        # 非标准参数（如 access_key, quota_id, user_id 等）归入 extra_body
        unknown_keys = [k for k in options if k not in call_vlm_known_keys]
        for key in unknown_keys:
            extra_body[key] = options.pop(key)

        return call_vlm(
            self.client,
            request.payload,
            self.spec.model,
            provider=provider,
            extra_body=extra_body or None,
            **options,
        )
