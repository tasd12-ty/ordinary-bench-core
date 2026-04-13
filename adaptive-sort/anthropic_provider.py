"""
Anthropic Claude API provider for adaptive sort evaluation.

Directly calls Claude API (not via OpenAI-compatible endpoint) to compare
object distances in 3D scenes.
"""

from __future__ import annotations

import base64
import json
import logging
import sys
from copy import deepcopy
from pathlib import Path

_API_TEST = str(Path(__file__).resolve().parent.parent / "VLM-test" / "API-test")
if _API_TEST not in sys.path:
    sys.path.append(_API_TEST)

from providers.base import ProviderAdapter, ProviderRequest

logger = logging.getLogger(__name__)


class AnthropicAdapter(ProviderAdapter):
    """Provider adapter using the Anthropic Python SDK directly."""

    def __init__(self, provider_spec):
        super().__init__(provider_spec)
        import anthropic
        api_key = provider_spec.api_key or None
        # Let the SDK use ANTHROPIC_API_KEY env var if no key provided
        self._client = anthropic.Anthropic(
            **({"api_key": api_key} if api_key else {}),
        )
        self._model = provider_spec.model
        self._max_tokens = int(provider_spec.options.get("max_tokens", 1024))
        self._temperature = float(provider_spec.options.get("temperature", 0.0))

    def prepare_request(
        self, *, system_prompt: str, user_prompt: str, image_inputs
    ) -> ProviderRequest:
        # Build content blocks for the user message
        content = []

        # Add images first
        for img in (image_inputs or []):
            if img["kind"] == "file":
                img_path = Path(img["value"])
                if img_path.exists():
                    suffix = img_path.suffix.lower()
                    media_type = {
                        ".png": "image/png",
                        ".jpg": "image/jpeg",
                        ".jpeg": "image/jpeg",
                    }.get(suffix, "image/png")
                    with open(img_path, "rb") as f:
                        data = base64.standard_b64encode(f.read()).decode("utf-8")
                    content.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": data,
                        },
                    })
            elif img["kind"] == "url":
                content.append({
                    "type": "image",
                    "source": {"type": "url", "url": img["value"]},
                })

        # Add text prompt
        content.append({"type": "text", "text": user_prompt})

        payload = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "system": system_prompt,
            "messages": [{"role": "user", "content": content}],
        }
        if self._temperature > 0:
            payload["temperature"] = self._temperature

        return ProviderRequest(
            payload=payload,
            prompt_snapshot=deepcopy(payload),
        )

    def append_correction(
        self, request: ProviderRequest, *, assistant_text: str, correction_text: str
    ) -> ProviderRequest:
        payload = deepcopy(request.payload)
        payload["messages"].append({"role": "assistant", "content": assistant_text})
        payload["messages"].append({"role": "user", "content": correction_text})
        return ProviderRequest(payload=payload, prompt_snapshot=deepcopy(payload))

    def call(self, request: ProviderRequest) -> str:
        payload = request.payload
        response = self._client.messages.create(**payload)
        # Extract text from response
        text_parts = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
        return "\n".join(text_parts)
