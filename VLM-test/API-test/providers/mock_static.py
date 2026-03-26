"""Static mock provider for local smoke tests."""

from __future__ import annotations

from copy import deepcopy

from .base import ProviderAdapter, ProviderRequest


class MockStaticAdapter(ProviderAdapter):
    def prepare_request(self, *, system_prompt: str, user_prompt: str, image_inputs):
        payload = {
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "image_inputs": list(image_inputs),
        }
        return ProviderRequest(payload=payload, prompt_snapshot=deepcopy(payload))

    def append_correction(self, request: ProviderRequest, *, assistant_text: str, correction_text: str):
        payload = deepcopy(request.payload)
        payload["assistant_text"] = assistant_text
        payload["correction_text"] = correction_text
        return ProviderRequest(payload=payload, prompt_snapshot=deepcopy(payload))

    def call(self, request: ProviderRequest) -> str:
        return str(self.spec.options.get("raw_response", "[]"))
