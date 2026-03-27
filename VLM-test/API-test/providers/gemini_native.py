"""Gemini-native provider adapter."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, List

from .base import ProviderAdapter, ProviderRequest
from gemini_vlm_client import call_gemini_vlm
from vlm_client import load_image_base64


def _serialize_gemini_prompt(system_instruction, prompt: list) -> list[dict[str, Any]]:
    saved = []
    if system_instruction:
        system_text_parts = system_instruction.get("parts", [])
        system_text = "\n".join(
            part.get("text", "") for part in system_text_parts if "text" in part
        )
        saved.append({"role": "system", "content": system_text})

    for message in prompt:
        content_items = []
        for part in message.get("parts", []):
            if "text" in part:
                content_items.append({"type": "text", "text": part["text"]})
            elif "inlineData" in part:
                content_items.append({
                    "type": "image_url",
                    "image_url": {"url": part["inlineData"].get("data", "")},
                })
        if len(content_items) == 1 and content_items[0]["type"] == "text":
            saved.append({"role": message.get("role", "user"), "content": content_items[0]["text"]})
        else:
            saved.append({"role": message.get("role", "user"), "content": content_items})
    return saved


def _build_prompt(
    system_prompt: str,
    user_prompt: str,
    image_inputs: List[dict[str, str]],
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    system_instruction = None
    if system_prompt:
        system_instruction = {"role": "user", "parts": [{"text": system_prompt}]}

    user_parts = []
    for image in image_inputs:
        if image["kind"] == "file":
            image_b64 = load_image_base64(image["value"])
            user_parts.append({
                "inlineData": {
                    "mimeType": "image/png",
                    "data": image_b64,
                }
            })
        elif image["kind"] == "url":
            user_parts.append({
                "inlineData": {
                    "mimeType": "image/png",
                    "data": image["value"],
                }
            })
        else:
            continue
    user_parts.append({"text": user_prompt})
    prompt = [{"role": "user", "parts": user_parts}]
    return prompt, system_instruction


class GeminiNativeAdapter(ProviderAdapter):
    def prepare_request(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        image_inputs: List[dict[str, str]],
    ) -> ProviderRequest:
        prompt, system_instruction = _build_prompt(system_prompt, user_prompt, image_inputs)
        snapshot = _serialize_gemini_prompt(system_instruction, prompt)
        return ProviderRequest(
            payload={"prompt": prompt, "system_instruction": system_instruction},
            prompt_snapshot=snapshot,
        )

    def append_correction(
        self,
        request: ProviderRequest,
        *,
        assistant_text: str,
        correction_text: str,
    ) -> ProviderRequest:
        prompt = deepcopy(request.payload["prompt"])
        prompt.extend([
            {"role": "model", "parts": [{"text": assistant_text or ""}]},
            {"role": "user", "parts": [{"text": correction_text}]},
        ])
        system_instruction = deepcopy(request.payload["system_instruction"])
        snapshot = _serialize_gemini_prompt(system_instruction, prompt)
        return ProviderRequest(
            payload={"prompt": prompt, "system_instruction": system_instruction},
            prompt_snapshot=snapshot,
        )

    _NON_API_OPTION_KEYS = frozenset({
        "max_concurrency",
    })

    def call(self, request: ProviderRequest) -> str:
        options = {
            key: value
            for key, value in self.spec.options.items()
            if key not in self._NON_API_OPTION_KEYS
        }
        return call_gemini_vlm(
            prompt=request.payload["prompt"],
            system_instruction=request.payload["system_instruction"],
            model=self.spec.model,
            api_url=self.spec.base_url,
            **options,
        )
