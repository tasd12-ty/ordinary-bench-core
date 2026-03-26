"""Base classes for provider adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List


@dataclass(slots=True)
class ProviderRequest:
    payload: Any
    prompt_snapshot: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)


class ProviderAdapter:
    def __init__(self, provider_spec):
        self.spec = provider_spec

    def prepare_request(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        image_inputs: List[dict[str, str]],
    ) -> ProviderRequest:
        raise NotImplementedError

    def append_correction(
        self,
        request: ProviderRequest,
        *,
        assistant_text: str,
        correction_text: str,
    ) -> ProviderRequest:
        raise NotImplementedError

    def call(self, request: ProviderRequest) -> str:
        raise NotImplementedError

    def serialize_request(self, request: ProviderRequest) -> Any:
        return request.prompt_snapshot if request.prompt_snapshot is not None else request.payload
