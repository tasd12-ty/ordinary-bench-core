"""Provider adapter registry."""

from .base import ProviderAdapter, ProviderRequest


def create_provider_adapter(provider_spec):
    adapter = provider_spec.adapter
    if adapter == "openai_chat":
        from .openai_chat import OpenAIChatAdapter
        return OpenAIChatAdapter(provider_spec)
    if adapter == "gemini_native":
        from .gemini_native import GeminiNativeAdapter
        return GeminiNativeAdapter(provider_spec)
    if adapter == "ali_internal":
        from .ali_internal import AliInternalAdapter
        return AliInternalAdapter(provider_spec)
    if adapter == "mock_static":
        from .mock_static import MockStaticAdapter
        return MockStaticAdapter(provider_spec)
    raise ValueError(f"Unknown provider adapter: {adapter!r}")


__all__ = [
    "ProviderAdapter",
    "ProviderRequest",
    "create_provider_adapter",
]
