"""VRF TOML job 配置加载。"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[no-redef]


def _resolve_env(value: str) -> str:
    if isinstance(value, str) and value.startswith("env:"):
        return os.environ.get(value[4:], "")
    return value


@dataclass
class ProviderSpec:
    adapter: str = "openai_chat"
    model: str = ""
    base_url: str = ""
    api_key: str = ""
    options: dict = field(default_factory=dict)


@dataclass
class JobSpec:
    provider: ProviderSpec = field(default_factory=ProviderSpec)
    questions_dir: str = ""
    images_dir: str = ""
    image_mode: str = "single"
    split: str | None = None
    max_scenes: int | None = None
    results_dir: str = "output/results"
    run_name: str = "vrf_run"
    batch_size: int = 20
    react_max_rounds: int = 1
    missing_threshold: float = 0.3
    save_prompt: bool = False

    @classmethod
    def from_toml(cls, path: str | Path) -> JobSpec:
        with open(path, "rb") as f:
            raw = tomllib.load(f)

        prov_raw = raw.get("provider", {})
        provider = ProviderSpec(
            adapter=prov_raw.get("adapter", "openai_chat"),
            model=prov_raw.get("model", ""),
            base_url=_resolve_env(prov_raw.get("base_url", "")),
            api_key=_resolve_env(prov_raw.get("api_key", "")),
            options=prov_raw.get("options", {}),
        )

        inp = raw.get("input", {})
        sel = raw.get("selection", {})
        out = raw.get("output", {})
        images = raw.get("images", {})

        return cls(
            provider=provider,
            questions_dir=inp.get("questions_dir", ""),
            images_dir=images.get("images_dir", ""),
            image_mode=images.get("mode", "single"),
            split=sel.get("split"),
            max_scenes=sel.get("max_scenes"),
            results_dir=out.get("results_dir", "output/results"),
            run_name=out.get("run_name", "vrf_run"),
            batch_size=inp.get("batch_size", 20),
            react_max_rounds=prov_raw.get("options", {}).get("react_max_rounds", 1),
            missing_threshold=prov_raw.get("options", {}).get("missing_threshold", 0.3),
            save_prompt=out.get("save_prompt", False),
        )

    def to_metadata(self) -> dict:
        return {
            "adapter": self.provider.adapter,
            "model": self.provider.model,
            "image_mode": self.image_mode,
        }
