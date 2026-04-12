"""Coordinate prediction TOML job config."""

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
    scenes_dir: str = ""
    images_dir: str = ""
    image_mode: str = "single"  # single / multi_view / top_view
    n_views: int = 4
    split: str | None = None
    max_scenes: int | None = None
    results_dir: str = "output/results"
    run_name: str = "coord_pred"
    tau: float = 0.10
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
            scenes_dir=inp.get("scenes_dir", ""),
            images_dir=images.get("images_dir", ""),
            image_mode=images.get("mode", "single"),
            n_views=images.get("n_views", 4),
            split=sel.get("split"),
            max_scenes=sel.get("max_scenes"),
            results_dir=out.get("results_dir", "output/results"),
            run_name=out.get("run_name", "coord_pred"),
            tau=inp.get("tau", 0.10),
            save_prompt=out.get("save_prompt", False),
        )

    def to_metadata(self) -> dict:
        return {
            "adapter": self.provider.adapter,
            "model": self.provider.model,
            "image_mode": self.image_mode,
            "tau": self.tau,
        }
