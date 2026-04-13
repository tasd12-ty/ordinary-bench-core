"""
TOML job specification for adaptive sort evaluation.

Simplified variant of VLM-test/API-test/job_spec.py with sorting-specific config.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
import re
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib
from typing import Any, Dict, Optional


_ENV_PATTERN = re.compile(r"\$\{([A-Z0-9_]+)\}")


def _expand_env_string(value: str) -> str:
    if value.startswith("env:"):
        return os.environ.get(value[4:], "")

    def repl(match: re.Match[str]) -> str:
        return os.environ.get(match.group(1), "")

    return _ENV_PATTERN.sub(repl, value)


def _resolve_env_refs(value: Any) -> Any:
    if isinstance(value, str):
        return _expand_env_string(value)
    if isinstance(value, list):
        return [_resolve_env_refs(item) for item in value]
    if isinstance(value, dict):
        return {key: _resolve_env_refs(item) for key, item in value.items()}
    return value


def _resolve_path(base_dir: Path, raw_path: str) -> str:
    expanded = _expand_env_string(raw_path)
    if not expanded:
        return expanded
    candidate = Path(expanded)
    if candidate.is_absolute():
        return str(candidate)
    if "://" in expanded or expanded.startswith("data:"):
        return expanded
    return str((base_dir / candidate).resolve())


@dataclass(slots=True)
class ProviderSpec:
    adapter: str
    model: str
    base_url: str = ""
    api_key: str = ""
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class InputSpec:
    scenes_dir: str
    tau: float = 0.10


@dataclass(slots=True)
class ImageSpec:
    mode: str = "single"
    single_view_root: str = ""
    multi_view_root: str = ""
    n_views: int = 1


@dataclass(slots=True)
class SelectionSpec:
    split: str = ""
    scene: str = ""
    max_scenes: Optional[int] = None


@dataclass(slots=True)
class SortingSpec:
    pivot_strategy: str = "middle"
    max_retries: int = 3
    retry_base_delay: float = 2.0
    max_concurrency: int = 4


@dataclass(slots=True)
class OutputSpec:
    results_dir: str
    run_name: str = ""


@dataclass(slots=True)
class AdaptiveSortJobSpec:
    provider: ProviderSpec
    input: InputSpec
    images: ImageSpec
    selection: SelectionSpec
    sorting: SortingSpec
    output: OutputSpec
    job_name: str = ""
    source_path: str = ""

    @classmethod
    def from_toml(cls, path: str | Path) -> "AdaptiveSortJobSpec":
        job_path = Path(path).resolve()
        with open(job_path, "rb") as handle:
            data = tomllib.load(handle)
        data.setdefault("output", {})
        data["output"].setdefault("run_name", job_path.stem)
        return cls.from_dict(data, base_dir=job_path.parent, source_path=str(job_path))

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        *,
        base_dir: str | Path,
        source_path: str = "",
    ) -> "AdaptiveSortJobSpec":
        base = Path(base_dir).resolve()
        resolved = _resolve_env_refs(data)

        provider_raw = dict(resolved.get("provider", {}))
        input_raw = dict(resolved.get("input", {}))
        images_raw = dict(resolved.get("images", {}))
        selection_raw = dict(resolved.get("selection", {}))
        sorting_raw = dict(resolved.get("sorting", {}))
        output_raw = dict(resolved.get("output", {}))

        provider = ProviderSpec(
            adapter=str(provider_raw.get("adapter", "")).strip(),
            model=str(provider_raw.get("model", "")).strip(),
            base_url=str(provider_raw.get("base_url", "")).strip(),
            api_key=str(provider_raw.get("api_key", "")).strip(),
            options=dict(provider_raw.get("options", {})),
        )
        input_spec = InputSpec(
            scenes_dir=_resolve_path(base, str(input_raw.get("scenes_dir", ""))),
            tau=float(input_raw.get("tau", 0.10)),
        )
        image_spec = ImageSpec(
            mode=str(images_raw.get("mode", "single")),
            single_view_root=_resolve_path(base, str(images_raw.get("single_view_root", ""))),
            multi_view_root=_resolve_path(base, str(images_raw.get("multi_view_root", ""))),
            n_views=int(images_raw.get("n_views", 1)),
        )
        selection = SelectionSpec(
            split=str(selection_raw.get("split", "")),
            scene=str(selection_raw.get("scene", "")),
            max_scenes=selection_raw.get("max_scenes"),
        )
        sorting = SortingSpec(
            pivot_strategy=str(sorting_raw.get("pivot_strategy", "middle")),
            max_retries=int(sorting_raw.get("max_retries", 3)),
            retry_base_delay=float(sorting_raw.get("retry_base_delay", 2.0)),
            max_concurrency=int(sorting_raw.get("max_concurrency", 4)),
        )
        output = OutputSpec(
            results_dir=_resolve_path(base, str(output_raw.get("results_dir", ""))),
            run_name=str(output_raw.get("run_name", "")).strip(),
        )

        job = cls(
            provider=provider,
            input=input_spec,
            images=image_spec,
            selection=selection,
            sorting=sorting,
            output=output,
            job_name=str(resolved.get("job_name", output.run_name or Path(source_path).stem)),
            source_path=source_path,
        )
        job.validate()
        return job

    def validate(self) -> None:
        if not self.provider.adapter:
            raise ValueError("provider.adapter is required")
        if not self.provider.model:
            raise ValueError("provider.model is required")
        if not self.input.scenes_dir:
            raise ValueError("input.scenes_dir is required")
        if not self.output.results_dir:
            raise ValueError("output.results_dir is required")

    @property
    def run_name(self) -> str:
        return self.output.run_name or self.job_name or self.provider.model

    def to_metadata(self) -> dict:
        return {
            "job_name": self.job_name,
            "model": self.provider.model,
            "adapter": self.provider.adapter,
            "tau": self.input.tau,
            "pivot_strategy": self.sorting.pivot_strategy,
            "image_mode": self.images.mode,
        }
