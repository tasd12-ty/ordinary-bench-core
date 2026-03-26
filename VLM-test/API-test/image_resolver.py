"""Scene image resolution helpers."""

from __future__ import annotations

from pathlib import Path
import random


_URL_PREFIXES = ("http://", "https://", "oss://", "data:")


def _is_url_root(root: str) -> bool:
    return root.startswith(_URL_PREFIXES)


def _single_image_value(root: str, scene_id: str) -> str:
    if _is_url_root(root):
        return f"{root.rstrip('/')}/{scene_id}.png"
    return str(Path(root) / f"{scene_id}.png")


def _multi_view_image_values(root: str, scene_id: str, n_views: int) -> list[str]:
    values = []
    for idx in range(n_views):
        if _is_url_root(root):
            values.append(f"{root.rstrip('/')}/{scene_id}/view_{idx}.png")
        else:
            values.append(str(Path(root) / scene_id / f"view_{idx}.png"))
    return values


def _candidate_single_images(root: str) -> list[Path]:
    if _is_url_root(root):
        raise ValueError("wrong_single image mode requires a local single_view_root")
    return sorted(Path(root).glob("*.png"))


def _stable_scene_seed(scene_id: str) -> int:
    total = 0
    for idx, ch in enumerate(scene_id):
        total += (idx + 1) * ord(ch)
    return total


def resolve_scene_images(scene_id: str, image_spec) -> list[dict[str, str]]:
    mode = image_spec.mode
    if mode == "none":
        return []
    if mode == "single":
        value = _single_image_value(image_spec.single_view_root, scene_id)
        kind = "url" if _is_url_root(image_spec.single_view_root) else "file"
        return [{"kind": kind, "value": value}]
    if mode == "wrong_single":
        candidates = [path for path in _candidate_single_images(image_spec.single_view_root) if path.stem != scene_id]
        if not candidates:
            raise ValueError("No candidate wrong images available")
        rng = random.Random(image_spec.wrong_image_seed + _stable_scene_seed(scene_id))
        wrong_path = rng.choice(candidates)
        return [{"kind": "file", "value": str(wrong_path)}]
    if mode == "multi_view":
        values = _multi_view_image_values(image_spec.multi_view_root, scene_id, image_spec.n_views)
        kind = "url" if _is_url_root(image_spec.multi_view_root) else "file"
        return [{"kind": kind, "value": value} for value in values]
    raise ValueError(f"Unsupported image mode: {mode!r}")
