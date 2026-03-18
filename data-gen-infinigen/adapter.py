#!/usr/bin/env python3
"""Adapt Infinigen frame metadata to ordinary-bench scene JSON."""

import argparse
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


COLOR_TERMS = [
    "black", "white", "gray", "grey", "red", "green", "blue", "yellow",
    "brown", "orange", "pink", "purple", "beige", "silver", "gold",
]


def _slug(text: str) -> str:
    chars = []
    for ch in text.lower():
        if ch.isalnum():
            chars.append(ch)
        elif chars and chars[-1] != "_":
            chars.append("_")
    return "".join(chars).strip("_") or "object"


def _load_json(path: Path) -> Any:
    with path.open("r") as f:
        return json.load(f)


def _load_camview(path: Path) -> Dict[str, np.ndarray]:
    if path.suffix == ".npz":
        data = np.load(path)
        return {k: data[k] for k in data.files}
    if path.suffix == ".json":
        raw = _load_json(path)
        return {k: np.asarray(v, dtype=np.float64) for k, v in raw.items()}
    raise ValueError(f"Unsupported camview format: {path}")


def _transform(T: np.ndarray, p: np.ndarray) -> np.ndarray:
    assert T.shape == (4, 4)
    return (T[:3, :3] @ p.T + T[:3, [3]]).T


def _bbox_corners(min_pt: Sequence[float], max_pt: Sequence[float]) -> np.ndarray:
    min_x, min_y, min_z = min_pt
    max_x, max_y, max_z = max_pt
    return np.asarray([
        [min_x, min_y, min_z],
        [max_x, min_y, min_z],
        [min_x, max_y, min_z],
        [max_x, max_y, min_z],
        [min_x, min_y, max_z],
        [max_x, min_y, max_z],
        [min_x, max_y, max_z],
        [max_x, max_y, max_z],
    ], dtype=np.float64)


def _world_to_bench(world_xyz: np.ndarray) -> np.ndarray:
    x, y, z = world_xyz.tolist()
    return np.asarray([x, z, -y], dtype=np.float64)


def _project_point(K: np.ndarray, camera_pose: np.ndarray, point_world: np.ndarray) -> Tuple[np.ndarray, float]:
    point_camera = _transform(np.linalg.inv(camera_pose), point_world.reshape(1, 3))[0]
    depth = float(point_camera[2])
    if depth <= 0:
        return np.asarray([math.nan, math.nan], dtype=np.float64), depth
    uvw = K @ point_camera
    uv = uvw[:2] / uvw[2]
    return uv.astype(np.float64), depth


def _extract_primary_label(meta: dict) -> str:
    tags = meta.get("tags") or []
    name = str(meta.get("name") or "").strip()
    for tag in tags:
        tag = str(tag).strip()
        if tag and tag.lower() not in {"object", "furniture"}:
            return tag
    if name:
        return name
    return "object"


def _extract_color(meta: dict) -> str:
    fields = [meta.get("name", "")] + list(meta.get("tags") or []) + list(meta.get("materials") or [])
    joined = " ".join(str(x).lower() for x in fields)
    for color in COLOR_TERMS:
        if color in joined:
            return "gray" if color == "grey" else color
    return ""


def _extract_material(meta: dict) -> str:
    materials = meta.get("materials") or []
    if materials:
        return _slug(str(materials[0]))
    return ""


def _screen_center_score(uv: np.ndarray, hw: np.ndarray) -> float:
    h, w = float(hw[0]), float(hw[1])
    center = np.asarray([w / 2.0, h / 2.0], dtype=np.float64)
    dist = np.linalg.norm(uv - center)
    diag = math.hypot(w, h)
    return max(0.0, 1.0 - dist / max(diag, 1.0))


def _query_score(text: str, queries: Sequence[str]) -> float:
    if not queries:
        return 0.0
    lowered = text.lower()
    matches = sum(1 for q in queries if q.lower() in lowered)
    return float(matches)


@dataclass
class Candidate:
    object_index: Optional[int]
    instance_index: int
    label: str
    meta_name: str
    tags: List[str]
    materials: List[str]
    world_center: np.ndarray
    bench_center: np.ndarray
    uv: np.ndarray
    depth: float
    world_diag: float
    rotation: float
    query_score: float
    center_score: float
    sort_score: float


def _size_bucket(diag: float, all_diags: Sequence[float]) -> str:
    if not all_diags:
        return "medium"
    sorted_diags = sorted(all_diags)
    lo = sorted_diags[max(0, len(sorted_diags) // 3 - 1)]
    hi = sorted_diags[max(0, (2 * len(sorted_diags)) // 3 - 1)]
    if diag <= lo:
        return "small"
    if diag >= hi:
        return "large"
    return "medium"


def _camera_sort_key(camera_id: str) -> Tuple[int, str]:
    tail = camera_id.rsplit("_", 1)[-1]
    if tail.isdigit():
        return (int(tail), camera_id)
    return (10**9, camera_id)


def _find_first(scene_root: Path, patterns: Sequence[str]) -> Optional[Path]:
    for pattern in patterns:
        matches = sorted(scene_root.glob(pattern))
        if matches:
            return matches[0]
    return None


def _in_frame(uv: np.ndarray, hw: np.ndarray, margin: int) -> bool:
    if not np.isfinite(uv).all():
        return False
    h = float(hw[0]) if hw.size >= 1 else 0.0
    w = float(hw[1]) if hw.size >= 2 else 0.0
    if w > 0 and (uv[0] < margin or uv[0] > w - margin):
        return False
    if h > 0 and (uv[1] < margin or uv[1] > h - margin):
        return False
    return True


def discover_view_bundles(scene_root: Path) -> List[Dict[str, Optional[Path]]]:
    camera_ids = set()
    for stem in ("Image", "camview", "Objects"):
        root = scene_root / "frames" / stem
        if root.exists():
            for path in root.glob("camera_*"):
                if path.is_dir():
                    camera_ids.add(path.name)

    if not camera_ids:
        return [discover_frame_bundle(scene_root)]

    bundles: List[Dict[str, Optional[Path]]] = []
    for camera_id in sorted(camera_ids, key=_camera_sort_key):
        image = _find_first(scene_root, [
            f"frames/Image/{camera_id}/Image_*.png",
            "Image_*.png",
        ])
        camview = _find_first(scene_root, [
            f"frames/camview/{camera_id}/camview_*.npz",
            f"frames/camview/{camera_id}/camview_*.json",
            "camview_*.npz",
            "camview_*.json",
        ])
        objects = _find_first(scene_root, [
            f"frames/Objects/{camera_id}/Objects_*.json",
            "frames/Objects/camera_0/Objects_*.json",
            "Objects_*.json",
        ])
        if camview is None or objects is None:
            continue
        bundles.append({
            "camera_id": camera_id,
            "image": image,
            "camview": camview,
            "objects": objects,
        })

    if not bundles:
        raise FileNotFoundError(f"Could not discover any camera bundles under {scene_root}")
    return bundles


def discover_frame_bundle(scene_root: Path) -> Dict[str, Optional[Path]]:
    patterns = {
        "image": ["frames/Image/camera_0/Image_*.png", "Image_*.png"],
        "camview": ["frames/camview/camera_0/camview_*.npz", "frames/camview/camera_0/camview_*.json", "camview_*.npz", "camview_*.json"],
        "objects": ["frames/Objects/camera_0/Objects_*.json", "Objects_*.json"],
    }
    bundle: Dict[str, Optional[Path]] = {"camera_id": "camera_0"}
    for key, globs in patterns.items():
        bundle[key] = _find_first(scene_root, globs)
        if key == "image" and bundle[key] is None:
            continue
        if key != "image" and bundle[key] is None:
            raise FileNotFoundError(f"Could not find {key} under {scene_root}")
    return bundle


def build_candidates(
    object_meta: List[dict],
    camview: Dict[str, np.ndarray],
    queries: Sequence[str],
    min_depth: float,
    min_screen_margin: int,
) -> List[Candidate]:
    K = np.asarray(camview["K"], dtype=np.float64)
    T = np.asarray(camview["T"], dtype=np.float64)
    hw = np.asarray(camview.get("HW", [0, 0]), dtype=np.float64).reshape(-1)
    fallback_hw = hw if hw.size == 2 else np.asarray([0.0, 0.0], dtype=np.float64)
    candidates: List[Candidate] = []

    for meta in object_meta:
        min_pt = meta.get("min")
        max_pt = meta.get("max")
        model_mats = meta.get("model_matrices") or []
        if min_pt is None or max_pt is None:
            continue
        local_min = np.asarray(min_pt, dtype=np.float64)
        local_max = np.asarray(max_pt, dtype=np.float64)
        local_center = (local_min + local_max) / 2.0
        local_corners = _bbox_corners(local_min, local_max)
        label = _extract_primary_label(meta)
        searchable = " ".join(
            [str(meta.get("name", ""))]
            + [str(x) for x in meta.get("tags") or []]
            + [str(x) for x in meta.get("materials") or []]
        )

        for instance_index, model_mat in enumerate(model_mats):
            M = np.asarray(model_mat, dtype=np.float64)
            if M.shape != (4, 4):
                continue
            world_center = _transform(M, local_center.reshape(1, 3))[0]
            uv, depth = _project_point(K, T, world_center)
            if depth <= min_depth or not np.isfinite(uv).all():
                continue
            if not _in_frame(uv, hw, min_screen_margin):
                continue

            world_corners = _transform(M, local_corners)
            world_ranges = world_corners.max(axis=0) - world_corners.min(axis=0)
            world_diag = float(np.linalg.norm(world_ranges))
            rot_z = math.degrees(math.atan2(M[1, 0], M[0, 0]))
            q_score = _query_score(searchable, queries)
            c_score = _screen_center_score(uv, fallback_hw)
            score = q_score * 10.0 + world_diag + c_score

            candidates.append(Candidate(
                object_index=int(meta.get("object_index")) if meta.get("object_index") is not None else None,
                instance_index=instance_index,
                label=_slug(label),
                meta_name=str(meta.get("name", "")),
                tags=[str(x) for x in meta.get("tags") or []],
                materials=[str(x) for x in meta.get("materials") or []],
                world_center=world_center,
                bench_center=_world_to_bench(world_center),
                uv=uv,
                depth=depth,
                world_diag=world_diag,
                rotation=rot_z,
                query_score=q_score,
                center_score=c_score,
                sort_score=score,
            ))

    candidates.sort(key=lambda c: (-c.sort_score, c.depth, c.meta_name))
    return candidates


def _candidate_to_object(
    cand: Candidate,
    object_id: str,
    diags: Sequence[float],
    uv: np.ndarray,
    depth: float,
    include_source: bool,
) -> dict:
    meta_stub = {
        "name": cand.meta_name,
        "tags": cand.tags,
        "materials": cand.materials,
    }
    payload = {
        "id": object_id,
        "shape": cand.label,
        "size": _size_bucket(cand.world_diag, diags),
        "material": _extract_material(meta_stub),
        "color": _extract_color(meta_stub),
        "3d_coords": [round(float(x), 6) for x in cand.bench_center.tolist()],
        "pixel_coords": [
            int(round(float(uv[0]))),
            int(round(float(uv[1]))),
            round(float(depth), 6),
        ],
        "rotation": round(float(cand.rotation), 4),
    }
    if include_source:
        payload["source_name"] = cand.meta_name
        payload["source_tags"] = cand.tags
        payload["source_materials"] = cand.materials
        payload["source_object_index"] = cand.object_index
        payload["source_instance_index"] = cand.instance_index
    return payload


def _camera_directions(camera_pose: np.ndarray) -> Dict[str, List[float]]:
    right = camera_pose[:3, 0]
    below = camera_pose[:3, 1]
    front = camera_pose[:3, 2]
    directions = {
        "behind": _world_to_bench(-front),
        "front": _world_to_bench(front),
        "left": _world_to_bench(-right),
        "right": _world_to_bench(right),
        "above": _world_to_bench(-below),
        "below": _world_to_bench(below),
    }
    return {
        key: [round(float(x), 6) for x in vec.tolist()]
        for key, vec in directions.items()
    }


def _copy_view_images(
    view_bundles: Sequence[Dict[str, Optional[Path]]],
    multi_view_dir: Optional[Path],
) -> List[Optional[Path]]:
    copied: List[Optional[Path]] = []
    if multi_view_dir is not None:
        multi_view_dir.mkdir(parents=True, exist_ok=True)
        for stale in multi_view_dir.glob("view_*.png"):
            stale.unlink()
    for view_index, bundle in enumerate(view_bundles):
        image_src = bundle.get("image")
        if multi_view_dir is None or image_src is None:
            copied.append(None)
            continue
        view_dst = multi_view_dir / f"view_{view_index}.png"
        shutil.copy2(image_src, view_dst)
        copied.append(view_dst)
    return copied


def adapt_scene(
    source_root: Path,
    scene_id: str,
    split: str,
    max_objects: int,
    query_terms: Sequence[str],
    min_depth: float,
    min_screen_margin: int,
    image_dst: Optional[Path] = None,
    multi_view_dir: Optional[Path] = None,
) -> dict:
    view_bundles = discover_view_bundles(source_root)
    bundle = view_bundles[0]
    camview = _load_camview(bundle["camview"])  # type: ignore[arg-type]
    object_meta = _load_json(bundle["objects"])  # type: ignore[arg-type]
    candidates = build_candidates(object_meta, camview, query_terms, min_depth, min_screen_margin)
    selected = candidates[:max_objects]
    if len(selected) < 3:
        raise RuntimeError(
            f"Adapter found only {len(selected)} visible candidate objects under {source_root}"
        )

    diags = [c.world_diag for c in selected]
    object_ids = [f"obj_{idx}" for idx, _ in enumerate(selected)]
    objects = [
        _candidate_to_object(
            cand=cand,
            object_id=object_ids[idx],
            diags=diags,
            uv=cand.uv,
            depth=cand.depth,
            include_source=True,
        )
        for idx, cand in enumerate(selected)
    ]

    if image_dst is not None and bundle["image"] is not None:
        image_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(bundle["image"], image_dst)

    copied_view_images = _copy_view_images(view_bundles, multi_view_dir)
    views = []
    source_views = []
    for view_index, view_bundle in enumerate(view_bundles):
        view_camview = _load_camview(view_bundle["camview"])  # type: ignore[arg-type]
        hw = np.asarray(view_camview.get("HW", [0, 0]), dtype=np.int64).reshape(-1)
        camera_pose = np.asarray(view_camview["T"], dtype=np.float64)
        view_objects = []
        K = np.asarray(view_camview["K"], dtype=np.float64)
        T = np.asarray(view_camview["T"], dtype=np.float64)
        for object_id, cand in zip(object_ids, selected):
            uv, depth = _project_point(K, T, cand.world_center)
            if depth <= min_depth:
                continue
            if not _in_frame(uv, hw, min_screen_margin):
                continue
            view_objects.append(
                _candidate_to_object(
                    cand=cand,
                    object_id=object_id,
                    diags=diags,
                    uv=uv,
                    depth=depth,
                    include_source=False,
                )
            )

        copied_image = copied_view_images[view_index]
        if copied_image is not None:
            image_name = copied_image.name
        elif view_bundle["image"] is not None:
            image_name = view_bundle["image"].name  # type: ignore[union-attr]
        else:
            image_name = ""

        views.append({
            "view_id": f"view_{view_index}",
            "image_path": image_name,
            "camera": {
                "camera_id": str(view_bundle.get("camera_id") or f"camera_{view_index}"),
                "intrinsics": np.asarray(view_camview["K"], dtype=np.float64).round(6).tolist(),
                "pose_cv_world": camera_pose.round(6).tolist(),
                "position_cv_world": camera_pose[:3, 3].round(6).tolist(),
                "position": _world_to_bench(camera_pose[:3, 3]).round(6).tolist(),
                "height": int(hw[0]) if hw.size >= 1 else 0,
                "width": int(hw[1]) if hw.size >= 2 else 0,
            },
            "directions": _camera_directions(camera_pose),
            "objects": view_objects,
        })
        source_views.append({
            "camera_id": str(view_bundle.get("camera_id") or f"camera_{view_index}"),
            "image_path": str(view_bundle["image"]) if view_bundle["image"] is not None else "",
            "camview_path": str(view_bundle["camview"]),
            "objects_path": str(view_bundle["objects"]),
        })

    scene = {
        "scene_id": scene_id,
        "split": split,
        "image_index": 0,
        "n_objects": len(objects),
        "objects": objects,
        "world_constraints": {},
        "views": views,
        "source": {
            "backend": "infinigen",
            "image_path": str(bundle["image"]) if bundle["image"] is not None else "",
            "camview_path": str(bundle["camview"]),
            "objects_path": str(bundle["objects"]),
            "query_terms": list(query_terms),
            "view_sources": source_views,
        },
    }
    return scene


def export_native_record(
    source_root: Path,
    scene_id: str,
    dest_dir: Path,
    copy_image: bool = False,
) -> dict:
    """Preserve the original Infinigen record alongside the adapted scene.

    The goal is to keep Infinigen-native metadata in a stable, benchmark-local
    location without replacing or reformatting it.
    """
    view_bundles = discover_view_bundles(source_root)
    dest_dir.mkdir(parents=True, exist_ok=True)
    for legacy in dest_dir.glob("Objects_*.json"):
        legacy.unlink()
    for legacy in dest_dir.glob("camview_*"):
        legacy.unlink()
    for legacy in dest_dir.glob("Image_*.png"):
        legacy.unlink()
    for stale_dir in dest_dir.glob("camera_*"):
        if stale_dir.is_dir():
            shutil.rmtree(stale_dir)

    copied_views = []
    for view_index, view_bundle in enumerate(view_bundles):
        camera_id = str(view_bundle.get("camera_id") or f"camera_{view_index}")
        view_dir = dest_dir / camera_id
        view_dir.mkdir(parents=True, exist_ok=True)
        copied = {}
        for key in ("objects", "camview"):
            src = view_bundle[key]
            assert src is not None
            dst = view_dir / src.name
            shutil.copy2(src, dst)
            copied[key] = dst

        if copy_image and view_bundle["image"] is not None:
            src = view_bundle["image"]
            dst = view_dir / src.name
            shutil.copy2(src, dst)
            copied["image"] = dst

        copied_views.append({
            "camera_id": camera_id,
            "files": {
                "objects": str(copied["objects"]),
                "camview": str(copied["camview"]),
                "image": str(copied["image"]) if "image" in copied else "",
            },
            "original_files": {
                "objects": str(view_bundle["objects"]),
                "camview": str(view_bundle["camview"]),
                "image": str(view_bundle["image"]) if view_bundle["image"] is not None else "",
            },
        })

    primary = copied_views[0]

    manifest = {
        "scene_id": scene_id,
        "backend": "infinigen",
        "source_root": str(source_root),
        "primary_camera_id": primary["camera_id"],
        "files": primary["files"],
        "original_files": primary["original_files"],
        "views": copied_views,
    }
    manifest_path = dest_dir / "manifest.json"
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Adapt Infinigen metadata into ordinary-bench scene JSON")
    parser.add_argument("--source-root", required=True, help="Infinigen scene root containing frames/")
    parser.add_argument("--scene-id", default="ifg_000000")
    parser.add_argument("--split", default="ifg")
    parser.add_argument("--max-objects", type=int, default=6)
    parser.add_argument("--query-term", action="append", default=[], help="Prefer objects whose metadata contains this term")
    parser.add_argument("--min-depth", type=float, default=0.2)
    parser.add_argument("--min-screen-margin", type=int, default=4)
    parser.add_argument("--image-dst", default=None, help="Optional destination for the selected RGB image")
    parser.add_argument("--multi-view-dir", default=None, help="Optional destination directory for multi-view RGB images")
    parser.add_argument("--scene-out", default=None, help="Optional scene JSON output path")
    parser.add_argument("--native-record-dir", default=None, help="Optional destination directory for native Infinigen metadata")
    args = parser.parse_args()

    image_dst = Path(args.image_dst) if args.image_dst else None
    multi_view_dir = Path(args.multi_view_dir) if args.multi_view_dir else None
    scene = adapt_scene(
        source_root=Path(args.source_root),
        scene_id=args.scene_id,
        split=args.split,
        max_objects=args.max_objects,
        query_terms=args.query_term,
        min_depth=args.min_depth,
        min_screen_margin=args.min_screen_margin,
        image_dst=image_dst,
        multi_view_dir=multi_view_dir,
    )

    if args.scene_out:
        out = Path(args.scene_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as f:
            json.dump(scene, f, indent=2)
    else:
        print(json.dumps(scene, indent=2))

    if args.native_record_dir:
        export_native_record(
            source_root=Path(args.source_root),
            scene_id=args.scene_id,
            dest_dir=Path(args.native_record_dir),
        )


if __name__ == "__main__":
    main()
