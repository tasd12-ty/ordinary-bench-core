#!/usr/bin/env python3
"""从现有的 data-gen 场景创建伪 Infinigen 帧包。"""

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image


def _bench_to_world(point: Sequence[float]) -> np.ndarray:
    x_b, y_b, z_b = point
    return np.asarray([x_b, -z_b, y_b], dtype=np.float64)


def _camera_pose_from_view(view: dict) -> np.ndarray:
    camera = view["camera"]
    position = _bench_to_world(camera["position"])
    look_at = _bench_to_world(camera["look_at"])
    forward = look_at - position
    forward = forward / np.linalg.norm(forward)

    # 伪 Infinigen 使用 CV 风格相机坐标系：+X 向右，+Y 向下，+Z 向前。
    world_down = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
    right = np.cross(world_down, forward)
    right = right / np.linalg.norm(right)
    down = np.cross(forward, right)
    down = down / np.linalg.norm(down)

    pose = np.eye(4, dtype=np.float64)
    pose[:3, 0] = right
    pose[:3, 1] = down
    pose[:3, 2] = forward
    pose[:3, 3] = position
    return pose


def _project_point(K: np.ndarray, pose_cv_world: np.ndarray, point_world: np.ndarray) -> Tuple[np.ndarray, float]:
    point_camera = np.linalg.inv(pose_cv_world)[:3, :] @ np.append(point_world, 1.0)
    depth = float(point_camera[2])
    if depth <= 0:
        return np.asarray([math.nan, math.nan], dtype=np.float64), depth
    uvw = K @ point_camera
    uv = uvw[:2] / uvw[2]
    return uv, depth


def _solve_intrinsics(view: dict, width: int, height: int) -> np.ndarray:
    pose = _camera_pose_from_view(view)
    rotation = pose[:3, :3]
    center = pose[:3, 3]
    u_rows: List[List[float]] = []
    v_rows: List[List[float]] = []
    u_targets: List[float] = []
    v_targets: List[float] = []

    for obj in view["objects"]:
        point_world = _bench_to_world(obj["3d_coords"])
        point_camera = rotation.T @ (point_world - center)
        depth = float(point_camera[2])
        if depth <= 1e-6:
            continue
        u_rows.append([float(point_camera[0] / depth), 1.0])
        v_rows.append([float(point_camera[1] / depth), 1.0])
        u_targets.append(float(obj["pixel_coords"][0]))
        v_targets.append(float(obj["pixel_coords"][1]))

    if len(u_rows) < 2 or len(v_rows) < 2:
        focal = float(max(width, height))
        return np.asarray([
            [focal, 0.0, width / 2.0],
            [0.0, focal, height / 2.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

    fx, cx = np.linalg.lstsq(np.asarray(u_rows), np.asarray(u_targets), rcond=None)[0]
    fy, cy = np.linalg.lstsq(np.asarray(v_rows), np.asarray(v_targets), rcond=None)[0]
    return np.asarray([
        [float(fx), 0.0, float(cx)],
        [0.0, float(fy), float(cy)],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)


def _object_name(obj: dict) -> str:
    parts = [obj.get("color", ""), obj.get("material", ""), obj.get("shape", "object")]
    return "".join(part.title() for part in parts if part) or "Object"


def _object_materials(obj: dict) -> List[str]:
    color = str(obj.get("color", "")).strip()
    material = str(obj.get("material", "")).strip()
    if color and material:
        return [f"{color}_{material}"]
    if material:
        return [material]
    return []


def _half_extents(obj: dict) -> np.ndarray:
    base = {
        "small": 0.35,
        "medium": 0.50,
        "large": 0.70,
    }.get(str(obj.get("size", "medium")), 0.50)
    shape = str(obj.get("shape", "object"))
    if shape == "sphere":
        return np.asarray([base, base, base], dtype=np.float64)
    if shape == "cylinder":
        return np.asarray([base * 0.55, base, base * 0.55], dtype=np.float64)
    if shape == "cube":
        return np.asarray([base * 0.75, base * 0.75, base * 0.75], dtype=np.float64)
    return np.asarray([base * 0.60, base * 0.60, base * 0.60], dtype=np.float64)


def _object_entry(index: int, obj: dict) -> dict:
    center_world = _bench_to_world(obj["3d_coords"])
    half_extents = _half_extents(obj)
    model_matrix = np.eye(4, dtype=np.float64)
    model_matrix[:3, 3] = center_world
    return {
        "object_index": index,
        "name": _object_name(obj),
        "tags": [obj.get("shape", "object")],
        "materials": _object_materials(obj),
        "min": (-half_extents).round(6).tolist(),
        "max": half_extents.round(6).tolist(),
        "model_matrices": [model_matrix.round(6).tolist()],
    }


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def _copy_images(scene_id: str, image_dir: Path, views: Sequence[dict], dest_root: Path) -> Tuple[int, int]:
    width = 0
    height = 0
    for index, view in enumerate(views):
        src = image_dir / scene_id / view["image_path"]
        dst = dest_root / "frames" / "Image" / f"camera_{index}" / f"Image_0_0_0001_{index}.png"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        if width == 0 or height == 0:
            with Image.open(src) as image:
                width, height = image.size
    return width, height


def _write_camviews(views: Sequence[dict], dest_root: Path, width: int, height: int) -> None:
    for index, view in enumerate(views):
        pose = _camera_pose_from_view(view)
        intrinsics = _solve_intrinsics(view, width, height)
        payload = {
            "K": intrinsics.round(6).tolist(),
            "T": pose.round(6).tolist(),
            "HW": [height, width],
        }
        _write_json(
            dest_root / "frames" / "camview" / f"camera_{index}" / f"camview_0001_00_{index:02d}.json",
            payload,
        )


def _write_objects(scene: dict, views: Sequence[dict], dest_root: Path) -> None:
    payload = [_object_entry(index + 1, obj) for index, obj in enumerate(scene["objects"])]
    for index, _view in enumerate(views):
        _write_json(
            dest_root / "frames" / "Objects" / f"camera_{index}" / f"Objects_0_0_0001_{index}.json",
            payload,
        )


def bootstrap_scene(data_root: Path, scene_id: str, dest_root: Path, n_views: int) -> Dict[str, int]:
    scene_path = data_root / "scenes" / f"{scene_id}.json"
    if not scene_path.exists():
        raise FileNotFoundError(f"Scene JSON not found: {scene_path}")

    scene = json.loads(scene_path.read_text())
    views = list(scene.get("views", []))[:n_views]
    if not views:
        raise RuntimeError(f"Scene {scene_id} does not contain any views")

    image_dir = data_root / "images" / "multi_view"
    width, height = _copy_images(scene_id, image_dir, views, dest_root)
    _write_camviews(views, dest_root, width, height)
    _write_objects(scene, views, dest_root)

    manifest = {
        "scene_id": scene_id,
        "source_scene_json": str(scene_path),
        "source_multi_view_dir": str(image_dir / scene_id),
        "n_views": len(views),
        "image_size": [width, height],
    }
    _write_json(dest_root / "manifest.json", manifest)
    return {
        "n_views": len(views),
        "width": width,
        "height": height,
        "n_objects": len(scene.get("objects", [])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap a pseudo-Infinigen scene from data-gen output")
    parser.add_argument("--data-root", default="/Users/tsyq/code/ordinary-bench-core/data-gen/output")
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--dest-root", required=True)
    parser.add_argument("--n-views", type=int, default=4)
    args = parser.parse_args()

    summary = bootstrap_scene(
        data_root=Path(args.data_root),
        scene_id=args.scene_id,
        dest_root=Path(args.dest_root),
        n_views=args.n_views,
    )
    print(json.dumps({
        "scene_id": args.scene_id,
        "dest_root": str(Path(args.dest_root).resolve()),
        **summary,
    }, indent=2))


if __name__ == "__main__":
    main()
