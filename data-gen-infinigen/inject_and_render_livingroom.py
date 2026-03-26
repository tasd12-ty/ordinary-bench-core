#!/usr/bin/env python3
"""打开 Infinigen 室内场景，在桌面上放置几何体探针对象，并渲染一张图像。"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable, Optional

import bpy
from mathutils import Vector


TABLETOP_SPECS = [
    {"shape": "sphere", "color": (0.84, 0.18, 0.18, 1.0), "metallic": 0.0, "roughness": 0.65, "uv_offset": (-0.34, -0.10), "radius": 0.085},
    {"shape": "cube", "color": (0.20, 0.37, 0.86, 1.0), "metallic": 0.8, "roughness": 0.18, "uv_offset": (-0.12, 0.10), "radius": 0.075},
    {"shape": "cylinder", "color": (0.98, 0.86, 0.16, 1.0), "metallic": 0.85, "roughness": 0.20, "uv_offset": (0.08, -0.08), "radius": 0.070},
    {"shape": "sphere", "color": (0.15, 0.58, 0.24, 1.0), "metallic": 0.9, "roughness": 0.12, "uv_offset": (0.24, 0.08), "radius": 0.070},
]


def _world_bbox(obj: bpy.types.Object) -> tuple[Vector, Vector]:
    corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    mins = Vector((min(c.x for c in corners), min(c.y for c in corners), min(c.z for c in corners)))
    maxs = Vector((max(c.x for c in corners), max(c.y for c in corners), max(c.z for c in corners)))
    return mins, maxs


def _candidate_table_score(obj: bpy.types.Object) -> tuple[int, float, float]:
    mins, maxs = _world_bbox(obj)
    extents = maxs - mins
    area = float(extents.x * extents.y)
    center = (mins + maxs) / 2.0
    name = obj.name.lower()
    name_bonus = 3 if "table" in name else 2 if "desk" in name else 0
    return (name_bonus, area, -abs(center.x) - abs(center.y))


def find_table_candidate(target_substring: Optional[str] = None) -> bpy.types.Object:
    mesh_objects = [obj for obj in bpy.data.objects if obj.type == "MESH" and obj.visible_get()]
    if target_substring:
        target_substring = target_substring.lower()
        targeted = [obj for obj in mesh_objects if target_substring in obj.name.lower()]
        if targeted:
            targeted.sort(key=_candidate_table_score, reverse=True)
            return targeted[0]
    named = sorted(
        [
            obj
            for obj in mesh_objects
            if any(token in obj.name.lower() for token in ("table", "desk"))
            and "lamp" not in obj.name.lower()
        ],
        key=_candidate_table_score,
        reverse=True,
    )
    if named:
        return named[0]

    plausible = []
    for obj in mesh_objects:
        mins, maxs = _world_bbox(obj)
        extents = maxs - mins
        center = (mins + maxs) / 2.0
        if 0.35 <= maxs.z <= 1.35 and extents.x >= 0.6 and extents.y >= 0.4 and extents.z <= 1.4:
            plausible.append(obj)
    if not plausible:
        raise RuntimeError("Could not find a plausible tabletop object in the Infinigen scene")
    plausible.sort(key=_candidate_table_score, reverse=True)
    return plausible[0]


def create_material(name: str, color, metallic: float, roughness: float) -> bpy.types.Material:
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf is not None:
        bsdf.inputs["Base Color"].default_value = color
        bsdf.inputs["Metallic"].default_value = metallic
        bsdf.inputs["Roughness"].default_value = roughness
    return mat


def add_probe_object(shape: str, location: Vector, radius: float, rotation_z: float, material: bpy.types.Material) -> bpy.types.Object:
    if shape == "sphere":
        bpy.ops.mesh.primitive_uv_sphere_add(radius=radius, location=location)
    elif shape == "cube":
        bpy.ops.mesh.primitive_cube_add(size=radius * 2.0, location=location)
    elif shape == "cylinder":
        bpy.ops.mesh.primitive_cylinder_add(radius=radius * 0.8, depth=radius * 2.4, location=location)
    else:
        raise ValueError(shape)
    obj = bpy.context.object
    obj.rotation_euler[2] = rotation_z
    if obj.data.materials:
        obj.data.materials[0] = material
    else:
        obj.data.materials.append(material)
    return obj


def add_objects_on_table(table: bpy.types.Object) -> list[bpy.types.Object]:
    mins, maxs = _world_bbox(table)
    tabletop_z = float(maxs.z)
    center = (mins + maxs) / 2.0
    usable_x = max((maxs.x - mins.x) * 0.72, 0.26)
    usable_y = max((maxs.y - mins.y) * 0.55, 0.18)
    placed = []
    for idx, spec in enumerate(TABLETOP_SPECS):
        mat = create_material(
            name=f"ProbeMat_{idx}",
            color=spec["color"],
            metallic=spec["metallic"],
            roughness=spec["roughness"],
        )
        x = center.x + spec["uv_offset"][0] * usable_x
        y = center.y + spec["uv_offset"][1] * usable_y
        z = tabletop_z + spec["radius"] * (1.0 if spec["shape"] != "cube" else 0.85)
        placed.append(
            add_probe_object(
                shape=spec["shape"],
                location=Vector((x, y, z)),
                radius=spec["radius"],
                rotation_z=0.45 if spec["shape"] == "cube" else 0.0,
                material=mat,
            )
        )
    return placed


def _iter_light_objects() -> Iterable[bpy.types.Object]:
    return [obj for obj in bpy.data.objects if obj.type == "LIGHT"]


def _scene_center_xy() -> Vector:
    mesh_objects = [obj for obj in bpy.data.objects if obj.type == "MESH" and obj.visible_get()]
    mins = Vector((math.inf, math.inf, math.inf))
    maxs = Vector((-math.inf, -math.inf, -math.inf))
    for obj in mesh_objects:
        obj_mins, obj_maxs = _world_bbox(obj)
        mins = Vector((min(mins.x, obj_mins.x), min(mins.y, obj_mins.y), min(mins.z, obj_mins.z)))
        maxs = Vector((max(maxs.x, obj_maxs.x), max(maxs.y, obj_maxs.y), max(maxs.z, obj_maxs.z)))
    center = (mins + maxs) / 2.0
    return Vector((center.x, center.y, 0.0))


def ensure_lighting(table_center: Vector) -> None:
    if list(_iter_light_objects()):
        return
    bpy.ops.object.light_add(type="AREA", location=(table_center.x + 2.5, table_center.y - 2.8, table_center.z + 2.6))
    key = bpy.context.object
    key.data.energy = 4500
    key.data.size = 2.2
    bpy.ops.object.light_add(type="AREA", location=(table_center.x - 2.0, table_center.y + 2.2, table_center.z + 1.8))
    fill = bpy.context.object
    fill.data.energy = 1800
    fill.data.size = 3.0
    bpy.ops.object.light_add(type="SUN", location=(0, 0, 6))
    sun = bpy.context.object
    sun.data.energy = 1.2


def ensure_camera(table: bpy.types.Object) -> bpy.types.Object:
    mins, maxs = _world_bbox(table)
    center = (mins + maxs) / 2.0
    look_at = Vector((center.x, center.y, mins.z + (maxs.z - mins.z) * 0.55))
    room_center = _scene_center_xy()
    toward_room = room_center - Vector((center.x, center.y, 0.0))
    if toward_room.length < 1e-4:
        toward_room = Vector((-1.0, 1.0, 0.0))
    toward_room.normalize()
    side = Vector((-toward_room.y, toward_room.x, 0.0))
    camera = bpy.data.objects.get("Camera")
    if camera is None or camera.type != "CAMERA":
        bpy.ops.object.camera_add()
        camera = bpy.context.object
        camera.name = "Camera"
    camera.location = Vector(
        (
            center.x + toward_room.x * 3.2 + side.x * 0.9,
            center.y + toward_room.y * 3.2 + side.y * 0.9,
            look_at.z + 1.55,
        )
    )
    direction = look_at - camera.location
    camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    camera.data.lens = 38
    camera.data.clip_start = 0.05
    camera.data.clip_end = 200
    bpy.context.scene.camera = camera
    ensure_lighting(center)
    return camera


def render_scene(
    output_image: Path,
    resolution_x: int = 1400,
    resolution_y: int = 900,
    samples: int = 96,
    engine: str = "CYCLES",
) -> None:
    scene = bpy.context.scene
    scene.render.engine = engine
    scene.render.resolution_x = resolution_x
    scene.render.resolution_y = resolution_y
    scene.render.resolution_percentage = 100
    if engine == "CYCLES":
        scene.cycles.samples = samples
        scene.cycles.use_denoising = True
        try:
            prefs = bpy.context.preferences.addons["cycles"].preferences
            for compute_type in ("METAL", "CUDA", "OPTIX", "HIP", "ONEAPI"):
                try:
                    prefs.compute_device_type = compute_type
                    prefs.get_devices()
                    for device in prefs.devices:
                        device.use = True
                    scene.cycles.device = "GPU"
                    break
                except Exception:
                    continue
        except Exception:
            pass
    else:
        if hasattr(scene, "eevee"):
            scene.eevee.taa_render_samples = max(samples, 8)
    output_image.parent.mkdir(parents=True, exist_ok=True)
    scene.render.filepath = str(output_image)
    bpy.ops.render.render(write_still=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inject tabletop probes into an Infinigen living room and render")
    parser.add_argument("--scene-blend", required=True)
    parser.add_argument("--output-image", required=True)
    parser.add_argument("--save-blend", default=None, help="Optional path to save the modified blend file")
    parser.add_argument("--target-name", default=None, help="Optional object-name substring to target a specific table/desk")
    parser.add_argument("--engine", choices=("CYCLES", "BLENDER_EEVEE_NEXT", "BLENDER_EEVEE"), default="CYCLES")
    parser.add_argument("--samples", type=int, default=96)
    parser.add_argument("--resolution-x", type=int, default=1400)
    parser.add_argument("--resolution-y", type=int, default=900)
    args = parser.parse_args()

    bpy.ops.wm.open_mainfile(filepath=args.scene_blend)
    table = find_table_candidate(args.target_name)
    add_objects_on_table(table)
    ensure_camera(table)
    render_scene(
        Path(args.output_image),
        resolution_x=args.resolution_x,
        resolution_y=args.resolution_y,
        samples=args.samples,
        engine=args.engine,
    )
    if args.save_blend:
        bpy.ops.wm.save_as_mainfile(filepath=args.save_blend)
    print(f"Rendered to {args.output_image}")


if __name__ == "__main__":
    main()
