#!/usr/bin/env python3
"""在 Infinigen 客厅场景中布置大型中心桌，并渲染四个基准测试风格的视角。"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import bpy
from mathutils import Vector


DATA_GEN_LAYOUT = [
    {
        "id": "obj_0",
        "shape": "sphere",
        "material": "rubber",
        "color_name": "brown_rubber",
        "color": (0.65, 0.48, 0.26, 1.0),
        "coord": (-1.1400768756866455, 0.44635891914367676),
        "radius": 0.18,
        "metallic": 0.0,
        "roughness": 0.78,
    },
    {
        "id": "obj_1",
        "shape": "cylinder",
        "material": "metal",
        "color_name": "green_metal",
        "color": (0.22, 0.69, 0.29, 1.0),
        "coord": (1.7727065086364746, 2.4809017181396484),
        "radius": 0.16,
        "depth": 0.28,
        "metallic": 0.92,
        "roughness": 0.18,
    },
    {
        "id": "obj_2",
        "shape": "cylinder",
        "material": "metal",
        "color_name": "yellow_metal",
        "color": (0.91, 0.76, 0.18, 1.0),
        "coord": (1.8638617992401123, 0.5772521495819092),
        "radius": 0.16,
        "depth": 0.28,
        "metallic": 0.95,
        "roughness": 0.16,
    },
    {
        "id": "obj_3",
        "shape": "sphere",
        "material": "metal",
        "color_name": "bronze_metal",
        "color": (0.83, 0.57, 0.35, 1.0),
        "coord": (-2.9853742122650146, -2.9687623977661133),
        "radius": 0.16,
        "metallic": 0.96,
        "roughness": 0.2,
    },
]

STRUCTURAL_TOKENS = (
    "floor",
    "wall",
    "ceiling",
    "exterior",
    "window",
    "door",
    "skirting",
    "ceilinglight",
)

HIDE_NAME_TOKENS = ("desklampfactory", "simpledeskfactory", "sidetablefactory")


def _world_bbox(obj: bpy.types.Object) -> tuple[Vector, Vector]:
    corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    mins = Vector((min(c.x for c in corners), min(c.y for c in corners), min(c.z for c in corners)))
    maxs = Vector((max(c.x for c in corners), max(c.y for c in corners), max(c.z for c in corners)))
    return mins, maxs


def _hide_object(obj: bpy.types.Object) -> None:
    obj.hide_render = True
    obj.hide_set(True)


def _iter_meshes() -> Iterable[bpy.types.Object]:
    return [obj for obj in bpy.data.objects if obj.type == "MESH"]


def find_living_room_floor() -> bpy.types.Object:
    candidates = [obj for obj in _iter_meshes() if "living-room" in obj.name.lower() and ".floor" in obj.name.lower()]
    if not candidates:
        raise RuntimeError("Could not find living-room floor mesh in Infinigen scene")
    candidates.sort(key=lambda obj: (_world_bbox(obj)[1] - _world_bbox(obj)[0]).x * (_world_bbox(obj)[1] - _world_bbox(obj)[0]).y, reverse=True)
    return candidates[0]


def remove_existing_table_lamps() -> None:
    for obj in bpy.data.objects:
        lowered = obj.name.lower()
        if any(token in lowered for token in HIDE_NAME_TOKENS):
            _hide_object(obj)


def clear_table_zone(center_xy: Vector, radius: float = 1.4) -> None:
    for obj in _iter_meshes():
        lowered = obj.name.lower()
        if any(token in lowered for token in STRUCTURAL_TOKENS):
            continue
        mins, maxs = _world_bbox(obj)
        center = (mins + maxs) / 2.0
        dist = math.hypot(center.x - center_xy.x, center.y - center_xy.y)
        if dist < radius:
            _hide_object(obj)


def make_material(name: str, color, metallic: float, roughness: float) -> bpy.types.Material:
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf is not None:
        bsdf.inputs["Base Color"].default_value = color
        bsdf.inputs["Metallic"].default_value = metallic
        bsdf.inputs["Roughness"].default_value = roughness
    return mat


def add_large_center_table(room_center: Vector, floor_z: float) -> dict:
    table_width = 3.2
    table_depth = 2.1
    tabletop_thickness = 0.08
    table_height = 0.82
    leg_thickness = 0.09

    top_z = floor_z + table_height
    bpy.ops.mesh.primitive_cube_add(location=(room_center.x, room_center.y, top_z))
    top = bpy.context.object
    top.name = "CenterTableTop"
    top.scale = (table_width / 2.0, table_depth / 2.0, tabletop_thickness / 2.0)
    top.data.materials.append(
        make_material(
            "CenterTableTopMat",
            color=(0.36, 0.31, 0.25, 1.0),
            metallic=0.05,
            roughness=0.68,
        )
    )
    leg_positions = [
        (room_center.x - table_width / 2.0 + 0.14, room_center.y - table_depth / 2.0 + 0.14),
        (room_center.x + table_width / 2.0 - 0.14, room_center.y - table_depth / 2.0 + 0.14),
        (room_center.x - table_width / 2.0 + 0.14, room_center.y + table_depth / 2.0 - 0.14),
        (room_center.x + table_width / 2.0 - 0.14, room_center.y + table_depth / 2.0 - 0.14),
    ]
    for index, (x, y) in enumerate(leg_positions):
        bpy.ops.mesh.primitive_cube_add(location=(x, y, floor_z + table_height / 2.0))
        leg = bpy.context.object
        leg.name = f"CenterTableLeg_{index}"
        leg.scale = (leg_thickness / 2.0, leg_thickness / 2.0, table_height / 2.0)
        leg.data.materials.append(
            make_material(
                f"CenterTableLegMat_{index}",
                color=(0.05, 0.05, 0.06, 1.0),
                metallic=0.7,
                roughness=0.4,
            )
        )

    return {
        "top": top,
        "center": Vector((room_center.x, room_center.y, top_z + tabletop_thickness / 2.0)),
        "width": table_width,
        "depth": table_depth,
        "top_z": top_z + tabletop_thickness / 2.0,
    }


def add_probe_objects(table: dict) -> list[dict]:
    raw_points = [Vector((entry["coord"][0], entry["coord"][1], 0.0)) for entry in DATA_GEN_LAYOUT]
    centroid = sum(raw_points, Vector((0.0, 0.0, 0.0))) / len(raw_points)
    centered = [point - centroid for point in raw_points]
    rotated = [Vector((point.y, -point.x, 0.0)) for point in centered]
    range_x = max(point.x for point in rotated) - min(point.x for point in rotated)
    range_y = max(point.y for point in rotated) - min(point.y for point in rotated)
    usable_x = table["width"] * 0.58
    usable_y = table["depth"] * 0.58
    layout_scale = min(usable_x / max(range_x, 1e-6), usable_y / max(range_y, 1e-6))

    placed = []
    for spec, point in zip(DATA_GEN_LAYOUT, rotated):
        mat = make_material(
            name=f"{spec['id']}_mat",
            color=spec["color"],
            metallic=spec["metallic"],
            roughness=spec["roughness"],
        )
        location = Vector(
            (
                table["center"].x + point.x * layout_scale,
                table["center"].y + point.y * layout_scale,
                table["top_z"] + spec["radius"] * (1.0 if spec["shape"] != "cube" else 0.85),
            )
        )
        if spec["shape"] == "sphere":
            bpy.ops.mesh.primitive_uv_sphere_add(radius=spec["radius"], location=location)
        elif spec["shape"] == "cylinder":
            bpy.ops.mesh.primitive_cylinder_add(
                radius=spec["radius"] * 0.88,
                depth=spec.get("depth", spec["radius"] * 2.0),
                location=location,
            )
        else:
            raise ValueError(spec["shape"])
        obj = bpy.context.object
        obj.name = spec["id"]
        obj.data.materials.append(mat)
        placed.append(
            {
                "id": spec["id"],
                "shape": spec["shape"],
                "material": spec["material"],
                "color": spec["color_name"],
                "location": [round(location.x, 4), round(location.y, 4), round(location.z, 4)],
            }
        )
    return placed


def add_fill_lighting(room_center: Vector, top_z: float) -> None:
    for obj in list(bpy.data.objects):
        if obj.type == "LIGHT" and obj.name.startswith("CenterTableLight"):
            bpy.data.objects.remove(obj, do_unlink=True)

    bpy.ops.object.light_add(type="AREA", location=(room_center.x - 1.2, room_center.y - 1.0, top_z + 1.45))
    key = bpy.context.object
    key.name = "CenterTableLight_Key"
    key.data.energy = 1200
    key.data.shape = "RECTANGLE"
    key.data.size = 1.4
    key.data.size_y = 1.0

    bpy.ops.object.light_add(type="AREA", location=(room_center.x + 1.25, room_center.y + 1.1, top_z + 1.35))
    fill = bpy.context.object
    fill.name = "CenterTableLight_Fill"
    fill.data.energy = 650
    fill.data.shape = "RECTANGLE"
    fill.data.size = 1.6
    fill.data.size_y = 1.2


def ensure_render_settings(engine: str, samples: int, resolution_x: int, resolution_y: int) -> None:
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
            scene.cycles.device = "CPU"
    else:
        if hasattr(scene, "eevee"):
            scene.eevee.taa_render_samples = max(samples, 8)


def _camera_pose(look_at: Vector, azimuth_deg: float, horizontal_radius: float, z_height: float) -> Vector:
    azimuth = math.radians(azimuth_deg)
    return Vector(
        (
            look_at.x + horizontal_radius * math.cos(azimuth),
            look_at.y + horizontal_radius * math.sin(azimuth),
            z_height,
        )
    )


def render_multiview(output_dir: Path, look_at: Vector, engine: str, samples: int, resolution_x: int, resolution_y: int) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    ensure_render_settings(engine, samples, resolution_x, resolution_y)

    scene = bpy.context.scene
    camera = bpy.data.objects.get("CenterTableCamera")
    if camera is None or camera.type != "CAMERA":
        bpy.ops.object.camera_add()
        camera = bpy.context.object
        camera.name = "CenterTableCamera"
    camera.data.lens = 34
    camera.data.clip_start = 0.05
    camera.data.clip_end = 120
    scene.camera = camera

    views = []
    azimuths = [45.0, 135.0, 225.0, 315.0]
    horizontal_radius = 3.35
    z_height = look_at.z + 1.25
    for index, azimuth in enumerate(azimuths):
        camera.location = _camera_pose(look_at, azimuth, horizontal_radius, z_height)
        direction = look_at - camera.location
        camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
        image_path = output_dir / f"view_{index}.png"
        scene.render.filepath = str(image_path)
        bpy.ops.render.render(write_still=True)
        views.append(
            {
                "view_id": f"view_{index}",
                "azimuth": azimuth,
                "horizontal_radius": horizontal_radius,
                "camera_z": round(z_height, 4),
                "position": [round(v, 4) for v in camera.location],
                "look_at": [round(v, 4) for v in look_at],
                "image_path": str(image_path),
            }
        )
    return views


def main() -> None:
    parser = argparse.ArgumentParser(description="Render an Infinigen living room with a center table and four benchmark-like views")
    parser.add_argument("--scene-blend", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--save-blend", default=None)
    parser.add_argument("--engine", choices=("CYCLES", "BLENDER_EEVEE_NEXT", "BLENDER_EEVEE"), default="BLENDER_EEVEE_NEXT")
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--resolution-x", type=int, default=960)
    parser.add_argument("--resolution-y", type=int, default=640)
    args = parser.parse_args()

    bpy.ops.wm.open_mainfile(filepath=args.scene_blend)
    living_floor = find_living_room_floor()
    floor_mins, floor_maxs = _world_bbox(living_floor)
    room_center = (floor_mins + floor_maxs) / 2.0

    remove_existing_table_lamps()
    clear_table_zone(room_center, radius=1.45)
    table = add_large_center_table(room_center, floor_z=float(floor_maxs.z))
    placed_objects = add_probe_objects(table)
    add_fill_lighting(room_center, table["top_z"])

    output_dir = Path(args.output_dir)
    views = render_multiview(
        output_dir=output_dir,
        look_at=Vector((room_center.x, room_center.y, table["top_z"] + 0.05)),
        engine=args.engine,
        samples=args.samples,
        resolution_x=args.resolution_x,
        resolution_y=args.resolution_y,
    )
    metadata = {
        "scene_blend": args.scene_blend,
        "output_dir": str(output_dir),
        "room_center": [round(room_center.x, 4), round(room_center.y, 4), round(room_center.z, 4)],
        "table": {
            "width": table["width"],
            "depth": table["depth"],
            "top_z": round(table["top_z"], 4),
        },
        "objects": placed_objects,
        "views": views,
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    if args.save_blend:
        bpy.ops.wm.save_as_mainfile(filepath=args.save_blend)
    print(f"Rendered 4 views to {output_dir}")


if __name__ == "__main__":
    main()
