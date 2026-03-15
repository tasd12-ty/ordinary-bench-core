#!/usr/bin/env python
"""Auto-generated Blender render script. Run with:
    blender --background --python render_scene.py
"""
import os
import sys
import math

import bpy
from mathutils import Vector

# ── Configuration ──────────────────────────────────────────────────────────

ASSETS_DIR = '/Users/tsyq/code/ordinary-bench/data-gen/blender/assets'
BASE_SCENE = os.path.join(ASSETS_DIR, "base_scene_v5.blend")
SHAPE_DIR = os.path.join(ASSETS_DIR, "shapes_v5")
MATERIAL_DIR = os.path.join(ASSETS_DIR, "materials_v5")

OUTPUT_IMAGE = 'output/analysis/blender_scripts/n04_000004_gt.png'
RESOLUTION_X = 480
RESOLUTION_Y = 320
SAMPLES = 256

CAMERA_AZIMUTH = 45.0
CAMERA_ELEVATION = 30.0
CAMERA_DISTANCE = 12.0

OBJECTS = {
    'obj_0': {'shape': 'SmoothCube_v2', 'material': 'Rubber', 'rgba': [1.0, 0.9333333333333333, 0.2, 1.0], 'scale': 0.4949747468305832, 'pos': (2.817045211791992, 2.8147425651550293)},
    'obj_1': {'shape': 'SmoothCylinder', 'material': 'Rubber', 'rgba': [0.5058823529411764, 0.2901960784313726, 0.09803921568627451, 1.0], 'scale': 0.7, 'pos': (1.0834777355194092, -0.411224901676178)},
    'obj_2': {'shape': 'SmoothCylinder', 'material': 'Rubber', 'rgba': [0.16470588235294117, 0.29411764705882354, 0.8431372549019608, 1.0], 'scale': 0.35, 'pos': (-2.827517032623291, 1.998293161392212)},
    'obj_3': {'shape': 'Sphere', 'material': 'Rubber', 'rgba': [0.1607843137254902, 0.8156862745098039, 0.8156862745098039, 1.0], 'scale': 0.35, 'pos': (-2.7478151321411133, -0.42710182070732117)},
}

# ── Blender version helpers ────────────────────────────────────────────────

BLENDER_VERSION = bpy.app.version
IS_280_PLUS = BLENDER_VERSION >= (2, 80, 0)


def delete_object(obj):
    for o in bpy.data.objects:
        o.select_set(False)
    obj.select_set(True)
    bpy.ops.object.delete()


# ── Asset loading ──────────────────────────────────────────────────────────

def load_materials():
    for fn in os.listdir(MATERIAL_DIR):
        if not fn.endswith(".blend"):
            continue
        name = os.path.splitext(fn)[0]
        blend_path = os.path.join(MATERIAL_DIR, fn)
        bpy.ops.wm.append(
            filepath=os.path.join(blend_path, "NodeTree", name),
            directory=os.path.join(blend_path, "NodeTree"),
            filename=name,
        )


def add_object(shape_name, scale, loc_xy, theta=0.0):
    count = sum(1 for o in bpy.data.objects if o.name.startswith(shape_name))
    blend_path = os.path.join(SHAPE_DIR, shape_name + ".blend")
    if not os.path.isfile(blend_path):
        print(f"WARNING: shape file not found: {blend_path}")
        return None
    bpy.ops.wm.append(
        filepath=os.path.join(blend_path, "Object", shape_name),
        directory=os.path.join(blend_path, "Object"),
        filename=shape_name,
    )
    new_name = f"{shape_name}_{count}"
    bpy.data.objects[shape_name].name = new_name
    obj = bpy.data.objects[new_name]
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    obj.rotation_euler[2] = theta
    bpy.ops.transform.resize(value=(scale, scale, scale))
    x, y = loc_xy
    bpy.ops.transform.translate(value=(x, y, 0))
    return obj


def add_material(obj, mat_group_name, rgba):
    mat_count = len(bpy.data.materials)
    mat = bpy.data.materials.new(name=f"Material_{mat_count}")
    mat.use_nodes = True
    assert len(obj.data.materials) == 0
    obj.data.materials.append(mat)
    output_node = None
    for n in mat.node_tree.nodes:
        if n.type == "OUTPUT_MATERIAL":
            output_node = n
            break
    if output_node is None:
        raise RuntimeError("No Material Output node found")
    group_node = mat.node_tree.nodes.new("ShaderNodeGroup")
    group_node.node_tree = bpy.data.node_groups[mat_group_name]
    for inp in group_node.inputs:
        if inp.name == "Color":
            inp.default_value = rgba
    mat.node_tree.links.new(
        group_node.outputs["Shader"],
        output_node.inputs["Surface"],
    )


# ── Scene setup ────────────────────────────────────────────────────────────

def clear_scene_objects():
    """Delete existing scene objects, keeping Camera and Lights."""
    keep_prefixes = ("Camera", "Lamp", "Light", "Ground", "ground")
    to_delete = []
    for obj in bpy.data.objects:
        if not any(obj.name.startswith(p) for p in keep_prefixes):
            to_delete.append(obj)
    for obj in to_delete:
        delete_object(obj)


def setup_camera():
    camera = bpy.data.objects["Camera"]
    az_rad = math.radians(CAMERA_AZIMUTH)
    el_rad = math.radians(CAMERA_ELEVATION)
    x = CAMERA_DISTANCE * math.cos(el_rad) * math.cos(az_rad)
    y = CAMERA_DISTANCE * math.cos(el_rad) * math.sin(az_rad)
    z = CAMERA_DISTANCE * math.sin(el_rad)
    camera.location = (x, y, z)
    direction = Vector((0, 0, 0)) - Vector((x, y, z))
    rot_quat = direction.to_track_quat("-Z", "Y")
    camera.rotation_euler = rot_quat.to_euler()


def setup_render():
    render = bpy.context.scene.render
    render.engine = "CYCLES"
    render.resolution_x = RESOLUTION_X
    render.resolution_y = RESOLUTION_Y
    render.resolution_percentage = 100
    bpy.context.scene.cycles.samples = SAMPLES
    bpy.context.scene.cycles.blur_glossy = 2.0
    # Try to enable GPU if available
    try:
        cycles_prefs = bpy.context.preferences.addons["cycles"].preferences
        for compute_type in ("CUDA", "OPTIX", "HIP", "ONEAPI", "METAL"):
            try:
                cycles_prefs.compute_device_type = compute_type
                for device in cycles_prefs.devices:
                    device.use = True
                bpy.context.scene.cycles.device = "GPU"
                break
            except Exception:
                continue
    except Exception:
        pass


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    if not os.path.isfile(BASE_SCENE):
        print(f"ERROR: base scene not found: {BASE_SCENE}")
        sys.exit(1)

    bpy.ops.wm.open_mainfile(filepath=BASE_SCENE)
    load_materials()
    clear_scene_objects()

    for obj_id, info in OBJECTS.items():
        obj = add_object(info["shape"], info["scale"], info["pos"])
        if obj is not None:
            bpy.context.view_layer.objects.active = obj
            obj.select_set(True)
            add_material(obj, info["material"], info["rgba"])

    setup_camera()
    setup_render()

    os.makedirs(os.path.dirname(os.path.abspath(OUTPUT_IMAGE)) or ".", exist_ok=True)
    bpy.context.scene.render.filepath = os.path.abspath(OUTPUT_IMAGE)
    bpy.ops.render.render(write_still=True)
    print(f"Rendered: {OUTPUT_IMAGE}")


main()
