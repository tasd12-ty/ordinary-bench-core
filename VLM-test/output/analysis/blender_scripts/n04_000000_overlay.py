#!/usr/bin/env python
"""Auto-generated Blender overlay render script. Run with:
    blender --background --python n04_000000_overlay.py

Renders GT objects (solid) and reconstructed objects (transparent) in the
same scene, with red displacement lines connecting corresponding pairs.
"""
import os
import sys
import math

import bpy
import bmesh
from mathutils import Vector

# ── Configuration ──────────────────────────────────────────────────────────

ASSETS_DIR = '/Users/tsyq/code/ordinary-bench/data-gen/blender/assets'
BASE_SCENE = os.path.join(ASSETS_DIR, "base_scene_v5.blend")
SHAPE_DIR = os.path.join(ASSETS_DIR, "shapes_v5")
MATERIAL_DIR = os.path.join(ASSETS_DIR, "materials_v5")

OUTPUT_IMAGE = 'output/analysis/blender_scripts/n04_000000_overlay.png'
RESOLUTION_X = 480
RESOLUTION_Y = 320
SAMPLES = 256

CAMERA_AZIMUTH = 45.0
CAMERA_ELEVATION = 30.0
CAMERA_DISTANCE = 12.0

GT_OBJECTS = {
    'obj_0': {'shape': 'Sphere', 'material': 'Rubber', 'rgba': [0.5058823529411764, 0.2901960784313726, 0.09803921568627451, 1.0], 'scale': 0.7, 'pos': (-1.1400768756866455, 0.44635891914367676)},
    'obj_1': {'shape': 'SmoothCylinder', 'material': 'MyMetal', 'rgba': [0.11372549019607843, 0.4117647058823529, 0.0784313725490196, 1.0], 'scale': 0.7, 'pos': (1.7727065086364746, 2.4809017181396484)},
    'obj_2': {'shape': 'SmoothCylinder', 'material': 'MyMetal', 'rgba': [1.0, 0.9333333333333333, 0.2, 1.0], 'scale': 0.7, 'pos': (1.8638617992401123, 0.5772521495819092)},
    'obj_3': {'shape': 'Sphere', 'material': 'MyMetal', 'rgba': [0.5058823529411764, 0.2901960784313726, 0.09803921568627451, 1.0], 'scale': 0.7, 'pos': (-2.9853742122650146, -2.9687623977661133)},
}

RECON_OBJECTS = {
    'obj_0': {'shape': 'Sphere', 'material': 'Rubber', 'rgba': [0.5058823529411764, 0.2901960784313726, 0.09803921568627451, 1.0], 'scale': 0.7, 'pos': (0.6031079718021527, 0.9313677768199699)},
    'obj_1': {'shape': 'SmoothCylinder', 'material': 'MyMetal', 'rgba': [0.11372549019607843, 0.4117647058823529, 0.0784313725490196, 1.0], 'scale': 0.7, 'pos': (1.1871908289943234, 1.5067895098714876)},
    'obj_2': {'shape': 'SmoothCylinder', 'material': 'MyMetal', 'rgba': [1.0, 0.9333333333333333, 0.2, 1.0], 'scale': 0.7, 'pos': (1.2424302686218045, 1.4637088306433366)},
    'obj_3': {'shape': 'Sphere', 'material': 'MyMetal', 'rgba': [0.5058823529411764, 0.2901960784313726, 0.09803921568627451, 1.0], 'scale': 0.7, 'pos': (-3.521611849493354, -3.366115728235673)},
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


def add_transparent_material(obj, rgba, alpha=0.3):
    """Add a semi-transparent material using Principled BSDF for recon objects."""
    mat_count = len(bpy.data.materials)
    mat = bpy.data.materials.new(name=f"TransparentMat_{mat_count}")
    mat.use_nodes = True
    mat.blend_method = "BLEND" if hasattr(mat, "blend_method") else None

    # Remove default nodes
    for node in mat.node_tree.nodes:
        mat.node_tree.nodes.remove(node)

    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    # Output node
    output_node = nodes.new("ShaderNodeOutputMaterial")
    output_node.location = (400, 0)

    # Mix between transparent and the colored shader
    mix_node = nodes.new("ShaderNodeMixShader")
    mix_node.location = (200, 0)
    mix_node.inputs[0].default_value = 1.0 - alpha  # Fac: higher = more of second input

    # Transparent BSDF
    transparent_node = nodes.new("ShaderNodeBsdfTransparent")
    transparent_node.location = (0, 100)

    # Principled BSDF for colored portion
    principled_node = nodes.new("ShaderNodeBsdfPrincipled")
    principled_node.location = (0, -100)
    principled_node.inputs["Base Color"].default_value = rgba
    # Make it slightly emissive so it's visible even when transparent
    if "Emission Color" in [inp.name for inp in principled_node.inputs]:
        principled_node.inputs["Emission Color"].default_value = rgba
        principled_node.inputs["Emission Strength"].default_value = 0.3

    links.new(transparent_node.outputs["BSDF"], mix_node.inputs[1])
    links.new(principled_node.outputs["BSDF"], mix_node.inputs[2])
    links.new(mix_node.outputs["Shader"], output_node.inputs["Surface"])

    if len(obj.data.materials) == 0:
        obj.data.materials.append(mat)
    else:
        obj.data.materials[0] = mat


def create_displacement_line(start_xy, end_xy, radius=0.02):
    """Create a thin red cylinder between two ground-plane points."""
    sx, sy = start_xy
    ex, ey = end_xy
    dx, dy = ex - sx, ey - sy
    length = math.sqrt(dx * dx + dy * dy)
    if length < 1e-6:
        return None

    mid_x = (sx + ex) / 2.0
    mid_y = (sy + ey) / 2.0
    mid_z = 0.15  # slightly above ground so it's visible

    bpy.ops.mesh.primitive_cylinder_add(
        radius=radius,
        depth=length,
        location=(mid_x, mid_y, mid_z),
    )
    line_obj = bpy.context.object
    line_obj.name = "DisplacementLine"

    # Rotate to point from start to end
    angle = math.atan2(dy, dx)
    line_obj.rotation_euler = (math.radians(90), 0, angle)

    # Red material
    mat = bpy.data.materials.new(name="RedLineMat")
    mat.use_nodes = True
    for node in mat.node_tree.nodes:
        mat.node_tree.nodes.remove(node)

    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    output_node = nodes.new("ShaderNodeOutputMaterial")
    output_node.location = (200, 0)
    emission_node = nodes.new("ShaderNodeEmission")
    emission_node.location = (0, 0)
    emission_node.inputs["Color"].default_value = (1.0, 0.0, 0.0, 1.0)
    emission_node.inputs["Strength"].default_value = 3.0
    links.new(emission_node.outputs["Emission"], output_node.inputs["Surface"])

    line_obj.data.materials.append(mat)
    return line_obj


# ── Scene setup ────────────────────────────────────────────────────────────

def clear_scene_objects():
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
    render.film_transparent = False
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

    # Add GT objects (solid)
    for obj_id, info in GT_OBJECTS.items():
        obj = add_object(info["shape"], info["scale"], info["pos"])
        if obj is not None:
            bpy.context.view_layer.objects.active = obj
            obj.select_set(True)
            add_material(obj, info["material"], info["rgba"])

    # Add recon objects (transparent)
    for obj_id, info in RECON_OBJECTS.items():
        obj = add_object(info["shape"], info["scale"], info["pos"])
        if obj is not None:
            add_transparent_material(obj, info["rgba"], alpha=0.3)

    # Add displacement lines
    for obj_id in GT_OBJECTS:
        if obj_id in RECON_OBJECTS:
            gt_pos = GT_OBJECTS[obj_id]["pos"]
            rc_pos = RECON_OBJECTS[obj_id]["pos"]
            create_displacement_line(gt_pos, rc_pos)

    setup_camera()
    setup_render()

    os.makedirs(os.path.dirname(os.path.abspath(OUTPUT_IMAGE)) or ".", exist_ok=True)
    bpy.context.scene.render.filepath = os.path.abspath(OUTPUT_IMAGE)
    bpy.ops.render.render(write_still=True)
    print(f"Rendered overlay: {OUTPUT_IMAGE}")


main()
