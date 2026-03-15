"""
Generate standalone Blender Python scripts for rendering reconstructed scenes.

Produces .py scripts that can be executed via:
    blender --background --python <script.py>

Three rendering modes:
  1. Single scene (GT or recon) with generate_blender_script()
  2. Side-by-side GT vs recon pair with generate_comparison_script()
  3. Overlay (GT solid + recon transparent + displacement lines) with generate_overlay_script()

The generated scripts are fully standalone -- they only use bpy and standard
library imports available inside Blender's Python interpreter.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import textwrap
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SHAPE_MAP = {
    "sphere": "Sphere",
    "cube": "SmoothCube_v2",
    "cylinder": "SmoothCylinder",
}

SIZE_MAP = {
    "large": 0.7,
    "small": 0.35,
}

MATERIAL_MAP = {
    "rubber": "Rubber",
    "metal": "MyMetal",
}

COLOR_MAP = {
    "gray": [87, 87, 87],
    "red": [173, 35, 35],
    "blue": [42, 75, 215],
    "green": [29, 105, 20],
    "brown": [129, 74, 25],
    "purple": [129, 38, 192],
    "cyan": [41, 208, 208],
    "yellow": [255, 238, 51],
}

_DEFAULT_ASSETS_DIR = str(
    Path(__file__).resolve().parent.parent.parent / "data-gen" / "blender" / "assets"
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _rgba_from_color_name(name: str) -> List[float]:
    """Return [R, G, B, A] in 0-1 range for a named color."""
    rgb = COLOR_MAP.get(name, [128, 128, 128])
    return [c / 255.0 for c in rgb] + [1.0]


def _build_object_defs(
    positions: Dict[str, list],
    object_info: Dict[str, dict],
) -> str:
    """Build a Python literal for object definitions to embed in a script."""
    entries = []
    for obj_id in sorted(positions.keys()):
        info = object_info.get(obj_id, {})
        shape = info.get("shape", "sphere")
        color = info.get("color", "gray")
        material = info.get("material", "rubber")
        size = info.get("size", "large")
        pos = positions[obj_id]
        x, y = float(pos[0]), float(pos[1])

        blend_shape = SHAPE_MAP.get(shape, "Sphere")
        blend_mat = MATERIAL_MAP.get(material, "Rubber")
        rgba = _rgba_from_color_name(color)

        scale = SIZE_MAP.get(size, 0.7)
        # Cubes use size / sqrt(2) as per the existing renderer
        if shape == "cube":
            scale /= math.sqrt(2)

        entries.append(
            f"    {obj_id!r}: {{"
            f"'shape': {blend_shape!r}, "
            f"'material': {blend_mat!r}, "
            f"'rgba': {rgba!r}, "
            f"'scale': {scale!r}, "
            f"'pos': ({x!r}, {y!r})"
            f"}},"
        )
    return "{\n" + "\n".join(entries) + "\n}"


# ---------------------------------------------------------------------------
# Core script template
# ---------------------------------------------------------------------------

_SCENE_SCRIPT_TEMPLATE = textwrap.dedent(r'''
#!/usr/bin/env python
"""Auto-generated Blender render script. Run with:
    blender --background --python {script_name}
"""
import os
import sys
import math

import bpy
from mathutils import Vector

# ── Configuration ──────────────────────────────────────────────────────────

ASSETS_DIR = {assets_dir!r}
BASE_SCENE = os.path.join(ASSETS_DIR, "base_scene_v5.blend")
SHAPE_DIR = os.path.join(ASSETS_DIR, "shapes_v5")
MATERIAL_DIR = os.path.join(ASSETS_DIR, "materials_v5")

OUTPUT_IMAGE = {output_image!r}
RESOLUTION_X = {res_x}
RESOLUTION_Y = {res_y}
SAMPLES = {samples}

CAMERA_AZIMUTH = {cam_azimuth}
CAMERA_ELEVATION = {cam_elevation}
CAMERA_DISTANCE = {cam_distance}

OBJECTS = {object_defs}

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
        print(f"WARNING: shape file not found: {{blend_path}}")
        return None
    bpy.ops.wm.append(
        filepath=os.path.join(blend_path, "Object", shape_name),
        directory=os.path.join(blend_path, "Object"),
        filename=shape_name,
    )
    new_name = f"{{shape_name}}_{{count}}"
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
    mat = bpy.data.materials.new(name=f"Material_{{mat_count}}")
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
        print(f"ERROR: base scene not found: {{BASE_SCENE}}")
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
    print(f"Rendered: {{OUTPUT_IMAGE}}")


main()
''').lstrip()

# ---------------------------------------------------------------------------
# Overlay script template (GT solid + recon transparent + displacement lines)
# ---------------------------------------------------------------------------

_OVERLAY_SCRIPT_TEMPLATE = textwrap.dedent(r'''
#!/usr/bin/env python
"""Auto-generated Blender overlay render script. Run with:
    blender --background --python {script_name}

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

ASSETS_DIR = {assets_dir!r}
BASE_SCENE = os.path.join(ASSETS_DIR, "base_scene_v5.blend")
SHAPE_DIR = os.path.join(ASSETS_DIR, "shapes_v5")
MATERIAL_DIR = os.path.join(ASSETS_DIR, "materials_v5")

OUTPUT_IMAGE = {output_image!r}
RESOLUTION_X = {res_x}
RESOLUTION_Y = {res_y}
SAMPLES = {samples}

CAMERA_AZIMUTH = {cam_azimuth}
CAMERA_ELEVATION = {cam_elevation}
CAMERA_DISTANCE = {cam_distance}

GT_OBJECTS = {gt_object_defs}

RECON_OBJECTS = {recon_object_defs}

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
        print(f"WARNING: shape file not found: {{blend_path}}")
        return None
    bpy.ops.wm.append(
        filepath=os.path.join(blend_path, "Object", shape_name),
        directory=os.path.join(blend_path, "Object"),
        filename=shape_name,
    )
    new_name = f"{{shape_name}}_{{count}}"
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
    mat = bpy.data.materials.new(name=f"Material_{{mat_count}}")
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
    mat = bpy.data.materials.new(name=f"TransparentMat_{{mat_count}}")
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
        print(f"ERROR: base scene not found: {{BASE_SCENE}}")
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
    print(f"Rendered overlay: {{OUTPUT_IMAGE}}")


main()
''').lstrip()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_blender_script(
    positions: Dict[str, list],
    object_info: Dict[str, dict],
    output_image_path: str,
    assets_dir: str = _DEFAULT_ASSETS_DIR,
    camera_azimuth: float = 45.0,
    camera_elevation: float = 30.0,
    camera_distance: float = 12.0,
    resolution: tuple = (480, 320),
    samples: int = 256,
) -> str:
    """Generate a standalone Blender Python script to render a scene.

    Args:
        positions: Object positions ``{obj_id: [x, y]}`` (2-D, placed at z=0).
        object_info: Per-object metadata
            ``{obj_id: {"shape", "color", "material", "size"}}``.
        output_image_path: File path for the rendered image.
        assets_dir: Root directory of Blender assets
            (``data-gen/blender/assets/``).
        camera_azimuth: Azimuth angle in degrees.
        camera_elevation: Elevation angle in degrees.
        camera_distance: Distance from scene centre.
        resolution: ``(width, height)`` in pixels.
        samples: Cycles render samples.

    Returns:
        Complete Blender Python script as a string.
    """
    object_defs = _build_object_defs(positions, object_info)
    script = _SCENE_SCRIPT_TEMPLATE.format(
        script_name="render_scene.py",
        assets_dir=assets_dir,
        output_image=output_image_path,
        res_x=resolution[0],
        res_y=resolution[1],
        samples=samples,
        cam_azimuth=camera_azimuth,
        cam_elevation=camera_elevation,
        cam_distance=camera_distance,
        object_defs=object_defs,
    )
    return script


def generate_comparison_script(
    gt_positions: Dict[str, list],
    recon_positions: Dict[str, list],
    object_info: Dict[str, dict],
    output_dir: str,
    scene_id: str,
    assets_dir: str = _DEFAULT_ASSETS_DIR,
    camera_azimuth: float = 45.0,
    camera_elevation: float = 30.0,
    camera_distance: float = 12.0,
    resolution: tuple = (480, 320),
    samples: int = 256,
) -> Tuple[str, str]:
    """Generate a pair of Blender scripts for GT / recon side-by-side renders.

    Applies Procrustes alignment to *recon_positions* so the two renders
    are visually comparable.

    Args:
        gt_positions: Ground-truth positions ``{obj_id: [x, y]}``.
        recon_positions: Reconstructed positions ``{obj_id: [x, y]}``.
        object_info: Per-object metadata dict.
        output_dir: Directory for generated scripts and rendered images.
        scene_id: Scene identifier used in file names.
        assets_dir: Blender assets root.
        camera_azimuth: Azimuth angle in degrees.
        camera_elevation: Elevation angle in degrees.
        camera_distance: Camera distance.
        resolution: ``(width, height)``.
        samples: Cycles samples.

    Returns:
        Tuple of ``(gt_script_path, recon_script_path)`` that were written.
    """
    from reconstruct.utils import procrustes_align  # noqa: delayed import

    # Procrustes-align recon to GT
    common_ids = sorted(set(gt_positions.keys()) & set(recon_positions.keys()))
    if len(common_ids) < 2:
        raise ValueError(
            f"Need at least 2 common objects for Procrustes alignment, "
            f"got {len(common_ids)}"
        )

    gt_mat = np.array([gt_positions[oid][:2] for oid in common_ids])
    recon_mat = np.array([recon_positions[oid][:2] for oid in common_ids])
    recon_aligned, _rms = procrustes_align(recon_mat, gt_mat)

    aligned_positions: Dict[str, list] = {}
    for i, oid in enumerate(common_ids):
        aligned_positions[oid] = recon_aligned[i].tolist()

    os.makedirs(output_dir, exist_ok=True)

    # GT script
    gt_image = os.path.join(output_dir, f"{scene_id}_gt.png")
    gt_script = generate_blender_script(
        positions=gt_positions,
        object_info=object_info,
        output_image_path=gt_image,
        assets_dir=assets_dir,
        camera_azimuth=camera_azimuth,
        camera_elevation=camera_elevation,
        camera_distance=camera_distance,
        resolution=resolution,
        samples=samples,
    )
    gt_script_path = os.path.join(output_dir, f"{scene_id}_gt.py")
    with open(gt_script_path, "w") as f:
        f.write(gt_script)

    # Recon script (using aligned positions)
    recon_image = os.path.join(output_dir, f"{scene_id}_recon.png")
    recon_script = generate_blender_script(
        positions=aligned_positions,
        object_info=object_info,
        output_image_path=recon_image,
        assets_dir=assets_dir,
        camera_azimuth=camera_azimuth,
        camera_elevation=camera_elevation,
        camera_distance=camera_distance,
        resolution=resolution,
        samples=samples,
    )
    recon_script_path = os.path.join(output_dir, f"{scene_id}_recon.py")
    with open(recon_script_path, "w") as f:
        f.write(recon_script)

    return gt_script_path, recon_script_path


def generate_overlay_script(
    gt_positions: Dict[str, list],
    recon_positions: Dict[str, list],
    object_info: Dict[str, dict],
    output_dir: str,
    scene_id: str,
    assets_dir: str = _DEFAULT_ASSETS_DIR,
    camera_azimuth: float = 45.0,
    camera_elevation: float = 30.0,
    camera_distance: float = 12.0,
    resolution: tuple = (480, 320),
    samples: int = 256,
) -> str:
    """Generate a Blender script that overlays GT and recon in one scene.

    GT objects are rendered solid; reconstructed objects are rendered
    semi-transparent (alpha=0.3). Thin red displacement lines connect each
    GT object to its reconstructed counterpart.

    Procrustes alignment is applied to recon positions beforehand.

    Args:
        gt_positions: Ground-truth positions ``{obj_id: [x, y]}``.
        recon_positions: Reconstructed positions ``{obj_id: [x, y]}``.
        object_info: Per-object metadata dict.
        output_dir: Directory for the generated script and rendered image.
        scene_id: Scene identifier.
        assets_dir: Blender assets root.
        camera_azimuth: Azimuth angle in degrees.
        camera_elevation: Elevation angle in degrees.
        camera_distance: Camera distance.
        resolution: ``(width, height)``.
        samples: Cycles samples.

    Returns:
        Path to the written overlay script.
    """
    from reconstruct.utils import procrustes_align  # noqa: delayed import

    common_ids = sorted(set(gt_positions.keys()) & set(recon_positions.keys()))
    if len(common_ids) < 2:
        raise ValueError(
            f"Need at least 2 common objects for Procrustes alignment, "
            f"got {len(common_ids)}"
        )

    gt_mat = np.array([gt_positions[oid][:2] for oid in common_ids])
    recon_mat = np.array([recon_positions[oid][:2] for oid in common_ids])
    recon_aligned, _rms = procrustes_align(recon_mat, gt_mat)

    aligned_positions: Dict[str, list] = {}
    for i, oid in enumerate(common_ids):
        aligned_positions[oid] = recon_aligned[i].tolist()

    gt_object_defs = _build_object_defs(gt_positions, object_info)
    recon_object_defs = _build_object_defs(aligned_positions, object_info)

    os.makedirs(output_dir, exist_ok=True)
    output_image = os.path.join(output_dir, f"{scene_id}_overlay.png")

    script = _OVERLAY_SCRIPT_TEMPLATE.format(
        script_name=f"{scene_id}_overlay.py",
        assets_dir=assets_dir,
        output_image=output_image,
        res_x=resolution[0],
        res_y=resolution[1],
        samples=samples,
        cam_azimuth=camera_azimuth,
        cam_elevation=camera_elevation,
        cam_distance=camera_distance,
        gt_object_defs=gt_object_defs,
        recon_object_defs=recon_object_defs,
    )

    script_path = os.path.join(output_dir, f"{scene_id}_overlay.py")
    with open(script_path, "w") as f:
        f.write(script)

    return script_path


def render_with_blender(
    script_path: str,
    blender_path: str = "blender",
    timeout: Optional[int] = None,
) -> subprocess.CompletedProcess:
    """Run a generated Blender script in background mode.

    Args:
        script_path: Path to the Blender Python script.
        blender_path: Path to the Blender executable.
        timeout: Optional timeout in seconds.

    Returns:
        ``subprocess.CompletedProcess`` with stdout/stderr.
    """
    cmd = [blender_path, "--background", "--python", script_path]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        print(f"Blender render failed (exit {result.returncode}):")
        print(result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr)
    return result


# ---------------------------------------------------------------------------
# Scene JSON helpers (for CLI)
# ---------------------------------------------------------------------------

def _extract_positions_and_info(scene: dict) -> Tuple[Dict[str, list], Dict[str, dict]]:
    """Extract positions and object_info from a data-gen scene JSON.

    Returns:
        (positions, object_info) where positions = {obj_id: [x, y]} and
        object_info = {obj_id: {"shape", "color", "material", "size"}}.
    """
    positions: Dict[str, list] = {}
    object_info: Dict[str, dict] = {}
    for obj in scene.get("objects", []):
        obj_id = obj["id"]
        coords = obj.get("3d_coords", [0, 0, 0])
        positions[obj_id] = [coords[0], coords[1]]
        object_info[obj_id] = {
            "shape": obj.get("shape", "sphere"),
            "color": obj.get("color", "gray"),
            "material": obj.get("material", "rubber"),
            "size": obj.get("size", "large"),
        }
    return positions, object_info


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    """CLI: generate Blender render scripts from scene + reconstruction JSON."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate Blender scripts for GT / reconstruction rendering."
    )
    parser.add_argument(
        "--scene-json",
        required=True,
        help="GT scene JSON from data-gen/output/scenes/",
    )
    parser.add_argument(
        "--recon-json",
        required=True,
        help='Reconstruction JSON with {"positions": {"obj_0": [x,y], ...}}',
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for generated scripts and rendered images",
    )
    parser.add_argument(
        "--assets-dir",
        default=_DEFAULT_ASSETS_DIR,
        help="Path to data-gen/blender/assets/",
    )
    parser.add_argument(
        "--camera-azimuth",
        type=float,
        default=45.0,
        help="Camera azimuth in degrees (default: 45)",
    )
    parser.add_argument(
        "--camera-elevation",
        type=float,
        default=30.0,
        help="Camera elevation in degrees (default: 30)",
    )
    parser.add_argument(
        "--camera-distance",
        type=float,
        default=12.0,
        help="Camera distance (default: 12)",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        nargs=2,
        default=[480, 320],
        metavar=("W", "H"),
        help="Render resolution (default: 480 320)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=256,
        help="Cycles render samples (default: 256)",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Also run the generated scripts with Blender",
    )
    parser.add_argument(
        "--blender-path",
        default="blender",
        help="Path to Blender executable (default: blender)",
    )
    args = parser.parse_args()

    # Load scene JSON
    with open(args.scene_json) as f:
        scene = json.load(f)

    scene_id = scene.get("scene_id", Path(args.scene_json).stem)
    gt_positions, object_info = _extract_positions_and_info(scene)

    # Load reconstruction JSON
    with open(args.recon_json) as f:
        recon_data = json.load(f)

    recon_positions: Dict[str, list] = recon_data.get("positions", {})
    if not recon_positions:
        print("ERROR: recon JSON has no 'positions' key or it is empty.")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    resolution = tuple(args.resolution)
    cam_kwargs = dict(
        camera_azimuth=args.camera_azimuth,
        camera_elevation=args.camera_elevation,
        camera_distance=args.camera_distance,
        resolution=resolution,
        samples=args.samples,
    )

    # 1. Comparison scripts (GT + recon side-by-side)
    print(f"Generating comparison scripts for scene {scene_id}...")
    gt_script_path, recon_script_path = generate_comparison_script(
        gt_positions=gt_positions,
        recon_positions=recon_positions,
        object_info=object_info,
        output_dir=args.output_dir,
        scene_id=scene_id,
        assets_dir=args.assets_dir,
        **cam_kwargs,
    )
    print(f"  GT script:    {gt_script_path}")
    print(f"  Recon script: {recon_script_path}")

    # 2. Overlay script
    print(f"Generating overlay script for scene {scene_id}...")
    overlay_script_path = generate_overlay_script(
        gt_positions=gt_positions,
        recon_positions=recon_positions,
        object_info=object_info,
        output_dir=args.output_dir,
        scene_id=scene_id,
        assets_dir=args.assets_dir,
        **cam_kwargs,
    )
    print(f"  Overlay script: {overlay_script_path}")

    # 3. Optionally render
    if args.render:
        print("\nRendering with Blender...")
        for label, path in [
            ("GT", gt_script_path),
            ("Recon", recon_script_path),
            ("Overlay", overlay_script_path),
        ]:
            print(f"  Rendering {label}: {path}")
            result = render_with_blender(path, blender_path=args.blender_path)
            if result.returncode == 0:
                print(f"    OK")
            else:
                print(f"    FAILED (exit {result.returncode})")

    print("\nDone.")


if __name__ == "__main__":
    main()
