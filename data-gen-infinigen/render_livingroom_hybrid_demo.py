"""
Render a living-room style hybrid scene:
- background/furniture via existing GLB assets
- tabletop geometric probes via current ordinary-bench shape assets

Run with:
  /Applications/Blender.app/Contents/MacOS/Blender --background --python data-gen-infinigen/render_livingroom_hybrid_demo.py
"""

import math
import os
from pathlib import Path

import bpy
from mathutils import Vector, Euler


ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = ROOT / "data-gen" / "blender" / "assets"
SHAPE_DIR = ASSETS_DIR / "shapes_v5"
MATERIAL_DIR = ASSETS_DIR / "materials_v5"
MODEL_DIR = ROOT / "blender-assets-demo" / "models"
OUTPUT_DIR = Path(__file__).resolve().parent / "examples" / "hybrid-output"

OUTPUT_IMAGE = OUTPUT_DIR / "livingroom_hybrid_demo.png"
RENDER_WIDTH = 1440
RENDER_HEIGHT = 960
RENDER_SAMPLES = 96


COLOR_MAP = {
    "red": [173 / 255.0, 35 / 255.0, 35 / 255.0, 1.0],
    "blue": [42 / 255.0, 75 / 255.0, 215 / 255.0, 1.0],
    "green": [29 / 255.0, 105 / 255.0, 20 / 255.0, 1.0],
    "yellow": [1.0, 238 / 255.0, 51 / 255.0, 1.0],
}

SHAPE_FILE = {
    "sphere": "Sphere",
    "cube": "SmoothCube_v2",
    "cylinder": "SmoothCylinder",
}

MATERIAL_FILE = {
    "metal": "MyMetal",
    "rubber": "Rubber",
}

TABLETOP_OBJECTS = [
    {"shape": "sphere", "material": "rubber", "color": "red", "offset": (-0.42, -0.12), "scale": 0.20, "rot": 0.0},
    {"shape": "cube", "material": "metal", "color": "blue", "offset": (-0.08, 0.18), "scale": 0.18, "rot": 0.55},
    {"shape": "cylinder", "material": "metal", "color": "yellow", "offset": (0.22, -0.04), "scale": 0.17, "rot": 0.0},
    {"shape": "sphere", "material": "metal", "color": "green", "offset": (0.48, 0.20), "scale": 0.16, "rot": 0.0},
]


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for block in bpy.data.meshes:
        if block.users == 0:
            bpy.data.meshes.remove(block)
    for block in bpy.data.materials:
        if block.users == 0:
            bpy.data.materials.remove(block)


def setup_render():
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.render.resolution_x = RENDER_WIDTH
    scene.render.resolution_y = RENDER_HEIGHT
    scene.render.resolution_percentage = 100
    scene.cycles.samples = RENDER_SAMPLES
    scene.cycles.use_denoising = True
    scene.render.film_transparent = False
    try:
        cycles_prefs = bpy.context.preferences.addons["cycles"].preferences
        for compute_type in ("METAL", "CUDA", "OPTIX", "HIP", "ONEAPI"):
            try:
                cycles_prefs.compute_device_type = compute_type
                cycles_prefs.get_devices()
                for device in cycles_prefs.devices:
                    device.use = True
                scene.cycles.device = "GPU"
                break
            except Exception:
                continue
        else:
            scene.cycles.device = "CPU"
    except Exception:
        scene.cycles.device = "CPU"


def create_principled_material(name, base_color, roughness=0.55, metallic=0.0):
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = base_color
        bsdf.inputs["Roughness"].default_value = roughness
        bsdf.inputs["Metallic"].default_value = metallic
    return mat


def add_environment():
    world = bpy.data.worlds.get("World") or bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs["Color"].default_value = (0.97, 0.94, 0.90, 1.0)
        bg.inputs["Strength"].default_value = 0.35

    bpy.ops.mesh.primitive_plane_add(size=18, location=(0, 0, 0))
    floor = bpy.context.object
    floor.name = "Floor"
    floor.data.materials.append(create_principled_material("FloorMat", (0.72, 0.66, 0.58, 1.0), roughness=0.88))

    bpy.ops.mesh.primitive_plane_add(size=18, location=(0, -5.5, 4.4), rotation=(math.radians(90), 0, 0))
    back_wall = bpy.context.object
    back_wall.name = "BackWall"
    back_wall.data.materials.append(create_principled_material("BackWallMat", (0.93, 0.90, 0.84, 1.0), roughness=0.95))

    bpy.ops.mesh.primitive_plane_add(size=12, location=(-5.5, 0.2, 3.8), rotation=(0, math.radians(90), 0))
    side_wall = bpy.context.object
    side_wall.name = "SideWall"
    side_wall.data.materials.append(create_principled_material("SideWallMat", (0.88, 0.86, 0.80, 1.0), roughness=0.95))

    bpy.ops.mesh.primitive_plane_add(size=4.8, location=(0.25, 0.15, 0.01))
    rug = bpy.context.object
    rug.name = "Rug"
    rug.data.materials.append(create_principled_material("RugMat", (0.24, 0.30, 0.36, 1.0), roughness=0.92))

    bpy.ops.mesh.primitive_cube_add(location=(2.8, -4.8, 3.3), scale=(1.3, 0.04, 1.1))
    window = bpy.context.object
    window.name = "WindowLightBox"
    window.data.materials.append(create_principled_material("WindowMat", (0.82, 0.90, 1.0, 1.0), roughness=0.05))

    bpy.ops.object.light_add(type="AREA", location=(2.6, -4.3, 3.4))
    key = bpy.context.object
    key.data.energy = 6500
    key.data.shape = "RECTANGLE"
    key.data.size = 2.2
    key.data.size_y = 1.2
    key.rotation_euler = Euler((math.radians(88), 0, math.radians(0)))

    bpy.ops.object.light_add(type="AREA", location=(-2.4, 2.2, 4.4))
    fill = bpy.context.object
    fill.data.energy = 2200
    fill.data.size = 4.5
    fill.rotation_euler = Euler((math.radians(60), 0, math.radians(-120)))

    bpy.ops.object.light_add(type="SUN", location=(0, 0, 6))
    sun = bpy.context.object
    sun.data.energy = 1.4
    sun.rotation_euler = Euler((math.radians(45), math.radians(15), math.radians(28)))


def add_camera():
    bpy.ops.object.camera_add()
    camera = bpy.context.object
    camera.name = "Camera"
    bpy.context.scene.camera = camera
    camera.location = (5.8, 6.5, 4.3)
    look_at = Vector((0.0, 0.05, 1.0))
    direction = look_at - camera.location
    camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    camera.data.lens = 42
    camera.data.clip_start = 0.1
    camera.data.clip_end = 100.0


def import_glb(filepath, location, scale=1.0, rotation_z_deg=0.0):
    existing = set(bpy.data.objects.keys())
    bpy.ops.import_scene.gltf(filepath=str(filepath))
    new_objects = [obj for name, obj in bpy.data.objects.items() if name not in existing]
    new_set = {id(obj) for obj in new_objects}
    roots = [obj for obj in new_objects if obj.parent is None or id(obj.parent) not in new_set]
    if not roots:
        return None
    if len(roots) > 1:
        bpy.ops.object.empty_add(location=location)
        parent = bpy.context.object
        for root in roots:
            root.parent = parent
        target = parent
    else:
        target = roots[0]

    all_coords = []
    for obj in new_objects:
        if obj.type == "MESH":
            for corner in obj.bound_box:
                all_coords.append(obj.matrix_world @ Vector(corner))
    if all_coords:
        xs = [c.x for c in all_coords]
        ys = [c.y for c in all_coords]
        zs = [c.z for c in all_coords]
        bbox_size = max(max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs))
        normalize = 2.0 / bbox_size if bbox_size > 0 else 1.0
        effective_scale = normalize * scale
    else:
        effective_scale = scale

    target.scale = (effective_scale, effective_scale, effective_scale)
    target.location = location
    target.rotation_euler[2] = math.radians(rotation_z_deg)
    return target


def append_node_group(blend_path, group_name):
    bpy.ops.wm.append(
        filepath=str(blend_path / "NodeTree" / group_name),
        directory=str(blend_path / "NodeTree"),
        filename=group_name,
    )


def append_shape_object(shape_name):
    bpy.ops.wm.append(
        filepath=str((SHAPE_DIR / f"{shape_name}.blend") / "Object" / shape_name),
        directory=str((SHAPE_DIR / f"{shape_name}.blend") / "Object"),
        filename=shape_name,
    )
    obj = bpy.data.objects[shape_name]
    obj.name = f"{shape_name}_{len([o for o in bpy.data.objects if o.name.startswith(shape_name)])}"
    return obj


def apply_material(obj, material_name, rgba):
    mat = bpy.data.materials.new(name=f"{material_name}_{obj.name}")
    mat.use_nodes = True
    out = next(node for node in mat.node_tree.nodes if node.type == "OUTPUT_MATERIAL")
    group = mat.node_tree.nodes.new("ShaderNodeGroup")
    group.node_tree = bpy.data.node_groups[material_name]
    for inp in group.inputs:
        if inp.name == "Color":
            inp.default_value = rgba
    mat.node_tree.links.new(group.outputs["Shader"], out.inputs["Surface"])
    obj.data.materials.clear()
    obj.data.materials.append(mat)


def table_top_height(target):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    coords = []
    for obj in bpy.data.objects:
        if obj == target or obj.parent == target or target in obj.children_recursive:
            eval_obj = obj.evaluated_get(depsgraph)
            if hasattr(eval_obj, "bound_box"):
                for corner in eval_obj.bound_box:
                    coords.append(eval_obj.matrix_world @ Vector(corner))
    if not coords:
        return 0.8
    return max(co.z for co in coords)


def add_tabletop_objects(table):
    for mat_name in MATERIAL_FILE.values():
        append_node_group(MATERIAL_DIR / f"{mat_name}.blend", mat_name)

    top_z = table_top_height(table)
    for spec in TABLETOP_OBJECTS:
        shape_name = SHAPE_FILE[spec["shape"]]
        obj = append_shape_object(shape_name)
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        if spec["shape"] == "cube":
            scale = spec["scale"] / math.sqrt(2.0)
        else:
            scale = spec["scale"]
        bpy.ops.transform.resize(value=(scale, scale, scale))
        obj.location = (spec["offset"][0], spec["offset"][1], top_z + scale * 0.95)
        obj.rotation_euler[2] = spec["rot"]
        apply_material(obj, MATERIAL_FILE[spec["material"]], COLOR_MAP[spec["color"]])
        obj.select_set(False)


def build_scene():
    clear_scene()
    setup_render()
    add_environment()
    add_camera()

    table = import_glb(MODEL_DIR / "table.glb", location=(0.0, 0.0, 0.0), scale=1.18, rotation_z_deg=8)
    import_glb(MODEL_DIR / "chair.glb", location=(-1.75, 0.2, 0.0), scale=1.08, rotation_z_deg=92)
    import_glb(MODEL_DIR / "chair.glb", location=(1.95, -0.45, 0.0), scale=1.0, rotation_z_deg=-35)
    import_glb(MODEL_DIR / "plant.glb", location=(2.9, 1.45, 0.0), scale=1.3, rotation_z_deg=12)
    import_glb(MODEL_DIR / "human_1.glb", location=(-3.35, 1.55, 0.0), scale=1.03, rotation_z_deg=-18)

    add_tabletop_objects(table)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    bpy.context.scene.render.filepath = str(OUTPUT_IMAGE)
    bpy.ops.render.render(write_still=True)
    print(f"Rendered to {OUTPUT_IMAGE}")


if __name__ == "__main__":
    build_scene()
