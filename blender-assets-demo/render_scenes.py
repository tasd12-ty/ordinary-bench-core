"""
Blender render script for demo scenes using downloaded .glb models.

Scene 1: table + chair + plant + human_1 (indoor feel)
Scene 2: 2 humans + table + plant (outdoor / conversation feel)

Usage:
    /Applications/Blender.app/Contents/MacOS/Blender --background --python render_scenes.py

Renders to output/ at 480x320, CYCLES engine.
"""

import os
import sys
import math

import bpy
from mathutils import Vector, Euler

BLENDER_VERSION = bpy.app.version
IS_BLENDER_280_OR_LATER = BLENDER_VERSION >= (2, 80, 0)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, "models")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")

# Render settings (consistent with existing pipeline)
RENDER_WIDTH = 1920
RENDER_HEIGHT = 1280
RENDER_SAMPLES = 256
RENDER_ENGINE = "CYCLES"

# Camera settings (adapted from render_multiview.py)
CAMERA_DISTANCE = 10.0
CAMERA_ELEVATION = 25.0  # degrees
CAMERA_LOOK_AT = (0.0, 0.0, 0.5)  # slightly above ground


# ---------- Scene definitions ----------

SCENE_CONFIGS = [
    {
        "name": "scene_indoor",
        "description": "Table + chair + plant + human (indoor)",
        "objects": [
            {"model": "table.glb",   "pos": (0.0, 0.0, 0.0),   "scale": 1.0, "rot_z": 0},
            {"model": "chair.glb",   "pos": (-1.5, 0.0, 0.0),  "scale": 1.0, "rot_z": 90},
            {"model": "plant.glb",   "pos": (1.2, 1.0, 0.0),   "scale": 1.0, "rot_z": 0},
            {"model": "human_1.glb", "pos": (-2.5, 1.0, 0.0),  "scale": 1.0, "rot_z": -30},
        ],
        "camera_azimuth": 45.0,
    },
    {
        "name": "scene_outdoor",
        "description": "2 humans + table + plant (outdoor / conversation)",
        "objects": [
            {"model": "human_1.glb", "pos": (-1.5, -1.0, 0.0), "scale": 1.0, "rot_z": 45},
            {"model": "human_2.glb", "pos": (1.5, -1.0, 0.0),  "scale": 1.0, "rot_z": -45},
            {"model": "table.glb",   "pos": (0.0, 0.0, 0.0),   "scale": 0.9, "rot_z": 0},
            {"model": "plant.glb",   "pos": (2.0, 1.5, 0.0),   "scale": 1.2, "rot_z": 20},
        ],
        "camera_azimuth": 30.0,
    },
]


# ---------- Helpers ----------

def clear_scene():
    """Remove all objects, meshes, materials from the scene."""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

    # Purge orphan data
    for block in bpy.data.meshes:
        if block.users == 0:
            bpy.data.meshes.remove(block)
    for block in bpy.data.materials:
        if block.users == 0:
            bpy.data.materials.remove(block)


def setup_render_settings():
    """Configure render engine and resolution."""
    scene = bpy.context.scene
    scene.render.engine = RENDER_ENGINE
    scene.render.resolution_x = RENDER_WIDTH
    scene.render.resolution_y = RENDER_HEIGHT
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = False

    # Cycles settings
    scene.cycles.samples = RENDER_SAMPLES
    scene.cycles.use_denoising = True

    # Try GPU, fall back to CPU
    try:
        cycles_prefs = bpy.context.preferences.addons['cycles'].preferences
        for compute_type in ['METAL', 'CUDA', 'OPTIX', 'HIP', 'ONEAPI']:
            try:
                cycles_prefs.compute_device_type = compute_type
                cycles_prefs.get_devices()
                for device in cycles_prefs.devices:
                    device.use = True
                scene.cycles.device = 'GPU'
                print(f"  Using GPU ({compute_type})")
                break
            except Exception:
                continue
        else:
            scene.cycles.device = 'CPU'
            print("  Using CPU")
    except Exception:
        scene.cycles.device = 'CPU'
        print("  Using CPU (fallback)")


def add_ground_plane(size=20.0, color=(0.4, 0.4, 0.4, 1.0)):
    """Add a ground plane with a simple diffuse material."""
    bpy.ops.mesh.primitive_plane_add(size=size, location=(0, 0, 0))
    plane = bpy.context.object
    plane.name = "GroundPlane"

    mat = bpy.data.materials.new(name="GroundMaterial")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = color
        bsdf.inputs["Roughness"].default_value = 0.8
    plane.data.materials.append(mat)
    return plane


def add_sky_light():
    """Add environment lighting (world HDRI-like) and a sun lamp."""
    # World background
    world = bpy.data.worlds.get("World")
    if world is None:
        world = bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.use_nodes = True

    bg_node = world.node_tree.nodes.get("Background")
    if bg_node:
        bg_node.inputs["Color"].default_value = (0.7, 0.8, 1.0, 1.0)
        bg_node.inputs["Strength"].default_value = 0.8

    # Sun lamp (key light)
    bpy.ops.object.light_add(type='SUN', location=(5, -5, 10))
    sun = bpy.context.object
    sun.name = "SunLight"
    sun.data.energy = 3.0
    sun.rotation_euler = Euler((math.radians(50), math.radians(10), math.radians(30)))

    # Fill light (area light)
    bpy.ops.object.light_add(type='AREA', location=(-4, 3, 6))
    fill = bpy.context.object
    fill.name = "FillLight"
    fill.data.energy = 50.0
    fill.data.size = 5.0
    fill.rotation_euler = Euler((math.radians(60), 0, math.radians(-120)))

    return sun, fill


def add_camera(azimuth_deg, elevation_deg, distance, look_at):
    """Add and position a camera."""
    bpy.ops.object.camera_add()
    camera = bpy.context.object
    camera.name = "Camera"
    bpy.context.scene.camera = camera

    # Spherical to Cartesian
    az = math.radians(azimuth_deg)
    el = math.radians(elevation_deg)
    x = distance * math.cos(el) * math.cos(az) + look_at[0]
    y = distance * math.cos(el) * math.sin(az) + look_at[1]
    z = distance * math.sin(el) + look_at[2]

    camera.location = (x, y, z)

    # Point camera at look_at
    direction = Vector(look_at) - Vector((x, y, z))
    rot_quat = direction.to_track_quat('-Z', 'Y')
    camera.rotation_euler = rot_quat.to_euler()

    # Lens
    camera.data.lens = 35
    camera.data.clip_start = 0.1
    camera.data.clip_end = 100.0

    return camera


def import_glb(filepath, location=(0, 0, 0), scale=1.0, rotation_z_deg=0):
    """Import a .glb model and position it."""
    if not os.path.exists(filepath):
        print(f"  WARNING: Model not found: {filepath}")
        return None

    # Record existing objects
    existing = set(bpy.data.objects.keys())

    bpy.ops.import_scene.gltf(filepath=filepath)

    # Find newly imported objects
    new_objects = [obj for name, obj in bpy.data.objects.items() if name not in existing]

    if not new_objects:
        print(f"  WARNING: No objects imported from {filepath}")
        return None

    # Find root objects (no parent among new objects)
    new_set = set(id(o) for o in new_objects)
    roots = [o for o in new_objects if o.parent is None or id(o.parent) not in new_set]

    # If multiple roots, parent them to an empty
    if len(roots) > 1:
        bpy.ops.object.empty_add(location=location)
        parent_empty = bpy.context.object
        parent_empty.name = os.path.basename(filepath).replace('.glb', '_root')
        for r in roots:
            r.parent = parent_empty
        target = parent_empty
    else:
        target = roots[0]

    # Normalize size: compute bounding box of all new objects and scale
    # to fit within a reasonable range (roughly 1-2 units tall)
    all_coords = []
    for obj in new_objects:
        if hasattr(obj, 'bound_box') and obj.type == 'MESH':
            for corner in obj.bound_box:
                world_co = obj.matrix_world @ Vector(corner)
                all_coords.append(world_co)

    if all_coords:
        xs = [c.x for c in all_coords]
        ys = [c.y for c in all_coords]
        zs = [c.z for c in all_coords]
        bbox_size = max(max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs))
        if bbox_size > 0:
            # Normalize to ~2 units max dimension, then apply user scale
            normalize_factor = 2.0 / bbox_size
            effective_scale = normalize_factor * scale
        else:
            effective_scale = scale
    else:
        effective_scale = scale

    target.scale = (effective_scale, effective_scale, effective_scale)
    target.location = location
    target.rotation_euler[2] = math.radians(rotation_z_deg)

    print(f"  Imported: {os.path.basename(filepath)} -> {target.name} "
          f"(scale={effective_scale:.3f})")
    return target


# ---------- Main ----------

def build_and_render_scene(config):
    """Build a single scene from config and render it."""
    scene_name = config["name"]
    print(f"\n{'='*50}")
    print(f"Building scene: {scene_name}")
    print(f"  {config['description']}")
    print(f"{'='*50}")

    # Clean slate
    clear_scene()

    # Setup rendering
    setup_render_settings()

    # Add environment
    add_ground_plane()
    add_sky_light()

    # Add camera
    add_camera(
        azimuth_deg=config["camera_azimuth"],
        elevation_deg=CAMERA_ELEVATION,
        distance=CAMERA_DISTANCE,
        look_at=CAMERA_LOOK_AT,
    )

    # Import models
    for obj_config in config["objects"]:
        model_path = os.path.join(MODELS_DIR, obj_config["model"])
        import_glb(
            filepath=model_path,
            location=obj_config["pos"],
            scale=obj_config["scale"],
            rotation_z_deg=obj_config["rot_z"],
        )

    # Render
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, f"{scene_name}.png")
    bpy.context.scene.render.filepath = output_path
    print(f"  Rendering to {output_path} ...")
    bpy.ops.render.render(write_still=True)
    print(f"  Done: {output_path}")

    return output_path


def main():
    rendered = []
    for config in SCENE_CONFIGS:
        try:
            path = build_and_render_scene(config)
            rendered.append(path)
        except Exception as e:
            print(f"  ERROR rendering {config['name']}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*50}")
    print(f"Rendering complete: {len(rendered)}/{len(SCENE_CONFIGS)} scenes")
    for p in rendered:
        print(f"  {p}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
