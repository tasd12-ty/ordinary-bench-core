# Extended for 3D scene rendering with height variation

"""
3D scene rendering script for ORDINAL-SPATIAL-3D benchmark.

Renders scenes with objects at varying heights from multiple camera viewpoints
designed to make height differences visible.

Usage:
    blender --background --python render_3d.py -- [arguments]
"""

from __future__ import print_function
import math
import sys
import random
import argparse
import json
import os
from datetime import datetime as dt
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Any, Optional

INSIDE_BLENDER = True
try:
    import bpy
    import bpy_extras
    from mathutils import Vector, Euler
    BLENDER_VERSION = bpy.app.version
    IS_BLENDER_280_OR_LATER = BLENDER_VERSION >= (2, 80, 0)
except ImportError:
    INSIDE_BLENDER = False
    IS_BLENDER_280_OR_LATER = False
    BLENDER_VERSION = (0, 0, 0)

if INSIDE_BLENDER:
    # Import utils from data-gen/blender/ directory
    try:
        import utils
    except ImportError:
        # Add data-gen/blender to sys.path
        script_dir = os.path.dirname(os.path.abspath(__file__))
        parent_blender_dir = os.path.join(
            os.path.dirname(os.path.dirname(script_dir)),
            "data-gen", "blender"
        )
        for d in [script_dir, parent_blender_dir]:
            if d not in sys.path:
                sys.path.insert(0, d)
        try:
            import utils
        except ImportError:
            print("\nERROR: Cannot import utils.py")
            print(f"Tried: {script_dir}, {parent_blender_dir}")
            sys.exit(1)


@dataclass
class CameraConfig:
    """Configuration for a single camera viewpoint."""
    camera_id: str
    azimuth: float
    elevation: float
    distance: float
    look_at: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    def to_cartesian(self) -> Tuple[float, float, float]:
        azimuth_rad = math.radians(self.azimuth)
        elevation_rad = math.radians(self.elevation)
        x = self.distance * math.cos(elevation_rad) * math.cos(azimuth_rad)
        y = self.distance * math.cos(elevation_rad) * math.sin(azimuth_rad)
        z = self.distance * math.sin(elevation_rad)
        return (
            x + self.look_at[0],
            y + self.look_at[1],
            z + self.look_at[2],
        )

    def to_dict(self) -> Dict[str, Any]:
        pos = self.to_cartesian()
        return {
            "camera_id": self.camera_id,
            "azimuth": self.azimuth,
            "elevation": self.elevation,
            "distance": self.distance,
            "position": list(pos),
            "look_at": list(self.look_at),
        }


@dataclass
class MultiViewConfig:
    """多视角渲染配置（保留兼容性）。"""
    n_views: int = 4
    camera_distance: float = 14.0
    elevation: float = 35.0
    azimuth_start: float = 45.0
    look_at: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    def generate_cameras(self) -> List[CameraConfig]:
        cameras = []
        azimuth_step = 360.0 / self.n_views
        for i in range(self.n_views):
            azimuth = (self.azimuth_start + i * azimuth_step) % 360.0
            cameras.append(CameraConfig(
                camera_id=f"view_{i}",
                azimuth=azimuth,
                elevation=self.elevation,
                distance=self.camera_distance,
                look_at=self.look_at,
            ))
        return cameras


def generate_cube_face_cameras(
    look_at: Tuple[float, float, float] = (0.0, 0.0, 1.25),
    distance: float = 14.0,
) -> List[CameraConfig]:
    """生成正方体六面法向量方向的正交相机。

    六个视角对应正方体六个面的法向量方向观察：
      front:  +Y 方向看（azimuth=90°, elevation=0°）
      back:   -Y 方向看（azimuth=270°, elevation=0°）
      right:  +X 方向看（azimuth=0°, elevation=0°）
      left:   -X 方向看（azimuth=180°, elevation=0°）
      top:    从上往下看（elevation=90°）
      bottom: 从下往上看（elevation=-90°）

    四个水平视角 elevation=0°，相机与物体同高度水平拍摄。

    参数：
        look_at: 相机朝向的目标点（建议设为场景 z 方向中心）
        distance: 相机到 look_at 的距离
    """
    # 六面定义：(view_id, azimuth, elevation)
    faces = [
        ("front",  90.0,   0.0),
        ("back",   270.0,  0.0),
        ("right",  0.0,    0.0),
        ("left",   180.0,  0.0),
        ("top",    0.0,    90.0),
        ("bottom", 0.0,   -90.0),
    ]

    cameras = []
    for view_id, azimuth, elevation in faces:
        cameras.append(CameraConfig(
            camera_id=view_id,
            azimuth=azimuth,
            elevation=elevation,
            distance=distance,
            look_at=look_at,
        ))
    return cameras


def set_camera_position(camera_config: CameraConfig) -> None:
    """Set Blender camera position and orientation."""
    camera = bpy.data.objects['Camera']
    position = camera_config.to_cartesian()
    look_at = camera_config.look_at
    camera.location = position
    direction = Vector(look_at) - Vector(position)
    rot_quat = direction.to_track_quat('-Z', 'Y')
    camera.rotation_euler = rot_quat.to_euler()


def refresh_camera_state() -> None:
    """Flush Blender's camera transform updates."""
    view_layer = getattr(bpy.context, "view_layer", None)
    if view_layer is not None:
        view_layer.update()


def get_object_by_name(name: str, alternative_names: Optional[List[str]] = None):
    """Get Blender object by name with fallbacks."""
    if name in bpy.data.objects:
        return bpy.data.objects[name]
    if alternative_names:
        for alt_name in alternative_names:
            if alt_name in bpy.data.objects:
                return bpy.data.objects[alt_name]
    raise KeyError(f"Object not found: {name}")


def compute_pixel_coords_for_view(camera, objects_3d: List[Dict]) -> List[Dict]:
    """Compute pixel coordinates for all objects from current camera view."""
    updated_objects = []
    for obj in objects_3d:
        obj_copy = obj.copy()
        coords_3d = obj['3d_coords']
        pixel_coords = utils.get_camera_coords(camera, Vector(coords_3d))
        obj_copy['pixel_coords'] = pixel_coords
        updated_objects.append(obj_copy)
    return updated_objects


def compute_scene_center_3d(objects_3d: List[Dict]) -> Tuple[float, float, float]:
    """Compute center of mass of objects in 3D."""
    if not objects_3d:
        return (0.0, 0.0, 0.0)
    xs = [obj["3d_coords"][0] for obj in objects_3d]
    ys = [obj["3d_coords"][1] for obj in objects_3d]
    zs = [obj["3d_coords"][2] for obj in objects_3d]
    return (
        sum(xs) / len(xs),
        sum(ys) / len(ys),
        sum(zs) / len(zs),
    )


def compute_scene_extent_3d(objects_3d: List[Dict]) -> float:
    """Compute maximum extent of the scene in any axis."""
    if not objects_3d:
        return 6.0
    xs = [obj["3d_coords"][0] for obj in objects_3d]
    ys = [obj["3d_coords"][1] for obj in objects_3d]
    zs = [obj["3d_coords"][2] for obj in objects_3d]
    span_x = max(xs) - min(xs)
    span_y = max(ys) - min(ys)
    span_z = max(zs) - min(zs)
    return max(span_x, span_y, span_z)


def sample_z_coordinate(args) -> float:
    """Sample z coordinate based on distribution mode."""
    z_min = args.z_min
    z_max = args.z_max

    if args.z_distribution == "uniform":
        return random.uniform(z_min, z_max)
    elif args.z_distribution == "discrete_levels":
        levels = json.loads(args.z_levels) if args.z_levels else [0.0, 1.0, 2.0]
        valid = [z for z in levels if z_min <= z <= z_max]
        return random.choice(valid) if valid else random.uniform(z_min, z_max)
    elif args.z_distribution == "gaussian":
        z_mean = (z_min + z_max) / 2.0
        z_std = (z_max - z_min) / 4.0
        z = random.gauss(z_mean, z_std)
        return max(z_min, min(z_max, z))
    else:
        return random.uniform(z_min, z_max)


def distance_3d(pos1, pos2):
    """3D Euclidean distance."""
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(pos1, pos2)))


def _create_grid_material(name, color1, color2, scale=4.0, roughness=0.85):
    """创建棋盘格网格材质，用于地面和围挡墙面。"""
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    for node in nodes:
        nodes.remove(node)

    output_node = nodes.new(type='ShaderNodeOutputMaterial')
    output_node.location = (400, 0)

    bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
    bsdf.location = (100, 0)
    bsdf.inputs['Roughness'].default_value = roughness
    bsdf.inputs['Specular IOR Level'].default_value = 0.05

    checker = nodes.new(type='ShaderNodeTexChecker')
    checker.location = (-300, 0)
    checker.inputs['Scale'].default_value = scale
    checker.inputs['Color1'].default_value = color1
    checker.inputs['Color2'].default_value = color2

    tex_coord = nodes.new(type='ShaderNodeTexCoord')
    tex_coord.location = (-500, 0)

    links.new(tex_coord.outputs['Object'], checker.inputs['Vector'])
    links.new(checker.outputs['Color'], bsdf.inputs['Base Color'])
    links.new(bsdf.outputs['BSDF'], output_node.inputs['Surface'])

    return mat


def enhance_ground_and_shadows(box_size=5.0, wall_height=3.5):
    """构建三维参考盒子并增强光影。

    用一个连体网格创建 U 形盒子（地面 + 后墙 + 左右侧墙），
    所有面共享顶点，无接缝。统一材质形成完整的空间参考容器。

    参数：
        box_size: 盒子半尺寸（地面从 -box_size 到 +box_size）
        wall_height: 围挡墙高度
    """
    # ── 统一材质（地面和墙面共用）──
    box_mat = _create_grid_material(
        "Box_Grid",
        color1=(0.62, 0.62, 0.64, 1.0),
        color2=(0.78, 0.78, 0.80, 1.0),
        scale=4.0,
        roughness=0.90,
    )

    # ── 替换原有地面材质 ──
    ground = bpy.data.objects.get('Ground')
    if ground and ground.data:
        if ground.data.materials:
            ground.data.materials[0] = box_mat
        else:
            ground.data.materials.append(box_mat)

    # ── 构建三面围挡墙（独立网格，渲染时按视角动态隐藏）──
    s = box_size
    h = wall_height

    def _add_wall(name, verts, face):
        """创建单面墙体物体。"""
        mesh = bpy.data.meshes.new(name)
        mesh.from_pydata(verts, [], [face])
        mesh.update()
        obj = bpy.data.objects.new(name, mesh)
        if IS_BLENDER_280_OR_LATER:
            bpy.context.collection.objects.link(obj)
        else:
            bpy.context.scene.objects.link(obj)
        obj.data.materials.append(box_mat)
        for poly in mesh.polygons:
            poly.use_smooth = False
        return obj

    # 后墙（y=-s，法线朝 +y，从盒子内侧看可见）
    _add_wall("Wall_Back",
        [(-s, -s, 0), (s, -s, 0), (s, -s, h), (-s, -s, h)],
        (0, 1, 2, 3))

    # 左墙（x=-s，法线朝 +x，从盒子内侧看可见）
    _add_wall("Wall_Left",
        [(-s, -s, 0), (-s, s, 0), (-s, s, h), (-s, -s, h)],
        (0, 1, 2, 3))

    # 右墙（x=+s，法线朝 -x，从盒子内侧看可见）
    _add_wall("Wall_Right",
        [(s, s, 0), (s, -s, 0), (s, -s, h), (s, s, h)],
        (0, 1, 2, 3))

    # ── 增强光照与阴影 ──
    for lamp_name in ['Lamp_Key', 'Light_Key', 'Lamp_Back', 'Light_Back',
                       'Lamp_Fill', 'Light_Fill']:
        if lamp_name in bpy.data.objects:
            light = bpy.data.objects[lamp_name].data
            light.shadow_soft_size = 2.0
            light.energy *= 1.2

    # 顶部太阳光：向下投射垂直阴影
    bpy.ops.object.light_add(type='SUN', location=(0, 0, 10))
    sun = bpy.context.object
    sun.name = "Sun_TopDown"
    sun.data.energy = 3.0
    sun.data.angle = 0.05
    sun.rotation_euler = (0, 0, 0)


def render_single_view(
    camera_config: CameraConfig,
    output_image: str,
    objects_3d: List[Dict],
    args,
) -> Dict[str, Any]:
    """Render scene from a single camera viewpoint."""
    set_camera_position(camera_config)
    refresh_camera_state()
    camera = bpy.data.objects['Camera']
    objects_with_pixels = compute_pixel_coords_for_view(camera, objects_3d)

    bpy.context.scene.render.filepath = output_image
    bpy.ops.render.render(write_still=True)

    return {
        "view_id": camera_config.camera_id,
        "image_path": os.path.basename(output_image),
        "camera": camera_config.to_dict(),
        "objects": objects_with_pixels,
    }


def render_top_view(
    output_image: str,
    objects_3d: List[Dict],
    blender_objects: Optional[List[Any]],
    args,
) -> Dict[str, Any]:
    """Render orthographic top-down view."""
    camera = bpy.data.objects['Camera']
    original_type = camera.data.type
    original_ortho_scale = getattr(camera.data, "ortho_scale", None)
    original_clip_end = camera.data.clip_end

    # Compute scene bounds for framing
    center = compute_scene_center_3d(objects_3d)
    extent = compute_scene_extent_3d(objects_3d)
    top_height = max(4.0, center[2] + extent + 2.0)
    ortho_scale = max(6.0, extent + 2.0 * args.top_view_padding)

    top_camera = CameraConfig(
        camera_id="top_view",
        azimuth=0.0,
        elevation=90.0,
        distance=top_height,
        look_at=(center[0], center[1], 0.0),
    )

    set_camera_position(top_camera)
    refresh_camera_state()
    camera.data.type = 'ORTHO'
    camera.data.ortho_scale = ortho_scale
    camera.data.clip_end = max(original_clip_end, top_height + 20.0)
    refresh_camera_state()

    objects_with_pixels = compute_pixel_coords_for_view(camera, objects_3d)

    bpy.context.scene.render.filepath = output_image
    bpy.ops.render.render(write_still=True)

    view_data = {
        "view_id": "top_view",
        "image_path": os.path.basename(output_image),
        "camera": top_camera.to_dict(),
        "projection": "orthographic",
        "ortho_scale": ortho_scale,
        "objects": objects_with_pixels,
    }

    # Restore camera settings
    camera.data.type = original_type
    if original_ortho_scale is not None:
        camera.data.ortho_scale = original_ortho_scale
    camera.data.clip_end = original_clip_end

    return view_data


def render_side_view(
    output_image: str,
    objects_3d: List[Dict],
    args,
) -> Dict[str, Any]:
    """Render side view at low elevation to show height differences."""
    center = compute_scene_center_3d(objects_3d)
    extent = compute_scene_extent_3d(objects_3d)
    distance = max(14.0, extent * 2.0)

    side_camera = CameraConfig(
        camera_id="side_view",
        azimuth=90.0,         # View from +Y direction
        elevation=15.0,        # Low angle to emphasize height
        distance=distance,
        look_at=center,
    )

    set_camera_position(side_camera)
    refresh_camera_state()
    camera = bpy.data.objects['Camera']
    objects_with_pixels = compute_pixel_coords_for_view(camera, objects_3d)

    bpy.context.scene.render.filepath = output_image
    bpy.ops.render.render(write_still=True)

    return {
        "view_id": "side_view",
        "image_path": os.path.basename(output_image),
        "camera": side_camera.to_dict(),
        "objects": objects_with_pixels,
    }


def add_random_objects_3d(directions, num_objects, args, camera, _retry_count=0):
    """
    Add random objects to the scene with 3D placement (z != 0).

    Key difference from data-gen: objects are placed at (x, y, z) where z is
    sampled from the configured distribution. Minimum distance is checked in 3D.
    """
    MAX_RETRIES = 100
    if _retry_count >= MAX_RETRIES:
        raise RuntimeError(f"Failed to place objects after {MAX_RETRIES} attempts")

    # Scale placement area
    if num_objects <= 6:
        placement_range = 3.0
    elif num_objects <= 10:
        placement_range = 3.5
    else:
        placement_range = 4.0

    effective_min_dist = args.min_dist
    effective_margin = args.margin
    min_dist_3d = args.min_dist_3d

    if num_objects > 10:
        effective_min_dist = max(0.15, args.min_dist * 0.7)
        effective_margin = max(0.25, args.margin * 0.7)

    # Load properties
    with open(args.properties_json, 'r') as f:
        properties = json.load(f)
        color_name_to_rgba = {}
        for name, rgb in properties['colors'].items():
            rgba = [float(c) / 255.0 for c in rgb] + [1.0]
            color_name_to_rgba[name] = rgba
        material_mapping = [(v, k) for k, v in properties['materials'].items()]
        object_mapping = [(v, k) for k, v in properties['shapes'].items()]
        size_mapping = list(properties['sizes'].items())

    positions = []  # (x, y, z, radius)
    objects = []
    blender_objects = []

    for i in range(num_objects):
        size_name, r = random.choice(size_mapping)

        num_tries = 0
        while True:
            num_tries += 1
            if num_tries > args.max_retries:
                for obj in blender_objects:
                    utils.delete_object(obj)
                return add_random_objects_3d(
                    directions, num_objects, args, camera,
                    _retry_count=_retry_count + 1,
                )

            x = random.uniform(-placement_range, placement_range)
            y = random.uniform(-placement_range, placement_range)
            z = sample_z_coordinate(args)

            # Check 3D minimum distances
            dists_good = True
            margins_good = True
            for (xx, yy, zz, rr) in positions:
                dist = distance_3d((x, y, z), (xx, yy, zz))
                if dist - r - rr < min_dist_3d:
                    dists_good = False
                    break
                # 2D margin check for visual separation in views
                dx, dy = x - xx, y - yy
                for direction_name in ['left', 'right', 'front', 'behind']:
                    direction_vec = directions[direction_name]
                    margin_val = dx * direction_vec[0] + dy * direction_vec[1]
                    if 0 < margin_val < effective_margin:
                        margins_good = False
                        break
                if not margins_good:
                    break

            if dists_good and margins_good:
                break

        obj_name, obj_name_out = random.choice(object_mapping)
        color_name, rgba = random.choice(list(color_name_to_rgba.items()))

        if obj_name == 'Cube':
            r /= math.sqrt(2)

        theta = 360.0 * random.random()

        # Place object at 3D position
        # utils.add_object places at z=0, then we translate to target z
        utils.add_object(args.shape_dir, obj_name, r, (x, y), theta=theta)
        obj = bpy.context.object
        # Move object to target z height
        obj.location.z = z
        blender_objects.append(obj)
        positions.append((x, y, z, r))

        # Add material
        mat_name, mat_name_out = random.choice(material_mapping)
        utils.add_material(mat_name, Color=rgba)

        # Record object data with full 3D coords
        pixel_coords = utils.get_camera_coords(camera, obj.location)
        objects.append({
            "id": f"obj_{i}",
            "shape": obj_name_out,
            "size": size_name,
            "material": mat_name_out,
            "3d_coords": [float(obj.location.x), float(obj.location.y), float(obj.location.z)],
            "rotation": theta,
            "pixel_coords": pixel_coords,
            "color": color_name,
        })

    return objects, blender_objects


def render_multiview_3d_scene(
    args,
    num_objects: int,
    output_index: int,
    output_split: str,
    output_dir: str,
    mv_config: MultiViewConfig,
) -> Dict[str, Any]:
    """Render a complete 3D multi-view scene."""
    os.makedirs(output_dir, exist_ok=True)

    bpy.ops.wm.open_mainfile(filepath=args.base_scene_blendfile)
    utils.load_materials(args.material_dir)

    # Render settings
    render_args = bpy.context.scene.render
    render_args.engine = "CYCLES"
    render_args.resolution_x = args.width
    render_args.resolution_y = args.height
    render_args.resolution_percentage = 100

    if args.use_gpu == 1:
        if IS_BLENDER_280_OR_LATER:
            cycles_prefs = bpy.context.preferences.addons['cycles'].preferences
            for compute_type in ['CUDA', 'OPTIX', 'HIP', 'ONEAPI']:
                try:
                    cycles_prefs.compute_device_type = compute_type
                    for device in cycles_prefs.devices:
                        device.use = True
                    break
                except:
                    continue
        bpy.context.scene.cycles.device = 'GPU'

    bpy.context.scene.cycles.samples = args.render_num_samples
    bpy.context.scene.cycles.blur_glossy = 2.0
    if IS_BLENDER_280_OR_LATER:
        bpy.context.scene.cycles.transparent_max_bounces = args.render_max_bounces

    # Enhance ground and shadows for 3D height perception
    enhance_ground_and_shadows()

    scene_id = f"{output_split}_{output_index:06d}"
    scene_struct = {
        "scene_id": scene_id,
        "split": output_split,
        "image_index": output_index,
        "n_objects": num_objects,
        "is_3d": True,
        "z_range": [args.z_min, args.z_max],
        "z_distribution": args.z_distribution,
        "objects": [],
        "world_constraints": {},
        "views": [],
    }

    # Light jitter
    def rand(L):
        return 2.0 * L * (random.random() - 0.5)

    lamp_key = get_object_by_name('Lamp_Key', ['Light_Key', 'Key', 'KeyLight'])
    lamp_back = get_object_by_name('Lamp_Back', ['Light_Back', 'Back', 'BackLight'])
    lamp_fill = get_object_by_name('Lamp_Fill', ['Light_Fill', 'Fill', 'FillLight'])

    if args.key_light_jitter > 0:
        for i in range(3):
            lamp_key.location[i] += rand(args.key_light_jitter)
    if args.back_light_jitter > 0:
        for i in range(3):
            lamp_back.location[i] += rand(args.back_light_jitter)
    if args.fill_light_jitter > 0:
        for i in range(3):
            lamp_fill.location[i] += rand(args.fill_light_jitter)

    # Initial camera for placement
    cameras = mv_config.generate_cameras()
    set_camera_position(cameras[0])

    # Direction computation
    if IS_BLENDER_280_OR_LATER:
        bpy.ops.mesh.primitive_plane_add(size=10)
    else:
        bpy.ops.mesh.primitive_plane_add(radius=5)
    plane = bpy.context.object
    plane_normal = plane.data.vertices[0].normal
    camera = bpy.data.objects['Camera']

    if IS_BLENDER_280_OR_LATER:
        cam_behind = camera.matrix_world.to_quaternion() @ Vector((0, 0, -1))
        cam_left = camera.matrix_world.to_quaternion() @ Vector((-1, 0, 0))
    else:
        cam_behind = camera.matrix_world.to_quaternion() * Vector((0, 0, -1))
        cam_left = camera.matrix_world.to_quaternion() * Vector((-1, 0, 0))

    plane_behind = (cam_behind - cam_behind.project(plane_normal)).normalized()
    plane_left = (cam_left - cam_left.project(plane_normal)).normalized()
    temp_directions = {
        'behind': tuple(plane_behind),
        'front': tuple(-plane_behind),
        'left': tuple(plane_left),
        'right': tuple(-plane_left),
    }
    utils.delete_object(plane)

    # Place objects in 3D
    objects_3d, blender_objects = add_random_objects_3d(
        temp_directions, num_objects, args, camera
    )

    scene_struct["objects"] = objects_3d

    # 计算场景中心，水平视角 look_at 对准 z 方向中心
    center = compute_scene_center_3d(objects_3d)
    z_center = (args.z_min + args.z_max) / 2.0
    look_at_3d = (center[0], center[1], z_center)

    # 生成正方体六面正交相机
    cameras = generate_cube_face_cameras(
        look_at=look_at_3d,
        distance=mv_config.camera_distance,
    )

    # 视角 → 需要隐藏的遮挡物体映射
    # 每个视角隐藏相机正对的面，避免遮挡物体
    HIDE_MAP = {
        "back":   ["Wall_Back"],
        "left":   ["Wall_Left"],
        "right":  ["Wall_Right"],
        "bottom": ["Ground"],
    }

    # 渲染六个正交视角（动态隐藏遮挡面）
    for cam_config in cameras:
        # 渲染前隐藏遮挡物体
        hidden_objs = []
        for obj_name in HIDE_MAP.get(cam_config.camera_id, []):
            obj = bpy.data.objects.get(obj_name)
            if obj:
                obj.hide_render = True
                hidden_objs.append(obj)

        img_path = os.path.join(output_dir, f"{cam_config.camera_id}.png")
        view_data = render_single_view(cam_config, img_path, objects_3d, args)
        scene_struct["views"].append(view_data)

        # 渲染后恢复
        for obj in hidden_objs:
            obj.hide_render = False

    scene_struct["scene_center"] = list(center)
    scene_struct["camera_mode"] = "cube_faces"

    # Save scene metadata
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(scene_struct, f, indent=2)

    # 复制 front 视角作为单视角代表
    if args.output_single_view_dir:
        single_view_dir = args.output_single_view_dir
        os.makedirs(single_view_dir, exist_ok=True)
        single_img_path = os.path.join(single_view_dir, f"{scene_id}.png")
        import shutil
        front_path = os.path.join(output_dir, "front.png")
        if os.path.exists(front_path):
            shutil.copy(front_path, single_img_path)

    return scene_struct


def main(args):
    """Main entry point for 3D scene rendering."""
    random.seed(args.seed + args.start_idx)
    print(f"Starting 3D rendering: {args.num_images} scenes, 6 cube-face views each")
    print(f"  Z-range: [{args.z_min}, {args.z_max}], distribution: {args.z_distribution}")

    mv_config = MultiViewConfig(
        n_views=6,
        camera_distance=args.camera_distance,
        elevation=0.0,
        azimuth_start=0.0,
    )

    # 创建输出目录（六面视角统一存放在 multi_view/ 下）
    multiview_dir = os.path.join(args.output_dir, "multi_view")
    single_view_dir = os.path.join(args.output_dir, "single_view")
    os.makedirs(multiview_dir, exist_ok=True)
    os.makedirs(single_view_dir, exist_ok=True)

    args.output_single_view_dir = single_view_dir

    all_scenes = []
    successful = 0
    failed = 0

    for i in range(args.num_images):
        # 每个场景使用独立 seed：base_seed * 场景索引，确保可复现且互不相同
        scene_seed = args.seed * 1000 + (i + args.start_idx)
        random.seed(scene_seed)

        scene_id = f"{args.split}_{(i + args.start_idx):06d}"
        scene_output_dir = os.path.join(multiview_dir, scene_id)
        num_objects = random.randint(args.min_objects, args.max_objects)

        try:
            scene_struct = render_multiview_3d_scene(
                args,
                num_objects=num_objects,
                output_index=(i + args.start_idx),
                output_split=args.split,
                output_dir=scene_output_dir,
                mv_config=mv_config,
            )
            all_scenes.append(scene_struct)
            successful += 1
            print(f"  [{successful}/{args.num_images}] Rendered {scene_id}")
        except Exception as e:
            print(f"  [ERROR] Failed to render {scene_id}: {e}")
            failed += 1
            continue

    print(f"\nRendering complete: {successful} successful, {failed} failed")

    output_file = os.path.join(args.output_dir, f"{args.split}_scenes.json")
    output_data = {
        "info": {
            "date": dt.today().strftime("%Y-%m-%d"),
            "split": args.split,
            "n_views": args.n_views,
            "is_3d": True,
            "z_range": [args.z_min, args.z_max],
            "z_distribution": args.z_distribution,
            "render_top_view": bool(args.render_top_view),
            "camera_config": asdict(mv_config),
        },
        "scenes": all_scenes,
    }

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"Saved scenes to {output_file}")


# Argument parser
parser = argparse.ArgumentParser(description="3D scene rendering")

# Input options
parser.add_argument('--base_scene_blendfile', default='assets/base_scene_v5.blend')
parser.add_argument('--properties_json', default='assets/properties.json')
parser.add_argument('--shape_dir', default='assets/shapes_v5')
parser.add_argument('--material_dir', default='assets/materials_v5')

# Object settings
parser.add_argument('--min_objects', default=4, type=int)
parser.add_argument('--max_objects', default=10, type=int)
parser.add_argument('--min_dist', default=0.25, type=float)
parser.add_argument('--margin', default=0.4, type=float)
parser.add_argument('--max_retries', default=50, type=int)

# 3D placement settings
parser.add_argument('--z_min', default=0.0, type=float, help="Minimum z coordinate")
parser.add_argument('--z_max', default=2.5, type=float, help="Maximum z coordinate")
parser.add_argument('--z_distribution', default='uniform',
    help="Z distribution: uniform, discrete_levels, gaussian")
parser.add_argument('--z_levels', default=None, type=str,
    help="JSON array of discrete z levels")
parser.add_argument('--min_dist_3d', default=0.5, type=float,
    help="Minimum 3D Euclidean distance between objects")

# Multi-view settings
parser.add_argument('--n_views', default=4, type=int)
parser.add_argument('--camera_distance', default=14.0, type=float)
parser.add_argument('--elevation', default=35.0, type=float)
parser.add_argument('--azimuth_start', default=45.0, type=float)
parser.add_argument('--render_top_view', default=0, type=int)
parser.add_argument('--top_view_height', default=None, type=float)
parser.add_argument('--top_view_padding', default=0.35, type=float)

# Output settings
parser.add_argument('--output_dir', default='../output/')
parser.add_argument('--start_idx', default=0, type=int)
parser.add_argument('--num_images', default=5, type=int)
parser.add_argument('--split', default='train')
parser.add_argument('--seed', default=42, type=int)

# Render settings
parser.add_argument('--use_gpu', default=0, type=int)
parser.add_argument('--width', default=480, type=int)
parser.add_argument('--height', default=320, type=int)
parser.add_argument('--render_num_samples', default=256, type=int)
parser.add_argument('--render_max_bounces', default=8, type=int)
parser.add_argument('--key_light_jitter', default=1.0, type=float)
parser.add_argument('--fill_light_jitter', default=1.0, type=float)
parser.add_argument('--back_light_jitter', default=1.0, type=float)


if __name__ == '__main__':
    if INSIDE_BLENDER:
        argv = utils.extract_args()
        args = parser.parse_args(argv)
        main(args)
    elif '--help' in sys.argv or '-h' in sys.argv:
        parser.print_help()
    else:
        print('Run from Blender:')
        print('  blender --background --python render_3d.py -- [args]')
