"""
ORDINARY-BENCH 网格渲染脚本。

在可见的 4x4 网格上摆放物体并渲染场景。
每个物体严格居中于一个网格单元格。
支持可选的网格坐标标签。

用法：
    blender --background --python render_grid.py -- [arguments]
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
    try:
        import utils
    except ImportError:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        if script_dir not in sys.path:
            sys.path.insert(0, script_dir)
        try:
            import utils
        except ImportError:
            print("\nERROR: Cannot import utils.py")
            print(f"Tried adding script dir to sys.path: {script_dir}")
            sys.exit(1)


# ---------------------------------------------------------------------------
# 相机配置（复用自 render_multiview.py）
# ---------------------------------------------------------------------------

@dataclass
class CameraConfig:
    """单个相机视角的配置。"""
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
    """多视角渲染配置。"""
    n_views: int = 4
    camera_distance: float = 12.0
    elevation: float = 30.0
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


# ---------------------------------------------------------------------------
# 相机辅助函数（复用自 render_multiview.py）
# ---------------------------------------------------------------------------

def compute_top_view_frame(
    objects_3d: List[Dict],
    blender_objects: Optional[List[Any]] = None,
    padding: float = 0.35,
    aspect_ratio: float = 1.0,
    grid_size: Optional[Tuple[float, float]] = None,
) -> Tuple[Tuple[float, float, float], float]:
    """计算能容纳整个场景的居中正交视口。

    若提供 *grid_size* ``(width, height)``，则视口保证覆盖整个网格
    （以原点为中心）加上 *padding*。物体的包围盒仍会被考虑，
    若物体超出网格范围，视口将相应扩大而非裁剪物体。
    """
    # -- 基于网格的最小视口 --
    grid_width = 0.0
    if grid_size is not None:
        gw, gh = grid_size
        grid_width = max(gw + 2.0 * padding,
                         aspect_ratio * (gh + 2.0 * padding))

    # -- 基于物体的视口 --
    obj_width = 0.0
    if blender_objects:
        min_x = float("inf")
        min_y = float("inf")
        max_x = float("-inf")
        max_y = float("-inf")
        for obj in blender_objects:
            for corner in obj.bound_box:
                world_corner = obj.matrix_world @ Vector(corner)
                min_x = min(min_x, world_corner.x)
                min_y = min(min_y, world_corner.y)
                max_x = max(max_x, world_corner.x)
                max_y = max(max_y, world_corner.y)
        if min_x < max_x and min_y < max_y:
            span_x = max_x - min_x
            span_y = max_y - min_y
            obj_width = max(span_x + 2.0 * padding,
                            aspect_ratio * (span_y + 2.0 * padding))
    elif objects_3d:
        xs = [float(obj["3d_coords"][0]) for obj in objects_3d]
        ys = [float(obj["3d_coords"][1]) for obj in objects_3d]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        span_x = max_x - min_x
        span_y = max_y - min_y
        obj_width = max(span_x + 2.0 * padding,
                        aspect_ratio * (span_y + 2.0 * padding))

    if grid_size is not None:
        # 网格优先；以网格原点为中心。
        width = max(grid_width, obj_width)
        return (0.0, 0.0, 0.0), width

    if obj_width > 0:
        if blender_objects:
            center = (0.5 * (min_x + max_x), 0.5 * (min_y + max_y), 0.0)
        else:
            center = (0.5 * (min_x + max_x), 0.5 * (min_y + max_y), 0.0)
        return center, max(6.0, obj_width)

    return (0.0, 0.0, 0.0), 10.0


def compute_top_view_height(
    blender_objects: Optional[List[Any]] = None,
    min_clearance: float = 1.0,
) -> float:
    """将正交相机置于最高物体的正上方。"""
    if blender_objects:
        max_z = float("-inf")
        for obj in blender_objects:
            for corner in obj.bound_box:
                world_corner = obj.matrix_world @ Vector(corner)
                max_z = max(max_z, world_corner.z)
        if max_z != float("-inf"):
            return max_z + min_clearance
    return 4.0


def get_object_by_name(name: str, alternative_names: Optional[List[str]] = None):
    """按名称获取 Blender 对象，支持备用名称。"""
    if name in bpy.data.objects:
        return bpy.data.objects[name]
    if alternative_names:
        for alt_name in alternative_names:
            if alt_name in bpy.data.objects:
                return bpy.data.objects[alt_name]
    raise KeyError(f"Object not found: {name}")


def set_camera_position(camera_config: CameraConfig) -> None:
    """设置 Blender 相机的位置和朝向。"""
    camera = bpy.data.objects['Camera']
    position = camera_config.to_cartesian()
    look_at = camera_config.look_at
    camera.location = position
    direction = Vector(look_at) - Vector(position)
    rot_quat = direction.to_track_quat('-Z', 'Y')
    camera.rotation_euler = rot_quat.to_euler()


def refresh_camera_state() -> None:
    """刷新 Blender 相机变换更新。"""
    view_layer = getattr(bpy.context, "view_layer", None)
    if view_layer is not None:
        view_layer.update()


def compute_pixel_coords_for_view(camera, objects_3d: List[Dict]) -> List[Dict]:
    """从当前相机视角计算所有物体的像素坐标。"""
    updated_objects = []
    for obj in objects_3d:
        obj_copy = obj.copy()
        coords_3d = obj['3d_coords']
        pixel_coords = utils.get_camera_coords(camera, Vector(coords_3d))
        obj_copy['pixel_coords'] = pixel_coords
        updated_objects.append(obj_copy)
    return updated_objects


def compute_directions_for_view(camera) -> Dict[str, Tuple[float, float, float]]:
    """计算相对于当前相机视角的基本方向。"""
    if IS_BLENDER_280_OR_LATER:
        bpy.ops.mesh.primitive_plane_add(size=10)
    else:
        bpy.ops.mesh.primitive_plane_add(radius=5)
    plane = bpy.context.object
    plane_normal = plane.data.vertices[0].normal

    if IS_BLENDER_280_OR_LATER:
        cam_behind = camera.matrix_world.to_quaternion() @ Vector((0, 0, -1))
        cam_left = camera.matrix_world.to_quaternion() @ Vector((-1, 0, 0))
        cam_up = camera.matrix_world.to_quaternion() @ Vector((0, 1, 0))
    else:
        cam_behind = camera.matrix_world.to_quaternion() * Vector((0, 0, -1))
        cam_left = camera.matrix_world.to_quaternion() * Vector((-1, 0, 0))
        cam_up = camera.matrix_world.to_quaternion() * Vector((0, 1, 0))

    plane_behind = (cam_behind - cam_behind.project(plane_normal)).normalized()
    plane_left = (cam_left - cam_left.project(plane_normal)).normalized()
    plane_up = cam_up.project(plane_normal).normalized()

    utils.delete_object(plane)

    return {
        'behind': tuple(plane_behind),
        'front': tuple(-plane_behind),
        'left': tuple(plane_left),
        'right': tuple(-plane_left),
        'above': tuple(plane_up),
        'below': tuple(-plane_up),
    }


# ---------------------------------------------------------------------------
# 渲染辅助函数（复用自 render_multiview.py）
# ---------------------------------------------------------------------------

def render_single_view(
    camera_config: CameraConfig,
    output_image: str,
    objects_3d: List[Dict],
    args,
) -> Dict[str, Any]:
    """从单个相机视角渲染场景。"""
    set_camera_position(camera_config)
    refresh_camera_state()

    camera = bpy.data.objects['Camera']
    objects_with_pixels = compute_pixel_coords_for_view(camera, objects_3d)
    directions = compute_directions_for_view(camera)

    bpy.context.scene.render.filepath = output_image
    bpy.ops.render.render(write_still=True)

    return {
        "view_id": camera_config.camera_id,
        "image_path": os.path.basename(output_image),
        "camera": camera_config.to_dict(),
        "directions": directions,
        "objects": objects_with_pixels,
    }


def render_top_view(
    output_image: str,
    objects_3d: List[Dict],
    blender_objects: Optional[List[Any]],
    args,
) -> Dict[str, Any]:
    """渲染正交俯视图。"""
    camera = bpy.data.objects['Camera']
    original_type = camera.data.type
    original_ortho_scale = getattr(camera.data, "ortho_scale", None)
    original_clip_start = camera.data.clip_start
    original_clip_end = camera.data.clip_end

    top_height = getattr(args, "top_view_height", None)
    if top_height is None:
        top_height = compute_top_view_height(
            blender_objects,
            min_clearance=max(0.8, 0.5 * getattr(args, "top_view_padding", 0.35)),
        )
    aspect_ratio = float(args.width) / float(args.height)
    grid_w = getattr(args, "grid_cols", 4) * getattr(args, "cell_size", 1.5)
    grid_h = getattr(args, "grid_rows", 4) * getattr(args, "cell_size", 1.5)
    top_center, ortho_scale = compute_top_view_frame(
        objects_3d,
        blender_objects=blender_objects,
        padding=getattr(args, "top_view_padding", 0.35),
        aspect_ratio=aspect_ratio,
        grid_size=(grid_w, grid_h),
    )
    top_camera = CameraConfig(
        camera_id="top_view",
        azimuth=0.0,
        elevation=90.0,
        distance=top_height,
        look_at=top_center,
    )

    set_camera_position(top_camera)
    refresh_camera_state()
    camera.data.type = 'ORTHO'
    camera.data.ortho_scale = ortho_scale
    camera.data.clip_start = 0.01
    camera.data.clip_end = max(original_clip_end, top_height + 20.0)
    refresh_camera_state()

    objects_with_pixels = compute_pixel_coords_for_view(camera, objects_3d)

    bpy.context.scene.render.filepath = output_image
    bpy.ops.render.render(write_still=True)

    view_data = {
        "view_id": top_camera.camera_id,
        "image_path": os.path.basename(output_image),
        "camera": top_camera.to_dict(),
        "projection": "orthographic",
        "ortho_scale": camera.data.ortho_scale,
        "objects": objects_with_pixels,
    }

    camera.data.type = original_type
    if original_ortho_scale is not None:
        camera.data.ortho_scale = original_ortho_scale
    camera.data.clip_start = original_clip_start
    camera.data.clip_end = original_clip_end

    return view_data


# ---------------------------------------------------------------------------
# 网格专用函数
# ---------------------------------------------------------------------------

ROW_LABELS = ['A', 'B', 'C', 'D']


def cell_center(row: int, col: int, rows: int, cols: int, cell_size: float) -> Tuple[float, float]:
    """计算网格单元格 (row, col) 中心的世界坐标 (x, y)。"""
    grid_w = cols * cell_size
    grid_h = rows * cell_size
    x = (col + 0.5) * cell_size - grid_w / 2.0
    y = (row + 0.5) * cell_size - grid_h / 2.0
    return (x, y)


def cell_label(row: int, col: int) -> str:
    """返回易读的单元格标签，如 A1、B3 等。"""
    return f"{ROW_LABELS[row]}{col + 1}"


def create_grid_lines(rows: int = 4, cols: int = 4, cell_size: float = 1.5,
                       line_width: float = 0.02) -> List[Any]:
    """
    在地面平面上创建可见的网格线。

    网格线为细长扁平方块，使用自发光材质，
    无论光照和相机角度如何都保持可见。
    """
    grid_w = cols * cell_size   # 总宽度（X 方向）
    grid_h = rows * cell_size   # 总高度（Y 方向）
    half_w = grid_w / 2.0
    half_h = grid_h / 2.0
    z_offset = 0.002  # slightly above ground to avoid z-fighting

    grid_objects = []

    # 创建网格线的自发光材质
    mat = bpy.data.materials.new(name="GridLineMaterial")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    # 移除默认的 Principled BSDF 节点
    for node in list(nodes):
        if node.type != 'OUTPUT_MATERIAL':
            nodes.remove(node)
    emission = nodes.new('ShaderNodeEmission')
    emission.inputs['Color'].default_value = (0.1, 0.1, 0.1, 1.0)
    emission.inputs['Strength'].default_value = 1.0
    output_node = None
    for node in nodes:
        if node.type == 'OUTPUT_MATERIAL':
            output_node = node
            break
    links.new(emission.outputs['Emission'], output_node.inputs['Surface'])

    # 水平线（Y 固定，沿 X 方向延伸整个宽度）
    for i in range(rows + 1):
        y = i * cell_size - half_h
        if IS_BLENDER_280_OR_LATER:
            bpy.ops.mesh.primitive_cube_add(size=1, location=(0, y, z_offset))
        else:
            bpy.ops.mesh.primitive_cube_add(radius=0.5, location=(0, y, z_offset))
        line = bpy.context.object
        line.name = f"grid_line_h_{i}"
        line.scale = (grid_w, line_width, 0.002)
        line.data.materials.append(mat)
        grid_objects.append(line)

    # 垂直线（X 固定，沿 Y 方向延伸整个高度）
    for j in range(cols + 1):
        x = j * cell_size - half_w
        if IS_BLENDER_280_OR_LATER:
            bpy.ops.mesh.primitive_cube_add(size=1, location=(x, 0, z_offset))
        else:
            bpy.ops.mesh.primitive_cube_add(radius=0.5, location=(x, 0, z_offset))
        line = bpy.context.object
        line.name = f"grid_line_v_{j}"
        line.scale = (line_width, grid_h, 0.002)
        line.data.materials.append(mat)
        grid_objects.append(line)

    return grid_objects


def create_grid_labels(rows: int = 4, cols: int = 4, cell_size: float = 1.5) -> List[Any]:
    """
    在每个网格单元格中心创建文字标签（如 A1、B2、...、D4）。

    标签使用自发光材质并朝上，
    以便从俯视角清晰可读，并可从透视角部分可见。
    """
    label_objects = []

    # 创建标签的自发光材质
    mat = bpy.data.materials.new(name="GridLabelMaterial")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    for node in list(nodes):
        if node.type != 'OUTPUT_MATERIAL':
            nodes.remove(node)
    emission = nodes.new('ShaderNodeEmission')
    emission.inputs['Color'].default_value = (0.3, 0.3, 0.3, 1.0)
    emission.inputs['Strength'].default_value = 1.0
    output_node = None
    for node in nodes:
        if node.type == 'OUTPUT_MATERIAL':
            output_node = node
            break
    links.new(emission.outputs['Emission'], output_node.inputs['Surface'])

    for r in range(rows):
        for c in range(cols):
            cx, cy = cell_center(r, c, rows, cols, cell_size)
            label_text = cell_label(r, c)

            bpy.ops.object.text_add(location=(cx, cy, 0.003))
            txt_obj = bpy.context.object
            txt_obj.name = f"grid_label_{label_text}"
            txt_obj.data.body = label_text
            txt_obj.data.size = 0.3
            txt_obj.data.align_x = 'CENTER'
            txt_obj.data.align_y = 'CENTER'
            # 旋转使文字朝上（从俯视角可读）
            txt_obj.rotation_euler[0] = math.radians(90)
            # 偏移使居中文字对齐单元格中心
            txt_obj.location.x = cx
            txt_obj.location.y = cy

            # 应用材质
            txt_obj.data.materials.append(mat)
            label_objects.append(txt_obj)

    return label_objects


def add_grid_objects(
    num_objects: int,
    args,
    camera,
    rows: int = 4,
    cols: int = 4,
    cell_size: float = 1.5,
) -> Tuple[List[Dict], List[Any]]:
    """
    在网格单元格中心摆放物体。

    从网格中随机选取 num_objects 个单元格，
    在每个选定单元格的精确中心摆放一个物体。仅使用 'small' 尺寸。
    """
    # 加载属性文件
    with open(args.properties_json, 'r') as f:
        properties = json.load(f)
        color_name_to_rgba = {}
        for name, rgb in properties['colors'].items():
            rgba = [float(c) / 255.0 for c in rgb] + [1.0]
            color_name_to_rgba[name] = rgba
        material_mapping = [(v, k) for k, v in properties['materials'].items()]
        object_mapping = [(v, k) for k, v in properties['shapes'].items()]

    # 网格模式固定使用 small 尺寸
    size_scale = properties['sizes']['small']

    # 构建所有单元格列表并随机选取
    all_cells = [(r, c) for r in range(rows) for c in range(cols)]
    selected_cells = random.sample(all_cells, min(num_objects, len(all_cells)))

    objects_3d = []
    blender_objects = []

    for i, (r, c) in enumerate(selected_cells):
        cx, cy = cell_center(r, c, rows, cols, cell_size)

        # 随机选择形状、颜色和材质
        obj_name, obj_name_out = random.choice(object_mapping)
        color_name, rgba = random.choice(list(color_name_to_rgba.items()))
        mat_name, mat_name_out = random.choice(material_mapping)

        r_scale = size_scale
        if obj_name == 'Cube':
            r_scale /= math.sqrt(2)

        theta = 360.0 * random.random()

        # 在单元格中心添加物体
        utils.add_object(args.shape_dir, obj_name, r_scale, (cx, cy), theta=theta)
        obj = bpy.context.object
        blender_objects.append(obj)

        # 应用材质
        utils.add_material(mat_name, Color=rgba)

        # 记录元数据
        pixel_coords = utils.get_camera_coords(camera, obj.location)
        objects_3d.append({
            "id": f"obj_{i}",
            "shape": obj_name_out,
            "size": "small",
            "material": mat_name_out,
            "3d_coords": tuple(obj.location),
            "rotation": theta,
            "pixel_coords": pixel_coords,
            "color": color_name,
            "cell_row": r,
            "cell_col": c,
            "cell_label": cell_label(r, c),
        })

    return objects_3d, blender_objects


# ---------------------------------------------------------------------------
# 主场景渲染
# ---------------------------------------------------------------------------

def render_grid_scene(
    args,
    num_objects: int,
    output_index: int,
    output_split: str,
    output_dir: str,
    mv_config: MultiViewConfig,
) -> Dict[str, Any]:
    """渲染完整的网格场景。"""
    os.makedirs(output_dir, exist_ok=True)

    # 加载基础场景
    bpy.ops.wm.open_mainfile(filepath=args.base_scene_blendfile)
    utils.load_materials(args.material_dir)

    # 渲染设置
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

    # 场景元数据
    scene_id = f"{output_split}_{output_index:06d}"
    scene_struct = {
        "scene_id": scene_id,
        "split": output_split,
        "image_index": output_index,
        "n_objects": num_objects,
        "placement_mode": "grid",
        "grid_info": {
            "rows": args.grid_rows,
            "cols": args.grid_cols,
            "cell_size": args.cell_size,
            "grid_extent": [
                -(args.grid_cols * args.cell_size) / 2.0,
                (args.grid_cols * args.cell_size) / 2.0,
            ],
            "labels_visible": bool(args.grid_labels),
        },
        "objects": [],
        "world_constraints": {},
        "views": [],
    }

    # 光源抖动
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

    # 创建网格线
    create_grid_lines(
        rows=args.grid_rows,
        cols=args.grid_cols,
        cell_size=args.cell_size,
        line_width=args.grid_line_width,
    )

    # 可选地创建网格标签
    if args.grid_labels == 1:
        create_grid_labels(
            rows=args.grid_rows,
            cols=args.grid_cols,
            cell_size=args.cell_size,
        )

    # 设置物体摆放时的初始相机
    cameras = mv_config.generate_cameras()
    set_camera_position(cameras[0])
    camera = bpy.data.objects['Camera']

    # 在网格上摆放物体
    objects_3d, blender_objects = add_grid_objects(
        num_objects, args, camera,
        rows=args.grid_rows,
        cols=args.grid_cols,
        cell_size=args.cell_size,
    )
    scene_struct["objects"] = objects_3d

    # 从每个视角渲染
    for cam_config in cameras:
        img_path = os.path.join(output_dir, f"{cam_config.camera_id}.png")
        view_data = render_single_view(cam_config, img_path, objects_3d, args)
        scene_struct["views"].append(view_data)

    # 俯视图
    if getattr(args, "render_top_view", 0) == 1 and getattr(args, "output_top_view_dir", None):
        os.makedirs(args.output_top_view_dir, exist_ok=True)
        top_img_path = os.path.join(args.output_top_view_dir, f"{scene_id}.png")
        scene_struct["top_view"] = render_top_view(
            top_img_path, objects_3d, blender_objects, args,
        )

    # 保存元数据
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(scene_struct, f, indent=2)

    # 将 view_0 复制为单视角图像
    if args.output_single_view_dir:
        single_view_dir = args.output_single_view_dir
        os.makedirs(single_view_dir, exist_ok=True)
        single_img_path = os.path.join(single_view_dir, f"{scene_id}.png")
        import shutil
        view0_path = os.path.join(output_dir, "view_0.png")
        if os.path.exists(view0_path):
            shutil.copy(view0_path, single_img_path)

    return scene_struct


# ---------------------------------------------------------------------------
# 主程序入口
# ---------------------------------------------------------------------------

def main(args):
    """网格渲染的主入口函数。"""
    random.seed(args.seed + args.start_idx)
    print(f"Starting grid rendering: {args.num_images} scenes, {args.n_views} views each "
          f"(grid {args.grid_rows}x{args.grid_cols}, seed={args.seed + args.start_idx})")

    mv_config = MultiViewConfig(
        n_views=args.n_views,
        camera_distance=args.camera_distance,
        elevation=args.elevation,
        azimuth_start=args.azimuth_start,
    )

    multiview_dir = os.path.join(args.output_dir, "multi_view")
    single_view_dir = os.path.join(args.output_dir, "single_view")
    top_view_dir = os.path.join(args.output_dir, "top_view")
    os.makedirs(multiview_dir, exist_ok=True)
    os.makedirs(single_view_dir, exist_ok=True)
    if args.render_top_view == 1:
        os.makedirs(top_view_dir, exist_ok=True)

    args.output_single_view_dir = single_view_dir
    args.output_top_view_dir = top_view_dir

    all_scenes = []
    successful = 0
    failed = 0

    for i in range(args.num_images):
        scene_id = f"{args.split}_{(i + args.start_idx):06d}"
        scene_output_dir = os.path.join(multiview_dir, scene_id)

        num_objects = random.randint(args.min_objects, args.max_objects)

        try:
            scene_struct = render_grid_scene(
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
            "render_top_view": bool(args.render_top_view),
            "camera_config": asdict(mv_config),
            "grid": {
                "rows": args.grid_rows,
                "cols": args.grid_cols,
                "cell_size": args.cell_size,
                "labels_visible": bool(args.grid_labels),
            },
        },
        "scenes": all_scenes,
    }

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"Saved scenes to {output_file}")


# ---------------------------------------------------------------------------
# 参数解析器
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Grid-based scene rendering")

# 输入选项
parser.add_argument('--base_scene_blendfile', default='assets/base_scene_v5.blend')
parser.add_argument('--properties_json', default='assets/properties.json')
parser.add_argument('--shape_dir', default='assets/shapes_v5')
parser.add_argument('--material_dir', default='assets/materials_v5')

# 物体设置
parser.add_argument('--min_objects', default=4, type=int)
parser.add_argument('--max_objects', default=8, type=int)
parser.add_argument('--min_dist', default=0.25, type=float)
parser.add_argument('--margin', default=0.4, type=float)

# 网格设置
parser.add_argument('--grid_rows', default=4, type=int, help="Number of grid rows")
parser.add_argument('--grid_cols', default=4, type=int, help="Number of grid columns")
parser.add_argument('--cell_size', default=1.5, type=float, help="Grid cell size in BU")
parser.add_argument('--grid_line_width', default=0.02, type=float, help="Grid line width")
parser.add_argument('--grid_labels', default=0, type=int,
    help="Whether to render cell labels (0=off, 1=on)")

# 多视角设置
parser.add_argument('--n_views', default=4, type=int)
parser.add_argument('--camera_distance', default=12.0, type=float)
parser.add_argument('--elevation', default=30.0, type=float)
parser.add_argument('--azimuth_start', default=45.0, type=float)
parser.add_argument('--render_top_view', default=1, type=int)
parser.add_argument('--top_view_height', default=None, type=float)
parser.add_argument('--top_view_padding', default=0.5, type=float)

# 输出设置
parser.add_argument('--output_dir', default='../output/')
parser.add_argument('--start_idx', default=0, type=int)
parser.add_argument('--num_images', default=5, type=int)
parser.add_argument('--split', default='train')
parser.add_argument('--seed', default=42, type=int)

# 渲染设置
parser.add_argument('--use_gpu', default=0, type=int)
parser.add_argument('--width', default=480, type=int)
parser.add_argument('--height', default=480, type=int)
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
        print('  blender --background --python render_grid.py -- [args]')
