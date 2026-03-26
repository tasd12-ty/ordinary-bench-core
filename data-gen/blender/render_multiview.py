# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# 扩展以支持多视角渲染

"""
ORDINAL-SPATIAL 基准的多视角渲染脚本。

从多个相机视角渲染同一场景，用于多视角空间推理评测。

用法：
    blender --background --python render_multiview.py -- [arguments]

示例：
    blender --background --python render_multiview.py -- \
        --num_images 10 \
        --n_views 4 \
        --output_dir ../output/multiview/
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
        # 尝试将脚本目录加入 sys.path 后重试。
        script_dir = os.path.dirname(os.path.abspath(__file__))
        if script_dir not in sys.path:
            sys.path.insert(0, script_dir)
        try:
            import utils
        except ImportError:
            print("\nERROR: Cannot import utils.py")
            print(f"Tried adding script dir to sys.path: {script_dir}")
            print("Add image_generation to Blender's Python path.")
            sys.exit(1)


@dataclass
class CameraConfig:
    """单个相机视角的配置。"""
    camera_id: str
    azimuth: float      # 方位角（度），0 = +X 方向
    elevation: float    # 仰角（度），0 = 水平
    distance: float     # 距场景中心的距离
    look_at: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    def to_cartesian(self) -> Tuple[float, float, float]:
        """
        将球坐标转换为笛卡尔坐标。

        Returns:
            (x, y, z) 相机位置
        """
        azimuth_rad = math.radians(self.azimuth)
        elevation_rad = math.radians(self.elevation)

        x = self.distance * math.cos(elevation_rad) * math.cos(azimuth_rad)
        y = self.distance * math.cos(elevation_rad) * math.sin(azimuth_rad)
        z = self.distance * math.sin(elevation_rad)

        return (
            x + self.look_at[0],
            y + self.look_at[1],
            z + self.look_at[2]
        )

    def to_dict(self) -> Dict[str, Any]:
        """转换为 JSON 可序列化的字典。"""
        pos = self.to_cartesian()
        return {
            "camera_id": self.camera_id,
            "azimuth": self.azimuth,
            "elevation": self.elevation,
            "distance": self.distance,
            "position": list(pos),
            "look_at": list(self.look_at)
        }


@dataclass
class MultiViewConfig:
    """多视角渲染配置。"""
    n_views: int = 4
    camera_distance: float = 12.0
    elevation: float = 30.0
    azimuth_start: float = 45.0  # 起始方位角 45°，覆盖效果更好
    look_at: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    def generate_cameras(self) -> List[CameraConfig]:
        """
        为所有视角生成相机配置。

        Returns:
            CameraConfig 对象列表
        """
        cameras = []
        azimuth_step = 360.0 / self.n_views

        for i in range(self.n_views):
            azimuth = self.azimuth_start + i * azimuth_step
            # 归一化到 [0, 360)
            azimuth = azimuth % 360.0

            cameras.append(CameraConfig(
                camera_id=f"view_{i}",
                azimuth=azimuth,
                elevation=self.elevation,
                distance=self.camera_distance,
                look_at=self.look_at
            ))

        return cameras


def compute_top_view_ortho_scale(
    objects_3d: List[Dict],
    padding: float = 2.5,
) -> float:
    """计算能舒适容纳场景的正交投影缩放比例。"""
    if not objects_3d:
        return 10.0

    xs = [float(obj["3d_coords"][0]) for obj in objects_3d]
    ys = [float(obj["3d_coords"][1]) for obj in objects_3d]
    span_x = max(xs) - min(xs)
    span_y = max(ys) - min(ys)
    span = max(span_x, span_y)
    return max(6.0, span + 2.0 * padding)


def compute_top_view_frame(
    objects_3d: List[Dict],
    blender_objects: Optional[List[Any]] = None,
    padding: float = 0.35,
    aspect_ratio: float = 1.0,
) -> Tuple[Tuple[float, float, float], float]:
    """
    计算能容纳完整场景的居中正交帧参数。

    俯视图应以场景包围盒为中心而非世界原点，
    拟合宽度须考虑渲染宽高比。
    """
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
            center = (
                0.5 * (min_x + max_x),
                0.5 * (min_y + max_y),
                0.0,
            )
            width = max(
                span_x + 2.0 * padding,
                aspect_ratio * (span_y + 2.0 * padding),
            )
            return center, max(6.0, width)

    if not objects_3d:
        return (0.0, 0.0, 0.0), 10.0

    xs = [float(obj["3d_coords"][0]) for obj in objects_3d]
    ys = [float(obj["3d_coords"][1]) for obj in objects_3d]
    min_x = min(xs)
    max_x = max(xs)
    min_y = min(ys)
    max_y = max(ys)
    span_x = max_x - min_x
    span_y = max_y - min_y
    center = (
        0.5 * (min_x + max_x),
        0.5 * (min_y + max_y),
        0.0,
    )
    width = max(
        span_x + 2.0 * padding,
        aspect_ratio * (span_y + 2.0 * padding),
    )
    return center, max(6.0, width)


def compute_top_view_height(
    blender_objects: Optional[List[Any]] = None,
    min_clearance: float = 1.0,
) -> float:
    """将正交相机放置在最高物体正上方。"""
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
    """按名称获取 Blender 对象，支持备用名称回退。"""
    if name in bpy.data.objects:
        return bpy.data.objects[name]
    if alternative_names:
        for alt_name in alternative_names:
            if alt_name in bpy.data.objects:
                return bpy.data.objects[alt_name]
    raise KeyError(f"Object not found: {name}")


def set_camera_position(camera_config: CameraConfig) -> None:
    """
    设置 Blender 相机的位置和朝向。

    Args:
        camera_config: 包含位置参数的相机配置
    """
    camera = bpy.data.objects['Camera']
    position = camera_config.to_cartesian()
    look_at = camera_config.look_at

    # 设置位置
    camera.location = position

    # 计算方向并设置旋转
    direction = Vector(look_at) - Vector(position)
    rot_quat = direction.to_track_quat('-Z', 'Y')
    camera.rotation_euler = rot_quat.to_euler()


def refresh_camera_state() -> None:
    """
    在读取投影坐标前刷新 Blender 相机变换更新。

    若不刷新，渲染时已使用新视角，但 world_to_camera_view
    仍会在一步内看到旧相机矩阵，导致保存元数据中的逐视角
    pixel_coords 出错。
    """
    view_layer = getattr(bpy.context, "view_layer", None)
    if view_layer is not None:
        view_layer.update()


def compute_pixel_coords_for_view(camera, objects_3d: List[Dict]) -> List[Dict]:
    """
    从当前相机视角计算所有物体的像素坐标。

    Args:
        camera: Blender 相机对象
        objects_3d: 包含 3d_coords 的物体字典列表

    Returns:
        已更新 pixel_coords 的物体列表
    """
    updated_objects = []
    for obj in objects_3d:
        obj_copy = obj.copy()
        coords_3d = obj['3d_coords']
        pixel_coords = utils.get_camera_coords(camera, Vector(coords_3d))
        obj_copy['pixel_coords'] = pixel_coords
        updated_objects.append(obj_copy)
    return updated_objects


def compute_directions_for_view(camera) -> Dict[str, Tuple[float, float, float]]:
    """
    计算相对于当前相机视角的基本方向向量。

    Args:
        camera: Blender 相机对象

    Returns:
        方向名称到向量的映射字典
    """
    # 创建临时平面以获取地面法线
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

    # 删除临时平面
    utils.delete_object(plane)

    return {
        'behind': tuple(plane_behind),
        'front': tuple(-plane_behind),
        'left': tuple(plane_left),
        'right': tuple(-plane_left),
        'above': tuple(plane_up),
        'below': tuple(-plane_up)
    }


def render_single_view(
    camera_config: CameraConfig,
    output_image: str,
    objects_3d: List[Dict],
    args
) -> Dict[str, Any]:
    """
    从单个相机视角渲染场景。

    Args:
        camera_config: 相机配置
        output_image: 输出图像路径
        objects_3d: 包含 3D 坐标的物体列表
        args: 命令行参数

    Returns:
        视角元数据字典
    """
    # 设置相机位置
    set_camera_position(camera_config)
    refresh_camera_state()

    camera = bpy.data.objects['Camera']

    # 计算视角相关数据
    objects_with_pixels = compute_pixel_coords_for_view(camera, objects_3d)
    directions = compute_directions_for_view(camera)

    # 设置输出路径并渲染
    bpy.context.scene.render.filepath = output_image
    bpy.ops.render.render(write_still=True)

    # 构建视角元数据
    view_data = {
        "view_id": camera_config.camera_id,
        "image_path": os.path.basename(output_image),
        "camera": camera_config.to_dict(),
        "directions": directions,
        "objects": objects_with_pixels
    }

    return view_data


def render_top_view(
    output_image: str,
    objects_3d: List[Dict],
    blender_objects: Optional[List[Any]],
    args,
) -> Dict[str, Any]:
    """渲染额外的正交俯视图。"""
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
    top_center, ortho_scale = compute_top_view_frame(
        objects_3d,
        blender_objects=blender_objects,
        padding=getattr(args, "top_view_padding", 0.35),
        aspect_ratio=aspect_ratio,
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


def render_multiview_scene(
    args,
    num_objects: int,
    output_index: int,
    output_split: str,
    output_dir: str,
    mv_config: MultiViewConfig
) -> Dict[str, Any]:
    """
    渲染完整的多视角场景。

    Args:
        args: 命令行参数
        num_objects: 放置的物体数量
        output_index: 场景索引
        output_split: 数据集 split 名称
        output_dir: 本场景的输出目录
        mv_config: 多视角配置

    Returns:
        完整的场景元数据
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 加载基础场景
    bpy.ops.wm.open_mainfile(filepath=args.base_scene_blendfile)

    # 加载材质
    utils.load_materials(args.material_dir)

    # 设置渲染参数
    render_args = bpy.context.scene.render
    render_args.engine = "CYCLES"
    render_args.resolution_x = args.width
    render_args.resolution_y = args.height
    render_args.resolution_percentage = 100

    # GPU 设置
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

    # 初始化场景结构
    scene_id = f"{output_split}_{output_index:06d}"
    scene_struct = {
        "scene_id": scene_id,
        "split": output_split,
        "image_index": output_index,
        "n_objects": num_objects,
        "objects": [],
        "world_constraints": {},
        "views": []
    }

    # 为灯光添加随机抖动
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

    # 放置物体（使用第一个相机位置进行初始设置）
    cameras = mv_config.generate_cameras()
    set_camera_position(cameras[0])

    # 创建临时平面用于方向计算
    if IS_BLENDER_280_OR_LATER:
        bpy.ops.mesh.primitive_plane_add(size=10)
    else:
        bpy.ops.mesh.primitive_plane_add(radius=5)
    plane = bpy.context.object

    camera = bpy.data.objects['Camera']
    plane_normal = plane.data.vertices[0].normal

    if IS_BLENDER_280_OR_LATER:
        cam_behind = camera.matrix_world.to_quaternion() @ Vector((0, 0, -1))
        cam_left = camera.matrix_world.to_quaternion() @ Vector((-1, 0, 0))
    else:
        cam_behind = camera.matrix_world.to_quaternion() * Vector((0, 0, -1))
        cam_left = camera.matrix_world.to_quaternion() * Vector((-1, 0, 0))

    plane_behind = (cam_behind - cam_behind.project(plane_normal)).normalized()
    plane_left = (cam_left - cam_left.project(plane_normal)).normalized()

    # 存储物体放置所用的方向
    temp_directions = {
        'behind': tuple(plane_behind),
        'front': tuple(-plane_behind),
        'left': tuple(plane_left),
        'right': tuple(-plane_left),
    }

    utils.delete_object(plane)

    # 添加物体
    objects_3d, blender_objects = add_random_objects(
        temp_directions, num_objects, args, camera
    )

    scene_struct["objects"] = objects_3d

    # 从每个视角渲染
    for cam_config in cameras:
        img_path = os.path.join(output_dir, f"{cam_config.camera_id}.png")

        view_data = render_single_view(
            cam_config,
            img_path,
            objects_3d,
            args
        )

        scene_struct["views"].append(view_data)

    if getattr(args, "render_top_view", 0) == 1 and getattr(args, "output_top_view_dir", None):
        os.makedirs(args.output_top_view_dir, exist_ok=True)
        top_img_path = os.path.join(args.output_top_view_dir, f"{scene_id}.png")
        scene_struct["top_view"] = render_top_view(
            top_img_path,
            objects_3d,
            blender_objects,
            args,
        )

    # 保存场景元数据
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(scene_struct, f, indent=2)

    # 同时将单视角图像（view_0）渲染到 single_view 目录
    if args.output_single_view_dir:
        single_view_dir = args.output_single_view_dir
        os.makedirs(single_view_dir, exist_ok=True)
        single_img_path = os.path.join(single_view_dir, f"{scene_id}.png")

        # 复制 view_0 图像或重新渲染
        import shutil
        view0_path = os.path.join(output_dir, "view_0.png")
        if os.path.exists(view0_path):
            shutil.copy(view0_path, single_img_path)

    return scene_struct


def add_random_objects(directions, num_objects, args, camera, _retry_count=0):
    """
    向场景中添加随机物体。
    改编自 render_images.py，带有重试次数限制。

    放置区域随物体数量缩放，以适应密集场景。
    """
    MAX_RETRIES = 100  # 为密集场景增加重试次数
    if _retry_count >= MAX_RETRIES:
        raise RuntimeError(f"Failed to place objects after {MAX_RETRIES} attempts")

    # 根据物体数量缩放放置区域
    # 基础区域：6 个物体以内为 3.0
    # 更多物体时扩大以适应密度
    if num_objects <= 6:
        placement_range = 3.0
    elif num_objects <= 10:
        placement_range = 3.5
    else:
        # 11-15 个物体时使用更大区域
        placement_range = 4.0

    # 为密集场景调整间距
    effective_min_dist = args.min_dist
    effective_margin = args.margin
    if num_objects > 10:
        # 极密集场景降低间距要求
        effective_min_dist = max(0.15, args.min_dist * 0.7)
        effective_margin = max(0.25, args.margin * 0.7)

    # 加载属性
    with open(args.properties_json, 'r') as f:
        properties = json.load(f)
        color_name_to_rgba = {}
        for name, rgb in properties['colors'].items():
            rgba = [float(c) / 255.0 for c in rgb] + [1.0]
            color_name_to_rgba[name] = rgba
        material_mapping = [(v, k) for k, v in properties['materials'].items()]
        object_mapping = [(v, k) for k, v in properties['shapes'].items()]
        size_mapping = list(properties['sizes'].items())

    positions = []
    objects = []
    blender_objects = []

    for i in range(num_objects):
        size_name, r = random.choice(size_mapping)

        # 尝试放置物体
        num_tries = 0
        while True:
            num_tries += 1
            if num_tries > args.max_retries:
                for obj in blender_objects:
                    utils.delete_object(obj)
                return add_random_objects(
                    directions, num_objects, args, camera,
                    _retry_count=_retry_count + 1
                )

            x = random.uniform(-placement_range, placement_range)
            y = random.uniform(-placement_range, placement_range)

            # 检查距离
            dists_good = True
            margins_good = True
            for (xx, yy, rr) in positions:
                dx, dy = x - xx, y - yy
                dist = math.sqrt(dx * dx + dy * dy)
                if dist - r - rr < effective_min_dist:
                    dists_good = False
                    break
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

        # 随机选择颜色和形状
        obj_name, obj_name_out = random.choice(object_mapping)
        color_name, rgba = random.choice(list(color_name_to_rgba.items()))

        if obj_name == 'Cube':
            r /= math.sqrt(2)

        theta = 360.0 * random.random()

        # 添加物体
        utils.add_object(args.shape_dir, obj_name, r, (x, y), theta=theta)
        obj = bpy.context.object
        blender_objects.append(obj)
        positions.append((x, y, r))

        # 添加材质
        mat_name, mat_name_out = random.choice(material_mapping)
        utils.add_material(mat_name, Color=rgba)

        # 记录物体数据
        pixel_coords = utils.get_camera_coords(camera, obj.location)
        objects.append({
            "id": f"obj_{i}",
            "shape": obj_name_out,
            "size": size_name,
            "material": mat_name_out,
            "3d_coords": tuple(obj.location),
            "rotation": theta,
            "pixel_coords": pixel_coords,
            "color": color_name,
        })

    return objects, blender_objects


def main(args):
    """多视角渲染的主入口。"""
    # 可复现性种子：基础种子 + start_idx 确保不同增量批次
    # 生成不同但确定性的场景
    random.seed(args.seed + args.start_idx)
    print(f"Starting multi-view rendering: {args.num_images} scenes, {args.n_views} views each (seed={args.seed + args.start_idx})")

    # 创建多视角配置
    mv_config = MultiViewConfig(
        n_views=args.n_views,
        camera_distance=args.camera_distance,
        elevation=args.elevation,
        azimuth_start=args.azimuth_start
    )

    # 创建输出目录
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
            scene_struct = render_multiview_scene(
                args,
                num_objects=num_objects,
                output_index=(i + args.start_idx),
                output_split=args.split,
                output_dir=scene_output_dir,
                mv_config=mv_config
            )
            all_scenes.append(scene_struct)
            successful += 1
            print(f"  [{successful}/{args.num_images}] Rendered {scene_id}")

        except Exception as e:
            print(f"  [ERROR] Failed to render {scene_id}: {e}")
            failed += 1
            continue

    print(f"\nRendering complete: {successful} successful, {failed} failed")

    # 保存合并的场景文件
    output_file = os.path.join(args.output_dir, f"{args.split}_scenes.json")
    output_data = {
        "info": {
            "date": dt.today().strftime("%Y-%m-%d"),
            "split": args.split,
            "n_views": args.n_views,
            "render_top_view": bool(args.render_top_view),
            "camera_config": asdict(mv_config)
        },
        "scenes": all_scenes
    }

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"Saved scenes to {output_file}")


# 参数解析器
parser = argparse.ArgumentParser(description="Multi-view scene rendering")

# 输入选项
parser.add_argument('--base_scene_blendfile', default='assets/base_scene_v5.blend')
parser.add_argument('--properties_json', default='assets/properties.json')
parser.add_argument('--shape_dir', default='assets/shapes_v5')
parser.add_argument('--material_dir', default='assets/materials_v5')

# 物体设置
parser.add_argument('--min_objects', default=4, type=int)
parser.add_argument('--max_objects', default=10, type=int)
parser.add_argument('--min_dist', default=0.25, type=float)
parser.add_argument('--margin', default=0.4, type=float)
parser.add_argument('--max_retries', default=50, type=int)

# 多视角设置
parser.add_argument('--n_views', default=4, type=int,
    help="Number of camera viewpoints")
parser.add_argument('--camera_distance', default=12.0, type=float,
    help="Camera distance from scene center")
parser.add_argument('--elevation', default=30.0, type=float,
    help="Camera elevation angle in degrees")
parser.add_argument('--azimuth_start', default=45.0, type=float,
    help="Starting azimuth angle in degrees")
parser.add_argument('--render_top_view', default=0, type=int,
    help="Whether to render an extra top-down view into top_view/ (0 or 1)")
parser.add_argument('--top_view_height', default=None, type=float,
    help="Top-view camera height. Defaults to an auto-tight height above the tallest object")
parser.add_argument('--top_view_padding', default=0.35, type=float,
    help="Minimal safety margin for orthographic top-view framing")

# 输出设置
parser.add_argument('--output_dir', default='../output/multiview/',
    help="Output directory for rendered scenes")
parser.add_argument('--start_idx', default=0, type=int)
parser.add_argument('--num_images', default=5, type=int)
parser.add_argument('--split', default='train')
parser.add_argument('--seed', default=42, type=int,
    help="Random seed for reproducible scene generation")

# 渲染设置
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
        print('  blender --background --python render_multiview.py -- [args]')
        print()
        print('For help:')
        print('  python render_multiview.py --help')
