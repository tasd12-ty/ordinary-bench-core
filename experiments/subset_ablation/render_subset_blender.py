"""
Blender 内部脚本：从子集 scene JSON 重建并渲染场景。

支持单视角和多视角渲染。多视角时在同一 Blender 进程中渲染 4 个方位角，
避免重复启动 Blender 和重建场景。

用法 (单视角):
    blender --background --python render_subset_blender.py -- \
        --scene_json output/scenes/n10_000080__s0042.json \
        --output_image output/images/single_view/n10_000080__s0042.png \
        --base_scene ../../data-gen/blender/assets/base_scene_v5.blend \
        --properties_json ../../data-gen/blender/assets/properties.json \
        --shape_dir ../../data-gen/blender/assets/shapes_v5 \
        --material_dir ../../data-gen/blender/assets/materials_v5 \
        --width 480 --height 320 --samples 64

用法 (多视角):
    blender --background --python render_subset_blender.py -- \
        --scene_json output/scenes/n10_000080__s0042.json \
        --output_dir output/images/multi_view/n10_000080__s0042 \
        --multi_view \
        --base_scene ... --properties_json ... --shape_dir ... --material_dir ...
"""

from __future__ import print_function
import argparse
import json
import math
import os
import sys

INSIDE_BLENDER = True
try:
    import bpy
    import bpy_extras
    from mathutils import Vector
except ImportError:
    INSIDE_BLENDER = False

if INSIDE_BLENDER:
    # 加载 utils.py — 从 data-gen/blender/ 目录
    # 通过 --blender_utils_dir 参数或自动查找
    pass


def find_and_load_utils(args):
    """定位并加载 data-gen/blender/utils.py。需要 --blender_utils_dir 参数。"""
    if hasattr(args, "blender_utils_dir") and args.blender_utils_dir:
        utils_dir = args.blender_utils_dir
    else:
        # 从 properties_json 路径反推: .../assets/properties.json -> .../
        utils_dir = os.path.dirname(os.path.dirname(args.properties_json))

    if not os.path.isfile(os.path.join(utils_dir, "utils.py")):
        raise FileNotFoundError(
            f"utils.py not found in {utils_dir}. "
            f"Use --blender_utils_dir to specify the directory containing utils.py"
        )

    if utils_dir not in sys.path:
        sys.path.insert(0, utils_dir)

    import utils as blender_utils
    return blender_utils


def parse_args():
    """解析 -- 之后的命令行参数。"""
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_json", required=True, help="子集 scene JSON 路径")
    parser.add_argument("--output_image", default=None, help="输出图片路径 (单视角)")
    parser.add_argument("--output_dir", default=None, help="输出目录 (多视角)")
    parser.add_argument("--multi_view", action="store_true", help="渲染 4 视角")
    parser.add_argument("--base_scene", required=True, help="base_scene_v5.blend 路径")
    parser.add_argument("--properties_json", required=True, help="properties.json 路径")
    parser.add_argument("--shape_dir", required=True, help="shapes_v5 目录")
    parser.add_argument("--material_dir", required=True, help="materials_v5 目录")
    parser.add_argument("--blender_utils_dir", default=None, help="utils.py 所在目录")
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--use_gpu", type=int, default=0)
    # 相机参数（默认与原始渲染一致）
    parser.add_argument("--azimuth", type=float, default=45.0)
    parser.add_argument("--elevation", type=float, default=30.0)
    parser.add_argument("--camera_distance", type=float, default=12.0)
    return parser.parse_args(argv)


# 多视角默认方位角（与 data-gen 原始渲染一致）
MULTI_VIEW_AZIMUTHS = [45.0, 135.0, 225.0, 315.0]


def set_camera_position(camera, azimuth, elevation, distance, look_at=(0, 0, 0)):
    """设置相机位置（球坐标系，与原始 render_multiview.py 一致）。"""
    az_rad = math.radians(azimuth)
    el_rad = math.radians(elevation)

    x = distance * math.cos(el_rad) * math.cos(az_rad) + look_at[0]
    y = distance * math.cos(el_rad) * math.sin(az_rad) + look_at[1]
    z = distance * math.sin(el_rad) + look_at[2]

    camera.location = Vector((x, y, z))

    direction = Vector(look_at) - camera.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    camera.rotation_euler = rot_quat.to_euler()


def place_objects_from_json(scene_data, args, utils):
    """
    从 scene JSON 中放置物体到 Blender 场景。

    严格复制原始 render_multiview.py 的放置逻辑:
    - shape 映射: properties.json 中 "shapes" 字典的反向映射
    - size 映射: properties.json 中 "sizes" 字典
    - cube 特殊处理: r /= sqrt(2)
    - rotation: 直接传给 rotation_euler[2]（与原始行为一致）
    - 位置: 3d_coords[0], 3d_coords[1] 作为 (x, y)
    """
    # 加载属性映射
    with open(args.properties_json, "r") as f:
        properties = json.load(f)

    # 反向映射: "cube" -> "SmoothCube_v2"
    shape_map = {k: v for k, v in properties["shapes"].items()}
    size_map = {k: v for k, v in properties["sizes"].items()}
    mat_map = {k: v for k, v in properties["materials"].items()}

    color_to_rgba = {}
    for name, rgb in properties["colors"].items():
        rgba = [float(c) / 255.0 for c in rgb] + [1.0]
        color_to_rgba[name] = rgba

    blender_objects = []

    for obj_data in scene_data["objects"]:
        shape_name = obj_data["shape"]      # e.g. "cube"
        blend_name = shape_map[shape_name]  # e.g. "SmoothCube_v2"
        size_name = obj_data["size"]        # e.g. "large"
        r = size_map[size_name]             # e.g. 0.7

        # 注意: 原始 render_multiview.py:714 检查 `obj_name == 'Cube'`
        # 但 v5 shapes 的 obj_name 是 "SmoothCube_v2"，所以 sqrt(2) 调整
        # 实际上从未执行。此处不应用此调整以保持一致。

        x = obj_data["3d_coords"][0]
        y = obj_data["3d_coords"][1]
        theta = obj_data["rotation"]  # 度数，直接传入

        # 放置物体
        utils.add_object(args.shape_dir, blend_name, r, (x, y), theta=theta)
        obj = bpy.context.object
        blender_objects.append(obj)

        # 添加材质
        material_name = obj_data["material"]      # e.g. "rubber"
        mat_blend_name = mat_map[material_name]    # e.g. "Rubber"
        color_name = obj_data["color"]
        rgba = color_to_rgba[color_name]
        utils.add_material(mat_blend_name, Color=rgba)

    return blender_objects


def main():
    args = parse_args()

    if not INSIDE_BLENDER:
        print("ERROR: This script must run inside Blender.")
        print("Usage: blender --background --python render_subset_blender.py -- [args]")
        sys.exit(1)

    utils = find_and_load_utils(args)

    # 加载 scene JSON
    with open(args.scene_json, "r") as f:
        scene_data = json.load(f)

    # 如果 scene JSON 中有 camera 信息，使用它
    azimuth = args.azimuth
    elevation = args.elevation
    distance = args.camera_distance

    look_at = (0, 0, 0)
    if "camera" in scene_data:
        cam = scene_data["camera"]
        azimuth = cam.get("azimuth", azimuth)
        elevation = cam.get("elevation", elevation)
        distance = cam.get("distance", distance)
        la = cam.get("look_at", [0, 0, 0])
        look_at = tuple(la)

    # 加载基础场景
    bpy.ops.wm.open_mainfile(filepath=args.base_scene)

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
        try:
            cycles_prefs = bpy.context.preferences.addons['cycles'].preferences
            for compute_type in ['CUDA', 'OPTIX', 'HIP', 'ONEAPI']:
                try:
                    cycles_prefs.compute_device_type = compute_type
                    for device in cycles_prefs.devices:
                        device.use = True
                    break
                except Exception:
                    continue
            bpy.context.scene.cycles.device = 'GPU'
        except Exception:
            pass

    bpy.context.scene.cycles.samples = args.samples
    bpy.context.scene.cycles.blur_glossy = 2.0

    # 放置物体
    blender_objects = place_objects_from_json(scene_data, args, utils)

    camera = bpy.data.objects['Camera']

    if args.multi_view:
        # 多视角：在同一场景中渲染 4 个方位角
        out_dir = args.output_dir
        if not out_dir:
            # fallback: 从 output_image 推导目录
            out_dir = os.path.splitext(args.output_image)[0]
        os.makedirs(out_dir, exist_ok=True)

        for i, az in enumerate(MULTI_VIEW_AZIMUTHS):
            set_camera_position(camera, az, elevation, distance, look_at)
            out_path = os.path.join(out_dir, f"view_{i}.png")
            bpy.context.scene.render.filepath = out_path
            bpy.ops.render.render(write_still=True)

        print(f"Rendered {scene_data['scene_id']} -> {out_dir}/ "
              f"({len(blender_objects)} objects, {len(MULTI_VIEW_AZIMUTHS)} views)")
    else:
        # 单视角
        set_camera_position(camera, azimuth, elevation, distance, look_at)
        os.makedirs(os.path.dirname(args.output_image), exist_ok=True)
        bpy.context.scene.render.filepath = args.output_image
        bpy.ops.render.render(write_still=True)

        print(f"Rendered {scene_data['scene_id']} -> {args.output_image} "
              f"({len(blender_objects)} objects)")


if __name__ == "__main__":
    main()
