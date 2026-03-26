"""
Blender 动态场景渲染器（--background 模式）。

读取运动规划 JSON，一次性创建所有物体，然后逐帧更新位置并渲染。

用法：
    blender --background --python render_dynamic.py -- \
        --plan_json /path/to/plan.json \
        --base_scene_blendfile /path/to/base_scene_v5.blend \
        --shape_dir /path/to/shapes_v5 \
        --material_dir /path/to/materials_v5 \
        --properties_json /path/to/properties.json \
        --output_dir /path/to/output \
        --width 480 --height 320 --samples 128 \
        --camera_distance 12.0 --elevation 30.0 --azimuth 45.0

设计说明：位置公式内联实现（不导入 motion/ 包），以避免
Blender 内置 Python 的依赖问题。
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
    try:
        import utils
    except ImportError:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        if script_dir not in sys.path:
            sys.path.insert(0, script_dir)
        import utils


# ---------------------------------------------------------------------------
# 内联位置公式（镜像 motion/models.py，无需 numpy）
# ---------------------------------------------------------------------------

def get_position(obj_plan, t):
    """获取第 t 帧的 (x, y)。优先使用预计算位置，否则内联计算。"""
    if "positions" in obj_plan and t < len(obj_plan["positions"]):
        p = obj_plan["positions"][t]
        return (p[0], p[1])
    return _compute_position(obj_plan["motion"], t)


def get_velocity(obj_plan, t):
    """获取第 t 帧的 (vx, vy)。优先使用预计算速度，否则内联计算。"""
    if "velocities" in obj_plan and t < len(obj_plan["velocities"]):
        v = obj_plan["velocities"][t]
        return (v[0], v[1])
    return _compute_velocity(obj_plan["motion"], t)


def _compute_position(motion_dict, t):
    """回退方案：从序列化运动字典计算第 t 帧的 (x, y)。"""
    mtype = motion_dict["type"]
    if mtype == "static":
        return (motion_dict["x0"], motion_dict["y0"])
    elif mtype == "linear":
        return (
            motion_dict["x0"] + motion_dict["vx"] * t,
            motion_dict["y0"] + motion_dict["vy"] * t,
        )
    elif mtype == "circular":
        angle = motion_dict.get("phase0", 0.0) + motion_dict["omega"] * t
        return (
            motion_dict["cx"] + motion_dict["radius"] * math.cos(angle),
            motion_dict["cy"] + motion_dict["radius"] * math.sin(angle),
        )
    elif mtype == "accelerated_linear":
        return (
            motion_dict["x0"] + motion_dict["vx"] * t + 0.5 * motion_dict["ax"] * t * t,
            motion_dict["y0"] + motion_dict["vy"] * t + 0.5 * motion_dict["ay"] * t * t,
        )
    raise ValueError(f"Unknown motion type: {mtype}")


def _compute_velocity(motion_dict, t):
    """回退方案：计算第 t 帧的 (vx, vy)。"""
    mtype = motion_dict["type"]
    if mtype == "static":
        return (0.0, 0.0)
    elif mtype == "linear":
        return (motion_dict["vx"], motion_dict["vy"])
    elif mtype == "circular":
        angle = motion_dict.get("phase0", 0.0) + motion_dict["omega"] * t
        r = motion_dict["radius"]
        w = motion_dict["omega"]
        return (-r * w * math.sin(angle), r * w * math.cos(angle))
    elif mtype == "accelerated_linear":
        return (
            motion_dict["vx"] + motion_dict["ax"] * t,
            motion_dict["vy"] + motion_dict["ay"] * t,
        )
    return (0.0, 0.0)


# ---------------------------------------------------------------------------
# 相机设置（镜像 render_multiview.py 中的 CameraConfig）
# ---------------------------------------------------------------------------

def setup_camera(distance, elevation_deg, azimuth_deg, look_at=(0, 0, 0)):
    """将相机放置于球坐标位置，朝向目标点。"""
    az = math.radians(azimuth_deg)
    el = math.radians(elevation_deg)
    x = distance * math.cos(el) * math.cos(az) + look_at[0]
    y = distance * math.cos(el) * math.sin(az) + look_at[1]
    z = distance * math.sin(el) + look_at[2]

    camera = bpy.data.objects["Camera"]
    camera.location = (x, y, z)
    direction = Vector(look_at) - Vector((x, y, z))
    rot_quat = direction.to_track_quat("-Z", "Y")
    camera.rotation_euler = rot_quat.to_euler()
    return camera


# ---------------------------------------------------------------------------
# 主渲染逻辑
# ---------------------------------------------------------------------------

def main(args):
    # 加载规划
    with open(args.plan_json) as f:
        plan = json.load(f)

    objects_plan = plan["objects"]
    n_frames = plan["n_frames"]
    fps = plan.get("fps", 24)

    # 加载基础场景
    bpy.ops.wm.open_mainfile(filepath=args.base_scene_blendfile)

    # 加载材质
    utils.load_materials(args.material_dir)

    # 渲染设置
    render = bpy.context.scene.render
    render.engine = "CYCLES"
    render.resolution_x = args.width
    render.resolution_y = args.height
    render.resolution_percentage = 100
    bpy.context.scene.cycles.samples = args.samples
    bpy.context.scene.cycles.blur_glossy = 2.0
    bpy.context.scene.cycles.transparent_max_bounces = 8

    if args.use_gpu:
        cycles_prefs = bpy.context.preferences.addons["cycles"].preferences
        for compute_type in ["CUDA", "OPTIX", "HIP", "ONEAPI"]:
            try:
                cycles_prefs.compute_device_type = compute_type
                for device in cycles_prefs.devices:
                    device.use = True
                break
            except Exception:
                continue
        bpy.context.scene.cycles.device = "GPU"

    # 加载颜色 RGBA 映射
    with open(args.properties_json) as f:
        properties = json.load(f)
    color_name_to_rgba = {}
    for name, rgb in properties["colors"].items():
        color_name_to_rgba[name] = [c / 255.0 for c in rgb] + [1.0]
    # 形状名称 -> blend 文件名
    shape_mapping = properties["shapes"]
    material_mapping = properties["materials"]

    # 设置相机（检查规划是否含相机运动）
    camera_plan = plan.get("camera", None)
    camera = setup_camera(
        args.camera_distance, args.elevation, args.azimuth,
    )

    # 在第 0 帧位置创建所有物体
    blender_objects = []
    for obj_plan in objects_plan:
        x0, y0 = get_position(obj_plan, 0)

        blend_name = shape_mapping.get(obj_plan["shape"], obj_plan["shape"])
        size_radius = obj_plan["size_radius"]
        rotation_rad = math.radians(obj_plan["rotation"])

        utils.add_object(args.shape_dir, blend_name, size_radius, (x0, y0), theta=rotation_rad)
        obj = bpy.context.object
        blender_objects.append(obj)

        # 应用材质
        mat_blend_name = material_mapping.get(obj_plan["material"], obj_plan["material"])
        rgba = color_name_to_rgba.get(obj_plan["color"], [0.5, 0.5, 0.5, 1.0])
        utils.add_material(mat_blend_name, Color=rgba)

    # 准备输出目录
    frames_dir = os.path.join(args.output_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    # 逐帧渲染循环
    scene_frames = []
    for t in range(n_frames):
        try:
            # 第一遍：更新所有物体位置
            for idx, obj_plan in enumerate(objects_plan):
                x, y = get_position(obj_plan, t)
                blender_objects[idx].location.x = x
                blender_objects[idx].location.y = y

            # 若规划含逐帧相机参数则更新相机
            if camera_plan and "frames" in camera_plan:
                cf = camera_plan["frames"]
                if t < len(cf):
                    cp = cf[t]
                    camera = setup_camera(
                        cp.get("distance", args.camera_distance),
                        cp.get("elevation", args.elevation),
                        cp.get("azimuth", args.azimuth),
                        look_at=tuple(cp.get("look_at", (0, 0, 0))),
                    )

            # 所有位置设置完毕后统一更新场景
            bpy.context.view_layer.update()

            # 第二遍：收集帧数据
            frame_objects = []
            for idx, obj_plan in enumerate(objects_plan):
                x, y = get_position(obj_plan, t)
                vx, vy = get_velocity(obj_plan, t)
                bobj = blender_objects[idx]
                pixel_coords = utils.get_camera_coords(camera, bobj.location)

                frame_objects.append({
                    "id": obj_plan["obj_id"],
                    "shape": obj_plan["shape"],
                    "size": obj_plan["size"],
                    "material": obj_plan["material"],
                    "color": obj_plan["color"],
                    "3d_coords": [x, y, 0.0],
                    "pixel_coords": list(pixel_coords),
                    "velocity": [vx, vy, 0.0],
                })

            # 渲染帧
            frame_path = os.path.join(frames_dir, f"frame_{t:04d}.png")
            bpy.context.scene.render.filepath = frame_path
            bpy.ops.render.render(write_still=True)

            scene_frames.append({
                "frame_id": t,
                "timestamp": t / fps,
                "objects": frame_objects,
            })

            print(f"  Frame {t+1}/{n_frames} rendered")

        except Exception as e:
            print(f"  WARNING: Frame {t} failed: {e}")
            scene_frames.append({
                "frame_id": t,
                "timestamp": t / fps,
                "objects": [],
                "error": str(e),
            })
            continue

    # 构建时序场景 JSON
    scene_id = plan.get("scene_id", "unknown")
    temporal_scene = {
        "scene_id": scene_id,
        "scene_type": "dynamic",
        "n_objects": len(objects_plan),
        "n_frames": n_frames,
        "fps": fps,
        "duration_seconds": n_frames / fps,
        "objects": [
            {
                "id": op["obj_id"],
                "shape": op["shape"],
                "size": op["size"],
                "material": op["material"],
                "color": op["color"],
                "motion": op["motion"],
            }
            for op in objects_plan
        ],
        "frames": scene_frames,
        "events": [],
    }

    scene_json_path = os.path.join(args.output_dir, "temporal_scene.json")
    with open(scene_json_path, "w") as f:
        json.dump(temporal_scene, f, indent=2)

    print(f"Saved temporal scene: {scene_json_path}")
    print(f"Frames: {frames_dir}")


# ---------------------------------------------------------------------------
# 参数解析器
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Dynamic scene renderer for Blender")

parser.add_argument("--plan_json", required=True, help="运动规划 JSON 路径")
parser.add_argument("--base_scene_blendfile", default="assets/base_scene_v5.blend")
parser.add_argument("--properties_json", default="assets/properties.json")
parser.add_argument("--shape_dir", default="assets/shapes_v5")
parser.add_argument("--material_dir", default="assets/materials_v5")
parser.add_argument("--output_dir", required=True)
parser.add_argument("--width", type=int, default=480)
parser.add_argument("--height", type=int, default=320)
parser.add_argument("--samples", type=int, default=128)
parser.add_argument("--use_gpu", type=int, default=0)
parser.add_argument("--camera_distance", type=float, default=12.0)
parser.add_argument("--elevation", type=float, default=30.0)
parser.add_argument("--azimuth", type=float, default=45.0)


if __name__ == "__main__":
    if INSIDE_BLENDER:
        argv = utils.extract_args()
        args = parser.parse_args(argv)
        main(args)
    elif "--help" in sys.argv or "-h" in sys.argv:
        parser.print_help()
    else:
        print("Run from Blender:")
        print("  blender --background --python render_dynamic.py -- [args]")
