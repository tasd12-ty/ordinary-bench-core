"""
ORDINARY-BENCH 的 3D 网格渲染脚本。

将物体摆放在可见的 4×4×4 3D 网格中，从 6 个正交视角
（顶部、底部、前方、后方、左侧、右侧）渲染场景。

用法：
    blender --background --python render_grid3d.py -- [arguments]
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
            sys.exit(1)


# ---------------------------------------------------------------------------
# 相机配置
# ---------------------------------------------------------------------------

@dataclass
class CameraConfig:
    """单个相机视角的配置参数。"""
    camera_id: str
    azimuth: float
    elevation: float
    distance: float
    look_at: Tuple[float, float, float] = (0.0, 0.0, 3.0)

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


# 6 orthographic views looking at grid center (0, 0, 3.0)
ORTHO_VIEWS = [
    # (view_name, azimuth, elevation)
    ("top",    0.0,  90.0),
    ("bottom", 0.0, -90.0),
    ("front",  0.0,   0.0),   # camera at -Y, looking +Y
    ("back",   180.0, 0.0),   # camera at +Y, looking -Y
    ("left",   270.0, 0.0),   # camera at -X, looking +X
    ("right",  90.0,  0.0),   # camera at +X, looking -X
]


# ---------------------------------------------------------------------------
# 相机辅助函数
# ---------------------------------------------------------------------------

def get_object_by_name(name: str, alternative_names: Optional[List[str]] = None):
    """按名称查找 Blender 对象，支持备用名称回退。"""
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


# ---------------------------------------------------------------------------
# 3D Grid functions
# ---------------------------------------------------------------------------

ROW_LABELS = ['A', 'B', 'C', 'D']


def cell_center_3d(
    row: int, col: int, layer: int,
    rows: int, cols: int, layers: int,
    cell_size: float,
) -> Tuple[float, float, float]:
    """计算 3D 网格单元中心的世界坐标 (x, y, z)。"""
    grid_w = cols * cell_size
    grid_h = rows * cell_size
    x = (col + 0.5) * cell_size - grid_w / 2.0
    y = (row + 0.5) * cell_size - grid_h / 2.0
    z = (layer + 0.5) * cell_size  # Z starts from 0 upward
    return (x, y, z)


def cell_label_3d(row: int, col: int, layer: int) -> str:
    """返回可读的 3D 标签，如 A1-1、B3-4。"""
    return f"{ROW_LABELS[row]}{col + 1}-{layer + 1}"


def create_emission_material(name: str, color: Tuple[float, ...]) -> Any:
    """创建自发光材质（不受光照影响，始终可见）。"""
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    for node in list(nodes):
        if node.type != 'OUTPUT_MATERIAL':
            nodes.remove(node)
    emission = nodes.new('ShaderNodeEmission')
    emission.inputs['Color'].default_value = color
    emission.inputs['Strength'].default_value = 1.0
    output_node = None
    for node in nodes:
        if node.type == 'OUTPUT_MATERIAL':
            output_node = node
            break
    links.new(emission.outputs['Emission'], output_node.inputs['Surface'])
    return mat


def create_grid_3d(
    rows: int = 4, cols: int = 4, layers: int = 4,
    cell_size: float = 1.5, line_width: float = 0.02,
) -> List[Any]:
    """
    创建 3D 线框网格。

    沿 X、Y、Z 三个轴方向各生成一组线段，构成三维格栅。
    4×4×4 网格共计 3 * (N+1)^2 = 75 根线段。
    """
    grid_w = cols * cell_size    # X 方向总长
    grid_h = rows * cell_size    # Y 方向总长
    grid_d = layers * cell_size  # Z 方向总长
    half_w = grid_w / 2.0
    half_h = grid_h / 2.0
    z_base = 0.0  # grid starts at z=0

    grid_objects = []
    mat = create_emission_material("GridLine3DMaterial", (0.15, 0.15, 0.15, 1.0))

    def add_line(name, location, scale):
        if IS_BLENDER_280_OR_LATER:
            bpy.ops.mesh.primitive_cube_add(size=1, location=location)
        else:
            bpy.ops.mesh.primitive_cube_add(radius=0.5, location=location)
        line = bpy.context.object
        line.name = name
        line.scale = scale
        line.data.materials.append(mat)
        grid_objects.append(line)

    # X 方向线段：位于每个 (y 边界, z 边界) 的交叉处
    for j in range(rows + 1):
        y = j * cell_size - half_h
        for k in range(layers + 1):
            z = z_base + k * cell_size
            add_line(
                f"grid_x_{j}_{k}",
                (0, y, z),
                (grid_w, line_width, line_width),
            )

    # Y 方向线段：位于每个 (x 边界, z 边界) 的交叉处
    for i in range(cols + 1):
        x = i * cell_size - half_w
        for k in range(layers + 1):
            z = z_base + k * cell_size
            add_line(
                f"grid_y_{i}_{k}",
                (x, 0, z),
                (line_width, grid_h, line_width),
            )

    # Z 方向线段：位于每个 (x 边界, y 边界) 的交叉处
    for i in range(cols + 1):
        x = i * cell_size - half_w
        for j in range(rows + 1):
            y = j * cell_size - half_h
            z_center = z_base + grid_d / 2.0
            add_line(
                f"grid_z_{i}_{j}",
                (x, y, z_center),
                (line_width, line_width, grid_d),
            )

    return grid_objects


def create_grid_labels_3d(
    rows: int = 4, cols: int = 4, layers: int = 4,
    cell_size: float = 1.5,
) -> List[Any]:
    """在每个 3D 单元中心创建文字标签（如 A1-1、B3-4）。"""
    label_objects = []
    mat = create_emission_material("GridLabel3DMaterial", (0.3, 0.3, 0.3, 1.0))

    for r in range(rows):
        for c in range(cols):
            for la in range(layers):
                cx, cy, cz = cell_center_3d(r, c, la, rows, cols, layers, cell_size)
                label_text = cell_label_3d(r, c, la)

                bpy.ops.object.text_add(location=(cx, cy, cz))
                txt_obj = bpy.context.object
                txt_obj.name = f"grid_label_{label_text}"
                txt_obj.data.body = label_text
                txt_obj.data.size = 0.2
                txt_obj.data.align_x = 'CENTER'
                txt_obj.data.align_y = 'CENTER'
                txt_obj.data.materials.append(mat)
                label_objects.append(txt_obj)

    return label_objects


def hide_ground_plane() -> None:
    """从场景中移除地面平面，确保底部视角不受遮挡。"""
    for name in ['Ground', 'ground', 'Plane', 'Floor']:
        if name in bpy.data.objects:
            utils.delete_object(bpy.data.objects[name])


def add_grid_objects_3d(
    num_objects: int,
    args,
    camera,
    rows: int = 4, cols: int = 4, layers: int = 4,
    cell_size: float = 1.5,
) -> Tuple[List[Dict], List[Any]]:
    """将物体摆放在 3D 网格单元的中心位置。"""
    with open(args.properties_json, 'r') as f:
        properties = json.load(f)
        color_name_to_rgba = {}
        for name, rgb in properties['colors'].items():
            rgba = [float(c) / 255.0 for c in rgb] + [1.0]
            color_name_to_rgba[name] = rgba
        material_mapping = [(v, k) for k, v in properties['materials'].items()]
        object_mapping = [(v, k) for k, v in properties['shapes'].items()]

    size_scale = properties['sizes']['small']

    # 4×4×4 网格的全部 64 个单元
    all_cells = [
        (r, c, la)
        for r in range(rows)
        for c in range(cols)
        for la in range(layers)
    ]
    selected_cells = random.sample(all_cells, min(num_objects, len(all_cells)))

    objects_3d = []
    blender_objects = []

    for i, (r, c, la) in enumerate(selected_cells):
        cx, cy, cz = cell_center_3d(r, c, la, rows, cols, layers, cell_size)

        obj_name, obj_name_out = random.choice(object_mapping)
        color_name, rgba = random.choice(list(color_name_to_rgba.items()))
        mat_name, mat_name_out = random.choice(material_mapping)

        r_scale = size_scale
        if obj_name == 'Cube':
            r_scale /= math.sqrt(2)

        theta = 360.0 * random.random()

        utils.add_object(args.shape_dir, obj_name, r_scale, (cx, cy, cz), theta=theta)
        obj = bpy.context.object
        blender_objects.append(obj)

        utils.add_material(mat_name, Color=rgba)

        pixel_coords = utils.get_camera_coords(camera, obj.location)
        objects_3d.append({
            "id": f"obj_{i}",
            "shape": obj_name_out,
            "size": "small",
            "material": mat_name_out,
            "3d_coords": (cx, cy, cz),
            "rotation": theta,
            "pixel_coords": pixel_coords,
            "color": color_name,
            "cell_row": r,
            "cell_col": c,
            "cell_layer": la,
            "cell_label": cell_label_3d(r, c, la),
        })

    return objects_3d, blender_objects


# ---------------------------------------------------------------------------
# 正交视角渲染
# ---------------------------------------------------------------------------

def render_ortho_view(
    view_name: str,
    camera_config: CameraConfig,
    output_image: str,
    objects_3d: List[Dict],
    ortho_scale: float,
    args,
) -> Dict[str, Any]:
    """渲染单个正交视角。"""
    set_camera_position(camera_config)
    refresh_camera_state()

    camera = bpy.data.objects['Camera']

    # 切换到正交投影
    original_type = camera.data.type
    original_ortho_scale = getattr(camera.data, "ortho_scale", None)
    original_clip_end = camera.data.clip_end

    camera.data.type = 'ORTHO'
    camera.data.ortho_scale = ortho_scale
    camera.data.clip_start = 0.01
    camera.data.clip_end = max(original_clip_end, 50.0)
    refresh_camera_state()

    objects_with_pixels = compute_pixel_coords_for_view(camera, objects_3d)

    bpy.context.scene.render.filepath = output_image
    bpy.ops.render.render(write_still=True)

    view_data = {
        "view_id": view_name,
        "image_path": os.path.basename(output_image),
        "camera": camera_config.to_dict(),
        "projection": "orthographic",
        "ortho_scale": ortho_scale,
        "objects": objects_with_pixels,
    }

    # 恢复相机原始设置
    camera.data.type = original_type
    if original_ortho_scale is not None:
        camera.data.ortho_scale = original_ortho_scale
    camera.data.clip_end = original_clip_end

    return view_data


# ---------------------------------------------------------------------------
# 主场景渲染
# ---------------------------------------------------------------------------

def render_grid3d_scene(
    args,
    num_objects: int,
    output_index: int,
    output_split: str,
    output_dir: str,
) -> Dict[str, Any]:
    """从 6 个正交视角渲染完整的 3D 网格场景。"""
    os.makedirs(output_dir, exist_ok=True)

    bpy.ops.wm.open_mainfile(filepath=args.base_scene_blendfile)
    utils.load_materials(args.material_dir)

    # 渲染参数设置
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

    # 移除地面平面以适配 3D 渲染
    hide_ground_plane()

    # 场景元数据
    grid_center_z = (args.grid_layers * args.cell_size) / 2.0
    scene_id = f"{output_split}_{output_index:06d}"
    scene_struct = {
        "scene_id": scene_id,
        "split": output_split,
        "image_index": output_index,
        "n_objects": num_objects,
        "placement_mode": "grid3d",
        "grid_info": {
            "rows": args.grid_rows,
            "cols": args.grid_cols,
            "layers": args.grid_layers,
            "cell_size": args.cell_size,
            "grid_extent_xy": [
                -(args.grid_cols * args.cell_size) / 2.0,
                (args.grid_cols * args.cell_size) / 2.0,
            ],
            "grid_extent_z": [0.0, args.grid_layers * args.cell_size],
            "labels_visible": bool(args.grid_labels),
        },
        "objects": [],
        "world_constraints": {},
        "views": [],
    }

    # 灯光随机抖动
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

    # 创建 3D 线框网格
    create_grid_3d(
        rows=args.grid_rows,
        cols=args.grid_cols,
        layers=args.grid_layers,
        cell_size=args.cell_size,
        line_width=args.grid_line_width,
    )

    # 可选：添加单元格标签
    if args.grid_labels == 1:
        create_grid_labels_3d(
            rows=args.grid_rows,
            cols=args.grid_cols,
            layers=args.grid_layers,
            cell_size=args.cell_size,
        )

    # 设置初始相机位置以计算像素坐标
    camera = bpy.data.objects['Camera']
    initial_cam = CameraConfig(
        camera_id="init", azimuth=0.0, elevation=0.0,
        distance=15.0, look_at=(0.0, 0.0, grid_center_z),
    )
    set_camera_position(initial_cam)
    refresh_camera_state()

    # 摆放物体
    objects_3d, blender_objects = add_grid_objects_3d(
        num_objects, args, camera,
        rows=args.grid_rows,
        cols=args.grid_cols,
        layers=args.grid_layers,
        cell_size=args.cell_size,
    )
    scene_struct["objects"] = objects_3d

    # 渲染 6 个正交视角
    look_at = (0.0, 0.0, grid_center_z)
    for view_name, azimuth, elevation in ORTHO_VIEWS:
        cam_config = CameraConfig(
            camera_id=view_name,
            azimuth=azimuth,
            elevation=elevation,
            distance=15.0,
            look_at=look_at,
        )
        view_dir = os.path.join(output_dir, view_name)
        os.makedirs(view_dir, exist_ok=True)
        img_path = os.path.join(view_dir, f"{scene_id}.png")

        view_data = render_ortho_view(
            view_name, cam_config, img_path,
            objects_3d, args.ortho_scale, args,
        )
        scene_struct["views"].append(view_data)

    # 保存元数据
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(scene_struct, f, indent=2)

    return scene_struct


# ---------------------------------------------------------------------------
# 主程序入口
# ---------------------------------------------------------------------------

def main(args):
    """3D 网格渲染的主入口。"""
    random.seed(args.seed + args.start_idx)
    print(f"Starting 3D grid rendering: {args.num_images} scenes, "
          f"grid {args.grid_rows}x{args.grid_cols}x{args.grid_layers} "
          f"(seed={args.seed + args.start_idx})")

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    all_scenes = []
    successful = 0
    failed = 0

    for i in range(args.num_images):
        scene_id = f"{args.split}_{(i + args.start_idx):06d}"
        scene_output_dir = os.path.join(output_dir, "scenes_render", scene_id)

        num_objects = random.randint(args.min_objects, args.max_objects)

        try:
            scene_struct = render_grid3d_scene(
                args,
                num_objects=num_objects,
                output_index=(i + args.start_idx),
                output_split=args.split,
                output_dir=scene_output_dir,
            )
            all_scenes.append(scene_struct)
            successful += 1
            print(f"  [{successful}/{args.num_images}] Rendered {scene_id}")
        except Exception as e:
            print(f"  [ERROR] Failed to render {scene_id}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
            continue

    print(f"\nRendering complete: {successful} successful, {failed} failed")

    output_file = os.path.join(output_dir, f"{args.split}_scenes.json")
    output_data = {
        "info": {
            "date": dt.today().strftime("%Y-%m-%d"),
            "split": args.split,
            "n_views": 6,
            "view_names": [v[0] for v in ORTHO_VIEWS],
            "projection": "orthographic",
            "grid": {
                "rows": args.grid_rows,
                "cols": args.grid_cols,
                "layers": args.grid_layers,
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

parser = argparse.ArgumentParser(description="3D grid scene rendering")

# 输入选项
parser.add_argument('--base_scene_blendfile', default='assets/base_scene_v5.blend')
parser.add_argument('--properties_json', default='assets/properties.json')
parser.add_argument('--shape_dir', default='assets/shapes_v5')
parser.add_argument('--material_dir', default='assets/materials_v5')

# 物体设置
parser.add_argument('--min_objects', default=4, type=int)
parser.add_argument('--max_objects', default=12, type=int)

# 3D 网格设置
parser.add_argument('--grid_rows', default=4, type=int)
parser.add_argument('--grid_cols', default=4, type=int)
parser.add_argument('--grid_layers', default=4, type=int)
parser.add_argument('--cell_size', default=1.5, type=float)
parser.add_argument('--grid_line_width', default=0.02, type=float)
parser.add_argument('--grid_labels', default=0, type=int)
parser.add_argument('--ortho_scale', default=8.0, type=float)

# 渲染兼容设置
parser.add_argument('--render_top_view', default=0, type=int, help="unused, kept for compat")
parser.add_argument('--top_view_padding', default=0.5, type=float, help="unused, kept for compat")

# 输出设置
parser.add_argument('--output_dir', default='../output/')
parser.add_argument('--start_idx', default=0, type=int)
parser.add_argument('--num_images', default=5, type=int)
parser.add_argument('--split', default='train')
parser.add_argument('--seed', default=42, type=int)

# 渲染参数
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
        print('  blender --background --python render_grid3d.py -- [args]')
