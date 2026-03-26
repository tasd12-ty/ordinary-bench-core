# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

# 已针对 Blender 5.0 进行兼容性修改

import sys, random, os
import bpy, bpy_extras


"""
与 Blender 交互的工具函数集合
兼容 Blender 2.80+ / 4.x / 5.0
"""

# Blender 版本检查
BLENDER_VERSION = bpy.app.version
IS_BLENDER_280_OR_LATER = BLENDER_VERSION >= (2, 80, 0)


def extract_args(input_argv=None):
  """
  提取 "--" 之后的命令行参数。Blender 会忽略 -- 之后的命令行标志，
  因此此函数用于将 Blender 调用中的命令行参数转发到自定义脚本。
  """
  if input_argv is None:
    input_argv = sys.argv
  output_argv = []
  if '--' in input_argv:
    idx = input_argv.index('--')
    output_argv = input_argv[(idx + 1):]
  return output_argv


def parse_args(parser, argv=None):
  return parser.parse_args(extract_args(argv))


# 也许有更好的实现方式？
def delete_object(obj):
  """删除指定的 Blender 对象。"""
  if IS_BLENDER_280_OR_LATER:
    # Blender 2.80+ / 5.0
    for o in bpy.data.objects:
      o.select_set(False)
    obj.select_set(True)
    bpy.ops.object.delete()
  else:
    # Blender 2.79 及更早版本
    for o in bpy.data.objects:
      o.select = False
    obj.select = True
    bpy.ops.object.delete()


def get_camera_coords(cam, pos):
  """
  对于指定点，从相机视角获取其 3D 坐标和 2D 像素坐标。

  输入：
  - cam: 相机对象
  - pos: 给定 3D 世界坐标的向量

  返回元组：
  - (px, py, pz): px 和 py 为 2D 图像坐标；pz 为深度值，范围 [-1, 1]
  """
  scene = bpy.context.scene
  x, y, z = bpy_extras.object_utils.world_to_camera_view(scene, cam, pos)
  scale = scene.render.resolution_percentage / 100.0
  w = int(scale * scene.render.resolution_x)
  h = int(scale * scene.render.resolution_y)
  px = int(round(x * w))
  py = int(round(h - y * h))
  return (px, py, z)


def set_layer(obj, layer_idx):
  """
  将对象移动到指定图层。
  Blender 2.80+ 中图层已被集合替代。
  layer_idx 0 = 可见（Scene Collection），layer_idx > 0 = 隐藏集合
  """
  if IS_BLENDER_280_OR_LATER:
    # Blender 2.80+ / 5.0：使用集合替代图层
    # 从所有当前集合中取消链接
    for col in obj.users_collection:
      col.objects.unlink(obj)

    if layer_idx == 0:
      # 链接到场景集合（可见）
      bpy.context.scene.collection.objects.link(obj)
    else:
      # 创建或获取隐藏集合
      hidden_col_name = f"HiddenLayer_{layer_idx}"
      if hidden_col_name not in bpy.data.collections:
        hidden_col = bpy.data.collections.new(hidden_col_name)
        bpy.context.scene.collection.children.link(hidden_col)
        # 从视图层排除以隐藏
        bpy.context.view_layer.layer_collection.children[hidden_col_name].exclude = True
      else:
        hidden_col = bpy.data.collections[hidden_col_name]
      hidden_col.objects.link(obj)
  else:
    # Blender 2.79 及更早版本
    obj.layers[layer_idx] = True
    for i in range(len(obj.layers)):
      obj.layers[i] = (i == layer_idx)


def add_object(object_dir, name, scale, loc, theta=0):
  """
  从文件加载对象。假设 object_dir 目录下有名为 "$name.blend" 的文件，
  其中包含名为 "$name" 的单个对象，该对象尺寸为单位尺寸且以原点为中心。

  - scale: 场景中对象大小的标量
  - loc: (x, y) 元组，表示对象在地面平面上的坐标
  """
  # 先统计场景中同名对象数量，以便给新对象分配唯一名称
  count = 0
  for obj in bpy.data.objects:
    if obj.name.startswith(name):
      count += 1

  blend_path = os.path.join(object_dir, '%s.blend' % name)
  inner_path = os.path.join('Object', name)
  bpy.ops.wm.append(filepath=os.path.join(blend_path, inner_path),
                     directory=os.path.join(blend_path, 'Object'),
                     filename=name)

  # 重命名以避免冲突
  new_name = '%s_%d' % (name, count)
  bpy.data.objects[name].name = new_name

  # 将新对象设为活动对象，然后旋转、缩放和平移
  x, y = loc
  obj = bpy.data.objects[new_name]

  if IS_BLENDER_280_OR_LATER:
    # Blender 2.80+ / 5.0
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
  else:
    # Blender 2.79 及更早版本
    bpy.context.scene.objects.active = obj

  bpy.context.object.rotation_euler[2] = theta
  bpy.ops.transform.resize(value=(scale, scale, scale))
  # v5 形状的原点在底部，因此 z=0 可将其放置在地面上
  bpy.ops.transform.translate(value=(x, y, 0))


def load_materials(material_dir):
  """
  从目录加载材质。假设该目录包含每个文件一种材质的 .blend 文件。
  X.blend 文件包含名为 X 的单个 NodeTree 条目；
  该 NodeTree 条目必须有一个接受 RGBA 值的 "Color" 输入。
  """
  for fn in os.listdir(material_dir):
    if not fn.endswith('.blend'): continue
    name = os.path.splitext(fn)[0]
    blend_path = os.path.join(material_dir, fn)
    bpy.ops.wm.append(filepath=os.path.join(blend_path, 'NodeTree', name),
                       directory=os.path.join(blend_path, 'NodeTree'),
                       filename=name)


def add_material(name, **properties):
  """
  创建新材质并分配给活动对象。"name" 应为之前通过 load_materials 加载的材质名称。
  """
  # 统计场景中已有材质数量
  mat_count = len(bpy.data.materials)

  if IS_BLENDER_280_OR_LATER:
    # Blender 2.80+ / 5.0：使用 bpy.data.materials.new() 替代 bpy.ops.material.new()
    mat = bpy.data.materials.new(name='Material_%d' % mat_count)
    mat.use_nodes = True
  else:
    # Blender 2.79 及更早版本
    bpy.ops.material.new()
    mat = bpy.data.materials['Material']
    mat.name = 'Material_%d' % mat_count

  # 将新材质附加到活动对象
  # 确保对象尚无材质
  obj = bpy.context.active_object
  assert len(obj.data.materials) == 0
  obj.data.materials.append(mat)

  # 查找新材质的输出节点
  # Blender 5.0 中节点名称可能被本地化，因此使用类型匹配
  output_node = None
  for n in mat.node_tree.nodes:
    if n.type == 'OUTPUT_MATERIAL':
      output_node = n
      break

  if output_node is None:
    raise RuntimeError("Could not find Material Output node in material")

  # 向活动材质的节点树中添加新的 GroupNode，
  # 并将预加载节点组的节点树复制到新组节点中。
  # 该复制似乎是按值复制，因此可以创建多个同类型材质而不互相覆盖。
  group_node = mat.node_tree.nodes.new('ShaderNodeGroup')
  group_node.node_tree = bpy.data.node_groups[name]

  # 查找并设置新组节点的 "Color" 输入
  for inp in group_node.inputs:
    if inp.name in properties:
      inp.default_value = properties[inp.name]

  # 将新组节点的输出连接到 MaterialOutput 节点的输入
  mat.node_tree.links.new(
      group_node.outputs['Shader'],
      output_node.inputs['Surface'],
  )

