# 动态场景 VLM Benchmark 调研报告

> **项目**: ordinary-bench 动态场景扩展
> **日期**: 2026-03-11
> **目标**: 调研如何为 VLM 空间理解基准增加动态视频场景维度，聚焦**数据生成**和**测试评估**两大方向

---

## 目录

- [第一部分：生成 — 动态场景数据制造](#第一部分生成--动态场景数据制造)
  - [1.1 现有管线分析](#11-现有管线分析)
  - [1.2 动画方案调研](#12-动画方案调研)
  - [1.3 元数据扩展设计](#13-元数据扩展设计)
  - [1.4 参考 Benchmark 的数据生成方式](#14-参考-benchmark-的数据生成方式)
- [第二部分：测试 — VLM 时空理解评估](#第二部分测试--vlm-时空理解评估)
  - [2.1 现有 Benchmark 评测方法调研](#21-现有-benchmark-评测方法调研)
  - [2.2 问题类型设计](#22-问题类型设计)
  - [2.3 输入形式](#23-输入形式)
  - [2.4 评分方法设计](#24-评分方法设计)
  - [2.5 渐进式难度设计](#25-渐进式难度设计)
- [第三部分：DSL 扩展方案](#第三部分dsl-扩展方案)
  - [3.1 现有代码架构分析](#31-现有代码架构分析)
  - [3.2 时间维度扩展设计](#32-时间维度扩展设计)
  - [3.3 新约束类型设计](#33-新约束类型设计)
  - [3.4 评分系统改造](#34-评分系统改造)
  - [3.5 文件组织与实现路线图](#35-文件组织与实现路线图)
- [附录：参考文献](#附录参考文献)

---

# 第一部分：生成 — 动态场景数据制造

## 1.1 现有管线分析

### 1.1.1 当前 Blender 渲染管线能力

现有管线 (`data-gen/blender/render_multiview.py`) 的核心架构：

- **CameraConfig**：球坐标（方位角、仰角、距离）+ 笛卡尔坐标转换
- **MultiViewConfig**：等间距方位角分布的多视角配置
- **渲染流程**：`add_random_objects()` 放置静态对象 → `render_multiview_scene()` 对每个相机位置调用 `render_single_view()` → 保存 `metadata.json`

**核心限制**：

| 限制 | 说明 | 影响 |
|------|------|------|
| **完全静态** | `add_random_objects()` 只在初始化时设置位置，之后不再改变 | 无法生成运动场景 |
| **无时间循环** | 外层循环是"不同相机视角"，而非"不同时间步" | 需要新增帧循环 |
| **单帧渲染** | `bpy.ops.render.render(write_still=True)` 只渲染一帧 | 需要改为逐帧渲染 |
| **空 world_constraints** | `world_constraints` 字段存在但为空 `{}` | 可用于存储运动约束 |

### 1.1.2 静态场景 JSON 元数据结构

当前格式：
```json
{
  "scene_id": "n04_000000",
  "n_objects": 4,
  "objects": [
    {
      "id": "obj_0",
      "shape": "sphere",
      "size": "large",
      "material": "rubber",
      "3d_coords": [-1.14, 0.45, 0.0],
      "rotation": 105.1,
      "pixel_coords": [287, 150, 12.42],
      "color": "brown"
    }
  ],
  "views": [...]
}
```

**关键特点**：z 坐标通常为 0（平面场景），实际上是 2D 场景伪装成 3D。时间扩展需决定是否引入真正的高度变化。

### 1.1.3 扩展方向

扩展到时间维度的核心改动：
1. 在 `render_single_view()` 外层包裹**帧循环**
2. 为每帧每视角记录物体状态
3. 新增运动模型抽象层
4. 轨迹碰撞检测（确保运动中不穿透）

---

## 1.2 动画方案调研

### 1.2.1 关键帧动画（推荐首选）

通过 Blender Python API 在特定时间点设定物体的位置/旋转/缩放，自动插值中间帧。

```python
import bpy

obj = bpy.data.objects['Sphere']

# 第1帧：起始位置
bpy.context.scene.frame_set(1)
obj.location = (0, 0, 0)
obj.keyframe_insert(data_path="location", frame=1)

# 第60帧：终止位置
bpy.context.scene.frame_set(60)
obj.location = (5, 3, 0)
obj.keyframe_insert(data_path="location", frame=60)

# 插值类型设置（线性 / Bezier / 常量）
for fcurve in obj.animation_data.action.fcurves:
    for kfp in fcurve.keyframe_points:
        kfp.interpolation = 'LINEAR'
```

| 维度 | 优势 | 劣势 |
|------|------|------|
| 可控性 | 每帧物体状态精确已知，GT 零误差 | — |
| 复杂度 | 支持任意轨迹（直线、曲线、螺旋） | 复杂交互需手动编排 |
| 效率 | 无需物理计算，渲染快 | — |
| 真实性 | — | 不遵循牛顿力学 |
| 碰撞 | — | 碰撞检测/响应需手动处理 |

**适用场景**：匀速/匀加速直线运动、轨道运动、旋转运动、组合运动（先平移后旋转）。

### 1.2.2 物理模拟（刚体）

```python
obj = bpy.data.objects['Cube']
bpy.context.view_layer.objects.active = obj
bpy.ops.rigidbody.object_add(type='ACTIVE')
obj.rigid_body.mass = 1.0
obj.rigid_body.friction = 0.5
obj.rigid_body.restitution = 0.3  # 弹性系数

# 烘焙模拟以获取每帧状态
bpy.ops.ptcache.bake_all(bake=True)
```

| 维度 | 优势 | 劣势 |
|------|------|------|
| 真实性 | 物理真实的碰撞、弹跳、滚动 | — |
| 交互 | 复杂多体交互自动生成 | — |
| 可控性 | — | 结果不易精确控制，混沌性 |
| GT 获取 | — | 需要烘焙后才能提取状态 |

**适用场景**：碰撞后运动、物理推理 benchmark。

### 1.2.3 三种方案对比

| 方案 | GT 精确度 | 场景复杂度 | 实现难度 | 推荐阶段 |
|------|-----------|-----------|---------|---------|
| 关键帧动画 | ★★★ 零误差 | ★★ 中等 | ★ 低 | **Phase 1** |
| 刚体物理 | ★★ 烘焙后可得 | ★★★ 自动交互 | ★★ 中 | Phase 2 |
| Kubric 风格 | ★★★ 完整GT | ★★★ 高 | ★★★ 高 | Phase 3 |

### 1.2.4 多视角视频渲染配置

**方案 A：多 Camera 对象 + 逐相机渲染动画**（推荐）

```python
def create_multi_camera_rig(n_cameras=4, distance=12.0, elevation=30.0):
    """创建环绕相机阵列"""
    cameras = []
    for i in range(n_cameras):
        azimuth = (360.0 / n_cameras) * i + 45.0
        az_rad = math.radians(azimuth)
        el_rad = math.radians(elevation)
        x = distance * math.cos(el_rad) * math.cos(az_rad)
        y = distance * math.cos(el_rad) * math.sin(az_rad)
        z = distance * math.sin(el_rad)
        bpy.ops.object.camera_add(location=(x, y, z))
        cam = bpy.context.object
        cam.name = f"Camera_{i}"
        direction = Vector((0,0,0)) - Vector((x,y,z))
        cam.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()
        cameras.append(cam)
    return cameras

def render_synchronized_frames(cameras, output_dir, frame_start, frame_end):
    """每帧依次渲染所有相机视角（保证同一帧完全同步）"""
    scene = bpy.context.scene
    for frame in range(frame_start, frame_end + 1):
        scene.frame_set(frame)
        for cam in cameras:
            scene.camera = cam
            filepath = f"{output_dir}/{cam.name}/frame_{frame:04d}.png"
            scene.render.filepath = filepath
            bpy.ops.render.render(write_still=True)
```

### 1.2.5 帧率/时长/抽帧策略

| 参数 | 推荐值 | 理由 |
|------|--------|------|
| 帧率 | 24 fps | 标准电影帧率，VLM 输入通常下采样到 1-4 fps |
| 动画时长 | 2-5 秒 | 足够展示空间关系变化，避免冗余 |
| 总帧数 | 48-120 帧 | 24fps × 2-5s |
| VLM 采样 | 8-16 帧 | 参考 4D-Bench (6帧)、STI-Bench (30帧) |
| 渲染分辨率 | 480×320 | 与现有 ordinary-bench 保持一致 |

**采样策略**：
- **均匀采样**：每隔 N 帧取一帧 → 适合匀速运动
- **关键时刻采样**：在空间关系发生变化的帧采样 → 适合事件驱动场景
- **混合采样**：均匀帧 + 运动变化关键帧（推荐）

---

## 1.3 元数据扩展设计

### 1.3.1 扩展后的动态场景 JSON 格式

```json
{
  "scene_id": "dyn_000001",
  "scene_type": "dynamic",
  "n_objects": 4,

  "animation": {
    "fps": 24,
    "frame_start": 1,
    "frame_end": 120,
    "duration_seconds": 5.0,
    "motion_type": "keyframe"
  },

  "objects": [
    {
      "id": "obj_0",
      "shape": "sphere",
      "size": "large",
      "material": "rubber",
      "color": "brown",
      "initial_state": {
        "position": [-1.14, 0.45, 0.0],
        "rotation_euler": [0.0, 0.0, 1.83],
        "velocity": [2.0, 0.0, 0.0]
      }
    }
  ],

  "frames": [
    {
      "frame": 1,
      "timestamp": 0.0,
      "objects": [
        {
          "id": "obj_0",
          "position": [-1.14, 0.45, 0.0],
          "rotation_euler": [0.0, 0.0, 1.83],
          "velocity": [2.0, 0.0, 0.0],
          "acceleration": [0.0, 0.0, 0.0]
        }
      ]
    },
    {
      "frame": 24,
      "timestamp": 1.0,
      "objects": [
        {
          "id": "obj_0",
          "position": [0.86, 0.45, 0.0],
          "velocity": [1.8, 0.0, 0.0],
          "acceleration": [-0.2, 0.0, 0.0]
        }
      ]
    }
  ],

  "events": [
    {
      "type": "spatial_relation_change",
      "frame": 48,
      "timestamp": 2.0,
      "description": "obj_0 overtakes obj_1 in distance to obj_2",
      "participants": ["obj_0", "obj_1", "obj_2"]
    },
    {
      "type": "collision",
      "frame": 36,
      "timestamp": 1.5,
      "participants": ["obj_0", "obj_2"],
      "contact_point": [1.2, 0.8, 0.0]
    }
  ],

  "views": [
    {
      "view_id": "view_0",
      "camera": {
        "azimuth": 45.0,
        "elevation": 30.0,
        "distance": 12.0
      },
      "frame_dir": "view_0_frames/",
      "per_frame_pixel_coords": {
        "1": [{"id": "obj_0", "pixel_coords": [287, 150, 12.42], "visible": true}],
        "24": [{"id": "obj_0", "pixel_coords": [310, 160, 11.8], "visible": true}]
      }
    }
  ]
}
```

### 1.3.2 时间戳标注体系（三层）

```
Level 1: 帧级 (Frame-level)
  - frame_index: 整数帧号
  - timestamp: 浮点秒数 = (frame - start) / fps
  - 每帧完整物体状态（位置、速度、加速度）

Level 2: 事件级 (Event-level)
  - 碰撞事件：碰撞帧、参与物体、接触点
  - 空间关系变化事件：变化帧、旧关系、新关系
  - 运动状态变化：开始/停止/方向改变

Level 3: 区间级 (Interval-level)
  - 时间区间内的稳定空间关系
  - 例：[frame 1-47] obj_0 在 obj_1 左边
  - 例：[frame 48-120] obj_0 在 obj_1 右边
```

### 1.3.3 Ground Truth 自动生成

从 Blender 动画数据提取 GT 的核心函数：

```python
def extract_per_frame_state(objects, frame_start, frame_end, fps, sample_every=1):
    """从 Blender 场景提取逐帧物体状态"""
    frames_data = []
    for frame in range(frame_start, frame_end + 1, sample_every):
        bpy.context.scene.frame_set(frame)
        bpy.context.view_layer.update()
        frame_state = {
            "frame": frame,
            "timestamp": (frame - frame_start) / fps,
            "objects": []
        }
        for obj in objects:
            bl_obj = bpy.data.objects[obj["blender_name"]]
            pos = bl_obj.matrix_world.translation
            velocity = estimate_velocity(bl_obj, frame, fps)     # 中心差分
            acceleration = estimate_acceleration(bl_obj, frame, fps)  # 二阶差分
            frame_state["objects"].append({
                "id": obj["id"],
                "position": [pos.x, pos.y, pos.z],
                "velocity": velocity,
                "acceleration": acceleration
            })
        frames_data.append(frame_state)
    return frames_data
```

速度通过中心差分 `v = (pos(t+1) - pos(t-1)) / (2dt)` 估算，加速度通过二阶差分 `a = (pos(t+1) - 2*pos(t) + pos(t-1)) / dt²` 估算。关键帧动画方案下速度/加速度可从插值函数精确计算，误差为零。

### 1.3.4 事件自动检测

```python
def detect_spatial_relation_changes(frames_data, objects):
    """自动检测空间关系变化事件"""
    events = []
    for i, j in combinations(range(len(objects)), 2):
        prev_relation = None
        for frame_data in frames_data:
            # 计算当前空间关系（距离排序、方位等）
            relation = compute_spatial_relation(frame_data, objects[i], objects[j])
            if prev_relation and relation != prev_relation:
                events.append({
                    "type": "spatial_relation_change",
                    "frame": frame_data["frame"],
                    "participants": [objects[i]["id"], objects[j]["id"]],
                    "old_relation": prev_relation,
                    "new_relation": relation
                })
            prev_relation = relation
    return events
```

---

## 1.4 参考 Benchmark 的数据生成方式

### 1.4.1 4D-Bench (ICCV 2025)

**数据来源**：从 Objaverse-XL（超过 1000 万个 3D 模型）中筛选动态 3D 物体，渲染为多视角视频。

**数据生成流水线**：
1. **运动分析**：像素变化检测识别物体运动的时间边界，提取动态片段
2. **视觉质量评估**：手动标注数千张图片 → 微调 CLIP 图像编码器作为质量分类器
3. **问答生成**：GPT-4o + Qwen2-VL 通过 CoT 推理生成 QA → 盲测过滤（确保问题需要视觉理解）→ 人工审核

**规模**：751 QA 对，580+ 物体类别，K=3 视角 × N=6 帧/视角

**启示**：Objaverse-XL 是极佳的 3D 资产来源；CLIP 质量过滤和盲测过滤思路值得借鉴。

### 1.4.2 IR3D-Bench

**数据生成**：基于 CLEVR 数据集（3-10 个几何基元），VLM 代理通过 MCP 协议生成可执行 Blender Python 脚本。

**真值格式**（JSON）：
```json
{
  "camera": {"location": [0.0, -10.0, 5.0], "lens": 50.0},
  "objects": [
    {"name": "red large metal sphere", "location": [X, Y, Z],
     "size_params": {...}, "material": {"base_color": [R,G,B,A]}}
  ]
}
```

**规模**：15,000 验证图像 → 筛选 1,500 场景对

**启示**：JSON 结构化场景表示与我们的格式非常接近；MCP 协议连接 VLM 与 Blender 是新颖的评测范式。

### 1.4.3 Spatial457

**渲染引擎**：Unreal Engine 4 + HDRI 环境贴图

**5级难度设计**：

| 等级 | 内容 | 新增能力 |
|------|------|---------|
| L1 | 单物体识别 | 基础识别 |
| L2 | 多物体 + 属性匹配/计数 | 多目标处理 |
| L3 | 2D 空间关系（左右前后） | 相机视角空间推理 |
| L4 | 遮挡 + 3D 姿态 | 3D 理解 |
| L5 | 6D 空间推理 + **碰撞预测** | 运动推断 |

**规模**：1,000 张图像，约 28,436 个问题

**启示**：L5 的碰撞预测已涉及动态推理，可作为从静态到动态的桥梁；RPDR 指标适合评估"模型在哪个维度上最弱"。

### 1.4.4 PhysBench (ICLR 2025)

**数据特征**：10,002 条交织的 video-image-text 数据，覆盖四大物理领域：
1. 物体固有属性理解
2. 物体间交互与空间关系
3. 环境整体理解
4. **物理驱动的动力学**（运动、力和时间变化）

**启示**：「物理驱动的动力学」与我们的动态场景方向直接相关；真实+合成混合数据策略值得考虑。

### 1.4.5 补充参考：CLEVRER 与 Kubric

**CLEVRER** (ICLR 2020)：Blender + Bullet Physics 联合生成碰撞/运动视频。任务类型包括描述性、解释性、预测性、反事实问题。证明了 Blender+物理引擎生成物理推理视频基准的可行性。

**Kubric** (Google Research)：Blender（渲染）+ PyBullet（物理模拟）。输出完整真值标注：实例分割掩码、深度图、光流。Apache 2.0 许可，模块化设计。

### 1.4.6 渲染时间估算

| 配置 | 分辨率 | 采样数 | 单帧时间 | 适用 |
|------|--------|--------|---------|------|
| 快速测试 | 240×160 | 64 | ~2s | 调试 |
| 标准（推荐） | 480×320 | 128 | ~8s | 发布 |
| 高质量 | 640×480 | 256 | ~20s | 论文 |

标准配置下：120帧 × 4视角 = 480 帧 × ~8s ≈ 1小时/场景。100 个场景 ≈ 100 GPU 小时。使用 EEVEE 引擎可降至 <1s/帧。

---

# 第二部分：测试 — VLM 时空理解评估

## 2.1 现有 Benchmark 评测方法调研

### 2.1.1 总览

| 基准 | 年份 | 数据量 | 输入 | 答案格式 | 核心维度 | 最优模型 |
|------|------|--------|------|----------|----------|----------|
| VLM4D | 2025 | 1.8K QA | 视频 | MCQ | 平移/旋转/计数/假阳性 | Gemini-2.5-Pro 62% |
| STI-Bench | 2025 | 2K QA | 30帧 | MCQ(5选1) | 量化空间/位移/速度/轨迹 | Qwen2.5-VL 41.3% |
| MotionBench | 2025 | 8K QA | 8-64帧 | MCQ(4选1) | 6类运动细粒度 | — |
| 4D-Bench | 2025 | 751 QA | 多视角视频 | MCQ+描述 | 4D物体理解 | GPT-4o 63% |
| Spatial4D | 2025 | 40K QA | 64帧 | MCQ+数值 | 18任务6类别 | — |
| MVBench | 2024 | 4K | 视频帧 | MCQ(3-5选) | 20时序任务 | — |
| TempCompass | 2024 | 7.5K | 视频 | MCQ/YN/匹配/生成 | 速度/方向/属性/顺序 | — |

**关键发现**：人类基线 ~90-98%，当前最优模型仅 40-63%，VLM 在时空理解上的差距巨大。

### 2.1.2 VLM4D — 时空感知意识 (ICCV 2025)

**问题类别**：

| 类别 | 测试能力 | 示例 |
|------|----------|------|
| 平移运动 (TM) | 线性运动追踪，方向感知 | "物体A在向什么方向移动？" |
| 旋转运动 (RM) | 朝向变化，区分物体自转vs相机运动 | "物体在顺时针还是逆时针旋转？" |
| 时空计数 (STM) | 多物体运动整合计数 | 同时追踪多个运动物体 |
| 假阳性检测 (FP) | 批判性思维，事件辨别 | 识别不存在的运动事件（~10%） |

**评分方式**：大模型充当评委（GPT-o3/o4-mini 评估完整推理过程）。人类基线 98.8%。

### 2.1.3 STI-Bench — 精确时空量化推理 (ICCV 2025)

**与 ordinary-bench 最相关的基准**。动态理解任务包括：

| 任务 | 测试能力 | 示例 |
|------|----------|------|
| 位移与路径长度 | 帧间追踪，运动积分 | "车辆从1s到18s行驶了多远？" |
| 速度与加速度 | 空间导数，尺度一致性 | "相机的平均速度是多少？" |
| 自我中心朝向 | 旋转表征，角度参考 | "相机水平朝向偏转了多少度？" |
| 轨迹描述 | 轨迹分段，运动抽象化 | "直行30m，左转85度，直行20m" |

**真值计算方法**（极具参考价值）：
```
位移: d = √[(xn-x0)² + (yn-y0)² + (zn-z0)²]
路径长度: L = Σ √[(xi-xi-1)² + (yi-yi-1)² + (zi-zi-1)²]
速度: v = d / Δt
加速度: a = (vi - vi-1) / Δt
轨迹简化: Ramer-Douglas-Peucker 算法
```

输入：均匀采样 **30 帧/视频**。精度要求分场景：桌面=毫米级，室内=厘米级，室外=米级。

### 2.1.4 MotionBench — 细粒度运动理解 (CVPR 2025)

**六大运动类别**：运动识别(MR)、位置相关运动(LM)、动作顺序(AO)、重复计数(RC)、运动相关物体(MO)、相机运动(CM)。

**关键特点**：标注密度 68.4 词/秒（为现有基准 2 倍），8,052 QA 对。

### 2.1.5 其他重要基准

- **Spatial4D-Bench**：引入 **平均相对准确率 (MRA)** 指标 — `MRA = (1/|C|) × Σ 1(|ŷ-y|/y < 1-θ)`，多阈值评估比单一准确率更信息丰富，对数值型时空问题极有参考价值
- **MVBench**：**"静态到动态"方法论** — 系统地将静态图像理解任务转换为时间推理任务，完美适配我们的 QRR/TRR 扩展
- **TempCompass**：**冲突视频对设计** — 翻转/拼接创造对抗性测试，防止模型利用单帧偏差或语言先验

---

## 2.2 问题类型设计

基于 ordinary-bench 现有的 QRR（距离排序关系）和 TRR（钟表方位关系）框架，设计四个问题维度。

### 2.2.1 维度一：时序空间关系 (Temporal QRR/TRR)

**核心思想**：将静态 QRR/TRR 约束扩展到多个时间点，测试 VLM 对空间关系随时间变化的理解。

#### 时序距离比较 (Temporal QRR)

**问题模板 A — 跨时间距离变化**：
> 在视频开始时，物体A和物体B之间的距离 与 视频结束时的距离相比，哪个更大？
> (a) 开始时更大 (b) 结束时更大 (c) 大致相等

**问题模板 B — 距离关系反转检测**：
> 视频开始时，A 比 B 更靠近 C。这个关系在视频中是否发生了反转？
> (a) 是，A 变得比 B 更远离 C (b) 否，A 始终更近 (c) 距离变得大致相等

**真值计算**：
```python
d_AB_t1 = dist(pos_A(t1), pos_B(t1))
d_AB_t2 = dist(pos_A(t2), pos_B(t2))
result = compare(d_AB_t1, d_AB_t2, tau=0.10)  # <, ~=, >
```

#### 时序方位变化 (Temporal TRR)

**问题模板 A — 方位变化方向**：
> 视频开始时，从物体A看，物体B大约在3点钟方向。到视频结束时，B 移动到了什么方向？
> (a) 12点钟 (b) 6点钟 (c) 9点钟 (d) 仍在3点钟

**问题模板 B — 方位变化幅度**：
> 物体B相对于物体A的钟表方位变化了大约多少？
> (a) 约90度 (b) 约180度 (c) 约270度 (d) 几乎没有变化

### 2.2.2 维度二：速度/加速度判断

**参考**：STI-Bench 的速度任务 + TempCompass 的速度维度。

#### 相对速度比较

**问题模板**：
> 红色球体和蓝色方块都在运动。哪个物体移动得更快？
> (a) 红色球体更快 (b) 蓝色方块更快 (c) 速度大致相同

**真值计算**：`v_obj = path_length(obj, t1, t2) / (t2 - t1)`，使用 `tau_velocity=0.15` 比较。

#### 加速/减速检测

**问题模板**：
> 红色球体在视频过程中是在加速、减速还是匀速运动？
> (a) 加速 (b) 减速 (c) 大致匀速 (d) 先加速后减速

#### 速度排序

**问题模板**：
> 将三个物体按运动速度从快到慢排列：红色球体、蓝色方块、绿色圆柱
> (a) 红>蓝>绿 (b) 蓝>红>绿 (c) 绿>蓝>红 (d) 红>绿>蓝

### 2.2.3 维度三：轨迹/方向预测

**参考**：VLM4D 的运动方向检测 + MVBench 的移动方向任务。

#### 运动方向判断

> 在视频的中间时刻，红色球体正在向什么方向移动？
> (a) 向左 (b) 向右 (c) 向上 (d) 向观察者靠近

#### 轨迹形状识别

> 红色球体在视频中的运动轨迹最接近什么形状？
> (a) 直线 (b) 圆弧 (c) 之字形 (d) 抛物线

#### 碰撞/交汇预测

> 根据红色球体和蓝色方块当前的运动趋势，它们的路径是否会交叉？
> (a) 会，它们正在相互靠近 (b) 不会，它们在远离 (c) 轨迹平行

#### 最近距离时刻

> 红色球体和蓝色方块在什么时段距离最近？
> (a) 开始阶段 (b) 中间阶段 (c) 结束阶段 (d) 距离始终不变

### 2.2.4 维度四：多视角一致性

**参考**：4D-Bench + Spatial4D-Bench 的多视角设计。

#### 跨视角运动一致性

> 视频1从正面拍摄物体A向右移动。视频2从上方俯视同一场景。在视频2中，物体A应该向什么方向移动？
> (a) 向下 (b) 向上 (c) 向右 (d) 向左

#### 跨视角空间推理

> 从视角1看，A 在 B 的左边并向右移动。从视角2（与视角1成90度角）看，A 应该在 B 的什么位置？
> (a) 前方，朝观察者移动 (b) 后方，远离观察者 (c) 前方，向右移动

---

## 2.3 输入形式

### 2.3.1 帧序列提取策略

| 阶段 | 采样方法 | 帧数 | 说明 |
|------|----------|------|------|
| Phase 1 | 均匀采样 + 时间戳 | 8-16 | "Frame 1 (t=0.0s), Frame 2 (t=0.5s), ..." |
| Phase 2 | 均匀 + 关键帧 | 16-32 | 在运动方向变化点额外采样 |
| Phase 3 | 事件帧 + 均匀背景帧 | 32-64 | 关键事件帧 + 均匀采样混合 |

**关键帧检测算法**：
```python
def extract_keyframes(positions, n_uniform=16, n_key=4):
    """混合采样：均匀帧 + 运动变化关键帧"""
    uniform_indices = np.linspace(0, len(positions)-1, n_uniform, dtype=int)
    # 检测运动方向突变点（速度向量夹角变化）
    velocities = np.diff(positions, axis=0)
    angle_changes = []
    for i in range(1, len(velocities)):
        cos_angle = np.dot(velocities[i], velocities[i-1]) / (
            np.linalg.norm(velocities[i]) * np.linalg.norm(velocities[i-1]) + 1e-8)
        angle_changes.append((i, np.arccos(np.clip(cos_angle, -1, 1))))
    key_indices = [k[0] for k in sorted(angle_changes, key=lambda x: -x[1])[:n_key]]
    return sorted(set(uniform_indices.tolist() + key_indices))
```

### 2.3.2 视频输入 vs 帧序列输入

| 方面 | 视频输入 | 帧序列输入 |
|------|----------|-----------|
| 时间连续性 | 模型自行感知 | 需显式提供时间戳 |
| 运动平滑性 | 自然的运动插值 | 可能丢失关键运动细节 |
| 物体追踪 | 模型需自行关联 | 可通过标注辅助追踪 |
| 上下文消耗 | 视频 token 更紧凑 | 多图像 token 消耗大 |
| 适用模型 | Gemini, GPT-4o, Qwen-VL | 所有 VLM（通用性强） |

### 2.3.3 提示词模板

**帧序列模式**（推荐作为通用模式）：
```
以下 {n_frames} 张图片是从一段视频中等时间间隔提取的帧。
视频总时长：{duration} 秒。

各帧时间戳：
- 图片 1: t = 0.0s
- 图片 2: t = {interval}s
- ...
- 图片 {n}: t = {duration}s

{object_legend}  （物体颜色/形状标注）

问题：{question_text}
选项：(a) ... (b) ... (c) ... (d) ...

请分析各帧之间物体空间关系的变化，选择正确答案。
```

**视频模式**：
```
请仔细观看以下视频。视频展示了多个物体在 {duration} 秒内的运动过程，
帧率为 {fps} FPS。

问题：{question_text}
选项：(a) ... (b) ... (c) ... (d) ...

请选择正确答案。
```

---

## 2.4 评分方法设计

### 2.4.1 扩展现有 scoring.py 框架

沿用 ordinary-bench 现有的逐题评分框架（`correct`/`hour_correct`/`quadrant_correct`），扩展到时序维度：

```python
def score_temporal_qrr(prediction, ground_truth, tau=0.10):
    """时序距离比较评分 — 精确匹配 + 变化方向部分分"""
    correct = (prediction == ground_truth)
    direction_correct = (get_change_direction(prediction) == get_change_direction(ground_truth))
    partial_score = 1.0 if correct else (0.5 if direction_correct else 0.0)
    return {'correct': correct, 'direction_correct': direction_correct, 'partial_score': partial_score}

def score_temporal_trr(prediction, ground_truth, hour_tolerance=1):
    """时序方位变化评分 — 三粒度 + 旋转方向"""
    correct = (prediction == ground_truth)
    hour_correct = abs(prediction - ground_truth) <= hour_tolerance or \
                   abs(prediction - ground_truth) >= (12 - hour_tolerance)
    quadrant_correct = (prediction // 3) == (ground_truth // 3)
    return {'correct': correct, 'hour_correct': hour_correct, 'quadrant_correct': quadrant_correct}

def score_velocity(prediction, ground_truth, question_type):
    """速度/加速度判断评分"""
    if question_type == 'speed_ordering':
        from scipy.stats import kendalltau
        tau, _ = kendalltau(prediction, ground_truth)
        return {'correct': prediction == ground_truth, 'ordering_score': (tau + 1) / 2}
    elif question_type == 'acceleration_detection':
        correct = (prediction == ground_truth)
        direction_match = (sign(prediction) == sign(ground_truth))
        return {'correct': correct, 'partial_score': 1.0 if correct else (0.3 if direction_match else 0.0)}
    else:
        return {'correct': prediction == ground_truth}
```

### 2.4.2 容忍度参数设计

| 维度 | 参数 | 容忍度 | 依据 |
|------|------|--------|------|
| 时序 QRR | tau | 0.10 (沿用静态) | 距离比较的相对容忍度 |
| 时序 TRR | hour_tolerance | ±1 小时 | 30度方位容忍 |
| 速度比较 | tau_velocity | 0.15 | 速度估计更困难，放宽 |
| 加速度 | — | 方向对即给分 | 量级难精确判断 |
| 方向判断 | adjacent | ±45度 | 8方向中邻近方向 |
| 数值型 | MRA 阈值集 | {0.5,...,0.95} | 借鉴 Spatial4D-Bench |

### 2.4.3 综合评估指标

```python
class DynamicSceneBenchmarkMetrics:
    def compute_all_metrics(self, results):
        metrics = {}
        # 1. 总体准确率
        metrics['overall_accuracy'] = compute_accuracy(results)
        # 2. 各维度准确率
        for dim in ['temporal_qrr', 'temporal_trr', 'velocity', 'trajectory', 'multiview']:
            metrics[f'{dim}_accuracy'] = compute_accuracy(filter_by(results, dim))
        # 3. 时序一致性分数（同一场景不同时间点回答的逻辑一致性）
        metrics['temporal_consistency'] = compute_temporal_consistency(results)
        # 4. 部分得分平均
        metrics['mean_partial_score'] = mean([r['partial_score'] for r in results])
        # 5. 各难度级别准确率
        for phase in [1, 2, 3]:
            metrics[f'phase_{phase}_accuracy'] = compute_accuracy(filter_by_phase(results, phase))
        return metrics
```

**时序一致性分数**：若 d(A,B) 在 t1 < t2 < t3 三个时刻都被问到，回答应满足逻辑单调性（若物体持续靠近，则距离应单调递减）。

---

## 2.5 渐进式难度设计

### 2.5.1 Phase 1：基础时空感知

**场景**：2-3 个物体，匀速直线运动，单一方向，固定相机

| 问题子类 | 示例 | 帧数 | 难度 |
|----------|------|------|------|
| 单时间点 QRR | "在 t=2s 时，A 和 B 谁离 C 更近？" | 8 | 低 |
| 跨时间 QRR | "A-B 距离是变大了还是变小了？" | 8 | 低-中 |
| 单时间点 TRR | "在 t=3s 时，B 在 A 的几点钟方向？" | 8 | 低 |
| 跨时间 TRR | "B 相对 A 的方位变了没？" | 8 | 低-中 |
| 简单速度比较 | "A 和 B 谁移动得更快？" | 8 | 低 |
| 简单方向 | "A 在向什么方向移动？" | 8 | 低 |

**场景示例**：
```
场景 P1-01: 桌面匀速运动
- 红色球体: (0,0) → (5,0)，匀速向右
- 蓝色方块: (3,3) → (3,0)，匀速向下
- 绿色圆柱: (5,3) 静止

Q1 [temporal_qrr]: "开始时谁离绿色圆柱更近？" → 蓝色方块 (d=3.0 < d=6.7)
Q2 [temporal_qrr]: "结束时呢？" → 红色球体 (d=3.0 < d=3.0 → 大致相等)
Q3 [velocity]: "谁更快？" → 红色球体 (位移5 vs 位移3)
Q4 [direction]: "蓝色方块向哪走？" → 向下
```

### 2.5.2 Phase 2：中级时空推理

**场景**：3-5 个物体，非匀速运动（加速/减速），曲线轨迹

| 问题子类 | 示例 | 帧数 | 难度 |
|----------|------|------|------|
| 加速度检测 | "物体 A 在加速还是减速？" | 16 | 中 |
| 距离关系反转 | "A-B 距离关系是否反转了？" | 16 | 中 |
| TRR 连续变化 | "B 绕 A 是顺时针还是逆时针？" | 16 | 中 |
| 速度排序 | "三个物体按速度排列" | 16 | 中-高 |
| 轨迹形状 | "A 的运动轨迹是什么形状？" | 16 | 中 |

**场景示例**：
```
场景 P2-01: 曲线运动与加减速
- 红色球体: 圆弧运动 (r=3, 0°→180°)，匀速
- 蓝色方块: 直线运动 (0,0)→(6,0)，先加速后减速
- 绿色圆柱: (3,3) 静止

Q1 [temporal_qrr]: "红球与绿柱的距离如何变化？" → 先变大后变小
Q2 [temporal_trr]: "红球方位从3点变到几点？" → 9点（半圈）
Q3 [velocity]: "蓝块在加速还是减速？" → 先加速后减速
Q4 [trajectory]: "红球轨迹是什么形状？" → 圆弧
```

### 2.5.3 Phase 3：高级时空推理

**场景**：5-8 个物体，物体间交互（碰撞、追逐、环绕），遮挡，多视角

| 问题子类 | 示例 | 帧数 | 难度 |
|----------|------|------|------|
| 碰撞/交汇预测 | "A 和 B 的路径是否会交叉？" | 32 | 高 |
| 遮挡推理 | "A 被遮挡期间，A 相对 B 距离如何变化？" | 32 | 高 |
| 多物体排序链 | "在 t=5s 时，按与 D 的距离排列 A,B,C" | 32 | 高 |
| 跨视角 QRR/TRR | "从视角2看，距离/方位关系如何？" | 32 | 高 |
| 综合推理 | "A 绕 B 运动时，A-C 距离如何随时间变化？" | 32 | 很高 |

### 2.5.4 三阶段总结

| 维度 | Phase 1 | Phase 2 | Phase 3 |
|------|---------|---------|---------|
| 物体数 | 2-3 | 3-5 | 5-8 |
| 运动类型 | 匀速直线 | 加减速 + 曲线 | 交互 + 碰撞 + 环绕 |
| 帧数 | 8 | 16 | 32-64 |
| 相机 | 固定 | 可移动 | 多视角 |
| 问题维度 | 时序QRR/TRR + 简单速度/方向 | +加速度 +轨迹形状 +排序 | +碰撞预测 +遮挡 +多视角 +综合 |
| 预期准确率 | 人 ~95%, 模型 ~50-65% | 人 ~90%, 模型 ~35-50% | 人 ~85%, 模型 ~25-40% |
| 评分重点 | 精确匹配 | +部分得分 +Kendall tau | +时序/视角一致性 |

---

# 第三部分：DSL 扩展方案

## 3.1 现有代码架构分析

### 3.1.1 核心模块概览

```
VLM-test/
├── dsl/
│   ├── comparators.py   # Comparator 枚举 (<, ~=, >), compare() 函数, tau=0.10
│   └── predicates.py    # QRRConstraint, TRRConstraint, MetricType, compute_angle_2d
├── extraction.py        # parse_objects(), extract_gt() → {"qrr": [...], "trr": [...]}
├── question_bank.py     # enumerate_qrr(), enumerate_trr() → list[dict]
├── generate_questions.py # process_scene() 主入口
└── API-test/
    ├── scoring.py       # score_qrr(), score_trr_hour/quadrant/adjacent(), score_batch_scene()
    ├── prompts.py       # 系统提示和用户提示模板
    └── response_parser.py # 多重容错的响应解析
```

### 3.1.2 关键数据结构

**Comparator** (`comparators.py`)：三值有序集合 `LT("<")`, `APPROX("~=")`, `GT(">")`。`compare(a, b, tau=0.10)` 基于相对容差比较：`|a-b| <= tau * max(a,b)` 时为 APPROX。

**MetricType** (`predicates.py`)：四种度量 `DIST_3D`、`DIST_2D`、`DEPTH_GAP`、`SIZE_RATIO`。当前无时间维度度量。

**QRRConstraint**：
```python
@dataclass
class QRRConstraint:
    pair1: Tuple[str, str]      # 两个对象 ID
    pair2: Tuple[str, str]      # 另两个对象 ID
    metric: MetricType          # 度量类型
    comparator: Comparator      # 比较结果
    # 语义: metric(pair1) comparator metric(pair2)
```

**TRRConstraint**：
```python
@dataclass
class TRRConstraint:
    target: str       # 被定位的对象
    ref1: str         # 参考点（站立位置）
    ref2: str         # 参考方向（12点方向）
    hour: int         # 1-12 钟面位置
    quadrant: int     # 1-4 象限
    angle_deg: float  # 精确角度
```

### 3.1.3 数据流

```
Blender 生成场景 JSON
    → extraction.py: parse_objects() 提取对象坐标
    → predicates.py: compute_qrr()/compute_trr() 计算约束
    → question_bank.py: enumerate_qrr()/enumerate_trr() 生成问题
    → prompts.py: 构造 VLM 提示词
    → response_parser.py: 解析 VLM 回答
    → scoring.py: 评分
```

### 3.1.4 时间扩展的主要障碍

| 模块 | 障碍 | 说明 |
|------|------|------|
| `comparators.py` | 只能比较两个标量 | 无法表达"速度变化趋势" |
| `predicates.py` | MetricType 无时间度量 | 缺少速度、加速度、接近速率 |
| `extraction.py` | `parse_objects()` 无状态 | 只处理单帧，无时间序列概念 |
| `question_bank.py` | qid 无帧信息 | `qrr_0001` 不编码场景/帧 |
| `scoring.py` | 无时间维度统计 | 无法区分动态/静态问题表现 |
| `render_multiview.py` | 对象完全静止 | 无帧循环，无运动模型 |

---

## 3.2 时间维度扩展设计

### 3.2.1 新增 MetricType

```python
class TemporalMetricType(Enum):
    DIST_3D_AT_T = "dist3D_at_t"       # 指定帧的3D距离（复用现有）
    DIST_DELTA = "dist_delta"           # 两时刻间距差: dist(t2) - dist(t1)
    SPEED = "speed"                     # 对象移动速度（标量）
    APPROACH_RATE = "approach_rate"     # 接近速率: d(dist)/dt
    DISPLACEMENT = "displacement"       # 位移量
```

### 3.2.2 QRRConstraint 时间扩展

**方案一：时间切片 QRR（保守扩展）**

```python
@dataclass
class TemporalQRRConstraint(QRRConstraint):
    frame_t: int        # 在哪一帧计算
    # 语义: metric(pair1, frame_t) comparator metric(pair2, frame_t)
```

向后兼容，本质是"在特定帧截取的 QRR"。

**方案二：跨帧对比 QRR**

```python
@dataclass
class CrossFrameQRRConstraint:
    pair: Tuple[str, str]    # 同一对对象
    metric: TemporalMetricType
    frame_t1: int
    frame_t2: int
    comparator: Comparator
    # 语义: metric(pair, t2) comparator metric(pair, t1)
    # 例: t2时 A-B 距离 < t1时 A-B 距离 → 接近
```

**方案三：VRR — 速度相对关系（新约束类型）**

```python
@dataclass
class VRRConstraint:
    pair1: Tuple[str, str]
    pair2: Tuple[str, str]
    metric: TemporalMetricType   # APPROACH_RATE 或 SPEED
    comparator: Comparator
    frame_window: Tuple[int, int]   # [t_start, t_end]
    # 语义: approach_rate(pair1, window) comparator approach_rate(pair2, window)
```

**推荐**：同时实现方案一和方案三。方案一作为"时间切片 QRR"，方案三作为真正的新约束类型。

### 3.2.3 TRRConstraint 时间扩展

**时刻指定 TRR**：

```python
@dataclass
class TemporalTRRConstraint(TRRConstraint):
    frame_t: int    # 在哪一帧观察
```

**方向变化 TRR**：

```python
@dataclass
class TRRDeltaConstraint:
    target: str
    ref1: str
    ref2: str
    frame_t1: int
    frame_t2: int
    hour_delta: int        # 钟面方向变化量
    direction: str         # "clockwise" / "counterclockwise" / "stationary"
```

---

## 3.3 新约束类型设计

### 3.3.1 速度比较约束 (VRRConstraint)

**目的**：测试 VLM 是否能判断哪一对对象的接近/远离速率更快。

**真值计算**：
```python
approach_rate(pair, [t1, t2]) = (dist_3d(t1) - dist_3d(t2)) / (t2 - t1)
# 正值 = 接近，负值 = 远离
# 用 compare() 比较两个 approach_rate 值
```

**稳定性过滤**：仅在整个窗口内单调变化时才采纳为真值。

### 3.3.2 加速度检测约束

```python
@dataclass
class AccelerationConstraint:
    pair: Tuple[str, str]
    frame_t1: int
    frame_t2: int
    frame_t3: int
    motion_type: str        # "approaching" / "separating"
    acceleration: Comparator  # LT(减速) / APPROX(匀速) / GT(加速)
    # 语义: [t1,t2] 的接近速率 vs [t2,t3] 的接近速率
```

### 3.3.3 轨迹预测约束（最高难度）

```python
@dataclass
class TrajectoryConstraint:
    type: str               # "qrr_prediction" | "trr_prediction"
    observation_frames: List[int]    # 模型能看到的帧
    prediction_frame: int            # 需要预测的帧
    inner_constraint: Union[QRRConstraint, TRRConstraint]  # 需预测的约束
```

### 3.3.4 运动模型抽象（Blender 端）

```python
class MotionSpec(ABC):
    obj_id: str
    duration_frames: int
    @abstractmethod
    def apply_at_frame(self, t: int, current_pos: Vector) -> Vector: ...

class LinearMotionSpec(MotionSpec):
    velocity: Tuple[float, float, float]   # 每帧位移向量

class CircularMotionSpec(MotionSpec):
    center: Tuple[float, float]
    radius: float
    angular_velocity: float   # 弧度/帧

class OscillatingMotionSpec(MotionSpec):
    direction: Tuple[float, float, float]
    amplitude: float
    period_frames: int
```

---

## 3.4 评分系统改造

### 3.4.1 新评分函数

| 约束类型 | 评分方式 | 部分得分 |
|----------|----------|----------|
| 时序 QRR | 精确匹配 + 方向正确 | 方向对 0.5 分 |
| 时序 TRR | 三粒度 (精确/小时/象限) + 旋转方向 | 与静态 TRR 一致 |
| 速度关系 VRR | 精确匹配 + 运动类型对 | 运动类型对 0.5 分 |
| 加速度检测 | 精确匹配 + 加减速方向 | 方向对 0.3 分 |
| 速度排序 | Kendall tau 排序相关 | (tau+1)/2 归一化 |
| 轨迹预测 | 递归调用对应 QRR/TRR 评分 | 与内层一致 |

### 3.4.2 聚合器扩展

新增统计分组：
- `static_qrr_accuracy`：静态帧 QRR（基线对比）
- `vrr_accuracy`：速度比较准确率
- `acceleration_accuracy`：加速度检测准确率
- `trajectory_prediction_accuracy`：轨迹预测准确率
- `by_horizon`：按预测跨度分组（短期 1-2 帧 / 中期 3-5 帧 / 长期 6+ 帧）
- `temporal_consistency`：时序逻辑一致性

### 3.4.3 QID 命名规范

| 约束类型 | QID 格式 | 示例 |
|----------|----------|------|
| 单帧 QRR | `qrr_f{t:02d}_{seq:04d}` | `qrr_f03_0042` |
| 跨帧 VRR | `vrr_f{t1:02d}f{t2:02d}_{seq:04d}` | `vrr_f01f08_0003` |
| 单帧 TRR | `trr_f{t:02d}_{seq:04d}` | `trr_f05_0012` |
| 方向变化 TRR | `trd_f{t1:02d}f{t2:02d}_{seq:04d}` | `trd_f01f10_0001` |
| 加速度 | `acc_f{t1}f{t2}f{t3}_{seq:04d}` | `acc_f01f05f09_0002` |
| 轨迹预测 | `tpred_f{obs}t{pred}_{seq:04d}` | `tpred_f0-8t12_0001` |

---

## 3.5 文件组织与实现路线图

### 3.5.1 新增文件

**时序 DSL**：
```
VLM-test/
├── dsl/
│   ├── temporal_predicates.py    # VRRConstraint, AccelerationConstraint, TrajectoryConstraint
│   └── temporal_comparators.py   # 速度比较专用容差逻辑
├── temporal_extraction.py        # parse_temporal_scene(), extract_temporal_gt()
├── temporal_question_bank.py     # enumerate_vrr(), enumerate_acceleration() 等
├── generate_temporal_questions.py # 时序问题生成入口
└── API-test/
    ├── temporal_scoring.py       # 时序评分函数集
    └── prompts_temporal.py       # 时序提示模板
```

**数据生成**：
```
data-gen/
├── blender/
│   ├── render_temporal.py        # 时序渲染脚本
│   └── motion_models.py          # MotionSpec 类族
├── temporal_pipeline.py          # 时序数据生成管线
└── config_temporal.toml          # 时序场景默认配置
```

**输出目录**：
```
data-gen/output/temporal/
├── scenes/                    # 时序场景 JSON（含 frames 字段）
├── images/sequences/{scene_id}/
│   ├── frames/                # 单视角帧序列
│   │   ├── frame_0000.png
│   │   └── ...
│   └── multi_view/{view_id}/  # 多视角帧序列
└── splits/                    # 时序数据集索引
```

### 3.5.2 向后兼容策略

保持现有静态场景代码**完全不变**，时序扩展以新增文件实现。接口通过 `scene.get("n_frames", 1) == 1` 判断是否为时序场景，统一入口自动路由。

### 3.5.3 实现路线图

| 阶段 | 内容 | 工时估算 | 依赖 |
|------|------|---------|------|
| **第一阶段** | 时间切片 QRR/TRR（在已有静态数据上测试） | 3-5 天 | 无 |
| **第二阶段** | Blender 动画管线 + 线性运动模型 | 5-7 天 | 第一阶段 |
| **第三阶段** | VRR 约束 + 速度比较问题 + 评分 | 5-7 天 | 第二阶段 |
| **第四阶段** | 加速度检测 + 轨迹预测（高难度） | 5-7 天 | 第三阶段 |
| **总计** | | **约 3-4 周** | |

### 3.5.4 关键技术决策

1. **容差参数**：`tau_distance=0.10`（沿用）、`tau_velocity=0.15`（速度更难判断）、`tau_acceleration=0.20`（加速度更模糊）
2. **帧数选择**：初期 8-16 帧、2-4 秒短序列（VLM 上下文窗口限制）
3. **运动设计原则**：优先单调运动（真值可靠），线性运动为默认，圆周/振荡作为高难度变体

---

# 附录：参考文献

## 数据生成相关

- [4D-Bench: Benchmarking MLLMs for 4D Object Understanding](https://arxiv.org/abs/2503.17827) (ICCV 2025)
- [IR3D-Bench: Inverse Rendering 3D Benchmark](https://arxiv.org/abs/2506.23329) (2025)
- [Spatial457: 6D Spatial Reasoning Diagnostic Benchmark](https://arxiv.org/abs/2502.08636) (2025)
- [PhysBench: Physical World Understanding Benchmark](https://arxiv.org/abs/2501.16411) (ICLR 2025)
- [Kubric: A Scalable Dataset Generator](https://github.com/google-research/kubric) (Google Research)
- [CLEVRER: Collision Events for Video Representation and Reasoning](https://arxiv.org/abs/1910.01442) (ICLR 2020)

## 测试评估相关

- [VLM4D: Towards Spatiotemporal Awareness in Vision Language Models](https://arxiv.org/abs/2508.02095) (ICCV 2025)
- [STI-Bench: Are MLLMs Ready for Precise Spatial-Temporal World Understanding?](https://arxiv.org/abs/2503.23765) (ICCV 2025)
- [MotionBench: Fine-grained Video Motion Understanding](https://arxiv.org/abs/2501.02955) (CVPR 2025)
- [Spatial4D-Bench: A Versatile 4D Spatial Intelligence Benchmark](https://arxiv.org/abs/2601.00092) (2025)
- [MVBench: A Comprehensive Multi-modal Video Understanding Benchmark](https://arxiv.org/abs/2311.17005) (CVPR 2024)
- [TempCompass: Do Video LLMs Really Understand Videos?](https://arxiv.org/abs/2403.00476) (ACL 2024)

## 设计原则来源

| 原则 | 来源 | 应用 |
|------|------|------|
| 静态到动态方法论 | MVBench | QRR/TRR → 时序 QRR/TRR |
| 冲突视频对设计 | TempCompass | 防止单帧偏差 |
| 平均相对准确率多阈值评估 | Spatial4D-Bench | 数值型时空问题 |
| 精度分场景 | STI-Bench | 不同尺度不同容忍度 |
| 大模型充当评委 | VLM4D | 开放式回答评估 |
| 盲测过滤 | 4D-Bench | 确保问题需要视觉理解 |
