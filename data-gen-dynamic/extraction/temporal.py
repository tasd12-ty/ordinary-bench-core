"""
时序提取：与 VLM-test/extraction.py 兼容的帧切片、
运动学计算和空间关系事件检测。
"""

import math
from typing import Dict, List, Optional, Tuple

import numpy as np


def frame_to_static_scene(temporal_scene: dict, frame_idx: int) -> dict:
    """
    将单帧提取为与 VLM-test/extraction.py::parse_objects() 兼容的静态场景字典。

    返回字典格式与 data-gen 场景 JSON 相同：
        {"objects": [{"id", "shape", "size", "material", "color",
                       "3d_coords", "pixel_coords", "rotation"}, ...]}
    """
    frame = temporal_scene["frames"][frame_idx]
    objects = []
    # 从顶层 objects 列表获取静态属性
    static_attrs = {
        o.get("id", o.get("obj_id", "")): o
        for o in temporal_scene.get("objects", [])
    }

    for fobj in frame["objects"]:
        obj_id = fobj["id"]
        static = static_attrs.get(obj_id, {})
        objects.append({
            "id": obj_id,
            "shape": fobj.get("shape", static.get("shape", "")),
            "size": fobj.get("size", static.get("size", "")),
            "material": fobj.get("material", static.get("material", "")),
            "color": fobj.get("color", static.get("color", "")),
            "3d_coords": fobj["3d_coords"],
            "pixel_coords": fobj.get("pixel_coords", [0, 0, 0]),
            "rotation": static.get("rotation", 0.0),
        })

    return {
        "scene_id": temporal_scene["scene_id"],
        "frame_id": frame["frame_id"],
        "timestamp": frame.get("timestamp", frame_idx),
        "n_objects": len(objects),
        "objects": objects,
    }


def compute_frame_kinematics(temporal_scene: dict) -> dict:
    """
    通过有限差分（中心差分）计算每个物体的速度和加速度。

    Returns:
        {obj_id: {"velocity": (n_frames, 2), "acceleration": (n_frames, 2)}}
        为 numpy 数组。
    """
    frames = temporal_scene["frames"]
    n_frames = len(frames)
    if n_frames == 0:
        return {}

    # 收集位置：{obj_id: (n_frames, 2)}
    obj_ids = [o["id"] for o in frames[0]["objects"]]
    positions = {oid: np.zeros((n_frames, 2)) for oid in obj_ids}

    for t, frame in enumerate(frames):
        for fobj in frame["objects"]:
            oid = fobj["id"]
            c = fobj["3d_coords"]
            positions[oid][t] = [c[0], c[1]]

    fps = temporal_scene.get("fps", 24)
    dt = 1.0 / fps
    result = {}

    for oid in obj_ids:
        pos = positions[oid]
        vel = np.zeros_like(pos)
        acc = np.zeros_like(pos)

        # 中心差分（边界处用前向/后向差分）
        for t in range(n_frames):
            if t == 0:
                vel[t] = (pos[1] - pos[0]) / dt if n_frames > 1 else 0
            elif t == n_frames - 1:
                vel[t] = (pos[-1] - pos[-2]) / dt
            else:
                vel[t] = (pos[t + 1] - pos[t - 1]) / (2 * dt)

        for t in range(n_frames):
            if t == 0:
                acc[t] = (vel[1] - vel[0]) / dt if n_frames > 1 else 0
            elif t == n_frames - 1:
                acc[t] = (vel[-1] - vel[-2]) / dt
            else:
                acc[t] = (vel[t + 1] - vel[t - 1]) / (2 * dt)

        result[oid] = {"velocity": vel, "acceleration": acc}

    return result


def _pairwise_distances(frame_objects: list) -> Dict[Tuple[str, str], float]:
    """计算单帧中所有物体对的 2D 距离。"""
    n = len(frame_objects)
    dists = {}
    for i in range(n):
        for j in range(i + 1, n):
            a = frame_objects[i]
            b = frame_objects[j]
            ca = a["3d_coords"]
            cb = b["3d_coords"]
            dx = ca[0] - cb[0]
            dy = ca[1] - cb[1]
            d = math.sqrt(dx * dx + dy * dy)
            dists[(a["id"], b["id"])] = d
    return dists


def _compute_hour(dx: float, dy: float) -> int:
    """由 dx, dy 偏移量计算 TRR 小时方向（1-12 钟面方向）。"""
    angle_deg = math.degrees(math.atan2(-dy, dx))  # 屏幕坐标：y 轴向下
    # 转换为钟面：12 点方向 = 向上 = 90°
    clock_angle = (90 - angle_deg) % 360
    hour = int(clock_angle / 30) + 1
    return min(max(hour, 1), 12)


def detect_trr_changes(
    temporal_scene: dict,
) -> List[dict]:
    """
    检测物体对之间 TRR 小时方向发生变化的帧。

    返回事件字典列表：
        {"frame": t, "from_obj": id, "to_obj": id, "type": "trr_change",
         "old_hour": h1, "new_hour": h2}
    """
    frames = temporal_scene["frames"]
    if len(frames) < 2:
        return []

    obj_ids = [o["id"] for o in frames[0]["objects"]]
    n = len(obj_ids)
    events = []

    # 计算初始小时方向
    prev_hours = {}
    for fobj_a in frames[0]["objects"]:
        for fobj_b in frames[0]["objects"]:
            if fobj_a["id"] == fobj_b["id"]:
                continue
            ca = fobj_a["3d_coords"]
            cb = fobj_b["3d_coords"]
            dx = cb[0] - ca[0]
            dy = cb[1] - ca[1]
            prev_hours[(fobj_a["id"], fobj_b["id"])] = _compute_hour(dx, dy)

    for t in range(1, len(frames)):
        curr_hours = {}
        for fobj_a in frames[t]["objects"]:
            for fobj_b in frames[t]["objects"]:
                if fobj_a["id"] == fobj_b["id"]:
                    continue
                ca = fobj_a["3d_coords"]
                cb = fobj_b["3d_coords"]
                dx = cb[0] - ca[0]
                dy = cb[1] - ca[1]
                key = (fobj_a["id"], fobj_b["id"])
                h = _compute_hour(dx, dy)
                curr_hours[key] = h

                prev_h = prev_hours.get(key)
                if prev_h is not None and prev_h != h:
                    events.append({
                        "frame": t,
                        "from_obj": key[0],
                        "to_obj": key[1],
                        "type": "trr_change",
                        "old_hour": prev_h,
                        "new_hour": h,
                    })

        prev_hours = curr_hours

    return events


def detect_qrr_changes(
    temporal_scene: dict, tau: float = 0.10,
) -> List[dict]:
    """
    检测物体对间距离排序发生变化的帧（QRR 关系反转）。

    当锚点 A 的距离顺序 d(A,B) vs d(A,C) 在相邻帧间翻转
    （超出 tau 容差）时，判定为反转。

    返回事件字典列表：
        {"frame": t, "anchor": id, "pair": (id1, id2), "type": "qrr_reversal"}
    """
    frames = temporal_scene["frames"]
    if len(frames) < 2:
        return []

    events = []
    prev_dists = _pairwise_distances(frames[0]["objects"])
    obj_ids = [o["id"] for o in frames[0]["objects"]]
    n = len(obj_ids)

    for t in range(1, len(frames)):
        curr_dists = _pairwise_distances(frames[t]["objects"])

        # 对每个锚点，检查其他物体的所有配对
        for a_idx in range(n):
            anchor = obj_ids[a_idx]
            others = [oid for oid in obj_ids if oid != anchor]
            for i in range(len(others)):
                for j in range(i + 1, len(others)):
                    b, c = others[i], others[j]
                    # 获取到锚点的距离
                    key_ab = tuple(sorted([anchor, b]))
                    key_ac = tuple(sorted([anchor, c]))

                    prev_ab = prev_dists.get(key_ab, 0)
                    prev_ac = prev_dists.get(key_ac, 0)
                    curr_ab = curr_dists.get(key_ab, 0)
                    curr_ac = curr_dists.get(key_ac, 0)

                    prev_diff = prev_ab - prev_ac
                    curr_diff = curr_ab - curr_ac

                    # 反转：符号改变且两个差值均超过 tau
                    if (abs(prev_diff) > tau and abs(curr_diff) > tau
                            and prev_diff * curr_diff < 0):
                        events.append({
                            "frame": t,
                            "anchor": anchor,
                            "pair": (b, c),
                            "type": "qrr_reversal",
                        })

        prev_dists = curr_dists

    return events


def detect_occlusions(
    temporal_scene: dict,
    depth_threshold: float = 0.02,
    pixel_dist_threshold: float = 50.0,
) -> List[dict]:
    """
    使用像素坐标和深度值检测遮挡事件。

    当两物体像素位置相近（在 pixel_dist_threshold 内）且深度值差异显著
    （> depth_threshold）时，判定为遮挡。深度值较小（距相机较近）的物体为遮挡者。

    返回事件字典列表：
        {"frame": t, "occluder": id, "occluded": id, "type": "occlusion",
         "pixel_dist": float, "depth_diff": float}
    """
    frames = temporal_scene["frames"]
    events = []

    for t, frame in enumerate(frames):
        objs = frame.get("objects", [])
        n = len(objs)
        for i in range(n):
            for j in range(i + 1, n):
                a = objs[i]
                b = objs[j]
                pa = a.get("pixel_coords", [0, 0, 0])
                pb = b.get("pixel_coords", [0, 0, 0])

                # 2D 像素距离
                px_dist = math.sqrt((pa[0] - pb[0])**2 + (pa[1] - pb[1])**2)
                if px_dist > pixel_dist_threshold:
                    continue

                # 深度比较（值越小 = 距相机越近）
                depth_a = pa[2] if len(pa) > 2 else 0
                depth_b = pb[2] if len(pb) > 2 else 0
                depth_diff = abs(depth_a - depth_b)

                if depth_diff > depth_threshold:
                    if depth_a < depth_b:
                        occluder, occluded = a["id"], b["id"]
                    else:
                        occluder, occluded = b["id"], a["id"]

                    events.append({
                        "frame": t,
                        "occluder": occluder,
                        "occluded": occluded,
                        "type": "occlusion",
                        "pixel_dist": round(px_dist, 2),
                        "depth_diff": round(depth_diff, 4),
                    })

    return events
