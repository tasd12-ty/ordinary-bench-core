"""
从数据生成的场景 JSON 中提取 GT 约束。

将场景 JSON 物体解析为 dsl/predicates.py 所需格式，
然后提取所有 QRR、TRR 和 FDR 真值约束。
"""

import json
from pathlib import Path
from typing import Dict, List, Any

from dsl.predicates import (
    MetricType, extract_all_qrr, extract_all_qrr_shared_anchor,
    extract_all_trr, extract_all_fdr,
)


def parse_objects(scene: dict) -> Dict[str, Dict]:
    """
    将场景 JSON 物体转换为 DSL 谓词所需的字典格式。

    场景 JSON 格式（来自 data-gen）：
      {"id": "obj_0", "shape": "sphere", "size": "large",
       "material": "rubber", "color": "brown",
       "3d_coords": [-1.14, 0.45, 0.0],
       "pixel_coords": [287, 150, 12.42],
       "rotation": 105.1}

    返回：
      {obj_id: {"position_3d": [...], "position_2d": [...], "depth": ..., "size": ..., ...}}
    """
    objects = {}
    for obj in scene.get("objects", []):
        obj_id = obj["id"]
        coords_3d = obj.get("3d_coords", [0, 0, 0])
        pixel = obj.get("pixel_coords", [0, 0, 0])

        objects[obj_id] = {
            "id": obj_id,
            "shape": obj.get("shape", ""),
            "color": obj.get("color", ""),
            "size": obj.get("size", "medium"),
            "material": obj.get("material", ""),
            "position_3d": coords_3d,
            "3d_coords": coords_3d,
            "position_2d": pixel[:2],
            "pixel_coords": pixel,
            "depth": pixel[2] if len(pixel) > 2 else 0.0,
            "rotation": obj.get("rotation", 0.0),
        }
    return objects


def object_description(obj: dict) -> str:
    """构造人类可读的物体描述，如 'large brown rubber sphere'。"""
    parts = []
    if obj.get("size"):
        parts.append(str(obj["size"]))
    if obj.get("color"):
        parts.append(obj["color"])
    if obj.get("material"):
        parts.append(obj["material"])
    if obj.get("shape"):
        parts.append(obj["shape"])
    return " ".join(parts)


def extract_gt(scene: dict, tau: float = 0.10) -> dict:
    """
    从场景中提取所有 GT 约束。

    返回：
      {"qrr": [QRRConstraint.to_dict(), ...], "trr": [TRRConstraint.to_dict(), ...], "fdr": [...]}
    """
    objects = parse_objects(scene)
    if len(objects) < 2:
        return {"qrr": [], "trr": [], "fdr": []}

    # QRR：使用三维距离，包含不相交对和共享锚点两种变体
    qrr_constraints = extract_all_qrr(
        objects, MetricType.DIST_3D, tau=tau, disjoint_only=True
    )
    qrr_constraints += extract_all_qrr_shared_anchor(
        objects, MetricType.DIST_3D, tau=tau,
    )
    qrr_list = [c.to_dict() for c in qrr_constraints]

    # TRR：使用三维坐标（视角不变）
    trr_constraints = extract_all_trr(objects, use_3d=True)
    trr_list = [c.to_dict() for c in trr_constraints]

    # FDR：每个锚点的全距离排序
    fdr_constraints = extract_all_fdr(objects, tau=tau)
    fdr_list = [c.to_dict() for c in fdr_constraints]

    return {"qrr": qrr_list, "trr": trr_list, "fdr": fdr_list}


def load_scene(path: str) -> dict:
    """加载场景 JSON 文件。"""
    with open(path) as f:
        return json.load(f)
