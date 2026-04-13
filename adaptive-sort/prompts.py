"""
全局距离对比较查询的 Prompt 模板。

每个划分步骤要求 VLM 将多个距离对与一个 pivot 距离对进行比较。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

_VLM_TEST = str(Path(__file__).resolve().parent.parent / "VLM-test")
if _VLM_TEST not in sys.path:
    sys.path.append(_VLM_TEST)

from extraction import object_description

DistPair = Tuple[str, str]


SYSTEM_PROMPT = """\
You are a spatial reasoning assistant analyzing a 3D scene image.

You will compare distances between pairs of objects. A **pivot pair** defines \
a reference distance. For each **candidate pair**, determine whether the \
distance between that candidate pair is:
- **shorter** than the pivot distance → answer "<"
- **approximately the same** → answer "~="
- **longer** than the pivot distance → answer ">"

Respond ONLY with a JSON array. Each element must have "qid" and "answer":
[{"qid": "cmp_001", "answer": "<"}, {"qid": "cmp_002", "answer": ">"}, ...]

Do NOT include any explanation or additional text."""


def pair_key(pair: DistPair) -> str:
    return f"{pair[0]}_{pair[1]}"


def format_partition_prompt(
    objects: dict,
    pivot: DistPair,
    candidates: list[DistPair],
    qid_prefix: str = "cmp",
) -> tuple[str, list[str]]:
    """格式化全局距离对划分步骤的用户 prompt。

    Args:
        objects: {obj_id: obj_dict}，包含描述字段。
        pivot: pivot 距离对 (obj_a, obj_b)。
        candidates: 与 pivot 比较的候选距离对。
        qid_prefix: 问题 ID 的前缀。

    Returns:
        (user_prompt, expected_qids) 元组。
    """
    # 物体描述
    obj_lines = []
    for obj_id in sorted(objects.keys()):
        desc = object_description(objects[obj_id])
        obj_lines.append(f"  - {obj_id}: {desc}")
    objects_block = "\n".join(obj_lines)

    pivot_a_desc = object_description(objects[pivot[0]])
    pivot_b_desc = object_description(objects[pivot[1]])

    # 问题
    question_lines = []
    expected_qids = []
    for idx, cand in enumerate(candidates, 1):
        qid = f"{qid_prefix}_{idx:03d}"
        expected_qids.append(qid)
        question_lines.append(
            f"[{qid}] d({cand[0]}, {cand[1]})"
        )
    questions_block = "\n".join(question_lines)

    prompt = f"""\
Objects in the image:
{objects_block}

Pivot distance: d({pivot[0]}, {pivot[1]}) — distance between {pivot[0]} ({pivot_a_desc}) and {pivot[1]} ({pivot_b_desc}).

For each pair below, is their distance shorter (<), similar (~=), or longer (>) than the pivot?
{questions_block}"""

    return prompt, expected_qids
