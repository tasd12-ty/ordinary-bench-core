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


SYSTEM_PROMPT_WITH_APPROX = """\
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

SYSTEM_PROMPT_STRICT = """\
You are a spatial reasoning assistant analyzing a 3D scene image.

You will compare distances between pairs of objects. A **pivot pair** defines \
a reference distance. For each **candidate pair**, determine whether the \
distance between that candidate pair is:
- **shorter** than the pivot distance → answer "<"
- **longer** than the pivot distance → answer ">"

You must choose exactly one of "<" or ">". Do NOT answer "~=" or "equal".

Respond ONLY with a JSON array. Each element must have "qid" and "answer":
[{"qid": "cmp_001", "answer": "<"}, {"qid": "cmp_002", "answer": ">"}, ...]

Do NOT include any explanation or additional text."""

# 默认保持向后兼容
SYSTEM_PROMPT = SYSTEM_PROMPT_WITH_APPROX


def get_system_prompt(allow_approx: bool = True) -> str:
    """根据 allow_approx 开关返回对应的 system prompt。"""
    return SYSTEM_PROMPT_WITH_APPROX if allow_approx else SYSTEM_PROMPT_STRICT


def pair_key(pair: DistPair) -> str:
    return f"{pair[0]}_{pair[1]}"


def format_partition_prompt(
    objects: dict,
    pivot: DistPair,
    candidates: list[DistPair],
    qid_prefix: str = "cmp",
    allow_approx: bool = True,
) -> tuple[str, list[str]]:
    """格式化全局距离对划分步骤的用户 prompt。

    Args:
        objects: {obj_id: obj_dict}，包含描述字段。
        pivot: pivot 距离对 (obj_a, obj_b)。
        candidates: 与 pivot 比较的候选距离对。
        qid_prefix: 问题 ID 的前缀。
        allow_approx: 是否允许 VLM 回答 ~=。False 时只允许 < / >。

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

    if allow_approx:
        comparison_line = "For each pair below, is their distance shorter (<), similar (~=), or longer (>) than the pivot?"
    else:
        comparison_line = "For each pair below, is their distance shorter (<) or longer (>) than the pivot? You must choose one."

    prompt = f"""\
Objects in the image:
{objects_block}

Pivot distance: d({pivot[0]}, {pivot[1]}) — distance between {pivot[0]} ({pivot_a_desc}) and {pivot[1]} ({pivot_b_desc}).

{comparison_line}
{questions_block}"""

    return prompt, expected_qids


def append_thinking_directive(prompt: str, enable_thinking: bool) -> str:
    """在 prompt 末尾追加 Qwen3 系列的思考模式软开关指令。

    Args:
        prompt: 原始 user prompt。
        enable_thinking: True 追加 /think，False 追加 /no_think。

    Returns:
        追加指令后的 prompt。
    """
    directive = "/think" if enable_thinking else "/no_think"
    return f"{prompt}\n{directive}"
