"""
三维 Batch 模式 VLM 提示模板。

在 2D 版本基础上，新增三维 TRR 提示（含水平钟面方向 + 垂直仰角分级）。
System prompt 指示 VLM 返回 JSON 数组，User prompt 列出物体和问题。
"""


# ── 混合题型 System Prompt（2D 版本，保持兼容） ──

BATCH_SYSTEM_PROMPT = """\
You are a spatial reasoning assistant analyzing a 3D scene image.
You will receive a list of objects visible in the image and a set of spatial questions.

Question types:
1. QRR (distance comparison): Compare 3D distances, either between two pairs of objects
   or from a common anchor object to two candidate objects.
   Answer with exactly one of: "<" (first pair closer), "~=" (approximately equal), ">" (first pair farther).
2. TRR (clock direction): Imagine standing at ref1, facing toward ref2 (12 o'clock direction).
   Answer with the clock hour (integer 1-12) where the target object appears.
3. FDR (full distance ranking): Given an anchor object, rank all other objects
   by their 3D distance from the anchor, from nearest to farthest.
   Answer with a JSON list of object IDs in order: ["nearest_id", ..., "farthest_id"].
   If two objects appear at very similar distances, give your best estimate — close pairs are scored with tolerance.

Respond ONLY with a JSON array. Each element must have "qid" and "answer".
For QRR: answer is a string "<", "~=", or ">".
For TRR: answer is an integer 1-12.
For FDR: answer is a list of object ID strings.

Example:
[{"qid": "qrr_0001", "answer": "<"}, {"qid": "trr_0001", "answer": 7}, {"qid": "fdr_0001", "answer": ["obj_2", "obj_1", "obj_3"]}]"""


# ── 多视角 System Prompt（2D 版本，保持兼容） ──

MULTI_VIEW_SYSTEM_PROMPT = """\
You are a spatial reasoning assistant analyzing a 3D scene from multiple viewpoints.
You will receive {n_views} images of the same scene taken from different camera angles,
followed by a list of objects visible in the scene and a set of spatial questions.

Analyze ALL provided views to determine spatial relationships more accurately.

Question types:
1. QRR (distance comparison): Compare 3D distances, either between two pairs of objects
   or from a common anchor object to two candidate objects.
   Answer with exactly one of: "<" (first pair closer), "~=" (approximately equal), ">" (first pair farther).
2. TRR (clock direction): Imagine standing at ref1, facing toward ref2 (12 o'clock direction).
   Answer with the clock hour (integer 1-12) where the target object appears.
3. FDR (full distance ranking): Given an anchor object, rank all other objects
   by their 3D distance from the anchor, from nearest to farthest.
   Answer with a JSON list of object IDs in order: ["nearest_id", ..., "farthest_id"].
   If two objects appear at very similar distances, give your best estimate — close pairs are scored with tolerance.

Respond ONLY with a JSON array. Each element must have "qid" and "answer".
For QRR: answer is a string "<", "~=", or ">".
For TRR: answer is an integer 1-12.
For FDR: answer is a list of object ID strings.

Example:
[{{"qid": "qrr_0001", "answer": "<"}}, {{"qid": "trr_0001", "answer": 7}}, {{"qid": "fdr_0001", "answer": ["obj_2", "obj_1", "obj_3"]}}]"""


def format_batch_user_prompt(objects: list, questions: list) -> str:
    """
    构造 batch 模式的 user prompt。

    列出场景中的物体描述，然后逐条列出本 batch 的问题。
    QRR 问题用自然语言描述距离比较，TRR 问题用钟面方向 + 仰角描述，
    FDR 问题用排序描述。
    """
    lines = ["Objects in the image:"]
    for obj in objects:
        lines.append(f"  - {obj['id']}: {obj['desc']}")
    lines.append("")
    lines.append("Questions:")

    for q in questions:
        if q["type"] == "qrr":
            p1a, p1b = q["pair1"]
            p2a, p2b = q["pair2"]
            if q.get("variant") == "shared_anchor" and q.get("anchor"):
                anchor = q["anchor"]
                cand1 = next(obj for obj in q["pair1"] if obj != anchor)
                cand2 = next(obj for obj in q["pair2"] if obj != anchor)
                lines.append(
                    f"[{q['qid']}] From anchor {anchor}, compare the distance to {cand1} "
                    f"vs the distance to {cand2}. "
                    f"Answer: < / ~= / >"
                )
            else:
                lines.append(
                    f"[{q['qid']}] Compare the distance between {p1a} and {p1b} "
                    f"vs the distance between {p2a} and {p2b}. "
                    f"Answer: < / ~= / >"
                )
        elif q["type"] == "trr":
            # 三维 TRR：同时询问水平方向和垂直仰角
            lines.append(
                f"[{q['qid']}] Standing at {q['ref1']}, facing {q['ref2']} "
                f"(12 o'clock), what clock hour (1-12) is {q['target']} at? "
                f"Also, is the target above, slightly_above, level, slightly_below, or below?"
            )
        elif q["type"] == "fdr":
            others = [o["id"] for o in objects if o["id"] != q["anchor"]]
            lines.append(
                f"[{q['qid']}] Rank all other objects by distance from {q['anchor']}, "
                f"nearest to farthest. Objects to rank: {', '.join(others)}. "
                f"Answer: ordered JSON list of object IDs."
            )

    return "\n".join(lines)


# ── 单题型 System Prompts（按题型分开问答时使用） ──

QRR_SYSTEM_PROMPT = """\
You are a spatial reasoning assistant analyzing a 3D scene image.
You will receive a list of objects visible in the image and a set of distance comparison questions.

For each question, compare 3D distances — either between two pairs of objects,
or from a common anchor object to two candidate objects.
Answer with exactly one of: "<" (first pair/candidate closer), "~=" (approximately equal), ">" (first pair/candidate farther).

Respond ONLY with a JSON array. Each element must have "qid" and "answer" (a string: "<", "~=", or ">").

Example:
[{"qid": "qrr_0001", "answer": "<"}, {"qid": "qrr_0002", "answer": "~="}]"""


TRR_SYSTEM_PROMPT = """\
You are a spatial reasoning assistant analyzing a 3D scene image.
You will receive a list of objects visible in the image and a set of clock-direction questions.

For each question, imagine standing at ref1, facing toward ref2 (that is your 12 o'clock direction).
Determine the clock hour (integer 1-12) where the target object appears.

Respond ONLY with a JSON array. Each element must have "qid" and "answer" (an integer 1-12).

Example:
[{"qid": "trr_0001", "answer": 7}, {"qid": "trr_0002", "answer": 11}]"""


FDR_SYSTEM_PROMPT = """\
You are a spatial reasoning assistant analyzing a 3D scene image.
You will receive a list of objects visible in the image and a set of distance ranking questions.

For each question, given an anchor object, rank all other listed objects by their 3D distance
from the anchor, from nearest to farthest. Answer with a JSON list of object IDs in order.
If two objects appear at very similar distances, give your best estimate — close pairs are scored with tolerance.

Respond ONLY with a JSON array. Each element must have "qid" and "answer" (a list of object ID strings).

Example:
[{"qid": "fdr_0001", "answer": ["obj_2", "obj_1", "obj_3"]}]"""


# ── 三维 TRR System Prompt（钟面方向 + 垂直仰角） ──

TRR_3D_SYSTEM_PROMPT = """\
You are a spatial reasoning assistant analyzing a 3D scene image.
You will receive a list of objects visible in the image and a set of 3D directional questions.

For each question, imagine standing at ref1, facing toward ref2 (that is your 12 o'clock direction).
You must determine TWO things about the target object:

1. Horizontal direction (clock hour): the clock hour (integer 1-12) where the target appears
   when looking from above (bird's-eye view). 12 = straight ahead toward ref2, 3 = right,
   6 = behind, 9 = left.

2. Elevation: whether the target is above, slightly above, at the same level, slightly below,
   or below ref1. Choose exactly one of:
     "above"          — clearly higher
     "slightly_above" — a little higher
     "level"          — roughly the same height
     "slightly_below" — a little lower
     "below"          — clearly lower

Respond ONLY with a JSON array. Each element must have "qid" and "answer".
The "answer" must be an object with two fields: "hour" (integer 1-12) and "elevation" (string).

Example:
[{"qid": "trr_0001", "answer": {"hour": 7, "elevation": "slightly_above"}}, \
{"qid": "trr_0002", "answer": {"hour": 11, "elevation": "below"}}]"""


# ── 题型到 System Prompt 的映射（使用三维 TRR） ──

TYPE_SYSTEM_PROMPTS = {
    "qrr": QRR_SYSTEM_PROMPT,
    "trr": TRR_3D_SYSTEM_PROMPT,
    "fdr": FDR_SYSTEM_PROMPT,
}


# ── 三维混合题型 Batch System Prompt（QRR + TRR_3D + FDR） ──

BATCH_3D_SYSTEM_PROMPT = """\
You are a spatial reasoning assistant analyzing a 3D scene image.
You will receive a list of objects visible in the image and a set of spatial questions.

Question types:
1. QRR (distance comparison): Compare 3D distances, either between two pairs of objects
   or from a common anchor object to two candidate objects.
   Answer with exactly one of: "<" (first pair closer), "~=" (approximately equal), ">" (first pair farther).

2. TRR (3D direction): Imagine standing at ref1, facing toward ref2 (12 o'clock direction).
   Determine two things about the target object:
   a) Horizontal direction: the clock hour (integer 1-12) from a bird's-eye view.
      12 = straight ahead toward ref2, 3 = right, 6 = behind, 9 = left.
   b) Elevation: one of "above", "slightly_above", "level", "slightly_below", "below".

3. FDR (full distance ranking): Given an anchor object, rank all other objects
   by their 3D distance from the anchor, from nearest to farthest.
   Answer with a JSON list of object IDs in order: ["nearest_id", ..., "farthest_id"].
   If two objects appear at very similar distances, give your best estimate — close pairs are scored with tolerance.

Respond ONLY with a JSON array. Each element must have "qid" and "answer".
For QRR: answer is a string "<", "~=", or ">".
For TRR: answer is an object {"hour": N, "elevation": "band"}.
For FDR: answer is a list of object ID strings.

Example:
[{"qid": "qrr_0001", "answer": "<"}, \
{"qid": "trr_0001", "answer": {"hour": 7, "elevation": "slightly_above"}}, \
{"qid": "fdr_0001", "answer": ["obj_2", "obj_1", "obj_3"]}]"""


# ── ReAct 纠正提示：当解析失败或缺失过多时，追加此 prompt 要求模型重新输出 ──

REACT_CORRECTION_PROMPT = """\
Your previous response could not be fully parsed. {n_missing} out of {n_total} answers are missing.

Missing question IDs: {missing_qids}

Please output ONLY a valid JSON array containing the answers for the missing questions.
Do NOT include any explanation, markdown fences, or extra text.
Format: [{{"qid": "...", "answer": ...}}, ...]"""
