"""VRF 评测 prompt 模板。"""

VRF_SYSTEM_PROMPT = """\
You are a spatial reasoning assistant analyzing a 3D scene image.
You will receive a list of objects visible in the image and a set of composite spatial claims.

Each question contains multiple spatial statements about distances and directions between objects.
Your task: judge whether ALL statements in each question are correct.
- Answer **true** if every statement is correct.
- Answer **false** if any statement is wrong.

Respond ONLY with a JSON array. Each element must have "qid" and "answer" (boolean).

Example:
[{"qid": "vrf_0001", "answer": true}, {"qid": "vrf_0002", "answer": false}]"""


def format_vrf_user_prompt(objects: list, questions: list) -> str:
    """Format VRF user prompt with object list and composite claims."""
    lines = ["Objects in the image:"]
    for obj in objects:
        lines.append(f"  - {obj['id']}: {obj['desc']}")
    lines.append("")
    lines.append("Questions:")

    for q in questions:
        lines.append(f"\n[{q['qid']}] Are ALL of the following spatial relationships correct?")
        for ci, claim in enumerate(q["claims"]):
            letter = chr(ord("a") + ci)
            lines.append(f"  ({letter}) {claim['claim_text']}")
        lines.append("  Answer: true / false")

    return "\n".join(lines)
