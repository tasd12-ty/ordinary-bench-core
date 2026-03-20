from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parent.parent
VLM_ROOT = REPO_ROOT / "VLM-test"
API_ROOT = VLM_ROOT / "API-test"

for path in (str(API_ROOT), str(VLM_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from extraction import object_description  # noqa: E402


QUESTION_TYPES = ("qrr", "trr", "fdr")
QRR_OPTIONS = ["<", "~=", ">"]
TRR_OPTIONS = list(range(1, 13))

COLOR_ZH = {
    "blue": "蓝色",
    "brown": "棕色",
    "cyan": "青色",
    "gray": "灰色",
    "green": "绿色",
    "purple": "紫色",
    "red": "红色",
    "yellow": "黄色",
}

SIZE_ZH = {
    "small": "小",
    "large": "大",
}

MATERIAL_ZH = {
    "rubber": "橡胶",
    "metal": "金属",
}

SHAPE_ZH = {
    "cube": "立方体",
    "cylinder": "圆柱",
    "sphere": "球体",
}


def slugify(value: str) -> str:
    text = re.sub(r"[^0-9A-Za-z._-]+", "-", value.strip())
    text = re.sub(r"-{2,}", "-", text).strip("-")
    return text or "anonymous"


def load_scene_doc(scene_id: str, scenes_dir: str) -> dict:
    path = Path(scenes_dir) / f"{scene_id}.json"
    with open(path) as f:
        return json.load(f)


def load_question_documents(scene_id: str, questions_dir: str) -> Dict[str, dict]:
    base = Path(questions_dir)
    split_docs: Dict[str, dict] = {}
    for qtype in QUESTION_TYPES:
        path = base / qtype / f"{scene_id}.json"
        if path.exists():
            with open(path) as f:
                split_docs[qtype] = json.load(f)

    if split_docs:
        return split_docs

    flat_path = base / f"{scene_id}.json"
    if not flat_path.exists():
        raise FileNotFoundError(f"No question file found for {scene_id} under {questions_dir}")

    with open(flat_path) as f:
        flat_doc = json.load(f)

    grouped_questions = {qtype: [] for qtype in QUESTION_TYPES}
    for batch in flat_doc.get("batches", []):
        for question in batch.get("questions", []):
            qtype = question.get("type")
            if qtype in grouped_questions:
                grouped_questions[qtype].append(question)

    docs: Dict[str, dict] = {}
    for qtype, questions in grouped_questions.items():
        if not questions:
            continue
        docs[qtype] = {
            "scene_id": flat_doc.get("scene_id"),
            "image_path": flat_doc.get("image_path"),
            "objects": flat_doc.get("objects", []),
            "n_objects": flat_doc.get("n_objects"),
            "question_type": qtype,
            "tau": flat_doc.get("tau", 0.10),
            "total_questions": len(questions),
            "n_batches": 1,
            "batches": [{
                "batch_id": 0,
                "n_questions": len(questions),
                "questions": questions,
            }],
        }
    return docs


def iter_question_batches(question_docs: Dict[str, dict]) -> List[dict]:
    batches: List[dict] = []
    for qtype in QUESTION_TYPES:
        doc = question_docs.get(qtype)
        if doc is None:
            continue
        for batch in doc.get("batches", []):
            batches.append({
                "batch_id": f"{qtype}_{batch['batch_id']}",
                "question_type": qtype,
                "n_questions": batch.get("n_questions", len(batch.get("questions", []))),
                "questions": list(batch.get("questions", [])),
            })
    return batches


def flatten_questions(question_docs: Dict[str, dict]) -> List[dict]:
    questions: List[dict] = []
    for batch in iter_question_batches(question_docs):
        questions.extend(batch["questions"])
    return questions


def build_object_catalog(scene_doc: dict) -> List[dict]:
    objects = []
    for obj in sorted(scene_doc.get("objects", []), key=lambda row: row["id"]):
        objects.append({
            "id": obj["id"],
            "label": obj["id"],
            "desc": describe_object_zh(obj),
            "desc_en": object_description(obj),
            "pixel_coords": list(obj.get("pixel_coords", [])),
            "shape": obj.get("shape"),
            "color": obj.get("color"),
            "material": obj.get("material"),
            "size": obj.get("size"),
        })
    return objects


def describe_object_zh(obj: dict) -> str:
    color = COLOR_ZH.get(str(obj.get("color", "")).lower(), str(obj.get("color", "")))
    size = SIZE_ZH.get(str(obj.get("size", "")).lower(), str(obj.get("size", "")))
    material = MATERIAL_ZH.get(str(obj.get("material", "")).lower(), str(obj.get("material", "")))
    shape = SHAPE_ZH.get(str(obj.get("shape", "")).lower(), str(obj.get("shape", "")))
    return "".join(part for part in [color, size, material, shape] if part)


def build_object_lookup(objects: Iterable[dict]) -> Dict[str, str]:
    return {obj["id"]: obj["desc"] for obj in objects}


def format_human_question(question: dict, object_lookup: Dict[str, str]) -> str:
    if question["type"] == "qrr":
        if question.get("variant") == "shared_anchor" and question.get("anchor"):
            anchor = question["anchor"]
            cand1 = next(obj for obj in question["pair1"] if obj != anchor)
            cand2 = next(obj for obj in question["pair2"] if obj != anchor)
            return (
                f"以{object_lookup[anchor]}为参照，请比较它到{object_lookup[cand1]}的距离，"
                f"和到{object_lookup[cand2]}的距离。"
            )
        p1a, p1b = question["pair1"]
        p2a, p2b = question["pair2"]
        return (
            f"请比较{object_lookup[p1a]}与{object_lookup[p1b]}之间的距离，"
            f"以及{object_lookup[p2a]}与{object_lookup[p2b]}之间的距离。"
        )

    if question["type"] == "trr":
        return (
            f"假设你站在{object_lookup[question['ref1']]}处，面向{object_lookup[question['ref2']]}，"
            f"并把这个方向看作 12 点钟方向，那么{object_lookup[question['target']]}位于几点钟方向？"
        )

    if question["type"] == "fdr":
        candidates = [oid for oid in sorted(object_lookup) if oid != question["anchor"]]
        candidate_text = ", ".join(object_lookup[oid] for oid in candidates)
        return (
            f"请以{object_lookup[question['anchor']]}为基准，"
            f"将其余物体按距离从近到远排序。候选物体：{candidate_text}。"
        )

    raise ValueError(f"Unsupported question type: {question['type']}")


def _boxes_overlap(box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]) -> bool:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    return ax1 < bx2 and ax2 > bx1 and ay1 < by2 and ay2 > by1


def _clamp(value: int, lower: int, upper: int) -> int:
    return max(lower, min(value, upper))


def render_labeled_image(source_image: str, objects: List[dict], output_image: str) -> None:
    image = Image.open(source_image).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    width, height = image.size

    placed_boxes: List[Tuple[int, int, int, int]] = []
    sorted_objects = sorted(
        objects,
        key=lambda row: row.get("pixel_coords", [0, 0, 0])[2] if len(row.get("pixel_coords", [])) >= 3 else 0,
        reverse=True,
    )

    for index, obj in enumerate(sorted_objects):
        coords = obj.get("pixel_coords", [])
        if len(coords) < 2:
            continue

        anchor_x = int(coords[0])
        anchor_y = int(coords[1])
        label = obj["label"]
        left_side = anchor_x < width * 0.5
        upper_half = anchor_y < height * 0.45

        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        dx = 34 if left_side else -(text_width + 34)
        dy = 26 if upper_half else -(text_height + 26)
        step_y = 18 if upper_half else -18

        text_x = _clamp(anchor_x + dx, 8, width - text_width - 14)
        text_y = _clamp(anchor_y + dy, 8, height - text_height - 12)
        box = (text_x - 6, text_y - 4, text_x + text_width + 6, text_y + text_height + 4)

        for _ in range(10):
            if not any(_boxes_overlap(box, placed) for placed in placed_boxes):
                break
            text_y = _clamp(text_y + step_y, 8, height - text_height - 12)
            box = (text_x - 6, text_y - 4, text_x + text_width + 6, text_y + text_height + 4)

        line_end_x = box[0] if left_side else box[2]
        line_end_y = (box[1] + box[3]) // 2
        draw.line([(anchor_x, anchor_y), (line_end_x, line_end_y)], fill=(0, 0, 0), width=3)
        draw.line([(anchor_x, anchor_y), (line_end_x, line_end_y)], fill=(255, 255, 255), width=1)
        draw.rectangle(box, fill=(0, 0, 0), outline=(255, 255, 255), width=1)
        draw.text((text_x, text_y), label, fill=(255, 255, 255), font=font)
        placed_boxes.append(box)

    output_path = Path(output_image)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def normalize_human_answer(answer: object, qtype: str) -> Optional[object]:
    if answer is None:
        return None

    if qtype == "qrr":
        text = str(answer).strip()
        if text == "=":
            text = "~="
        return text if text in QRR_OPTIONS else None

    if qtype == "trr":
        try:
            value = int(str(answer).strip())
        except (TypeError, ValueError):
            return None
        return value if 1 <= value <= 12 else None

    if qtype == "fdr":
        if isinstance(answer, list):
            items = [str(item).strip() for item in answer if str(item).strip()]
            return items or None
        if isinstance(answer, str):
            text = answer.strip()
            if not text:
                return None
            if text.startswith("["):
                try:
                    parsed = json.loads(text)
                except json.JSONDecodeError:
                    parsed = None
                if isinstance(parsed, list):
                    items = [str(item).strip() for item in parsed if str(item).strip()]
                    return items or None
            items = [part.strip() for part in re.split(r"[\n,]+", text) if part.strip()]
            return items or None
        return None

    return None
