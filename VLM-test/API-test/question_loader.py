"""Question loading and scene discovery helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from question_bank import make_batches


QUESTION_TYPES = ("qrr", "trr", "fdr")


def _flatten_batches(doc: dict) -> list[dict]:
    questions = []
    for batch in doc.get("batches", []):
        questions.extend(batch.get("questions", []))
    return questions


def _load_json(path: Path) -> dict:
    with open(path) as handle:
        return json.load(handle)


def _discover_split_scene_ids(questions_dir: Path, question_types: Iterable[str]) -> list[str]:
    scene_ids = set()
    for qtype in question_types:
        type_dir = questions_dir / qtype
        if not type_dir.exists():
            continue
        scene_ids.update(path.stem for path in type_dir.glob("*.json"))
    return sorted(scene_ids)


def _discover_flat_scene_ids(questions_dir: Path) -> list[str]:
    return sorted(path.stem for path in questions_dir.glob("*.json"))


def _read_split_manifest(manifest_dir: Path, split_name: str) -> set[str]:
    manifest_path = manifest_dir / split_name
    with open(manifest_path) as handle:
        return {item["scene_id"] for item in json.load(handle)}


def discover_scene_ids(job) -> list[str]:
    questions_dir = Path(job.input.questions_dir)
    selection = job.selection

    if selection.scene:
        scene_ids = [selection.scene]
    else:
        if job.input.question_layout == "v1":
            scene_ids = _discover_flat_scene_ids(questions_dir)
        elif job.input.question_layout == "v2":
            scene_ids = _discover_split_scene_ids(questions_dir, job.input.question_types)
        else:
            split_ids = _discover_split_scene_ids(questions_dir, job.input.question_types)
            scene_ids = split_ids or _discover_flat_scene_ids(questions_dir)

        if selection.split:
            scene_ids = [scene_id for scene_id in scene_ids if scene_id.startswith(selection.split)]

        if selection.test_only or selection.train_only:
            manifest_dir = Path(selection.split_manifest_dir)
            manifest_name = "test_scenes.json" if selection.test_only else "train_scenes.json"
            selected = _read_split_manifest(manifest_dir, manifest_name)
            scene_ids = [scene_id for scene_id in scene_ids if scene_id in selected]

    if selection.max_scenes is not None:
        scene_ids = scene_ids[: selection.max_scenes]

    return scene_ids


def _load_split_scene(scene_id: str, questions_dir: Path, question_types: list[str]) -> tuple[dict, dict[str, list[dict]]]:
    scene_meta = None
    questions_by_type: dict[str, list[dict]] = {}
    for qtype in question_types:
        path = questions_dir / qtype / f"{scene_id}.json"
        if not path.exists():
            continue
        data = _load_json(path)
        if scene_meta is None:
            scene_meta = data
        questions_by_type[qtype] = _flatten_batches(data)
    if scene_meta is None:
        raise FileNotFoundError(f"No split-format question files found for {scene_id}")
    return scene_meta, questions_by_type


def _load_flat_scene(scene_id: str, questions_dir: Path) -> tuple[dict, dict[str, list[dict]]]:
    path = questions_dir / f"{scene_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"No flat-format question file found for {scene_id}")
    data = _load_json(path)
    questions = _flatten_batches(data)
    grouped = {qtype: [question for question in questions if question["type"] == qtype] for qtype in QUESTION_TYPES}
    return data, grouped


def load_scene_questions(scene_id: str, job):
    questions_dir = Path(job.input.questions_dir)
    layout = job.input.question_layout
    question_types = list(job.input.question_types)

    if layout == "v1":
        scene_meta, questions_by_type = _load_flat_scene(scene_id, questions_dir)
    elif layout == "v2":
        scene_meta, questions_by_type = _load_split_scene(scene_id, questions_dir, question_types)
    else:
        try:
            scene_meta, questions_by_type = _load_split_scene(scene_id, questions_dir, question_types)
        except FileNotFoundError:
            scene_meta, questions_by_type = _load_flat_scene(scene_id, questions_dir)

    if job.input.question_grouping == "mixed":
        ordered = []
        for qtype in QUESTION_TYPES:
            if qtype in question_types:
                ordered.extend(questions_by_type.get(qtype, []))
        groups = [{
            "group_id": "mixed",
            "question_type": None,
            "questions": ordered,
            "batches": make_batches(ordered, job.input.batch_size),
        }]
    else:
        groups = []
        for qtype in question_types:
            type_questions = questions_by_type.get(qtype, [])
            if not type_questions:
                continue
            groups.append({
                "group_id": qtype,
                "question_type": qtype,
                "questions": type_questions,
                "batches": make_batches(type_questions, job.input.batch_size),
            })

    if not groups or not any(group["questions"] for group in groups):
        raise ValueError(f"No questions loaded for {scene_id}")

    return scene_meta, groups
