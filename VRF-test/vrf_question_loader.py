"""VRF 问题加载与场景发现。"""

import json
from pathlib import Path
from typing import List


def _flatten_batches(doc: dict) -> List[dict]:
    questions = []
    for batch in doc.get("batches", []):
        questions.extend(batch.get("questions", []))
    return questions


def discover_scene_ids(questions_dir: Path, split: str | None = None, max_scenes: int | None = None) -> List[str]:
    """Discover scene IDs from VRF question directory."""
    scene_ids = sorted(p.stem for p in questions_dir.glob("*.json"))
    if split:
        scene_ids = [sid for sid in scene_ids if sid.startswith(split)]
    if max_scenes is not None:
        scene_ids = scene_ids[:max_scenes]
    return scene_ids


def load_scene_questions(scene_id: str, questions_dir: Path) -> tuple[dict, List[dict]]:
    """Load VRF questions for a single scene. Returns (metadata, questions)."""
    path = questions_dir / f"{scene_id}.json"
    with open(path) as f:
        data = json.load(f)
    questions = _flatten_batches(data)
    return data, questions
