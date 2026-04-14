"""Session adapters for human-baseline v2 modes."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from common import QRR_OPTIONS, normalize_human_answer, slugify
from session_v2 import ProgressiveSessionManager


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _pair_label(pair: tuple[str, str], object_lookup: Dict[str, str]) -> str:
    a, b = pair
    return f"{a} ({object_lookup.get(a, a)}) - {b} ({object_lookup.get(b, b)})"


class ProgressiveModeAdapter:
    """Compatibility wrapper for the existing progressive manager."""

    mode_id = "progressive"
    label = "递进式测试"
    description = "先看单视角，再对错题追加更多视角并即时批改。"

    def __init__(self, manager: ProgressiveSessionManager) -> None:
        self.manager = manager

    def is_configured(self) -> bool:
        return self.manager.questions_dir.is_dir() and self.manager.scenes_dir.is_dir()

    def describe(self) -> dict:
        return {
            "id": self.mode_id,
            "label": self.label,
            "description": self.description,
            "configured": self.is_configured(),
        }

    def get_progress_summary(self, annotator_id: str) -> dict:
        summary = self.manager.get_progress_summary(annotator_id)
        summary["test_mode"] = self.mode_id
        return summary

    def get_current_round(self, annotator_id: str) -> Optional[dict]:
        payload = self.manager.get_current_round(annotator_id)
        if payload is not None:
            payload["test_mode"] = self.mode_id
        return payload

    def allocate_scene(self, annotator_id: str) -> Optional[dict]:
        payload = self.manager.allocate_scene(annotator_id)
        if payload is not None:
            payload["test_mode"] = self.mode_id
        return payload

    def submit_round(self, annotator_id: str, submission: dict) -> dict:
        result = self.manager.submit_round(annotator_id, submission)
        result["test_mode"] = self.mode_id
        return result


class AdaptiveSortSessionAdapter:
    """File-backed adaptive-sort workflow for human annotators."""

    mode_id = "adaptive_sort"
    label = "Adaptive Sort"
    description = "固定一个 pivot 距离对，逐步比较 candidate 距离对，不显示标准答案。"

    def __init__(self, tasks_dir: str | Path, responses_dir: str | Path) -> None:
        self.tasks_dir = Path(tasks_dir)
        self.responses_dir = Path(responses_dir)
        self._manifest_cache: Optional[dict] = None
        self._scene_entries: Optional[List[dict]] = None
        self._task_cache: Dict[str, dict] = {}
        self._task_meta: Dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Capability / manifest
    # ------------------------------------------------------------------

    def is_configured(self) -> bool:
        return bool(self._load_scene_entries())

    def describe(self) -> dict:
        scene_entries = self._load_scene_entries()
        return {
            "id": self.mode_id,
            "label": self.label,
            "description": self.description,
            "configured": bool(scene_entries),
            "tasks_dir": str(self.tasks_dir),
            "n_scenes": len(scene_entries),
        }

    def _manifest_path(self) -> Path:
        return self.tasks_dir / "manifest.json"

    def _load_manifest(self) -> dict:
        if self._manifest_cache is not None:
            return self._manifest_cache

        path = self._manifest_path()
        if not path.is_file():
            self._manifest_cache = {}
            return self._manifest_cache

        with open(path) as fh:
            self._manifest_cache = json.load(fh)
        return self._manifest_cache

    def _load_scene_entries(self) -> List[dict]:
        if self._scene_entries is not None:
            return self._scene_entries

        manifest = self._load_manifest()
        entries: List[dict] = []
        for raw in manifest.get("scenes", []):
            if not isinstance(raw, dict):
                continue
            scene_id = str(raw.get("scene_id", "")).strip()
            task_file = str(raw.get("task_file", "")).strip()
            if not scene_id or not task_file:
                continue
            task_path = self.tasks_dir / task_file
            if not task_path.is_file():
                continue
            entries.append({
                "scene_id": scene_id,
                "task_file": task_file,
                "task_path": task_path,
                "title": raw.get("title") or scene_id,
            })

        self._scene_entries = entries
        return self._scene_entries

    def _find_scene_entry(self, scene_id: str) -> Optional[dict]:
        for entry in self._load_scene_entries():
            if entry["scene_id"] == scene_id:
                return entry
        return None

    def _load_scene_task(self, scene_id: str) -> dict:
        if scene_id in self._task_cache:
            return self._task_cache[scene_id]

        entry = self._find_scene_entry(scene_id)
        if entry is None:
            raise FileNotFoundError(f"Adaptive-sort task not found for scene {scene_id}")

        with open(entry["task_path"]) as fh:
            raw = json.load(fh)

        objects: List[dict] = []
        for row in sorted(raw.get("objects", []), key=lambda item: str(item.get("id", ""))):
            obj_id = str(row.get("id", "")).strip()
            if not obj_id:
                continue
            label = str(row.get("label") or obj_id)
            desc = str(row.get("desc") or label or obj_id)
            normalized = dict(row)
            normalized["id"] = obj_id
            normalized["label"] = label
            normalized["desc"] = desc
            objects.append(normalized)

        object_lookup = {obj["id"]: obj["desc"] for obj in objects}
        allow_approx = bool(raw.get("allow_approx", True))
        images_raw = raw.get("images") or {}
        images = {
            "single_view": images_raw.get("single_view"),
            "multi_view": list(images_raw.get("multi_view") or []),
        }

        normalized_steps: List[dict] = []
        total_questions = 0
        for index, step in enumerate(raw.get("steps", []), start=1):
            pivot = tuple(step.get("pivot") or ())
            if len(pivot) != 2:
                continue
            step_id = str(step.get("step_id") or f"step_{index:03d}")
            level = int(step.get("level", 0))

            questions: List[dict] = []
            for cand_index, candidate in enumerate(step.get("candidates", []), start=1):
                pair = tuple(candidate.get("pair") or ())
                if len(pair) != 2:
                    continue
                qid = str(candidate.get("qid") or f"{step_id}_cmp_{cand_index:03d}")
                prompt_text = (
                    f"相对于 pivot 距离对 {_pair_label(pivot, object_lookup)}，"
                    f"请判断候选距离对 {_pair_label(pair, object_lookup)} 更近、更远，"
                    f"还是大致相等。"
                )
                questions.append({
                    "qid": qid,
                    "type": "adaptive_sort_cmp",
                    "candidate_pair": [pair[0], pair[1]],
                    "prompt_text": prompt_text,
                    "answer_options": list(QRR_OPTIONS if allow_approx else ["<", ">"]),
                    "gt_answer": candidate.get("gt_answer"),
                })

            if not questions:
                continue

            total_questions += len(questions)
            normalized_steps.append({
                "step_id": step_id,
                "level": level,
                "pivot": [pivot[0], pivot[1]],
                "questions": questions,
            })

        task = {
            "scene_id": scene_id,
            "title": str(raw.get("title") or entry["title"]),
            "objects": objects,
            "object_lookup": object_lookup,
            "images": images,
            "allow_approx": allow_approx,
            "steps": normalized_steps,
            "source_task_file": str(entry["task_path"]),
        }

        self._task_cache[scene_id] = task
        self._task_meta[scene_id] = {
            "n_steps": len(normalized_steps),
            "total_questions": total_questions,
        }
        return task

    # ------------------------------------------------------------------
    # Progress persistence
    # ------------------------------------------------------------------

    def _progress_path(self, annotator_id: str) -> Path:
        safe_id = slugify(annotator_id)
        return self.responses_dir / safe_id / "adaptive_sort_progress_v2.json"

    def _load_progress(self, annotator_id: str) -> dict:
        path = self._progress_path(annotator_id)
        if path.is_file():
            with open(path) as fh:
                return json.load(fh)

        now = _utc_now()
        return {
            "annotator_id": slugify(annotator_id),
            "test_mode": self.mode_id,
            "scenes_completed": [],
            "current_scene": None,
            "created_at": now,
            "updated_at": now,
        }

    def _save_progress(self, annotator_id: str, progress: dict) -> None:
        path = self._progress_path(annotator_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        progress["updated_at"] = _utc_now()
        with open(path, "w") as fh:
            json.dump(progress, fh, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_progress_summary(self, annotator_id: str) -> dict:
        progress = self._load_progress(annotator_id)
        scene_entries = self._load_scene_entries()
        completed = set(progress.get("scenes_completed", []))

        total_scenes = len(scene_entries)
        total_steps = 0
        total_questions = 0
        answered_questions = 0
        completed_steps = 0

        for entry in scene_entries:
            task = self._load_scene_task(entry["scene_id"])
            total_steps += len(task["steps"])
            total_questions += sum(len(step["questions"]) for step in task["steps"])
            if entry["scene_id"] in completed:
                completed_steps += len(task["steps"])
                answered_questions += sum(len(step["questions"]) for step in task["steps"])

        current = progress.get("current_scene")
        current_scene_id = None
        current_step = None
        if current:
            current_scene_id = current.get("scene_id")
            current_step = int(current.get("step_index", 0)) + 1
            for record in current.get("step_records", []):
                completed_steps += 1
                answered_questions += len(record.get("responses", []))

        scenes_completed = len(completed)
        return {
            "annotator_id": slugify(annotator_id),
            "test_mode": self.mode_id,
            "total_scenes": total_scenes,
            "scenes_completed": scenes_completed,
            "scenes_remaining": max(total_scenes - scenes_completed, 0),
            "total_steps": total_steps,
            "completed_steps": completed_steps,
            "total_questions": total_questions,
            "answered_questions": answered_questions,
            "progress_pct": round(100.0 * scenes_completed / total_scenes, 1) if total_scenes else 0.0,
            "current_scene_id": current_scene_id,
            "current_step": current_step,
        }

    def get_current_round(self, annotator_id: str) -> Optional[dict]:
        progress = self._load_progress(annotator_id)
        current = progress.get("current_scene")
        if not current:
            return None

        scene_id = current.get("scene_id")
        task = self._load_scene_task(scene_id)
        step_index = int(current.get("step_index", 0))
        if step_index < 0 or step_index >= len(task["steps"]):
            return None

        step = task["steps"][step_index]
        n_steps_total = len(task["steps"])
        return {
            "test_mode": self.mode_id,
            "scene_id": scene_id,
            "scene_title": task["title"],
            "step_id": step["step_id"],
            "step_index": step_index + 1,
            "n_steps_total": n_steps_total,
            "level": step["level"],
            "round_label": f"Step {step_index + 1}/{n_steps_total} · Level {step['level']}",
            "images": task["images"],
            "objects": task["objects"],
            "pivot_pair": list(step["pivot"]),
            "questions": step["questions"],
            "allow_approx": task["allow_approx"],
        }

    def allocate_scene(self, annotator_id: str) -> Optional[dict]:
        progress = self._load_progress(annotator_id)
        current = progress.get("current_scene")
        if current:
            return self.get_current_round(annotator_id)

        completed = set(progress.get("scenes_completed", []))
        scene_id = None
        for entry in self._load_scene_entries():
            if entry["scene_id"] not in completed:
                scene_id = entry["scene_id"]
                break

        if scene_id is None:
            return None

        progress["current_scene"] = {
            "scene_id": scene_id,
            "step_index": 0,
            "step_records": [],
            "status": "in_progress",
        }
        self._save_progress(annotator_id, progress)
        return self.get_current_round(annotator_id)

    def submit_round(self, annotator_id: str, submission: dict) -> dict:
        progress = self._load_progress(annotator_id)
        current = progress.get("current_scene")
        if not current:
            raise ValueError("No adaptive-sort scene in progress")

        scene_id = current.get("scene_id")
        if submission.get("scene_id") != scene_id:
            raise ValueError(f"Scene mismatch: expected {scene_id}")

        task = self._load_scene_task(scene_id)
        step_index = int(current.get("step_index", 0))
        if step_index < 0 or step_index >= len(task["steps"]):
            raise ValueError(f"Invalid step index {step_index}")

        step = task["steps"][step_index]
        submitted_step_id = str(submission.get("step_id") or step["step_id"])
        if submitted_step_id != step["step_id"]:
            raise ValueError(f"Step mismatch: expected {step['step_id']}, got {submitted_step_id}")

        expected_qids = [q["qid"] for q in step["questions"]]
        response_map = {}
        for row in submission.get("responses", []):
            qid = str(row.get("qid", "")).strip()
            if qid:
                response_map[qid] = row.get("answer")

        missing_qids = [qid for qid in expected_qids if qid not in response_map]
        if missing_qids:
            raise ValueError(f"Missing answers for qids: {missing_qids}")

        normalized_responses = []
        for question in step["questions"]:
            answer = normalize_human_answer(response_map.get(question["qid"]), "adaptive_sort_cmp")
            if answer is None:
                raise ValueError(f"Invalid answer for {question['qid']}")
            if not task["allow_approx"] and answer == "~=":
                raise ValueError(f"Approximate answer is not allowed for {question['qid']}")
            normalized_responses.append({
                "qid": question["qid"],
                "candidate_pair": question["candidate_pair"],
                "answer": answer,
            })

        step_record = {
            "step_id": step["step_id"],
            "level": step["level"],
            "pivot_pair": step["pivot"],
            "responses": normalized_responses,
            "submitted_at": _utc_now(),
            "elapsed_seconds": float(submission.get("elapsed_seconds", 0.0)),
        }
        current.setdefault("step_records", []).append(step_record)
        current["step_index"] = step_index + 1

        if current["step_index"] < len(task["steps"]):
            self._save_progress(annotator_id, progress)
            next_action = "next_step"
        else:
            next_action = self._complete_scene(annotator_id, progress, task)

        return {
            "accepted": True,
            "test_mode": self.mode_id,
            "next_action": next_action,
            "submission_summary": {
                "scene_id": scene_id,
                "scene_title": task["title"],
                "step_id": step["step_id"],
                "step_index": step_index + 1,
                "n_steps_total": len(task["steps"]),
                "n_responses": len(normalized_responses),
                "level": step["level"],
            },
            "progress": self.get_progress_summary(annotator_id),
        }

    # ------------------------------------------------------------------
    # Completion / persistence
    # ------------------------------------------------------------------

    def _complete_scene(self, annotator_id: str, progress: dict, task: dict) -> str:
        current = progress["current_scene"]
        current["status"] = "completed"
        completed = progress.setdefault("scenes_completed", [])
        scene_id = current["scene_id"]
        if scene_id not in completed:
            completed.append(scene_id)

        self._save_response_file(annotator_id, task, current)
        progress["current_scene"] = None
        self._save_progress(annotator_id, progress)

        completed_set = set(completed)
        remaining = [
            entry["scene_id"]
            for entry in self._load_scene_entries()
            if entry["scene_id"] not in completed_set
        ]
        return "scene_complete" if remaining else "all_done"

    def _save_response_file(self, annotator_id: str, task: dict, current: dict) -> None:
        safe_id = slugify(annotator_id)
        scene_id = current["scene_id"]
        flat_responses: List[dict] = []
        for record in current.get("step_records", []):
            for row in record.get("responses", []):
                flat_responses.append({
                    "step_id": record["step_id"],
                    "level": record["level"],
                    "pivot_pair": record["pivot_pair"],
                    "qid": row["qid"],
                    "candidate_pair": row["candidate_pair"],
                    "answer": row["answer"],
                    "elapsed_seconds": record.get("elapsed_seconds", 0.0),
                })

        payload = {
            "schema_version": 2,
            "scene_id": scene_id,
            "annotator_id": safe_id,
            "model": f"human/{safe_id}",
            "submitted_at": _utc_now(),
            "test_type": self.mode_id,
            "selected_test_type": self.mode_id,
            "source_task_file": task["source_task_file"],
            "scene_title": task["title"],
            "allow_approx": task["allow_approx"],
            "n_objects": len(task["objects"]),
            "n_steps": len(task["steps"]),
            "total_comparisons": sum(len(step["questions"]) for step in task["steps"]),
            "step_records": current.get("step_records", []),
            "responses": flat_responses,
            "raw_response": json.dumps(flat_responses, ensure_ascii=False),
        }

        out_dir = self.responses_dir / safe_id
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{scene_id}__{safe_id}__adaptive_sort.json"
        with open(out_path, "w") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
