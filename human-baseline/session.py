"""SessionManager for human baseline question allocation and progress tracking.

Handles loading test scenes, allocating pages of questions to annotators,
tracking per-annotator progress, and saving responses in a format compatible
with analyze_responses.py.
"""
from __future__ import annotations

import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from common import (
    QUESTION_TYPES,
    QRR_OPTIONS,
    TRR_OPTIONS,
    build_object_catalog,
    build_object_lookup,
    flatten_questions,
    format_human_question,
    load_question_documents,
    load_scene_doc,
    slugify,
)


# Default proportions per question type when assembling a page.
# These are target counts for a page of size 12; they scale with page_size.
_TYPE_WEIGHTS: Dict[str, float] = {
    "qrr": 5.0,
    "trr": 4.0,
    "fdr": 1.5,
}


class SessionManager:
    """Manages question allocation, progress tracking, and response submission
    for human baseline annotations.

    Each annotator gets an independent progress file that records which question
    IDs have been answered.  Pages are allocated one scene at a time with a
    proportional mix of QRR, TRR, and FDR questions.
    """

    def __init__(
        self,
        questions_dir: str | Path,
        scenes_dir: str | Path,
        images_dir: str | Path,
        multi_view_images_dir: str | Path,
        tasks_dir: str | Path,
        responses_dir: str | Path,
        test_scenes_file: Optional[str | Path] = None,
        page_size: int = 12,
        min_page_size: int = 10,
    ) -> None:
        self.questions_dir = Path(questions_dir)
        self.scenes_dir = Path(scenes_dir)
        self.images_dir = Path(images_dir)
        self.multi_view_images_dir = Path(multi_view_images_dir)
        self.tasks_dir = Path(tasks_dir)
        self.responses_dir = Path(responses_dir)
        self.test_scenes_file = Path(test_scenes_file) if test_scenes_file else None
        self.page_size = page_size
        self.min_page_size = min_page_size

        self._test_scene_ids: Optional[List[str]] = None
        self._all_questions: Optional[Dict[str, Dict[str, List[dict]]]] = None

    # ------------------------------------------------------------------
    # Scene discovery
    # ------------------------------------------------------------------

    def _load_test_scene_ids(self) -> List[str]:
        """Load test-only scene IDs.

        Reads from the explicit test_scenes_file (or ``data-gen/output/test_scenes.json``
        by default).  If no file exists, falls back to filtering scene IDs whose
        numeric index is >= 80.
        """
        if self._test_scene_ids is not None:
            return self._test_scene_ids

        # Try the explicit test-scenes manifest first.
        candidates: List[Path] = []
        if self.test_scenes_file is not None:
            candidates.append(self.test_scenes_file)
        # Convention: test_scenes.json lives next to the scenes directory.
        candidates.append(self.scenes_dir.parent / "test_scenes.json")

        for path in candidates:
            if path.is_file():
                with open(path) as fh:
                    data = json.load(fh)
                if isinstance(data, list):
                    scene_ids = [
                        entry["scene_id"] if isinstance(entry, dict) else str(entry)
                        for entry in data
                    ]
                    self._test_scene_ids = sorted(scene_ids)
                    return self._test_scene_ids

        # Fallback: discover from the questions directory and filter index >= 80.
        all_ids: Set[str] = set()
        for qtype in QUESTION_TYPES:
            qtype_dir = self.questions_dir / qtype
            if qtype_dir.is_dir():
                all_ids.update(p.stem for p in qtype_dir.glob("*.json"))

        test_ids: List[str] = []
        for sid in sorted(all_ids):
            # Scene IDs follow the pattern ``nXX_YYYYYY`` where YYYYYY is the index.
            parts = sid.rsplit("_", 1)
            if len(parts) == 2:
                try:
                    index = int(parts[1])
                except ValueError:
                    continue
                if index >= 80:
                    test_ids.append(sid)

        self._test_scene_ids = test_ids
        return self._test_scene_ids

    # ------------------------------------------------------------------
    # Question loading
    # ------------------------------------------------------------------

    def _load_all_questions(self) -> Dict[str, Dict[str, List[dict]]]:
        """Load all questions for all test scenes.

        Returns ``{scene_id: {qtype: [question, ...]}}``.  Each question dict
        is the raw question object from the question JSON files.
        """
        if self._all_questions is not None:
            return self._all_questions

        scene_ids = self._load_test_scene_ids()
        result: Dict[str, Dict[str, List[dict]]] = {}

        for scene_id in scene_ids:
            try:
                docs = load_question_documents(scene_id, str(self.questions_dir))
            except FileNotFoundError:
                continue

            scene_qs: Dict[str, List[dict]] = {}
            for qtype in QUESTION_TYPES:
                doc = docs.get(qtype)
                if doc is None:
                    continue
                questions: List[dict] = []
                for batch in doc.get("batches", []):
                    questions.extend(batch.get("questions", []))
                if questions:
                    scene_qs[qtype] = questions

            if scene_qs:
                result[scene_id] = scene_qs

        self._all_questions = result
        return self._all_questions

    # ------------------------------------------------------------------
    # Progress persistence
    # ------------------------------------------------------------------

    def _progress_path(self, annotator_id: str) -> Path:
        """Return the path to an annotator's progress file."""
        safe_id = slugify(annotator_id)
        return self.responses_dir / safe_id / "progress.json"

    def _load_progress(self, annotator_id: str) -> dict:
        """Load or create ``progress.json`` for *annotator_id*.

        Progress structure::

            {
                "annotator_id": "...",
                "answered": {"scene_id": ["qid", ...]},
                "pages_completed": 0,
                "created_at": "ISO timestamp",
                "updated_at": "ISO timestamp"
            }
        """
        path = self._progress_path(annotator_id)
        if path.is_file():
            with open(path) as fh:
                return json.load(fh)

        now = datetime.now(timezone.utc).isoformat()
        return {
            "annotator_id": slugify(annotator_id),
            "answered": {},
            "pages_completed": 0,
            "created_at": now,
            "updated_at": now,
        }

    def _save_progress(self, annotator_id: str, progress: dict) -> None:
        """Persist ``progress.json`` for *annotator_id*."""
        path = self._progress_path(annotator_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        progress["updated_at"] = datetime.now(timezone.utc).isoformat()
        with open(path, "w") as fh:
            json.dump(progress, fh, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # Progress summary
    # ------------------------------------------------------------------

    def get_progress_summary(self, annotator_id: str) -> dict:
        """Return progress statistics for an annotator.

        Includes total / answered / remaining counts broken down by question
        type and by scene.
        """
        progress = self._load_progress(annotator_id)
        all_qs = self._load_all_questions()

        total = 0
        answered = 0
        by_type: Dict[str, Dict[str, int]] = {
            qt: {"total": 0, "answered": 0} for qt in QUESTION_TYPES
        }
        scenes_touched = 0
        scenes_complete = 0

        for scene_id, scene_qs in all_qs.items():
            answered_qids: Set[str] = set(progress.get("answered", {}).get(scene_id, []))
            scene_total = 0
            scene_answered = 0

            for qtype, questions in scene_qs.items():
                n = len(questions)
                a = sum(1 for q in questions if q["qid"] in answered_qids)
                by_type[qtype]["total"] += n
                by_type[qtype]["answered"] += a
                scene_total += n
                scene_answered += a

            total += scene_total
            answered += scene_answered
            if scene_answered > 0:
                scenes_touched += 1
            if scene_answered == scene_total and scene_total > 0:
                scenes_complete += 1

        return {
            "annotator_id": slugify(annotator_id),
            "total_questions": total,
            "answered_questions": answered,
            "remaining_questions": total - answered,
            "progress_pct": round(100.0 * answered / total, 1) if total > 0 else 0.0,
            "by_type": by_type,
            "total_scenes": len(all_qs),
            "scenes_touched": scenes_touched,
            "scenes_complete": scenes_complete,
            "pages_completed": progress.get("pages_completed", 0),
        }

    # ------------------------------------------------------------------
    # Scene selection
    # ------------------------------------------------------------------

    def _select_scene(self, progress: dict) -> Optional[str]:
        """Select the best scene for the next page.

        Priority order:
        1. Untouched scenes (no answered questions yet) -- pick the first one
           in sorted order for determinism.
        2. Scenes with the most unanswered questions.

        Returns ``None`` if every question in every scene has been answered.
        """
        all_qs = self._load_all_questions()
        answered_map: Dict[str, Set[str]] = {
            sid: set(qids)
            for sid, qids in progress.get("answered", {}).items()
        }

        untouched: List[str] = []
        partial: List[Tuple[int, str]] = []  # (unanswered_count, scene_id)

        for scene_id in sorted(all_qs.keys()):
            scene_qs = all_qs[scene_id]
            total = sum(len(qs) for qs in scene_qs.values())
            n_answered = len(answered_map.get(scene_id, set()))

            remaining = total - n_answered
            if remaining <= 0:
                continue

            if n_answered == 0:
                untouched.append(scene_id)
            else:
                partial.append((remaining, scene_id))

        if untouched:
            return untouched[0]

        if partial:
            # Pick the scene with the most unanswered questions.
            partial.sort(key=lambda pair: (-pair[0], pair[1]))
            return partial[0][1]

        return None

    # ------------------------------------------------------------------
    # Question selection
    # ------------------------------------------------------------------

    def _select_questions(
        self,
        scene_id: str,
        scene_questions: Dict[str, List[dict]],
        answered_qids: Set[str],
        page_size: int,
        min_page_size: int,
    ) -> List[dict]:
        """Select questions for a page, mixing types proportionally.

        Each returned question dict is augmented with a ``"_repeat"`` boolean
        flag indicating whether it was previously answered.

        Algorithm:
        1. Compute per-type target counts proportional to ``_TYPE_WEIGHTS``,
           scaled to *page_size*.
        2. Fill from unanswered questions first; if a type has fewer
           unanswered than its target, take what is available.
        3. If total selected < *min_page_size*, supplement with already-answered
           questions (marked as repeats).
        """
        # Separate unanswered and answered pools per type.
        unanswered: Dict[str, List[dict]] = {}
        answered_pool: Dict[str, List[dict]] = {}
        for qtype, questions in scene_questions.items():
            unanswered[qtype] = [q for q in questions if q["qid"] not in answered_qids]
            answered_pool[qtype] = [q for q in questions if q["qid"] in answered_qids]

        # Compute per-type targets.
        active_types = [qt for qt in QUESTION_TYPES if qt in scene_questions and scene_questions[qt]]
        if not active_types:
            return []

        total_weight = sum(_TYPE_WEIGHTS.get(qt, 1.0) for qt in active_types)
        targets: Dict[str, int] = {}
        assigned = 0
        for qt in active_types:
            t = max(1, round(page_size * _TYPE_WEIGHTS.get(qt, 1.0) / total_weight))
            targets[qt] = t
            assigned += t

        # Adjust rounding so sum == page_size.
        diff = page_size - assigned
        if diff != 0 and active_types:
            # Add/subtract from the type with the largest target.
            adjust_type = max(active_types, key=lambda qt: targets[qt])
            targets[adjust_type] = max(1, targets[adjust_type] + diff)

        selected: List[dict] = []

        # Phase 1: fill with unanswered questions.
        for qt in QUESTION_TYPES:
            if qt not in targets:
                continue
            pool = list(unanswered.get(qt, []))
            random.shuffle(pool)
            take = min(targets[qt], len(pool))
            for q in pool[:take]:
                entry = dict(q)
                entry["_repeat"] = False
                selected.append(entry)

        # Phase 2: if under-filled, try unanswered from other types that have surplus.
        if len(selected) < min_page_size:
            remaining_needed = min_page_size - len(selected)
            selected_qids = {q["qid"] for q in selected}
            surplus: List[dict] = []
            for qt in QUESTION_TYPES:
                pool = unanswered.get(qt, [])
                for q in pool:
                    if q["qid"] not in selected_qids:
                        surplus.append(q)
            random.shuffle(surplus)
            for q in surplus[:remaining_needed]:
                entry = dict(q)
                entry["_repeat"] = False
                selected.append(entry)

        # Phase 3: if still under min_page_size, supplement with repeats.
        if len(selected) < min_page_size:
            remaining_needed = min_page_size - len(selected)
            selected_qids = {q["qid"] for q in selected}
            repeats: List[dict] = []
            for qt in QUESTION_TYPES:
                pool = answered_pool.get(qt, [])
                for q in pool:
                    if q["qid"] not in selected_qids:
                        repeats.append(q)
            random.shuffle(repeats)
            for q in repeats[:remaining_needed]:
                entry = dict(q)
                entry["_repeat"] = True
                selected.append(entry)

        return selected

    # ------------------------------------------------------------------
    # Image bundle
    # ------------------------------------------------------------------

    def _build_page_images(self, scene_id: str, test_type: str) -> dict:
        """Build image paths for the page.

        Returns a dict with ``single_view``, ``multi_view``, and optionally
        ``labeled_single_view`` / ``labeled_multi_view`` keys.
        """
        bundle: Dict[str, Any] = {
            "single_view": f"images/single_view/{scene_id}.png",
            "multi_view": [],
        }

        # Add multi-view images.
        mv_dir = self.multi_view_images_dir / scene_id
        if mv_dir.is_dir():
            for i in range(4):
                view_path = f"images/multi_view/{scene_id}/view_{i}.png"
                bundle["multi_view"].append(view_path)

        # Include labeled versions from tasks_dir if they exist.
        labeled_dir = self.tasks_dir / "images" / scene_id
        if labeled_dir.is_dir():
            labeled_sv = labeled_dir / "single_view_labels.png"
            if labeled_sv.is_file():
                bundle["labeled_single_view"] = (
                    f"tasks/images/{scene_id}/single_view_labels.png"
                )

            labeled_mv: List[str] = []
            for i in range(4):
                labeled_path = labeled_dir / f"view_{i}_labels.png"
                if labeled_path.is_file():
                    labeled_mv.append(
                        f"tasks/images/{scene_id}/view_{i}_labels.png"
                    )
            if labeled_mv:
                bundle["labeled_multi_view"] = labeled_mv

        return bundle

    # ------------------------------------------------------------------
    # Page allocation
    # ------------------------------------------------------------------

    def allocate_page(self, annotator_id: str, test_type: str = "single_view") -> Optional[dict]:
        """Allocate the next page of questions for *annotator_id*.

        A page consists of one scene and 10-15 questions drawn proportionally
        from QRR, TRR, and FDR pools.

        Parameters
        ----------
        annotator_id:
            Identifier for the human annotator.
        test_type:
            ``"single_view"`` or ``"multi_view"``.

        Returns
        -------
        dict or None
            Page allocation dict with keys: ``page_id``, ``scene_id``,
            ``test_type``, ``objects``, ``images``, ``questions``, ``n_new``,
            ``n_repeat``.  Returns ``None`` when all questions are exhausted.
        """
        progress = self._load_progress(annotator_id)
        all_qs = self._load_all_questions()

        scene_id = self._select_scene(progress)
        if scene_id is None:
            return None

        scene_qs = all_qs.get(scene_id, {})
        answered_qids: Set[str] = set(progress.get("answered", {}).get(scene_id, []))

        selected = self._select_questions(
            scene_id=scene_id,
            scene_questions=scene_qs,
            answered_qids=answered_qids,
            page_size=self.page_size,
            min_page_size=self.min_page_size,
        )

        if not selected:
            return None

        # Load scene document for object info.
        try:
            scene_doc = load_scene_doc(scene_id, str(self.scenes_dir))
        except (FileNotFoundError, json.JSONDecodeError):
            scene_doc = {"objects": []}

        objects = build_object_catalog(scene_doc)
        object_lookup = build_object_lookup(objects)

        # Enrich questions with prompt text and answer metadata.
        enriched: List[dict] = []
        for q in selected:
            entry = dict(q)
            entry["prompt_text"] = format_human_question(q, object_lookup)
            if q["type"] == "qrr":
                entry["answer_options"] = QRR_OPTIONS
            elif q["type"] == "trr":
                entry["answer_options"] = TRR_OPTIONS
            elif q["type"] == "fdr":
                candidates = [obj for obj in objects if obj["id"] != q.get("anchor")]
                entry["ranking_candidates"] = [
                    {"id": obj["id"], "desc": obj["desc"]} for obj in candidates
                ]
            enriched.append(entry)

        n_new = sum(1 for q in selected if not q.get("_repeat", False))
        n_repeat = sum(1 for q in selected if q.get("_repeat", False))

        page_number = progress.get("pages_completed", 0) + 1
        page_id = f"{slugify(annotator_id)}_page_{page_number:04d}"

        images = self._build_page_images(scene_id, test_type)

        return {
            "page_id": page_id,
            "scene_id": scene_id,
            "test_type": test_type,
            "n_objects": len(objects),
            "objects": objects,
            "images": images,
            "questions": enriched,
            "n_new": n_new,
            "n_repeat": n_repeat,
            "total_questions": len(enriched),
        }

    # ------------------------------------------------------------------
    # Page submission
    # ------------------------------------------------------------------

    def submit_page(self, annotator_id: str, page_submission: dict) -> dict:
        """Record page responses and persist them.

        Parameters
        ----------
        annotator_id:
            The annotator who completed the page.
        page_submission:
            Must contain at least ``page_id``, ``scene_id``, ``test_type``,
            and ``responses`` (a list of ``{"qid": ..., "answer": ...}``
            dicts).

        Returns
        -------
        dict
            Updated progress summary for the annotator.
        """
        safe_id = slugify(annotator_id)
        scene_id = page_submission["scene_id"]
        page_id = page_submission.get("page_id", f"{safe_id}_{scene_id}")
        test_type = page_submission.get("test_type", "single_view")
        responses: List[dict] = page_submission.get("responses", [])
        now = datetime.now(timezone.utc).isoformat()

        # -- Update progress ------------------------------------------------
        progress = self._load_progress(annotator_id)
        answered_map: Dict[str, List[str]] = progress.setdefault("answered", {})
        scene_answered: Set[str] = set(answered_map.get(scene_id, []))
        for resp in responses:
            qid = resp.get("qid")
            if qid:
                scene_answered.add(qid)
        answered_map[scene_id] = sorted(scene_answered)
        progress["pages_completed"] = progress.get("pages_completed", 0) + 1
        self._save_progress(annotator_id, progress)

        # -- Build analyze_responses-compatible JSON payload ----------------
        # Group responses by question type for batch structure.
        all_qs = self._load_all_questions()
        scene_qs = all_qs.get(scene_id, {})
        qid_to_type: Dict[str, str] = {}
        for qtype, questions in scene_qs.items():
            for q in questions:
                qid_to_type[q["qid"]] = qtype

        batches_by_type: Dict[str, List[dict]] = {}
        for resp in responses:
            qtype = qid_to_type.get(resp.get("qid", ""), "unknown")
            batches_by_type.setdefault(qtype, []).append(resp)

        batches: List[dict] = []
        for qtype in QUESTION_TYPES:
            type_responses = batches_by_type.get(qtype)
            if not type_responses:
                continue
            batches.append({
                "batch_id": f"{qtype}_page",
                "question_type": qtype,
                "responses": type_responses,
                "raw_response": json.dumps(type_responses, ensure_ascii=False),
            })

        payload = {
            "schema_version": 1,
            "scene_id": scene_id,
            "annotator_id": safe_id,
            "model": f"human/{safe_id}",
            "submitted_at": now,
            "test_type": test_type,
            "page_id": page_id,
            "batches": batches,
            "responses": responses,
            "raw_response": json.dumps(responses, ensure_ascii=False),
        }

        # Persist the page response JSON.
        out_dir = self.responses_dir / safe_id
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{scene_id}__{page_id}.json"
        with open(out_path, "w") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)

        return self.get_progress_summary(annotator_id)
