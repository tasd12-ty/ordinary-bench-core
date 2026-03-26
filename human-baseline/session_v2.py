"""人类基准测试的渐进式会话管理器。

以场景为中心的模型，分三轮逐步增加视觉信息：
  第 1 轮：单视角（1 张图像）
  第 2 轮：四视角（4 张多视角图像）
  第 3 轮：五视角（单视角 + 4 张多视角图像）

每轮仅呈现前一轮回答错误的问题。
当所有问题均正确（或完成第 3 轮）后，该场景标记为已完成并分配下一个场景。
"""
from __future__ import annotations

import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

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
from grading import grade_round

# 各轮次对应的视角模式标签。
VIEW_MODES = {
    1: "single_view",
    2: "four_view",
    3: "five_view",
}

VIEW_MODE_LABELS = {
    1: "单视角",
    2: "四视角",
    3: "五视角",
}

# TRR 目标采样数量。
TRR_SAMPLE_SIZE = 5


class ProgressiveSessionManager:
    """管理逐场景渐进式测试，并在每轮之间进行评分。"""

    def __init__(
        self,
        questions_dir: str | Path,
        scenes_dir: str | Path,
        images_dir: str | Path,
        multi_view_images_dir: str | Path,
        tasks_dir: str | Path,
        responses_dir: str | Path,
        test_scenes_file: Optional[str | Path] = None,
    ) -> None:
        self.questions_dir = Path(questions_dir)
        self.scenes_dir = Path(scenes_dir)
        self.images_dir = Path(images_dir)
        self.multi_view_images_dir = Path(multi_view_images_dir)
        self.tasks_dir = Path(tasks_dir)
        self.responses_dir = Path(responses_dir)
        self.test_scenes_file = Path(test_scenes_file) if test_scenes_file else None

        self._test_scene_ids: Optional[List[str]] = None
        self._all_questions: Optional[Dict[str, Dict[str, List[dict]]]] = None

    # ------------------------------------------------------------------
    # 场景发现
    # ------------------------------------------------------------------

    def _load_test_scene_ids(self) -> List[str]:
        if self._test_scene_ids is not None:
            return self._test_scene_ids

        candidates: List[Path] = []
        if self.test_scenes_file is not None:
            candidates.append(self.test_scenes_file)
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

        all_ids: Set[str] = set()
        for qtype in QUESTION_TYPES:
            qtype_dir = self.questions_dir / qtype
            if qtype_dir.is_dir():
                all_ids.update(p.stem for p in qtype_dir.glob("*.json"))

        test_ids: List[str] = []
        for sid in sorted(all_ids):
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
    # 问题加载
    # ------------------------------------------------------------------

    def _load_all_questions(self) -> Dict[str, Dict[str, List[dict]]]:
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
    # 进度持久化
    # ------------------------------------------------------------------

    def _progress_path(self, annotator_id: str) -> Path:
        safe_id = slugify(annotator_id)
        return self.responses_dir / safe_id / "progress_v2.json"

    def _load_progress(self, annotator_id: str) -> dict:
        path = self._progress_path(annotator_id)
        if path.is_file():
            with open(path) as fh:
                return json.load(fh)

        now = datetime.now(timezone.utc).isoformat()
        return {
            "annotator_id": slugify(annotator_id),
            "scenes_completed": [],
            "current_scene": None,
            "created_at": now,
            "updated_at": now,
        }

    def _save_progress(self, annotator_id: str, progress: dict) -> None:
        path = self._progress_path(annotator_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        progress["updated_at"] = datetime.now(timezone.utc).isoformat()
        with open(path, "w") as fh:
            json.dump(progress, fh, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # 问题选择
    # ------------------------------------------------------------------

    def _select_questions(
        self,
        scene_id: str,
        scene_questions: Dict[str, List[dict]],
        annotator_id: str,
    ) -> List[dict]:
        """为一个场景选取约 12 道题：全部 QRR + 5 道 TRR + 全部 FDR。"""
        seed = int(hashlib.md5(f"{scene_id}:{annotator_id}".encode()).hexdigest(), 16)
        rng = random.Random(seed)

        selected: List[dict] = []

        # 全部 QRR 问题。
        selected.extend(scene_questions.get("qrr", []))

        # 采样 TRR 问题。
        trr_pool = list(scene_questions.get("trr", []))
        rng.shuffle(trr_pool)
        selected.extend(trr_pool[:TRR_SAMPLE_SIZE])

        # 全部 FDR 问题。
        selected.extend(scene_questions.get("fdr", []))

        return selected

    # ------------------------------------------------------------------
    # 图像包
    # ------------------------------------------------------------------

    def _build_images(self, scene_id: str, round_number: int) -> dict:
        """为指定轮次构建图像路径。

        第 1 轮：仅单视角。
        第 2 轮：四张多视角图像。
        第 3 轮：单视角 + 四张多视角图像。
        """
        bundle: Dict[str, Any] = {}

        sv_path = f"images/single_view/{scene_id}.png"
        mv_paths: List[str] = []
        mv_dir = self.multi_view_images_dir / scene_id
        if mv_dir.is_dir():
            for i in range(4):
                mv_paths.append(f"images/multi_view/{scene_id}/view_{i}.png")

        # tasks 目录中的带标注版本。
        labeled_dir = self.tasks_dir / "images" / scene_id
        labeled_sv = None
        labeled_mv: List[str] = []

        if labeled_dir.is_dir():
            lsv = labeled_dir / "single_view_labels.png"
            if lsv.is_file():
                labeled_sv = f"tasks/images/{scene_id}/single_view_labels.png"
            for i in range(4):
                lmv = labeled_dir / f"view_{i}_labels.png"
                if lmv.is_file():
                    labeled_mv.append(f"tasks/images/{scene_id}/view_{i}_labels.png")

        if round_number == 1:
            bundle["single_view"] = labeled_sv or sv_path
        elif round_number == 2:
            bundle["multi_view"] = labeled_mv if labeled_mv else mv_paths
        elif round_number == 3:
            bundle["single_view"] = labeled_sv or sv_path
            bundle["multi_view"] = labeled_mv if labeled_mv else mv_paths

        return bundle

    # ------------------------------------------------------------------
    # 场景分配
    # ------------------------------------------------------------------

    def allocate_scene(self, annotator_id: str) -> Optional[dict]:
        """分配下一个场景并准备第 1 轮数据。

        返回包含第 1 轮配置的场景数据，若所有场景已完成则返回 None。
        """
        progress = self._load_progress(annotator_id)
        all_qs = self._load_all_questions()
        completed = set(progress.get("scenes_completed", []))

        # 查找下一个未测试的场景。
        scene_id = None
        for sid in self._load_test_scene_ids():
            if sid not in completed and sid in all_qs:
                scene_id = sid
                break

        if scene_id is None:
            return None

        scene_qs = all_qs[scene_id]
        selected = self._select_questions(scene_id, scene_qs, annotator_id)

        if not selected:
            return None

        # 加载场景文档。
        try:
            scene_doc = load_scene_doc(scene_id, str(self.scenes_dir))
        except (FileNotFoundError, json.JSONDecodeError):
            scene_doc = {"objects": []}

        objects = build_object_catalog(scene_doc)
        object_lookup = build_object_lookup(objects)

        # 为问题附加元数据。
        enriched = self._enrich_questions(selected, objects, object_lookup)

        # 将当前场景保存到进度中。
        progress["current_scene"] = {
            "scene_id": scene_id,
            "selected_questions": selected,
            "enriched_questions": enriched,
            "objects": objects,
            "rounds": [],
            "status": "in_progress",
        }
        self._save_progress(annotator_id, progress)

        return self.get_current_round(annotator_id)

    def _enrich_questions(
        self,
        questions: List[dict],
        objects: List[dict],
        object_lookup: Dict[str, str],
    ) -> List[dict]:
        """为问题附加 prompt_text、answer_options 和 ranking_candidates 字段。"""
        enriched: List[dict] = []
        for q in questions:
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
        return enriched

    # ------------------------------------------------------------------
    # 当前轮次
    # ------------------------------------------------------------------

    def get_current_round(self, annotator_id: str) -> Optional[dict]:
        """获取标注者当前轮次的数据。

        返回包含 scene_id、round_number、view_mode、images、objects、
        questions 等字段的字典。若无正在进行的场景则返回 None。
        """
        progress = self._load_progress(annotator_id)
        current = progress.get("current_scene")

        if current is None:
            return None

        if current.get("status") == "completed":
            return None

        scene_id = current["scene_id"]
        rounds = current.get("rounds", [])

        # 确定当前轮次编号及需要呈现的问题。
        if not rounds:
            round_number = 1
            qids_to_present = [q["qid"] for q in current["selected_questions"]]
        else:
            last_round = rounds[-1]
            wrong_qids = last_round.get("wrong_qids", [])
            if not wrong_qids or last_round["round_number"] >= 3:
                # 场景应已完成。
                return None
            round_number = last_round["round_number"] + 1
            qids_to_present = wrong_qids

        qids_set = set(qids_to_present)
        enriched = current.get("enriched_questions", [])
        questions_for_round = [q for q in enriched if q["qid"] in qids_set]

        # 保留原始顺序。
        qid_order = {qid: i for i, qid in enumerate(qids_to_present)}
        questions_for_round.sort(key=lambda q: qid_order.get(q["qid"], 0))

        images = self._build_images(scene_id, round_number)
        objects = current.get("objects", [])

        n_total_selected = len(current["selected_questions"])

        return {
            "scene_id": scene_id,
            "round_number": round_number,
            "view_mode": VIEW_MODES[round_number],
            "round_label": f"第{round_number}轮 — {VIEW_MODE_LABELS[round_number]}",
            "images": images,
            "objects": objects,
            "questions": questions_for_round,
            "n_total_selected": n_total_selected,
            "n_this_round": len(questions_for_round),
            "previous_rounds": [
                {
                    "round_number": r["round_number"],
                    "view_mode": r["view_mode"],
                    "n_correct": r["grading"]["n_correct"],
                    "n_total": r["grading"]["n_total"],
                }
                for r in rounds
            ],
        }

    # ------------------------------------------------------------------
    # 轮次提交
    # ------------------------------------------------------------------

    def submit_round(self, annotator_id: str, submission: dict) -> dict:
        """提交当前轮次的回答，进行评分并推进进度。

        参数
        ----------
        submission:
            包含 ``scene_id``、``round_number``、``responses``
            （{qid, answer} 列表）以及 ``elapsed_seconds`` 的字典。

        返回值
        -------
        包含 ``grading``、``next_action``、``progress`` 的字典。
        """
        progress = self._load_progress(annotator_id)
        current = progress.get("current_scene")

        if current is None:
            raise ValueError("No scene in progress")

        scene_id = current["scene_id"]
        if submission.get("scene_id") != scene_id:
            raise ValueError(f"Scene mismatch: expected {scene_id}")

        rounds = current.get("rounds", [])
        expected_round = len(rounds) + 1
        submitted_round = submission.get("round_number", expected_round)

        if submitted_round != expected_round:
            raise ValueError(f"Round mismatch: expected {expected_round}, got {submitted_round}")

        # 构建答案映射。
        answers: Dict[str, Any] = {}
        for resp in submission.get("responses", []):
            answers[resp["qid"]] = resp["answer"]

        # 确定本轮包含的问题。
        all_selected = current["selected_questions"]
        if not rounds:
            questions_this_round = all_selected
        else:
            wrong_qids = set(rounds[-1].get("wrong_qids", []))
            questions_this_round = [q for q in all_selected if q["qid"] in wrong_qids]

        # 评分。
        grading_result = grade_round(questions_this_round, answers)

        # 记录本轮数据。
        round_record = {
            "round_number": submitted_round,
            "view_mode": VIEW_MODES.get(submitted_round, "unknown"),
            "qids_presented": [q["qid"] for q in questions_this_round],
            "answers": answers,
            "grading": grading_result,
            "wrong_qids": grading_result["wrong_qids"],
            "submitted_at": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": submission.get("elapsed_seconds", 0),
        }
        rounds.append(round_record)
        current["rounds"] = rounds

        # 确定下一步操作。
        if grading_result["all_correct"] or submitted_round >= 3:
            next_action = self._complete_scene(annotator_id, progress)
        else:
            next_action = "next_round"
            self._save_progress(annotator_id, progress)

        return {
            "grading": grading_result,
            "next_action": next_action,
            "progress": self.get_progress_summary(annotator_id),
        }

    def _complete_scene(self, annotator_id: str, progress: dict) -> str:
        """将当前场景标记为已完成并保存回答文件。"""
        current = progress["current_scene"]
        scene_id = current["scene_id"]

        current["status"] = "completed"
        completed = progress.setdefault("scenes_completed", [])
        if scene_id not in completed:
            completed.append(scene_id)

        # 构建最终答案：后面轮次的答案覆盖前面轮次的答案。
        final_answers: Dict[str, Any] = {}
        for r in current["rounds"]:
            for qid, answer in r.get("answers", {}).items():
                final_answers[qid] = answer

        # 保存兼容 analyze_responses 的回答文件。
        self._save_response_file(annotator_id, current, final_answers)

        # 清除当前场景。
        progress["current_scene"] = None
        self._save_progress(annotator_id, progress)

        # 检查是否还有剩余场景。
        all_qs = self._load_all_questions()
        completed_set = set(completed)
        has_more = any(
            sid not in completed_set
            for sid in self._load_test_scene_ids()
            if sid in all_qs
        )
        return "scene_complete" if has_more else "all_done"

    def _save_response_file(
        self,
        annotator_id: str,
        scene_data: dict,
        final_answers: Dict[str, Any],
    ) -> None:
        """保存兼容 analyze_responses.py 的回答文件。"""
        safe_id = slugify(annotator_id)
        scene_id = scene_data["scene_id"]
        rounds = scene_data.get("rounds", [])

        # 构建回答列表。
        responses: List[dict] = []
        for q in scene_data["selected_questions"]:
            qid = q["qid"]
            if qid in final_answers:
                responses.append({"qid": qid, "answer": final_answers[qid]})

        # 按题型分组以构建批次。
        qid_to_type: Dict[str, str] = {}
        for q in scene_data["selected_questions"]:
            qid_to_type[q["qid"]] = q["type"]

        batches_by_type: Dict[str, List[dict]] = {}
        for resp in responses:
            qtype = qid_to_type.get(resp["qid"], "unknown")
            batches_by_type.setdefault(qtype, []).append(resp)

        batches: List[dict] = []
        for qtype in QUESTION_TYPES:
            type_responses = batches_by_type.get(qtype)
            if not type_responses:
                continue
            batches.append({
                "batch_id": f"{qtype}_progressive",
                "question_type": qtype,
                "responses": type_responses,
                "raw_response": json.dumps(type_responses, ensure_ascii=False),
            })

        # 确定回答条件。
        total_rounds = len(rounds)
        if total_rounds == 1 and rounds[0]["grading"]["all_correct"]:
            condition = "round_1_perfect"
        elif total_rounds == 2 and rounds[-1]["grading"]["all_correct"]:
            condition = "round_2_perfect"
        else:
            condition = "round_3_final"

        # 视角模式历史记录。
        view_history = [r["view_mode"] for r in rounds]

        # 各轮次详情。
        round_details = {}
        for r in rounds:
            rn = r["round_number"]
            round_details[f"round_{rn}_correct"] = r["grading"]["n_correct"]
            round_details[f"round_{rn}_total"] = r["grading"]["n_total"]

        now = datetime.now(timezone.utc).isoformat()

        payload = {
            "schema_version": 2,
            "scene_id": scene_id,
            "annotator_id": safe_id,
            "model": f"human/{safe_id}",
            "submitted_at": now,
            "test_type": "progressive",
            "n_objects": len(scene_data.get("objects", [])),
            "response_condition": condition,
            "response_condition_detail": {
                "total_rounds": total_rounds,
                **round_details,
            },
            "view_mode_history": view_history,
            "batches": batches,
            "responses": responses,
            "raw_response": json.dumps(responses, ensure_ascii=False),
        }

        out_dir = self.responses_dir / safe_id
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{scene_id}__{safe_id}.json"
        with open(out_path, "w") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # 进度汇总
    # ------------------------------------------------------------------

    def get_progress_summary(self, annotator_id: str) -> dict:
        progress = self._load_progress(annotator_id)
        all_qs = self._load_all_questions()
        test_scenes = [sid for sid in self._load_test_scene_ids() if sid in all_qs]
        completed = set(progress.get("scenes_completed", []))

        total_scenes = len(test_scenes)
        scenes_completed = len(completed)
        scenes_remaining = total_scenes - scenes_completed

        # 统计所有测试场景的问题总数。
        total_questions = 0
        for sid in test_scenes:
            scene_qs = all_qs.get(sid, {})
            # 使用相同的选择逻辑来计数。
            selected = self._select_questions(sid, scene_qs, annotator_id)
            total_questions += len(selected)

        # 统计已完成场景的已回答问题数。
        answered_questions = 0
        for sid in completed:
            scene_qs = all_qs.get(sid, {})
            selected = self._select_questions(sid, scene_qs, annotator_id)
            answered_questions += len(selected)

        current = progress.get("current_scene")
        current_scene_id = current["scene_id"] if current else None
        current_round = None
        if current and current.get("rounds"):
            current_round = len(current["rounds"]) + 1
        elif current:
            current_round = 1

        return {
            "annotator_id": slugify(annotator_id),
            "total_scenes": total_scenes,
            "scenes_completed": scenes_completed,
            "scenes_remaining": scenes_remaining,
            "total_questions": total_questions,
            "answered_questions": answered_questions,
            "progress_pct": round(100.0 * scenes_completed / total_scenes, 1) if total_scenes > 0 else 0.0,
            "current_scene_id": current_scene_id,
            "current_round": current_round,
        }
