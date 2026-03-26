"""人类基准测试的问题分配与进度追踪 SessionManager。

负责加载测试场景、向标注者分配问题页、
追踪各标注者的进度，以及以兼容 analyze_responses.py 的格式保存回答。
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


# 组成一页时各题型的默认比例权重。
# 以页面大小为 12 时的目标数量为基准，实际按 page_size 等比缩放。
_TYPE_WEIGHTS: Dict[str, float] = {
    "qrr": 5.0,
    "trr": 4.0,
    "fdr": 1.5,
}


class SessionManager:
    """管理人类基准标注的问题分配、进度追踪与回答提交。

    每位标注者拥有独立的进度文件，记录已回答的问题 ID。
    每次按单场景分配一页问题，QRR、TRR 和 FDR 按比例混合。
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
    # 场景发现
    # ------------------------------------------------------------------

    def _load_test_scene_ids(self) -> List[str]:
        """加载仅属于测试集的场景 ID。

        优先读取显式指定的 test_scenes_file（默认为 ``data-gen/output/test_scenes.json``）。
        若文件不存在，则回退到从问题目录中发现场景 ID 并过滤数字索引 >= 80 的条目。
        """
        if self._test_scene_ids is not None:
            return self._test_scene_ids

        # 优先尝试显式指定的测试场景清单。
        candidates: List[Path] = []
        if self.test_scenes_file is not None:
            candidates.append(self.test_scenes_file)
        # 约定：test_scenes.json 与场景目录同级。
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

        # 回退：从问题目录发现场景 ID 并过滤索引 >= 80 的条目。
        all_ids: Set[str] = set()
        for qtype in QUESTION_TYPES:
            qtype_dir = self.questions_dir / qtype
            if qtype_dir.is_dir():
                all_ids.update(p.stem for p in qtype_dir.glob("*.json"))

        test_ids: List[str] = []
        for sid in sorted(all_ids):
            # 场景 ID 格式为 ``nXX_YYYYYY``，其中 YYYYYY 为数字索引。
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
        """加载所有测试场景的全部问题。

        返回 ``{scene_id: {qtype: [question, ...]}}``。
        每个问题字典为问题 JSON 文件中的原始问题对象。
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
    # 进度持久化
    # ------------------------------------------------------------------

    def _progress_path(self, annotator_id: str) -> Path:
        """返回标注者进度文件的路径。"""
        safe_id = slugify(annotator_id)
        return self.responses_dir / safe_id / "progress.json"

    def _load_progress(self, annotator_id: str) -> dict:
        """加载或创建 *annotator_id* 的 ``progress.json``。

        进度结构::

            {
                "annotator_id": "...",
                "answered": {"scene_id": ["qid", ...]},
                "pages_completed": 0,
                "created_at": "ISO 时间戳",
                "updated_at": "ISO 时间戳"
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
        """将 *annotator_id* 的 ``progress.json`` 持久化到磁盘。"""
        path = self._progress_path(annotator_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        progress["updated_at"] = datetime.now(timezone.utc).isoformat()
        with open(path, "w") as fh:
            json.dump(progress, fh, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # 进度汇总
    # ------------------------------------------------------------------

    def get_progress_summary(self, annotator_id: str) -> dict:
        """返回标注者的进度统计信息。

        按题型和场景细分，包含总数 / 已回答 / 剩余数量。
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
    # 场景选择
    # ------------------------------------------------------------------

    def _select_scene(self, progress: dict) -> Optional[str]:
        """为下一页选取最合适的场景。

        优先级顺序：
        1. 未接触场景（尚无已回答问题）——按排序顺序取第一个以保证确定性。
        2. 未回答问题最多的场景。

        若所有场景的所有问题均已回答，则返回 ``None``。
        """
        all_qs = self._load_all_questions()
        answered_map: Dict[str, Set[str]] = {
            sid: set(qids)
            for sid, qids in progress.get("answered", {}).items()
        }

        untouched: List[str] = []
        partial: List[Tuple[int, str]] = []  # (未回答数量, 场景 ID)

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
            # 选取未回答问题最多的场景。
            partial.sort(key=lambda pair: (-pair[0], pair[1]))
            return partial[0][1]

        return None

    # ------------------------------------------------------------------
    # 问题选择
    # ------------------------------------------------------------------

    def _select_questions(
        self,
        scene_id: str,
        scene_questions: Dict[str, List[dict]],
        answered_qids: Set[str],
        page_size: int,
        min_page_size: int,
    ) -> List[dict]:
        """按比例混合各题型，为一页选取问题。

        每个返回的问题字典附加 ``"_repeat"`` 布尔标志，
        指示该问题是否已被回答过。

        算法：
        1. 按 ``_TYPE_WEIGHTS`` 比例计算各题型的目标数量，并缩放至 *page_size*。
        2. 优先从未回答问题中填充；若某题型未回答数量不足目标值，则取所有可用问题。
        3. 若已选总数 < *min_page_size*，则补充已回答问题（标记为重复）。
        """
        # 按题型分别建立未回答和已回答问题池。
        unanswered: Dict[str, List[dict]] = {}
        answered_pool: Dict[str, List[dict]] = {}
        for qtype, questions in scene_questions.items():
            unanswered[qtype] = [q for q in questions if q["qid"] not in answered_qids]
            answered_pool[qtype] = [q for q in questions if q["qid"] in answered_qids]

        # 计算各题型的目标数量。
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

        # 调整舍入误差使总和等于 page_size。
        diff = page_size - assigned
        if diff != 0 and active_types:
            # 对目标数量最大的题型进行加减调整。
            adjust_type = max(active_types, key=lambda qt: targets[qt])
            targets[adjust_type] = max(1, targets[adjust_type] + diff)

        selected: List[dict] = []

        # 第一阶段：用未回答问题填充。
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

        # 第二阶段：若填充不足，从有剩余的其他题型中补充未回答问题。
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

        # 第三阶段：若仍未达到最小页面大小，用已回答问题补充（标记为重复）。
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
    # 图像包
    # ------------------------------------------------------------------

    def _build_page_images(self, scene_id: str, test_type: str) -> dict:
        """为当前页面构建图像路径。

        返回包含 ``single_view``、``multi_view`` 键以及可选的
        ``labeled_single_view`` / ``labeled_multi_view`` 键的字典。
        """
        bundle: Dict[str, Any] = {
            "single_view": f"images/single_view/{scene_id}.png",
            "multi_view": [],
        }

        # 添加多视角图像。
        mv_dir = self.multi_view_images_dir / scene_id
        if mv_dir.is_dir():
            for i in range(4):
                view_path = f"images/multi_view/{scene_id}/view_{i}.png"
                bundle["multi_view"].append(view_path)

        # 若 tasks_dir 中存在带标注版本，则一并包含。
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
    # 页面分配
    # ------------------------------------------------------------------

    def allocate_page(self, annotator_id: str, test_type: str = "single_view") -> Optional[dict]:
        """为 *annotator_id* 分配下一页问题。

        一页由一个场景和 10-15 道题组成，按比例从 QRR、TRR、FDR 问题池中抽取。

        参数
        ----------
        annotator_id:
            人类标注者的标识符。
        test_type:
            ``"single_view"`` 或 ``"multi_view"``。

        返回值
        -------
        dict 或 None
            包含以下键的页面分配字典：``page_id``、``scene_id``、
            ``test_type``、``objects``、``images``、``questions``、``n_new``、
            ``n_repeat``。所有问题耗尽时返回 ``None``。
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

        # 加载场景文档以获取物体信息。
        try:
            scene_doc = load_scene_doc(scene_id, str(self.scenes_dir))
        except (FileNotFoundError, json.JSONDecodeError):
            scene_doc = {"objects": []}

        objects = build_object_catalog(scene_doc)
        object_lookup = build_object_lookup(objects)

        # 为问题附加提示文本和答案元数据。
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
    # 页面提交
    # ------------------------------------------------------------------

    def submit_page(self, annotator_id: str, page_submission: dict) -> dict:
        """记录并持久化一页的回答。

        参数
        ----------
        annotator_id:
            完成本页的标注者。
        page_submission:
            至少须包含 ``page_id``、``scene_id``、``test_type``
            以及 ``responses``（``{"qid": ..., "answer": ...}`` 字典列表）。

        返回值
        -------
        dict
            该标注者的最新进度汇总。
        """
        safe_id = slugify(annotator_id)
        scene_id = page_submission["scene_id"]
        page_id = page_submission.get("page_id", f"{safe_id}_{scene_id}")
        test_type = page_submission.get("test_type", "single_view")
        responses: List[dict] = page_submission.get("responses", [])
        now = datetime.now(timezone.utc).isoformat()

        # -- 更新进度 ------------------------------------------------
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

        # -- 构建兼容 analyze_responses 的 JSON 载荷 ----------------
        # 按题型分组回答以构建批次结构。
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

        # 将本页回答的 JSON 持久化到磁盘。
        out_dir = self.responses_dir / safe_id
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{scene_id}__{page_id}.json"
        with open(out_path, "w") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)

        return self.get_progress_summary(annotator_id)
