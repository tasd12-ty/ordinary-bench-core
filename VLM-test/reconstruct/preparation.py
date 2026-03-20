"""
Preparation helpers for scene reconstruction.

This module separates:
  1. loading/normalizing question + scene metadata
  2. extracting auditable relation constraints from model-scoring output
  3. serializing a per-scene prepared bundle for later reconstruction
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


VALID_QRR_COMPARATORS = {"<", "~=", ">"}
QUESTION_TYPES = ("qrr", "trr", "fdr")


@dataclass
class PreparedSceneInput:
    """Serializable reconstruction input with audit information."""

    scene_id: str
    model: Optional[str] = None
    use_correct_only: bool = True
    metadata: Dict[str, object] = field(default_factory=dict)
    object_ids: List[str] = field(default_factory=list)
    gt_positions: Dict[str, List[float]] = field(default_factory=dict)
    per_question_audit: List[dict] = field(default_factory=list)
    qrr_constraints: List[dict] = field(default_factory=list)
    trr_constraints: List[dict] = field(default_factory=list)
    fdr_constraints: List[dict] = field(default_factory=list)
    qrr_from_fdr: List[dict] = field(default_factory=list)
    qrr_all: List[dict] = field(default_factory=list)
    summary: Dict[str, object] = field(default_factory=dict)
    integrity: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "scene_id": self.scene_id,
            "model": self.model,
            "use_correct_only": self.use_correct_only,
            "metadata": self.metadata,
            "object_ids": self.object_ids,
            "gt_positions": self.gt_positions,
            "per_question_audit": self.per_question_audit,
            "constraints": {
                "qrr_direct": self.qrr_constraints,
                "trr": self.trr_constraints,
                "fdr": self.fdr_constraints,
                "qrr_from_fdr": self.qrr_from_fdr,
                "qrr_all": self.qrr_all or (self.qrr_constraints + self.qrr_from_fdr),
            },
            "summary": self.summary,
            "integrity": self.integrity,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PreparedSceneInput":
        constraints = data.get("constraints", {})
        qrr_constraints = list(constraints.get("qrr_direct", data.get("qrr_constraints", [])))
        trr_constraints = list(constraints.get("trr", data.get("trr_constraints", [])))
        fdr_constraints = list(constraints.get("fdr", data.get("fdr_constraints", [])))
        qrr_from_fdr = list(constraints.get("qrr_from_fdr", data.get("qrr_from_fdr", [])))
        qrr_all = list(constraints.get("qrr_all", data.get("qrr_all", []))) or (qrr_constraints + qrr_from_fdr)
        return cls(
            scene_id=data["scene_id"],
            model=data.get("model"),
            use_correct_only=bool(data.get("use_correct_only", True)),
            metadata=dict(data.get("metadata", {})),
            object_ids=list(data.get("object_ids", [])),
            gt_positions=dict(data.get("gt_positions", {})),
            per_question_audit=list(data.get("per_question_audit", [])),
            qrr_constraints=qrr_constraints,
            trr_constraints=trr_constraints,
            fdr_constraints=fdr_constraints,
            qrr_from_fdr=qrr_from_fdr,
            qrr_all=qrr_all,
            summary=dict(data.get("summary", {})),
            integrity=dict(data.get("integrity", {})),
        )


def _flatten_batches(doc: dict) -> List[dict]:
    questions: List[dict] = []
    for batch in doc.get("batches", []):
        questions.extend(batch.get("questions", []))
    return questions


def load_questions_auto(questions_dir: str, scene_id: str) -> Tuple[List[dict], dict]:
    """Load flattened questions for a scene from flat or split layout.

    Preference order:
      1. split layout (`questions/{qrr,trr,fdr}/{scene_id}.json`)
      2. legacy flat layout (`questions/{scene_id}.json`)

    When both layouts exist, split is preferred because it is the current
    recommended storage format and avoids silently dropping question types.
    """
    base = Path(questions_dir)
    flat_path = base / f"{scene_id}.json"

    split_paths = {qtype: base / qtype / f"{scene_id}.json" for qtype in QUESTION_TYPES}
    found = {qtype: path for qtype, path in split_paths.items() if path.exists()}
    if found:
        questions: List[dict] = []
        scene_meta = None
        for qtype in QUESTION_TYPES:
            path = found.get(qtype)
            if path is None:
                continue
            with open(path) as f:
                doc = json.load(f)
            if scene_meta is None:
                scene_meta = {
                    "scene_id": doc.get("scene_id"),
                    "image_path": doc.get("image_path"),
                    "n_objects": doc.get("n_objects"),
                }
            questions.extend(_flatten_batches(doc))

        meta = {
            "layout": "split",
            "paths": {qtype: str(path) for qtype, path in found.items()},
            "scene_meta": scene_meta or {},
        }

        if flat_path.exists():
            with open(flat_path) as f:
                flat_doc = json.load(f)
            flat_questions = _flatten_batches(flat_doc)
            meta["alternate_flat_path"] = str(flat_path)
            meta["alternate_flat_question_count"] = len(flat_questions)
            if len(flat_questions) != len(questions):
                meta["layout_warning"] = (
                    f"flat/split mismatch for {scene_id}: "
                    f"flat={len(flat_questions)} split={len(questions)}"
                )
        return questions, meta

    if flat_path.exists():
        with open(flat_path) as f:
            doc = json.load(f)
        return _flatten_batches(doc), {
            "layout": "flat",
            "paths": {"flat": str(flat_path)},
            "scene_meta": {
                "scene_id": doc.get("scene_id"),
                "image_path": doc.get("image_path"),
                "n_objects": doc.get("n_objects"),
            },
        }

    return [], {"layout": "missing", "paths": {}, "scene_meta": {}}


def load_scene_gt_positions(scene_path: str) -> Optional[Dict[str, List[float]]]:
    """Load 2D GT positions from a scene JSON file."""
    path = Path(scene_path)
    if not path.exists():
        return None

    with open(path) as f:
        scene = json.load(f)

    positions = {}
    for obj in scene.get("objects", []):
        coords = obj.get("3d_coords", obj.get("position_3d"))
        if coords is None or len(coords) < 2:
            continue
        positions[obj["id"]] = [float(coords[0]), float(coords[1])]
    return positions or None


def infer_object_ids_from_questions(questions: Iterable[dict]) -> List[str]:
    """Infer the object universe touched by the question set."""
    obj_set = set()
    for q in questions:
        if q["type"] == "qrr":
            obj_set.update(q.get("pair1", []))
            obj_set.update(q.get("pair2", []))
        elif q["type"] == "trr":
            obj_set.update([q["target"], q["ref1"], q["ref2"]])
        elif q["type"] == "fdr":
            obj_set.add(q["anchor"])
            obj_set.update(q.get("gt_ranking", []))
    return sorted(obj_set)


def _find_duplicates(items: Iterable[str]) -> List[str]:
    seen = set()
    dupes = set()
    for item in items:
        if item in seen:
            dupes.add(item)
        seen.add(item)
    return sorted(dupes)


def _canonicalize_fdr_prediction(predicted: object, gt_ranking: List[str], anchor: str) -> Tuple[List[str], Optional[str]]:
    if not isinstance(predicted, list):
        return [], "invalid_prediction_type"

    allowed = set(gt_ranking)
    ranking: List[str] = []
    seen = set()
    for item in predicted:
        if not isinstance(item, str):
            continue
        if item == anchor or item not in allowed or item in seen:
            continue
        ranking.append(item)
        seen.add(item)

    if len(ranking) < 2:
        return ranking, "insufficient_valid_prediction"
    return ranking, None


def _decompose_fdr_record(fdr_constraint: dict) -> List[dict]:
    ranking = list(fdr_constraint["ranking"])
    anchor = fdr_constraint["anchor"]
    n = len(ranking)
    n_pairs = n * (n - 1) // 2 if n >= 2 else 1
    weight = float(fdr_constraint.get("weight", 1.0)) / n_pairs
    source_qid = fdr_constraint["qid"]

    derived = []
    for i in range(len(ranking)):
        for j in range(i + 1, len(ranking)):
            nearer = ranking[i]
            farther = ranking[j]
            derived.append({
                "qid": f"{source_qid}__pair_{i}_{j}",
                "source_qid": source_qid,
                "source_type": "fdr_decomposition",
                "anchor": anchor,
                "pair1": sorted((anchor, nearer)),
                "pair2": sorted((anchor, farther)),
                "comparator": "<",
                "weight": weight,
                "variant": "shared_anchor",
            })
    return derived


def _question_stub(question: dict) -> dict:
    keys = (
        "qid", "type", "variant", "anchor", "pair1", "pair2",
        "target", "ref1", "ref2", "gt_comparator", "gt_hour",
        "gt_quadrant", "gt_ranking", "gt_tie_groups", "metric",
    )
    return {k: question[k] for k in keys if k in question}


def prepare_reconstruction_input_from_scoring(
    scoring_result: dict,
    questions: List[dict],
    gt_positions: Optional[Dict[str, List[float]]] = None,
    scene_id: Optional[str] = None,
    model: Optional[str] = None,
    use_correct_only: bool = True,
    metadata: Optional[Dict[str, object]] = None,
) -> PreparedSceneInput:
    """Prepare an auditable per-scene bundle from scoring output."""
    metadata = dict(metadata or {})
    per_question = list(scoring_result.get("per_question", []))
    q_lookup = {q["qid"]: q for q in questions}
    score_lookup = {row["qid"]: row for row in per_question if "qid" in row}

    duplicate_question_ids = _find_duplicates(q["qid"] for q in questions)
    duplicate_score_ids = _find_duplicates(row.get("qid", "") for row in per_question if row.get("qid"))
    missing_score_qids = [q["qid"] for q in questions if q["qid"] not in score_lookup]
    extra_score_qids = [row["qid"] for row in per_question if row.get("qid") not in q_lookup]

    object_ids = infer_object_ids_from_questions(questions)
    qrr_constraints: List[dict] = []
    trr_constraints: List[dict] = []
    fdr_constraints: List[dict] = []
    qrr_from_fdr: List[dict] = []
    audit_rows: List[dict] = []
    skip_reason_counts: Dict[str, int] = {}

    def mark_skip(reason: str) -> str:
        skip_reason_counts[reason] = skip_reason_counts.get(reason, 0) + 1
        return reason

    for question in questions:
        qid = question["qid"]
        row = score_lookup.get(qid)
        qtype = question["type"]
        audit = {
            "qid": qid,
            "type": qtype,
            "question": _question_stub(question),
            "score": row,
            "selected": False,
            "skip_reason": None,
            "prepared_constraints": [],
        }

        if row is None:
            audit["skip_reason"] = mark_skip("missing_score")
            audit_rows.append(audit)
            continue

        if qtype == "qrr":
            predicted = row.get("predicted")
            if predicted is None:
                audit["skip_reason"] = mark_skip("missing_prediction")
            elif use_correct_only and not row.get("correct", False):
                audit["skip_reason"] = mark_skip("incorrect_answer")
            else:
                comparator = question["gt_comparator"] if use_correct_only else str(predicted)
                if comparator not in VALID_QRR_COMPARATORS:
                    audit["skip_reason"] = mark_skip("invalid_comparator")
                else:
                    constraint = {
                        "qid": qid,
                        "source_type": "qrr",
                        "pair1": list(question["pair1"]),
                        "pair2": list(question["pair2"]),
                        "comparator": comparator,
                        "weight": 1.0,
                        "variant": question.get("variant", "disjoint"),
                        "anchor": question.get("anchor"),
                    }
                    qrr_constraints.append(constraint)
                    audit["selected"] = True
                    audit["prepared_constraints"] = [constraint]

        elif qtype == "trr":
            predicted = row.get("predicted")
            if predicted is None or predicted == -1:
                audit["skip_reason"] = mark_skip("missing_prediction")
            elif use_correct_only:
                if row.get("hour_correct", False):
                    constraint = {
                        "qid": qid,
                        "source_type": "trr",
                        "target": question["target"],
                        "ref1": question["ref1"],
                        "ref2": question["ref2"],
                        "hour": question["gt_hour"],
                        "weight": 1.0,
                        "level": "hour",
                    }
                    trr_constraints.append(constraint)
                    audit["selected"] = True
                    audit["prepared_constraints"] = [constraint]
                elif row.get("quadrant_correct", False):
                    constraint = {
                        "qid": qid,
                        "source_type": "trr",
                        "target": question["target"],
                        "ref1": question["ref1"],
                        "ref2": question["ref2"],
                        "hour": question["gt_hour"],
                        "weight": 0.5,
                        "level": "quadrant",
                    }
                    trr_constraints.append(constraint)
                    audit["selected"] = True
                    audit["prepared_constraints"] = [constraint]
                else:
                    audit["skip_reason"] = mark_skip("incorrect_answer")
            else:
                try:
                    pred_hour = int(predicted)
                except (TypeError, ValueError):
                    pred_hour = -1
                if 1 <= pred_hour <= 12:
                    constraint = {
                        "qid": qid,
                        "source_type": "trr",
                        "target": question["target"],
                        "ref1": question["ref1"],
                        "ref2": question["ref2"],
                        "hour": pred_hour,
                        "weight": 1.0,
                        "level": "hour",
                    }
                    trr_constraints.append(constraint)
                    audit["selected"] = True
                    audit["prepared_constraints"] = [constraint]
                else:
                    audit["skip_reason"] = mark_skip("invalid_hour_prediction")

        elif qtype == "fdr":
            predicted = row.get("predicted", [])
            if use_correct_only:
                if row.get("pairwise_accuracy", 0.0) < 0.5:
                    audit["skip_reason"] = mark_skip("low_pairwise_accuracy")
                else:
                    constraint = {
                        "qid": qid,
                        "source_type": "fdr",
                        "anchor": question["anchor"],
                        "ranking": list(question["gt_ranking"]),
                        "weight": 1.0,
                    }
                    derived = _decompose_fdr_record(constraint)
                    fdr_constraints.append(constraint)
                    qrr_from_fdr.extend(derived)
                    audit["selected"] = True
                    audit["prepared_constraints"] = [constraint, *derived]
            else:
                ranking, reason = _canonicalize_fdr_prediction(
                    predicted=predicted,
                    gt_ranking=list(question["gt_ranking"]),
                    anchor=question["anchor"],
                )
                if reason is not None:
                    audit["skip_reason"] = mark_skip(reason)
                else:
                    constraint = {
                        "qid": qid,
                        "source_type": "fdr",
                        "anchor": question["anchor"],
                        "ranking": ranking,
                        "weight": 1.0,
                    }
                    derived = _decompose_fdr_record(constraint)
                    fdr_constraints.append(constraint)
                    qrr_from_fdr.extend(derived)
                    audit["selected"] = True
                    audit["prepared_constraints"] = [constraint, *derived]
        else:
            audit["skip_reason"] = mark_skip("unsupported_question_type")

        audit_rows.append(audit)

    qrr_all = qrr_constraints + qrr_from_fdr
    integrity = {
        "n_questions": len(questions),
        "n_scored_rows": len(per_question),
        "duplicate_question_ids": duplicate_question_ids,
        "duplicate_score_qids": duplicate_score_ids,
        "missing_score_qids": missing_score_qids,
        "extra_score_qids": extra_score_qids,
        "skip_reason_counts": skip_reason_counts,
        "question_layout": metadata.get("question_layout"),
        "question_layout_warning": metadata.get("question_layout_warning"),
        "alternate_flat_path": metadata.get("alternate_flat_path"),
        "alternate_flat_question_count": metadata.get("alternate_flat_question_count"),
    }
    summary = {
        "n_objects": len(object_ids),
        "n_questions": len(questions),
        "n_qrr_direct": len(qrr_constraints),
        "n_qrr_direct_disjoint": sum(1 for c in qrr_constraints if c.get("variant", "disjoint") != "shared_anchor"),
        "n_qrr_direct_shared_anchor": sum(1 for c in qrr_constraints if c.get("variant") == "shared_anchor"),
        "n_trr": len(trr_constraints),
        "n_fdr": len(fdr_constraints),
        "n_qrr_from_fdr": len(qrr_from_fdr),
        "n_qrr_total": len(qrr_all),
        "n_selected_questions": sum(1 for row in audit_rows if row["selected"]),
        "n_skipped_questions": sum(1 for row in audit_rows if not row["selected"]),
    }

    return PreparedSceneInput(
        scene_id=scene_id or str(metadata.get("scene_id") or "unknown_scene"),
        model=model,
        use_correct_only=use_correct_only,
        metadata=metadata,
        object_ids=object_ids,
        gt_positions=dict(gt_positions or {}),
        per_question_audit=audit_rows,
        qrr_constraints=qrr_constraints,
        trr_constraints=trr_constraints,
        fdr_constraints=fdr_constraints,
        qrr_from_fdr=qrr_from_fdr,
        qrr_all=qrr_all,
        summary=summary,
        integrity=integrity,
    )
