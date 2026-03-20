#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from common import (
    flatten_questions,
    load_question_documents,
    normalize_human_answer,
    slugify,
)

from scoring import aggregate_batch_results, score_batch_scene


def discover_response_files(responses_dir: str) -> List[Path]:
    return sorted(Path(responses_dir).rglob("*.json"))


def load_response_payload(path: Path) -> Optional[dict]:
    with open(path) as f:
        data = json.load(f)
    if "scene_id" not in data:
        return None
    if "responses" not in data and "batches" not in data:
        return None
    return data


def infer_annotator_id(path: Path, payload: dict) -> str:
    annotator = payload.get("annotator_id")
    if annotator:
        return slugify(str(annotator))
    if path.parent.name and path.parent.name != path.anchor:
        return slugify(path.parent.name)
    return "anonymous"


def build_question_lookup(scene_id: str, questions_dir: str) -> Tuple[List[dict], Dict[str, dict]]:
    question_docs = load_question_documents(scene_id, questions_dir)
    questions = flatten_questions(question_docs)
    question_lookup = {question["qid"]: question for question in questions}
    return questions, question_lookup


def parse_raw_response(raw_response: object) -> List[dict]:
    if isinstance(raw_response, list):
        return [row for row in raw_response if isinstance(row, dict)]
    if not isinstance(raw_response, str) or not raw_response.strip():
        return []
    try:
        parsed = json.loads(raw_response)
    except json.JSONDecodeError:
        return []
    return [row for row in parsed if isinstance(row, dict)] if isinstance(parsed, list) else []


def normalize_batch_responses(
    batch_rows: Iterable[dict],
    question_lookup: Dict[str, dict],
) -> List[dict]:
    normalized = []
    for row in batch_rows:
        qid = row.get("qid")
        if qid not in question_lookup:
            continue
        question = question_lookup[qid]
        answer = normalize_human_answer(row.get("answer"), question["type"])
        if answer is None:
            continue
        normalized.append({"qid": qid, "answer": answer})
    return normalized


def synthesize_batches(payload: dict, question_lookup: Dict[str, dict]) -> List[dict]:
    if payload.get("batches"):
        batches = []
        for batch in payload["batches"]:
            batch_copy = dict(batch)
            if not batch_copy.get("responses"):
                batch_copy["responses"] = parse_raw_response(batch_copy.get("raw_response"))
            batches.append(batch_copy)
        return batches

    rows = list(payload.get("responses", [])) or parse_raw_response(payload.get("raw_response"))
    responses = normalize_batch_responses(rows, question_lookup)
    return [{
        "batch_id": "all_0",
        "question_type": "mixed",
        "elapsed_seconds": payload.get("elapsed_seconds", 0.0),
        "responses": responses,
        "raw_response": json.dumps(responses, ensure_ascii=False),
    }]


def process_payload(
    payload: dict,
    response_path: Path,
    questions_dir: str,
    model_name: str,
) -> Tuple[dict, List[dict]]:
    scene_id = payload["scene_id"]
    questions, question_lookup = build_question_lookup(scene_id, questions_dir)
    predictions: Dict[str, object] = {}
    raw_records: List[dict] = []

    for batch in synthesize_batches(payload, question_lookup):
        normalized_rows = normalize_batch_responses(batch.get("responses", []), question_lookup)
        raw_response = batch.get("raw_response")
        if not raw_response:
            raw_response = json.dumps(normalized_rows, ensure_ascii=False)

        for row in normalized_rows:
            predictions[row["qid"]] = row["answer"]

        qtype = batch.get("question_type")
        if qtype == "mixed" and normalized_rows:
            first_qid = normalized_rows[0]["qid"]
            qtype = question_lookup[first_qid]["type"]

        raw_records.append({
            "scene_id": scene_id,
            "batch_id": batch.get("batch_id", "all_0"),
            "question_type": qtype or "mixed",
            "model": model_name,
            "timestamp": payload.get("submitted_at"),
            "elapsed_seconds": round(float(batch.get("elapsed_seconds", 0.0)), 2),
            "react_rounds": 0,
            "raw_response": raw_response,
            "source_response_file": str(response_path),
            "response_condition": payload.get("response_condition"),
            "selected_test_type": payload.get("selected_test_type"),
        })

    scores = score_batch_scene(predictions, questions)
    scene_result = {
        "scene_id": scene_id,
        "model": model_name,
        "n_objects": payload.get("n_objects", 0),
        "n_batches": len(raw_records),
        "total_questions": len(questions),
        "scores": scores,
        "source_response_file": str(response_path),
        "submitted_at": payload.get("submitted_at"),
        "response_condition": payload.get("response_condition"),
        "response_condition_detail": payload.get("response_condition_detail", {}),
        "selected_test_type": payload.get("selected_test_type"),
        "view_mode_history": payload.get("view_mode_history", []),
    }
    return scene_result, raw_records


def save_results_for_annotator(
    annotator_id: str,
    scene_payloads: List[Tuple[Path, dict]],
    questions_dir: str,
    output_dir: str,
) -> None:
    model_name = f"human/{annotator_id}"
    model_dir = Path(output_dir) / f"human--{slugify(annotator_id)}"
    raw_dir = model_dir / "raw"
    scenes_dir = model_dir / "scenes"
    raw_dir.mkdir(parents=True, exist_ok=True)
    scenes_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for index, (path, payload) in enumerate(sorted(scene_payloads, key=lambda item: item[1]["scene_id"]), start=1):
        scene_result, raw_records = process_payload(
            payload=payload,
            response_path=path,
            questions_dir=questions_dir,
            model_name=model_name,
        )

        for raw_record in raw_records:
            raw_path = raw_dir / f"{scene_result['scene_id']}_{raw_record['batch_id']}.json"
            with open(raw_path, "w") as f:
                json.dump(raw_record, f, indent=2, ensure_ascii=False)

        scene_path = scenes_dir / f"{scene_result['scene_id']}.json"
        with open(scene_path, "w") as f:
            json.dump(scene_result, f, indent=2, ensure_ascii=False)

        results.append(scene_result)
        scores = scene_result["scores"]
        print(
            f"[{index}/{len(scene_payloads)}] {annotator_id} {scene_result['scene_id']}: "
            f"QRR {scores['qrr_correct']}/{scores['qrr_total']}, "
            f"TRR {scores['trr_hour_correct']}/{scores['trr_total']}, "
            f"FDR {scores['fdr_exact_correct']}/{scores['fdr_total']}, "
            f"missing {scores['missing']}"
        )

    summary = aggregate_batch_results(results)
    summary["model"] = model_name
    summary["annotator_id"] = annotator_id
    summary["n_scenes"] = len(results)

    with open(model_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nSaved compatible results for {annotator_id} to {model_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze human baseline responses")
    parser.add_argument("--responses-dir", default="human-baseline/output/responses")
    parser.add_argument("--questions-dir", default="VLM-test/output/questions")
    parser.add_argument("--output-dir", default="human-baseline/output/results")
    args = parser.parse_args()

    response_files = discover_response_files(args.responses_dir)
    if not response_files:
        raise SystemExit(f"No response JSON files found under {args.responses_dir}")

    grouped: Dict[str, List[Tuple[Path, dict]]] = defaultdict(list)
    for path in response_files:
        payload = load_response_payload(path)
        if payload is None:
            continue
        annotator_id = infer_annotator_id(path, payload)
        grouped[annotator_id].append((path, payload))

    if not grouped:
        raise SystemExit("No valid response payloads found")

    for annotator_id, payloads in sorted(grouped.items()):
        save_results_for_annotator(
            annotator_id=annotator_id,
            scene_payloads=payloads,
            questions_dir=args.questions_dir,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
