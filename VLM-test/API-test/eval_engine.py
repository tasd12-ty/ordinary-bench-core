"""Unified evaluation engine."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from pathlib import Path
import time

from job_spec import JobSpec
from image_resolver import resolve_scene_images
from prompts import (
    BATCH_SYSTEM_PROMPT,
    MULTI_VIEW_SYSTEM_PROMPT,
    REACT_CORRECTION_PROMPT,
    TYPE_SYSTEM_PROMPTS,
    format_batch_user_prompt,
)
from providers import create_provider_adapter
from question_loader import discover_scene_ids, load_scene_questions
from response_parser import parse_batch_response
from result_store import ResultStore
from scoring import aggregate_batch_results, score_batch_scene


logger = logging.getLogger(__name__)

NO_IMAGE_SYSTEM_PROMPT = """\
You are a spatial reasoning assistant. You will receive a description of objects \
in a 3D scene (NO image is provided) and a set of spatial questions.

Based on the object descriptions ALONE, answer the spatial questions to the best \
of your ability. If you cannot determine the answer, make your best guess.

Question types:
1. QRR (distance comparison): Compare 3D distances, either between two pairs of objects
   or from a common anchor object to two candidate objects.
   Answer with exactly one of: "<" (first pair closer), "~=" (approximately equal), ">" (first pair farther).
2. TRR (clock direction): Imagine standing at ref1, facing toward ref2 (12 o'clock direction).
   Answer with the clock hour (integer 1-12) where the target object appears.
3. FDR (full distance ranking): Given an anchor object, rank all other objects by their 3D distance
   from the anchor, from nearest to farthest. Answer with a JSON list of object ID strings.

Respond ONLY with a JSON array. Each element must have "qid" and "answer".
"""


def _batch_label(group: dict, batch_id: int) -> str:
    if group["question_type"]:
        return f"{group['question_type']}_{batch_id}"
    return f"batch_{batch_id}"


def _select_system_prompt(job: JobSpec, group: dict) -> str:
    if job.input.question_grouping == "by_type":
        return TYPE_SYSTEM_PROMPTS[group["question_type"]]
    if job.images.mode == "multi_view":
        return MULTI_VIEW_SYSTEM_PROMPT.format(n_views=job.images.n_views)
    if job.images.mode == "none":
        return NO_IMAGE_SYSTEM_PROMPT
    return BATCH_SYSTEM_PROMPT


def _call_with_react(
    *,
    adapter,
    initial_request,
    expected_qids: list[str],
    job: JobSpec,
) -> tuple[dict, str, object, int, float]:
    request = initial_request
    start = time.time()
    raw_response = adapter.call(request)
    elapsed = time.time() - start

    predictions = parse_batch_response(raw_response, expected_qids)
    missing = sum(1 for value in predictions.values() if value is None)

    react_rounds = 0
    while (
        react_rounds < job.prompt.react_max_rounds
        and expected_qids
        and missing > len(expected_qids) * job.prompt.missing_threshold
    ):
        react_rounds += 1
        missing_qids = [qid for qid, value in predictions.items() if value is None]
        logger.info(
            "    ReAct #%s, missing %s/%s",
            react_rounds,
            missing,
            len(expected_qids),
        )

        chunk_size = max(1, job.prompt.react_chunk_size)
        for start_idx in range(0, len(missing_qids), chunk_size):
            chunk = missing_qids[start_idx : start_idx + chunk_size]
            correction_text = REACT_CORRECTION_PROMPT.format(
                missing_qids=", ".join(chunk),
                n_missing=len(chunk),
                n_total=len(expected_qids),
            )
            correction_request = adapter.append_correction(
                request,
                assistant_text=raw_response,
                correction_text=correction_text,
            )
            chunk_start = time.time()
            correction_response = adapter.call(correction_request)
            elapsed += time.time() - chunk_start
            correction_predictions = parse_batch_response(correction_response, chunk)
            for qid, value in correction_predictions.items():
                if value is not None:
                    predictions[qid] = value
            raw_response = (
                (raw_response or "")
                + f"\n\n--- ReAct #{react_rounds} ---\n"
                + (correction_response or "")
            )

        missing = sum(1 for value in predictions.values() if value is None)

    return predictions, raw_response, request, react_rounds, elapsed


def process_scene(scene_id: str, job: JobSpec, store: ResultStore) -> dict:
    adapter = create_provider_adapter(job.provider)
    scene_meta, groups = load_scene_questions(scene_id, job)
    image_inputs = resolve_scene_images(scene_id, job.images)

    all_predictions = {}
    all_questions = []
    total_batches = 0

    for group in groups:
        if not group["questions"]:
            continue
        system_prompt = _select_system_prompt(job, group)
        for batch in group["batches"]:
            batch_id = batch["batch_id"]
            batch_label = _batch_label(group, batch_id)
            questions = batch["questions"]
            expected_qids = [question["qid"] for question in questions]
            user_prompt = format_batch_user_prompt(scene_meta["objects"], questions)
            request = adapter.prepare_request(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                image_inputs=image_inputs,
            )

            logger.info(
                "  %s %s (%s questions)",
                scene_id,
                batch_label,
                len(questions),
            )
            predictions, raw_response, final_request, react_rounds, elapsed = _call_with_react(
                adapter=adapter,
                initial_request=request,
                expected_qids=expected_qids,
                job=job,
            )

            raw_record = {
                "scene_id": scene_id,
                "batch_id": batch_label,
                "question_type": group["question_type"],
                "model": job.provider.model,
                "timestamp": time.time(),
                "elapsed_seconds": round(elapsed, 2),
                "react_rounds": react_rounds,
                "raw_response": raw_response,
            }
            if job.prompt.save_prompt:
                raw_record["prompt"] = adapter.serialize_request(final_request)
            store.save_raw(scene_id=scene_id, batch_label=batch_label, record=raw_record)

            all_predictions.update(predictions)
            all_questions.extend(questions)
            total_batches += 1

    scores = score_batch_scene(all_predictions, all_questions, ablation=job.prompt.ablation)
    return {
        "scene_id": scene_id,
        "model": job.provider.model,
        "n_objects": scene_meta["n_objects"],
        "n_batches": total_batches,
        "total_questions": len(all_questions),
        "scores": scores,
    }


def run_job(job: JobSpec) -> tuple[list[dict], dict]:
    scene_ids = discover_scene_ids(job)
    if not scene_ids:
        raise SystemExit("No scene files found for job")

    logger.info(
        "Running job %s: %s scenes, adapter=%s, model=%s",
        job.job_name or job.run_name,
        len(scene_ids),
        job.provider.adapter,
        job.provider.model,
    )

    store = ResultStore(job)
    results: list[dict] = []
    failed_scenes: list[str] = []

    concurrency = int(job.provider.options.get("max_concurrency", 4))
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        futures = {
            pool.submit(process_scene, scene_id, job, store): scene_id
            for scene_id in scene_ids
        }
        for future in as_completed(futures):
            scene_id = futures[future]
            try:
                result = future.result()
                results.append(result)
                store.save_scene_result(result)
                scores = result["scores"]
                if job.prompt.ablation:
                    logger.info(
                        "  %s: answerable %s/%s, refusal %s/%s, halluc %s, missing %s",
                        scene_id,
                        scores.get("answerable_correct", 0),
                        scores.get("answerable_total", 0),
                        scores.get("refusal_correct", 0),
                        scores.get("refusal_total", 0),
                        scores.get("refusal_hallucinated", 0),
                        scores["missing"],
                    )
                else:
                    logger.info(
                        "  %s: QRR %s/%s, TRR hour %s/%s, FDR exact %s/%s, missing %s",
                        scene_id,
                        scores["qrr_correct"],
                        scores["qrr_total"],
                        scores["trr_hour_correct"],
                        scores["trr_total"],
                        scores["fdr_exact_correct"],
                        scores["fdr_total"],
                        scores["missing"],
                    )
            except Exception as exc:
                logger.error("  %s failed: %s", scene_id, exc)
                failed_scenes.append(scene_id)

    summary = aggregate_batch_results(results)
    summary.update(job.to_metadata())
    summary["run_name"] = job.run_name
    summary["n_scenes"] = len(results)
    summary["n_failed"] = len(failed_scenes)
    if failed_scenes:
        summary["failed_scenes"] = failed_scenes
    store.save_summary(summary)

    return results, summary
