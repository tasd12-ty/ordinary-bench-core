"""VRF 评测引擎：场景发现 → 问题加载 → API 调用 → 评分。"""

from __future__ import annotations

import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_VLM_API_DIR = Path(__file__).resolve().parent.parent / "VLM-test" / "API-test"
if str(_VLM_API_DIR) not in sys.path:
    sys.path.insert(0, str(_VLM_API_DIR))

from providers import create_provider_adapter
from response_parser import parse_batch_response

from vrf_job_spec import JobSpec
from vrf_prompts import VRF_SYSTEM_PROMPT, format_vrf_user_prompt
from vrf_question_loader import discover_scene_ids, load_scene_questions
from vrf_scoring import score_batch_scene, aggregate_results

logger = logging.getLogger(__name__)


def _resolve_images(scene_id: str, images_dir: str, mode: str) -> list[dict]:
    """Simple image resolver for VRF (single view only for now)."""
    if mode == "none" or not images_dir:
        return []
    path = Path(images_dir) / f"{scene_id}.png"
    if path.exists():
        return [{"kind": "file", "value": str(path)}]
    return []


def process_scene(scene_id: str, job: JobSpec, results_dir: Path) -> dict:
    adapter = create_provider_adapter(job.provider)
    questions_dir = Path(job.questions_dir)
    scene_meta, questions = load_scene_questions(scene_id, questions_dir)
    image_inputs = _resolve_images(scene_id, job.images_dir, job.image_mode)

    expected_qids = [q["qid"] for q in questions]
    user_prompt = format_vrf_user_prompt(scene_meta["objects"], questions)

    request = adapter.prepare_request(
        system_prompt=VRF_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        image_inputs=image_inputs,
    )

    logger.info("  %s (%d questions)", scene_id, len(questions))
    start = time.time()
    raw_response = adapter.call(request)
    elapsed = time.time() - start

    predictions = parse_batch_response(raw_response, expected_qids)
    missing = sum(1 for v in predictions.values() if v is None)

    # Simple react retry
    if (
        missing > len(expected_qids) * job.missing_threshold
        and job.react_max_rounds > 0
    ):
        missing_qids = [qid for qid, v in predictions.items() if v is None]
        correction_text = (
            f"Your previous response could not be fully parsed. "
            f"{len(missing_qids)} out of {len(expected_qids)} answers are missing.\n"
            f"Missing: {', '.join(missing_qids)}\n"
            f"Please output ONLY a valid JSON array for the missing questions."
        )
        correction_request = adapter.append_correction(
            request, assistant_text=raw_response, correction_text=correction_text,
        )
        retry_start = time.time()
        retry_response = adapter.call(correction_request)
        elapsed += time.time() - retry_start
        for qid, val in parse_batch_response(retry_response, missing_qids).items():
            if val is not None:
                predictions[qid] = val
        raw_response += f"\n\n--- ReAct retry ---\n{retry_response}"

    # Save raw
    raw_dir = results_dir / "raw" / scene_id
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_record = {
        "scene_id": scene_id,
        "model": job.provider.model,
        "elapsed_seconds": round(elapsed, 2),
        "raw_response": raw_response,
    }
    if job.save_prompt:
        raw_record["prompt"] = adapter.serialize_request(request)
    with open(raw_dir / "vrf_0.json", "w") as f:
        json.dump(raw_record, f, indent=2)

    scores = score_batch_scene(predictions, questions)
    return {
        "scene_id": scene_id,
        "model": job.provider.model,
        "n_objects": scene_meta["n_objects"],
        "total_questions": len(questions),
        "scores": scores,
    }


def run_job(job: JobSpec) -> tuple[list[dict], dict]:
    questions_dir = Path(job.questions_dir)
    scene_ids = discover_scene_ids(questions_dir, job.split, job.max_scenes)
    if not scene_ids:
        raise SystemExit("No VRF question files found")

    logger.info(
        "VRF job: %s scenes, adapter=%s, model=%s",
        len(scene_ids), job.provider.adapter, job.provider.model,
    )

    results_dir = Path(job.results_dir) / job.run_name
    results_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    failed: list[str] = []

    concurrency = int(job.provider.options.get("max_concurrency", 4))
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        futures = {
            pool.submit(process_scene, sid, job, results_dir): sid
            for sid in scene_ids
        }
        for future in as_completed(futures):
            sid = futures[future]
            try:
                result = future.result()
                results.append(result)
                s = result["scores"]
                logger.info(
                    "  %s: VRF %d/%d (true=%d/%d, false=%d/%d), missing %d",
                    sid,
                    s["vrf_correct"], s["vrf_total"],
                    s["vrf_true_correct"], s["vrf_true_total"],
                    s["vrf_false_correct"], s["vrf_false_total"],
                    s["missing"],
                )
            except Exception as exc:
                logger.error("  %s failed: %s", sid, exc)
                failed.append(sid)

    summary = aggregate_results(results)
    summary.update(job.to_metadata())
    summary["run_name"] = job.run_name
    summary["n_scenes"] = len(results)
    summary["n_failed"] = len(failed)
    if failed:
        summary["failed_scenes"] = failed

    with open(results_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info("Summary saved to %s/summary.json", results_dir)

    return results, summary
