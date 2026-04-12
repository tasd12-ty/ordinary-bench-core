"""Coordinate prediction eval engine: scene discovery -> prompt -> API -> evaluate."""

from __future__ import annotations

import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# Local imports FIRST (before adding external paths that may shadow module names)
from job_spec import JobSpec
from prompts import COORD_SYSTEM_PROMPT, format_coord_user_prompt
from coord_parser import parse_coordinate_response
from coord_scoring import evaluate_predictions, aggregate_results

# Now add external paths for VLM-test modules
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_VLM_DIR = _PROJECT_ROOT / "VLM-test"
_API_DIR = _VLM_DIR / "API-test"
for _p in (str(_VLM_DIR), str(_API_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from providers import create_provider_adapter
from extraction import parse_objects, object_description, load_scene
from dsl.predicates import (
    MetricType,
    extract_all_qrr,
    extract_all_qrr_shared_anchor,
    extract_all_trr,
)
from reconstruct.preparation import load_scene_gt_positions
from reconstruct.constraints import QRREntry, TRREntry

logger = logging.getLogger(__name__)


def _resolve_images(scene_id: str, images_dir: str, mode: str, n_views: int = 4) -> List[dict]:
    """Resolve image paths based on image_mode."""
    base = Path(images_dir)
    if mode == "single":
        p = base / "single_view" / f"{scene_id}.png"
        return [{"kind": "file", "value": str(p)}] if p.exists() else []
    elif mode == "multi_view":
        paths = []
        for i in range(n_views):
            p = base / "multi_view" / scene_id / f"view_{i}.png"
            if p.exists():
                paths.append({"kind": "file", "value": str(p)})
        return paths
    elif mode == "top_view":
        p = base / "top_view" / f"{scene_id}.png"
        return [{"kind": "file", "value": str(p)}] if p.exists() else []
    return []


def _extract_gt_constraints(
    objects: Dict, tau: float,
) -> tuple[List[QRREntry], List[TRREntry]]:
    """Extract GT QRR and TRR constraints, convert to Entry types for CSR."""
    # QRR constraints — use DIST_2D to match the 2D evaluation plane
    qrr_dsl = extract_all_qrr(objects, MetricType.DIST_2D, tau=tau, disjoint_only=True)
    qrr_dsl += extract_all_qrr_shared_anchor(objects, MetricType.DIST_2D, tau=tau)
    qrr_entries = [
        QRREntry(
            pair1=tuple(c.pair1),
            pair2=tuple(c.pair2),
            comparator=str(c.comparator),
            variant=c.variant,
            anchor=c.anchor,
        )
        for c in qrr_dsl
    ]

    # TRR constraints
    trr_dsl = extract_all_trr(objects, use_3d=True)
    trr_entries = [
        TRREntry(
            target=c.target,
            ref1=c.ref1,
            ref2=c.ref2,
            hour=c.hour,
            level="hour",
        )
        for c in trr_dsl
    ]
    return qrr_entries, trr_entries


def discover_scene_ids(
    scenes_dir: Path, split: Optional[str], max_scenes: Optional[int],
) -> List[str]:
    """Discover scene IDs from scenes directory."""
    files = sorted(scenes_dir.glob("*.json"))
    if split:
        files = [f for f in files if f.stem.startswith(split)]
    ids = [f.stem for f in files]
    if max_scenes and max_scenes > 0:
        ids = ids[:max_scenes]
    return ids


def process_scene(scene_id: str, job: JobSpec, results_dir: Path) -> dict:
    """Process a single scene: load -> prompt -> call API -> parse -> evaluate."""
    scenes_dir = Path(job.scenes_dir)
    scene_path = scenes_dir / f"{scene_id}.json"

    # Load scene
    scene = load_scene(str(scene_path))
    objects = parse_objects(scene)
    obj_list = [
        {"id": oid, "desc": object_description(objects[oid])}
        for oid in sorted(objects.keys())
    ]
    expected_ids = sorted(objects.keys())

    # GT positions
    gt_raw = load_scene_gt_positions(str(scene_path))
    gt_positions = {oid: np.array(pos) for oid, pos in gt_raw.items()} if gt_raw else {}

    # GT constraints for CSR
    qrr_entries, trr_entries = _extract_gt_constraints(objects, job.tau)

    # Images
    image_inputs = _resolve_images(scene_id, job.images_dir, job.image_mode, job.n_views)

    # Build prompt
    user_prompt = format_coord_user_prompt(obj_list, job.image_mode, job.n_views)

    # API call
    adapter = create_provider_adapter(job.provider)
    request = adapter.prepare_request(
        system_prompt=COORD_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        image_inputs=image_inputs,
    )

    logger.info("  %s (%d objects, mode=%s, %d images)",
                scene_id, len(objects), job.image_mode, len(image_inputs))
    start = time.time()
    raw_response = adapter.call(request)
    elapsed = time.time() - start

    # Parse
    predicted_raw = parse_coordinate_response(raw_response, expected_ids)
    n_missing = sum(1 for v in predicted_raw.values() if v is None)

    # ReAct retry if >50% missing
    if n_missing > len(expected_ids) * 0.5:
        missing_ids = [oid for oid, v in predicted_raw.items() if v is None]
        correction = (
            f"Your response could not be fully parsed. "
            f"{n_missing}/{len(expected_ids)} objects are missing coordinates.\n"
            f"Missing: {', '.join(missing_ids)}\n"
            f"Please output a JSON object with [x, y] for these missing objects."
        )
        retry_req = adapter.append_correction(
            request, assistant_text=raw_response, correction_text=correction,
        )
        retry_start = time.time()
        retry_response = adapter.call(retry_req)
        elapsed += time.time() - retry_start

        retry_parsed = parse_coordinate_response(retry_response, missing_ids)
        for oid, val in retry_parsed.items():
            if val is not None:
                predicted_raw[oid] = val
        raw_response += f"\n\n--- ReAct retry ---\n{retry_response}"

    # Convert to numpy
    predicted = {
        oid: np.array(coords)
        for oid, coords in predicted_raw.items()
        if coords is not None
    }

    # Evaluate
    metrics = evaluate_predictions(predicted, gt_positions, qrr_entries, trr_entries, job.tau)

    # Save raw
    raw_dir = results_dir / "raw" / scene_id
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_record = {
        "scene_id": scene_id,
        "model": job.provider.model,
        "image_mode": job.image_mode,
        "elapsed_seconds": round(elapsed, 2),
        "raw_response": raw_response,
        "predicted_positions": {
            oid: coords.tolist() for oid, coords in predicted.items()
        },
        "gt_positions": {oid: pos.tolist() for oid, pos in gt_positions.items()},
        "metrics": metrics,
    }
    if job.save_prompt:
        raw_record["prompt"] = adapter.serialize_request(request)
    with open(raw_dir / "coord_0.json", "w") as f:
        json.dump(raw_record, f, indent=2, ensure_ascii=False)

    return {
        "scene_id": scene_id,
        "model": job.provider.model,
        "n_objects": len(objects),
        "metrics": metrics,
    }


def run_job(job: JobSpec) -> tuple[list[dict], dict]:
    """Discover scenes, run in parallel, aggregate."""
    scenes_dir = Path(job.scenes_dir)
    scene_ids = discover_scene_ids(scenes_dir, job.split, job.max_scenes)
    if not scene_ids:
        raise SystemExit("No scene files found")

    logger.info(
        "Coord prediction: %d scenes, adapter=%s, model=%s, mode=%s",
        len(scene_ids), job.provider.adapter, job.provider.model, job.image_mode,
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
                m = result["metrics"]
                logger.info(
                    "  %s: tau=%.3f, nrms=%.3f, csr_qrr=%s, csr_trr=%s, missing=%d",
                    sid,
                    m["kendall_tau"], m["nrms"],
                    f'{m["csr_qrr"]:.3f}' if m["csr_qrr"] is not None else "N/A",
                    f'{m["csr_trr"]:.3f}' if m["csr_trr"] is not None else "N/A",
                    m["n_missing"],
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
