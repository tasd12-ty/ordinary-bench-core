"""
Batch reconstruction of evaluated scenes.

Runs the reconstruction pipeline on all scene results for a given model,
producing per-scene reconstruction metrics and aggregate statistics.
"""

import json
import sys
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from reconstruct import (
    PreparedSceneInput,
    reconstruct_from_prepared,
    prepare_reconstruction_input_from_scoring,
    load_questions_auto,
    load_scene_gt_positions,
)
from analysis.aggregate import load_scene_results


def load_scene_gt(scene_path: str) -> Optional[Dict[str, np.ndarray]]:
    """Load ground truth 2D positions from scene JSON."""
    loaded = load_scene_gt_positions(scene_path)
    if not loaded:
        return None
    return {oid: np.array(pos, dtype=np.float64) for oid, pos in loaded.items()}


def prepare_single_scene(
    scene_result: dict,
    questions: List[dict],
    gt_positions: Optional[Dict[str, np.ndarray]] = None,
    use_correct_only: bool = True,
    question_metadata: Optional[dict] = None,
) -> PreparedSceneInput:
    """Prepare a single scene for reconstruction without solving it."""
    gt_serialized = None
    if gt_positions is not None:
        gt_serialized = {
            oid: np.asarray(pos, dtype=np.float64).tolist()
            for oid, pos in gt_positions.items()
        }

    question_metadata = dict(question_metadata or {})
    metadata = {
        "scene_id": scene_result.get("scene_id"),
        "model": scene_result.get("model"),
        "n_objects": scene_result.get("n_objects"),
        "question_layout": question_metadata.get("layout", scene_result.get("question_layout", "auto")),
        "question_paths": question_metadata.get("paths", {}),
        "question_scene_meta": question_metadata.get("scene_meta", {}),
        "question_layout_warning": question_metadata.get("layout_warning"),
        "alternate_flat_path": question_metadata.get("alternate_flat_path"),
        "alternate_flat_question_count": question_metadata.get("alternate_flat_question_count"),
    }

    return prepare_reconstruction_input_from_scoring(
        scoring_result=scene_result["scores"],
        questions=questions,
        gt_positions=gt_serialized,
        scene_id=scene_result.get("scene_id"),
        model=scene_result.get("model"),
        use_correct_only=use_correct_only,
        metadata=metadata,
    )


def reconstruct_single_scene(
    scene_result: dict,
    questions: List[dict],
    gt_positions: Optional[Dict[str, np.ndarray]] = None,
    use_correct_only: bool = True,
    n_restarts: int = 10,
    question_metadata: Optional[dict] = None,
) -> dict:
    """Reconstruct a single scene and return metrics.

    Args:
        scene_result: from load_scene_results()
        questions: flattened question list for this scene
        gt_positions: ground truth positions
        use_correct_only: if True, use only correct answers
        n_restarts: number of optimization restarts

    Returns:
        dict with scene_id, status, metrics, positions
    """
    prepared = prepare_single_scene(
        scene_result=scene_result,
        questions=questions,
        gt_positions=gt_positions,
        use_correct_only=use_correct_only,
        question_metadata=question_metadata,
    )

    result = reconstruct_from_prepared(
        prepared_input=prepared,
        n_restarts=n_restarts,
    )

    output = result.to_dict()
    output["scene_id"] = scene_result["scene_id"]
    output["n_objects"] = scene_result.get("n_objects", 0)
    output["model"] = scene_result.get("model", "unknown")
    output["use_correct_only"] = use_correct_only
    output["prepared_summary"] = prepared.summary
    output["prepared_integrity"] = prepared.integrity

    return output


def prepare_all_scenes(
    results_dir: str,
    questions_dir: str,
    scenes_dir: str,
    output_dir: Optional[str] = None,
    use_correct_only: bool = True,
    max_scenes: Optional[int] = None,
) -> List[dict]:
    """Prepare reconstruction inputs for all evaluated scenes."""
    scene_results = load_scene_results(results_dir)
    if max_scenes:
        scene_results = scene_results[:max_scenes]

    prepared_outputs = []
    scene_output_dir = None
    if output_dir:
        scene_output_dir = Path(output_dir) / "scenes"
        scene_output_dir.mkdir(parents=True, exist_ok=True)

    for i, scene_result in enumerate(scene_results):
        scene_id = scene_result["scene_id"]
        questions, question_meta = load_questions_auto(questions_dir, scene_id)
        if not questions:
            print(f"  [{i+1}/{len(scene_results)}] {scene_id}: no questions found, skipping")
            continue

        scene_path = os.path.join(scenes_dir, f"{scene_id}.json")
        gt_positions = load_scene_gt(scene_path) if os.path.exists(scene_path) else None

        try:
            prepared = prepare_single_scene(
                scene_result=scene_result,
                questions=questions,
                gt_positions=gt_positions,
                use_correct_only=use_correct_only,
                question_metadata=question_meta,
            )
            prepared_dict = prepared.to_dict()
            prepared_outputs.append(prepared_dict)
            print(
                f"  [{i+1}/{len(scene_results)}] {scene_id}: "
                f"qrr={prepared.summary['n_qrr_total']} "
                f"(direct {prepared.summary['n_qrr_direct']}, fdr {prepared.summary['n_qrr_from_fdr']}) "
                f"trr={prepared.summary['n_trr']} skipped={prepared.summary['n_skipped_questions']}"
            )
            if scene_output_dir is not None:
                with open(scene_output_dir / f"{scene_id}.json", "w") as f:
                    json.dump(prepared_dict, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"  [{i+1}/{len(scene_results)}] {scene_id}: ERROR {e}")
            continue

    if output_dir:
        summary = {
            "n_scenes": len(prepared_outputs),
            "use_correct_only": use_correct_only,
            "status": "prepared",
            "scenes": [
                {
                    "scene_id": row["scene_id"],
                    "n_objects": row["summary"]["n_objects"],
                    "n_qrr_total": row["summary"]["n_qrr_total"],
                    "n_trr": row["summary"]["n_trr"],
                    "n_skipped_questions": row["summary"]["n_skipped_questions"],
                    "integrity": row["integrity"],
                }
                for row in prepared_outputs
            ],
        }
        with open(Path(output_dir) / "summary.json", "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\nSaved {len(prepared_outputs)} prepared scenes to {output_dir}")

    return prepared_outputs


def reconstruct_all_scenes(
    results_dir: str,
    questions_dir: str,
    scenes_dir: str,
    output_path: Optional[str] = None,
    use_correct_only: bool = True,
    n_restarts: int = 10,
    max_scenes: Optional[int] = None,
) -> List[dict]:
    """Reconstruct all evaluated scenes for a model.

    Args:
        results_dir: path to model results directory
        questions_dir: path to questions directory
        scenes_dir: path to scene data directory (for GT)
        output_path: optional path to save results JSON
        use_correct_only: if True, use only correct answers
        n_restarts: number of optimization restarts
        max_scenes: optional limit on number of scenes

    Returns:
        List of per-scene reconstruction results
    """
    scene_results = load_scene_results(results_dir)
    if max_scenes:
        scene_results = scene_results[:max_scenes]

    all_outputs = []

    for i, scene_result in enumerate(scene_results):
        scene_id = scene_result["scene_id"]

        # Load questions
        questions, question_meta = load_questions_auto(questions_dir, scene_id)
        if not questions:
            print(f"  [{i+1}/{len(scene_results)}] {scene_id}: no questions found, skipping")
            continue

        # Load GT
        scene_path = os.path.join(scenes_dir, f"{scene_id}.json")
        gt_positions = None
        if os.path.exists(scene_path):
            gt_positions = load_scene_gt(scene_path)

        # Reconstruct
        try:
            output = reconstruct_single_scene(
                scene_result, questions, gt_positions,
                use_correct_only=use_correct_only,
                n_restarts=n_restarts,
                question_metadata=question_meta,
            )
            status = output["status"]
            csr_q = output["metrics"]["csr_qrr"]
            csr_t = output["metrics"]["csr_trr"]
            nrms = output["metrics"].get("nrms", "N/A")
            print(f"  [{i+1}/{len(scene_results)}] {scene_id}: "
                  f"status={status} csr_qrr={csr_q:.3f} csr_trr={csr_t:.3f} "
                  f"nrms={nrms if isinstance(nrms, str) else f'{nrms:.4f}'}")
            all_outputs.append(output)
        except Exception as e:
            print(f"  [{i+1}/{len(scene_results)}] {scene_id}: ERROR {e}")
            continue

    # Save if output_path specified
    if output_path and all_outputs:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(all_outputs, f, indent=2, default=str)
        print(f"\nSaved {len(all_outputs)} reconstruction results to {output_path}")

    return all_outputs


def summarize_reconstructions(results: List[dict]) -> dict:
    """Aggregate reconstruction metrics across scenes."""
    if not results:
        return {}

    metrics_keys = ["csr_qrr", "csr_trr", "spread", "best_loss"]
    gt_keys = ["kendall_tau", "nrms"]

    summary = {
        "n_scenes": len(results),
        "status_counts": {},
        "feasible_rate": 0.0,
    }

    # Status counts
    for r in results:
        status = r["status"]
        summary["status_counts"][status] = summary["status_counts"].get(status, 0) + 1

    summary["feasible_rate"] = sum(
        1 for r in results if r["feasible"]
    ) / len(results)

    # Aggregate metrics
    for key in metrics_keys:
        values = [r["metrics"][key] for r in results
                  if r["metrics"].get(key) is not None]
        if values:
            summary[f"{key}_mean"] = float(np.mean(values))
            summary[f"{key}_std"] = float(np.std(values))
            summary[f"{key}_median"] = float(np.median(values))

    for key in gt_keys:
        values = [r["metrics"][key] for r in results
                  if r["metrics"].get(key) is not None]
        if values:
            summary[f"{key}_mean"] = float(np.mean(values))
            summary[f"{key}_std"] = float(np.std(values))
            summary[f"{key}_n"] = len(values)

    # By split
    by_split = {}
    for r in results:
        split = r["scene_id"].rsplit("_", 1)[0]
        by_split.setdefault(split, []).append(r)

    summary["by_split"] = {}
    for split, split_results in sorted(by_split.items()):
        s = {"n_scenes": len(split_results)}
        for key in metrics_keys + gt_keys:
            values = [r["metrics"][key] for r in split_results
                      if r["metrics"].get(key) is not None]
            if values:
                s[f"{key}_mean"] = float(np.mean(values))
        summary["by_split"][split] = s

    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Batch scene reconstruction")
    parser.add_argument("--results-dir", "-r", required=True,
                        help="Path to model results directory")
    parser.add_argument("--questions-dir", "-q",
                        default="VLM-test/output/questions",
                        help="Path to questions directory")
    parser.add_argument("--scenes-dir", "-s",
                        default="data-gen/output/scenes",
                        help="Path to scene data directory")
    parser.add_argument("--output", "-o", default=None,
                        help="Output path for results JSON")
    parser.add_argument("--belief", action="store_true",
                        help="Use VLM predictions (not only correct) for belief reconstruction")
    parser.add_argument("--restarts", type=int, default=10)
    parser.add_argument("--max-scenes", type=int, default=None)
    args = parser.parse_args()

    results = reconstruct_all_scenes(
        results_dir=args.results_dir,
        questions_dir=args.questions_dir,
        scenes_dir=args.scenes_dir,
        output_path=args.output,
        use_correct_only=not args.belief,
        n_restarts=args.restarts,
        max_scenes=args.max_scenes,
    )

    if results:
        summary = summarize_reconstructions(results)
        print(f"\n=== Summary ===")
        print(f"Scenes: {summary['n_scenes']}")
        print(f"Feasible rate: {summary['feasible_rate']:.1%}")
        print(f"Status: {summary['status_counts']}")
        for key in ["csr_qrr", "csr_trr", "kendall_tau", "nrms"]:
            if f"{key}_mean" in summary:
                print(f"{key}: {summary[f'{key}_mean']:.4f} ± {summary.get(f'{key}_std', 0):.4f}")
