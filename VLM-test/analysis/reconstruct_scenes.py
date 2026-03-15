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

from reconstruct import reconstruct_from_scoring, SolverConfig, ReconstructResult
from analysis.aggregate import load_scene_results, load_questions


def load_scene_gt(scene_path: str) -> Optional[Dict[str, np.ndarray]]:
    """Load ground truth 2D positions from scene JSON."""
    with open(scene_path) as f:
        scene = json.load(f)

    positions = {}
    for obj in scene.get("objects", []):
        coords = obj.get("3d_coords", obj.get("position_3d"))
        if coords is not None:
            positions[obj["id"]] = np.array(coords[:2], dtype=np.float64)
    return positions if positions else None


def reconstruct_single_scene(
    scene_result: dict,
    questions: List[dict],
    gt_positions: Optional[Dict[str, np.ndarray]] = None,
    use_correct_only: bool = True,
    n_restarts: int = 10,
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
    scoring = scene_result["scores"]

    result = reconstruct_from_scoring(
        scoring_result=scoring,
        questions=questions,
        gt_positions=gt_positions,
        n_restarts=n_restarts,
        use_correct_only=use_correct_only,
    )

    output = result.to_dict()
    output["scene_id"] = scene_result["scene_id"]
    output["n_objects"] = scene_result.get("n_objects", 0)
    output["model"] = scene_result.get("model", "unknown")
    output["use_correct_only"] = use_correct_only

    return output


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
        questions = load_questions(questions_dir, scene_id)
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
