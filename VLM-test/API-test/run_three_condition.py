"""
Three-condition experiment runner.

Condition A: Correct image + spatial questions (normal)
Condition B: Wrong image (random scene) + spatial questions
Condition C: No image + spatial questions (text-only)

Usage:
    python API-test/run_three_condition.py --condition [A|B|C] [--split n04] [--max-scenes 10]
"""

import json
import os
import sys
import random
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import *
from vlm_client import make_client, call_vlm, load_image_base64, build_messages
from prompts import BATCH_SYSTEM_PROMPT, format_batch_user_prompt, REACT_CORRECTION_PROMPT
from response_parser import parse_batch_response
from scoring import score_batch_scene


# No-image system prompt: same as batch but without image reference
NO_IMAGE_SYSTEM_PROMPT = """\
You are a spatial reasoning assistant. You will receive a description of objects \
in a 3D scene (NO image is provided) and a set of spatial questions.

Based on the object descriptions ALONE, answer the spatial questions to the best \
of your ability. If you cannot determine the answer, make your best guess.

Question types:
1. QRR (distance comparison): Compare the 3D distance between two pairs of objects.
   Answer with exactly one of: "<" (first pair closer), "~=" (approximately equal), ">" (first pair farther).
2. TRR (clock direction): Imagine standing at ref1, facing toward ref2 (12 o'clock direction).
   Answer with the clock hour (integer 1-12) where the target object appears.

Respond ONLY with a JSON array. Each element must have "qid" and "answer".
For QRR: answer is a string "<", "~=", or ">".
For TRR: answer is an integer 1-12.

Example:
[{"qid": "qrr_0001", "answer": "<"}, {"qid": "trr_0001", "answer": 7}]"""


def run_condition(
    condition: str,
    questions_dir: str = "output/questions",
    images_dir: str = "../data-gen/output/images/single_view",
    output_dir: str = None,
    split: str = None,
    max_scenes: int = None,
    wrong_image_seed: int = 42,
):
    """Run a single condition of the three-condition experiment.

    Args:
        condition: "A" (correct image), "B" (wrong image), "C" (no image)
        questions_dir: path to question files
        images_dir: path to scene images
        output_dir: output directory (auto-generated if None)
        split: optional split filter (e.g., "n04")
        max_scenes: optional limit
        wrong_image_seed: seed for selecting wrong images in condition B
    """
    condition = condition.upper()
    assert condition in ("A", "B", "C"), f"Invalid condition: {condition}"

    if output_dir is None:
        model_slug = VLM_MODEL.replace("/", "--").replace(":", "--")
        output_dir = f"output/results/{model_slug}_condition_{condition}"

    os.makedirs(os.path.join(output_dir, "raw"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "scenes"), exist_ok=True)

    # Discover scenes
    q_dir = Path(questions_dir)
    scene_files = sorted(q_dir.glob("*.json"))
    if split:
        scene_files = [f for f in scene_files if f.stem.startswith(split)]
    if max_scenes:
        scene_files = scene_files[:max_scenes]

    print(f"Condition {condition}: {len(scene_files)} scenes")
    print(f"Model: {VLM_MODEL}")
    print(f"Output: {output_dir}")

    # For condition B: collect all available image paths for random selection
    all_image_paths = []
    if condition == "B":
        img_dir = Path(images_dir)
        all_image_paths = sorted(img_dir.glob("*.png"))
        rng = random.Random(wrong_image_seed)

    client = make_client(VLM_BASE_URL, VLM_API_KEY)

    def process_scene(scene_file):
        with open(scene_file) as f:
            scene_data = json.load(f)

        scene_id = scene_data["scene_id"]
        objects = scene_data["objects"]

        # Check if already processed
        scene_out = os.path.join(output_dir, "scenes", f"{scene_id}.json")
        if os.path.exists(scene_out):
            return None

        all_predictions = {}
        all_questions = []

        for batch in scene_data["batches"]:
            batch_id = batch["batch_id"]
            questions = batch["questions"]
            all_questions.extend(questions)

            user_prompt = format_batch_user_prompt(objects, questions)

            # Build messages based on condition
            if condition == "A":
                # Normal: correct image
                img_path = os.path.join(images_dir, f"{scene_id}.png")
                if not os.path.exists(img_path):
                    print(f"  {scene_id}: image not found at {img_path}")
                    return None
                img_b64 = load_image_base64(img_path)
                messages = build_messages(BATCH_SYSTEM_PROMPT, user_prompt, img_b64)

            elif condition == "B":
                # Wrong image: random scene image
                candidates = [p for p in all_image_paths
                              if p.stem != scene_id]
                if not candidates:
                    return None
                wrong_img = rng.choice(candidates)
                img_b64 = load_image_base64(str(wrong_img))
                messages = build_messages(BATCH_SYSTEM_PROMPT, user_prompt, img_b64)

            elif condition == "C":
                # No image: text-only
                messages = [
                    {"role": "system", "content": NO_IMAGE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ]

            try:
                response = call_vlm(
                    client, messages, VLM_MODEL,
                    max_tokens=VLM_MAX_TOKENS,
                    temperature=0.0,
                )
                parsed = parse_batch_response(response)
                all_predictions.update(parsed)

                # Save raw response
                raw_out = os.path.join(output_dir, "raw",
                                       f"{scene_id}_batch_{batch_id}.json")
                with open(raw_out, "w") as f:
                    json.dump({
                        "scene_id": scene_id,
                        "batch_id": batch_id,
                        "condition": condition,
                        "model": VLM_MODEL,
                        "raw_response": response,
                    }, f, indent=2)

            except Exception as e:
                print(f"  {scene_id} batch {batch_id}: ERROR {e}")

        # Score
        scores = score_batch_scene(all_predictions, all_questions)

        result = {
            "scene_id": scene_id,
            "condition": condition,
            "model": VLM_MODEL,
            "n_objects": scene_data.get("n_objects", len(objects)),
            "n_batches": len(scene_data["batches"]),
            "total_questions": len(all_questions),
            "scores": scores,
        }

        with open(scene_out, "w") as f:
            json.dump(result, f, indent=2)

        s = scores
        print(f"  {scene_id}: QRR={s['qrr_correct']}/{s['qrr_total']} "
              f"TRR_h={s['trr_hour_correct']}/{s['trr_total']} "
              f"missing={s['missing']}")

        return result

    # Process scenes concurrently
    results = []
    with ThreadPoolExecutor(max_workers=VLM_CONCURRENCY) as executor:
        futures = {executor.submit(process_scene, f): f for f in scene_files}
        for future in as_completed(futures):
            result = future.result()
            if result:
                results.append(result)

    # Save summary
    if results:
        from scoring import aggregate_batch_results
        summary = aggregate_batch_results(results)
        summary["condition"] = condition
        summary["model"] = VLM_MODEL
        summary["n_scenes"] = len(results)

        with open(os.path.join(output_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\nCondition {condition} complete: {len(results)} scenes")
        print(f"  QRR accuracy: {summary['overall']['qrr_accuracy']:.2%}")
        print(f"  TRR hour accuracy: {summary['overall']['trr_hour_accuracy']:.2%}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Three-condition experiment")
    parser.add_argument("--condition", "-c", required=True,
                        choices=["A", "B", "C"],
                        help="Experiment condition")
    parser.add_argument("--questions-dir", "-q",
                        default="output/questions")
    parser.add_argument("--images-dir", "-i",
                        default="../data-gen/output/images/single_view")
    parser.add_argument("--output-dir", "-o", default=None)
    parser.add_argument("--split", "-s", default=None)
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--wrong-image-seed", type=int, default=42)
    args = parser.parse_args()

    run_condition(
        condition=args.condition,
        questions_dir=args.questions_dir,
        images_dir=args.images_dir,
        output_dir=args.output_dir,
        split=args.split,
        max_scenes=args.max_scenes,
        wrong_image_seed=args.wrong_image_seed,
    )
