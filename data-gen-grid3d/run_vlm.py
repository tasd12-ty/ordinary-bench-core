#!/usr/bin/env python3
"""
Run VLM API on 3D grid position questions.

Reads question JSONs, sends 6 orthographic view images + prompt to VLM,
parses response, and saves predictions.

Usage:
    # Single scene
    VLM_MODEL=openai/gpt-4o python run_vlm.py --question output/questions/g04_000000.json

    # All scenes
    VLM_MODEL=openai/gpt-4o python run_vlm.py --questions-dir output/questions

Environment variables:
    VLM_BASE_URL   — API endpoint (default: OpenRouter)
    VLM_API_KEY    — API key
    VLM_MODEL      — Model identifier
    VLM_CONCURRENCY — Parallel workers (default: 2)
    VLM_TIMEOUT    — Request timeout in seconds (default: 120)
    VLM_MAX_RETRIES — Retry count (default: 5)
"""

import argparse
import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Add VLM-test/API-test to import path for reuse
API_TEST_DIR = Path(__file__).resolve().parent.parent / "VLM-test" / "API-test"
if str(API_TEST_DIR) not in sys.path:
    sys.path.insert(0, str(API_TEST_DIR))

from vlm_client import make_client, call_vlm, load_image_base64
from response_parser import extract_json
from config import CONFIG

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def build_messages(question: dict, data_dir: Path) -> list:
    """
    Build OpenAI-compatible messages with 6 labeled images + prompt.

    Each image is preceded by a text label identifying the view and axes.
    """
    system_prompt = question["system_prompt"]

    # Build user content: interleave view labels + images, then prompt body
    content = []
    for i, img_spec in enumerate(question["images"], 1):
        # Text label before each image
        content.append({
            "type": "text",
            "text": f"[Image {i} — {img_spec['label']}]",
        })
        # Image
        img_path = data_dir / img_spec["path"]
        if not img_path.exists():
            logger.warning(f"Image not found: {img_path}")
            continue
        b64 = load_image_base64(str(img_path))
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}"},
        })

    # Object list and answer format (everything after the image descriptions)
    objects = question["objects"]
    lines = ["\nObjects in this scene:"]
    for j, obj in enumerate(objects, 1):
        lines.append(f"  {j}. {obj['desc']}")
    lines.append("")
    lines.append(
        "For each object, determine its grid position by combining information "
        "from at least two views. Answer as JSON:"
    )
    lines.append("")
    lines.append("[")
    for j, obj in enumerate(objects):
        comma = "," if j < len(objects) - 1 else ""
        lines.append(f'  {{"object": "{obj["desc"]}", "cell": "?"}}{comma}')
    lines.append("]")

    content.append({"type": "text", "text": "\n".join(lines)})

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content},
    ]


def parse_predictions(raw_response: str, objects: list) -> list:
    """Parse VLM response into prediction list."""
    try:
        parsed = extract_json(raw_response)
    except Exception as e:
        logger.error(f"Failed to parse response: {e}")
        return [{"object": obj["desc"], "cell": None} for obj in objects]

    if not isinstance(parsed, list):
        logger.warning(f"Expected list, got {type(parsed).__name__}")
        return [{"object": obj["desc"], "cell": None} for obj in objects]

    # Build lookup from parsed response
    pred_by_obj = {}
    for item in parsed:
        if isinstance(item, dict):
            obj_name = item.get("object", "")
            cell = item.get("cell", None)
            pred_by_obj[obj_name] = cell

    # Match against expected objects
    predictions = []
    for obj in objects:
        desc = obj["desc"]
        cell = pred_by_obj.get(desc)
        predictions.append({"object": desc, "cell": cell})

    return predictions


def run_scene(question_path: Path, data_dir: Path, output_dir: Path, client, config: dict) -> dict:
    """Run VLM on one scene question."""
    with open(question_path) as f:
        question = json.load(f)

    scene_id = question["scene_id"]
    logger.info(f"Processing {scene_id} ({question['n_objects']} objects)...")

    # Build messages
    messages = build_messages(question, data_dir)

    # Call VLM
    raw_response = call_vlm(
        client, messages, config["model"],
        temperature=config["temperature"],
        max_tokens=config["max_tokens"],
        max_retries=config["max_retries"],
        retry_base_delay=config["retry_base_delay"],
        timeout=config["timeout"],
        provider=config.get("provider", ""),
    )

    # Parse
    predictions = parse_predictions(raw_response, question["objects"])

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_path = output_dir / f"{scene_id}.json"
    with open(pred_path, "w") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)

    n_valid = sum(1 for p in predictions if p["cell"] is not None)
    logger.info(f"  {scene_id}: {n_valid}/{len(predictions)} objects parsed → {pred_path.name}")

    return {
        "scene_id": scene_id,
        "n_objects": len(predictions),
        "n_valid": n_valid,
        "predictions": predictions,
        "raw_response": raw_response,
    }


def main():
    parser = argparse.ArgumentParser(description="Run VLM on 3D grid questions")
    parser.add_argument("--question", "-q", default=None,
                        help="Single question JSON file")
    parser.add_argument("--questions-dir", default=None,
                        help="Directory of question JSONs")
    parser.add_argument("--data-dir", "-d", default="output",
                        help="Data directory containing images/ (default: output)")
    parser.add_argument("--output-dir", "-o", default=None,
                        help="Output predictions directory (default: {data-dir}/predictions)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir) if args.output_dir else data_dir / "predictions"

    config = CONFIG
    client = make_client(config["base_url"], config["api_key"])
    concurrency = config["max_concurrency"]

    logger.info(f"Model: {config['model']}")
    logger.info(f"Output: {output_dir}")

    # Collect question files
    question_paths = []
    if args.question:
        question_paths = [Path(args.question)]
    elif args.questions_dir:
        question_paths = sorted(Path(args.questions_dir).glob("*.json"))
    else:
        parser.print_help()
        print("\nError: provide --question or --questions-dir")
        sys.exit(1)

    if not question_paths:
        print("No question files found.")
        sys.exit(1)

    logger.info(f"Scenes: {len(question_paths)}, Concurrency: {concurrency}")

    # Run
    results = []
    if concurrency <= 1 or len(question_paths) == 1:
        for qp in question_paths:
            try:
                result = run_scene(qp, data_dir, output_dir, client, config)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed {qp.stem}: {e}")
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {
                pool.submit(run_scene, qp, data_dir, output_dir, client, config): qp
                for qp in question_paths
            }
            for future in as_completed(futures):
                qp = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Failed {qp.stem}: {e}")

    # Summary
    total = sum(r["n_objects"] for r in results)
    valid = sum(r["n_valid"] for r in results)
    print(f"\nDone: {len(results)} scenes, {valid}/{total} objects parsed")
    print(f"Predictions: {output_dir}")


if __name__ == "__main__":
    main()
