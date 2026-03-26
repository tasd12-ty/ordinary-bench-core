"""
三条件实验运行器。

条件 A：正确图片 + 空间问题（正常流程）
条件 B：错误图片（随机场景）+ 空间问题
条件 C：无图片 + 空间问题（纯文本）

用法：
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


# 无图片模式的 system prompt：与 batch 模式相同，但不引用图片
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
3. FDR (full distance ranking): Given an anchor object, rank all other objects
   by their 3D distance from the anchor, from nearest to farthest.
   Answer with a JSON list of object ID strings.

Respond ONLY with a JSON array. Each element must have "qid" and "answer".
For QRR: answer is a string "<", "~=", or ">".
For TRR: answer is an integer 1-12.
For FDR: answer is a list of object ID strings.

Example:
[{"qid": "qrr_0001", "answer": "<"}, {"qid": "trr_0001", "answer": 7}, {"qid": "fdr_0001", "answer": ["obj_2", "obj_1", "obj_3"]}]"""


def run_condition(
    condition: str,
    questions_dir: str = "output/questions",
    images_dir: str = "../data-gen/output/images/single_view",
    output_dir: str = None,
    split: str = None,
    max_scenes: int = None,
    wrong_image_seed: int = 42,
):
    """运行三条件实验的单个条件。

    Args:
        condition: "A"（正确图片）、"B"（错误图片）、"C"（无图片）
        questions_dir: 问题文件目录
        images_dir: 场景图片目录
        output_dir: 输出目录（为 None 时自动生成）
        split: 可选的 split 过滤（如 "n04"）
        max_scenes: 可选的场景数量限制
        wrong_image_seed: 条件 B 中随机选错误图片的随机种子
    """
    condition = condition.upper()
    assert condition in ("A", "B", "C"), f"Invalid condition: {condition}"

    if output_dir is None:
        model_slug = VLM_MODEL.replace("/", "--").replace(":", "--")
        output_dir = f"output/results/{model_slug}_condition_{condition}"

    os.makedirs(os.path.join(output_dir, "raw"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "scenes"), exist_ok=True)

    # 收集场景文件列表
    q_dir = Path(questions_dir)
    scene_files = sorted(q_dir.glob("*.json"))
    if split:
        scene_files = [f for f in scene_files if f.stem.startswith(split)]
    if max_scenes:
        scene_files = scene_files[:max_scenes]

    print(f"Condition {condition}: {len(scene_files)} scenes")
    print(f"Model: {VLM_MODEL}")
    print(f"Output: {output_dir}")

    # 条件 B：预收集所有可用图片路径，用于随机错误图片选取
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

        # 跳过已处理的场景
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

            # 根据条件构建消息
            if condition == "A":
                # 正常：使用正确图片
                img_path = os.path.join(images_dir, f"{scene_id}.png")
                if not os.path.exists(img_path):
                    print(f"  {scene_id}: image not found at {img_path}")
                    return None
                img_b64 = load_image_base64(img_path)
                messages = build_messages(BATCH_SYSTEM_PROMPT, user_prompt, img_b64)

            elif condition == "B":
                # 错误图片：随机选取其他场景的图片
                candidates = [p for p in all_image_paths
                              if p.stem != scene_id]
                if not candidates:
                    return None
                wrong_img = rng.choice(candidates)
                img_b64 = load_image_base64(str(wrong_img))
                messages = build_messages(BATCH_SYSTEM_PROMPT, user_prompt, img_b64)

            elif condition == "C":
                # 无图片：纯文本模式
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

                # 保存原始响应
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

        # 评分
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
              f"(D={s['qrr_disjoint_correct']}/{s['qrr_disjoint_total']}, "
              f"SA={s['qrr_shared_anchor_correct']}/{s['qrr_shared_anchor_total']}) "
              f"TRR_h={s['trr_hour_correct']}/{s['trr_total']} "
              f"missing={s['missing']}")

        return result

    # 并发处理所有场景
    results = []
    with ThreadPoolExecutor(max_workers=VLM_CONCURRENCY) as executor:
        futures = {executor.submit(process_scene, f): f for f in scene_files}
        for future in as_completed(futures):
            result = future.result()
            if result:
                results.append(result)

    # 保存汇总结果
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
        print(f"    disjoint: {summary['overall']['qrr_disjoint_accuracy']:.2%}")
        print(f"    shared_anchor: {summary['overall']['qrr_shared_anchor_accuracy']:.2%}")
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
