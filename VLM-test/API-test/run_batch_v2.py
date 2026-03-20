#!/usr/bin/env python3
"""
Batch 提问模式 v2：从按题型分目录的 questions/ 加载问题。

从 output/questions/{qrr,trr,fdr}/{scene_id}.json 加载各类型问题，
合并后分批发送给 VLM，解析响应并评分。

用法：
    python run_batch_v2.py                        # 全部场景
    python run_batch_v2.py --test-only            # 只跑测试集
    python run_batch_v2.py --train-only           # 只跑训练集
    python run_batch_v2.py --split n04            # 指定 split
    python run_batch_v2.py --scene n04_000000     # 单场景
    python run_batch_v2.py --batch-size 15        # 自定义 batch 大小
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# 将 VLM-test/ 加入 sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import CONFIG
from vlm_client import make_client, load_image_base64, build_messages, call_vlm
from prompts import (
    BATCH_SYSTEM_PROMPT, format_batch_user_prompt, REACT_CORRECTION_PROMPT,
    TYPE_SYSTEM_PROMPTS,
)
from response_parser import parse_batch_response
from scoring import score_batch_scene, aggregate_batch_results
from question_bank import make_batches

QUESTION_TYPES = ["qrr", "trr", "fdr"]
REACT_MAX_ROUNDS = 2
REACT_MISSING_THRESHOLD = 0.2
REACT_CHUNK_SIZE = 50  # Max missing qids per correction prompt

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _model_dir_name(model: str) -> str:
    return model.replace("/", "--")


def load_scene_questions(scene_id: str, questions_dir: Path):
    """从各题型子目录分别加载问题，返回 (scene_meta, {type: questions_list})。"""
    questions_by_type = {}
    scene_meta = None

    for qtype in QUESTION_TYPES:
        path = questions_dir / qtype / f"{scene_id}.json"
        if not path.exists():
            continue
        with open(path) as f:
            data = json.load(f)
        if scene_meta is None:
            scene_meta = data
        type_questions = []
        for batch in data["batches"]:
            type_questions.extend(batch["questions"])
        if type_questions:
            questions_by_type[qtype] = type_questions

    return scene_meta, questions_by_type


def discover_scenes(questions_dir: Path, split: str = None) -> list:
    """从所有题型子目录收集场景 ID。"""
    scene_id_set = set()
    for qtype in QUESTION_TYPES:
        type_dir = questions_dir / qtype
        if type_dir.exists():
            scene_id_set.update(f.stem for f in type_dir.glob("*.json"))
    scene_ids = sorted(scene_id_set)
    if split:
        scene_ids = [s for s in scene_ids if s.startswith(split)]
    return scene_ids


def process_scene(scene_id: str, config: dict) -> dict:
    """处理单个场景的所有问题（按题型分开问答），返回评分结果。"""
    questions_dir = Path(config["questions_dir"])
    scene_meta, questions_by_type = load_scene_questions(scene_id, questions_dir)

    if scene_meta is None or not questions_by_type:
        raise ValueError(f"No questions found for {scene_id}")

    batch_size = config.get("batch_size", 20)

    # 加载图片（所有题型共享同一张图）
    image_path = Path(config["images_dir"]) / f"{scene_id}.png"
    image_b64 = load_image_base64(str(image_path))

    client = make_client(config["base_url"], config["api_key"])

    model_dir = _model_dir_name(config["model"])
    raw_dir = Path(config["results_dir"]) / model_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    all_predictions = {}
    all_questions = []
    total_batches = 0

    vlm_kwargs = dict(
        temperature=config["temperature"],
        max_tokens=config["max_tokens"],
        max_retries=config["max_retries"],
        retry_base_delay=config["retry_base_delay"],
        timeout=config["timeout"],
        provider=config.get("provider", ""),
    )

    # 按题型分开问答，每种题型使用专用 system prompt
    for qtype in QUESTION_TYPES:
        type_questions = questions_by_type.get(qtype, [])
        if not type_questions:
            continue

        all_questions.extend(type_questions)
        system_prompt = TYPE_SYSTEM_PROMPTS[qtype]
        batches = make_batches(type_questions, batch_size)

        for batch in batches:
            batch_id = f"{qtype}_{batch['batch_id']}"
            questions = batch["questions"]
            expected_qids = [q["qid"] for q in questions]

            user_prompt = format_batch_user_prompt(scene_meta["objects"], questions)
            messages = build_messages(system_prompt, user_prompt, image_b64)

            logger.info(f"  {scene_id} {batch_id} ({len(questions)} {qtype.upper()})")
            t0 = time.time()
            raw_response = call_vlm(client, messages, config["model"], **vlm_kwargs)
            elapsed = time.time() - t0

            predictions = parse_batch_response(raw_response, expected_qids)
            n_missing = sum(1 for v in predictions.values() if v is None)

            # ReAct 纠正循环（分段纠正，每次最多 REACT_CHUNK_SIZE 个 qid）
            react_round = 0
            while (react_round < REACT_MAX_ROUNDS
                   and n_missing > len(expected_qids) * REACT_MISSING_THRESHOLD):
                react_round += 1
                missing_qids = [qid for qid, v in predictions.items() if v is None]
                logger.info(f"  {scene_id} {batch_id} ReAct #{react_round}, "
                            f"missing {n_missing}/{len(expected_qids)}")

                # Split missing qids into chunks to avoid overly long prompts
                for chunk_start in range(0, len(missing_qids), REACT_CHUNK_SIZE):
                    chunk = missing_qids[chunk_start:chunk_start + REACT_CHUNK_SIZE]

                    correction_messages = messages + [
                        {"role": "assistant", "content": raw_response or ""},
                        {"role": "user", "content": REACT_CORRECTION_PROMPT.format(
                            missing_qids=", ".join(chunk),
                            n_missing=len(chunk),
                            n_total=len(expected_qids),
                        )},
                    ]

                    t1 = time.time()
                    correction_response = call_vlm(
                        client, correction_messages, config["model"], **vlm_kwargs)
                    elapsed += time.time() - t1

                    correction_preds = parse_batch_response(correction_response, chunk)
                    for qid, val in correction_preds.items():
                        if val is not None:
                            predictions[qid] = val

                    raw_response = (raw_response or "") + f"\n\n--- ReAct #{react_round} chunk {chunk_start // REACT_CHUNK_SIZE + 1} ---\n" + (correction_response or "")

                n_missing = sum(1 for v in predictions.values() if v is None)

            # 保存原始响应
            raw_record = {
                "scene_id": scene_id, "batch_id": batch_id,
                "question_type": qtype,
                "model": config["model"], "timestamp": time.time(),
                "elapsed_seconds": round(elapsed, 2),
                "react_rounds": react_round,
                "raw_response": raw_response,
            }
            with open(raw_dir / f"{scene_id}_{batch_id}.json", "w") as f:
                json.dump(raw_record, f, indent=2, ensure_ascii=False)

            all_predictions.update(predictions)
            total_batches += 1

    # 评分
    scores = score_batch_scene(all_predictions, all_questions)

    return {
        "scene_id": scene_id,
        "model": config["model"],
        "n_objects": scene_meta["n_objects"],
        "n_batches": total_batches,
        "total_questions": len(all_questions),
        "scores": scores,
    }


def main():
    parser = argparse.ArgumentParser(description="Batch 提问模式 v2（分题型目录）")
    parser.add_argument("--split", default=None, help="只跑指定 split（如 n04）")
    parser.add_argument("--scene", default=None, help="只跑单个场景（如 n04_000000）")
    parser.add_argument("--batch-size", type=int, default=20, help="Batch 大小（默认 20）")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--test-only", action="store_true",
                       help="只跑测试集场景（从 test_scenes.json 读取）")
    group.add_argument("--train-only", action="store_true",
                       help="只跑训练集场景（从 train_scenes.json 读取）")
    args = parser.parse_args()

    config = CONFIG.copy()
    config["batch_size"] = args.batch_size

    if not config["api_key"]:
        logger.error("请设置环境变量 VLM_API_KEY")
        sys.exit(1)

    questions_dir = Path(config["questions_dir"])

    if args.scene:
        scene_ids = [args.scene]
    else:
        scene_ids = discover_scenes(questions_dir, args.split)
        if args.test_only or args.train_only:
            split_file_dir = Path(__file__).resolve().parent.parent.parent / "data-gen" / "output"
            split_name = "test_scenes.json" if args.test_only else "train_scenes.json"
            with open(split_file_dir / split_name) as f:
                split_ids = {s["scene_id"] for s in json.load(f)}
            scene_ids = [s for s in scene_ids if s in split_ids]

    if not scene_ids:
        logger.error("未找到场景文件")
        sys.exit(1)

    logger.info(f"Batch 模式 v2：{len(scene_ids)} 个场景，模型 {config['model']}")

    results = []
    with ThreadPoolExecutor(max_workers=config["max_concurrency"]) as pool:
        futures = {pool.submit(process_scene, sid, config): sid for sid in scene_ids}
        for future in as_completed(futures):
            sid = futures[future]
            try:
                result = future.result()
                results.append(result)
                s = result["scores"]
                fdr_info = ""
                if s["fdr_total"] > 0:
                    fdr_info = (f", FDR exact {s['fdr_exact_correct']}/{s['fdr_total']}"
                                f" kendall {s['fdr_kendall_mean']:.3f}")
                logger.info(
                    f"  {sid}: QRR {s['qrr_correct']}/{s['qrr_total']} "
                    f"(D {s['qrr_disjoint_correct']}/{s['qrr_disjoint_total']}, "
                    f"SA {s['qrr_shared_anchor_correct']}/{s['qrr_shared_anchor_total']}), "
                    f"TRR hour {s['trr_hour_correct']}/{s['trr_total']}"
                    f"{fdr_info}, missing {s['missing']}"
                )
            except Exception as e:
                logger.error(f"  {sid} 失败: {e}")

    # 保存逐场景结果
    model_dir = _model_dir_name(config["model"])
    scenes_dir = Path(config["results_dir"]) / model_dir / "scenes"
    scenes_dir.mkdir(parents=True, exist_ok=True)
    for r in results:
        with open(scenes_dir / f"{r['scene_id']}.json", "w") as f:
            json.dump(r, f, indent=2, ensure_ascii=False)

    # 汇总
    summary = aggregate_batch_results(results)
    summary["model"] = config["model"]
    summary["n_scenes"] = len(results)

    summary_path = Path(config["results_dir"]) / model_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # 打印结果
    o = summary["overall"]
    print(f"\n=== Batch 模式 v2 结果 ({len(results)} 场景) ===")
    print(f"模型: {config['model']}")
    print(f"QRR 准确率: {o['qrr_accuracy']:.2%} ({o['qrr_correct']}/{o['qrr_total']})")
    print(
        f"  disjoint: {o['qrr_disjoint_accuracy']:.2%} "
        f"({o['qrr_disjoint_correct']}/{o['qrr_disjoint_total']})"
    )
    print(
        f"  shared_anchor: {o['qrr_shared_anchor_accuracy']:.2%} "
        f"({o['qrr_shared_anchor_correct']}/{o['qrr_shared_anchor_total']})"
    )
    print(f"TRR hour 准确率: {o['trr_hour_accuracy']:.2%} ({o['trr_hour_correct']}/{o['trr_total']})")
    print(f"TRR quadrant 准确率: {o['trr_quadrant_accuracy']:.2%} ({o['trr_quadrant_correct']}/{o['trr_total']})")
    if o["fdr_total"] > 0:
        print(f"FDR exact 准确率: {o['fdr_exact_accuracy']:.2%} ({o['fdr_exact_correct']}/{o['fdr_total']})")
        print(f"FDR Kendall τ 均值: {o['fdr_kendall_mean']:.4f}")
        print(f"FDR pairwise 均值: {o['fdr_pairwise_mean']:.4f}")
        print(f"FDR top-1 均值: {o['fdr_top1_mean']:.4f}")
    print(f"缺失: {o['missing']}")


if __name__ == "__main__":
    main()
