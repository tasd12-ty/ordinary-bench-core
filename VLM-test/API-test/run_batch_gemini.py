#!/usr/bin/env python3
"""
Gemini 系列模型单视角批量测试脚本。

通过阿里内部平台代理调用 Gemini API，使用 Gemini 原生 prompt 协议。
图片通过 OSS URL 传递（平台自动下载转 base64）。

用法：
    python run_batch_gemini.py                        # 全部场景（v1 格式）
    python run_batch_gemini.py --v2                   # 全部场景（v2 分题型格式）
    python run_batch_gemini.py --test-only            # 只跑测试集
    python run_batch_gemini.py --split n04            # 指定 split
    python run_batch_gemini.py --scene n04_000000     # 单场景
    python run_batch_gemini.py --v2 --types qrr       # 只测 QRR（v2 模式）
    python run_batch_gemini.py --v2 --types qrr fdr   # 只测 QRR + FDR

环境变量覆盖（可选）：
    GEMINI_VLM_MODEL       — 模型名称（默认 gemini-2.5-pro-06-17）
    GEMINI_VLM_CONCURRENCY — 并发数（默认 8，上限 10）
    GEMINI_VLM_MAX_TOKENS  — maxOutputTokens
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gemini_config import GEMINI_CONFIG
from gemini_vlm_client import (
    build_gemini_prompt,
    build_gemini_correction_prompt,
    call_gemini_vlm,
)
from prompts import (
    BATCH_SYSTEM_PROMPT,
    format_batch_user_prompt,
    REACT_CORRECTION_PROMPT,
    TYPE_SYSTEM_PROMPTS,
)
from response_parser import parse_batch_response
from scoring import score_batch_scene, aggregate_batch_results

QUESTION_TYPES = ["qrr", "trr", "fdr"]
REACT_MAX_ROUNDS = 2
REACT_MISSING_THRESHOLD = 0.2

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def _build_saved_prompt(system_instruction, prompt):
    """将 Gemini 格式的 system_instruction 和 prompt 合并为统一的消息数组保存。"""
    saved = []
    if system_instruction:
        system_text_parts = system_instruction.get("parts", [])
        system_text = "\n".join(
            part.get("text", "") for part in system_text_parts if "text" in part
        )
        saved.append({"role": "system", "content": system_text})
    for message in prompt:
        role = message.get("role", "user")
        parts = message.get("parts", [])
        content_items = []
        for part in parts:
            if "text" in part:
                content_items.append({"type": "text", "text": part["text"]})
            elif "inlineData" in part:
                inline = part["inlineData"]
                content_items.append({
                    "type": "image_url",
                    "image_url": {"url": inline.get("data", "")},
                })
        if len(content_items) == 1 and content_items[0]["type"] == "text":
            saved.append({"role": role, "content": content_items[0]["text"]})
        else:
            saved.append({"role": role, "content": content_items})
    return saved

def _model_dir_name(model: str) -> str:
    return model.replace("/", "--").replace(".", "_")


def _scene_oss_image_url(scene_id: str, config: dict) -> str:
    oss_base = config["oss_base"].rstrip("/")
    return f"{oss_base}/data-gen/output/images/single_view/{scene_id}.png"


def _build_vlm_kwargs(config: dict) -> dict:
    return {
        "api_url": config["api_url"],
        "model": config["model"],
        "access_key": config["access_key"],
        "quota_id": config["quota_id"],
        "user_id": config["user_id"],
        "app": config["app"],
        "temperature": config["temperature"],
        "max_output_tokens": config["max_output_tokens"],
        "include_thoughts": config["include_thoughts"],
        "thinking_budget": config["thinking_budget"],
        "max_retries": config["max_retries"],
        "retry_base_delay": config["retry_base_delay"],
        "timeout": config["timeout"],
    }


def _call_and_correct(
    prompt: list,
    system_instruction,
    expected_qids: list,
    vlm_kwargs: dict,
    scene_id: str,
    batch_id: str,
) -> tuple:
    """
    调用 Gemini VLM 并执行 ReAct 纠正循环。

    返回 (predictions, raw_response, elapsed, react_round)。
    """
    t0 = time.time()
    raw_response = call_gemini_vlm(
        prompt=prompt, system_instruction=system_instruction, **vlm_kwargs
    )
    elapsed = time.time() - t0

    predictions = parse_batch_response(raw_response, expected_qids)
    n_missing = sum(1 for v in predictions.values() if v is None)

    react_round = 0
    while (
        react_round < REACT_MAX_ROUNDS
        and n_missing > len(expected_qids) * REACT_MISSING_THRESHOLD
    ):
        react_round += 1
        missing_qids = [qid for qid, v in predictions.items() if v is None]
        logger.info(
            f"  {scene_id} {batch_id} ReAct #{react_round}, "
            f"missing {n_missing}/{len(expected_qids)}"
        )

        correction_text = REACT_CORRECTION_PROMPT.format(
            missing_qids=", ".join(missing_qids[:20]),
            n_missing=n_missing,
            n_total=len(expected_qids),
        )
        correction_prompt, correction_sys = build_gemini_correction_prompt(
            prompt, system_instruction, raw_response, correction_text
        )

        t1 = time.time()
        correction_response = call_gemini_vlm(
            prompt=correction_prompt,
            system_instruction=correction_sys,
            **vlm_kwargs,
        )
        elapsed += time.time() - t1

        correction_preds = parse_batch_response(correction_response, missing_qids)
        for qid, val in correction_preds.items():
            if val is not None:
                predictions[qid] = val

        raw_response = (
            (raw_response or "")
            + f"\n\n--- ReAct #{react_round} ---\n"
            + (correction_response or "")
        )
        n_missing = sum(1 for v in predictions.values() if v is None)

    return predictions, raw_response, elapsed, react_round


# ── v1 模式 ──


def process_scene_v1(scene_id: str, config: dict) -> dict:
    questions_path = Path(config["questions_dir"]) / f"{scene_id}.json"
    with open(questions_path) as f:
        scene_data = json.load(f)

    image_oss_url = _scene_oss_image_url(scene_id, config)
    vlm_kwargs = _build_vlm_kwargs(config)

    model_dir = _model_dir_name(config["model"])
    raw_dir = Path(config["results_dir"]) / model_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    all_questions = []
    all_predictions = {}

    for batch in scene_data["batches"]:
        batch_id = batch["batch_id"]
        questions = batch["questions"]
        all_questions.extend(questions)
        expected_qids = [q["qid"] for q in questions]

        user_prompt = format_batch_user_prompt(scene_data["objects"], questions)
        prompt, system_instruction = build_gemini_prompt(
            BATCH_SYSTEM_PROMPT, user_prompt, image_oss_url
        )

        logger.info(
            f"  {scene_id} batch {batch_id}/{scene_data['n_batches'] - 1} "
            f"({len(questions)} questions)"
        )

        predictions, raw_response, elapsed, react_round = _call_and_correct(
            prompt, system_instruction, expected_qids, vlm_kwargs,
            scene_id, str(batch_id),
        )

        raw_record = {
            "scene_id": scene_id,
            "batch_id": batch_id,
            "model": config["model"],
            "timestamp": time.time(),
            "elapsed_seconds": round(elapsed, 2),
            "react_rounds": react_round,
            "raw_response": raw_response,
        }
        with open(raw_dir / f"{scene_id}_batch_{batch_id}.json", "w") as f:
            json.dump(raw_record, f, indent=2, ensure_ascii=False)

        all_predictions.update(predictions)

    scores = score_batch_scene(all_predictions, all_questions)

    return {
        "scene_id": scene_id,
        "model": config["model"],
        "n_objects": scene_data["n_objects"],
        "n_batches": scene_data["n_batches"],
        "total_questions": len(all_questions),
        "scores": scores,
    }


# ── v2 模式 ──


def _load_scene_questions_v2(scene_id, questions_dir, question_types=None):
    types_to_load = question_types or QUESTION_TYPES
    questions_by_type = {}
    scene_meta = None

    for qtype in types_to_load:
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


def _discover_scenes_v2(questions_dir, split=None, question_types=None):
    types_to_scan = question_types or QUESTION_TYPES
    scene_id_set = set()
    for qtype in types_to_scan:
        type_dir = questions_dir / qtype
        if type_dir.exists():
            scene_id_set.update(f.stem for f in type_dir.glob("*.json"))
    scene_ids = sorted(scene_id_set)
    if split:
        scene_ids = [s for s in scene_ids if s.startswith(split)]
    return scene_ids


def process_scene_v2(scene_id: str, config: dict) -> dict:
    from question_bank import make_batches

    questions_dir = Path(config["questions_dir"])
    question_types = config.get("question_types", QUESTION_TYPES)
    scene_meta, questions_by_type = _load_scene_questions_v2(
        scene_id, questions_dir, question_types
    )

    if scene_meta is None or not questions_by_type:
        raise ValueError(f"No questions found for {scene_id}")

    batch_size = config.get("batch_size", 20)
    image_oss_url = _scene_oss_image_url(scene_id, config)
    vlm_kwargs = _build_vlm_kwargs(config)

    model_dir = _model_dir_name(config["model"])
    raw_dir = Path(config["results_dir"]) / model_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    all_predictions = {}
    all_questions = []
    total_batches = 0

    for qtype in question_types:
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

            user_prompt = format_batch_user_prompt(
                scene_meta["objects"], questions
            )
            prompt, system_instruction = build_gemini_prompt(
                system_prompt, user_prompt, image_oss_url
            )

            logger.info(
                f"  {scene_id} {batch_id} ({len(questions)} {qtype.upper()})"
            )

            predictions, raw_response, elapsed, react_round = _call_and_correct(
                prompt, system_instruction, expected_qids, vlm_kwargs,
                scene_id, batch_id,
            )

            saved_prompt = _build_saved_prompt(system_instruction, prompt)
            raw_record = {
                "scene_id": scene_id,
                "batch_id": batch_id,
                "question_type": qtype,
                "model": config["model"],
                "timestamp": time.time(),
                "elapsed_seconds": round(elapsed, 2),
                "react_rounds": react_round,
                "prompt": saved_prompt,
                "raw_response": raw_response,
            }
            with open(raw_dir / f"{scene_id}_{batch_id}.json", "w") as f:
                json.dump(raw_record, f, indent=2, ensure_ascii=False)

            all_predictions.update(predictions)
            total_batches += 1

    scores = score_batch_scene(all_predictions, all_questions)

    return {
        "scene_id": scene_id,
        "model": config["model"],
        "n_objects": scene_meta["n_objects"],
        "n_batches": total_batches,
        "total_questions": len(all_questions),
        "scores": scores,
    }


# ── 主入口 ──


def main():
    parser = argparse.ArgumentParser(
        description="Gemini 系列模型单视角批量测试（支持 v1/v2 问题格式）"
    )
    parser.add_argument("--split", default=None, help="只跑指定 split（如 n04）")
    parser.add_argument("--scene", default=None, help="只跑单个场景")
    parser.add_argument("--v2", action="store_true", help="使用 v2 分题型目录格式")
    parser.add_argument("--batch-size", type=int, default=20, help="Batch 大小（v2 模式）")
    parser.add_argument(
        "--types", nargs="+", choices=QUESTION_TYPES, default=None,
        help="只测指定题型（v2 模式）",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--test-only", action="store_true", help="只跑测试集")
    group.add_argument("--train-only", action="store_true", help="只跑训练集")
    args = parser.parse_args()

    config = GEMINI_CONFIG.copy()
    config["batch_size"] = args.batch_size
    config["question_types"] = args.types or QUESTION_TYPES

    if config["max_concurrency"] > 10:
        logger.warning(
            f"并发数 {config['max_concurrency']} 超过上限 10，已自动调整为 10"
        )
        config["max_concurrency"] = 10

    questions_dir = Path(config["questions_dir"])
    split_file_dir = (
        Path(__file__).resolve().parent.parent.parent / "data-gen" / "output"
    )

    if args.scene:
        scene_ids = [args.scene]
    elif args.v2:
        scene_ids = _discover_scenes_v2(
            questions_dir, args.split, config["question_types"]
        )
    else:
        files = sorted(questions_dir.glob("*.json"))
        scene_ids = [f.stem for f in files]
        if args.split:
            scene_ids = [s for s in scene_ids if s.startswith(args.split)]

    if args.test_only or args.train_only:
        split_name = "test_scenes.json" if args.test_only else "train_scenes.json"
        split_path = split_file_dir / split_name
        if split_path.exists():
            with open(split_path) as f:
                split_ids = {s["scene_id"] for s in json.load(f)}
            scene_ids = [s for s in scene_ids if s in split_ids]
        else:
            logger.warning(f"未找到 {split_path}，跳过训练/测试集过滤")

    if not scene_ids:
        logger.error("未找到场景文件")
        sys.exit(1)

    process_fn = process_scene_v2 if args.v2 else process_scene_v1
    mode_label = "v2（分题型）" if args.v2 else "v1（混合）"

    logger.info(
        f"Gemini 单视角测试 {mode_label}：{len(scene_ids)} 个场景，"
        f"模型 {config['model']}，并发 {config['max_concurrency']}"
    )

    results = []
    failed_scenes = []
    with ThreadPoolExecutor(max_workers=config["max_concurrency"]) as pool:
        futures = {
            pool.submit(process_fn, sid, config): sid for sid in scene_ids
        }
        for future in as_completed(futures):
            sid = futures[future]
            try:
                result = future.result()
                results.append(result)
                scores = result["scores"]
                fdr_info = ""
                if scores["fdr_total"] > 0:
                    fdr_info = (
                        f", FDR exact {scores['fdr_exact_correct']}"
                        f"/{scores['fdr_total']}"
                        f" kendall {scores['fdr_kendall_mean']:.3f}"
                    )
                logger.info(
                    f"  {sid}: QRR {scores['qrr_correct']}/{scores['qrr_total']} "
                    f"(D {scores['qrr_disjoint_correct']}/{scores['qrr_disjoint_total']}, "
                    f"SA {scores['qrr_shared_anchor_correct']}"
                    f"/{scores['qrr_shared_anchor_total']}), "
                    f"TRR hour {scores['trr_hour_correct']}/{scores['trr_total']}"
                    f"{fdr_info}, missing {scores['missing']}"
                )
            except Exception as exc:
                logger.error(f"  {sid} 失败: {exc}")
                failed_scenes.append(sid)

    model_dir = _model_dir_name(config["model"])
    scenes_dir = Path(config["results_dir"]) / model_dir / "scenes"
    scenes_dir.mkdir(parents=True, exist_ok=True)
    for result in results:
        with open(scenes_dir / f"{result['scene_id']}.json", "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

    summary = aggregate_batch_results(results)
    summary["model"] = config["model"]
    summary["n_scenes"] = len(results)
    summary["n_failed"] = len(failed_scenes)
    if failed_scenes:
        summary["failed_scenes"] = failed_scenes

    summary_path = Path(config["results_dir"]) / model_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    overall = summary["overall"]
    print(f"\n=== Gemini 单视角测试结果 {mode_label} ({len(results)} 场景) ===")
    print(f"模型: {config['model']}")
    print(
        f"QRR 准确率: {overall['qrr_accuracy']:.2%} "
        f"({overall['qrr_correct']}/{overall['qrr_total']})"
    )
    print(
        f"  disjoint: {overall['qrr_disjoint_accuracy']:.2%} "
        f"({overall['qrr_disjoint_correct']}/{overall['qrr_disjoint_total']})"
    )
    print(
        f"  shared_anchor: {overall['qrr_shared_anchor_accuracy']:.2%} "
        f"({overall['qrr_shared_anchor_correct']}"
        f"/{overall['qrr_shared_anchor_total']})"
    )
    print(
        f"TRR hour 准确率: {overall['trr_hour_accuracy']:.2%} "
        f"({overall['trr_hour_correct']}/{overall['trr_total']})"
    )
    print(
        f"TRR quadrant 准确率: {overall['trr_quadrant_accuracy']:.2%} "
        f"({overall['trr_quadrant_correct']}/{overall['trr_total']})"
    )
    if overall["fdr_total"] > 0:
        print(
            f"FDR exact 准确率: {overall['fdr_exact_accuracy']:.2%} "
            f"({overall['fdr_exact_correct']}/{overall['fdr_total']})"
        )
        print(f"FDR Kendall τ 均值: {overall['fdr_kendall_mean']:.4f}")
        print(f"FDR pairwise 均值: {overall['fdr_pairwise_mean']:.4f}")
        print(f"FDR top-1 均值: {overall['fdr_top1_mean']:.4f}")
    print(f"缺失: {overall['missing']}")
    if failed_scenes:
        print(f"失败场景: {len(failed_scenes)} 个")


if __name__ == "__main__":
    main()
