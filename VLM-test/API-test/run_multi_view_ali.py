#!/usr/bin/env python3
"""
DashScope OpenAI 兼容模式多视角批量测试脚本（支持 v1/v2 问题格式）
"""
import argparse
import json
import logging
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ali_config import ALI_CONFIG
from ali_vlm_client import (
    build_multi_view_prompt_messages,
    build_correction_prompt,
    call_ali_vlm,
)
from prompts import (
    MULTI_VIEW_SYSTEM_PROMPT,
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


def _model_dir_name(model: str) -> str:
    """将模型名转为安全的目录名。"""
    return model.replace("/", "--").replace(".", "_")


def _scene_multi_view_oss_urls(
    scene_id: str, config: dict, n_views: int
) -> list:
    """根据场景 ID 构造多视角 OSS 图片 URL 列表。"""
    oss_base = config["oss_base"].rstrip("/")
    return [
        f"{oss_base}/data-gen/output/images/multi_view/{scene_id}/view_{i}.png"
        for i in range(n_views)
    ]


# ── v1 模式：从混合 questions/{scene_id}.json 加载 ──


def process_scene_v1(scene_id: str, config: dict, n_views: int) -> dict:
    """处理单个场景（v1 混合格式，多视角），返回评分结果。"""
    questions_path = Path(config["questions_dir"]) / f"{scene_id}.json"
    with open(questions_path) as f:
        scene_data = json.load(f)

    image_oss_urls = _scene_multi_view_oss_urls(scene_id, config, n_views)
    system_prompt = MULTI_VIEW_SYSTEM_PROMPT.format(n_views=n_views)

    model_dir = _model_dir_name(config["model"]) + "_multi_view"
    raw_dir = Path(config["results_dir"]) / model_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    all_questions = []
    all_predictions = {}

    vlm_kwargs = {
        "base_url": config["base_url"],
        "api_key": config["api_key"],
        "model": config["model"],
        "temperature": config["temperature"],
        "max_tokens": config["max_tokens"],
        "max_retries": config["max_retries"],
        "retry_base_delay": config["retry_base_delay"],
        "timeout": config["timeout"],
    }

    for batch in scene_data["batches"]:
        batch_id = batch["batch_id"]
        questions = batch["questions"]
        all_questions.extend(questions)
        expected_qids = [q["qid"] for q in questions]

        user_prompt = format_batch_user_prompt(scene_data["objects"], questions)
        prompt = build_multi_view_prompt_messages(
            system_prompt, user_prompt, image_urls
        )

        logger.info(
            f"  {scene_id} batch {batch_id}/{scene_data['n_batches'] - 1} "
            f"({len(questions)} questions, {n_views} views)"
        )
        t0 = time.time()
        raw_response = call_ali_vlm(prompt=prompt, **vlm_kwargs)
        elapsed = time.time() - t0

        predictions = parse_batch_response(raw_response, expected_qids)
        n_missing = sum(1 for v in predictions.values() if v is None)

        # ReAct 纠正循环
        react_round = 0
        while (
            react_round < REACT_MAX_ROUNDS
            and n_missing > len(expected_qids) * REACT_MISSING_THRESHOLD
        ):
            react_round += 1
            missing_qids = [qid for qid, v in predictions.items() if v is None]
            logger.info(
                f"  {scene_id} batch {batch_id} ReAct #{react_round}，"
                f"缺失 {n_missing}/{len(expected_qids)}"
            )

            correction_text = REACT_CORRECTION_PROMPT.format(
                missing_qids=", ".join(missing_qids[:20]),
                n_missing=n_missing,
                n_total=len(expected_qids),
            )
            correction_prompt = build_correction_prompt(
                prompt, raw_response, correction_text
            )

            t1 = time.time()
            correction_response = call_ali_vlm(
                messages=correction_prompt, **vlm_kwargs
            )
            elapsed += time.time() - t1

            correction_preds = parse_batch_response(
                correction_response, missing_qids
            )
            for qid, val in correction_preds.items():
                if val is not None:
                    predictions[qid] = val

            raw_response = (
                (raw_response or "")
                + f"\n\n--- ReAct #{react_round} ---\n"
                + (correction_response or "")
            )
            n_missing = sum(1 for v in predictions.values() if v is None)

        raw_record = {
            "scene_id": scene_id,
            "batch_id": batch_id,
            "model": config["model"],
            "n_views": n_views,
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
        "n_views": n_views,
        "n_objects": scene_data["n_objects"],
        "n_batches": scene_data["n_batches"],
        "total_questions": len(all_questions),
        "scores": scores,
    }


# ── v2 模式：从分题型目录加载 ──


def _load_scene_questions_v2(scene_id, questions_dir, question_types=None):
    """从各题型子目录分别加载问题。"""
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
    """从题型子目录收集场景 ID。"""
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


def process_scene_v2(scene_id: str, config: dict, n_views: int) -> dict:
    """处理单个场景（v2 分题型格式，多视角），返回评分结果。"""
    from question_bank import make_batches

    questions_dir = Path(config["questions_dir"])
    question_types = config.get("question_types", QUESTION_TYPES)
    scene_meta, questions_by_type = _load_scene_questions_v2(
        scene_id, questions_dir, question_types
    )

    if scene_meta is None or not questions_by_type:
        raise ValueError(f"No questions found for {scene_id}")

    batch_size = config.get("batch_size", 20)
    image_oss_urls = _scene_multi_view_oss_urls(scene_id, config, n_views)

    model_dir = _model_dir_name(config["model"]) + "_multi_view"
    raw_dir = Path(config["results_dir"]) / model_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    all_predictions = {}
    all_questions = []
    total_batches = 0

    vlm_kwargs = {
        "base_url": config["base_url"],
        "api_key": config["api_key"],
        "model": config["model"],
        "temperature": config["temperature"],
        "max_tokens": config["max_tokens"],
        "max_retries": config["max_retries"],
        "retry_base_delay": config["retry_base_delay"],
        "timeout": config["timeout"],
    }

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
            prompt = build_multi_view_prompt_messages(
                system_prompt, user_prompt, image_oss_urls
            )

            logger.info(
                f"  {scene_id} {batch_id} "
                f"({len(questions)} {qtype.upper()}, {n_views} views)"
            )
            t0 = time.time()
            raw_response = call_ali_vlm(prompt=prompt, **vlm_kwargs)
            elapsed = time.time() - t0

            predictions = parse_batch_response(raw_response, expected_qids)
            n_missing = sum(1 for v in predictions.values() if v is None)

            # ReAct 纠正循环
            react_round = 0
            while (
                react_round < REACT_MAX_ROUNDS
                and n_missing > len(expected_qids) * REACT_MISSING_THRESHOLD
            ):
                react_round += 1
                missing_qids = [
                    qid for qid, v in predictions.items() if v is None
                ]
                logger.info(
                    f"  {scene_id} {batch_id} ReAct #{react_round}, "
                    f"missing {n_missing}/{len(expected_qids)}"
                )

                correction_text = REACT_CORRECTION_PROMPT.format(
                    missing_qids=", ".join(missing_qids[:20]),
                    n_missing=n_missing,
                    n_total=len(expected_qids),
                )
                correction_prompt = build_correction_prompt(
                    prompt, raw_response, correction_text
                )

                t1 = time.time()
                correction_response = call_ali_vlm(
                    prompt=correction_prompt, **vlm_kwargs
                )
                elapsed += time.time() - t1

                correction_preds = parse_batch_response(
                    correction_response, missing_qids
                )
                for qid, val in correction_preds.items():
                    if val is not None:
                        predictions[qid] = val

                raw_response = (
                    (raw_response or "")
                    + f"\n\n--- ReAct #{react_round} ---\n"
                    + (correction_response or "")
                )
                n_missing = sum(1 for v in predictions.values() if v is None)

            raw_record = {
                "scene_id": scene_id,
                "batch_id": batch_id,
                "question_type": qtype,
                "model": config["model"],
                "n_views": n_views,
                "timestamp": time.time(),
                "elapsed_seconds": round(elapsed, 2),
                "react_rounds": react_round,
                "prompt": prompt,
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
        "n_views": n_views,
        "n_objects": scene_meta["n_objects"],
        "n_batches": total_batches,
        "total_questions": len(all_questions),
        "scores": scores,
    }


# ── 主入口 ──


def main():
    parser = argparse.ArgumentParser(
        description="DashScope OpenAI 兼容模式多视角批量测试（支持 v1/v2 问题格式）"
    )
    parser.add_argument(
        "--split", default=None, help="只跑指定 split（如 n04）"
    )
    parser.add_argument(
        "--scene", default=None, help="只跑单个场景（如 n04_000000）"
    )
    parser.add_argument(
        "--n-views",
        type=int,
        default=4,
        choices=[1, 2, 3, 4],
        help="发送的视角数量（默认 4）",
    )
    parser.add_argument(
        "--v2",
        action="store_true",
        help="使用 v2 分题型目录格式（默认 v1 混合格式）",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=20,
        help="Batch 大小（仅 v2 模式，默认 20）",
    )
    parser.add_argument(
        "--types",
        nargs="+",
        choices=QUESTION_TYPES,
        default=None,
        help="只测指定题型（仅 v2 模式，如 --types qrr trr）",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--test-only",
        action="store_true",
        help="只跑测试集场景（从 test_scenes.json 读取）",
    )
    group.add_argument(
        "--train-only",
        action="store_true",
        help="只跑训练集场景（从 train_scenes.json 读取）",
    )
    args = parser.parse_args()

    config = ALI_CONFIG.copy()
    config["batch_size"] = args.batch_size
    config["question_types"] = args.types or QUESTION_TYPES

    # 强制并发不超过 10
    if config["max_concurrency"] > 10:
        logger.warning(
            f"并发数 {config['max_concurrency']} 超过上限 10，已自动调整为 10"
        )
        config["max_concurrency"] = 10

    questions_dir = Path(config["questions_dir"])
    split_file_dir = (
        Path(__file__).resolve().parent.parent.parent / "data-gen" / "output"
    )

    # 收集场景 ID
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

    # 按训练/测试集过滤
    if args.test_only or args.train_only:
        split_name = (
            "test_scenes.json" if args.test_only else "train_scenes.json"
        )
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

    # 选择处理函数
    if args.v2:
        process_fn = lambda sid, cfg: process_scene_v2(sid, cfg, args.n_views)
    else:
        process_fn = lambda sid, cfg: process_scene_v1(sid, cfg, args.n_views)

    mode_label = "v2（分题型）" if args.v2 else "v1（混合）"

    logger.info(
        f"阿里内部 API 多视角测试 {mode_label}：{len(scene_ids)} 个场景，"
        f"{args.n_views} 视角，模型 {config['model']}，"
        f"并发 {config['max_concurrency']}"
    )

    # 并发处理场景
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

    # 保存逐场景结果
    model_dir = _model_dir_name(config["model"]) + "_multi_view"
    scenes_dir = Path(config["results_dir"]) / model_dir / "scenes"
    scenes_dir.mkdir(parents=True, exist_ok=True)
    for result in results:
        scene_path = scenes_dir / f"{result['scene_id']}.json"
        with open(scene_path, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

    # 汇总
    summary = aggregate_batch_results(results)
    summary["model"] = config["model"]
    summary["n_views"] = args.n_views
    summary["n_scenes"] = len(results)
    summary["n_failed"] = len(failed_scenes)
    if failed_scenes:
        summary["failed_scenes"] = failed_scenes

    summary_path = Path(config["results_dir"]) / model_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # 打印结果
    overall = summary["overall"]
    print(
        f"\n=== DashScope 多视角测试结果 {mode_label} "
        f"({len(results)} 场景, {args.n_views} 视角) ==="
    )
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
