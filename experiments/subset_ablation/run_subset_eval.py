"""
子集消融实验 VLM 评估器。

自包含的评估脚本，复用现有 VLM 客户端和 provider，
但使用自定义 prompt（含 N/A 选项）和评分逻辑。

用法 (从 experiments/subset_ablation/ 目录运行):
    # 设置环境变量
    export VLM_BASE_URL="https://openrouter.ai/api/v1"
    export VLM_API_KEY="sk-..."
    export VLM_MODEL="openai/gpt-4o"

    # 运行评估 (需要 uv run 以使用项目虚拟环境)
    cd experiments/subset_ablation
    uv run python run_subset_eval.py --questions-dir output/questions/qrr \
        --images-dir output/images/single_view \
        --output-dir output/results/subset \
        --concurrency 4

    # 只跑 2 个子集试试
    uv run python run_subset_eval.py --questions-dir output/questions/qrr \
        --images-dir output/images/single_view \
        --output-dir output/results/subset \
        --limit 2
"""

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# 添加 VLM-test 到 sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
API_TEST_DIR = PROJECT_ROOT / "VLM-test" / "API-test"
VLM_TEST_DIR = PROJECT_ROOT / "VLM-test"
for p in [str(API_TEST_DIR), str(VLM_TEST_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from vlm_client import call_vlm, load_image_base64, make_client
from response_parser import extract_json

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── 自定义 Prompt（含 N/A 选项）──

SUBSET_SYSTEM_PROMPT = """\
You are a spatial reasoning assistant analyzing a 3D scene image.
You will receive a list of objects visible in the image and a set of distance comparison (QRR) questions.

IMPORTANT: Some questions reference object IDs that are NOT in the "Objects visible" list.
These objects are not present in the image. If ANY object mentioned in a question is not \
in the visible list, you MUST answer "N/A" for that question.

For each question, compare 3D distances — either between two pairs of objects, \
or from a common anchor object to two candidate objects.

Answer with exactly one of:
  "<"   — first pair/candidate is closer
  "~="  — approximately equal distance
  ">"   — first pair/candidate is farther
  "N/A" — one or more objects in the question are not visible in the image

Respond ONLY with a JSON array. Each element must have "qid" and "answer".

Example:
[{"qid": "mqrr_0001", "answer": "<"}, {"qid": "mqrr_0002", "answer": "N/A"}, {"qid": "mqrr_0003", "answer": "~="}]"""


def format_user_prompt(objects: list, questions: list) -> str:
    """构造 user prompt。objects 只列出子集图中可见的物体。"""
    lines = ["Objects visible in the image:"]
    for obj in objects:
        lines.append(f"  - {obj['id']}: {obj['desc']}")
    lines.append("")
    lines.append("Questions:")

    for q in questions:
        p1a, p1b = q["pair1"]
        p2a, p2b = q["pair2"]
        if q.get("variant") == "shared_anchor" and q.get("anchor"):
            anchor = q["anchor"]
            cand1 = next(obj for obj in q["pair1"] if obj != anchor)
            cand2 = next(obj for obj in q["pair2"] if obj != anchor)
            lines.append(
                f"[{q['qid']}] From anchor {anchor}, compare the distance to {cand1} "
                f"vs the distance to {cand2}. Answer: < / ~= / > / N/A"
            )
        else:
            lines.append(
                f"[{q['qid']}] Compare the distance between {p1a} and {p1b} "
                f"vs the distance between {p2a} and {p2b}. Answer: < / ~= / > / N/A"
            )

    return "\n".join(lines)


def parse_response(raw: str, expected_qids: list) -> dict:
    """解析 VLM 响应，支持 N/A 答案。"""
    result = {qid: None for qid in expected_qids}

    try:
        data = extract_json(raw)
    except ValueError as e:
        logger.error(f"Parse failed: {e}")
        return result

    if not isinstance(data, list):
        return result

    for item in data:
        if not isinstance(item, dict):
            continue
        qid = item.get("qid", "")
        answer = item.get("answer")
        if qid in result:
            # 规范化答案
            if isinstance(answer, str):
                answer = answer.strip()
                if answer.upper() in ("N/A", "NA"):
                    answer = "N/A"
                elif answer in ("<", "~=", ">"):
                    pass
                else:
                    answer = None  # 无法识别
            else:
                answer = None
            result[qid] = answer

    return result


def score_scene(predictions: dict, questions: list) -> dict:
    """评分一个子集场景。"""
    answerable_correct = 0
    answerable_total = 0
    refusal_correct = 0  # VLM 正确回答 N/A
    refusal_hallucinated = 0  # VLM 对不可答题给出了 < / ~= / >
    refusal_total = 0
    missing = 0

    per_question = []

    for q in questions:
        qid = q["qid"]
        pred = predictions.get(qid)
        is_answerable = q.get("answerable", True)
        gt = q.get("gt_comparator")

        row = {
            "qid": qid,
            "variant": q.get("variant", ""),
            "source": q.get("source", ""),
            "answerable": is_answerable,
            "predicted": pred,
        }

        if pred is None:
            missing += 1
            # 缺失回答计入对应分母作为失败
            if is_answerable:
                answerable_total += 1
            else:
                refusal_total += 1
            row["status"] = "missing"
        elif is_answerable:
            answerable_total += 1
            correct = (pred == gt)
            if correct:
                answerable_correct += 1
            row["gt"] = gt
            row["correct"] = correct
            row["status"] = "correct" if correct else "wrong"
        else:
            refusal_total += 1
            if pred == "N/A":
                refusal_correct += 1
                row["status"] = "correct_refusal"
            else:
                refusal_hallucinated += 1
                row["status"] = "hallucinated"
                row["hallucinated_answer"] = pred
            row["missing_objects"] = q.get("missing_objects", [])
            row["n_missing"] = q.get("n_missing", 0)

        per_question.append(row)

    return {
        "answerable_correct": answerable_correct,
        "answerable_total": answerable_total,
        "answerable_acc": round(answerable_correct / answerable_total, 4) if answerable_total else 0,
        "refusal_correct": refusal_correct,
        "refusal_hallucinated": refusal_hallucinated,
        "refusal_total": refusal_total,
        "refusal_rate": round(refusal_correct / refusal_total, 4) if refusal_total else 0,
        "missing": missing,
        "per_question": per_question,
    }


def process_one_scene(
    question_file: Path,
    images_dir: Path,
    client,
    model: str,
    provider: str = "",
    batch_size: int = 20,
    react_max_rounds: int = 2,
    missing_threshold: float = 0.2,
) -> dict:
    """处理一个子集场景：加载问题 → 调用 VLM → 评分。"""
    with open(question_file) as f:
        qdata = json.load(f)

    scene_id = qdata["scene_id"]
    image_path = images_dir / f"{scene_id}.png"

    if not image_path.exists():
        return {"scene_id": scene_id, "status": "no_image", "error": str(image_path)}

    # 加载图片
    image_b64 = load_image_base64(str(image_path))
    image_url = f"data:image/png;base64,{image_b64}"

    # 收集所有问题
    all_questions = []
    for batch in qdata["batches"]:
        all_questions.extend(batch["questions"])

    # 分 batch 调用 VLM，含 ReAct 重问循环
    all_predictions = {}

    for i in range(0, len(all_questions), batch_size):
        chunk = all_questions[i:i + batch_size]
        user_prompt = format_user_prompt(qdata["objects"], chunk)

        content = [
            {"type": "image_url", "image_url": {"url": image_url}},
            {"type": "text", "text": user_prompt},
        ]
        messages = [
            {"role": "system", "content": SUBSET_SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]

        expected_qids = [q["qid"] for q in chunk]

        try:
            raw_response = call_vlm(client, messages, model, provider=provider)
            preds = parse_response(raw_response, expected_qids)

            # ReAct 重问循环：缺答过多时追加纠正 prompt
            missing_qids = [qid for qid, v in preds.items() if v is None]
            react_round = 0
            while (
                react_round < react_max_rounds
                and len(missing_qids) > len(expected_qids) * missing_threshold
            ):
                react_round += 1
                logger.info(
                    f"  {scene_id} batch {i//batch_size} ReAct #{react_round}: "
                    f"{len(missing_qids)}/{len(expected_qids)} missing"
                )
                correction = (
                    f"Your previous response is missing {len(missing_qids)} answers. "
                    f"Missing question IDs: {', '.join(missing_qids[:50])}\n\n"
                    f"Please output ONLY a valid JSON array for the missing questions. "
                    f"Format: [{{'qid': '...', 'answer': '...'}}]\n"
                    f"Remember: answer < / ~= / > / N/A"
                )
                messages.append({"role": "assistant", "content": raw_response})
                messages.append({"role": "user", "content": correction})

                raw_response = call_vlm(client, messages, model, provider=provider)
                retry_preds = parse_response(raw_response, missing_qids)

                # 合并补充的回答
                for qid, val in retry_preds.items():
                    if val is not None and preds.get(qid) is None:
                        preds[qid] = val
                missing_qids = [qid for qid, v in preds.items() if v is None]

            all_predictions.update(preds)
        except Exception as e:
            logger.error(f"{scene_id} batch {i//batch_size}: {e}")
            for qid in expected_qids:
                all_predictions[qid] = None

    # 评分
    scores = score_scene(all_predictions, all_questions)
    scores["scene_id"] = scene_id
    scores["parent_scene_id"] = qdata.get("parent_scene_id", "")
    scores["n_objects_in_image"] = qdata["n_objects"]
    scores["n_objects_parent"] = qdata.get("n_objects_parent", 0)
    scores["total_questions"] = len(all_questions)

    return scores


def main():
    parser = argparse.ArgumentParser(description="子集消融实验 VLM 评估")
    parser.add_argument("--questions-dir", required=True, help="questions/qrr/ 目录")
    parser.add_argument("--images-dir", required=True, help="images/single_view/ 目录")
    parser.add_argument("--output-dir", required=True, help="结果输出目录")
    parser.add_argument("--concurrency", type=int, default=4, help="并行线程数")
    parser.add_argument("--limit", type=int, default=None, help="只评估前 N 个子集")
    parser.add_argument("--base-url", default=None,
                        help="VLM API URL (默认读取 VLM_BASE_URL 环境变量)")
    parser.add_argument("--api-key", default=None,
                        help="API key (默认读取 VLM_API_KEY 环境变量)")
    parser.add_argument("--model", default=None,
                        help="模型 ID (默认读取 VLM_MODEL 环境变量)")
    parser.add_argument("--provider", default="", help="OpenRouter provider 前缀")
    parser.add_argument("--react-rounds", type=int, default=2,
                        help="ReAct 重问最大轮数 (默认 2)")
    parser.add_argument("--missing-threshold", type=float, default=0.2,
                        help="缺答率超过此值触发重问 (默认 0.2)")
    parser.add_argument("--batch-size", type=int, default=20,
                        help="每次 API 调用的问题数 (默认 20)")
    args = parser.parse_args()

    base_url = args.base_url or os.environ.get("VLM_BASE_URL", "https://openrouter.ai/api/v1")
    api_key = args.api_key or os.environ.get("VLM_API_KEY", "")
    model = args.model or os.environ.get("VLM_MODEL", "")

    if not api_key:
        print("ERROR: Set VLM_API_KEY environment variable or use --api-key")
        sys.exit(1)
    if not model:
        print("ERROR: Set VLM_MODEL environment variable or use --model")
        sys.exit(1)

    questions_dir = Path(args.questions_dir)
    images_dir = Path(args.images_dir)
    output_dir = Path(args.output_dir)
    scenes_out = output_dir / "scenes"
    scenes_out.mkdir(parents=True, exist_ok=True)

    # 发现问题文件
    question_files = sorted(questions_dir.glob("*.json"))
    if args.limit:
        question_files = question_files[:args.limit]

    # 跳过已有结果
    todo = []
    for qf in question_files:
        result_path = scenes_out / qf.name
        if result_path.exists():
            continue
        todo.append(qf)

    logger.info(f"Total: {len(question_files)}, already done: {len(question_files) - len(todo)}, todo: {len(todo)}")
    logger.info(f"Model: {model}, Base URL: {base_url}")

    if not todo:
        logger.info("Nothing to do.")
        return

    client = make_client(base_url, api_key)
    ok = 0
    errors = 0

    def _process(qf):
        return process_one_scene(
            qf, images_dir, client, model, args.provider,
            batch_size=args.batch_size,
            react_max_rounds=args.react_rounds,
            missing_threshold=args.missing_threshold,
        )

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {executor.submit(_process, qf): qf for qf in todo}

        for i, future in enumerate(as_completed(futures)):
            qf = futures[future]
            try:
                result = future.result()
                out_path = scenes_out / qf.name
                with open(out_path, "w") as f:
                    json.dump(result, f, indent=2)

                if result.get("status") == "no_image":
                    errors += 1
                    logger.warning(f"No image: {result['scene_id']}")
                else:
                    ok += 1
                    acc = result.get("answerable_acc", 0)
                    ref = result.get("refusal_rate", 0)
                    logger.info(
                        f"[{i+1}/{len(todo)}] {result['scene_id']}: "
                        f"acc={acc:.2f} refusal_rate={ref:.2f}"
                    )
            except Exception as e:
                errors += 1
                logger.error(f"Failed {qf.stem}: {e}")

    logger.info(f"Done: {ok} ok, {errors} errors")

    # 汇总
    all_results = []
    for f in sorted(scenes_out.glob("*.json")):
        with open(f) as fh:
            all_results.append(json.load(fh))

    total_ans_c = sum(r.get("answerable_correct", 0) for r in all_results)
    total_ans_t = sum(r.get("answerable_total", 0) for r in all_results)
    total_ref_c = sum(r.get("refusal_correct", 0) for r in all_results)
    total_ref_t = sum(r.get("refusal_total", 0) for r in all_results)

    summary = {
        "model": model,
        "n_scenes": len(all_results),
        "answerable_acc": round(total_ans_c / total_ans_t, 4) if total_ans_t else 0,
        "answerable_correct": total_ans_c,
        "answerable_total": total_ans_t,
        "refusal_rate": round(total_ref_c / total_ref_t, 4) if total_ref_t else 0,
        "refusal_correct": total_ref_c,
        "refusal_total": total_ref_t,
    }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== Summary ===")
    print(f"Model: {model}")
    print(f"Scenes: {len(all_results)}")
    print(f"Answerable accuracy: {summary['answerable_acc']:.4f} ({total_ans_c}/{total_ans_t})")
    print(f"Refusal rate: {summary['refusal_rate']:.4f} ({total_ref_c}/{total_ref_t})")


if __name__ == "__main__":
    main()
