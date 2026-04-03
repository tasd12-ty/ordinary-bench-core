"""
子集消融实验 — 多视角 VLM 评估器。

支持三种图片输入模式：
  --mode multi_view   : 传 4 张视角图给 VLM（默认）
  --mode single_view  : 使用 single_view 目录的单张图
  --mode pick_view N  : 从 multi_view 中选第 N 个视角作为单张图

每个子集场景的输出为一个 JSON，包含：
- answerable_acc: 可回答问题的准确率
- refusal_rate: N/A 问题的正确拒答率
- hallucination_rate: N/A 问题的幻觉率（给出了具体答案）
- per_question: 逐题详情

用法:
    cd experiments/subset_ablation
    export VLM_BASE_URL="https://openrouter.ai/api/v1"
    export VLM_API_KEY="sk-..."
    export VLM_MODEL="openai/gpt-4o"

    # 多视角（4 张图传给 VLM）
    uv run python run_subset_eval_multiview.py \
        --questions-dir output/questions/qrr \
        --images-dir output/images \
        --output-dir output/results/gpt4o_multiview \
        --mode multi_view

    # 用多视角中的 view_0 做单视角测试
    uv run python run_subset_eval_multiview.py \
        --questions-dir output/questions/qrr \
        --images-dir output/images \
        --output-dir output/results/gpt4o_view0 \
        --mode pick_view --view-index 0

    # 传统单视角
    uv run python run_subset_eval_multiview.py \
        --questions-dir output/questions/qrr \
        --images-dir output/images \
        --output-dir output/results/gpt4o_single \
        --mode single_view
"""

import argparse
import json
import logging
import os
import sys
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

# ── System Prompt ──

SYSTEM_PROMPT_MULTI = """\
You are a spatial reasoning assistant analyzing a 3D scene from multiple camera views.
You will receive 4 images of the SAME scene from different angles, a list of objects visible in the scene, and a set of distance comparison (QRR) questions.

IMPORTANT: Some questions reference object IDs that are NOT in the "Objects visible" list.
These objects are not present in the scene. If ANY object mentioned in a question is not in the visible list, you MUST answer "N/A" for that question.

For each question, compare 3D distances — either between two pairs of objects, or from a common anchor object to two candidate objects. Use ALL views to reason about the 3D spatial layout.

Answer with exactly one of:
  "<"   — first pair/candidate is closer
  "~="  — approximately equal distance
  ">"   — first pair/candidate is farther
  "N/A" — one or more objects in the question are not visible in the scene

Respond ONLY with a JSON array. Each element must have "qid" and "answer".
Example: [{"qid": "mqrr_0001", "answer": "<"}, {"qid": "mqrr_0002", "answer": "N/A"}]"""

SYSTEM_PROMPT_SINGLE = """\
You are a spatial reasoning assistant analyzing a 3D scene image.
You will receive a list of objects visible in the image and a set of distance comparison (QRR) questions.

IMPORTANT: Some questions reference object IDs that are NOT in the "Objects visible" list.
These objects are not present in the image. If ANY object mentioned in a question is not in the visible list, you MUST answer "N/A" for that question.

For each question, compare 3D distances — either between two pairs of objects, or from a common anchor object to two candidate objects.

Answer with exactly one of:
  "<"   — first pair/candidate is closer
  "~="  — approximately equal distance
  ">"   — first pair/candidate is farther
  "N/A" — one or more objects in the question are not visible in the image

Respond ONLY with a JSON array. Each element must have "qid" and "answer".
Example: [{"qid": "mqrr_0001", "answer": "<"}, {"qid": "mqrr_0002", "answer": "N/A"}]"""


# ── 图片加载 ──

def load_images(scene_id: str, images_dir: Path, mode: str, view_index: int = 0):
    """加载图片，返回 (b64_list, system_prompt)。

    Returns:
        b64_list: base64 编码的图片列表（1 张或 4 张）
        system_prompt: 对应的 system prompt
    """
    if mode == "multi_view":
        mv_dir = images_dir / "multi_view" / scene_id
        b64_list = []
        for i in range(4):
            p = mv_dir / f"view_{i}.png"
            if not p.exists():
                return None, None
            b64_list.append(load_image_base64(str(p)))
        return b64_list, SYSTEM_PROMPT_MULTI

    elif mode == "pick_view":
        p = images_dir / "multi_view" / scene_id / f"view_{view_index}.png"
        if not p.exists():
            return None, None
        return [load_image_base64(str(p))], SYSTEM_PROMPT_SINGLE

    else:  # single_view
        p = images_dir / "single_view" / f"{scene_id}.png"
        if not p.exists():
            return None, None
        return [load_image_base64(str(p))], SYSTEM_PROMPT_SINGLE


# ── Prompt 构建 ──

def format_user_prompt(objects: list, questions: list) -> str:
    obj_desc = ", ".join(
        f"{o['id']} ({o.get('desc', '')})" if isinstance(o, dict) else str(o)
        for o in objects
    )
    lines = [f"Objects visible: {obj_desc}", "", "Questions:"]
    for q in questions:
        if q.get("variant") == "shared_anchor":
            anchor = q.get("anchor", "")
            p1 = [x for x in q["pair1"] if x != anchor]
            p2 = [x for x in q["pair2"] if x != anchor]
            c1 = p1[0] if p1 else q["pair1"][1]
            c2 = p2[0] if p2 else q["pair2"][1]
            lines.append(
                f'  {q["qid"]}: From anchor {anchor}, compare distance to {c1} vs distance to {c2}.'
            )
        else:
            p1a, p1b = q["pair1"]
            p2a, p2b = q["pair2"]
            lines.append(
                f'  {q["qid"]}: Compare dist({p1a},{p1b}) vs dist({p2a},{p2b}).'
            )
    return "\n".join(lines)


# ── 响应解析 ──

def parse_response(raw: str, expected_qids: list) -> dict:
    result = {qid: None for qid in expected_qids}
    try:
        data = extract_json(raw)
    except ValueError:
        return result

    if not isinstance(data, list):
        return result

    for item in data:
        if not isinstance(item, dict):
            continue
        qid = item.get("qid", "")
        answer = item.get("answer")
        if qid in result:
            if isinstance(answer, str):
                a = answer.strip()
                if a in ("<", "~=", ">", "N/A"):
                    result[qid] = a
                elif a.upper() == "N/A" or a.lower() in ("na", "n/a"):
                    result[qid] = "N/A"

    return result


# ── 评分 ──

def score_scene(predictions: dict, questions: list) -> dict:
    answerable_correct = 0
    answerable_total = 0
    refusal_correct = 0
    refusal_hallucinated = 0
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

        per_question.append(row)

    return {
        "answerable_correct": answerable_correct,
        "answerable_total": answerable_total,
        "answerable_acc": round(answerable_correct / answerable_total, 4) if answerable_total else 0,
        "refusal_correct": refusal_correct,
        "refusal_hallucinated": refusal_hallucinated,
        "refusal_total": refusal_total,
        "refusal_rate": round(refusal_correct / refusal_total, 4) if refusal_total else 0,
        "hallucination_rate": round(refusal_hallucinated / refusal_total, 4) if refusal_total else 0,
        "missing": missing,
        "per_question": per_question,
    }


# ── 单场景处理 ──

def process_one_scene(
    question_file: Path,
    images_dir: Path,
    client,
    model: str,
    mode: str,
    view_index: int = 0,
    provider: str = "",
    batch_size: int = 20,
    react_max_rounds: int = 2,
    missing_threshold: float = 0.2,
) -> dict:
    with open(question_file) as f:
        qdata = json.load(f)

    scene_id = qdata["scene_id"]

    # 加载图片
    b64_list, system_prompt = load_images(scene_id, images_dir, mode, view_index)
    if b64_list is None:
        return {"scene_id": scene_id, "status": "no_image"}

    # 收集所有问题
    all_questions = []
    for batch in qdata["batches"]:
        all_questions.extend(batch["questions"])

    all_predictions = {}

    for i in range(0, len(all_questions), batch_size):
        chunk = all_questions[i:i + batch_size]
        user_prompt = format_user_prompt(qdata["objects"], chunk)

        # 构造消息：图片 + 文本
        content = []
        for img_b64 in b64_list:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{img_b64}"},
            })
        content.append({"type": "text", "text": user_prompt})

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]

        expected_qids = [q["qid"] for q in chunk]

        try:
            raw_response = call_vlm(client, messages, model, provider=provider)
            preds = parse_response(raw_response, expected_qids)

            # ReAct 重问循环
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
    scores["mode"] = mode
    if mode == "pick_view":
        scores["view_index"] = view_index

    return scores


# ── 汇总报告 ──

def generate_summary(results: list, model: str, mode: str, view_index: int = 0) -> dict:
    """按父场景分组汇总。"""
    from collections import defaultdict

    by_parent = defaultdict(list)
    for r in results:
        pid = r.get("parent_scene_id", "unknown")
        by_parent[pid].append(r)

    parent_summaries = {}
    for pid in sorted(by_parent):
        rs = by_parent[pid]
        ans_c = sum(r.get("answerable_correct", 0) for r in rs)
        ans_t = sum(r.get("answerable_total", 0) for r in rs)
        ref_c = sum(r.get("refusal_correct", 0) for r in rs)
        ref_h = sum(r.get("refusal_hallucinated", 0) for r in rs)
        ref_t = sum(r.get("refusal_total", 0) for r in rs)

        parent_summaries[pid] = {
            "n_subsets": len(rs),
            "n_objects_parent": rs[0].get("n_objects_parent", 0),
            "answerable_acc": round(ans_c / ans_t, 4) if ans_t else 0,
            "answerable_correct": ans_c,
            "answerable_total": ans_t,
            "refusal_rate": round(ref_c / ref_t, 4) if ref_t else 0,
            "hallucination_rate": round(ref_h / ref_t, 4) if ref_t else 0,
            "refusal_correct": ref_c,
            "refusal_hallucinated": ref_h,
            "refusal_total": ref_t,
        }

    total_ans_c = sum(r.get("answerable_correct", 0) for r in results)
    total_ans_t = sum(r.get("answerable_total", 0) for r in results)
    total_ref_c = sum(r.get("refusal_correct", 0) for r in results)
    total_ref_h = sum(r.get("refusal_hallucinated", 0) for r in results)
    total_ref_t = sum(r.get("refusal_total", 0) for r in results)

    summary = {
        "model": model,
        "mode": mode,
        "view_index": view_index if mode == "pick_view" else None,
        "n_scenes": len(results),
        "overall": {
            "answerable_acc": round(total_ans_c / total_ans_t, 4) if total_ans_t else 0,
            "answerable_correct": total_ans_c,
            "answerable_total": total_ans_t,
            "refusal_rate": round(total_ref_c / total_ref_t, 4) if total_ref_t else 0,
            "hallucination_rate": round(total_ref_h / total_ref_t, 4) if total_ref_t else 0,
            "refusal_correct": total_ref_c,
            "refusal_hallucinated": total_ref_h,
            "refusal_total": total_ref_t,
        },
        "by_parent": parent_summaries,
    }
    return summary


# ── 主程序 ──

def main():
    parser = argparse.ArgumentParser(description="子集消融实验 — 多视角 VLM 评估")
    parser.add_argument("--questions-dir", required=True, help="questions/qrr/ 目录")
    parser.add_argument("--images-dir", required=True,
                        help="images/ 目录（包含 single_view/ 和 multi_view/ 子目录）")
    parser.add_argument("--output-dir", required=True, help="结果输出目录")
    parser.add_argument("--mode", choices=["multi_view", "single_view", "pick_view"],
                        default="multi_view", help="图片模式 (默认 multi_view)")
    parser.add_argument("--view-index", type=int, default=0,
                        help="pick_view 模式下选择的视角索引 0-3 (默认 0)")
    parser.add_argument("--concurrency", type=int, default=4, help="并行线程数")
    parser.add_argument("--limit", type=int, default=None, help="只评估前 N 个子集")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--provider", default="")
    parser.add_argument("--react-rounds", type=int, default=2)
    parser.add_argument("--missing-threshold", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=20)
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

    mode_desc = args.mode
    if args.mode == "pick_view":
        mode_desc = f"pick_view (view_{args.view_index})"

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

    logger.info(f"Mode: {mode_desc}")
    logger.info(f"Model: {model}")
    logger.info(f"Total: {len(question_files)}, done: {len(question_files) - len(todo)}, todo: {len(todo)}")

    if not todo:
        logger.info("Nothing to do.")
        return

    client = make_client(base_url, api_key)
    ok = 0
    errors = 0

    def _process(qf):
        return process_one_scene(
            qf, images_dir, client, model, args.mode, args.view_index,
            args.provider, args.batch_size, args.react_rounds, args.missing_threshold,
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
                    hal = result.get("hallucination_rate", 0)
                    logger.info(
                        f"[{i+1}/{len(todo)}] {result['scene_id']}: "
                        f"acc={acc:.2f} refusal={ref:.2f} hallucination={hal:.2f}"
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

    summary = generate_summary(all_results, model, args.mode, args.view_index)

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # 打印报告
    o = summary["overall"]
    print(f"\n{'='*60}")
    print(f"Model: {model}")
    print(f"Mode:  {mode_desc}")
    print(f"Scenes: {summary['n_scenes']}")
    print(f"{'='*60}")
    print(f"Answerable accuracy:  {o['answerable_acc']:.4f} ({o['answerable_correct']}/{o['answerable_total']})")
    print(f"Refusal rate:         {o['refusal_rate']:.4f} ({o['refusal_correct']}/{o['refusal_total']})")
    print(f"Hallucination rate:   {o['hallucination_rate']:.4f} ({o['refusal_hallucinated']}/{o['refusal_total']})")
    print(f"{'='*60}")
    print(f"\nPer parent scene:")
    print(f"{'Parent':<16} {'N':>3} {'Subsets':>7} {'Ans Acc':>8} {'Refusal':>8} {'Halluc':>8}")
    print(f"{'-'*55}")
    for pid, ps in sorted(summary["by_parent"].items()):
        print(f"{pid:<16} {ps['n_objects_parent']:>3} {ps['n_subsets']:>7} "
              f"{ps['answerable_acc']:>8.4f} {ps['refusal_rate']:>8.4f} {ps['hallucination_rate']:>8.4f}")


if __name__ == "__main__":
    main()
