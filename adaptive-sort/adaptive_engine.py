"""
自适应排序评测引擎 — 全局距离对快速排序。

使用 VLM 作为比较器对场景中所有 C(N,2) 距离对进行排序。
第 0 层 pivot = GT 中位数；第 1 层及以上 pivot = 随机选取。

支持三层并发：
  - 场景间并发：多个场景同时处理
  - 层内并发：同一场景快排同层的多个子数组划分并发执行
  - 全局信号量：统一控制同时进行的 API 调用总数
  - 大场景优先调度：物体多的场景先启动，最大化并发利用率
"""

from __future__ import annotations

import base64
import json
import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations
from math import comb
from pathlib import Path
from typing import Any, Optional, Tuple

import sys
_SELF_DIR = str(Path(__file__).resolve().parent)
_VLM_TEST = str(Path(__file__).resolve().parent.parent / "VLM-test")
_API_TEST = str(Path(__file__).resolve().parent.parent / "VLM-test" / "API-test")
if _SELF_DIR not in sys.path:
    sys.path.insert(0, _SELF_DIR)
for _p in (_VLM_TEST, _API_TEST):
    if _p not in sys.path:
        sys.path.append(_p)

from sorting import quicksort_global, SortResult, pair_key, DistPair
from gt_ranking import compute_gt_global_ranking
from prompts import get_system_prompt, format_partition_prompt
from scoring import score_global
from result_store import sort_result_to_dict, save_scene_result, save_summary
from job_spec import AdaptiveSortJobSpec
from mock_oracle import make_mock_comparator

from extraction import parse_objects, load_scene
from image_resolver import resolve_scene_images
from response_parser import parse_batch_response
from providers.base import ProviderAdapter

logger = logging.getLogger(__name__)

APPROX_ANSWER_ALIASES = {
    "~", "~=", "≈", "=", "eq", "approx", "approximate",
    "approximately equal", "similar", "about the same",
    "约等于", "近似", "差不多", "相等", "等于",
}

STRICT_APPROX_RETRY_PROMPT = """\

Correction for this retry:
Your previous answer used approximate equality (`~=` or equivalent wording), \
which is invalid for this run. You must choose exactly one of "<" or ">" for \
every qid. Do not output "~=", "equal", "approximately equal", or any Chinese \
equivalent such as "约等于". Return JSON only."""


# ── 场景发现 ──────────────────────────────────────────────────

def discover_scenes(scenes_dir: str, selection) -> list[tuple[str, Path]]:
    """发现场景 JSON 文件，支持可选过滤条件。"""
    sdir = Path(scenes_dir)
    if not sdir.is_dir():
        raise FileNotFoundError(f"Scenes directory not found: {sdir}")

    results = []
    for p in sorted(sdir.glob("*.json")):
        sid = p.stem
        if selection.split and not sid.startswith(selection.split):
            continue
        if selection.scene and sid != selection.scene:
            continue
        results.append((sid, p))

    if selection.max_scenes and len(results) > selection.max_scenes:
        results = results[: selection.max_scenes]
    return results


# ── 比较器工厂 ──────────────────────────────────────────────────

def _make_comparator_fn(
    adapter: ProviderAdapter,
    objects: dict,
    image_inputs: list[dict],
    max_retries: int = 3,
    retry_base_delay: float = 2.0,
    allow_approx: bool = True,
):
    """创建比较器：(pivot_pair, candidate_pairs) -> (results, usage)。

    Args:
        allow_approx: 是否允许 VLM 回答 ~=。False 时提示词只给 < / >，
            VLM 返回的 ~= 会被视为非法并触发重试。
    """
    _openai_client = getattr(adapter, "client", None)
    _model = getattr(adapter.spec, "model", "")
    _options = getattr(adapter.spec, "options", {})
    _system_prompt = get_system_prompt(allow_approx)
    _valid_answers = ("<", "~=", ">") if allow_approx else ("<", ">")

    def comparator_fn(pivot: DistPair, candidates: list[DistPair]):
        user_prompt, expected_qids = format_partition_prompt(
            objects, pivot, candidates, allow_approx=allow_approx,
        )
        usage_info = {"prompt_tokens": 0, "completion_tokens": 0}
        prompt_for_attempt = user_prompt

        last_error = None
        for attempt in range(max_retries + 1):
            try:
                if _openai_client is not None:
                    raw_response, usage_info = _call_openai_with_usage(
                        _openai_client, _model, _options,
                        _system_prompt, prompt_for_attempt, image_inputs,
                    )
                else:
                    request = adapter.prepare_request(
                        system_prompt=_system_prompt,
                        user_prompt=prompt_for_attempt,
                        image_inputs=image_inputs,
                    )
                    raw_response = adapter.call(request)

                predictions = parse_batch_response(raw_response, expected_qids)

                # 检查缺失
                missing = [q for q, a in predictions.items() if a is None]
                if missing and attempt < max_retries:
                    logger.warning(
                        "Attempt %d: %d/%d missing, retrying...",
                        attempt + 1, len(missing), len(expected_qids),
                    )
                    time.sleep(retry_base_delay * (2 ** attempt))
                    continue

                # 按 pair_key 构建结果
                result = {}
                invalid_approx = []
                invalid_other = []
                for idx, cand in enumerate(candidates, 1):
                    qid = f"cmp_{idx:03d}"
                    ans = _normalize_answer(predictions.get(qid))
                    if ans is None:
                        raise ValueError(f"Missing answer for {qid} after {max_retries + 1} attempts")
                    # 当不允许 ~= 时，视为无效输出，需要重试
                    if not allow_approx and ans == "~=":
                        invalid_approx.append(qid)
                    elif ans not in _valid_answers:
                        invalid_other.append((qid, ans))
                    result[pair_key(cand)] = ans

                # 存在无效的 ~= 回答时触发重试
                if (invalid_approx or invalid_other) and attempt < max_retries:
                    if invalid_approx and not allow_approx:
                        prompt_for_attempt = user_prompt + STRICT_APPROX_RETRY_PROMPT
                    logger.warning(
                        "Attempt %d: invalid answers for %d/%d qids "
                        "(approx=%d, other=%d), retrying...",
                        attempt + 1, len(invalid_approx) + len(invalid_other),
                        len(candidates), len(invalid_approx), len(invalid_other),
                    )
                    time.sleep(retry_base_delay * (2 ** attempt))
                    continue

                # 最后一次重试仍有无效回答，报错
                if invalid_approx or invalid_other:
                    details = []
                    if invalid_approx:
                        details.append(f"approximate answers are not allowed for {invalid_approx}")
                    if invalid_other:
                        details.append(f"other invalid answers: {invalid_other}")
                    raise ValueError(
                        f"Invalid answers after {max_retries + 1} attempts "
                        f"(allow_approx={allow_approx}): " + "; ".join(details)
                    )
                return result, usage_info

            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    logger.warning("Attempt %d failed: %s. Retrying...", attempt + 1, e)
                    time.sleep(retry_base_delay * (2 ** attempt))
                else:
                    raise RuntimeError(
                        f"Comparator failed after {max_retries + 1} attempts: {last_error}"
                    ) from last_error

        raise RuntimeError(f"Comparator failed: {last_error}")

    return comparator_fn


def _normalize_answer(answer: Any) -> str | None:
    """Normalize model answer variants before validation."""
    if answer is None:
        return None
    ans = str(answer).strip().strip(' \t\r\n"\'`.,，。!！;；:：')
    if ans.lower() in APPROX_ANSWER_ALIASES:
        return "~="
    return ans


def _strip_thinking_tags(text: str) -> str:
    """剥离 Qwen3 思考模式的 <think> 标签，保留实际输出内容。"""
    m = re.search(r"</think>\s*(.*)", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    if re.search(r"<think>", text, re.IGNORECASE):
        return re.split(r"<think>", text, flags=re.IGNORECASE)[0].strip()
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()


def _extract_response_text(resp) -> str:
    """从 API 响应中提取实际内容文本。

    处理 vLLM + Qwen3 的 reasoning 分离：
      - message.content: 实际回答
      - message.reasoning: 思考过程（vLLM --reasoning-parser qwen3）

    如果 content 为空但 reasoning 不为空，说明 max_tokens 不够，
    模型把所有 token 都用在了思考上。此时尝试从 reasoning 中提取 JSON。
    """
    import re as _re
    msg = resp.choices[0].message
    text = msg.content or ""

    # 剥离可能混在 content 中的 <think> 标签
    if text:
        text = _strip_thinking_tags(text)

    # content 为空时，尝试从 reasoning 字段中提取 JSON
    if not text:
        reasoning = getattr(msg, "reasoning", None) or ""
        if reasoning:
            logger.warning(
                "Content empty but reasoning present (%d chars, finish=%s). "
                "Likely max_tokens too small for thinking mode.",
                len(reasoning), resp.choices[0].finish_reason,
            )
            json_match = _re.search(r'\[.*\]', reasoning, _re.DOTALL)
            if json_match:
                text = json_match.group(0)
                logger.info("Extracted JSON (%d chars) from reasoning field", len(text))

    return text


def _call_openai_with_usage(client, model, options, system_prompt, user_prompt, image_inputs):
    """直接调用 OpenAI 兼容 API，返回 (text, usage)。

    默认关闭 OpenRouter reasoning；options.extra_body 显式配置时覆盖默认值。
    """
    content = []
    for img in (image_inputs or []):
        if img["kind"] == "file":
            p = Path(img["value"])
            if p.exists():
                with open(p, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode("utf-8")
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"},
                })
        elif img["kind"] == "url":
            content.append({
                "type": "image_url",
                "image_url": {"url": img["value"]},
            })
    content.append({"type": "text", "text": user_prompt})

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content},
    ]

    temp = float(options.get("temperature", 0.0))
    max_tok = max(int(options.get("max_tokens", 1024)), 4096)
    req_timeout = int(options.get("timeout", 120))
    extra_body = _resolve_extra_body(options)

    kwargs = dict(
        model=model, messages=messages,
        temperature=temp, max_tokens=max_tok, timeout=req_timeout,
    )
    if extra_body:
        kwargs["extra_body"] = extra_body

    resp = client.chat.completions.create(**kwargs)
    text = _extract_response_text(resp)

    usage = {"prompt_tokens": 0, "completion_tokens": 0}
    if resp.usage:
        usage["prompt_tokens"] = resp.usage.prompt_tokens or 0
        usage["completion_tokens"] = resp.usage.completion_tokens or 0

    return text, usage


def _resolve_extra_body(options: dict) -> dict:
    """Resolve OpenAI-compatible extra_body with reasoning disabled by default."""
    if "extra_body" not in options:
        return {"reasoning": {"effort": "none"}}
    extra_body = options.get("extra_body") or {}
    if not isinstance(extra_body, dict):
        raise TypeError("provider.options.extra_body must be a table/dict")
    return dict(extra_body)


# ── 场景处理 ──────────────────────────────────────────────────
# ── 场景处理 ──────────────────────────────────────────────────

def process_scene(
    scene_id: str,
    scene_path: Path,
    adapter: ProviderAdapter,
    job: AdaptiveSortJobSpec,
    global_executor: Optional[ThreadPoolExecutor] = None,
    global_semaphore: Optional[threading.Semaphore] = None,
) -> dict:
    """处理单个场景：对所有 C(N,2) 距离对进行全局快速排序。

    Args:
        scene_id: 场景标识符。
        scene_path: 场景 JSON 文件路径。
        adapter: VLM provider 适配器。
        job: 评测任务配置。
        global_executor: 全局共享线程池。传入时复用该线程池进行
            层内并发划分；为 None 时创建场景独占线程池（向后兼容）。
        global_semaphore: 全局信号量。传入时用于限制跨场景的
            同时 API 调用总数。
    """
    t0 = time.time()
    scene = load_scene(str(scene_path))
    objects = parse_objects(scene)
    n_objects = len(objects)
    obj_ids = sorted(objects.keys())

    logger.info("Processing scene %s (%d objects, %d pairs)",
                scene_id, n_objects, n_objects * (n_objects - 1) // 2)

    # 所有 C(N,2) 距离对
    all_pairs: list[DistPair] = list(combinations(obj_ids, 2))

    # GT 全局排序（用于第 0 层 pivot 选择）
    allow_approx = job.input.allow_approx
    gt_ranking, gt_tie_groups = compute_gt_global_ranking(
        objects, job.input.tau, allow_approx=allow_approx,
    )

    # 解析图像
    image_inputs = resolve_scene_images(scene_id, job.images)

    # 创建比较器
    is_mock = job.provider.adapter == "mock_oracle"
    if is_mock:
        comparator_fn = make_mock_comparator(objects, job.input.tau, allow_approx=allow_approx)
    else:
        comparator_fn = _make_comparator_fn(
            adapter, objects, image_inputs,
            max_retries=job.sorting.max_retries,
            retry_base_delay=job.sorting.retry_base_delay,
            allow_approx=allow_approx,
        )

    # 全局快速排序 — 优先使用全局共享线程池
    if global_executor is not None:
        sr = quicksort_global(
            all_pairs, comparator_fn, gt_ranking,
            executor=global_executor, semaphore=global_semaphore,
        )
    else:
        with ThreadPoolExecutor(max_workers=max(job.sorting.max_concurrency, 2)) as local_executor:
            sr = quicksort_global(all_pairs, comparator_fn, gt_ranking, executor=local_executor)

    # 评分
    if sr.failed:
        scores = _failure_scores(scene_id, n_objects, len(gt_ranking), sr)
    else:
        scores = score_global(
            sr.ranking, sr.tie_groups,
            gt_ranking, gt_tie_groups,
            sr.total_comparisons, sr.total_api_calls, sr.num_levels,
            sr.total_prompt_tokens, sr.total_completion_tokens,
            n_objects,
        )
        scores["scene_id"] = scene_id
        scores["failed"] = False

    # 保存
    output_dir = Path(job.output.results_dir) / job.run_name
    sr_dict = sort_result_to_dict(sr)
    save_scene_result(
        output_dir, scene_id, job.provider.model, n_objects,
        sr_dict, scores, gt_ranking, gt_tie_groups,
    )

    elapsed = time.time() - t0
    if scores.get("failed"):
        logger.error("Scene %s failed in %.1fs | %s", scene_id, elapsed, scores.get("fail_reason", ""))
    else:
        logger.info(
            "Scene %s done in %.1fs | kt=%.4f pw=%.4f savings=%.1f%% calls=%d",
            scene_id, elapsed,
            scores.get("kendall_tau", 0),
            scores.get("pairwise_accuracy", 0),
            scores.get("comparison_savings", 0) * 100,
            scores.get("total_api_calls", 0),
        )

    return scores


def _failure_scores(scene_id: str, n_objects: int, n_pairs: int, sr: SortResult) -> dict:
    """Build a summary-compatible record for failed scene sorts."""
    exhaustive_disjoint = 3 * comb(n_objects, 4) if n_objects >= 4 else 0
    exhaustive_shared = n_objects * comb(n_objects - 1, 2) if n_objects >= 3 else 0
    return {
        "scene_id": scene_id,
        "failed": True,
        "fail_reason": sr.fail_reason,
        "n_pairs": n_pairs,
        "exact_match": False,
        "kendall_tau": None,
        "pairwise_accuracy": None,
        "total_comparisons": sr.total_comparisons,
        "exhaustive_comparisons": exhaustive_disjoint + exhaustive_shared,
        "comparison_savings": None,
        "total_api_calls": sr.total_api_calls,
        "num_levels": sr.num_levels,
        "prompt_tokens": sr.total_prompt_tokens,
        "completion_tokens": sr.total_completion_tokens,
    }


# ── 场景排序辅助 ──────────────────────────────────────────────

def _count_objects_fast(scene_path: Path) -> int:
    """快速读取场景 JSON 中的物体数量（仅解析 objects 字段长度）。"""
    try:
        with open(scene_path) as fh:
            scene = json.load(fh)
        return len(scene.get("objects", []))
    except Exception:
        return 0

# ── 主入口 ────────────────────────────────────────────────────

def run_evaluation(job: AdaptiveSortJobSpec, adapter: ProviderAdapter):
    """对所有场景运行全局距离对快速排序评测。

    并发策略：
      1. 场景按物体数量降序排列（大场景优先启动，保持高并发利用率）
      2. 全局 ThreadPoolExecutor 供所有场景共享
      3. 全局 Semaphore 限制同时进行的 API 调用总数 = max_concurrency
      4. 场景间通过 scene_executor 并发处理
      5. 每个场景内部的快排层间串行、层内并发（复用全局线程池）
    """
    scenes = discover_scenes(job.input.scenes_dir, job.selection)
    if not scenes:
        logger.warning("No scenes found in %s", job.input.scenes_dir)
        return

    # ── 大场景优先排序 ──
    scenes_with_size = [
        (_count_objects_fast(path), sid, path)
        for sid, path in scenes
    ]
    scenes_with_size.sort(key=lambda x: x[0], reverse=True)

    logger.info(
        "Found %d scenes to evaluate (sorted by n_objects desc: %s..%s)",
        len(scenes_with_size),
        scenes_with_size[0][0] if scenes_with_size else "?",
        scenes_with_size[-1][0] if scenes_with_size else "?",
    )

    max_concurrency = max(job.sorting.max_concurrency, 1)

    # 场景间并发数：0 = 自动（不限制，由信号量控制实际并发）
    max_scene_conc = job.sorting.max_scene_concurrency
    if max_scene_conc <= 0:
        max_scene_conc = min(len(scenes_with_size), max_concurrency)

    # 全局线程池的 worker 数 = max_concurrency + 场景并发数
    # 场景并发线程负责驱动各场景的 BFS 循环（大部分时间在等待
    # 层内任务完成），层内任务线程负责实际 API 调用。
    # 信号量确保同时进行的 API 调用不超过 max_concurrency。
    global_pool_size = max_concurrency + max_scene_conc
    global_semaphore = threading.Semaphore(max_concurrency)

    logger.info(
        "Concurrency config: max_api_concurrency=%d, max_scene_concurrency=%d, "
        "global_pool_size=%d",
        max_concurrency, max_scene_conc, global_pool_size,
    )

    scene_scores: list[dict] = []
    scene_scores_lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=global_pool_size) as global_executor:

        # 场景调度器：用独立线程池驱动场景间并发
        # 每个场景的 process_scene 内部复用 global_executor 做层内并发
        scene_futures = {}
        scene_executor = ThreadPoolExecutor(max_workers=max_scene_conc)

        try:
            for n_objects, scene_id, scene_path in scenes_with_size:
                fut = scene_executor.submit(
                    process_scene,
                    scene_id, scene_path, adapter, job,
                    global_executor, global_semaphore,
                )
                scene_futures[fut] = scene_id

            for fut in as_completed(scene_futures):
                scene_id = scene_futures[fut]
                try:
                    scores = fut.result()
                    with scene_scores_lock:
                        scene_scores.append(scores)
                except Exception as exc:
                    logger.error("Scene %s failed: %s", scene_id, exc, exc_info=True)
                    with scene_scores_lock:
                        scene_scores.append({
                            "scene_id": scene_id,
                            "failed": True,
                            "fail_reason": str(exc),
                        })
        finally:
            scene_executor.shutdown(wait=True)

    # 保存汇总
    output_dir = Path(job.output.results_dir) / job.run_name
    save_summary(output_dir, job.provider.model, scene_scores)

    # 打印
    _print_summary(job, scene_scores, output_dir)


def _print_summary(job: AdaptiveSortJobSpec, scene_scores: list[dict], output_dir: Path):
    """打印评测汇总信息。"""
    if not scene_scores:
        return

    successful = [s for s in scene_scores if not s.get("failed")]
    failed = [s for s in scene_scores if s.get("failed")]
    n = len(successful)
    mean_kt = sum(s.get("kendall_tau", 0) for s in successful) / n if n else 0
    mean_pw = sum(s.get("pairwise_accuracy", 0) for s in successful) / n if n else 0
    total_cmp = sum(s.get("total_comparisons", 0) for s in successful)
    total_exh = sum(s.get("exhaustive_comparisons", 0) for s in successful)
    total_api = sum(s.get("total_api_calls", 0) for s in successful)
    total_ptok = sum(s.get("prompt_tokens", 0) for s in successful)
    total_ctok = sum(s.get("completion_tokens", 0) for s in successful)
    savings = (1 - total_cmp / total_exh) * 100 if total_exh > 0 else 0

    print(f"\n{'='*60}")
    print(f"Global Distance-Pair Quicksort Complete")
    print(f"{'='*60}")
    print(f"Model:              {job.provider.model}")
    print(f"Scenes:             {len(scene_scores)} ({n} succeeded, {len(failed)} failed)")
    print(f"Mean Kendall tau:   {mean_kt:.4f}")
    print(f"Mean pairwise acc:  {mean_pw:.4f}")
    print(f"Total comparisons:  {total_cmp} (vs {total_exh} exhaustive QRR)")
    print(f"Comparison savings: {savings:.1f}%")
    print(f"Total API calls:    {total_api}")
    print(f"Prompt tokens:      {total_ptok}")
    print(f"Completion tokens:  {total_ctok}")
    print(f"Total tokens:       {total_ptok + total_ctok}")
    print(f"Results saved to:   {output_dir}")
    print(f"{'='*60}\n")
