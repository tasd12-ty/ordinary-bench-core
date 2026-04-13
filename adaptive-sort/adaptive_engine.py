"""
自适应排序评测引擎 — 全局距离对快速排序。

使用 VLM 作为比较器对场景中所有 C(N,2) 距离对进行排序。
第 0 层 pivot = GT 中位数；第 1 层及以上 pivot = 随机选取。
"""

from __future__ import annotations

import base64
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
from pathlib import Path
from typing import Tuple

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
from gt_ranking import compute_gt_global_ranking, gt_comparator_for_dist_pairs
from prompts import SYSTEM_PROMPT, format_partition_prompt
from scoring import score_global
from result_store import sort_result_to_dict, save_scene_result, save_summary
from job_spec import AdaptiveSortJobSpec
from mock_oracle import make_mock_comparator

from extraction import parse_objects, load_scene
from image_resolver import resolve_scene_images
from response_parser import parse_batch_response
from providers.base import ProviderAdapter

logger = logging.getLogger(__name__)


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
):
    """创建比较器：(pivot_pair, candidate_pairs) -> (results, usage)。"""
    _openai_client = getattr(adapter, "client", None)
    _model = getattr(adapter.spec, "model", "")
    _options = getattr(adapter.spec, "options", {})

    def comparator_fn(pivot: DistPair, candidates: list[DistPair]):
        user_prompt, expected_qids = format_partition_prompt(
            objects, pivot, candidates,
        )
        usage_info = {"prompt_tokens": 0, "completion_tokens": 0}

        last_error = None
        for attempt in range(max_retries + 1):
            try:
                if _openai_client is not None:
                    raw_response, usage_info = _call_openai_with_usage(
                        _openai_client, _model, _options,
                        SYSTEM_PROMPT, user_prompt, image_inputs,
                    )
                else:
                    request = adapter.prepare_request(
                        system_prompt=SYSTEM_PROMPT,
                        user_prompt=user_prompt,
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
                for idx, cand in enumerate(candidates, 1):
                    qid = f"cmp_{idx:03d}"
                    ans = predictions.get(qid)
                    if ans is None:
                        raise ValueError(f"Missing answer for {qid} after {max_retries + 1} attempts")
                    ans = ans.strip()
                    if ans in ("~", "≈", "=", "eq", "approx"):
                        ans = "~="
                    if ans not in ("<", "~=", ">"):
                        raise ValueError(f"Invalid answer '{ans}' for {qid}")
                    result[pair_key(cand)] = ans
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


def _call_openai_with_usage(client, model, options, system_prompt, user_prompt, image_inputs):
    """直接调用 OpenAI 兼容 API，返回 (text, usage)。"""
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
    kwargs = dict(
        model=model, messages=messages,
        temperature=temp, max_tokens=max_tok, timeout=120,
        extra_body={"reasoning": {"effort": "none"}},
    )

    resp = client.chat.completions.create(**kwargs)
    text = resp.choices[0].message.content or ""

    # 内容为空时的回退处理（思考模型消耗了所有 token）
    if not text:
        logger.warning("Empty content, retrying without reasoning constraint")
        kwargs["extra_body"] = {}
        kwargs["max_tokens"] = max(max_tok, 8192)
        resp = client.chat.completions.create(**kwargs)
        text = resp.choices[0].message.content or ""

    usage = {"prompt_tokens": 0, "completion_tokens": 0}
    if resp.usage:
        usage["prompt_tokens"] = resp.usage.prompt_tokens or 0
        usage["completion_tokens"] = resp.usage.completion_tokens or 0

    return text, usage


# ── 场景处理 ──────────────────────────────────────────────────

def process_scene(
    scene_id: str,
    scene_path: Path,
    adapter: ProviderAdapter,
    job: AdaptiveSortJobSpec,
) -> dict:
    """处理单个场景：对所有 C(N,2) 距离对进行全局快速排序。"""
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
    gt_ranking, gt_tie_groups = compute_gt_global_ranking(objects, job.input.tau)

    # 解析图像
    image_inputs = resolve_scene_images(scene_id, job.images)

    # 创建比较器
    is_mock = job.provider.adapter == "mock_oracle"
    if is_mock:
        comparator_fn = make_mock_comparator(objects, job.input.tau)
    else:
        comparator_fn = _make_comparator_fn(
            adapter, objects, image_inputs,
            max_retries=job.sorting.max_retries,
            retry_base_delay=job.sorting.retry_base_delay,
        )

    # 全局快速排序
    with ThreadPoolExecutor(max_workers=max(job.sorting.max_concurrency, 2)) as executor:
        sr = quicksort_global(all_pairs, comparator_fn, gt_ranking, executor)

    # 评分
    scores = score_global(
        sr.ranking, sr.tie_groups,
        gt_ranking, gt_tie_groups,
        sr.total_comparisons, sr.total_api_calls, sr.num_levels,
        sr.total_prompt_tokens, sr.total_completion_tokens,
        n_objects,
    )

    # 保存
    output_dir = Path(job.output.results_dir) / job.run_name
    sr_dict = sort_result_to_dict(sr)
    save_scene_result(
        output_dir, scene_id, job.provider.model, n_objects,
        sr_dict, scores, gt_ranking, gt_tie_groups,
    )

    elapsed = time.time() - t0
    logger.info(
        "Scene %s done in %.1fs | kt=%.4f pw=%.4f savings=%.1f%% calls=%d",
        scene_id, elapsed,
        scores.get("kendall_tau", 0),
        scores.get("pairwise_accuracy", 0),
        scores.get("comparison_savings", 0) * 100,
        scores.get("total_api_calls", 0),
    )

    return scores


# ── 主入口 ────────────────────────────────────────────────────

def run_evaluation(job: AdaptiveSortJobSpec, adapter: ProviderAdapter):
    """对所有场景运行全局距离对快速排序评测。"""
    scenes = discover_scenes(job.input.scenes_dir, job.selection)
    if not scenes:
        logger.warning("No scenes found in %s", job.input.scenes_dir)
        return

    logger.info("Found %d scenes to evaluate", len(scenes))

    scene_scores = []
    for scene_id, scene_path in scenes:
        try:
            scores = process_scene(scene_id, scene_path, adapter, job)
            scene_scores.append(scores)
        except Exception as e:
            logger.error("Scene %s failed: %s", scene_id, e, exc_info=True)

    # 保存汇总
    output_dir = Path(job.output.results_dir) / job.run_name
    save_summary(output_dir, job.provider.model, scene_scores)

    # 打印
    if scene_scores:
        n = len(scene_scores)
        mean_kt = sum(s.get("kendall_tau", 0) for s in scene_scores) / n
        mean_pw = sum(s.get("pairwise_accuracy", 0) for s in scene_scores) / n
        total_cmp = sum(s.get("total_comparisons", 0) for s in scene_scores)
        total_exh = sum(s.get("exhaustive_comparisons", 0) for s in scene_scores)
        total_api = sum(s.get("total_api_calls", 0) for s in scene_scores)
        total_ptok = sum(s.get("prompt_tokens", 0) for s in scene_scores)
        total_ctok = sum(s.get("completion_tokens", 0) for s in scene_scores)
        savings = (1 - total_cmp / total_exh) * 100 if total_exh > 0 else 0

        print(f"\n{'='*60}")
        print(f"Global Distance-Pair Quicksort Complete")
        print(f"{'='*60}")
        print(f"Model:              {job.provider.model}")
        print(f"Scenes:             {n}")
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
