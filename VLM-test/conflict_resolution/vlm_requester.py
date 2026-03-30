"""
VLM 重问模块
============

对冲突检测识别出的问题，调用 VLM API 重新获取答案。

复用已有的评测基础设施：
    - providers/   — API 适配器（OpenAI、Gemini 等）
    - prompts.py   — 按题型构建 system prompt + user prompt
    - response_parser.py — 从 VLM 原始输出中解析 JSON 答案

设计原则：
    - 按题型分组发送（QRR / TRR / FDR 各自独立调用），
      与原始评测保持一致，确保 prompt 格式相同
    - 单次失败不中断流程，返回 None 表示该题未获取到新答案
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent / "API-test"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from prompts import TYPE_SYSTEM_PROMPTS, format_batch_user_prompt
from providers import create_provider_adapter
from response_parser import parse_batch_response


def reask_questions(
    questions: List[dict],
    scene_objects: List[dict],
    image_inputs: List[dict],
    provider_spec,
) -> Dict[str, Any]:
    """对冲突问题调用 VLM，返回新预测。

    Args:
        questions: 需重问的问题列表（含 qid、type、pair1/pair2 等字段）
        scene_objects: 场景中的物体描述列表（[{"id": "obj_0", "desc": "..."}]）
        image_inputs: 图片输入（[{"kind": "file", "value": "path"}]）
        provider_spec: VLM 服务配置（含 adapter、model、api_key 等）

    Returns:
        {qid: prediction} 映射。prediction 为 VLM 的新回答：
            QRR → "<" / "~=" / ">"
            TRR → 1~12 的整数
            FDR → 物体 ID 排序列表
        调用失败的题目值为 None。
    """
    if not questions:
        return {}

    # 创建 API 适配器（每次重问都新建，避免状态污染）
    adapter = create_provider_adapter(provider_spec)

    # 按题型分组：不同题型使用不同的 system prompt
    by_type: Dict[str, List[dict]] = defaultdict(list)
    for q in questions:
        by_type[q["type"]].append(q)

    predictions: Dict[str, Any] = {}

    for qtype, qs in by_type.items():
        # 选择题型对应的 system prompt（与原始评测一致）
        system_prompt = TYPE_SYSTEM_PROMPTS.get(
            qtype, TYPE_SYSTEM_PROMPTS.get("qrr", "")
        )
        # 构建 user prompt：物体列表 + 问题列表
        user_prompt = format_batch_user_prompt(scene_objects, qs)
        expected_qids = [q["qid"] for q in qs]

        # 构建请求（含图片）
        request = adapter.prepare_request(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            image_inputs=image_inputs,
        )

        try:
            # 调用 VLM API
            raw = adapter.call(request)
            # 解析响应中的 JSON 答案
            parsed = parse_batch_response(raw, expected_qids)
            predictions.update(parsed)
        except Exception as e:
            # 单次失败不中断整个消解流程
            print(f"    VLM 调用失败 ({qtype}): {e}")
            for qid in expected_qids:
                predictions[qid] = None

    return predictions
