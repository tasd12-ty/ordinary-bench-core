"""
Gemini 系列模型客户端。

通过阿里内部平台代理调用 Gemini API，prompt 协议遵循 Gemini 原生格式：
- 使用 parts（而非 OpenAI 的 content）
- 图片使用 inlineData（平台自动将 URL 下载转 base64）
- system prompt 使用顶层 systemInstruction 字段
- 参数放在 params 对象中（maxOutputTokens、use_gemini_httpstream_api 等）
- 响应中需过滤 thought 部分

包含指数退避重试。
"""

import json
import logging
import time
from typing import Optional

import requests

logger = logging.getLogger(__name__)

_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


def _build_image_part(oss_url: str) -> dict:
    """构造 Gemini 格式的图片 part（平台会自动将 URL 下载转 base64）。"""
    return {
        "inlineData": {
            "mimeType": "image/png",
            "data": oss_url,
        }
    }


def build_gemini_prompt(
    system_prompt: str,
    user_text: str,
    image_oss_url: str,
) -> tuple:
    """
    构造 Gemini 格式的 prompt 和 systemInstruction。

    返回 (prompt_list, system_instruction_dict)。
    """
    system_instruction = None
    if system_prompt:
        system_instruction = {
            "role": "user",
            "parts": [{"text": system_prompt}],
        }

    user_parts = [
        _build_image_part(image_oss_url),
        {"text": user_text},
    ]
    prompt = [{"role": "user", "parts": user_parts}]

    return prompt, system_instruction


def build_gemini_multi_view_prompt(
    system_prompt: str,
    user_text: str,
    image_oss_urls: list,
) -> tuple:
    """
    构造多视角图片的 Gemini 格式 prompt。

    返回 (prompt_list, system_instruction_dict)。
    """
    system_instruction = None
    if system_prompt:
        system_instruction = {
            "role": "user",
            "parts": [{"text": system_prompt}],
        }

    user_parts = [_build_image_part(url) for url in image_oss_urls]
    user_parts.append({"text": user_text})

    prompt = [{"role": "user", "parts": user_parts}]

    return prompt, system_instruction


def build_gemini_correction_prompt(
    original_prompt: list,
    original_system_instruction: Optional[dict],
    assistant_response: str,
    correction_text: str,
) -> tuple:
    """
    构造 ReAct 纠正轮的 Gemini prompt。

    在原始 prompt 后追加 model 回复和 user 纠正请求。
    返回 (prompt_list, system_instruction_dict)。
    """
    correction_prompt = original_prompt + [
        {"role": "model", "parts": [{"text": assistant_response or ""}]},
        {"role": "user", "parts": [{"text": correction_text}]},
    ]
    return correction_prompt, original_system_instruction


def call_gemini_vlm(
    api_url: str,
    prompt: list,
    model: str,
    access_key: str,
    quota_id: str,
    system_instruction: Optional[dict] = None,
    user_id: str = "345245",
    app: str = "llm_application",
    temperature: float = 0.0,
    max_output_tokens: int = 65536,
    include_thoughts: str = "true",
    thinking_budget: int = 2000,
    max_retries: int = 5,
    retry_base_delay: float = 2.0,
    timeout: int = 180,
) -> str:
    """
    调用 Gemini API（通过阿里内部平台代理），带指数退避重试和流式输出。

    使用 stream=True 实时接收数据，每收到一个 chunk 就打印进度，
    让用户可以随时观察运行状态。

    返回模型生成的文本内容（已过滤 thinking 部分）。
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "params": {
            "use_gemini_httpstream_api": True,
            "maxOutputTokens": max_output_tokens,
            "includeThoughts": include_thoughts,
            "thinkingBudget": thinking_budget,
        },
        "passparams": {},
        "app": app,
        "quota_id": quota_id,
        "access_key": access_key,
        "user_id": user_id,
    }

    if temperature > 0:
        payload["params"]["temperature"] = temperature

    if system_instruction:
        payload["systemInstruction"] = system_instruction

    headers = {"Content-Type": "application/json"}

    for attempt in range(max_retries):
        try:
            response = requests.post(
                api_url,
                json=payload,
                headers=headers,
                timeout=timeout,
                stream=True,
            )

            if response.status_code in _RETRYABLE_STATUS_CODES:
                response.close()
                delay = min(retry_base_delay * (2 ** attempt), 60.0)
                logger.warning(
                    f"重试 {attempt + 1}/{max_retries}，HTTP {response.status_code}，"
                    f"等待 {delay:.1f}s"
                )
                time.sleep(delay)
                continue

            response.raise_for_status()

            # 流式接收：逐 chunk 读取并实时打印进度
            chunks = []
            received_bytes = 0
            start_time = time.time()
            last_log_time = start_time

            for chunk in response.iter_content(chunk_size=4096):
                if chunk:
                    chunks.append(chunk)
                    received_bytes += len(chunk)
                    now = time.time()
                    # 每 3 秒打印一次接收进度
                    if now - last_log_time >= 3.0:
                        elapsed = now - start_time
                        logger.info(
                            f"    ⏳ 流式接收中… "
                            f"{received_bytes / 1024:.1f} KB, "
                            f"已用时 {elapsed:.1f}s"
                        )
                        last_log_time = now

            total_elapsed = time.time() - start_time
            logger.info(
                f"    ✅ 接收完成: {received_bytes / 1024:.1f} KB, "
                f"耗时 {total_elapsed:.1f}s"
            )

            body = b"".join(chunks)
            data = json.loads(body)
            content = _extract_gemini_content(data)
            if content:
                return content

            logger.warning(
                f"重试 {attempt + 1}/{max_retries}，响应内容为空，"
                f"响应体: {body.decode('utf-8', errors='replace')[:500]}"
            )
            delay = min(retry_base_delay * (2 ** attempt), 60.0)
            time.sleep(delay)

        except requests.exceptions.Timeout:
            delay = min(retry_base_delay * (2 ** attempt), 60.0)
            logger.warning(
                f"重试 {attempt + 1}/{max_retries}，请求超时，等待 {delay:.1f}s"
            )
            time.sleep(delay)

        except requests.exceptions.ConnectionError as exc:
            delay = min(retry_base_delay * (2 ** attempt), 60.0)
            logger.warning(
                f"重试 {attempt + 1}/{max_retries}，连接错误: {exc}，等待 {delay:.1f}s"
            )
            time.sleep(delay)

        except requests.exceptions.HTTPError as exc:
            raise RuntimeError(
                f"API 请求失败: HTTP {response.status_code}, "
                f"响应: {response.text[:500]}"
            ) from exc

    raise RuntimeError(f"Gemini API 调用失败，已重试 {max_retries} 次")


def _extract_gemini_content(data: dict) -> Optional[str]:
    """
    从 Gemini API 响应中提取生成的文本内容。

    响应格式层级（按优先级尝试）：
    1. data.data.completion.content — 平台已提取的纯文本（最可靠）
    2. data.data.message — 平台提取的 message 字段
    3. data.data.completion.candidates[0].content.parts — 原始 parts（需过滤 thought）
    """
    inner = data.get("data")
    if not isinstance(inner, dict):
        return None

    completion = inner.get("completion")
    if isinstance(completion, dict):
        # 优先使用平台已提取的 content 字段（不含 thinking）
        content = completion.get("content")
        if content and isinstance(content, str):
            return content

        # 从 candidates 中提取（需过滤 thought 部分）
        candidates = completion.get("candidates")
        if candidates and isinstance(candidates, list) and len(candidates) > 0:
            candidate_content = candidates[0].get("content", {})
            parts = candidate_content.get("parts", [])
            text_parts = [
                part["text"]
                for part in parts
                if isinstance(part, dict)
                and "text" in part
                and not part.get("thought", False)
            ]
            if text_parts:
                return "\n".join(text_parts)

    # 平台提取的 message 字段
    message = inner.get("message")
    if message and isinstance(message, str):
        return message

    return None
