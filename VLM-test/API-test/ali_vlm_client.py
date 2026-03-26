"""
阿里 DashScope OpenAI 兼容模式客户端。

使用 OpenAI SDK 通过 DashScope 兼容端点调用阿里系列模型（qwen 等）。
支持图文混合输入、多视角图片、ReAct 纠正轮。
包含指数退避重试。
"""

import logging
import time
from typing import Optional

from openai import OpenAI, APIError, APIConnectionError, RateLimitError, APITimeoutError

logger = logging.getLogger(__name__)


def make_client(base_url: str, api_key: str) -> OpenAI:
    """创建 DashScope OpenAI 兼容客户端实例。"""
    return OpenAI(base_url=base_url, api_key=api_key)


def build_prompt_messages(system_prompt: str, user_text: str, image_url: str) -> list:
    """
    构造 OpenAI vision 格式的消息列表（system + user with image）。

    图片通过 URL 传递（支持 OSS URL 或 HTTP URL）。
    """
    messages = []

    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    user_content = [
        {"type": "text", "text": user_text},
        {"type": "image_url", "image_url": {"url": image_url}},
    ]
    messages.append({"role": "user", "content": user_content})

    return messages


def build_multi_view_prompt_messages(
    system_prompt: str,
    user_text: str,
    image_urls: list,
) -> list:
    """
    构造多视角图片的消息列表。

    将多张图片 URL 和文本问题组合在同一条 user 消息中。
    """
    messages = []

    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    user_content = []
    for url in image_urls:
        user_content.append({
            "type": "image_url",
            "image_url": {"url": url},
        })
    user_content.append({"type": "text", "text": user_text})

    messages.append({"role": "user", "content": user_content})

    return messages


def build_correction_prompt(
    original_messages: list,
    assistant_response: str,
    correction_text: str,
) -> list:
    """
    构造 ReAct 纠正轮的消息：在原始消息后追加 assistant 回复和 user 纠正请求。
    """
    return original_messages + [
        {"role": "assistant", "content": assistant_response or ""},
        {"role": "user", "content": correction_text},
    ]


def call_ali_vlm(
    base_url: str,
    api_key: str,
    messages: list,
    model: str,
    temperature: float = 0.0,
    max_tokens: int = 65536,
    max_retries: int = 5,
    retry_base_delay: float = 2.0,
    timeout: int = 300,
) -> str:
    """
    通过 DashScope OpenAI 兼容端点调用阿里系列模型，带指数退避重试。

    返回模型生成的文本内容。
    """
    client = make_client(base_url, api_key)

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
            )

            content = response.choices[0].message.content
            if content:
                return content

            # content 为空，检查是否有 reasoning_content
            message = response.choices[0].message
            reasoning = getattr(message, "reasoning_content", None)
            if reasoning:
                logger.warning(
                    "content 为空但有 reasoning_content，尝试关闭思考模式重试"
                )
                try:
                    retry_response = client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        timeout=timeout,
                        extra_body={
                            "chat_template_kwargs": {"enable_thinking": False}
                        },
                    )
                    retry_content = retry_response.choices[0].message.content
                    if retry_content:
                        return retry_content
                except Exception as inner_error:
                    logger.warning(f"关闭思考模式重试失败: {inner_error}")

            logger.warning(
                f"重试 {attempt + 1}/{max_retries}，响应内容为空"
            )
            raise ValueError("VLM returned None content")

        except (RateLimitError, APITimeoutError, APIConnectionError, ValueError) as error:
            delay = min(retry_base_delay * (2 ** attempt), 60.0)
            logger.warning(
                f"重试 {attempt + 1}/{max_retries}，"
                f"{type(error).__name__}，等待 {delay:.1f}s"
            )
            time.sleep(delay)

        except APIError as error:
            if getattr(error, "status_code", None) and error.status_code >= 500:
                delay = min(retry_base_delay * (2 ** attempt), 60.0)
                logger.warning(
                    f"重试 {attempt + 1}/{max_retries}，"
                    f"服务器错误 {error.status_code}，等待 {delay:.1f}s"
                )
                time.sleep(delay)
            else:
                raise

    raise RuntimeError(f"调用失败，已重试 {max_retries} 次")