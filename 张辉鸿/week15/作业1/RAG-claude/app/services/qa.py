"""Multi-modal QA using Qwen-VL via DashScope API."""

import base64
from pathlib import Path

import dashscope
from dashscope import MultiModalConversation

from app.config import settings

dashscope.api_key = settings.dashscope_api_key


def answer_question(
    question: str,
    text_contexts: list[dict],
    image_contexts: list[dict],
) -> str:
    """Generate an answer using Qwen-VL with retrieved text + image context.

    Args:
        question: User's question
        text_contexts: List of {text, doc_id, page_start, page_end, score}
        image_contexts: List of {image_path, caption, doc_id, page_num, score}

    Returns:
        Generated answer string
    """
    messages = _build_messages(question, text_contexts, image_contexts)

    resp = MultiModalConversation.call(
        model="qwen-vl-max",
        messages=messages,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"Qwen-VL QA failed: {resp.message}")

    return resp.output["choices"][0]["message"]["content"][0]["text"]


def _build_messages(
    question: str,
    text_contexts: list[dict],
    image_contexts: list[dict],
) -> list[dict]:
    """Build the multi-modal messages for Qwen-VL."""
    # System prompt
    system_content = [
        {"text": (
            "你是一个多模态文档问答助手。根据提供的文本和图像内容回答问题。\n"
            "要求：\n"
            "1. 回答要准确、简洁，基于提供的上下文内容。\n"
            "2. 如果上下文不足以回答问题，请明确说明。\n"
            "3. 在回答中注明信息来源，包括文件名和页码。\n"
            "4. 如果答案来自图像中的信息，请特别说明。"
        )}
    ]

    user_content = _build_user_content(question, text_contexts, image_contexts)

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]


def _build_user_content(
    question: str,
    text_contexts: list[dict],
    image_contexts: list[dict],
) -> list[dict]:
    """Build the user message with text and image content blocks."""
    content: list[dict] = []

    # Text contexts
    if text_contexts:
        parts = ["## 检索到的相关文本内容:\n"]
        for i, ctx in enumerate(text_contexts):
            parts.append(
                f"[文本来源{i + 1}] 页码: {ctx.get('page_start', '?')}-{ctx.get('page_end', '?')}, "
                f"相关度: {ctx['score']:.3f}\n"
                f"{ctx['text']}\n"
            )
        content.append({"text": "\n".join(parts)})

    # Image contexts
    image_count = 0
    for ctx in image_contexts:
        path = ctx["image_path"]
        if Path(path).exists():
            image_count += 1
            with open(path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode()
            content.append({
                "image": f"data:image/png;base64,{img_b64}"
            })
            content.append({
                "text": f"[图像来源{image_count}] 页码: {ctx.get('page_num', '?')}, "
                        f"描述: {ctx.get('caption', '无描述')}"
            })

    # Question
    content.append({
        "text": f"\n## 用户问题:\n{question}\n\n请根据以上提供的文本和图像内容回答问题，并注明信息来源。"
    })

    return content
