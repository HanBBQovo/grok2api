"""
Utilities for approximate OpenAI-compatible usage accounting.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any, Sequence


_TOKEN_RE = re.compile(
    r"[A-Za-z0-9_]+|[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]|[^\s]",
    flags=re.UNICODE,
)
_PROMPT_OVERHEAD_TOKENS = 4
_COMPLETION_OVERHEAD_TOKENS = 2
_IMAGE_ATTACHMENT_TOKENS = 256
_FILE_ATTACHMENT_TOKENS = 128


def empty_chat_usage() -> dict[str, Any]:
    """Return an OpenAI-compatible empty usage payload."""
    return {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "prompt_tokens_details": {
            "cached_tokens": 0,
            "text_tokens": 0,
            "audio_tokens": 0,
            "image_tokens": 0,
        },
        "completion_tokens_details": {
            "text_tokens": 0,
            "audio_tokens": 0,
            "reasoning_tokens": 0,
        },
        "input_tokens_details": {
            "text_tokens": 0,
            "image_tokens": 0,
        },
    }


def estimate_text_tokens(text: str) -> int:
    """Estimate token count from plain text without external tokenizers."""
    if not text:
        return 0

    total = 0
    for part in _TOKEN_RE.findall(text):
        if part.isascii() and (part[0].isalnum() or part[0] == "_"):
            total += max(1, math.ceil(len(part) / 4))
        else:
            total += 1

    # Newlines often split into extra tokens in chat payloads.
    total += text.count("\n") // 4
    return max(total, 1)


def estimate_structured_tokens(payload: Any) -> int:
    """Estimate token count for structured payloads such as tool calls."""
    if payload in (None, "", [], {}):
        return 0

    try:
        serialized = json.dumps(
            payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True
        )
    except TypeError:
        serialized = str(payload)
    return estimate_text_tokens(serialized)


def estimate_chat_usage(
    *,
    prompt_text: str = "",
    completion_text: str = "",
    prompt_image_count: int = 0,
    prompt_file_count: int = 0,
    completion_tool_calls: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Build a non-zero, OpenAI-compatible usage payload for chat responses."""
    usage = empty_chat_usage()

    prompt_text_tokens = estimate_text_tokens(prompt_text)
    prompt_image_tokens = max(0, int(prompt_image_count or 0)) * _IMAGE_ATTACHMENT_TOKENS
    prompt_file_tokens = max(0, int(prompt_file_count or 0)) * _FILE_ATTACHMENT_TOKENS

    completion_text_tokens = estimate_text_tokens(completion_text)
    completion_tool_tokens = estimate_structured_tokens(completion_tool_calls)

    prompt_tokens = prompt_text_tokens + prompt_image_tokens + prompt_file_tokens
    completion_tokens = completion_text_tokens + completion_tool_tokens

    if prompt_tokens:
        prompt_tokens += _PROMPT_OVERHEAD_TOKENS
    if completion_tokens:
        completion_tokens += _COMPLETION_OVERHEAD_TOKENS

    total_tokens = prompt_tokens + completion_tokens

    usage["prompt_tokens"] = prompt_tokens
    usage["completion_tokens"] = completion_tokens
    usage["total_tokens"] = total_tokens
    usage["input_tokens"] = prompt_tokens
    usage["output_tokens"] = completion_tokens
    usage["prompt_tokens_details"] = {
        "cached_tokens": 0,
        "text_tokens": prompt_text_tokens + prompt_file_tokens,
        "audio_tokens": 0,
        "image_tokens": prompt_image_tokens,
    }
    usage["completion_tokens_details"] = {
        "text_tokens": completion_text_tokens + completion_tool_tokens,
        "audio_tokens": 0,
        "reasoning_tokens": 0,
    }
    usage["input_tokens_details"] = {
        "text_tokens": prompt_text_tokens + prompt_file_tokens,
        "image_tokens": prompt_image_tokens,
    }
    return usage


__all__ = [
    "empty_chat_usage",
    "estimate_chat_usage",
    "estimate_structured_tokens",
    "estimate_text_tokens",
]
