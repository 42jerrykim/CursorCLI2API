"""Convert Cursor NDJSON events to OpenAI Chat Completions format."""
import json
import time
import uuid
from typing import Any, AsyncIterator

from src.cursor_runner import run_agent


OPENAI_MODEL_ID = "cursor-agent"
CHAT_COMPLETION_CHUNK_OBJECT = "chat.completion.chunk"


def _extract_assistant_text(event: dict) -> str:
    """Extract assistant message text from Cursor assistant or result event."""
    if event.get("type") == "assistant":
        msg = event.get("message") or {}
        for part in msg.get("content") or []:
            if part.get("type") == "text" and "text" in part:
                return part["text"] or ""
    if event.get("type") == "result" and event.get("subtype") == "success":
        return event.get("result") or ""
    return ""


def _messages_to_prompt(messages: list[dict]) -> str:
    """Convert OpenAI messages to a single prompt string for the agent."""
    parts = []
    for m in messages:
        role = (m.get("role") or "user").strip().lower()
        content = m.get("content")
        if content is None:
            continue
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            texts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    texts.append(block.get("text") or "")
            text = "\n".join(texts)
        else:
            text = str(content)
        if role == "system":
            parts.append(f"System: {text}")
        elif role == "user":
            parts.append(f"User: {text}")
        elif role == "assistant":
            parts.append(f"Assistant: {text}")
        else:
            parts.append(f"{role}: {text}")
    return "\n\n".join(parts)


def _openai_chunk(delta_content: str, finish_reason: str | None = None) -> dict:
    """Build one OpenAI SSE chunk (chat.completion.chunk)."""
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": CHAT_COMPLETION_CHUNK_OBJECT,
        "created": int(time.time()),
        "model": OPENAI_MODEL_ID,
        "choices": [
            {
                "index": 0,
                "delta": {"content": delta_content} if delta_content else {},
                "finish_reason": finish_reason,
            }
        ],
    }


def _openai_completion(full_content: str, completion_id: str) -> dict:
    """Build full OpenAI ChatCompletion (non-streaming) response."""
    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": OPENAI_MODEL_ID,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": full_content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }


async def stream_completion(
    messages: list[dict],
    *,
    stream_partial: bool = True,
    force: bool | None = None,
    cwd: str | None = None,
    timeout: float | None = None,
) -> AsyncIterator[dict]:
    """
    Run agent and yield OpenAI-format stream chunks (for SSE).
    Each yielded dict should be serialized as JSON and sent as SSE data.
    With stream_partial, each Cursor event is a delta; otherwise we send only new suffix.
    """
    prompt = _messages_to_prompt(messages)
    accumulated = ""

    async for event in run_agent(
        prompt,
        stream_partial=stream_partial,
        force=force,
        cwd=cwd,
        timeout=timeout,
    ):
        text = _extract_assistant_text(event)
        if text:
            # Always send only the new suffix (delta). Cursor may send multiple
            # events with the same or cumulative full text; we must not repeat.
            if text.startswith(accumulated):
                delta = text[len(accumulated) :]
            elif accumulated and accumulated in text:
                # Same content possibly with prefix (e.g. from different event type) - send only suffix after accumulated
                pos = text.find(accumulated)
                delta = text[pos + len(accumulated) :]
            else:
                delta = text
            accumulated = text
            if delta:
                yield _openai_chunk(delta, finish_reason=None)
        if event.get("type") == "result" and event.get("subtype") == "success":
            yield _openai_chunk("", finish_reason="stop")


async def run_completion(
    messages: list[dict],
    *,
    stream_partial: bool = True,
    force: bool | None = None,
    cwd: str | None = None,
    timeout: float | None = None,
) -> dict:
    """
    Run agent and return a single OpenAI-format ChatCompletion (non-streaming).
    """
    prompt = _messages_to_prompt(messages)
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    accumulated = ""

    async for event in run_agent(
        prompt,
        stream_partial=stream_partial,
        force=force,
        cwd=cwd,
        timeout=timeout,
    ):
        text = _extract_assistant_text(event)
        if text:
            if text.startswith(accumulated):
                accumulated = text
            elif accumulated and accumulated in text:
                accumulated = text
            else:
                accumulated = accumulated + text

    full_content = accumulated
    return _openai_completion(full_content, completion_id)
