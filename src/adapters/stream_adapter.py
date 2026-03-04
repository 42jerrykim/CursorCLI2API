"""Convert Cursor NDJSON events to OpenAI Chat Completions format."""
import json
import logging
import time
import uuid
from typing import Any, AsyncIterator

import config
from src.cursor_runner import run_agent
from src.utils.log_helpers import truncate_content_for_log

logger = logging.getLogger(__name__)


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
    request_id: str | None = None,
    prompt_preview: str = "",
    stream_partial: bool = True,
    force: bool | None = None,
    cwd: str | None = None,
    timeout: float | None = None,
    include_thinking: bool | None = None,
) -> AsyncIterator[dict]:
    """
    Run agent and yield OpenAI-format stream chunks (for SSE).
    Each yielded dict should be serialized as JSON and sent as SSE data.
    With stream_partial, each Cursor event is a delta; otherwise we send only new suffix.
    When include_thinking is True (default), thinking is sent first in delta.content
    as "<thinking>...</thinking>\\n\\n" then the assistant reply, so TUI shows both without code changes.
    """
    req_id = request_id or ""
    if include_thinking is None:
        include_thinking = getattr(config, "INCLUDE_THINKING", True)
    prompt = _messages_to_prompt(messages)
    t0 = time.monotonic()
    logger.info("[stream_adapter] req_id=%s stream_completion start prompt_len=%d prompt_preview=%s include_thinking=%s", req_id, len(prompt), prompt_preview, include_thinking)
    accumulated = ""
    event_count = 0
    yield_count = 0
    thinking_emitted = False

    agent_gen = run_agent(
        prompt,
        request_id=request_id,
        prompt_preview=prompt_preview,
        stream_partial=stream_partial,
        force=force,
        cwd=cwd,
        timeout=timeout,
    )
    try:
        async for event in agent_gen:
            event_count += 1
            ev_type = event.get("type", "")
            subtype = event.get("subtype", "")

            if event_count <= 3 or ev_type in ("assistant", "result", "thinking"):
                preview = ""
                if config.LOG_CONTENT_MAX_LEN > 0 and ev_type in ("assistant", "result"):
                    preview = truncate_content_for_log(_extract_assistant_text(event), config.LOG_CONTENT_MAX_LEN)
                elif ev_type == "thinking" and subtype == "delta":
                    preview = truncate_content_for_log(event.get("text") or "", config.LOG_CONTENT_MAX_LEN) if config.LOG_CONTENT_MAX_LEN > 0 else ""
                if preview:
                    logger.debug("[stream_adapter] req_id=%s t=%.2fs event #%d type=%s content_preview=%s", req_id, time.monotonic() - t0, event_count, ev_type, preview)
                else:
                    logger.debug("[stream_adapter] req_id=%s t=%.2fs event #%d type=%s", req_id, time.monotonic() - t0, event_count, ev_type)

            # Emit thinking into delta.content so TUI shows it without code changes
            if ev_type == "thinking" and include_thinking:
                if subtype == "delta":
                    if not thinking_emitted:
                        yield_count += 1
                        yield _openai_chunk("<thinking>", finish_reason=None)
                        thinking_emitted = True
                    text = event.get("text") or ""
                    if text:
                        yield_count += 1
                        yield _openai_chunk(text, finish_reason=None)
                elif subtype == "completed":
                    yield_count += 1
                    yield _openai_chunk("</thinking>\n\n", finish_reason=None)

            # Assistant/result: existing delta logic (reply follows after </thinking>)
            text = _extract_assistant_text(event)
            if text:
                # Always send only the new suffix (delta). Cursor may send multiple
                # events with the same or cumulative full text; we must not repeat.
                if text.startswith(accumulated):
                    delta = text[len(accumulated) :]
                elif accumulated and accumulated in text:
                    pos = text.find(accumulated)
                    delta = text[pos + len(accumulated) :]
                else:
                    delta = text
                accumulated = text
                if delta:
                    yield_count += 1
                    if config.LOG_CONTENT_MAX_LEN > 0:
                        content_preview = truncate_content_for_log(delta, config.LOG_CONTENT_MAX_LEN)
                        if content_preview:
                            logger.debug("[stream_adapter] req_id=%s t=%.2fs yield #%d delta_len=%d content=%s", req_id, time.monotonic() - t0, yield_count, len(delta), content_preview)
                        else:
                            logger.debug("[stream_adapter] req_id=%s t=%.2fs yield #%d delta_len=%d", req_id, time.monotonic() - t0, yield_count, len(delta))
                    else:
                        logger.debug("[stream_adapter] req_id=%s t=%.2fs yield #%d delta_len=%d", req_id, time.monotonic() - t0, yield_count, len(delta))
                    yield _openai_chunk(delta, finish_reason=None)
            if event.get("type") == "result" and event.get("subtype") == "success":
                logger.info("[stream_adapter] req_id=%s t=%.2fs result success total_events=%d total_yields=%d", req_id, time.monotonic() - t0, event_count, yield_count)
                yield _openai_chunk("", finish_reason="stop")
                break
    finally:
        await agent_gen.aclose()


def _extract_thinking_text(event: dict) -> str:
    """Extract thinking delta text from Cursor thinking event."""
    if event.get("type") == "thinking" and event.get("subtype") == "delta":
        return event.get("text") or ""
    return ""


async def run_completion(
    messages: list[dict],
    *,
    request_id: str | None = None,
    prompt_preview: str = "",
    stream_partial: bool = True,
    force: bool | None = None,
    cwd: str | None = None,
    timeout: float | None = None,
    include_thinking: bool | None = None,
) -> dict:
    """
    Run agent and return a single OpenAI-format ChatCompletion (non-streaming).
    When include_thinking is True (default), full content includes [Thinking]...\\n\\n[Reply]... reply.
    """
    if include_thinking is None:
        include_thinking = getattr(config, "INCLUDE_THINKING", True)
    prompt = _messages_to_prompt(messages)
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    accumulated_reply = ""
    thinking_parts: list[str] = []

    async for event in run_agent(
        prompt,
        request_id=request_id,
        prompt_preview=prompt_preview,
        stream_partial=stream_partial,
        force=force,
        cwd=cwd,
        timeout=timeout,
    ):
        ev_type = event.get("type", "")
        if ev_type == "thinking" and include_thinking and event.get("subtype") == "delta":
            thinking_parts.append(_extract_thinking_text(event))
        text = _extract_assistant_text(event)
        if text:
            if text.startswith(accumulated_reply):
                accumulated_reply = text
            elif accumulated_reply and accumulated_reply in text:
                accumulated_reply = text
            else:
                accumulated_reply = accumulated_reply + text

    if include_thinking and thinking_parts:
        full_content = "<thinking>" + "".join(thinking_parts) + "</thinking>\n\n" + accumulated_reply
    else:
        full_content = accumulated_reply
    return _openai_completion(full_content, completion_id)
