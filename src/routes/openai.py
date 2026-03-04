"""OpenAI-compatible API routes."""
import asyncio
import json
import logging
import time
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

import config
from src.adapters.stream_adapter import (
    OPENAI_MODEL_ID,
    run_completion,
    stream_completion,
)
from src.cursor_runner import check_agent_available
from src.utils.log_helpers import truncate_content_for_log

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["openai"])


# Advertised context/output limit so clients use this instead of defaulting to 4096
CURSOR_AGENT_MAX_TOKENS = 200_000


@router.get("/models")
async def list_models() -> dict:
    """Return list of available models (OpenAI-compatible)."""
    return {
        "object": "list",
        "data": [
            {
                "id": OPENAI_MODEL_ID,
                "object": "model",
                "created": 0,
                "owned_by": "cursor-cli",
                "max_tokens": CURSOR_AGENT_MAX_TOKENS,
                "context_length": CURSOR_AGENT_MAX_TOKENS,
                "context_window": CURSOR_AGENT_MAX_TOKENS,
            }
        ],
    }


def _parse_messages(body: dict) -> list[dict]:
    messages = body.get("messages")
    if not messages or not isinstance(messages, list):
        raise HTTPException(status_code=400, detail="messages is required and must be a non-empty array")
    return messages


PROMPT_PREVIEW_LEN = 80


def _prompt_preview_from_messages(messages: list[dict]) -> str:
    """First user-visible content from messages, single line, truncated for logs."""
    if not messages:
        return ""
    first = messages[0]
    content = first.get("content")
    if content is None:
        return ""
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text") or "")
        text = " ".join(parts)
    else:
        text = str(content)
    return truncate_content_for_log(text.strip(), PROMPT_PREVIEW_LEN)


@router.post("/chat/completions")
async def create_chat_completion(body: dict) -> Any:
    """
    OpenAI-compatible chat completions.
    Supports stream=True (SSE) and stream=False (single JSON).
    """
    messages = _parse_messages(body)
    stream = body.get("stream", False)
    model = body.get("model") or OPENAI_MODEL_ID  # ignore model for now, use cursor-agent

    # Optional: pass force from body if we add support later
    force = body.get("cursor_force") if "cursor_force" in body else None
    # Optional: include Cursor thinking in content (default from config, typically True)
    include_thinking = body.get("include_thinking") if "include_thinking" in body else None
    if include_thinking is not None and not isinstance(include_thinking, bool):
        include_thinking = bool(include_thinking)

    try:
        check_agent_available()
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))

    try:
        request_id = uuid.uuid4().hex[:8]
        prompt_preview = _prompt_preview_from_messages(messages)
        if stream:
            logger.info("req_id=%s stream start prompt_preview=%s", request_id, prompt_preview)
            return StreamingResponse(
                _sse_stream(messages, request_id=request_id, prompt_preview=prompt_preview, force=force, include_thinking=include_thinking),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )
        logger.info("req_id=%s chat/completions request start prompt_preview=%s", request_id, prompt_preview)
        result = await run_completion(messages, request_id=request_id, prompt_preview=prompt_preview, force=force, include_thinking=include_thinking)
        return result
    except TimeoutError as e:
        raise HTTPException(status_code=504, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))


def _empty_delta_chunk():
    """OpenAI-format chunk with empty delta (for stream start)."""
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": OPENAI_MODEL_ID,
        "choices": [{"index": 0, "delta": {}, "finish_reason": None}],
    }


SSE_KEEPALIVE_INTERVAL = 20.0  # seconds; send comment so client read timeout doesn't fire


async def _sse_stream(
    messages: list[dict],
    *,
    request_id: str,
    prompt_preview: str = "",
    force: bool | None = None,
    include_thinking: bool | None = None,
):
    """Generate SSE lines from stream_completion chunks. Sends SSE keepalive comments
    while waiting for the agent so clients with a per-read timeout don't disconnect."""
    t0 = time.monotonic()
    logger.info("[stream] req_id=%s t=%.2fs sending empty-delta (stream start)", request_id, time.monotonic() - t0)
    yield f"data: {json.dumps(_empty_delta_chunk(), ensure_ascii=False)}\n\n"
    queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()

    async def consume():
        try:
            async for chunk in stream_completion(
                messages, request_id=request_id, prompt_preview=prompt_preview, force=force, include_thinking=include_thinking
            ):
                await queue.put(("chunk", chunk))
            await queue.put(("done", None))
        except (RuntimeError, TimeoutError) as e:
            await queue.put(("error", e))

    asyncio.create_task(consume())
    chunk_count = 0
    while True:
        try:
            kind, payload = await asyncio.wait_for(queue.get(), timeout=SSE_KEEPALIVE_INTERVAL)
        except asyncio.TimeoutError:
            elapsed = time.monotonic() - t0
            logger.info("[stream] req_id=%s t=%.2fs keepalive (no agent output yet)", request_id, elapsed)
            yield ": keepalive\n\n"
            continue
        if kind == "done":
            break
        if kind == "error":
            elapsed = time.monotonic() - t0
            logger.warning("[stream] req_id=%s t=%.2fs error: %s", request_id, elapsed, payload)
            yield f"data: {json.dumps({'error': {'message': str(payload)}, 'object': 'error'}, ensure_ascii=False)}\n\n"
            return
        chunk_count += 1
        delta_content = (payload.get("choices") or [{}])[0].get("delta", {}).get("content") or ""
        delta_len = len(delta_content)
        elapsed = time.monotonic() - t0
        content_preview = truncate_content_for_log(delta_content, config.LOG_CONTENT_MAX_LEN) if config.LOG_CONTENT_MAX_LEN > 0 else ""
        if content_preview:
            logger.debug("[stream] req_id=%s t=%.2fs chunk #%d delta_len=%d content=%s", request_id, elapsed, chunk_count, delta_len, content_preview)
        else:
            logger.debug("[stream] req_id=%s t=%.2fs chunk #%d delta_len=%d", request_id, elapsed, chunk_count, delta_len)
        line = json.dumps(payload, ensure_ascii=False)
        yield f"data: {line}\n\n"
    elapsed = time.monotonic() - t0
    logger.info("[stream] req_id=%s t=%.2fs done total_chunks=%d", request_id, elapsed, chunk_count)
    yield "data: [DONE]\n\n"
