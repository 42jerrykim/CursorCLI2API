"""OpenAI-compatible API routes."""
import asyncio
import json
import logging
import time
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

logger = logging.getLogger(__name__)

from src.adapters.stream_adapter import (
    OPENAI_MODEL_ID,
    run_completion,
    stream_completion,
)
from src.cursor_runner import check_agent_available

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

    try:
        check_agent_available()
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))

    try:
        if stream:
            prompt_preview = (messages[0].get("content") or "")[:80] if messages else ""
            logger.info("chat/completions stream request start prompt=%r", prompt_preview)
            return StreamingResponse(
                _sse_stream(messages, force=force),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )
        result = await run_completion(messages, force=force)
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


async def _sse_stream(messages: list[dict], force: bool | None = None):
    """Generate SSE lines from stream_completion chunks. Sends SSE keepalive comments
    while waiting for the agent so clients with a per-read timeout don't disconnect."""
    t0 = time.monotonic()
    logger.info("[stream] t=%.2fs sending empty-delta (stream start)", time.monotonic() - t0)
    yield f"data: {json.dumps(_empty_delta_chunk(), ensure_ascii=False)}\n\n"
    queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()

    async def consume():
        try:
            async for chunk in stream_completion(messages, force=force):
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
            logger.info("[stream] t=%.2fs keepalive (no agent output yet)", elapsed)
            yield ": keepalive\n\n"
            continue
        if kind == "done":
            break
        if kind == "error":
            elapsed = time.monotonic() - t0
            logger.warning("[stream] t=%.2fs error: %s", elapsed, payload)
            yield f"data: {json.dumps({'error': {'message': str(payload)}, 'object': 'error'}, ensure_ascii=False)}\n\n"
            return
        chunk_count += 1
        delta_len = len((payload.get("choices") or [{}])[0].get("delta", {}).get("content") or "")
        elapsed = time.monotonic() - t0
        logger.info("[stream] t=%.2fs chunk #%d delta_len=%d", elapsed, chunk_count, delta_len)
        line = json.dumps(payload, ensure_ascii=False)
        yield f"data: {line}\n\n"
    elapsed = time.monotonic() - t0
    logger.info("[stream] t=%.2fs done total_chunks=%d", elapsed, chunk_count)
    yield "data: [DONE]\n\n"
