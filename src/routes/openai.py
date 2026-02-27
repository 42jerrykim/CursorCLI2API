"""OpenAI-compatible API routes."""
import json
import time
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from src.adapters.stream_adapter import (
    OPENAI_MODEL_ID,
    run_completion,
    stream_completion,
)
from src.cursor_runner import check_agent_available

router = APIRouter(prefix="/v1", tags=["openai"])


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


async def _sse_stream(messages: list[dict], force: bool | None = None):
    """Generate SSE lines from stream_completion chunks."""
    # Send one empty-delta chunk immediately so the client sees the stream has started
    yield f"data: {json.dumps(_empty_delta_chunk(), ensure_ascii=False)}\n\n"
    try:
        async for chunk in stream_completion(messages, force=force):
            line = json.dumps(chunk, ensure_ascii=False)
            yield f"data: {line}\n\n"
        yield "data: [DONE]\n\n"
    except (RuntimeError, TimeoutError) as e:
        # Yield one error chunk so client gets a message before stream ends
        yield f"data: {json.dumps({'error': {'message': str(e)}, 'object': 'error'}, ensure_ascii=False)}\n\n"
