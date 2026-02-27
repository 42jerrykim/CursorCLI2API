"""Tests for OpenAI-compatible API routes."""
import json

import pytest


def test_get_models(client):
    """GET /v1/models returns 200 and list with cursor-agent."""
    r = client.get("/v1/models")
    assert r.status_code == 200
    data = r.json()
    assert data["object"] == "list"
    assert "data" in data
    assert len(data["data"]) >= 1
    ids = [m["id"] for m in data["data"]]
    assert "cursor-agent" in ids


def test_chat_completions_no_stream(client, mock_run_agent):
    """POST /v1/chat/completions with stream=false returns full message."""
    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "cursor-agent",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": False,
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert data["object"] == "chat.completion"
    assert "choices" in data
    assert len(data["choices"]) >= 1
    msg = data["choices"][0].get("message", {})
    assert msg.get("role") == "assistant"
    assert msg.get("content") == "Hello"
    assert data["choices"][0].get("finish_reason") == "stop"


def test_chat_completions_stream(client, mock_run_agent):
    """POST /v1/chat/completions with stream=true returns SSE with delta content and [DONE]."""
    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "cursor-agent",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": True,
        },
    )
    assert r.status_code == 200
    assert "text/event-stream" in r.headers.get("content-type", "")

    lines = r.text.strip().split("\n")
    data_lines = [ln for ln in lines if ln.startswith("data: ")]
    assert len(data_lines) >= 1

    saw_delta = False
    saw_done = False
    for line in data_lines:
        payload = line[6:]  # strip "data: "
        if payload == "[DONE]":
            saw_done = True
            continue
        try:
            chunk = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if chunk.get("object") == "chat.completion.chunk":
            choices = chunk.get("choices", [])
            if choices and choices[0].get("delta"):
                if choices[0]["delta"].get("content"):
                    saw_delta = True
                if choices[0].get("finish_reason") == "stop":
                    saw_done = True

    assert saw_delta, "Expected at least one chunk with delta.content"
    assert saw_done, "Expected [DONE] or finish_reason=stop"
