"""Pytest fixtures: TestClient and run_agent mock."""
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from src.main import app


@pytest.fixture
def client():
    """FastAPI TestClient."""
    return TestClient(app)


async def _mock_run_agent(*args, **kwargs):
    """Fake NDJSON events so adapter produces OpenAI-format output."""
    yield {
        "type": "assistant",
        "message": {"role": "assistant", "content": [{"type": "text", "text": "Hello"}]},
    }
    yield {"type": "result", "subtype": "success", "result": ""}


@pytest.fixture
def mock_run_agent():
    """Patch run_agent and check_agent_available so tests don't call real Cursor CLI."""
    with (
        patch("src.adapters.stream_adapter.run_agent", new=_mock_run_agent),
        patch("src.routes.openai.check_agent_available"),
    ):
        yield
