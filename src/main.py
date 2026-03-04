"""FastAPI app: OpenAI-compatible API over Cursor CLI."""
import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI

# Import config so env is loaded when app starts
import config  # noqa: F401

log_level = getattr(logging, config.LOG_LEVEL, logging.INFO)
use_color = (
    False
    if config.LOG_COLOR in ("0", "false", "no")
    else True
    if config.LOG_COLOR in ("1", "true", "yes")
    else sys.stdout.isatty()
)
if use_color:
    from src.utils.colored_log_formatter import ColoredLogFormatter

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        ColoredLogFormatter(use_color=True, fmt="%(levelname)s %(name)s %(message)s")
    )
    logging.basicConfig(level=log_level, handlers=[handler])
else:
    logging.basicConfig(level=log_level, format="%(levelname)s %(name)s %(message)s")
logging.getLogger("src").setLevel(log_level)

from fastapi.middleware.cors import CORSMiddleware

from src.routes.openai import router as openai_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(
    title="Cursor CLI to OpenAI API",
    description="Expose Cursor headless CLI as an OpenAI-compatible chat completions API (local only).",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(openai_router)


@app.get("/")
async def root():
    return {
        "service": "Cursor CLI to OpenAI API",
        "docs": "/docs",
        "openai_v1": "/v1",
        "models": "/v1/models",
        "chat_completions": "/v1/chat/completions",
    }


def run():
    """Run the server (for python -m src.main)."""
    import uvicorn
    import config as cfg
    uvicorn.run(
        "src.main:app",
        host=cfg.HOST,
        port=cfg.PORT,
        reload=False,
    )


if __name__ == "__main__":
    run()
