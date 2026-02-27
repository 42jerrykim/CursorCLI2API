"""FastAPI app: OpenAI-compatible API over Cursor CLI."""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.routes.openai import router as openai_router

# Import config so env is loaded when app starts
import config  # noqa: F401


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
