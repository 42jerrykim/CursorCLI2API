"""Server and Cursor CLI configuration."""
import os
from pathlib import Path

# Load .env if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Bind address: localhost only
HOST: str = os.getenv("HOST", "127.0.0.1")
PORT: int = int(os.getenv("PORT", "8080"))

# Cursor agent command (executable name or path)
AGENT_CMD: str = os.getenv("CURSOR_AGENT_CMD", "agent")

# Optional: absolute path to working directory for agent (default: current)
AGENT_CWD: str | None = os.getenv("CURSOR_AGENT_CWD") or None

# Request timeout in seconds (None = no timeout)
REQUEST_TIMEOUT: float | None = (
    float(t) if (t := os.getenv("REQUEST_TIMEOUT", "").strip()) else None
)

# Enable --force for agent (allow file modifications)
AGENT_FORCE: bool = os.getenv("CURSOR_AGENT_FORCE", "false").lower() in ("1", "true", "yes")
