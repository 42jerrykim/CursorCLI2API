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

# Enable --force for agent (allow file modifications). Default True; set CURSOR_AGENT_FORCE=false to disable.
AGENT_FORCE: bool = os.getenv("CURSOR_AGENT_FORCE", "true").lower() not in ("0", "false", "no")

# Max length of Cursor output content to include in logs (0 = do not log content).
LOG_CONTENT_MAX_LEN: int = max(0, int(os.getenv("LOG_CONTENT_MAX_LEN", "200")))

# Log level: INFO (default) for flow-only logs; DEBUG for per-event/chunk/line details.
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()

# Colored log output in terminal. "auto" (default) = color when stdout is TTY; "1"/"0" to force.
LOG_COLOR: str = os.getenv("LOG_COLOR", "auto").lower()
