"""Run Cursor CLI agent as subprocess and parse NDJSON stdout."""
import asyncio
import json
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import AsyncIterator

from src.utils.log_helpers import truncate_content_for_log

logger = logging.getLogger(__name__)

# Optional config (avoid circular import by lazy use in run_agent)
def _get_config():
    import config as cfg
    return cfg


def _resolve_agent_path(cmd: str) -> str:
    """Resolve agent command to an existing executable path. Prefer .exe on Windows."""
    if not cmd:
        return cmd
    p = Path(cmd)
    if p.is_file():
        return str(p.resolve())
    if sys.platform == "win32" and not (cmd.lower().endswith(".exe") or cmd.lower().endswith(".cmd")):
        exe_path = Path(cmd + ".exe")
        if exe_path.is_file():
            return str(exe_path.resolve())
        cmd_path = Path(cmd + ".cmd")
        if cmd_path.is_file():
            return str(cmd_path.resolve())
    found = shutil.which(cmd)
    if found:
        return found
    return cmd


def _content_preview_from_obj(obj: dict) -> str:
    """Extract a single string from Cursor NDJSON event for log preview."""
    ev_type = obj.get("type", "")
    if ev_type == "assistant":
        msg = obj.get("message") or {}
        parts = []
        for part in msg.get("content") or []:
            if part.get("type") == "text" and "text" in part:
                parts.append(part["text"] or "")
        return "".join(parts)
    if ev_type == "result":
        return obj.get("result") or ""
    if isinstance(obj.get("text"), str):
        return obj["text"]
    if isinstance(obj.get("content"), str):
        return obj["content"]
    if isinstance(obj.get("content"), list):
        for item in obj["content"]:
            if isinstance(item, dict) and item.get("type") == "text" and "text" in item:
                return item["text"] or ""
    return ""


def check_agent_available() -> None:
    """Raise RuntimeError if the configured agent command is not found."""
    cfg = _get_config()
    cmd = cfg.AGENT_CMD
    path = _resolve_agent_path(cmd)
    if path != cmd and Path(path).is_file():
        return
    if shutil.which(cmd) is not None:
        return
    if Path(cmd).is_file():
        return
    if sys.platform == "win32":
        if Path(cmd + ".exe").is_file() or Path(cmd + ".cmd").is_file():
            return
    raise RuntimeError(
        f'Cursor agent "{cmd}" not found. '
        "Install Cursor CLI (https://cursor.com/docs/cli/installation) or set "
        "CURSOR_AGENT_CMD to the full path of the executable (e.g. ...\\agent.exe on Windows)."
    )


async def run_agent(
    prompt: str,
    *,
    request_id: str | None = None,
    prompt_preview: str = "",
    stream_partial: bool = True,
    force: bool | None = None,
    cwd: str | None = None,
    timeout: float | None = None,
) -> AsyncIterator[dict]:
    """
    Run `agent -p --output-format stream-json [--stream-partial-output] "<prompt>"`
    and yield each NDJSON line as a parsed dict.
    """
    req_id = request_id or ""
    cfg = _get_config()
    cmd = _resolve_agent_path(cfg.AGENT_CMD)
    force = force if force is not None else cfg.AGENT_FORCE
    cwd = cwd or cfg.AGENT_CWD
    timeout = timeout if timeout is not None else cfg.REQUEST_TIMEOUT

    args = [
        cmd,
        "-p",
        "--output-format", "stream-json",
    ]
    if stream_partial:
        args.append("--stream-partial-output")
    if force:
        args.append("--force")
    args.append(prompt)
    # When cwd is None, subprocess uses the server's CWD (often the project dir). Cursor agent
    # may then index the project, which can add 30+ seconds before the first "thinking" event.
    run_cwd = cwd or os.getcwd()
    t0 = asyncio.get_event_loop().time()
    logger.info(
        "[cursor_runner] req_id=%s spawning agent cwd=%s prompt_preview=%s",
        req_id,
        run_cwd,
        prompt_preview,
    )

    try:
        proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=run_cwd,
        )
        logger.info("[cursor_runner] req_id=%s subprocess started pid=%s", req_id, proc.pid)
    except FileNotFoundError:
        raise RuntimeError(
            f'Cursor agent "{cmd}" not found. '
            "Install Cursor CLI or set CURSOR_AGENT_CMD to the full path of the executable."
        )

    async def read_stderr() -> str:
        data = await proc.stderr.read()
        return data.decode("utf-8", errors="replace").strip()

    line_count = 0
    try:
        if proc.stdout is None:
            err = await read_stderr()
            raise RuntimeError(f"agent failed: no stdout. stderr: {err}")

        buffer = b""
        first_chunk = True
        while True:
            try:
                if timeout is not None:
                    chunk = await asyncio.wait_for(proc.stdout.read(8192), timeout=timeout)
                else:
                    chunk = await proc.stdout.read(8192)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                raise TimeoutError("Cursor agent request timed out")

            if not chunk:
                break
            if first_chunk:
                elapsed = asyncio.get_event_loop().time() - t0
                logger.info("[cursor_runner] req_id=%s t=%.2fs first stdout chunk received len=%d", req_id, elapsed, len(chunk))
                first_chunk = False
            buffer += chunk
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                line = line.strip()
                if not line:
                    continue
                line_count += 1
                try:
                    obj = json.loads(line.decode("utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError) as e:
                    continue  # skip malformed lines
                if line_count <= 2 or obj.get("type") in ("thinking", "assistant", "result"):
                    cfg = _get_config()
                    max_len = getattr(cfg, "LOG_CONTENT_MAX_LEN", 0)
                    preview = truncate_content_for_log(_content_preview_from_obj(obj), max_len) if max_len > 0 else ""
                    if preview:
                        logger.info("[cursor_runner] req_id=%s t=%.2fs line #%d type=%s content_preview=%s", req_id, asyncio.get_event_loop().time() - t0, line_count, obj.get("type", ""), preview)
                    else:
                        logger.info("[cursor_runner] req_id=%s t=%.2fs line #%d type=%s", req_id, asyncio.get_event_loop().time() - t0, line_count, obj.get("type", ""))
                yield obj

        # drain remainder
        if buffer.strip():
            try:
                yield json.loads(buffer.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass

    finally:
        if proc.returncode is None:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            await proc.wait()

    elapsed = asyncio.get_event_loop().time() - t0
    logger.info("[cursor_runner] req_id=%s t=%.2fs process exited returncode=%s lines_parsed=%d", req_id, elapsed, proc.returncode, line_count)
    if proc.returncode != 0:
        err = await read_stderr()
        raise RuntimeError(f"agent exited with code {proc.returncode}. stderr: {err}")
