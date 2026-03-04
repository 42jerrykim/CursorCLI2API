"""ANSI-colored log formatter for terminal readability."""
import logging
import re
import sys


# ANSI escape codes (no color when not a TTY)
def _code(*values: int) -> str:
    return "\033[" + ";".join(str(v) for v in values) + "m" if sys.stdout.isatty() else ""


RESET = _code(0)
BOLD = _code(1)
DIM = _code(2)

# Level colors
LEVEL_DEBUG = _code(2, 37)   # dim white
LEVEL_INFO = _code(32)       # green
LEVEL_WARNING = _code(33)    # yellow
LEVEL_ERROR = _code(31)      # red

# Message part colors
TAG_STREAM = _code(36)           # cyan [stream]
TAG_STREAM_ADAPTER = _code(32)   # green [stream_adapter]
TAG_CURSOR_RUNNER = _code(35)    # magenta [cursor_runner]
REQ_ID = _code(1, 36)            # bold cyan req_id=xxx
PROMPT_PREVIEW = _code(33)       # yellow prompt_preview=...
KEY_T = _code(2, 37)             # dim t=12.34s
KEY_DONE = _code(1, 32)          # bold green done / result success
# Event type colors (cursor_runner stream-json)
TYPE_THINKING = _code(36)        # cyan type=thinking
TYPE_ASSISTANT = _code(32)        # green type=assistant
TYPE_RESULT = _code(1, 35)       # bold magenta type=result


def _color_level(levelno: int) -> str:
    if levelno >= logging.ERROR:
        return LEVEL_ERROR
    if levelno >= logging.WARNING:
        return LEVEL_WARNING
    if levelno >= logging.INFO:
        return LEVEL_INFO
    return LEVEL_DEBUG


def _color_message(message: str) -> str:
    if not sys.stdout.isatty():
        return message
    out = message
    # Tag highlights: [stream], [stream_adapter], [cursor_runner]
    out = re.sub(r"\[stream\]", f"{TAG_STREAM}[stream]{RESET}", out)
    out = re.sub(r"\[stream_adapter\]", f"{TAG_STREAM_ADAPTER}[stream_adapter]{RESET}", out)
    out = re.sub(r"\[cursor_runner\]", f"{TAG_CURSOR_RUNNER}[cursor_runner]{RESET}", out)
    # req_id=xxxxxxxx (8 hex chars)
    out = re.sub(r"(req_id=)([a-f0-9]{8})", rf"\1{REQ_ID}\2{RESET}", out)
    # prompt_preview=... (value until next " t=" or " total_" or end)
    out = re.sub(
        r'(prompt_preview=)([^"]*?)(?=\s+t=|\s+total_|$)',
        rf"\1{PROMPT_PREVIEW}\2{RESET}",
        out,
        count=1,
    )
    # t=12.34s -> dim
    out = re.sub(r"(t=)([\d.]+s)", rf"{KEY_T}\1\2{RESET}", out)
    # "done" and "result success" -> bold green
    out = re.sub(r"\b(done)\b", f"{KEY_DONE}\\1{RESET}", out)
    out = re.sub(r"\b(result success)\b", f"{KEY_DONE}\\1{RESET}", out)
    # type=thinking / type=assistant / type=result -> distinct colors
    out = re.sub(r"(type=)(thinking)\b", rf"\1{TYPE_THINKING}\2{RESET}", out)
    out = re.sub(r"(type=)(assistant)\b", rf"\1{TYPE_ASSISTANT}\2{RESET}", out)
    out = re.sub(r"(type=)(result)\b", rf"\1{TYPE_RESULT}\2{RESET}", out)
    return out


def colorize_line(line: str, levelno: int) -> str:
    """Apply colors to a full log line (levelname + message)."""
    if not sys.stdout.isatty():
        return line
    # Format is "LEVELNAME NAME MESSAGE" - levelname is first word (padded to 8), then name, then rest
    parts = line.split(None, 2)  # max 2 splits -> [levelname, name, message]
    if len(parts) < 3:
        return line
    levelname, name, message = parts
    level_color = _color_level(levelno)
    colored_level = f"{level_color}{levelname}{RESET}"
    colored_msg = _color_message(message)
    return f"{colored_level} {name} {colored_msg}"


class ColoredLogFormatter(logging.Formatter):
    """Formatter that adds ANSI colors for level, tags, req_id, and prompt_preview."""

    def __init__(self, use_color: bool = True, **kwargs) -> None:
        super().__init__(**kwargs)
        self._use_color = use_color and sys.stdout.isatty()

    def format(self, record: logging.LogRecord) -> str:
        line = super().format(record)
        if self._use_color:
            return colorize_line(line, record.levelno)
        return line
