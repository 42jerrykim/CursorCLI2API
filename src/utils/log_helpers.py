"""Helpers for logging Cursor output content (truncated, single-line)."""
import re


def truncate_content_for_log(text: str, max_len: int) -> str:
    """
    Normalize and truncate a string for log output: single line, limited length.
    If max_len <= 0, returns empty string.
    """
    if not text or max_len <= 0:
        return ""
    # Collapse newlines and multiple spaces to a single space
    one_line = re.sub(r"\s+", " ", (text or "").strip())
    if len(one_line) <= max_len:
        return one_line
    return one_line[: max_len - 3].rstrip() + "..."
