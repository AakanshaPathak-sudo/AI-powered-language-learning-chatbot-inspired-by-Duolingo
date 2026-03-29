"""Persist chat turns to append-only JSON Lines (JSONL) for auditing and analysis."""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from app.config import Settings

logger = logging.getLogger(__name__)

# Serialize concurrent writes from multiple requests in the same process.
_write_lock = threading.Lock()


def append_chat_record(settings: Settings, record: dict[str, Any]) -> None:
    """
    Append one JSON object as a single line to the configured log file.

    Each line is a standalone JSON object (JSONL), so files can grow safely
    without loading the full history. Disabled when chat_log_enabled is False.
    """
    if not settings.chat_log_enabled:
        return

    path = Path(settings.data_dir) / settings.chat_log_path
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, ensure_ascii=False, default=str) + "\n"
        with _write_lock:
            with open(path, "a", encoding="utf-8") as f:
                f.write(line)
    except OSError as e:
        # Do not fail the chat request if logging fails
        logger.warning("Could not write chat log to %s: %s", path, e)


def build_record(
    *,
    query: str,
    answer: Optional[str] = None,
    sources: Optional[list[dict[str, str]]] = None,
    error: Optional[str] = None,
    record_id: Optional[str] = None,
) -> dict[str, Any]:
    """Standard shape for one logged turn (success or failure)."""
    return {
        "id": record_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query": query,
        "answer": answer,
        "sources": sources or [],
        "error": error,
    }
