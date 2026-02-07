"""
In-memory ring buffer for log entries, exposed via a logging.Handler.

Each uvicorn worker process gets its own buffer (module-level singleton).
The /logs endpoint reads from it using cursor-based pagination so the Mac
debug dashboard can poll incrementally without duplicates or missed entries.
"""

import logging
import os
import threading
from collections import deque
from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class LogEntry:
    timestamp: float
    level: str
    logger_name: str
    message: str
    worker_pid: int
    sequence: int


class LogBuffer(logging.Handler):
    """Thread-safe ring buffer that captures log records."""

    def __init__(self, maxlen: int = 2000, level: int = logging.DEBUG) -> None:
        super().__init__(level)
        self._buf: deque[LogEntry] = deque(maxlen=maxlen)
        self._seq = 0
        self._lock = threading.Lock()
        self._pid = os.getpid()

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
        except Exception:
            msg = record.getMessage()

        with self._lock:
            self._seq += 1
            entry = LogEntry(
                timestamp=record.created,
                level=record.levelname,
                logger_name=record.name,
                message=msg,
                worker_pid=self._pid,
                sequence=self._seq,
            )
            self._buf.append(entry)

    def get_entries(
        self, since_sequence: int = 0, limit: int = 200
    ) -> tuple[list[dict[str, Any]], int]:
        """Return entries with sequence > since_sequence.

        Returns:
            (entries_as_dicts, latest_sequence)
        """
        with self._lock:
            latest = self._seq
            if since_sequence >= latest:
                return [], latest
            entries = [
                asdict(e) for e in self._buf if e.sequence > since_sequence
            ]
        # Keep only the most recent entries when the result exceeds the limit.
        if len(entries) > limit:
            entries = entries[-limit:]
        return entries, latest


# Module-level singleton — one per process
log_buffer = LogBuffer()
