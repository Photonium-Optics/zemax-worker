"""Diagnostics router – connection diagnostics, logs."""

import os

from fastapi import APIRouter, Depends

import main
from config import WORKER_COUNT
from log_buffer import log_buffer
from diagnostics.connection_diagnostics import get_diagnostic_report

router = APIRouter()


@router.get("/diagnostics/connection")
async def diagnostics_connection(_: None = Depends(main.verify_api_key)):
    """Return connection diagnostic events for debugging license/connection issues."""
    return get_diagnostic_report()


@router.get("/logs")
async def get_logs(
    since: int = 0,
    limit: int = 200,
    _: None = Depends(main.verify_api_key),
):
    """Return recent log entries from this worker process's ring buffer.

    Query params:
        since: sequence cursor — only entries with sequence > since are returned
        limit: max entries to return (default 200, capped at 1000)
    """
    limit = min(max(limit, 1), 1000)
    entries, latest_sequence = log_buffer.get_entries(since_sequence=since, limit=limit)
    return {
        "entries": entries,
        "latest_sequence": latest_sequence,
        "worker_pid": os.getpid(),
        "worker_count": WORKER_COUNT,
    }
