"""
Connection diagnostics for tracking OpticStudio license/connection failures.

Captures the full context of every connection failure and reconnect attempt
so we can diagnose the root cause of intermittent "Licence is not valid" errors.

Usage:
    Import and call the diagnostic functions from main.py at key points.
    View captured data via GET /diagnostics/connection endpoint.
"""

import json
import logging
import os
import time
import traceback
from collections import deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Ring buffer of events — keeps last 200
_events: deque[dict[str, Any]] = deque(maxlen=200)
_start_time = time.monotonic()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _uptime_s() -> float:
    return round(time.monotonic() - _start_time, 2)


def record_event(event_type: str, **kwargs: Any) -> None:
    """Record a diagnostic event with timestamp and uptime."""
    entry = {
        "time": _now_iso(),
        "uptime_s": _uptime_s(),
        "pid": os.getpid(),
        "event": event_type,
        **kwargs,
    }
    _events.append(entry)
    # Also log it for immediate visibility in worker logs
    logger.info(f"[DIAG] {event_type}: {json.dumps({k: v for k, v in kwargs.items() if k != 'traceback'}, default=str)}")


def record_connect_attempt(mode: str) -> None:
    """Record that we're about to call ZOS().connect()."""
    record_event("connect_attempt", mode=mode)


def record_connect_success(version: str) -> None:
    """Record successful connection."""
    record_event("connect_success", version=version)


def record_connect_failure(error: str, tb: str) -> None:
    """Record failed connection attempt."""
    record_event("connect_failure", error=error, traceback=tb)


def record_disconnect(reason: str) -> None:
    """Record intentional disconnect (close() call)."""
    record_event("disconnect", reason=reason)


def record_operation_start(endpoint: str) -> None:
    """Record the start of an analysis operation."""
    record_event("op_start", endpoint=endpoint)


def record_operation_success(endpoint: str, duration_ms: float) -> None:
    """Record successful operation completion."""
    record_event("op_success", endpoint=endpoint, duration_ms=round(duration_ms, 1))


def record_operation_error(endpoint: str, error: str, error_type: str, tb: str, duration_ms: float) -> None:
    """Record operation failure with full context."""
    record_event(
        "op_error",
        endpoint=endpoint,
        error=error,
        error_type=error_type,
        duration_ms=round(duration_ms, 1),
        traceback=tb,
    )


def record_reconnect_triggered(reason: str, reconnect_failures: int, backoff_s: float) -> None:
    """Record that a reconnect was triggered."""
    record_event(
        "reconnect_triggered",
        reason=reason,
        reconnect_failures=reconnect_failures,
        backoff_s=round(backoff_s, 1),
    )


def record_reconnect_skipped_backoff(remaining_s: float, attempt: int) -> None:
    """Record that reconnect was skipped due to backoff."""
    record_event(
        "reconnect_skipped_backoff",
        remaining_s=round(remaining_s, 1),
        attempt=attempt,
    )


def record_license_seat_info() -> dict[str, Any]:
    """
    Try to gather info about OpticStudio license seat usage.

    This is best-effort — depends on what's available on the system.
    Returns the info dict for logging.
    """
    import platform
    info: dict[str, Any] = {
        "platform": platform.system(),
        "web_concurrency": os.getenv("WEB_CONCURRENCY", "NOT_SET"),
    }

    is_windows = platform.system() == "Windows"

    # Check for OpticStudio processes
    if is_windows:
        try:
            import subprocess
            result = subprocess.run(
                ["tasklist", "/FI", "IMAGENAME eq OpticStudio.exe", "/FO", "CSV", "/NH"],
                capture_output=True, text=True, timeout=5,
            )
            lines = [l.strip() for l in result.stdout.strip().split("\n") if l.strip() and "OpticStudio" in l]
            info["opticstudio_processes"] = len(lines)
            info["opticstudio_pids"] = [l.split(",")[1].strip('"') for l in lines] if lines else []
        except Exception as e:
            info["opticstudio_process_check_error"] = str(e)

        # Check for ZOSAPI processes
        try:
            import subprocess
            result = subprocess.run(
                ["tasklist", "/FI", "IMAGENAME eq ZOSAPI.exe", "/FO", "CSV", "/NH"],
                capture_output=True, text=True, timeout=5,
            )
            lines = [l.strip() for l in result.stdout.strip().split("\n") if l.strip() and "ZOSAPI" in l]
            info["zosapi_processes"] = len(lines)
        except Exception:
            pass

        # Count ALL python processes (each uvicorn worker is a python process)
        try:
            import subprocess
            result = subprocess.run(
                ["tasklist", "/FI", "IMAGENAME eq python.exe", "/FO", "CSV", "/NH"],
                capture_output=True, text=True, timeout=5,
            )
            lines = [l.strip() for l in result.stdout.strip().split("\n") if l.strip() and "python" in l.lower()]
            info["python_processes"] = len(lines)
        except Exception:
            pass
    else:
        info["note"] = "Non-Windows platform — process checks skipped"

    # Check memory usage of this process
    try:
        import psutil
        proc = psutil.Process(os.getpid())
        mem = proc.memory_info()
        info["worker_rss_mb"] = round(mem.rss / 1024 / 1024, 1)
        info["worker_vms_mb"] = round(mem.vms / 1024 / 1024, 1)
        # System memory
        vm = psutil.virtual_memory()
        info["system_memory_percent"] = vm.percent
        info["system_memory_available_mb"] = round(vm.available / 1024 / 1024, 1)
    except ImportError:
        info["psutil_note"] = "psutil not installed — install for memory diagnostics"
    except Exception as e:
        info["memory_check_error"] = str(e)

    record_event("license_seat_info", **info)
    return info


def get_diagnostic_report() -> dict[str, Any]:
    """Get the full diagnostic report."""
    return {
        "pid": os.getpid(),
        "uptime_s": _uptime_s(),
        "total_events": len(_events),
        "events": list(_events),
        # Summary counts
        "summary": _compute_summary(),
    }


def _compute_summary() -> dict[str, Any]:
    """Compute summary statistics from events."""
    counts: dict[str, int] = {}
    last_connect_failure: Optional[dict] = None
    last_op_error: Optional[dict] = None

    for ev in _events:
        event_type = ev.get("event", "unknown")
        counts[event_type] = counts.get(event_type, 0) + 1
        if event_type == "connect_failure":
            last_connect_failure = ev
        if event_type == "op_error":
            last_op_error = ev

    return {
        "event_counts": counts,
        "last_connect_failure": last_connect_failure,
        "last_op_error": last_op_error,
    }
