"""
Timing utilities for profiling Zemax worker operations.

All timing logs use the [TIMING] prefix for easy grep filtering:
    grep "\[TIMING\]" zemax-worker.log
"""

import time
import logging
from contextlib import contextmanager
from typing import Generator


@contextmanager
def timed_operation(
    logger: logging.Logger, operation: str, level: str = "info"
) -> Generator[None, None, None]:
    """
    Context manager that logs operation timing.

    Usage:
        with timed_operation(logger, "/cross-section"):
            # ... operation code ...

    Logs:
        [TIMING] /cross-section START
        [TIMING] /cross-section COMPLETE: 1234.5ms
    """
    start = time.perf_counter()
    log_fn = getattr(logger, level)
    log_fn(f"[TIMING] {operation} START")
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000
        log_fn(f"[TIMING] {operation} COMPLETE: {elapsed_ms:.1f}ms")


def log_timing(logger: logging.Logger, operation: str, elapsed_ms: float) -> None:
    """
    Log a timing measurement.

    Usage:
        start = time.perf_counter()
        # ... operation ...
        elapsed_ms = (time.perf_counter() - start) * 1000
        log_timing(logger, "CrossSection.run", elapsed_ms)

    Logs:
        [TIMING] CrossSection.run: 1234.5ms
    """
    logger.info(f"[TIMING] {operation}: {elapsed_ms:.1f}ms")
