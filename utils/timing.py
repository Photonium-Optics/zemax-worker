"""
Timing utilities for profiling Zemax operations.

All timing logs use the [TIMING] prefix for easy grep filtering:
    grep "\\[TIMING\\]" service.log

Shared between zemax-analysis-service and zemax-worker.
"""

import asyncio
import time
import logging
from contextlib import contextmanager, asynccontextmanager
from typing import Generator, AsyncGenerator


@contextmanager
def timed_operation(
    logger: logging.Logger, operation: str, level: str = "info"
) -> Generator[None, None, None]:
    """
    Context manager that logs operation timing with success/failure distinction.

    Usage:
        with timed_operation(logger, "/cross-section"):
            # ... operation code ...

    Logs on success:
        [TIMING] /cross-section START
        [TIMING] /cross-section COMPLETE: 1234.5ms

    Logs on exception:
        [TIMING] /cross-section START
        [TIMING] /cross-section FAILED: 1234.5ms
    """
    start = time.perf_counter()
    log_fn = getattr(logger, level, logger.info)
    log_fn(f"[TIMING] {operation} START")
    success = True
    try:
        yield
    except Exception:
        success = False
        raise
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000
        status = "COMPLETE" if success else "FAILED"
        log_fn(f"[TIMING] {operation} {status}: {elapsed_ms:.1f}ms")


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


@asynccontextmanager
async def timed_lock_acquire(
    lock: asyncio.Lock, logger: logging.Logger, name: str = "lock"
) -> AsyncGenerator[None, None]:
    """
    Async context manager that acquires a lock and logs the wait time.

    Only the time spent waiting to acquire the lock is measured,
    not the time spent holding it.

    Args:
        lock: The asyncio.Lock to acquire
        logger: Logger instance for timing output
        name: Name of the lock for log messages (default: "lock")

    Usage:
        async with timed_lock_acquire(_zospy_lock, logger, name="zospy"):
            # ... code that needs the lock ...

    Logs:
        [TIMING] zospy_lock_wait: 12.3ms
    """
    start = time.perf_counter()
    await lock.acquire()
    elapsed_ms = (time.perf_counter() - start) * 1000
    log_timing(logger, f"{name}_lock_wait", elapsed_ms)
    try:
        yield
    finally:
        lock.release()
