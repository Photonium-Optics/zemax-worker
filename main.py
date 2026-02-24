"""
Zemax Worker

FastAPI worker that runs on Windows machine with Zemax OpticStudio.
This is the "dumb executor" that receives commands from the Mac orchestrator
and executes ZosPy operations against OpticStudio.

Prerequisites:
- Windows 10/11
- Zemax OpticStudio (Professional or Premium license for API access)
- Python 3.9-3.11
- ZosPy >= 1.2.0

Parallelism:
Each uvicorn worker is a separate process with its own OpticStudio connection.
Multiple workers enable parallel request processing, but each consumes a license seat.

License limits (per Ansys):
- Professional (subscription): 4 instances
- Premium (subscription): 8 instances
- Perpetual (legacy 19.4+): 2 instances

Examples:
  python main.py --workers 5                     # Recommended: auto-sets WEB_CONCURRENCY
  WEB_CONCURRENCY=5 python -m uvicorn main:app   # Alternative: set env var explicitly

The Mac-side zemax-analysis-service auto-detects the worker count from /health.
"""

import sys
# When run as `python main.py`, this module's __name__ is "__main__", not "main".
# Router modules do `import main` to access globals at request time.  Without this
# alias, Python would re-import main.py as a *separate* "main" module, triggering
# a circular import crash.  This one-liner ensures both names resolve to the same
# module object.
sys.modules.setdefault("main", sys.modules[__name__])

import asyncio
import base64
import logging
import os
import tempfile
import time
from contextlib import asynccontextmanager
from typing import Any, Callable, Optional

from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from zospy_handler import ZosPyHandler, ZosPyError
from config import (
    NOT_CONNECTED_ERROR, DEFAULT_PORT, DEFAULT_HOST, ZEMAX_API_KEY,
    WORKER_COUNT, _RECONNECT_BACKOFF_BASE, _RECONNECT_BACKOFF_MAX,
    _RECONNECT_COM_RELEASE_DELAY,
)
from utils.timing import timed_operation, timed_lock_acquire
from log_buffer import log_buffer
from diagnostics.connection_diagnostics import (
    record_connect_attempt, record_connect_success, record_connect_failure,
    record_disconnect, record_operation_start, record_operation_success,
    record_operation_error, record_reconnect_triggered, record_reconnect_skipped_backoff,
    record_license_seat_info,
)
import traceback as _tb_mod

# Configure logging
logging.basicConfig(level=logging.INFO)
logging.getLogger().addHandler(log_buffer)
# Uvicorn sets propagate=False on its loggers, so they never reach the root
# logger (and thus never reach log_buffer). Attach log_buffer directly so the
# Mac debug dashboard sees access logs and server lifecycle messages too.
for _uv_name in ("uvicorn.error", "uvicorn.access"):
    logging.getLogger(_uv_name).addHandler(log_buffer)
# Enable DEBUG for raw Zemax output logger (root is INFO, so explicit level needed)
logging.getLogger("zemax.raw").setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)


# =============================================================================
# Global State
# =============================================================================

# Initialize ZosPy handler (manages OpticStudio connection)
zospy_handler: Optional[ZosPyHandler] = None
_last_connection_error: Optional[str] = None

# Reconnect backoff imported from config.py
_reconnect_failures = 0
_last_reconnect_attempt: float = 0.0

# Async lock - serializes ZosPy operations within this process
# Each uvicorn worker process has its own lock and OpticStudio connection
_zospy_lock = asyncio.Lock()


# =============================================================================
# Connection Management
# =============================================================================


def _init_zospy() -> Optional[ZosPyHandler]:
    """Initialize ZosPy connection with error handling."""
    global _last_connection_error
    record_connect_attempt(mode="standalone")
    record_license_seat_info()
    handler = None
    try:
        handler = ZosPyHandler()
        version = handler.get_version()
        logger.info("ZosPy connection established.")
        record_connect_success(version=version)
        _last_connection_error = None
        return handler
    except Exception as e:
        # Clean up partially-constructed handler to avoid orphaned OpticStudio processes
        if handler is not None:
            try:
                handler.close()
            except Exception as close_err:
                logger.warning(f"Failed to close handler during init cleanup: {close_err}")
        tb = _tb_mod.format_exc()
        logger.error(f"Failed to initialize ZosPy: {e}")
        record_connect_failure(error=str(e), tb=tb)
        _last_connection_error = str(e)
        return None


def _backoff_delay() -> float:
    """Compute the current exponential backoff delay in seconds."""
    return min(
        _RECONNECT_BACKOFF_BASE * (2 ** (_reconnect_failures - 1)),
        _RECONNECT_BACKOFF_MAX,
    )


async def _reconnect_zospy() -> Optional[ZosPyHandler]:
    """
    Attempt to reconnect to OpticStudio with exponential backoff.

    Caller MUST hold _zospy_lock.
    """
    global zospy_handler, _reconnect_failures, _last_reconnect_attempt

    now = time.monotonic()
    if _reconnect_failures > 0:
        backoff = _backoff_delay()
        elapsed = now - _last_reconnect_attempt
        if elapsed < backoff:
            remaining = backoff - elapsed
            logger.warning(
                f"Reconnect backoff: {remaining:.1f}s remaining "
                f"(attempt {_reconnect_failures})"
            )
            record_reconnect_skipped_backoff(remaining_s=remaining, attempt=_reconnect_failures)
            return None

    _last_reconnect_attempt = now

    if zospy_handler:
        record_disconnect(reason="reconnect_replacing_existing")
        try:
            zospy_handler.close()
        except Exception as e:
            logger.warning(f"Failed to close ZosPy handler during reconnect: {e}")
        zospy_handler = None

        # Wait for COM to release the license seat before reconnecting,
        # otherwise the new connect() sees it as still occupied.
        logger.info(
            f"Waiting {_RECONNECT_COM_RELEASE_DELAY}s for COM license release..."
        )
        await asyncio.sleep(_RECONNECT_COM_RELEASE_DELAY)
    else:
        record_reconnect_triggered(
            reason="zospy_handler_was_None",
            reconnect_failures=_reconnect_failures,
            backoff_s=_backoff_delay() if _reconnect_failures > 0 else 0,
        )

    logger.info("Attempting to reconnect to OpticStudio...")
    zospy_handler = _init_zospy()

    if zospy_handler is not None:
        _reconnect_failures = 0
    else:
        _reconnect_failures += 1
        logger.warning(
            f"Reconnect failed (attempt {_reconnect_failures}, "
            f"next backoff {_backoff_delay():.0f}s)"
        )

    return zospy_handler


async def _ensure_connected() -> Optional[ZosPyHandler]:
    """Ensure ZosPy is connected, attempting reconnection if needed. Caller MUST hold _zospy_lock."""
    if zospy_handler is None:
        await _reconnect_zospy()  # sets the global
    return zospy_handler


async def _handle_zospy_error(operation_name: str, error: Exception) -> None:
    """Handle errors from ZosPy operations by always reconnecting.

    Previous versions conditionally skipped reconnect if a synchronous
    get_version() COM call succeeded, but that call could hang forever
    on a zombie OpticStudio, deadlocking the worker under _zospy_lock.
    """
    error_type = type(error).__name__
    logger.error(f"{operation_name} {error_type}: {error}")
    logger.warning(f"{operation_name}: reconnecting after {error_type}...")
    record_reconnect_triggered(
        reason=f"{error_type} in {operation_name}: {error}",
        reconnect_failures=_reconnect_failures,
        backoff_s=_backoff_delay() if _reconnect_failures > 0 else 0,
    )
    await _reconnect_zospy()


def _not_connected_error() -> str:
    """Build the not-connected error message, appending the last error detail if available."""
    if _last_connection_error:
        return f"{NOT_CONNECTED_ERROR}: {_last_connection_error}"
    return NOT_CONNECTED_ERROR


async def verify_api_key(x_api_key: Optional[str] = Header(None, alias="X-API-Key")):
    """Verify API key if configured."""
    if ZEMAX_API_KEY is not None:
        if x_api_key != ZEMAX_API_KEY:
            raise HTTPException(status_code=401, detail="Invalid API key")


# =============================================================================
# Endpoint Helpers
# =============================================================================


async def _run_endpoint(
    endpoint_name: str,
    response_cls: type[BaseModel],
    request: BaseModel,
    handler: Callable[[], dict[str, Any]],
    build_response: Optional[Callable[[dict[str, Any]], BaseModel]] = None,
) -> BaseModel:
    """
    Run a standard endpoint with the common boilerplate:
    timed_operation -> lock -> ensure_connected -> load_system -> handler -> response.

    Args:
        endpoint_name: Name for logging/timing (e.g. "/cross-section")
        response_cls: Pydantic response model class (must have success and error fields)
        request: Request with zmx_content field
        handler: Callable that takes no args and returns a dict from zospy_handler.
                 The system is already loaded when this is called.
        build_response: Optional callable(result) -> response_cls for custom mapping.
                        If None, result dict keys are splatted directly into response_cls.

    Returns:
        Instance of response_cls with success/error or handler results.
    """
    with timed_operation(logger, endpoint_name):
        async with timed_lock_acquire(_zospy_lock, logger, name="zospy"):
            if await _ensure_connected() is None:
                return response_cls(success=False, error=_not_connected_error())

            record_operation_start(endpoint=endpoint_name)
            op_t0 = time.monotonic()
            try:
                _load_system_from_request(request)
                result = handler()

                dur_ms = (time.monotonic() - op_t0) * 1000
                record_operation_success(endpoint=endpoint_name, duration_ms=dur_ms)

                if build_response:
                    return build_response(result)

                if not result.get("success", True):
                    return response_cls(
                        success=False,
                        error=result.get("error", f"{endpoint_name} failed"),
                    )

                # Filter to known model fields; exclude "success"/"error" to avoid
                # duplicate keyword args (we set success=True explicitly above).
                model_fields = set(response_cls.model_fields.keys())
                return response_cls(success=True, **{
                    k: v for k, v in result.items()
                    if k not in ("success", "error") and k in model_fields
                })
            except Exception as e:
                dur_ms = (time.monotonic() - op_t0) * 1000
                record_operation_error(
                    endpoint=endpoint_name,
                    error=str(e),
                    error_type=type(e).__name__,
                    tb=_tb_mod.format_exc(),
                    duration_ms=dur_ms,
                )
                await _handle_zospy_error(endpoint_name, e)
                return response_cls(success=False, error=str(e))


def _load_system_from_request(request: BaseModel) -> dict[str, Any]:
    """
    Load optical system from request into OpticStudio.

    Args:
        request: Request with zmx_content field (base64-encoded .zmx file)

    Returns:
        Dict with load result (num_surfaces, efl)

    Raises:
        ValueError: If zmx_content is missing or invalid
        ZosPyError: If loading fails
    """
    zmx_content = getattr(request, 'zmx_content', None)

    if not zmx_content:
        raise ValueError("Request must include 'zmx_content'")

    try:
        zmx_bytes = base64.b64decode(zmx_content)
    except Exception as e:
        raise ValueError(f"Invalid base64 zmx_content: {e}") from e

    logger.info(f"Loading system from ZMX: {len(zmx_bytes)} bytes (base64 {len(zmx_content)})")

    # Quick sanity check: count WAVM/PWAV lines to verify wavelengths survived conversion
    try:
        zmx_text = zmx_bytes.decode('utf-16-le', errors='replace')
        wavm_count = zmx_text.count('WAVM ')
        pwav_lines = [l.strip() for l in zmx_text.split('\r\n') if l.strip().startswith('PWAV')]
        logger.info(f"ZMX content: WAVM={wavm_count}, PWAV={pwav_lines}")
    except Exception as e:
        logger.debug(f"Could not inspect ZMX text: {e}")

    with tempfile.NamedTemporaryFile(mode='wb', suffix='.zmx', delete=False) as f:
        f.write(zmx_bytes)
        temp_file = f.name

    try:
        result = zospy_handler.load_zmx_file(temp_file)
        if result.get("num_surfaces", 0) == 0:
            raise ZosPyError("System loaded but has no surfaces")
        logger.info(f"Loaded system: {result.get('num_surfaces')} surfaces, EFL={result.get('efl')}")
        return result
    finally:
        if os.path.exists(temp_file):
            os.unlink(temp_file)


# =============================================================================
# Application Setup
# =============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    global zospy_handler

    # Startup: Connection is lazy - happens on first request via _ensure_connected()
    # This allows the server to start instantly without waiting for OpticStudio/ZosPy
    logger.info("Starting Zemax Worker (lazy connection mode - connects on first request)")
    logger.info(f"Reporting worker_count={WORKER_COUNT} (from WEB_CONCURRENCY env var)")
    from diagnostics.connection_diagnostics import record_event
    record_event(
        "worker_startup",
        worker_count=WORKER_COUNT,
        web_concurrency=os.getenv("WEB_CONCURRENCY", "NOT_SET"),
    )
    if WORKER_COUNT == 1 and not os.getenv("WEB_CONCURRENCY"):
        logger.warning(
            "WEB_CONCURRENCY not set — reporting worker_count=1. "
            "Use 'python main.py --workers N' to set this automatically."
        )

    yield

    # Shutdown: Clean up ZosPy connection
    if zospy_handler:
        try:
            zospy_handler.close()
        except Exception as e:
            logger.warning(f"Error closing ZosPy connection: {e}")
    logger.info("Zemax Worker stopped.")


app = FastAPI(
    title="Zemax Worker",
    description="Windows worker for Zemax OpticStudio operations via ZosPy",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS middleware (allow requests from Mac orchestrator)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to specific Tailscale IPs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register all endpoint routers.
# NOTE: This must happen AFTER all module-level globals and helpers are defined,
# because router modules use `import main` to access main.zospy_handler,
# main._zospy_lock, main._run_endpoint, etc. at request time.
from routers import register_routers
register_routers(app)


# =============================================================================
# Utilities
# =============================================================================


def _kill_orphaned_opticstudio() -> int:
    """Kill orphaned OpticStudio/ZOSAPI processes from previous crashed workers.

    Called once from __main__ before uvicorn.run() spawns workers.
    All OpticStudio instances on this machine are headless API instances
    (standalone mode), so it's safe to kill them all at startup.

    Returns the number of processes targeted for cleanup.
    """
    try:
        import psutil
    except ImportError:
        logger.warning("psutil not installed — skipping orphan cleanup")
        return 0

    targets = []
    for proc in psutil.process_iter(['pid', 'name']):
        try:
            name = (proc.info['name'] or '').lower()
            if name in ('opticstudio.exe', 'zosapi.exe'):
                targets.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    if not targets:
        return 0

    logger.warning(
        f"Found {len(targets)} orphaned OpticStudio/ZOSAPI process(es), "
        f"killing: {[(p.pid, p.info['name']) for p in targets]}"
    )

    for proc in targets:
        try:
            proc.terminate()
        except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
            logger.warning(f"Could not terminate PID {proc.pid}: {e}")

    gone, alive = psutil.wait_procs(targets, timeout=5)

    force_killed = 0
    access_denied = 0
    for proc in alive:
        try:
            proc.kill()
            force_killed += 1
        except psutil.NoSuchProcess:
            pass  # Exited between wait_procs and kill — not counted
        except psutil.AccessDenied as e:
            access_denied += 1
            logger.warning(f"Could not force-kill PID {proc.pid}: {e}")

    parts = [f"{len(gone)} terminated", f"{force_killed} force-killed"]
    if access_denied:
        parts.append(f"{access_denied} access denied")
    logger.info(
        f"Cleaned up {len(targets)} orphaned process(es) "
        f"({', '.join(parts)}), waiting 3s for license release..."
    )
    time.sleep(3)
    return len(targets)


if __name__ == "__main__":
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="Zemax Worker")
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Number of uvicorn worker processes (overrides WEB_CONCURRENCY)",
    )
    args = parser.parse_args()

    port = int(os.getenv("PORT", str(DEFAULT_PORT)))
    host = os.getenv("HOST", DEFAULT_HOST)
    dev_mode = os.getenv("DEV_MODE", "false").lower() == "true"

    # Resolve worker count: CLI flag > WEB_CONCURRENCY env var > default 1
    num_workers = args.workers if args.workers is not None else int(os.getenv("WEB_CONCURRENCY", "1"))

    # Set WEB_CONCURRENCY so child processes report the correct worker_count
    # in /health. This is the canonical way uvicorn children discover the total.
    os.environ["WEB_CONCURRENCY"] = str(num_workers)

    _kill_orphaned_opticstudio()

    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        workers=num_workers,
        reload=dev_mode,
        log_level="info",
    )
