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

import asyncio
import base64
import logging
import os
import tempfile
import time
from contextlib import asynccontextmanager
from typing import Any, Callable, Literal, Optional

from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from zospy_handler import ZosPyHandler, ZosPyError
from utils.timing import timed_operation, timed_lock_acquire
from log_buffer import log_buffer
from diagnostics.connection_diagnostics import (
    record_connect_attempt, record_connect_success, record_connect_failure,
    record_disconnect, record_operation_start, record_operation_success,
    record_operation_error, record_reconnect_triggered, record_reconnect_skipped_backoff,
    record_connection_alive_check, record_license_seat_info, get_diagnostic_report,
)
import traceback as _tb_mod

# Configure logging
logging.basicConfig(level=logging.INFO)
logging.getLogger().addHandler(log_buffer)
# Uvicorn sets propagate=False on its loggers, so they never reach the root
# logger (and thus never reach log_buffer). Attach log_buffer directly so the
# Mac debug dashboard sees access logs and server lifecycle messages too.
for _uv_name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
    logging.getLogger(_uv_name).addHandler(log_buffer)
# Enable DEBUG for raw Zemax output logger (root is INFO, so explicit level needed)
logging.getLogger("zemax.raw").setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)

# =============================================================================
# Constants
# =============================================================================

# Error messages
NOT_CONNECTED_ERROR = "OpticStudio not connected"

# Default server configuration
DEFAULT_PORT = 8787
DEFAULT_HOST = "0.0.0.0"

# API key for authentication (optional but recommended)
ZEMAX_API_KEY = os.getenv("ZEMAX_API_KEY", None)

# Number of uvicorn workers behind this URL. Read from WEB_CONCURRENCY (set by
# __main__ or the operator). The analysis service reads this via /health to size
# its task queue. Defaults to 1 if unset.
WORKER_COUNT = int(os.getenv("WEB_CONCURRENCY", "1"))

# =============================================================================
# Global State
# =============================================================================

# Initialize ZosPy handler (manages OpticStudio connection)
zospy_handler: Optional[ZosPyHandler] = None
_last_connection_error: Optional[str] = None

# Reconnect backoff: prevents rapid reconnect loops that starve the license.
_RECONNECT_BACKOFF_BASE = 3.0   # seconds
_RECONNECT_BACKOFF_MAX = 60.0   # seconds
_RECONNECT_COM_RELEASE_DELAY = 2.0  # seconds to wait after close() for COM cleanup
_reconnect_failures = 0
_last_reconnect_attempt: float = 0.0

# Async lock - serializes ZosPy operations within this process
# Each uvicorn worker process has its own lock and OpticStudio connection
_zospy_lock = asyncio.Lock()


def _init_zospy() -> Optional[ZosPyHandler]:
    """Initialize ZosPy connection with error handling."""
    global _last_connection_error
    record_connect_attempt(mode="standalone")
    record_license_seat_info()
    try:
        handler = ZosPyHandler()
        version = handler.get_version()
        logger.info("ZosPy connection established.")
        record_connect_success(version=version)
        _last_connection_error = None
        return handler
    except Exception as e:
        tb = _tb_mod.format_exc()
        logger.error(f"Failed to initialize ZosPy: {e}")
        record_connect_failure(error=str(e), tb=tb)
        _last_connection_error = str(e)
        return None


def _is_connection_alive() -> bool:
    """Check if the OpticStudio COM connection is still responsive."""
    if zospy_handler is None:
        record_connection_alive_check(alive=False, check_duration_ms=0)
        return False
    t0 = time.monotonic()
    try:
        alive = zospy_handler.get_version() != "Unknown"
        dur = (time.monotonic() - t0) * 1000
        record_connection_alive_check(alive=alive, check_duration_ms=dur)
        return alive
    except Exception:
        dur = (time.monotonic() - t0) * 1000
        record_connection_alive_check(alive=False, check_duration_ms=dur)
        return False


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
        except Exception:
            pass
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
    """Handle errors from ZosPy operations, reconnecting only when the connection is dead."""
    tb = _tb_mod.format_exc()
    if isinstance(error, ZosPyError):
        logger.error(f"{operation_name} ZosPyError: {error}")
        record_reconnect_triggered(
            reason=f"ZosPyError in {operation_name}: {error}",
            reconnect_failures=_reconnect_failures,
            backoff_s=_backoff_delay() if _reconnect_failures > 0 else 0,
        )
        await _reconnect_zospy()
        return

    # Generic exception (COM/analysis). Only reconnect if the connection died;
    # transient analysis errors should not destroy a working connection.
    logger.error(f"{operation_name} failed: {error}")
    if _is_connection_alive():
        logger.info(f"{operation_name}: connection still alive, not reconnecting")
    else:
        logger.warning(f"{operation_name}: connection dead, reconnecting...")
        record_reconnect_triggered(
            reason=f"connection_dead after {type(error).__name__} in {operation_name}: {error}",
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


async def _run_endpoint(
    endpoint_name: str,
    response_cls: type[BaseModel],
    request: BaseModel,
    handler: Callable[[], dict[str, Any]],
    build_response: Optional[Callable[[dict[str, Any]], BaseModel]] = None,
) -> BaseModel:
    """
    Run a standard endpoint with the common boilerplate:
    timed_operation → lock → ensure_connected → load_system → handler → response.

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

    logger.info("Loading system from ZMX content")
    try:
        zmx_bytes = base64.b64decode(zmx_content)
    except Exception as e:
        raise ValueError(f"Invalid base64 zmx_content: {e}") from e

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


# =============================================================================
# Request/Response Models
# =============================================================================


class SystemRequest(BaseModel):
    """
    Request containing an optical system.

    Requires zmx_content: Base64-encoded .zmx file from zemax-converter.
    """
    zmx_content: str = Field(description="Base64-encoded .zmx file content")


class HealthResponse(BaseModel):
    """Health check response."""
    success: bool = Field(description="Whether the worker is healthy")
    opticstudio_connected: bool = Field(description="Whether OpticStudio is connected")
    version: Optional[str] = Field(default=None, description="OpticStudio version")
    zospy_version: Optional[str] = Field(default=None, description="ZosPy version")
    worker_count: int = Field(description="Number of uvicorn worker processes serving this URL")
    connection_error: Optional[str] = Field(default=None, description="Error detail when opticstudio_connected is False")


class CrossSectionRequest(BaseModel):
    """Cross-section diagram request."""
    zmx_content: str = Field(description="Base64-encoded .zmx file content")
    number_of_rays: int = Field(default=11, ge=3, le=100, description="Number of rays per field")
    color_rays_by: Literal["Fields", "Wavelengths", "None"] = Field(default="Fields", description="Color rays by")


class CrossSectionResponse(BaseModel):
    """Cross-section diagram response."""
    success: bool = Field(description="Whether the operation succeeded")
    image: Optional[str] = Field(default=None, description="Base64-encoded PNG image or numpy array bytes")
    image_format: Optional[str] = Field(default=None, description="Image format: 'png' or 'numpy_array'")
    array_shape: Optional[list[int]] = Field(default=None, description="Shape for numpy_array reconstruction")
    array_dtype: Optional[str] = Field(default=None, description="Dtype for numpy_array reconstruction")
    paraxial: Optional[dict[str, Any]] = Field(default=None, description="Paraxial properties")
    surfaces: Optional[list[dict[str, Any]]] = Field(default=None, description="Surface geometry for fallback rendering")
    rays_total: Optional[int] = Field(default=None, description="Total rays traced")
    rays_through: Optional[int] = Field(default=None, description="Rays reaching image")
    error: Optional[str] = Field(default=None, description="Error message")


class RayTraceDiagnosticRequest(BaseModel):
    """Ray trace diagnostic request."""
    zmx_content: str = Field(description="Base64-encoded .zmx file content")
    num_rays: int = Field(default=21, description="Number of rays per field (determines grid density). Higher values = more accuracy but slower. 21 gives ~13 rays in circular pupil.")
    distribution: str = Field(default="hexapolar", description="Ray distribution (currently uses square grid)")


class RawRay(BaseModel):
    """Raw ray trace result for a single ray."""
    field_index: int = Field(description="0-indexed field number")
    field_x: float = Field(description="Field X coordinate")
    field_y: float = Field(description="Field Y coordinate")
    px: float = Field(description="Normalized pupil X coordinate (-1 to 1)")
    py: float = Field(description="Normalized pupil Y coordinate (-1 to 1)")
    reached_image: bool = Field(description="Whether the ray reached the image surface")
    failed_surface: Optional[int] = Field(default=None, description="Surface index where ray failed (if applicable)")
    failure_mode: Optional[str] = Field(default=None, description="Failure mode: MISS, TIR, VIGNETTE, etc.")


class RayTraceDiagnosticResponse(BaseModel):
    """
    Raw ray trace diagnostic response.

    This is a "dumb executor" response - returns raw per-ray data only.
    All aggregation, hotspot detection, and threshold calculations
    happen on the Mac side (zemax-analysis-service).
    """
    success: bool = Field(description="Whether the operation succeeded")
    paraxial: Optional[dict[str, Any]] = Field(default=None, description="Basic paraxial data (efl, bfl, fno, total_track)")
    num_surfaces: Optional[int] = Field(default=None, description="Number of surfaces in system")
    num_fields: Optional[int] = Field(default=None, description="Number of fields")
    raw_rays: Optional[list[RawRay]] = Field(default=None, description="Per-ray trace results")
    surface_semi_diameters: Optional[list[float]] = Field(default=None, description="Semi-diameters from LDE")
    error: Optional[str] = Field(default=None, description="Error message if operation failed")


class ZernikeResponse(BaseModel):
    """Raw Zernike coefficients response - conversion to Seidel happens on Mac."""
    success: bool = Field(description="Whether the operation succeeded")
    zernike_coefficients: Optional[list[float]] = Field(default=None, description="Raw Zernike coefficients Z1-Z37")
    wavelength_um: Optional[float] = Field(default=None, description="Wavelength in micrometers")
    num_surfaces: int = Field(default=0)
    error: Optional[str] = Field(default=None)


class SemiDiametersResponse(BaseModel):
    """Semi-diameter calculation response."""
    success: bool = Field(description="Whether the operation succeeded")
    semi_diameters: Optional[list[dict[str, Any]]] = Field(default=None)
    error: Optional[str] = Field(default=None)


class LoadSystemResponse(BaseModel):
    """Load system response."""
    success: bool = Field(description="Whether the operation succeeded")
    num_surfaces: Optional[int] = Field(default=None, description="Number of surfaces loaded")
    efl: Optional[float] = Field(default=None, description="Effective focal length")
    error: Optional[str] = Field(default=None, description="Error message")


class TraceRaysResponse(BaseModel):
    """Trace rays response."""
    success: bool = Field(description="Whether the operation succeeded")
    num_surfaces: Optional[int] = Field(default=None)
    num_fields: Optional[int] = Field(default=None)
    num_wavelengths: Optional[int] = Field(default=None)
    data: Optional[list[dict[str, Any]]] = Field(default=None, description="Ray trace data")
    error: Optional[str] = Field(default=None)


class WavefrontRequest(BaseModel):
    """Wavefront analysis request."""
    zmx_content: str = Field(description="Base64-encoded .zmx file content")
    field_index: int = Field(default=1, ge=1, description="Field index (1-indexed)")
    wavelength_index: int = Field(default=1, ge=1, description="Wavelength index (1-indexed)")
    sampling: str = Field(default="64x64", description="Pupil sampling grid (e.g., '32x32', '64x64', '128x128')")
    remove_tilt: bool = Field(default=False, description="Remove tilt from wavefront map")


class WavefrontResponse(BaseModel):
    """
    Wavefront analysis response.

    Returns raw wavefront data including:
    - RMS and P-V wavefront error in waves
    - Strehl ratio (if available)
    - Wavefront map as numpy array (Mac side renders to PNG)
    """
    success: bool = Field(description="Whether the operation succeeded")
    rms_waves: Optional[float] = Field(default=None, description="RMS wavefront error in waves")
    pv_waves: Optional[float] = Field(default=None, description="Peak-to-valley wavefront error in waves")
    strehl_ratio: Optional[float] = Field(default=None, description="Strehl ratio (0-1)")
    wavelength_um: Optional[float] = Field(default=None, description="Wavelength in micrometers")
    field_x: Optional[float] = Field(default=None, description="Field X coordinate")
    field_y: Optional[float] = Field(default=None, description="Field Y coordinate")
    image: Optional[str] = Field(default=None, description="Base64-encoded numpy array bytes")
    image_format: Optional[str] = Field(default=None, description="Image format: 'numpy_array'")
    array_shape: Optional[list[int]] = Field(default=None, description="Shape for numpy array reconstruction")
    array_dtype: Optional[str] = Field(default=None, description="Dtype for numpy array reconstruction")
    error: Optional[str] = Field(default=None, description="Error message if operation failed")


class SpotDiagramRequest(BaseModel):
    """Spot diagram analysis request."""
    zmx_content: str = Field(description="Base64-encoded .zmx file content")
    ray_density: int = Field(default=5, ge=1, le=20, description="Rays per axis (grid density)")
    reference: str = Field(default="centroid", description="Reference point: 'chief_ray' or 'centroid'")
    field_index: Optional[int] = Field(default=None, ge=1, description="Field index (1-indexed). None = all fields.")
    wavelength_index: Optional[int] = Field(default=None, ge=1, description="Wavelength index (1-indexed). None = all wavelengths.")


class SpotFieldData(BaseModel):
    """Spot diagram data for a single field point."""
    field_index: int = Field(description="0-indexed field number")
    field_x: float = Field(description="Field X coordinate")
    field_y: float = Field(description="Field Y coordinate")
    rms_radius: Optional[float] = Field(default=None, description="RMS spot radius in lens units")
    geo_radius: Optional[float] = Field(default=None, description="GEO (max) spot radius in lens units")
    centroid_x: Optional[float] = Field(default=None, description="Centroid X coordinate on image plane")
    centroid_y: Optional[float] = Field(default=None, description="Centroid Y coordinate on image plane")
    num_rays: Optional[int] = Field(default=None, description="Number of rays traced for this field")


class SpotRayPoint(BaseModel):
    """A single ray hit point on the image plane."""
    x: float = Field(description="X coordinate on image plane")
    y: float = Field(description="Y coordinate on image plane")


class SpotRayData(BaseModel):
    """Raw ray data for a single field/wavelength combination."""
    field_index: int = Field(description="0-based field index")
    field_x: float = Field(description="Field X coordinate")
    field_y: float = Field(description="Field Y coordinate")
    wavelength_index: int = Field(description="0-based wavelength index")
    rays: list[SpotRayPoint] = Field(default_factory=list, description="Ray hit points on image plane")


class SpotDiagramResponse(BaseModel):
    """
    Spot diagram analysis response.

    Returns spot metrics (RMS, GEO radius) and raw ray data for Mac-side rendering.
    ZOSAPI's StandardSpot does NOT support image export - use spot_rays for rendering.
    """
    success: bool = Field(description="Whether the operation succeeded")
    image: Optional[str] = Field(default=None, description="Always None - ZOSAPI StandardSpot doesn't support image export")
    image_format: Optional[str] = Field(default=None, description="Always None")
    array_shape: Optional[list[int]] = Field(default=None, description="Always None")
    array_dtype: Optional[str] = Field(default=None, description="Always None")
    spot_data: Optional[list[SpotFieldData]] = Field(default=None, description="Per-field spot metrics (RMS, GEO radius, centroid)")
    spot_rays: Optional[list[SpotRayData]] = Field(default=None, description="Raw ray X,Y positions for Mac-side rendering")
    airy_radius: Optional[float] = Field(default=None, description="Airy disk radius in lens units")
    error: Optional[str] = Field(default=None, description="Error message if operation failed")


class NativeSeidelResponse(BaseModel):
    """
    Native Seidel raw text response from OpticStudio SeidelCoefficients analysis.

    Returns the raw UTF-16 text output from OpticStudio's GetTextFile().
    Parsing happens on the Mac side (seidel_text_parser.py).
    """
    success: bool = Field(description="Whether the operation succeeded")
    seidel_text: Optional[str] = Field(default=None, description="Raw text from OpticStudio GetTextFile()")
    num_surfaces: Optional[int] = Field(default=None, description="Number of optical surfaces")
    error: Optional[str] = Field(default=None, description="Error message if operation failed")


class MTFRequest(BaseModel):
    """MTF analysis request."""
    zmx_content: str = Field(description="Base64-encoded .zmx file content")
    field_index: int = Field(default=0, ge=0, description="Field index (0 = all fields, 1+ = specific field, 1-indexed)")
    wavelength_index: int = Field(default=1, ge=1, description="Wavelength index (1-indexed)")
    sampling: str = Field(default="64x64", description="Pupil sampling grid")
    maximum_frequency: float = Field(default=0.0, ge=0, description="Maximum spatial frequency (cycles/mm). 0 = auto.")


class MTFFieldData(BaseModel):
    """MTF data for a single field point."""
    field_index: int = Field(description="0-indexed field number")
    field_x: float = Field(description="Field X coordinate")
    field_y: float = Field(description="Field Y coordinate")
    tangential: list[float] = Field(default_factory=list, description="Tangential MTF values")
    sagittal: list[float] = Field(default_factory=list, description="Sagittal MTF values")


class MTFResponse(BaseModel):
    """MTF analysis response."""
    success: bool = Field(description="Whether the operation succeeded")
    frequency: Optional[list[float]] = Field(default=None, description="Spatial frequency array (cycles/mm)")
    fields: Optional[list[MTFFieldData]] = Field(default=None, description="Per-field MTF data")
    diffraction_limit: Optional[list[float]] = Field(default=None, description="Diffraction-limited MTF curve")
    cutoff_frequency: Optional[float] = Field(default=None, description="Cutoff frequency (cycles/mm)")
    wavelength_um: Optional[float] = Field(default=None, description="Wavelength in micrometers")
    error: Optional[str] = Field(default=None, description="Error message if operation failed")


class HuygensMTFRequest(BaseModel):
    """Huygens MTF analysis request."""
    zmx_content: str = Field(description="Base64-encoded .zmx file content")
    field_index: int = Field(default=0, ge=0, description="Field index (0 = all fields, 1+ = specific field, 1-indexed)")
    wavelength_index: int = Field(default=1, ge=1, description="Wavelength index (1-indexed)")
    sampling: str = Field(default="64x64", description="Pupil sampling grid")
    maximum_frequency: float = Field(default=0.0, ge=0, description="Maximum spatial frequency (cycles/mm). 0 = auto.")




class RayFanRequest(BaseModel):
    """Ray Fan analysis request."""
    zmx_content: str = Field(description="Base64-encoded .zmx file content")
    field_index: int = Field(default=0, ge=0, description="Field index (0 = all fields, 1+ = specific field, 1-indexed)")
    wavelength_index: int = Field(default=0, ge=0, description="Wavelength index (0 = all wavelengths, 1+ = specific, 1-indexed)")
    plot_scale: float = Field(default=0.0, ge=0, description="Maximum vertical scale for plots; 0 = auto")
    number_of_rays: int = Field(default=20, ge=5, le=100, description="Number of rays traced on each side of origin")


class RayFanFieldData(BaseModel):
    """Ray fan data for a single field/wavelength combination."""
    field_index: int = Field(description="0-indexed field number")
    field_x: float = Field(description="Field X coordinate")
    field_y: float = Field(description="Field Y coordinate")
    wavelength_um: float = Field(default=0.0, description="Wavelength in micrometers")
    wavelength_index: int = Field(default=0, description="Wavelength index")
    tangential_py: list[float] = Field(default_factory=list, description="Tangential pupil Y coordinates")
    tangential_ey: list[float] = Field(default_factory=list, description="Tangential aberration EY values")
    sagittal_px: list[float] = Field(default_factory=list, description="Sagittal pupil X coordinates")
    sagittal_ex: list[float] = Field(default_factory=list, description="Sagittal aberration EX values")


class RayFanResponse(BaseModel):
    """Ray Fan analysis response."""
    success: bool = Field(description="Whether the operation succeeded")
    fans: Optional[list[RayFanFieldData]] = Field(default=None, description="Per-field/wavelength fan data")
    max_aberration: Optional[float] = Field(default=None, description="Maximum aberration value")
    num_fields: int = Field(default=0, description="Number of fields in the system")
    num_wavelengths: int = Field(default=0, description="Number of wavelengths")
    error: Optional[str] = Field(default=None, description="Error message if operation failed")

class PSFRequest(BaseModel):
    """PSF analysis request."""
    zmx_content: str = Field(description="Base64-encoded .zmx file content")
    field_index: int = Field(default=1, ge=1, description="Field index (1-indexed)")
    wavelength_index: int = Field(default=1, ge=1, description="Wavelength index (1-indexed)")
    sampling: str = Field(default="64x64", description="Pupil sampling grid")


class PSFResponse(BaseModel):
    """PSF analysis response."""
    success: bool = Field(description="Whether the operation succeeded")
    image: Optional[str] = Field(default=None, description="Base64-encoded numpy array bytes")
    image_format: Optional[str] = Field(default=None, description="Image format: 'numpy_array'")
    array_shape: Optional[list[int]] = Field(default=None, description="Shape for numpy array reconstruction")
    array_dtype: Optional[str] = Field(default=None, description="Dtype for numpy array reconstruction")
    strehl_ratio: Optional[float] = Field(default=None, description="Strehl ratio (0-1)")
    psf_peak: Optional[float] = Field(default=None, description="Peak PSF intensity")
    wavelength_um: Optional[float] = Field(default=None, description="Wavelength in micrometers")
    field_x: Optional[float] = Field(default=None, description="Field X coordinate")
    field_y: Optional[float] = Field(default=None, description="Field Y coordinate")
    error: Optional[str] = Field(default=None, description="Error message if operation failed")


class GeometricImageRequest(BaseModel):
    """Geometric Image Analysis request."""
    zmx_content: str = Field(description="Base64-encoded .zmx file content")
    field_size: float = Field(default=0.0, ge=0, description="Image width in field coordinates (0 = auto)")
    image_size: float = Field(default=50.0, gt=0, description="Detector size in lens units")
    rays_x_1000: int = Field(default=10, ge=1, le=100, description="Approximate ray count in thousands")
    number_of_pixels: int = Field(default=100, ge=10, le=1000, description="Pixels across image width")
    field: int = Field(default=1, ge=1, description="Field number (1-indexed)")
    wavelength: str = Field(default="All", description="Wavelength: 'All' or wavelength number")


class GeometricImageResponse(BaseModel):
    """Geometric Image Analysis response."""
    success: bool = Field(description="Whether the operation succeeded")
    image: Optional[str] = Field(default=None, description="Base64-encoded numpy array bytes")
    image_format: Optional[str] = Field(default=None, description="Image format: 'numpy_array'")
    array_shape: Optional[list[int]] = Field(default=None, description="Shape for numpy array reconstruction")
    array_dtype: Optional[str] = Field(default=None, description="Dtype for numpy array reconstruction")
    field_size: Optional[float] = Field(default=None, description="Field size used")
    image_size: Optional[float] = Field(default=None, description="Image size used")
    rays_x_1000: Optional[int] = Field(default=None, description="Rays x 1000 used")
    number_of_pixels: Optional[int] = Field(default=None, description="Number of pixels used")
    paraxial: Optional[dict[str, Any]] = Field(default=None, description="Paraxial properties")
    error: Optional[str] = Field(default=None, description="Error message if operation failed")


class MeritFunctionOperandRow(BaseModel):
    """A single merit function operand row."""
    operand_code: str = Field(description="Zemax operand code (e.g. EFFL, MTFA)")
    params: list[Optional[float]] = Field(default_factory=list, max_length=6, description="Up to 6 parameter values [Int1, Int2, Hx, Hy, Px, Py]")
    target: float = Field(default=0, description="Target value")
    weight: float = Field(default=1, description="Weight")


class MeritFunctionRequest(BaseModel):
    """Request to evaluate a merit function."""
    zmx_content: str = Field(description="Base64-encoded .zmx file content")
    operand_rows: list[MeritFunctionOperandRow] = Field(max_length=200, description="Merit function operand rows")


class EvaluatedOperandRow(BaseModel):
    """Result for a single evaluated operand row."""
    row_index: int = Field(description="0-based index in the original request")
    operand_code: str = Field(description="Zemax operand code")
    value: Optional[float] = Field(default=None, description="Computed value")
    target: float = Field(description="Target value")
    weight: float = Field(description="Weight")
    contribution: Optional[float] = Field(default=None, description="Contribution to total merit")
    error: Optional[str] = Field(default=None, description="Per-row error message")


class MeritFunctionResponse(BaseModel):
    """Response from merit function evaluation."""
    success: bool = Field(description="Whether the evaluation succeeded")
    total_merit: Optional[float] = Field(default=None, description="Total merit function value")
    evaluated_rows: Optional[list[EvaluatedOperandRow]] = Field(default=None, description="Per-row results")
    row_errors: Optional[list[dict[str, Any]]] = Field(default=None, description="Per-row errors for invalid/failed operands")
    error: Optional[str] = Field(default=None, description="Error message if failed")


class OptimizationWizardRequest(BaseModel):
    """Request to apply the SEQ Optimization Wizard to generate merit function operands."""
    zmx_content: str = Field(description="Base64-encoded .zmx file content")
    criterion: Literal["Spot", "Wavefront", "Angular", "Contrast"] = Field(default="Spot", description="Optimization criterion: Spot, Wavefront, Angular, or Contrast")
    reference: Literal["Centroid", "ChiefRay"] = Field(default="Centroid", description="Reference type: Centroid or ChiefRay")
    overall_weight: float = Field(default=1.0, ge=0, description="Overall weight for wizard operands")
    rings: int = Field(default=3, ge=1, le=20, description="Number of pupil rings")
    arms: Literal[6, 8, 10, 12] = Field(default=6, description="Number of pupil arms: 6, 8, 10, or 12")
    use_gaussian_quadrature: bool = Field(default=False, description="Use Gaussian quadrature sampling")
    use_glass_boundary_values: bool = Field(default=False, description="Apply glass thickness constraints")
    glass_min: float = Field(default=1.0, ge=0, description="Minimum glass thickness (mm)")
    glass_max: float = Field(default=50.0, ge=0, description="Maximum glass thickness (mm)")
    use_air_boundary_values: bool = Field(default=False, description="Apply air spacing constraints")
    air_min: float = Field(default=0.5, ge=0, description="Minimum air spacing (mm)")
    air_max: float = Field(default=1000.0, ge=0, description="Maximum air spacing (mm)")
    air_edge_thickness: float = Field(default=0.0, ge=0, description="Minimum edge thickness for air spaces (mm)")
    # Optimization Function
    type: Literal["RMS", "PTV"] = Field(default="RMS", description="Optimization type: RMS or PTV")
    spatial_frequency: float = Field(default=30.0, gt=0, description="Spatial frequency (cycles/mm)")
    xs_weight: float = Field(default=1.0, ge=0, description="Sagittal (X) weight")
    yt_weight: float = Field(default=1.0, ge=0, description="Tangential (Y) weight")
    use_maximum_distortion: bool = Field(default=False, description="Enable maximum distortion constraint")
    max_distortion_pct: float = Field(default=1.0, ge=0, description="Maximum distortion percentage")
    ignore_lateral_color: bool = Field(default=False, description="Ignore lateral color")
    # Pupil Integration
    obscuration: float = Field(default=0.0, ge=0, le=1, description="Pupil obscuration ratio (0-1)")
    # Boundary Values
    glass_edge_thickness: float = Field(default=0.0, ge=0, description="Minimum edge thickness for glass (mm)")
    # Optimization Goal
    optimization_goal: Literal["nominal", "manufacturing_yield"] = Field(default="nominal", description="Goal: nominal or manufacturing_yield")
    manufacturing_yield_weight: float = Field(default=1.0, ge=0, description="Manufacturing yield weight")
    # Bottom bar
    start_at: int = Field(default=1, ge=1, description="Starting surface index")
    use_all_configurations: bool = Field(default=True, description="Use all configurations")
    configuration_number: int = Field(default=1, ge=1, description="Specific configuration number")
    use_all_fields: bool = Field(default=True, description="Use all fields")
    field_number: int = Field(default=1, ge=1, description="Specific field number")
    assume_axial_symmetry: bool = Field(default=True, description="Assume axial symmetry")
    add_favorite_operands: bool = Field(default=False, description="Add favorite operands")
    delete_vignetted: bool = Field(default=True, description="Delete vignetted rays")


class WizardGeneratedRow(BaseModel):
    """A single merit function row generated by the optimization wizard."""
    row_index: int = Field(description="0-based row index")
    operand_code: str = Field(description="Zemax operand code (e.g. BLNK, DMFS, OPDX)")
    params: list[Optional[float]] = Field(default_factory=list, description="6 parameter values [Int1, Int2, Hx, Hy, Px, Py]")
    target: float = Field(default=0, description="Target value")
    weight: float = Field(default=0, description="Weight")
    value: Optional[float] = Field(default=None, description="Computed value after wizard apply")
    contribution: Optional[float] = Field(default=None, description="Contribution to total merit")


class OptimizationWizardResponse(BaseModel):
    """Response from the optimization wizard."""
    success: bool = Field(description="Whether the wizard succeeded")
    total_merit: Optional[float] = Field(default=None, description="Total merit function value after wizard")
    generated_rows: Optional[list[WizardGeneratedRow]] = Field(default=None, description="All generated operand rows")
    num_rows_generated: int = Field(default=0, description="Number of rows generated")
    error: Optional[str] = Field(default=None, description="Error message if failed")


class ParaxialResponse(BaseModel):
    """First-order (paraxial) optical properties response."""
    success: bool = Field(description="Whether the operation succeeded")
    efl: Optional[float] = Field(default=None, description="Effective focal length (mm)")
    bfl: Optional[float] = Field(default=None, description="Back focal length (mm)")
    fno: Optional[float] = Field(default=None, description="F-number")
    na: Optional[float] = Field(default=None, description="Numerical aperture")
    epd: Optional[float] = Field(default=None, description="Entrance pupil diameter (mm)")
    total_track: Optional[float] = Field(default=None, description="Total track length (mm)")
    max_field: Optional[float] = Field(default=None, description="Maximum field value")
    field_type: Optional[str] = Field(default=None, description="Field type (e.g. object_angle)")
    field_unit: Optional[str] = Field(default=None, description="Field unit (e.g. deg)")
    image_height: Optional[float] = Field(default=None, description="Paraxial image height (mm)")
    error: Optional[str] = Field(default=None, description="Error message if failed")


class SurfaceDataReportResponse(BaseModel):
    """Surface Data Report response."""
    success: bool = Field(description="Whether the operation succeeded")
    surfaces: Optional[list[dict[str, Any]]] = Field(default=None, description="Per-surface data (edge thickness, material, refractive index, power)")
    paraxial: Optional[dict[str, Any]] = Field(default=None, description="Paraxial properties")
    error: Optional[str] = Field(default=None, description="Error message if operation failed")


# =============================================================================
# Endpoints
# =============================================================================


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Health check endpoint.

    Returns the status of the worker and OpticStudio connection.
    Acquires _zospy_lock with a 2-second timeout to avoid reading
    zospy_handler during reconnection or other mutations.
    """
    lock_acquired = False
    try:
        await asyncio.wait_for(_zospy_lock.acquire(), timeout=2.0)
        lock_acquired = True
    except asyncio.TimeoutError:
        # Lock held by a long operation — worker is busy but healthy
        return HealthResponse(
            success=True,
            opticstudio_connected=False,
            version=None,
            zospy_version=None,
            worker_count=WORKER_COUNT,
            connection_error="Health check timed out (worker busy)",
        )
    except asyncio.CancelledError:
        # wait_for may have acquired the lock before cancellation propagated
        if lock_acquired:
            _zospy_lock.release()
        raise

    try:
        if zospy_handler is None:
            return HealthResponse(
                success=True,  # Worker is running
                opticstudio_connected=False,
                version=None,
                zospy_version=None,
                worker_count=WORKER_COUNT,
                connection_error=_last_connection_error,
            )

        try:
            status = zospy_handler.get_status()
            return HealthResponse(
                success=True,
                opticstudio_connected=status.get("connected", False),
                version=status.get("opticstudio_version"),
                zospy_version=status.get("zospy_version"),
                worker_count=WORKER_COUNT,
            )
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return HealthResponse(
                success=True,
                opticstudio_connected=False,
                version=None,
                zospy_version=None,
                worker_count=WORKER_COUNT,
                connection_error=str(e),
            )
    finally:
        if lock_acquired:
            _zospy_lock.release()


@app.get("/diagnostics/connection")
async def diagnostics_connection(_: None = Depends(verify_api_key)):
    """Return connection diagnostic events for debugging license/connection issues."""
    return get_diagnostic_report()


@app.get("/logs")
async def get_logs(
    since: int = 0,
    limit: int = 200,
    _: None = Depends(verify_api_key),
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


@app.post("/load-system", response_model=LoadSystemResponse)
async def load_system(request: SystemRequest, _: None = Depends(verify_api_key)) -> LoadSystemResponse:
    """Load an optical system into OpticStudio from zmx_content."""
    with timed_operation(logger, "/load-system"):
        async with timed_lock_acquire(_zospy_lock, logger, name="zospy"):
            if await _ensure_connected() is None:
                return LoadSystemResponse(success=False, error=_not_connected_error())

            try:
                result = _load_system_from_request(request)
                return LoadSystemResponse(
                    success=True,
                    num_surfaces=result.get("num_surfaces"),
                    efl=result.get("efl"),
                )
            except Exception as e:
                await _handle_zospy_error("Load system", e)
                return LoadSystemResponse(success=False, error=str(e))


@app.post("/cross-section", response_model=CrossSectionResponse)
async def get_cross_section(request: CrossSectionRequest, _: None = Depends(verify_api_key)) -> CrossSectionResponse:
    """Generate cross-section diagram using ZosPy's CrossSection analysis."""
    return await _run_endpoint(
        "/cross-section", CrossSectionResponse, request,
        lambda: zospy_handler.get_cross_section(
            number_of_rays=request.number_of_rays,
            color_rays_by=request.color_rays_by,
        ),
    )


@app.post("/calc-semi-diameters", response_model=SemiDiametersResponse)
async def calc_semi_diameters(request: SystemRequest, _: None = Depends(verify_api_key)) -> SemiDiametersResponse:
    """Calculate semi-diameters by tracing edge rays."""
    return await _run_endpoint(
        "/calc-semi-diameters", SemiDiametersResponse, request,
        lambda: zospy_handler.calc_semi_diameters(),
    )


@app.post("/ray-trace-diagnostic", response_model=RayTraceDiagnosticResponse)
async def ray_trace_diagnostic(
    request: RayTraceDiagnosticRequest,
    _: None = Depends(verify_api_key),
) -> RayTraceDiagnosticResponse:
    """
    Trace rays through the system and return raw per-ray results.

    This is a "dumb executor" endpoint - returns raw data only.
    All aggregation, hotspot detection, and threshold calculations
    happen on the Mac side (zemax-analysis-service).
    """
    return await _run_endpoint(
        "/ray-trace-diagnostic", RayTraceDiagnosticResponse, request,
        lambda: zospy_handler.ray_trace_diagnostic(
            num_rays=request.num_rays,
            distribution=request.distribution,
        ),
    )


@app.post("/seidel", response_model=ZernikeResponse)
async def get_zernike(request: SystemRequest, _: None = Depends(verify_api_key)) -> ZernikeResponse:
    """Get raw Zernike coefficients - conversion to Seidel happens on Mac side."""
    return await _run_endpoint(
        "/seidel", ZernikeResponse, request,
        lambda: zospy_handler.get_seidel(),
    )


@app.post("/seidel-native", response_model=NativeSeidelResponse)
async def get_seidel_native(request: SystemRequest, _: None = Depends(verify_api_key)) -> NativeSeidelResponse:
    """
    Get native Seidel text output from OpticStudio's SeidelCoefficients analysis.

    Returns raw text — parsing happens on the Mac side (seidel_text_parser.py).
    """
    return await _run_endpoint(
        "/seidel-native", NativeSeidelResponse, request,
        lambda: zospy_handler.get_seidel_native(),
    )


@app.post("/trace-rays", response_model=TraceRaysResponse)
async def trace_rays(
    request: SystemRequest,
    num_rays: int = 7,
    _: None = Depends(verify_api_key),
) -> TraceRaysResponse:
    """Trace rays through the system and return positions at each surface."""
    return await _run_endpoint(
        "/trace-rays", TraceRaysResponse, request,
        lambda: zospy_handler.trace_rays(num_rays=num_rays),
    )


@app.post("/wavefront", response_model=WavefrontResponse)
async def get_wavefront(
    request: WavefrontRequest,
    _: None = Depends(verify_api_key),
) -> WavefrontResponse:
    """
    Get wavefront error map and RMS wavefront error.

    This is a "dumb executor" endpoint - returns raw data only.
    Wavefront map image rendering happens on Mac side using matplotlib.
    """
    return await _run_endpoint(
        "/wavefront", WavefrontResponse, request,
        lambda: zospy_handler.get_wavefront(
            field_index=request.field_index,
            wavelength_index=request.wavelength_index,
            sampling=request.sampling,
            remove_tilt=request.remove_tilt,
        ),
    )


@app.post("/spot-diagram", response_model=SpotDiagramResponse)
async def get_spot_diagram(
    request: SpotDiagramRequest,
    _: None = Depends(verify_api_key),
) -> SpotDiagramResponse:
    """
    Generate spot diagram using ZosPy's StandardSpot analysis for metrics
    and batch ray tracing for raw ray positions.

    ZOSAPI's StandardSpot does NOT support image export. The response includes:
    - spot_data: Per-field metrics (RMS, GEO radius, centroid) from StandardSpot
    - spot_rays: Raw ray X,Y positions from batch ray tracing for Mac-side rendering

    This is a "dumb executor" endpoint - Mac side renders the spot diagram from spot_rays.
    """
    def _build_spot_response(result: dict) -> SpotDiagramResponse:
        if not result.get("success", False):
            logger.warning(f"[SPOT] Handler failure: {result.get('error')}")
            return SpotDiagramResponse(
                success=False,
                error=result.get("error", "Spot diagram analysis failed"),
            )

        spot_data = None
        if result.get("spot_data"):
            spot_data = [SpotFieldData(**sd) for sd in result["spot_data"]]

        # Convert spot_rays dicts to SpotRayData models
        spot_rays = None
        total_rays = 0
        if result.get("spot_rays"):
            spot_rays = []
            for ray_data in result["spot_rays"]:
                rays = [SpotRayPoint(x=r["x"], y=r["y"]) for r in ray_data.get("rays", [])]
                spot_rays.append(SpotRayData(
                    field_index=ray_data["field_index"],
                    field_x=ray_data["field_x"],
                    field_y=ray_data["field_y"],
                    wavelength_index=ray_data["wavelength_index"],
                    rays=rays,
                ))
            total_rays = sum(len(sr.rays) for sr in spot_rays)

        logger.info(
            f"[SPOT] Response: spot_data={len(spot_data) if spot_data else 0}, "
            f"spot_rays={len(spot_rays) if spot_rays else 0}, "
            f"total_rays={total_rays}, airy_radius={result.get('airy_radius')}"
        )

        return SpotDiagramResponse(
            success=True,
            image=result.get("image"),
            image_format=result.get("image_format"),
            array_shape=result.get("array_shape"),
            array_dtype=result.get("array_dtype"),
            spot_data=spot_data,
            spot_rays=spot_rays,
            airy_radius=result.get("airy_radius"),
        )

    return await _run_endpoint(
        "/spot-diagram", SpotDiagramResponse, request,
        lambda: zospy_handler.get_spot_diagram(
            ray_density=request.ray_density,
            reference=request.reference,
            field_index=request.field_index,
            wavelength_index=request.wavelength_index,
        ),
        build_response=_build_spot_response,
    )


@app.post("/evaluate-merit-function", response_model=MeritFunctionResponse)
async def evaluate_merit_function(
    request: MeritFunctionRequest,
    _: None = Depends(verify_api_key),
) -> MeritFunctionResponse:
    """
    Evaluate a merit function: construct operands in the MFE and compute.

    Loads the system from zmx_content, populates the Merit Function Editor
    with the provided operand rows, and returns computed values and contributions.
    """
    def _build_merit_response(result: dict) -> MeritFunctionResponse:
        raw_rows = result.get("evaluated_rows", [])
        evaluated = [EvaluatedOperandRow(**r) for r in raw_rows] if raw_rows else None
        row_errors = result.get("row_errors", []) if result.get("row_errors") else None
        return MeritFunctionResponse(
            success=result.get("success", False),
            error=result.get("error") if not result.get("success", False) else None,
            total_merit=result.get("total_merit"),
            evaluated_rows=evaluated,
            row_errors=row_errors,
        )

    def _call_handler():
        operand_dicts = [row.model_dump() for row in request.operand_rows]
        return zospy_handler.evaluate_merit_function(operand_dicts)

    return await _run_endpoint(
        "/evaluate-merit-function", MeritFunctionResponse, request,
        _call_handler,
        build_response=_build_merit_response,
    )


@app.post("/apply-optimization-wizard", response_model=OptimizationWizardResponse)
async def apply_optimization_wizard(
    request: OptimizationWizardRequest,
    _: None = Depends(verify_api_key),
) -> OptimizationWizardResponse:
    """
    Apply the SEQ Optimization Wizard to auto-generate merit function operands.

    Uses OpticStudio's SEQOptimizationWizard2 to populate the MFE based on
    image quality criteria, accounting for all active fields, wavelengths,
    and pupil sampling.
    """
    def _build_wizard_response(result: dict) -> OptimizationWizardResponse:
        raw_rows = result.get("generated_rows", [])
        generated = [WizardGeneratedRow(**r) for r in raw_rows] if raw_rows else None
        return OptimizationWizardResponse(
            success=result.get("success", False),
            error=result.get("error") if not result.get("success", False) else None,
            total_merit=result.get("total_merit"),
            generated_rows=generated,
            num_rows_generated=result.get("num_rows_generated", 0),
        )

    # Pass all wizard params except zmx_content (already loaded by _run_endpoint)
    wizard_params = request.model_dump(exclude={"zmx_content"})

    return await _run_endpoint(
        "/apply-optimization-wizard", OptimizationWizardResponse, request,
        lambda: zospy_handler.apply_optimization_wizard(**wizard_params),
        build_response=_build_wizard_response,
    )


class ZernikeCoefficientsRequest(BaseModel):
    """Zernike Standard Coefficients analysis request."""
    zmx_content: str = Field(description="Base64-encoded .zmx file content")
    field_index: int = Field(default=1, ge=1, description="Field index (1-indexed)")
    wavelength_index: int = Field(default=1, ge=1, description="Wavelength index (1-indexed)")
    sampling: str = Field(default="64x64", description="Pupil sampling grid (e.g., '64x64', '128x128')")
    maximum_term: int = Field(default=37, ge=1, le=231, description="Maximum Zernike term number")


class ZernikeCoefficientsDetailResponse(BaseModel):
    """Zernike Standard Coefficients response."""
    success: bool = Field(description="Whether the operation succeeded")
    coefficients: Optional[list[dict[str, Any]]] = Field(default=None, description="List of {term, value, formula} dicts")
    pv_to_chief: Optional[float] = Field(default=None, description="P-V wavefront error to chief ray (waves)")
    pv_to_centroid: Optional[float] = Field(default=None, description="P-V wavefront error to centroid (waves)")
    rms_to_chief: Optional[float] = Field(default=None, description="RMS wavefront error to chief ray (waves)")
    rms_to_centroid: Optional[float] = Field(default=None, description="RMS wavefront error to centroid (waves)")
    strehl_ratio: Optional[float] = Field(default=None, description="Strehl ratio (0-1)")
    surface: Optional[str] = Field(default=None, description="Analysis surface")
    field_x: Optional[float] = Field(default=None, description="Field X coordinate")
    field_y: Optional[float] = Field(default=None, description="Field Y coordinate")
    field_index: Optional[int] = Field(default=None, description="Field index used")
    wavelength_index: Optional[int] = Field(default=None, description="Wavelength index used")
    wavelength_um: Optional[float] = Field(default=None, description="Wavelength in micrometers")
    maximum_term: Optional[int] = Field(default=None, description="Maximum Zernike term computed")
    error: Optional[str] = Field(default=None, description="Error message if operation failed")


@app.post("/zernike-standard-coefficients", response_model=ZernikeCoefficientsDetailResponse)
async def get_zernike_standard_coefficients(
    request: ZernikeCoefficientsRequest,
    _: None = Depends(verify_api_key),
) -> ZernikeCoefficientsDetailResponse:
    """
    Get Zernike Standard Coefficients decomposition of the wavefront.

    Returns individual Zernike polynomial terms (Z1-Z37+), P-V and RMS wavefront
    error, and Strehl ratio.
    """
    return await _run_endpoint(
        "/zernike-standard-coefficients", ZernikeCoefficientsDetailResponse, request,
        lambda: zospy_handler.get_zernike_standard_coefficients(
            field_index=request.field_index,
            wavelength_index=request.wavelength_index,
            sampling=request.sampling,
            maximum_term=request.maximum_term,
        ),
    )


class ZernikeVsFieldRequest(BaseModel):
    """Zernike Coefficients vs Field analysis request."""
    zmx_content: str = Field(description="Base64-encoded .zmx file content")
    maximum_term: int = Field(default=37, ge=1, le=231, description="Maximum Zernike term number")
    wavelength_index: int = Field(default=1, ge=1, description="Wavelength index (1-indexed)")
    sampling: str = Field(default="64x64", description="Pupil sampling grid (e.g., '64x64', '128x128')")
    field_density: int = Field(default=20, ge=5, le=100, description="Number of field sample points")


class ZernikeVsFieldResponse(BaseModel):
    """Zernike Coefficients vs Field response."""
    success: bool = Field(description="Whether the operation succeeded")
    field_positions: Optional[list[float]] = Field(default=None, description="Field position values")
    coefficients: Optional[dict[str, list[float]]] = Field(default=None, description="Dict mapping term number (str) to list of coefficient values per field")
    wavelength_um: Optional[float] = Field(default=None, description="Wavelength in micrometers")
    field_unit: Optional[str] = Field(default=None, description="Field coordinate unit (e.g. deg, mm)")
    error: Optional[str] = Field(default=None, description="Error message if operation failed")


@app.post("/zernike-vs-field", response_model=ZernikeVsFieldResponse)
async def get_zernike_vs_field(
    request: ZernikeVsFieldRequest,
    _: None = Depends(verify_api_key),
) -> ZernikeVsFieldResponse:
    """
    Get Zernike Coefficients vs Field analysis.

    Returns how each Zernike polynomial coefficient varies across field positions.
    Critical for understanding field-dependent aberrations.
    """
    return await _run_endpoint(
        "/zernike-vs-field", ZernikeVsFieldResponse, request,
        lambda: zospy_handler.get_zernike_vs_field(
            maximum_term=request.maximum_term,
            wavelength_index=request.wavelength_index,
            sampling=request.sampling,
            field_density=request.field_density,
        ),
    )


class RmsVsFieldRequest(BaseModel):
    """RMS vs Field analysis request."""
    zmx_content: str = Field(description="Base64-encoded .zmx file content")
    ray_density: int = Field(default=5, ge=1, le=20, description="Ray density (1-20)")
    num_field_points: int = Field(default=20, ge=3, le=256, description="Number of field points (snapped to nearest FieldDensity enum)")
    reference: str = Field(default="centroid", description="Reference point: 'centroid' or 'chief_ray'")
    wavelength_index: int = Field(default=1, ge=1, description="Wavelength index (1-indexed)")


class RmsVsFieldDataPoint(BaseModel):
    """Single data point in RMS vs Field result."""
    field_value: float = Field(description="Field coordinate value")
    rms_radius_um: float = Field(description="RMS spot radius in micrometers")


class RmsVsFieldResponse(BaseModel):
    """RMS vs Field analysis response."""
    success: bool = Field(description="Whether the operation succeeded")
    data: Optional[list[RmsVsFieldDataPoint]] = Field(default=None, description="RMS vs field data points")
    diffraction_limit: Optional[list[RmsVsFieldDataPoint]] = Field(default=None, description="Diffraction limit curve")
    wavelength_um: Optional[float] = Field(default=None, description="Wavelength in micrometers")
    field_unit: Optional[str] = Field(default=None, description="Field coordinate unit (e.g. deg, mm)")
    error: Optional[str] = Field(default=None, description="Error message if operation failed")


@app.post("/rms-vs-field", response_model=RmsVsFieldResponse)
async def get_rms_vs_field(
    request: RmsVsFieldRequest,
    _: None = Depends(verify_api_key),
) -> RmsVsFieldResponse:
    """
    Get RMS spot radius vs field using native RmsField analysis.

    Auto-samples across the full field range, producing a smooth curve.
    """
    return await _run_endpoint(
        "/rms-vs-field", RmsVsFieldResponse, request,
        lambda: zospy_handler.get_rms_vs_field(
            ray_density=request.ray_density,
            num_field_points=request.num_field_points,
            reference=request.reference,
            wavelength_index=request.wavelength_index,
        ),
    )


@app.post("/mtf", response_model=MTFResponse)
async def get_mtf(
    request: MTFRequest,
    _: None = Depends(verify_api_key),
) -> MTFResponse:
    """
    Get MTF (Modulation Transfer Function) data using FFT MTF analysis.

    Returns raw frequency/modulation data. Image rendering happens on Mac side.
    """
    return await _run_endpoint(
        "/mtf", MTFResponse, request,
        lambda: zospy_handler.get_mtf(
            field_index=request.field_index,
            wavelength_index=request.wavelength_index,
            sampling=request.sampling,
            maximum_frequency=request.maximum_frequency,
        ),
    )



@app.post("/ray-fan", response_model=RayFanResponse)
async def get_ray_fan(
    request: RayFanRequest,
    _: None = Depends(verify_api_key),
) -> RayFanResponse:
    """
    Get Ray Fan (Ray Aberration) data.

    Returns raw pupil/aberration data for tangential and sagittal fans.
    Image rendering happens on the Mac analysis service side.
    """
    return await _run_endpoint(
        "/ray-fan", RayFanResponse, request,
        lambda: zospy_handler.get_ray_fan(
            field_index=request.field_index,
            wavelength_index=request.wavelength_index,
            plot_scale=request.plot_scale,
            number_of_rays=request.number_of_rays,
        ),
    )


@app.post("/huygens-mtf", response_model=MTFResponse)
async def get_huygens_mtf(
    request: HuygensMTFRequest,
    _: None = Depends(verify_api_key),
) -> MTFResponse:
    """
    Get Huygens MTF data. More accurate than FFT MTF for systems with
    significant aberrations or tilted/decentered elements.

    Returns raw frequency/modulation data. Image rendering happens on Mac side.
    """
    return await _run_endpoint(
        "/huygens-mtf", MTFResponse, request,
        lambda: zospy_handler.get_huygens_mtf(
            field_index=request.field_index,
            wavelength_index=request.wavelength_index,
            sampling=request.sampling,
            maximum_frequency=request.maximum_frequency,
        ),
    )


class ThroughFocusMTFRequest(BaseModel):
    """Through Focus MTF analysis request."""
    zmx_content: str = Field(description="Base64-encoded .zmx file content")
    field_index: int = Field(default=0, ge=0, description="Field index (0 = all fields, 1+ = specific field, 1-indexed)")
    wavelength_index: int = Field(default=1, ge=1, description="Wavelength index (1-indexed)")
    sampling: str = Field(default="64x64", description="Pupil sampling grid")
    delta_focus: float = Field(default=0.1, gt=0, description="Focus step size in mm")
    frequency: float = Field(default=0.0, ge=0, description="Spatial frequency (cycles/mm). 0 = default.")
    number_of_steps: int = Field(default=5, ge=1, le=50, description="Number of steps in each direction from focus")


class ThroughFocusMTFFieldData(BaseModel):
    """Through Focus MTF data for a single field point."""
    field_index: int = Field(description="0-indexed field number")
    field_x: float = Field(description="Field X coordinate")
    field_y: float = Field(description="Field Y coordinate")
    tangential: list[float] = Field(default_factory=list, description="Tangential MTF values at each focus position")
    sagittal: list[float] = Field(default_factory=list, description="Sagittal MTF values at each focus position")


class BestFocusData(BaseModel):
    """Best focus position data."""
    position: float = Field(description="Best focus position (mm)")
    mtf_value: float = Field(description="MTF value at best focus")


class ThroughFocusMTFResponse(BaseModel):
    """Through Focus MTF analysis response."""
    success: bool = Field(description="Whether the operation succeeded")
    focus_positions: Optional[list[float]] = Field(default=None, description="Defocus positions (mm)")
    fields: Optional[list[ThroughFocusMTFFieldData]] = Field(default=None, description="Per-field MTF data")
    best_focus: Optional[BestFocusData] = Field(default=None, description="Best focus position and MTF value")
    frequency: Optional[float] = Field(default=None, description="Spatial frequency (cycles/mm)")
    wavelength_um: Optional[float] = Field(default=None, description="Wavelength in micrometers")
    delta_focus: Optional[float] = Field(default=None, description="Focus step size (mm)")
    error: Optional[str] = Field(default=None, description="Error message if operation failed")


@app.post("/through-focus-mtf", response_model=ThroughFocusMTFResponse)
async def get_through_focus_mtf(
    request: ThroughFocusMTFRequest,
    _: None = Depends(verify_api_key),
) -> ThroughFocusMTFResponse:
    """
    Get Through Focus MTF data using FFT Through Focus MTF analysis.

    Shows how MTF varies at different focus positions. Returns raw data;
    image rendering happens on Mac side.
    """
    return await _run_endpoint(
        "/through-focus-mtf", ThroughFocusMTFResponse, request,
        lambda: zospy_handler.get_through_focus_mtf(
            sampling=request.sampling,
            delta_focus=request.delta_focus,
            frequency=request.frequency,
            number_of_steps=request.number_of_steps,
            field_index=request.field_index,
            wavelength_index=request.wavelength_index,
        ),
    )


@app.post("/psf", response_model=PSFResponse)
async def get_psf(
    request: PSFRequest,
    _: None = Depends(verify_api_key),
) -> PSFResponse:
    """
    Get PSF (Point Spread Function) data using FFT PSF analysis.

    Returns raw 2D intensity grid as numpy array. Image rendering happens on Mac side.
    """
    return await _run_endpoint(
        "/psf", PSFResponse, request,
        lambda: zospy_handler.get_psf(
            field_index=request.field_index,
            wavelength_index=request.wavelength_index,
            sampling=request.sampling,
        ),
    )


@app.post("/geometric-image-analysis", response_model=GeometricImageResponse)
async def get_geometric_image_analysis(
    request: GeometricImageRequest,
    _: None = Depends(verify_api_key),
) -> GeometricImageResponse:
    """
    Run Geometric Image Analysis to simulate how an extended scene looks
    through the optical system.

    Returns raw 2D intensity grid as numpy array. Image rendering happens on Mac side.
    """
    # Parse wavelength: could be "All" or a number string
    wavelength: str | int = request.wavelength
    try:
        wavelength = int(request.wavelength)
    except (ValueError, TypeError):
        pass

    return await _run_endpoint(
        "/geometric-image-analysis", GeometricImageResponse, request,
        lambda: zospy_handler.get_geometric_image_analysis(
            field_size=request.field_size,
            image_size=request.image_size,
            rays_x_1000=request.rays_x_1000,
            number_of_pixels=request.number_of_pixels,
            field=request.field,
            wavelength=wavelength,
        ),
    )


class HuygensPSFRequest(BaseModel):
    """Huygens PSF analysis request."""
    zmx_content: str = Field(description="Base64-encoded .zmx file content")
    field_index: int = Field(default=1, ge=1, description="Field index (1-indexed)")
    wavelength_index: int = Field(default=1, ge=1, description="Wavelength index (1-indexed)")
    sampling: str = Field(default="64x64", description="Pupil sampling grid")


@app.post("/huygens-psf", response_model=PSFResponse)
async def get_huygens_psf(
    request: HuygensPSFRequest,
    _: None = Depends(verify_api_key),
) -> PSFResponse:
    """
    Get Huygens PSF (Point Spread Function) data.

    More accurate than FFT PSF for highly aberrated systems.
    Returns raw 2D intensity grid as numpy array. Image rendering happens on Mac side.
    """
    return await _run_endpoint(
        "/huygens-psf", PSFResponse, request,
        lambda: zospy_handler.get_huygens_psf(
            field_index=request.field_index,
            wavelength_index=request.wavelength_index,
            sampling=request.sampling,
        ),
    )


@app.post("/paraxial", response_model=ParaxialResponse)
async def get_paraxial(request: SystemRequest, _: None = Depends(verify_api_key)) -> ParaxialResponse:
    """Get comprehensive first-order (paraxial) optical properties."""
    return await _run_endpoint(
        "/paraxial", ParaxialResponse, request,
        lambda: zospy_handler.get_paraxial(),
    )


@app.post("/surface-data-report", response_model=SurfaceDataReportResponse)
async def get_surface_data_report(
    request: SystemRequest,
    _: None = Depends(verify_api_key),
) -> SurfaceDataReportResponse:
    """
    Get Surface Data Report for every surface in the system.

    Returns per-surface: edge thickness, center thickness, material,
    refractive index, and surface power. Essential for manufacturability checks.
    """
    return await _run_endpoint(
        "/surface-data-report", SurfaceDataReportResponse, request,
        lambda: zospy_handler.get_surface_data_report(),
    )


class CardinalPointEntry(BaseModel):
    """Single cardinal point entry."""
    name: str = Field(description="Cardinal point name (e.g. 'Focal Length (Object)')")
    value: Optional[float] = Field(default=None, description="Cardinal point value")
    units: str = Field(default="mm", description="Units")


class CardinalPointsResponse(BaseModel):
    """Cardinal points analysis response."""
    success: bool = Field(description="Whether the operation succeeded")
    cardinal_points: Optional[list[CardinalPointEntry]] = Field(default=None, description="List of cardinal point entries")
    starting_surface: Optional[int] = Field(default=None, description="Starting surface number")
    ending_surface: Optional[int] = Field(default=None, description="Ending surface number")
    wavelength: Optional[float] = Field(default=None, description="Analysis wavelength")
    orientation: Optional[str] = Field(default=None, description="Analysis orientation (Y-Z or X-Z)")
    lens_units: Optional[str] = Field(default=None, description="Lens units (e.g. mm)")
    error: Optional[str] = Field(default=None, description="Error message if failed")


@app.post("/cardinal-points", response_model=CardinalPointsResponse)
async def get_cardinal_points(request: SystemRequest, _: None = Depends(verify_api_key)) -> CardinalPointsResponse:
    """Get cardinal points of the optical system."""
    return await _run_endpoint(
        "/cardinal-points", CardinalPointsResponse, request,
        lambda: zospy_handler.get_cardinal_points(),
    )


class RunOptimizationRequest(BaseModel):
    """Request to run OpticStudio optimization."""
    zmx_content: str = Field(description="Base64-encoded .zmx file content")
    algorithm: Literal["DLS", "Hammer"] = Field(default="DLS", description="Optimization algorithm: DLS (damped least squares) or Hammer")
    cycles: int = Field(default=5, ge=1, le=50, description="Number of automatic optimization cycles")
    operand_rows: Optional[list[MeritFunctionOperandRow]] = Field(default=None, description="Explicit MFE operand rows (mutually exclusive with setup_wizard)")
    setup_wizard: bool = Field(default=False, description="Use SEQ Optimization Wizard to populate MFE")
    wizard_params: Optional[dict[str, Any]] = Field(default=None, description="Wizard parameters when setup_wizard=True")


class VariableState(BaseModel):
    """State of a single variable parameter after optimization."""
    surface_index: int = Field(description="1-based surface index in OpticStudio LDE (surface 1 = first optical surface)")
    parameter: str = Field(description="Parameter name: radius, thickness, or conic")
    value: float = Field(description="Current value after optimization")
    is_variable: bool = Field(default=True, description="Whether the parameter is marked as variable")


class RunOptimizationResponse(BaseModel):
    """Response from optimization run."""
    success: bool = Field(description="Whether the optimization succeeded")
    merit_before: Optional[float] = Field(default=None, description="Merit function value before optimization")
    merit_after: Optional[float] = Field(default=None, description="Merit function value after optimization")
    cycles_completed: Optional[int] = Field(default=None, description="Number of optimization cycles completed")
    operand_results: Optional[list[dict[str, Any]]] = Field(default=None, description="Per-operand results after optimization")
    variable_states: Optional[list[VariableState]] = Field(default=None, description="Variable parameter states after optimization")
    error: Optional[str] = Field(default=None, description="Error message if failed")


@app.post("/run-optimization", response_model=RunOptimizationResponse)
async def run_optimization(
    request: RunOptimizationRequest,
    _: None = Depends(verify_api_key),
) -> RunOptimizationResponse:
    """
    Run OpticStudio's DLS or Hammer optimization for N cycles.

    Loads the system from zmx_content, populates the MFE (via explicit rows
    or wizard), runs the optimizer, and returns before/after merit values
    plus variable states from the LDE.
    """
    def _build_response(result: dict) -> RunOptimizationResponse:
        if not result.get("success"):
            return RunOptimizationResponse(
                success=False,
                error=result.get("error", "Optimization failed"),
            )

        variable_states = None
        raw_states = result.get("variable_states")
        if raw_states:
            variable_states = [VariableState(**vs) for vs in raw_states]

        return RunOptimizationResponse(
            success=True,
            merit_before=result.get("merit_before"),
            merit_after=result.get("merit_after"),
            cycles_completed=result.get("cycles_completed"),
            operand_results=result.get("operand_results"),
            variable_states=variable_states,
        )

    def _call_handler():
        operand_dicts = None
        if request.operand_rows:
            operand_dicts = [row.model_dump() for row in request.operand_rows]
        return zospy_handler.run_optimization(
            algorithm=request.algorithm,
            cycles=request.cycles,
            operand_rows=operand_dicts,
            setup_wizard=request.setup_wizard,
            wizard_params=request.wizard_params,
        )

    return await _run_endpoint(
        "/run-optimization", RunOptimizationResponse, request,
        _call_handler,
        build_response=_build_response,
    )


class OperandParameterInfo(BaseModel):
    """Metadata for a single operand parameter column."""
    column: str = Field(description="Column name: Comment, Param1-Param8")
    header: str = Field(description="Column header label from OpticStudio")
    data_type: str = Field(description="Cell data type as string")
    is_active: bool = Field(description="Whether this parameter column is active")
    is_read_only: bool = Field(description="Whether this parameter column is read-only")
    default_value: Optional[float | int | str] = Field(default=None, description="Default value after ChangeType()")


class OperandCatalogEntry(BaseModel):
    """Metadata for a single operand type."""
    code: str = Field(description="Operand code (e.g. 'EFFL')")
    type_name: str = Field(default="", description="Human-readable operand description")
    parameters: list[OperandParameterInfo] = Field(default_factory=list)


class OperandCatalogResponse(BaseModel):
    """Response from operand catalog discovery."""
    success: bool = Field(description="Whether the operation succeeded")
    operands: list[OperandCatalogEntry] = Field(default_factory=list)
    total_count: int = Field(default=0, description="Total number of operands discovered")
    error: Optional[str] = Field(default=None, description="Error message if failed")


@app.post("/operand-catalog", response_model=OperandCatalogResponse)
async def get_operand_catalog(
    _: None = Depends(verify_api_key),
) -> OperandCatalogResponse:
    """
    Discover all supported merit function operand types and their parameter metadata.

    No ZMX file needed -- only requires an OpticStudio connection.
    """
    with timed_operation(logger, "/operand-catalog"):
        async with timed_lock_acquire(_zospy_lock, logger, name="zospy"):
            if await _ensure_connected() is None:
                return OperandCatalogResponse(success=False, error=_not_connected_error())

            try:
                result = zospy_handler.get_operand_catalog()

                if not result.get("success"):
                    return OperandCatalogResponse(
                        success=False,
                        error=result.get("error", "/operand-catalog failed"),
                    )

                # Filter to known model fields, matching _run_endpoint pattern
                model_fields = set(OperandCatalogResponse.model_fields.keys())
                return OperandCatalogResponse(success=True, **{
                    k: v for k, v in result.items()
                    if k not in ("success", "error") and k in model_fields
                })
            except Exception as e:
                await _handle_zospy_error("/operand-catalog", e)
                return OperandCatalogResponse(success=False, error=str(e))


class SurfaceCurvatureRequest(BaseModel):
    """Surface Curvature analysis request."""
    zmx_content: str = Field(description="Base64-encoded .zmx file content")
    surface: int = Field(default=1, ge=1, description="Surface number (1-indexed)")
    sampling: str = Field(default="65x65", description="Grid sampling resolution (e.g., '65x65', '129x129')")
    show_as: str = Field(default="Surface", description="Display format: Surface, Contour, GreyScale, etc.")
    data: str = Field(default="TangentialCurvature", description="Curvature data type: TangentialCurvature, SagittalCurvature, X_Curvature, Y_Curvature")
    remove: str = Field(default="None_", description="Removal option: None_, BaseROC, BestFitSphere")


class SurfaceCurvatureResponse(BaseModel):
    """Surface Curvature analysis response."""
    success: bool = Field(description="Whether the operation succeeded")
    image: Optional[str] = Field(default=None, description="Base64-encoded numpy array bytes")
    image_format: Optional[str] = Field(default=None, description="Image format: 'numpy_array'")
    array_shape: Optional[list[int]] = Field(default=None, description="Shape for numpy array reconstruction")
    array_dtype: Optional[str] = Field(default=None, description="Dtype for numpy array reconstruction")
    min_curvature: Optional[float] = Field(default=None, description="Minimum curvature value")
    max_curvature: Optional[float] = Field(default=None, description="Maximum curvature value")
    mean_curvature: Optional[float] = Field(default=None, description="Mean curvature value")
    surface_number: Optional[int] = Field(default=None, description="Surface number analyzed")
    data_type: Optional[str] = Field(default=None, description="Curvature data type used")
    error: Optional[str] = Field(default=None, description="Error message if operation failed")


@app.post("/surface-curvature", response_model=SurfaceCurvatureResponse)
async def get_surface_curvature(
    request: SurfaceCurvatureRequest,
    _: None = Depends(verify_api_key),
) -> SurfaceCurvatureResponse:
    """
    Get surface curvature map for a specific surface.

    Returns raw curvature grid data as a numpy array. Image rendering
    happens on the Mac analysis service side.
    """
    return await _run_endpoint(
        "/surface-curvature", SurfaceCurvatureResponse, request,
        lambda: zospy_handler.get_surface_curvature(
            surface=request.surface,
            sampling=request.sampling,
            show_as=request.show_as,
            data=request.data,
            remove=request.remove,
        ),
    )


# =============================================================================
# Physical Optics Propagation
# =============================================================================


class PhysicalOpticsPropagationRequest(BaseModel):
    """Physical Optics Propagation (POP) analysis request."""
    zmx_content: str = Field(description="Base64-encoded .zmx file content")
    field_index: int = Field(default=1, ge=1, description="Field index (1-indexed)")
    wavelength_index: int = Field(default=1, ge=1, description="Wavelength index (1-indexed)")
    beam_type: str = Field(default="GaussianWaist", description="Beam type (GaussianWaist, GaussianAngle, TopHat, etc.)")
    waist_x: Optional[float] = Field(default=None, description="Beam waist X (mm)")
    waist_y: Optional[float] = Field(default=None, description="Beam waist Y (mm)")
    x_sampling: int = Field(default=64, description="X sampling points (power of 2)")
    y_sampling: int = Field(default=64, description="Y sampling points (power of 2)")
    x_width: float = Field(default=4.0, description="X display width (mm)")
    y_width: float = Field(default=4.0, description="Y display width (mm)")
    start_surface: int = Field(default=1, description="Start surface (1-indexed)")
    end_surface: str = Field(default="Image", description="End surface ('Image' or surface index)")
    use_polarization: bool = Field(default=False, description="Use polarization")
    data_type: str = Field(default="Irradiance", description="Data type: Irradiance or Phase")


class PhysicalOpticsPropagationResponse(BaseModel):
    """Physical Optics Propagation (POP) analysis response."""
    success: bool = Field(description="Whether the operation succeeded")
    image: Optional[str] = Field(default=None, description="Base64-encoded numpy array bytes")
    image_format: Optional[str] = Field(default=None, description="Image format: 'numpy_array'")
    array_shape: Optional[list[int]] = Field(default=None, description="Shape for numpy array reconstruction")
    array_dtype: Optional[str] = Field(default=None, description="Dtype for numpy array reconstruction")
    beam_params: Optional[dict[str, Any]] = Field(default=None, description="Propagated beam parameters")
    wavelength_um: Optional[float] = Field(default=None, description="Wavelength in micrometers")
    field_x: Optional[float] = Field(default=None, description="Field X coordinate")
    field_y: Optional[float] = Field(default=None, description="Field Y coordinate")
    data_type: Optional[str] = Field(default=None, description="Data type used (Irradiance, Phase)")
    error: Optional[str] = Field(default=None, description="Error message if operation failed")


@app.post("/physical-optics-propagation", response_model=PhysicalOpticsPropagationResponse)
async def get_physical_optics_propagation(
    request: PhysicalOpticsPropagationRequest,
    _: None = Depends(verify_api_key),
) -> PhysicalOpticsPropagationResponse:
    """
    Run Physical Optics Propagation (POP) analysis.

    Returns raw 2D beam profile as numpy array. Image rendering happens on Mac side.
    """
    # Parse end_surface: if numeric string, convert to int for the handler
    end_surface = request.end_surface
    try:
        end_surface_val = int(end_surface)
    except (ValueError, TypeError):
        end_surface_val = end_surface

    return await _run_endpoint(
        "/physical-optics-propagation", PhysicalOpticsPropagationResponse, request,
        lambda: zospy_handler.get_physical_optics_propagation(
            field_index=request.field_index,
            wavelength_index=request.wavelength_index,
            beam_type=request.beam_type,
            waist_x=request.waist_x,
            waist_y=request.waist_y,
            x_sampling=request.x_sampling,
            y_sampling=request.y_sampling,
            x_width=request.x_width,
            y_width=request.y_width,
            start_surface=request.start_surface,
            end_surface=end_surface_val,
            use_polarization=request.use_polarization,
            data_type=request.data_type,
        ),
    )


# =============================================================================
# Polarization Analyses
# =============================================================================


class PolarizationPupilMapRequest(BaseModel):
    """Polarization Pupil Map analysis request."""
    zmx_content: str = Field(description="Base64-encoded .zmx file content")
    field_index: int = Field(default=1, ge=1, description="Field index (1-indexed)")
    wavelength_index: int = Field(default=1, ge=1, description="Wavelength index (1-indexed)")
    surface: str = Field(default="Image", description="Target surface ('Image' or integer)")
    sampling: str = Field(default="11x11", description="Pupil sampling grid")
    jx: float = Field(default=1.0, description="Jones vector X-component")
    jy: float = Field(default=0.0, description="Jones vector Y-component")
    x_phase: float = Field(default=0.0, description="X-component phase (degrees)")
    y_phase: float = Field(default=0.0, description="Y-component phase (degrees)")


class PolarizationPupilMapResponse(BaseModel):
    """Polarization Pupil Map analysis response."""
    success: bool = Field(description="Whether the operation succeeded")
    pupil_map: Optional[list[list[float]]] = Field(default=None, description="2D pupil map data rows")
    pupil_map_columns: Optional[list[str]] = Field(default=None, description="Column names for the pupil map")
    pupil_map_shape: Optional[list[int]] = Field(default=None, description="Shape of the pupil map [rows, cols]")
    transmission: Optional[float] = Field(default=None, description="Total transmission")
    x_field: Optional[float] = Field(default=None, description="Resulting X electric field")
    y_field: Optional[float] = Field(default=None, description="Resulting Y electric field")
    x_phase: Optional[float] = Field(default=None, description="Resulting X phase (degrees)")
    y_phase: Optional[float] = Field(default=None, description="Resulting Y phase (degrees)")
    field_x: Optional[float] = Field(default=None, description="Field X coordinate")
    field_y: Optional[float] = Field(default=None, description="Field Y coordinate")
    field_index: Optional[int] = Field(default=None, description="Field index used")
    wavelength_index: Optional[int] = Field(default=None, description="Wavelength index used")
    wavelength_um: Optional[float] = Field(default=None, description="Wavelength in micrometers")
    surface: Optional[str] = Field(default=None, description="Surface analyzed")
    sampling: Optional[str] = Field(default=None, description="Sampling grid used")
    jx: Optional[float] = Field(default=None, description="Input Jones vector X")
    jy: Optional[float] = Field(default=None, description="Input Jones vector Y")
    input_x_phase: Optional[float] = Field(default=None, description="Input X phase")
    input_y_phase: Optional[float] = Field(default=None, description="Input Y phase")
    error: Optional[str] = Field(default=None, description="Error message if operation failed")


@app.post("/polarization-pupil-map", response_model=PolarizationPupilMapResponse)
async def get_polarization_pupil_map(
    request: PolarizationPupilMapRequest,
    _: None = Depends(verify_api_key),
) -> PolarizationPupilMapResponse:
    """Get Polarization Pupil Map showing polarization state across the pupil."""
    return await _run_endpoint(
        "/polarization-pupil-map", PolarizationPupilMapResponse, request,
        lambda: zospy_handler.get_polarization_pupil_map(
            field_index=request.field_index,
            wavelength_index=request.wavelength_index,
            surface=request.surface,
            sampling=request.sampling,
            jx=request.jx,
            jy=request.jy,
            x_phase=request.x_phase,
            y_phase=request.y_phase,
        ),
    )


class PolarizationTransmissionRequest(BaseModel):
    """Polarization Transmission analysis request."""
    zmx_content: str = Field(description="Base64-encoded .zmx file content")
    sampling: str = Field(default="32x32", description="Pupil sampling grid")
    unpolarized: bool = Field(default=False, description="Use unpolarized light")
    jx: float = Field(default=1.0, description="Jones vector X-component")
    jy: float = Field(default=0.0, description="Jones vector Y-component")
    x_phase: float = Field(default=0.0, description="X-component phase (degrees)")
    y_phase: float = Field(default=0.0, description="Y-component phase (degrees)")


class PolarizationTransmissionResponse(BaseModel):
    """Polarization Transmission analysis response."""
    success: bool = Field(description="Whether the operation succeeded")
    field_transmissions: Optional[list[dict]] = Field(default=None, description="Per-field transmission data")
    chief_ray_transmissions: Optional[list[dict]] = Field(default=None, description="Per-field chief ray transmission data")
    x_field: Optional[float] = Field(default=None, description="Resulting X electric field")
    y_field: Optional[float] = Field(default=None, description="Resulting Y electric field")
    x_phase: Optional[float] = Field(default=None, description="Resulting X phase")
    y_phase: Optional[float] = Field(default=None, description="Resulting Y phase")
    grid_size: Optional[str] = Field(default=None, description="Grid size used")
    num_fields: Optional[int] = Field(default=None, description="Number of fields in system")
    num_wavelengths: Optional[int] = Field(default=None, description="Number of wavelengths in system")
    field_info: Optional[list[dict]] = Field(default=None, description="Field position info")
    wavelength_info: Optional[list[dict]] = Field(default=None, description="Wavelength info")
    unpolarized: Optional[bool] = Field(default=None, description="Whether unpolarized mode was used")
    jx: Optional[float] = Field(default=None, description="Input Jones vector X")
    jy: Optional[float] = Field(default=None, description="Input Jones vector Y")
    input_x_phase: Optional[float] = Field(default=None, description="Input X phase")
    input_y_phase: Optional[float] = Field(default=None, description="Input Y phase")
    error: Optional[str] = Field(default=None, description="Error message if operation failed")


@app.post("/polarization-transmission", response_model=PolarizationTransmissionResponse)
async def get_polarization_transmission(
    request: PolarizationTransmissionRequest,
    _: None = Depends(verify_api_key),
) -> PolarizationTransmissionResponse:
    """Get Polarization Transmission showing transmission vs field with polarization effects."""
    return await _run_endpoint(
        "/polarization-transmission", PolarizationTransmissionResponse, request,
        lambda: zospy_handler.get_polarization_transmission(
            sampling=request.sampling,
            unpolarized=request.unpolarized,
            jx=request.jx,
            jy=request.jy,
            x_phase=request.x_phase,
            y_phase=request.y_phase,
        ),
    )


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

    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        workers=num_workers,
        reload=dev_mode,
        log_level="info",
    )
