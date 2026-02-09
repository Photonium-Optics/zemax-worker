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
from contextlib import asynccontextmanager
from typing import Any, Callable, Literal, Optional

from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from zospy_handler import ZosPyHandler, ZosPyError
from utils.timing import timed_operation, timed_lock_acquire
from log_buffer import log_buffer

# Configure logging
logging.basicConfig(level=logging.INFO)
logging.getLogger().addHandler(log_buffer)
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

# Async lock - serializes ZosPy operations within this process
# Each uvicorn worker process has its own lock and OpticStudio connection
_zospy_lock = asyncio.Lock()


def _init_zospy() -> Optional[ZosPyHandler]:
    """Initialize ZosPy connection with error handling."""
    global _last_connection_error
    try:
        handler = ZosPyHandler()
        logger.info("ZosPy connection established.")
        _last_connection_error = None
        return handler
    except Exception as e:
        logger.error(f"Failed to initialize ZosPy: {e}")
        _last_connection_error = str(e)
        return None


def _reconnect_zospy() -> Optional[ZosPyHandler]:
    """Attempt to reconnect to OpticStudio after a failure."""
    global zospy_handler

    # Clean up existing connection if any
    if zospy_handler:
        try:
            zospy_handler.close()
        except Exception:
            pass

    logger.info("Attempting to reconnect to OpticStudio...")
    zospy_handler = _init_zospy()
    return zospy_handler


def _ensure_connected() -> Optional[ZosPyHandler]:
    """
    Ensure ZosPy is connected, attempting reconnection if needed.

    Returns:
        ZosPyHandler if connected, None if connection failed.
    """
    global zospy_handler
    if zospy_handler is None:
        zospy_handler = _reconnect_zospy()
    return zospy_handler


def _handle_zospy_error(operation_name: str, error: Exception) -> None:
    """
    Handle ZosPy errors by logging and attempting reconnection.

    Args:
        operation_name: Name of the operation that failed (for logging)
        error: The exception that was raised
    """
    global zospy_handler
    if isinstance(error, ZosPyError):
        logger.error(f"{operation_name} failed: {error}")
        zospy_handler = _reconnect_zospy()
    else:
        logger.error(f"{operation_name} unexpected error: {error}")


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
            if _ensure_connected() is None:
                error_msg = f"{NOT_CONNECTED_ERROR}: {_last_connection_error}" if _last_connection_error else NOT_CONNECTED_ERROR
                return response_cls(success=False, error=error_msg)

            try:
                _load_system_from_request(request)
                result = handler()

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
                _handle_zospy_error(endpoint_name, e)
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
    reference: str = Field(default="chief_ray", description="Reference point: 'chief_ray' or 'centroid'")


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
    criterion: Literal["Spot", "Wavefront", "Contrast"] = Field(default="Spot", description="Optimization criterion: Spot, Wavefront, or Contrast")
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
        if _zospy_lock.locked():
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
            if _ensure_connected() is None:
                error_msg = f"{NOT_CONNECTED_ERROR}: {_last_connection_error}" if _last_connection_error else NOT_CONNECTED_ERROR
                return LoadSystemResponse(success=False, error=error_msg)

            try:
                result = _load_system_from_request(request)
                return LoadSystemResponse(
                    success=True,
                    num_surfaces=result.get("num_surfaces"),
                    efl=result.get("efl"),
                )
            except Exception as e:
                _handle_zospy_error("Load system", e)
                return LoadSystemResponse(success=False, error=str(e))


@app.post("/cross-section", response_model=CrossSectionResponse)
async def get_cross_section(request: SystemRequest, _: None = Depends(verify_api_key)) -> CrossSectionResponse:
    """Generate cross-section diagram using ZosPy's CrossSection analysis."""
    return await _run_endpoint(
        "/cross-section", CrossSectionResponse, request,
        lambda: zospy_handler.get_cross_section(),
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


@app.post("/paraxial", response_model=ParaxialResponse)
async def get_paraxial(request: SystemRequest, _: None = Depends(verify_api_key)) -> ParaxialResponse:
    """Get comprehensive first-order (paraxial) optical properties."""
    return await _run_endpoint(
        "/paraxial", ParaxialResponse, request,
        lambda: zospy_handler.get_paraxial(),
    )


class OperandParameterInfo(BaseModel):
    """Metadata for a single operand parameter column."""
    column: str = Field(description="Column name: Comment, Param1-Param8")
    header: str = Field(description="Column header label from OpticStudio")
    data_type: str = Field(description="Cell data type as string")
    is_active: bool = Field(description="Whether this parameter column is active")
    is_read_only: bool = Field(description="Whether this parameter column is read-only")


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
            if _ensure_connected() is None:
                error_msg = f"{NOT_CONNECTED_ERROR}: {_last_connection_error}" if _last_connection_error else NOT_CONNECTED_ERROR
                return OperandCatalogResponse(success=False, error=error_msg)

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
                _handle_zospy_error("/operand-catalog", e)
                return OperandCatalogResponse(success=False, error=str(e))


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
