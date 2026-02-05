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
  uvicorn main:app --host 0.0.0.0 --port 8787 --workers 1  # Single worker
  uvicorn main:app --host 0.0.0.0 --port 8787 --workers 3  # 3 parallel workers (uses 3 license seats)

On macOS, set TASK_QUEUE_WORKERS to match the number of workers here.
"""

import asyncio
import base64
import logging
import os
import tempfile
from contextlib import asynccontextmanager
from typing import Any, Callable, Optional

from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from zospy_handler import ZosPyHandler, ZosPyError
from utils.timing import timed_operation, timed_lock_acquire

# Configure logging
logging.basicConfig(level=logging.INFO)
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

# =============================================================================
# Global State
# =============================================================================

# Initialize ZosPy handler (manages OpticStudio connection)
zospy_handler: Optional[ZosPyHandler] = None

# Async lock - serializes ZosPy operations within this process
# Each uvicorn worker process has its own lock and OpticStudio connection
_zospy_lock = asyncio.Lock()


def _init_zospy() -> Optional[ZosPyHandler]:
    """Initialize ZosPy connection with error handling."""
    try:
        handler = ZosPyHandler()
        logger.info("ZosPy connection established.")
        return handler
    except Exception as e:
        logger.error(f"Failed to initialize ZosPy: {e}")
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
                return response_cls(success=False, error=NOT_CONNECTED_ERROR)

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


class SpotDiagramResponse(BaseModel):
    """
    Spot diagram analysis response.

    Returns spot diagram image and per-field spot data (RMS, GEO radius, centroid).
    This is a "dumb executor" response - Mac side handles rendering if needed.
    """
    success: bool = Field(description="Whether the operation succeeded")
    image: Optional[str] = Field(default=None, description="Base64-encoded PNG or numpy array bytes")
    image_format: Optional[str] = Field(default=None, description="Image format: 'png' or 'numpy_array'")
    array_shape: Optional[list[int]] = Field(default=None, description="Shape for numpy array reconstruction")
    array_dtype: Optional[str] = Field(default=None, description="Dtype for numpy array reconstruction")
    spot_data: Optional[list[SpotFieldData]] = Field(default=None, description="Per-field spot data")
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


# =============================================================================
# Endpoints
# =============================================================================


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Health check endpoint.

    Returns the status of the worker and OpticStudio connection.
    """
    if zospy_handler is None:
        return HealthResponse(
            success=True,  # Worker is running
            opticstudio_connected=False,
            version=None,
            zospy_version=None,
        )

    try:
        status = zospy_handler.get_status()
        return HealthResponse(
            success=True,
            opticstudio_connected=status.get("connected", False),
            version=status.get("opticstudio_version"),
            zospy_version=status.get("zospy_version"),
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            success=True,
            opticstudio_connected=False,
            version=None,
            zospy_version=None,
        )


@app.post("/load-system", response_model=LoadSystemResponse)
async def load_system(request: SystemRequest, _: None = Depends(verify_api_key)) -> LoadSystemResponse:
    """Load an optical system into OpticStudio from zmx_content."""
    with timed_operation(logger, "/load-system"):
        async with timed_lock_acquire(_zospy_lock, logger, name="zospy"):
            if _ensure_connected() is None:
                return LoadSystemResponse(success=False, error=NOT_CONNECTED_ERROR)

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
        ),
    )


@app.post("/spot-diagram", response_model=SpotDiagramResponse)
async def get_spot_diagram(
    request: SpotDiagramRequest,
    _: None = Depends(verify_api_key),
) -> SpotDiagramResponse:
    """
    Generate spot diagram using ZosPy's StandardSpot analysis.

    This is a "dumb executor" endpoint - returns raw data only.
    Spot diagram image rendering happens on Mac side if PNG export fails.
    """
    def _build_spot_response(result: dict) -> SpotDiagramResponse:
        if not result.get("success", False):
            return SpotDiagramResponse(
                success=False,
                error=result.get("error", "Spot diagram analysis failed"),
            )
        spot_data = None
        if result.get("spot_data"):
            spot_data = [SpotFieldData(**sd) for sd in result["spot_data"]]
        return SpotDiagramResponse(
            success=True,
            image=result.get("image"),
            image_format=result.get("image_format"),
            array_shape=result.get("array_shape"),
            array_dtype=result.get("array_dtype"),
            spot_data=spot_data,
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
        evaluated = [EvaluatedOperandRow(**r) for r in raw_rows] or None
        row_errors = result.get("row_errors", []) or None
        return MeritFunctionResponse(
            success=result.get("success", False),
            error=result.get("error") if not result.get("success", False) else None,
            total_merit=result.get("total_merit"),
            evaluated_rows=evaluated,
            row_errors=row_errors,
        )

    def _call_handler():
        operand_dicts = [
            {
                "operand_code": row.operand_code,
                "params": row.params,
                "target": row.target,
                "weight": row.weight,
            }
            for row in request.operand_rows
        ]
        return zospy_handler.evaluate_merit_function(operand_dicts)

    return await _run_endpoint(
        "/evaluate-merit-function", MeritFunctionResponse, request,
        _call_handler,
        build_response=_build_merit_response,
    )


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", str(DEFAULT_PORT)))
    host = os.getenv("HOST", DEFAULT_HOST)
    dev_mode = os.getenv("DEV_MODE", "false").lower() == "true"
    # Number of workers - each gets its own OpticStudio connection (uses 1 license seat each)
    # Default to 1, but can increase up to your license limit (Premium=8, Professional=4, Perpetual=2)
    num_workers = int(os.getenv("WORKERS", "1"))

    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        workers=num_workers,
        reload=dev_mode,  # Only enable reload in development
        log_level="info",
    )
