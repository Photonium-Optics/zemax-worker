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

CRITICAL: This worker MUST run with a single uvicorn worker (--workers 1).
ZosPy/COM requires single-threaded apartment (STA) semantics. Running with
multiple workers will cause race conditions and unpredictable failures.

Example: uvicorn main:app --host 0.0.0.0 --port 8787 --workers 1
"""

import os
import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from zospy_handler import ZosPyHandler, ZosPyError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize ZosPy handler (manages OpticStudio connection)
zospy_handler: Optional[ZosPyHandler] = None

# Thread safety lock - ZosPy/COM is single-threaded
# All ZosPy operations must be serialized
_zospy_lock = asyncio.Lock()

# API key for authentication (optional but recommended)
ZEMAX_API_KEY = os.getenv("ZEMAX_API_KEY", None)

# Error message constant
NOT_CONNECTED_ERROR = "OpticStudio not connected"


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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    global zospy_handler

    # Startup: Initialize ZosPy connection
    logger.info("Starting Zemax Worker - Connecting to OpticStudio...")
    zospy_handler = _init_zospy()
    if zospy_handler is None:
        logger.warning("OpticStudio not available at startup. Will attempt to connect on first request.")

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
    """Request containing an LLM JSON optical system."""
    system: dict[str, Any] = Field(description="LLM JSON optical system")


class HealthResponse(BaseModel):
    """Health check response."""
    success: bool = Field(description="Whether the worker is healthy")
    opticstudio_connected: bool = Field(description="Whether OpticStudio is connected")
    version: Optional[str] = Field(default=None, description="OpticStudio version")
    zospy_version: Optional[str] = Field(default=None, description="ZosPy version")


class CrossSectionResponse(BaseModel):
    """Cross-section diagram response."""
    success: bool = Field(description="Whether the operation succeeded")
    image: Optional[str] = Field(default=None, description="Base64-encoded PNG image")
    image_format: Optional[str] = Field(default=None, description="Image format (png, etc.)")
    paraxial: Optional[dict[str, Any]] = Field(default=None, description="Paraxial properties")
    surfaces: Optional[list[dict[str, Any]]] = Field(default=None, description="Surface geometry for fallback rendering")
    rays_total: Optional[int] = Field(default=None, description="Total rays traced")
    rays_through: Optional[int] = Field(default=None, description="Rays reaching image")
    error: Optional[str] = Field(default=None, description="Error message")


class RayTraceDiagnosticRequest(BaseModel):
    """Ray trace diagnostic request."""
    system: dict[str, Any] = Field(description="LLM JSON optical system")
    num_rays: int = Field(default=50, description="Number of rays per field")
    distribution: str = Field(default="hexapolar", description="Ray distribution")


class RayTraceDiagnosticResponse(BaseModel):
    """Ray trace diagnostic response."""
    success: bool = Field(description="Whether the operation succeeded")
    paraxial: Optional[dict[str, Any]] = Field(default=None)
    num_surfaces: Optional[int] = Field(default=None)
    num_fields: Optional[int] = Field(default=None)
    field_results: Optional[list[dict[str, Any]]] = Field(default=None)
    aggregate_surface_failures: Optional[list[dict[str, Any]]] = Field(default=None)
    hotspots: Optional[list[int]] = Field(default=None)
    error: Optional[str] = Field(default=None)


class SeidelResponse(BaseModel):
    """Seidel aberration response."""
    success: bool = Field(description="Whether the operation succeeded")
    seidel_coefficients: Optional[dict[str, Any]] = Field(default=None)
    per_surface: Optional[dict[str, Any]] = Field(default=None)
    chromatic: Optional[dict[str, Any]] = Field(default=None)
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
    """Load an optical system into OpticStudio."""
    async with _zospy_lock:
        if _ensure_connected() is None:
            return LoadSystemResponse(success=False, error=NOT_CONNECTED_ERROR)

        try:
            result = zospy_handler.load_system(request.system)
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
    async with _zospy_lock:
        if _ensure_connected() is None:
            return CrossSectionResponse(success=False, error=NOT_CONNECTED_ERROR)

        try:
            result = zospy_handler.get_cross_section(request.system)
            return CrossSectionResponse(
                success=True,
                image=result.get("image"),
                image_format=result.get("image_format"),
                paraxial=result.get("paraxial"),
                surfaces=result.get("surfaces"),
                rays_total=result.get("rays_total"),
                rays_through=result.get("rays_through"),
            )
        except Exception as e:
            _handle_zospy_error("Cross-section", e)
            return CrossSectionResponse(success=False, error=str(e))


@app.post("/calc-semi-diameters", response_model=SemiDiametersResponse)
async def calc_semi_diameters(request: SystemRequest, _: None = Depends(verify_api_key)) -> SemiDiametersResponse:
    """Calculate semi-diameters by tracing edge rays."""
    async with _zospy_lock:
        if _ensure_connected() is None:
            return SemiDiametersResponse(success=False, error=NOT_CONNECTED_ERROR)

        try:
            result = zospy_handler.calc_semi_diameters(request.system)
            return SemiDiametersResponse(
                success=True,
                semi_diameters=result.get("semi_diameters", []),
            )
        except Exception as e:
            _handle_zospy_error("Calc semi-diameters", e)
            return SemiDiametersResponse(success=False, error=str(e))


@app.post("/ray-trace-diagnostic", response_model=RayTraceDiagnosticResponse)
async def ray_trace_diagnostic(
    request: RayTraceDiagnosticRequest,
    _: None = Depends(verify_api_key),
) -> RayTraceDiagnosticResponse:
    """Run ray trace diagnostic using ZosPy's SingleRayTrace analysis."""
    async with _zospy_lock:
        if _ensure_connected() is None:
            return RayTraceDiagnosticResponse(success=False, error=NOT_CONNECTED_ERROR)

        try:
            result = zospy_handler.ray_trace_diagnostic(
                request.system,
                num_rays=request.num_rays,
                distribution=request.distribution,
            )
            return RayTraceDiagnosticResponse(
                success=True,
                paraxial=result.get("paraxial"),
                num_surfaces=result.get("num_surfaces"),
                num_fields=result.get("num_fields"),
                field_results=result.get("field_results"),
                aggregate_surface_failures=result.get("aggregate_surface_failures"),
                hotspots=result.get("hotspots"),
            )
        except Exception as e:
            _handle_zospy_error("Ray trace diagnostic", e)
            return RayTraceDiagnosticResponse(success=False, error=str(e))


@app.post("/seidel", response_model=SeidelResponse)
async def get_seidel(request: SystemRequest, _: None = Depends(verify_api_key)) -> SeidelResponse:
    """Get Seidel aberrations via ZosPy's Zernike analysis with conversion."""
    async with _zospy_lock:
        if _ensure_connected() is None:
            return SeidelResponse(success=False, error=NOT_CONNECTED_ERROR)

        try:
            result = zospy_handler.get_seidel(request.system)
            return SeidelResponse(
                success=True,
                seidel_coefficients=result.get("seidel_coefficients"),
                per_surface=result.get("per_surface"),
                chromatic=result.get("chromatic"),
                num_surfaces=result.get("num_surfaces", 0),
            )
        except Exception as e:
            _handle_zospy_error("Seidel", e)
            return SeidelResponse(success=False, error=str(e))


@app.post("/trace-rays", response_model=TraceRaysResponse)
async def trace_rays(
    request: SystemRequest,
    num_rays: int = 7,
    _: None = Depends(verify_api_key),
) -> TraceRaysResponse:
    """Trace rays through the system and return positions at each surface."""
    async with _zospy_lock:
        if _ensure_connected() is None:
            return TraceRaysResponse(success=False, error=NOT_CONNECTED_ERROR)

        try:
            result = zospy_handler.trace_rays(request.system, num_rays=num_rays)
            return TraceRaysResponse(
                success=True,
                num_surfaces=result.get("num_surfaces"),
                num_fields=result.get("num_fields"),
                num_wavelengths=result.get("num_wavelengths"),
                data=result.get("data"),
            )
        except Exception as e:
            _handle_zospy_error("Trace rays", e)
            return TraceRaysResponse(success=False, error=str(e))


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8787"))
    host = os.getenv("HOST", "0.0.0.0")
    dev_mode = os.getenv("DEV_MODE", "false").lower() == "true"

    # CRITICAL: Run with workers=1 for COM/STA compatibility
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        workers=1,  # Required for ZosPy/COM single-threaded apartment
        reload=dev_mode,  # Only enable reload in development
        log_level="info",
    )
