"""Core router – health, cross-section, semi-diameters, ray-trace diagnostic, ray analysis."""

import asyncio

from fastapi import APIRouter, Depends

import main
from models import (
    HealthResponse, CrossSectionRequest, CrossSectionResponse,
    SystemRequest, SemiDiametersResponse,
    RayTraceDiagnosticRequest, RayTraceDiagnosticResponse,
    RayAnalysisRequest, RayAnalysisResponse,
)
from config import WORKER_COUNT

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Health check endpoint.

    Returns the status of the worker and OpticStudio connection.
    Acquires _zospy_lock with a 2-second timeout to avoid reading
    zospy_handler during reconnection or other mutations.
    """
    lock_acquired = False
    try:
        await asyncio.wait_for(main._zospy_lock.acquire(), timeout=2.0)
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
            main._zospy_lock.release()
        raise

    try:
        if main.zospy_handler is None:
            return HealthResponse(
                success=True,  # Worker is running
                opticstudio_connected=False,
                version=None,
                zospy_version=None,
                worker_count=WORKER_COUNT,
                connection_error=main._last_connection_error,
            )

        try:
            status = main.zospy_handler.get_status()
            return HealthResponse(
                success=True,
                opticstudio_connected=status.get("connected", False),
                version=status.get("opticstudio_version"),
                zospy_version=status.get("zospy_version"),
                worker_count=WORKER_COUNT,
            )
        except Exception as e:
            main.logger.error(f"Health check failed: {e}")
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
            main._zospy_lock.release()


@router.post("/cross-section", response_model=CrossSectionResponse)
async def get_cross_section(request: CrossSectionRequest, _: None = Depends(main.verify_api_key)) -> CrossSectionResponse:
    """Generate cross-section diagram using ZosPy's CrossSection analysis."""
    return await main._run_endpoint(
        "/cross-section", CrossSectionResponse, request,
        lambda: main.zospy_handler.get_cross_section(
            number_of_rays=request.number_of_rays,
            color_rays_by=request.color_rays_by,
        ),
    )


@router.post("/calc-semi-diameters", response_model=SemiDiametersResponse)
async def calc_semi_diameters(request: SystemRequest, _: None = Depends(main.verify_api_key)) -> SemiDiametersResponse:
    """Calculate semi-diameters by tracing edge rays."""
    return await main._run_endpoint(
        "/calc-semi-diameters", SemiDiametersResponse, request,
        lambda: main.zospy_handler.calc_semi_diameters(),
    )


@router.post("/ray-trace-diagnostic", response_model=RayTraceDiagnosticResponse)
async def ray_trace_diagnostic(
    request: RayTraceDiagnosticRequest,
    _: None = Depends(main.verify_api_key),
) -> RayTraceDiagnosticResponse:
    """
    Trace rays through the system and return raw per-ray results.

    This is a "dumb executor" endpoint - returns raw data only.
    All aggregation, hotspot detection, and threshold calculations
    happen on the Mac side (zemax-analysis-service).
    """
    return await main._run_endpoint(
        "/ray-trace-diagnostic", RayTraceDiagnosticResponse, request,
        lambda: main.zospy_handler.ray_trace_diagnostic(
            num_rays=request.num_rays,
            distribution=request.distribution,
        ),
    )


@router.post("/ray-analysis", response_model=RayAnalysisResponse)
async def ray_analysis(
    request: RayAnalysisRequest,
    _: None = Depends(main.verify_api_key),
) -> RayAnalysisResponse:
    """
    Unified ray analysis: combined spot diagram + diagnostic in one batch trace.

    Returns both image-plane positions (for spot plots) and ray failure data
    (for diagnostic analysis) from a single IBatchRayTrace call.
    """
    return await main._run_endpoint(
        "/ray-analysis", RayAnalysisResponse, request,
        lambda: main.zospy_handler.unified_ray_analysis(
            num_rays=request.num_rays,
            distribution=request.distribution,
            field_index=request.field_index,
            wavelength_index=request.wavelength_index,
        ),
    )
