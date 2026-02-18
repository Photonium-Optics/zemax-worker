"""Geometry router – paraxial, surface data report, cardinal points, surface curvature."""

from fastapi import APIRouter, Depends

import main
from models import (
    SystemRequest, ParaxialResponse, SurfaceDataReportResponse,
    CardinalPointsResponse,
    SurfaceCurvatureRequest, SurfaceCurvatureResponse,
)

router = APIRouter()


@router.post("/paraxial", response_model=ParaxialResponse)
async def get_paraxial(request: SystemRequest, _: None = Depends(main.verify_api_key)) -> ParaxialResponse:
    """Get comprehensive first-order (paraxial) optical properties."""
    return await main._run_endpoint(
        "/paraxial", ParaxialResponse, request,
        lambda: main.zospy_handler.get_paraxial_data(),
    )


@router.post("/surface-data-report", response_model=SurfaceDataReportResponse)
async def get_surface_data_report(
    request: SystemRequest,
    _: None = Depends(main.verify_api_key),
) -> SurfaceDataReportResponse:
    """
    Get Surface Data Report for every surface in the system.

    Returns per-surface: edge thickness, center thickness, material,
    refractive index, and surface power. Essential for manufacturability checks.
    """
    return await main._run_endpoint(
        "/surface-data-report", SurfaceDataReportResponse, request,
        lambda: main.zospy_handler.get_surface_data_report(),
    )


@router.post("/cardinal-points", response_model=CardinalPointsResponse)
async def get_cardinal_points(request: SystemRequest, _: None = Depends(main.verify_api_key)) -> CardinalPointsResponse:
    """Get cardinal points of the optical system."""
    return await main._run_endpoint(
        "/cardinal-points", CardinalPointsResponse, request,
        lambda: main.zospy_handler.get_cardinal_points(),
    )


@router.post("/surface-curvature", response_model=SurfaceCurvatureResponse)
async def get_surface_curvature(
    request: SurfaceCurvatureRequest,
    _: None = Depends(main.verify_api_key),
) -> SurfaceCurvatureResponse:
    """
    Get surface curvature map for a specific surface.

    Returns raw curvature grid data as a numpy array. Image rendering
    happens on the Mac analysis service side.
    """
    return await main._run_endpoint(
        "/surface-curvature", SurfaceCurvatureResponse, request,
        lambda: main.zospy_handler.get_surface_curvature(
            surface=request.surface,
            sampling=request.sampling,
            show_as=request.show_as,
            data=request.data,
            remove=request.remove,
        ),
    )
