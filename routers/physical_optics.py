"""Physical optics router – POP, geometric image analysis."""

from fastapi import APIRouter, Depends

import main
from models import (
    GeometricImageRequest, GeometricImageResponse,
    PhysicalOpticsPropagationRequest, PhysicalOpticsPropagationResponse,
)

router = APIRouter()


@router.post("/geometric-image-analysis", response_model=GeometricImageResponse)
async def get_geometric_image_analysis(
    request: GeometricImageRequest,
    _: None = Depends(main.verify_api_key),
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

    return await main._run_endpoint(
        "/geometric-image-analysis", GeometricImageResponse, request,
        lambda: main.zospy_handler.get_geometric_image_analysis(
            field_size=request.field_size,
            image_size=request.image_size,
            rays_x_1000=request.rays_x_1000,
            number_of_pixels=request.number_of_pixels,
            field=request.field,
            wavelength=wavelength,
        ),
    )


@router.post("/physical-optics-propagation", response_model=PhysicalOpticsPropagationResponse)
async def get_physical_optics_propagation(
    request: PhysicalOpticsPropagationRequest,
    _: None = Depends(main.verify_api_key),
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

    return await main._run_endpoint(
        "/physical-optics-propagation", PhysicalOpticsPropagationResponse, request,
        lambda: main.zospy_handler.get_physical_optics_propagation(
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
