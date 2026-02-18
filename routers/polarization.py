"""Polarization router – pupil map, transmission."""

from fastapi import APIRouter, Depends

import main
from models import (
    PolarizationPupilMapRequest, PolarizationPupilMapResponse,
    PolarizationTransmissionRequest, PolarizationTransmissionResponse,
)

router = APIRouter()


@router.post("/polarization-pupil-map", response_model=PolarizationPupilMapResponse)
async def get_polarization_pupil_map(
    request: PolarizationPupilMapRequest,
    _: None = Depends(main.verify_api_key),
) -> PolarizationPupilMapResponse:
    """Get Polarization Pupil Map showing polarization state across the pupil."""
    return await main._run_endpoint(
        "/polarization-pupil-map", PolarizationPupilMapResponse, request,
        lambda: main.zospy_handler.get_polarization_pupil_map(
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


@router.post("/polarization-transmission", response_model=PolarizationTransmissionResponse)
async def get_polarization_transmission(
    request: PolarizationTransmissionRequest,
    _: None = Depends(main.verify_api_key),
) -> PolarizationTransmissionResponse:
    """Get Polarization Transmission showing transmission vs field with polarization effects."""
    return await main._run_endpoint(
        "/polarization-transmission", PolarizationTransmissionResponse, request,
        lambda: main.zospy_handler.get_polarization_transmission(
            sampling=request.sampling,
            unpolarized=request.unpolarized,
            jx=request.jx,
            jy=request.jy,
            x_phase=request.x_phase,
            y_phase=request.y_phase,
        ),
    )
