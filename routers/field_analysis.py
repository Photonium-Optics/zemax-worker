"""Field analysis router – RMS vs field."""

from fastapi import APIRouter, Depends

import main
from models import RmsVsFieldRequest, RmsVsFieldResponse

router = APIRouter()


@router.post("/rms-vs-field", response_model=RmsVsFieldResponse)
async def get_rms_vs_field(
    request: RmsVsFieldRequest,
    _: None = Depends(main.verify_api_key),
) -> RmsVsFieldResponse:
    """
    Get RMS spot radius vs field using native RmsField analysis.

    Auto-samples across the full field range, producing a smooth curve.
    """
    return await main._run_endpoint(
        "/rms-vs-field", RmsVsFieldResponse, request,
        lambda: main.zospy_handler.get_rms_vs_field(
            ray_density=request.ray_density,
            num_field_points=request.num_field_points,
            reference=request.reference,
            wavelength_index=request.wavelength_index,
        ),
    )
