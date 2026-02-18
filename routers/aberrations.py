"""Aberrations router – Seidel, wavefront, Zernike."""

from fastapi import APIRouter, Depends

import main
from models import (
    SystemRequest, NativeSeidelResponse,
    WavefrontRequest, WavefrontResponse,
    ZernikeCoefficientsRequest, ZernikeCoefficientsDetailResponse,
    ZernikeVsFieldRequest, ZernikeVsFieldResponse,
)

router = APIRouter()


@router.post("/seidel-native", response_model=NativeSeidelResponse)
async def get_seidel_native(request: SystemRequest, _: None = Depends(main.verify_api_key)) -> NativeSeidelResponse:
    """
    Get native Seidel text output from OpticStudio's SeidelCoefficients analysis.

    Returns raw text — parsing happens on the Mac side (seidel_text_parser.py).
    """
    return await main._run_endpoint(
        "/seidel-native", NativeSeidelResponse, request,
        lambda: main.zospy_handler.get_seidel_native(),
    )


@router.post("/wavefront", response_model=WavefrontResponse)
async def get_wavefront(
    request: WavefrontRequest,
    _: None = Depends(main.verify_api_key),
) -> WavefrontResponse:
    """
    Get wavefront error map and RMS wavefront error.

    This is a "dumb executor" endpoint - returns raw data only.
    Wavefront map image rendering happens on Mac side using matplotlib.
    """
    return await main._run_endpoint(
        "/wavefront", WavefrontResponse, request,
        lambda: main.zospy_handler.get_wavefront(
            field_index=request.field_index,
            wavelength_index=request.wavelength_index,
            sampling=request.sampling,
            remove_tilt=request.remove_tilt,
        ),
    )


@router.post("/zernike-standard-coefficients", response_model=ZernikeCoefficientsDetailResponse)
async def get_zernike_standard_coefficients(
    request: ZernikeCoefficientsRequest,
    _: None = Depends(main.verify_api_key),
) -> ZernikeCoefficientsDetailResponse:
    """
    Get Zernike Standard Coefficients decomposition of the wavefront.

    Returns individual Zernike polynomial terms (Z1-Z37+), P-V and RMS wavefront
    error, and Strehl ratio.
    """
    return await main._run_endpoint(
        "/zernike-standard-coefficients", ZernikeCoefficientsDetailResponse, request,
        lambda: main.zospy_handler.get_zernike_standard_coefficients(
            field_index=request.field_index,
            wavelength_index=request.wavelength_index,
            sampling=request.sampling,
            maximum_term=request.maximum_term,
        ),
    )


@router.post("/zernike-vs-field", response_model=ZernikeVsFieldResponse)
async def get_zernike_vs_field(
    request: ZernikeVsFieldRequest,
    _: None = Depends(main.verify_api_key),
) -> ZernikeVsFieldResponse:
    """
    Get Zernike Coefficients vs Field analysis.

    Returns how each Zernike polynomial coefficient varies across field positions.
    Critical for understanding field-dependent aberrations.
    """
    return await main._run_endpoint(
        "/zernike-vs-field", ZernikeVsFieldResponse, request,
        lambda: main.zospy_handler.get_zernike_vs_field(
            maximum_term=request.maximum_term,
            wavelength_index=request.wavelength_index,
            sampling=request.sampling,
            field_density=request.field_density,
        ),
    )
