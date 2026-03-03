"""Performance router – MTF, PSF, ray fan, standard spot metrics."""

import logging

from fastapi import APIRouter, Depends

import main
from models import (
    StandardSpotMetricsRequest, StandardSpotMetricsResponse,
    SpotFieldData,
    MTFRequest, MTFResponse,
    HuygensMTFRequest,
    ThroughFocusMTFRequest, ThroughFocusMTFResponse,
    ThroughFocusSpotRequest, ThroughFocusSpotResponse,
    ThroughFocusSpotFieldData, ThroughFocusSpotFocusEntry, ThroughFocusSpotBestFocus,
    PSFRequest, PSFResponse,
    HuygensPSFRequest,
    RayFanRequest, RayFanResponse,
    OPDFanRequest, OPDFanResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/standard-spot-metrics", response_model=StandardSpotMetricsResponse)
async def get_standard_spot_metrics(
    request: StandardSpotMetricsRequest,
    _: None = Depends(main.verify_api_key),
) -> StandardSpotMetricsResponse:
    """
    Run StandardSpot analysis for official ZOS-API RMS/GEO metrics only.
    No batch ray trace — metrics only.
    """
    def _build_response(result: dict) -> StandardSpotMetricsResponse:
        if not result.get("success", False):
            return StandardSpotMetricsResponse(
                success=False,
                error=result.get("error", "StandardSpot metrics failed"),
            )
        spot_data = None
        if result.get("spot_data"):
            spot_data = [SpotFieldData(**sd) for sd in result["spot_data"]]
        return StandardSpotMetricsResponse(
            success=True,
            spot_data=spot_data,
        )

    return await main._run_endpoint(
        "/standard-spot-metrics", StandardSpotMetricsResponse, request,
        lambda: main.zospy_handler.get_standard_spot_metrics(
            ray_density=request.ray_density,
            reference=request.reference,
            field_index=request.field_index,
            wavelength_index=request.wavelength_index,
        ),
        build_response=_build_response,
    )


@router.post("/mtf", response_model=MTFResponse)
async def get_mtf(
    request: MTFRequest,
    _: None = Depends(main.verify_api_key),
) -> MTFResponse:
    """
    Get MTF (Modulation Transfer Function) data using FFT MTF analysis.

    Returns raw frequency/modulation data. Image rendering happens on Mac side.
    """
    return await main._run_endpoint(
        "/mtf", MTFResponse, request,
        lambda: main.zospy_handler.get_mtf(
            field_index=request.field_index,
            wavelength_index=request.wavelength_index,
            sampling=request.sampling,
            maximum_frequency=request.maximum_frequency,
        ),
    )


@router.post("/huygens-mtf", response_model=MTFResponse)
async def get_huygens_mtf(
    request: HuygensMTFRequest,
    _: None = Depends(main.verify_api_key),
) -> MTFResponse:
    """
    Get Huygens MTF data. More accurate than FFT MTF for systems with
    significant aberrations or tilted/decentered elements.

    Returns raw frequency/modulation data. Image rendering happens on Mac side.
    """
    return await main._run_endpoint(
        "/huygens-mtf", MTFResponse, request,
        lambda: main.zospy_handler.get_huygens_mtf(
            field_index=request.field_index,
            wavelength_index=request.wavelength_index,
            sampling=request.sampling,
            maximum_frequency=request.maximum_frequency,
        ),
    )


@router.post("/through-focus-mtf", response_model=ThroughFocusMTFResponse)
async def get_through_focus_mtf(
    request: ThroughFocusMTFRequest,
    _: None = Depends(main.verify_api_key),
) -> ThroughFocusMTFResponse:
    """
    Get Through Focus MTF data using FFT Through Focus MTF analysis.

    Shows how MTF varies at different focus positions. Returns raw data;
    image rendering happens on Mac side.
    """
    return await main._run_endpoint(
        "/through-focus-mtf", ThroughFocusMTFResponse, request,
        lambda: main.zospy_handler.get_through_focus_mtf(
            sampling=request.sampling,
            delta_focus=request.delta_focus,
            frequency=request.frequency,
            number_of_steps=request.number_of_steps,
            field_index=request.field_index,
            wavelength_index=request.wavelength_index,
        ),
    )


@router.post("/psf", response_model=PSFResponse)
async def get_psf(
    request: PSFRequest,
    _: None = Depends(main.verify_api_key),
) -> PSFResponse:
    """
    Get PSF (Point Spread Function) data using FFT PSF analysis.

    Returns raw 2D intensity grid as numpy array. Image rendering happens on Mac side.
    """
    return await main._run_endpoint(
        "/psf", PSFResponse, request,
        lambda: main.zospy_handler.get_psf(
            field_index=request.field_index,
            wavelength_index=request.wavelength_index,
            sampling=request.sampling,
        ),
    )


@router.post("/huygens-psf", response_model=PSFResponse)
async def get_huygens_psf(
    request: HuygensPSFRequest,
    _: None = Depends(main.verify_api_key),
) -> PSFResponse:
    """
    Get Huygens PSF (Point Spread Function) data.

    More accurate than FFT PSF for highly aberrated systems.
    Returns raw 2D intensity grid as numpy array. Image rendering happens on Mac side.
    """
    return await main._run_endpoint(
        "/huygens-psf", PSFResponse, request,
        lambda: main.zospy_handler.get_huygens_psf(
            field_index=request.field_index,
            wavelength_index=request.wavelength_index,
            sampling=request.sampling,
        ),
    )


@router.post("/ray-fan", response_model=RayFanResponse)
async def get_ray_fan(
    request: RayFanRequest,
    _: None = Depends(main.verify_api_key),
) -> RayFanResponse:
    """
    Get Ray Fan (Ray Aberration) data.

    Returns raw pupil/aberration data for tangential and sagittal fans.
    Image rendering happens on the Mac analysis service side.
    """
    return await main._run_endpoint(
        "/ray-fan", RayFanResponse, request,
        lambda: main.zospy_handler.get_ray_fan(
            field_index=request.field_index,
            wavelength_index=request.wavelength_index,
            plot_scale=request.plot_scale,
            number_of_rays=request.number_of_rays,
        ),
    )


@router.post("/optical-path-fan", response_model=OPDFanResponse)
async def get_optical_path_fan(
    request: OPDFanRequest,
    _: None = Depends(main.verify_api_key),
) -> OPDFanResponse:
    """
    Get OPD Fan (Optical Path Difference Fan) data.

    Returns OPD in waves for tangential and sagittal fans.
    """
    return await main._run_endpoint(
        "/optical-path-fan", OPDFanResponse, request,
        lambda: main.zospy_handler.get_optical_path_fan(
            field_index=request.field_index,
            wavelength_index=request.wavelength_index,
            plot_scale=request.plot_scale,
            number_of_rays=request.number_of_rays,
        ),
    )


@router.post("/through-focus-spot", response_model=ThroughFocusSpotResponse)
async def get_through_focus_spot(
    request: ThroughFocusSpotRequest,
    _: None = Depends(main.verify_api_key),
) -> ThroughFocusSpotResponse:
    """
    Get Through Focus Spot Diagram data.

    Traces spot rays at multiple defocus positions to show how the spot
    evolves across focus. Returns raw ray data for Mac-side rendering.
    """
    def _build_response(result: dict) -> ThroughFocusSpotResponse:
        if not result.get("success", False):
            return ThroughFocusSpotResponse(
                success=False,
                error=result.get("error", "Through Focus Spot analysis failed"),
            )

        fields_out = None
        if result.get("fields"):
            fields_out = [
                ThroughFocusSpotFieldData(
                    field_index=f["field_index"],
                    field_x=f["field_x"],
                    field_y=f["field_y"],
                    focus_spots=[
                        ThroughFocusSpotFocusEntry(**spot)
                        for spot in f.get("focus_spots", [])
                    ],
                )
                for f in result["fields"]
            ]

        best_focus = None
        if result.get("best_focus"):
            best_focus = ThroughFocusSpotBestFocus(**result["best_focus"])

        return ThroughFocusSpotResponse(
            success=True,
            focus_positions=result.get("focus_positions"),
            fields=fields_out,
            best_focus=best_focus,
            airy_radius_um=result.get("airy_radius_um"),
            wavelength_um=result.get("wavelength_um"),
        )

    return await main._run_endpoint(
        "/through-focus-spot", ThroughFocusSpotResponse, request,
        lambda: main.zospy_handler.get_through_focus_spot(
            delta_focus=request.delta_focus,
            number_of_steps=request.number_of_steps,
            ray_density=request.ray_density,
            field_index=request.field_index,
            wavelength_index=request.wavelength_index,
        ),
        build_response=_build_response,
    )
