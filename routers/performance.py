"""Performance router – spot diagram, MTF, PSF, ray fan."""

import logging

from fastapi import APIRouter, Depends

import main
from models import (
    SpotDiagramRequest, SpotFieldData, SpotRayPoint, SpotRayData, SpotDiagramResponse,
    MTFRequest, MTFResponse,
    HuygensMTFRequest,
    ThroughFocusMTFRequest, ThroughFocusMTFResponse,
    PSFRequest, PSFResponse,
    HuygensPSFRequest,
    RayFanRequest, RayFanResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/spot-diagram", response_model=SpotDiagramResponse)
async def get_spot_diagram(
    request: SpotDiagramRequest,
    _: None = Depends(main.verify_api_key),
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
                    wavelength_um=ray_data.get("wavelength_um", 0.0),
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
            wavelength_info=result.get("wavelength_info"),
            num_fields=result.get("num_fields"),
            num_wavelengths=result.get("num_wavelengths"),
        )

    return await main._run_endpoint(
        "/spot-diagram", SpotDiagramResponse, request,
        lambda: main.zospy_handler.get_spot_diagram(
            ray_density=request.ray_density,
            reference=request.reference,
            field_index=request.field_index,
            wavelength_index=request.wavelength_index,
        ),
        build_response=_build_spot_response,
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
