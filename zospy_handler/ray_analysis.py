"""Unified ray analysis mixin – combined spot + diagnostic in one batch trace."""

import logging
import time
from typing import Any, Optional

from config import RAY_ERROR_CODES
from zospy_handler._base import _extract_value, _log_raw_output
from zospy_handler.pupil import (
    generate_hexapolar_coords,
    generate_square_grid_coords,
    generate_random_coords,
)
from utils.timing import log_timing

logger = logging.getLogger(__name__)


class RayAnalysisMixin:
    def unified_ray_analysis(
        self,
        num_rays: int = 50,
        distribution: str = "hexapolar",
        field_index: Optional[int] = None,
        wavelength_index: Optional[int] = None,
    ) -> dict[str, Any]:
        """
        Trace rays through the system and return both position AND diagnostic data.

        Single IBatchRayTrace call returns everything needed for spot diagrams
        (X, Y at image) and ray diagnostics (errCode, vignetteCode) combined.

        Args:
            num_rays: Number of rays per field/wavelength (determines grid density)
            distribution: Ray distribution type ('hexapolar', 'grid', or 'random')
            field_index: 0-based field index, or None for all fields
            wavelength_index: 0-based wavelength index, or None for all wavelengths

        Returns:
            Dict with:
                - paraxial: Basic paraxial data (efl, bfl, fno, total_track)
                - num_surfaces: Number of surfaces in system
                - num_fields: Number of fields
                - num_wavelengths: Number of wavelengths
                - wavelength_info: List of {index, wavelength_um}
                - raw_rays: List of per-ray results with position AND diagnostic info
                - surface_semi_diameters: List of semi-diameters from LDE
        """
        # Generate pupil coordinates based on distribution
        if distribution in ("square", "grid"):
            pupil_coords = generate_square_grid_coords(num_rays)
        elif distribution == "random":
            pupil_coords = generate_random_coords(num_rays)
        else:
            pupil_coords = generate_hexapolar_coords(num_rays)
        logger.info(
            f"Ray analysis: distribution={distribution}, requested={num_rays}, "
            f"actual={len(pupil_coords)} rays/field/wavelength"
        )

        # Get paraxial data
        paraxial = self.get_paraxial_data()

        # Get system info
        lde = self.oss.LDE
        num_surfaces = lde.NumberOfSurfaces - 1  # Exclude object surface
        fields = self.oss.SystemData.Fields
        num_fields = fields.NumberOfFields
        wavelengths = self.oss.SystemData.Wavelengths
        num_wavelengths = wavelengths.NumberOfWavelengths

        # Extract surface semi-diameters from LDE
        surface_semi_diameters = []
        for i in range(1, lde.NumberOfSurfaces):
            surface = lde.GetSurfaceAt(i)
            surface_semi_diameters.append(_extract_value(surface.SemiDiameter))

        # Determine which fields and wavelengths to trace
        if field_index is not None:
            field_indices = [field_index + 1]  # Convert 0-based to 1-based
        else:
            field_indices = list(range(1, num_fields + 1))

        if wavelength_index is not None:
            wl_indices = [wavelength_index + 1]  # Convert 0-based to 1-based
        else:
            wl_indices = list(range(1, num_wavelengths + 1))

        # Build wavelength info
        wavelength_info = []
        for wi in wl_indices:
            wl = wavelengths.GetWavelength(wi)
            wavelength_info.append({
                "index": wi - 1,  # 0-based for response
                "wavelength_um": _extract_value(wl.Wavelength, 0.5876),
            })

        # Calculate max field extent for Hx/Hy normalization (must use ALL fields)
        max_field_x = 0.0
        max_field_y = 0.0
        field_coords: dict[int, tuple[float, float]] = {}
        for fi in range(1, num_fields + 1):
            field = fields.GetField(fi)
            fx = _extract_value(field.X)
            fy = _extract_value(field.Y)
            field_coords[fi] = (fx, fy)
            max_field_x = max(max_field_x, abs(fx))
            max_field_y = max(max_field_y, abs(fy))

        raw_rays: list[dict[str, Any]] = []
        ray_trace_start = time.perf_counter()
        batch_trace = None

        try:
            batch_trace = self.oss.Tools.OpenBatchRayTrace()
            if batch_trace is None:
                logger.error("Could not open BatchRayTrace tool")
                return self._ray_analysis_error_result(
                    paraxial, num_surfaces, num_fields, num_wavelengths,
                    wavelength_info, surface_semi_diameters,
                )

            total_rays = len(field_indices) * len(wl_indices) * len(pupil_coords)
            norm_unpol = batch_trace.CreateNormUnpol(
                total_rays,
                self._zp.constants.Tools.RayTrace.RaysType.Real,
                self.oss.LDE.NumberOfSurfaces,
            )
            if norm_unpol is None:
                logger.error("Could not create NormUnpol ray trace")
                return self._ray_analysis_error_result(
                    paraxial, num_surfaces, num_fields, num_wavelengths,
                    wavelength_info, surface_semi_diameters,
                )

            opd_none = self._zp.constants.Tools.RayTrace.OPDMode.None_

            # Add all rays in one batch — order must match read loop below
            # Order: for each field -> for each wavelength -> for each pupil coord
            rays_added = 0
            for fi in field_indices:
                fx, fy = field_coords[fi]
                hx = float(fx / max_field_x) if max_field_x > 1e-10 else 0.0
                hy = float(fy / max_field_y) if max_field_y > 1e-10 else 0.0
                for wi in wl_indices:
                    for px, py in pupil_coords:
                        norm_unpol.AddRay(wi, hx, hy, float(px), float(py), opd_none)
                        rays_added += 1

            logger.debug(
                f"BatchRayTrace: added {rays_added} rays "
                f"({len(field_indices)} fields x {len(wl_indices)} wavelengths x {len(pupil_coords)} pupil)"
            )

            batch_trace.RunAndWaitForCompletion()
            norm_unpol.StartReadingResults()

            # Read results in same order as AddRay calls
            total_success = 0
            total_failed = 0
            for fi in field_indices:
                fx, fy = field_coords[fi]
                for wi_idx, wi in enumerate(wl_indices):
                    wl_um = wavelength_info[wi_idx]["wavelength_um"]
                    for px, py in pupil_coords:
                        result = norm_unpol.ReadNextResult()
                        success = result[0]
                        err_code = result[2]
                        vignette_code = result[3]
                        # result[4], result[5] = x, y at image surface (mm)

                        ray_result: dict[str, Any] = {
                            "field_index": fi - 1,  # 0-based
                            "field_x": fx,
                            "field_y": fy,
                            "wavelength_index": wi - 1,  # 0-based
                            "wavelength_um": wl_um,
                            "px": float(px),
                            "py": float(py),
                            "reached_image": False,
                            "failed_surface": None,
                            "failure_mode": None,
                            "x_um": None,
                            "y_um": None,
                        }

                        if success and err_code == 0 and vignette_code == 0:
                            ray_result["reached_image"] = True
                            # Convert mm -> µm for spot diagram positions
                            ray_result["x_um"] = float(result[4]) * 1000.0
                            ray_result["y_um"] = float(result[5]) * 1000.0
                            total_success += 1
                        else:
                            total_failed += 1
                            if vignette_code > 0:
                                ray_result["failed_surface"] = int(vignette_code)
                                ray_result["failure_mode"] = (
                                    "VIGNETTE" if err_code == 0
                                    else RAY_ERROR_CODES.get(err_code, f"ERROR_{err_code}")
                                )
                            else:
                                ray_result["failure_mode"] = RAY_ERROR_CODES.get(
                                    err_code, f"ERROR_{err_code}"
                                )

                        raw_rays.append(ray_result)

            logger.info(
                f"BatchRayTrace: {total_success} success, {total_failed} failed "
                f"out of {rays_added}"
            )

        except Exception as e:
            logger.error(
                f"BatchRayTrace FAILED: {type(e).__name__}: {e}", exc_info=True
            )
            # Discard partial results — they may be incomplete/misleading
            raw_rays = []
            return self._ray_analysis_error_result(
                paraxial, num_surfaces, num_fields, num_wavelengths,
                wavelength_info, surface_semi_diameters,
            )
        finally:
            if batch_trace is not None:
                try:
                    batch_trace.Close()
                except Exception:
                    pass
            ray_trace_elapsed_ms = (time.perf_counter() - ray_trace_start) * 1000
            log_timing(logger, "ray_analysis_all", ray_trace_elapsed_ms)

        result = {
            "paraxial": {
                "efl": paraxial.get("efl"),
                "bfl": paraxial.get("bfl"),
                "fno": paraxial.get("fno"),
                "total_track": paraxial.get("total_track"),
            },
            "num_surfaces": num_surfaces,
            "num_fields": num_fields,
            "num_wavelengths": num_wavelengths,
            "wavelength_info": wavelength_info,
            "raw_rays": raw_rays,
            "surface_semi_diameters": surface_semi_diameters,
        }
        _log_raw_output("/ray-analysis", result)
        return result

    @staticmethod
    def _ray_analysis_error_result(
        paraxial: dict,
        num_surfaces: int,
        num_fields: int,
        num_wavelengths: int,
        wavelength_info: list,
        surface_semi_diameters: list,
    ) -> dict[str, Any]:
        """Return a safe error result dict when batch trace fails to initialize."""
        return {
            "paraxial": {
                "efl": paraxial.get("efl"),
                "bfl": paraxial.get("bfl"),
                "fno": paraxial.get("fno"),
                "total_track": paraxial.get("total_track"),
            },
            "num_surfaces": num_surfaces,
            "num_fields": num_fields,
            "num_wavelengths": num_wavelengths,
            "wavelength_info": wavelength_info,
            "raw_rays": [],
            "surface_semi_diameters": surface_semi_diameters,
        }
