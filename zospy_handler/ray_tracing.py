"""Ray-tracing mixin – batch ray diagnostic trace."""

import logging
import time
from typing import Any

from config import RAY_ERROR_CODES
from zospy_handler._base import _extract_value, _log_raw_output, _compute_field_normalization, _normalize_field
from zospy_handler.pupil import generate_hexapolar_coords, generate_square_grid_coords
from utils.timing import log_timing

logger = logging.getLogger(__name__)


class RayTracingMixin:
    def ray_trace_diagnostic(
        self,
        num_rays: int = 50,
        distribution: str = "hexapolar",
    ) -> dict[str, Any]:
        """
        Trace rays through the system and return raw per-ray results.

        This is a "dumb executor" that returns raw data only - no aggregation,
        no hotspot detection, no threshold calculations. All post-processing
        happens on the Mac side (zemax-analysis-service).

        Note: System must be pre-loaded via load_zmx_file().

        Args:
            num_rays: Number of rays per field (determines grid density)
            distribution: Ray distribution type ('hexapolar' or 'square')

        Returns:
            Dict with:
                - paraxial: Basic paraxial data (efl, bfl, fno, total_track)
                - num_surfaces: Number of surfaces in system
                - num_fields: Number of fields
                - raw_rays: List of per-ray results with field, pupil coords, success/failure info
                - surface_semi_diameters: List of semi-diameters from LDE
        """
        # Generate pupil coordinates based on distribution
        if distribution in ("square", "grid"):
            pupil_coords = generate_square_grid_coords(num_rays)
        else:
            pupil_coords = generate_hexapolar_coords(num_rays)
        logger.info(f"Ray trace: distribution={distribution}, requested={num_rays}, actual={len(pupil_coords)} rays/field")

        # Get paraxial data
        paraxial = self.get_paraxial_data()

        # Get system info
        lde = self.oss.LDE
        num_surfaces = lde.NumberOfSurfaces - 1  # Exclude object surface
        fields = self.oss.SystemData.Fields
        num_fields = fields.NumberOfFields

        # Find primary wavelength index (don't hardcode to 1)
        wavelengths = self.oss.SystemData.Wavelengths
        primary_wl = 1
        for wi in range(1, int(wavelengths.NumberOfWavelengths) + 1):
            if wavelengths.GetWavelength(wi).IsPrimary:
                primary_wl = wi
                break

        # Extract surface semi-diameters from LDE
        # Use _extract_value for UnitField objects
        surface_semi_diameters = []
        for i in range(1, lde.NumberOfSurfaces):
            surface = lde.GetSurfaceAt(i)
            surface_semi_diameters.append(_extract_value(surface.SemiDiameter))

        # Collect raw ray results using batch ray trace (single COM roundtrip)
        raw_rays = []

        # Compute field normalization parameters (respects Radial vs Rectangular)
        is_radial, max_field_x, max_field_y, max_field_r = _compute_field_normalization(fields, num_fields)
        field_coords: dict[int, tuple[float, float]] = {}
        for fi in range(1, num_fields + 1):
            field = fields.GetField(fi)
            field_coords[fi] = (_extract_value(field.X), _extract_value(field.Y))

        ray_trace_start = time.perf_counter()
        batch_trace = None
        try:
            batch_trace = self.oss.Tools.OpenBatchRayTrace()
            if batch_trace is None:
                logger.error("Could not open BatchRayTrace tool")
                return {"paraxial": {}, "num_surfaces": num_surfaces, "num_fields": num_fields, "raw_rays": [], "surface_semi_diameters": surface_semi_diameters}

            total_rays = num_fields * len(pupil_coords)
            norm_unpol = batch_trace.CreateNormUnpol(
                total_rays,
                self._zp.constants.Tools.RayTrace.RaysType.Real,
                self.oss.LDE.NumberOfSurfaces,
            )
            if norm_unpol is None:
                logger.error("Could not create NormUnpol ray trace")
                return {"paraxial": {}, "num_surfaces": num_surfaces, "num_fields": num_fields, "raw_rays": [], "surface_semi_diameters": surface_semi_diameters}

            opd_none = self._zp.constants.Tools.RayTrace.OPDMode.None_

            # Add all rays in one batch — order must match read loop below
            rays_added = 0
            for fi in range(1, num_fields + 1):
                fx, fy = field_coords[fi]
                hx, hy = _normalize_field(fx, fy, is_radial, max_field_x, max_field_y, max_field_r)
                for px, py in pupil_coords:
                    norm_unpol.AddRay(primary_wl, hx, hy, float(px), float(py), opd_none)
                    rays_added += 1

            logger.debug(f"BatchRayTrace: added {rays_added} rays ({num_fields} fields x {len(pupil_coords)} pupil)")

            batch_trace.RunAndWaitForCompletion()
            norm_unpol.StartReadingResults()

            # Read results in same order as AddRay calls
            total_success = 0
            total_failed = 0
            for fi in range(1, num_fields + 1):
                fx, fy = field_coords[fi]
                for px, py in pupil_coords:
                    result = norm_unpol.ReadNextResult()
                    success, err_code, vignette_code = result[0], result[2], result[3]

                    ray_result = {
                        "field_index": fi - 1,
                        "field_x": fx,
                        "field_y": fy,
                        "px": float(px),
                        "py": float(py),
                        "reached_image": False,
                        "failed_surface": None,
                        "failure_mode": None,
                    }

                    if success and err_code == 0 and vignette_code == 0:
                        ray_result["reached_image"] = True
                        total_success += 1
                    else:
                        total_failed += 1
                        if vignette_code > 0:
                            ray_result["failed_surface"] = int(vignette_code)
                            ray_result["failure_mode"] = "VIGNETTE" if err_code == 0 else RAY_ERROR_CODES.get(err_code, f"ERROR_{err_code}")
                        else:
                            ray_result["failure_mode"] = RAY_ERROR_CODES.get(err_code, f"ERROR_{err_code}")

                    raw_rays.append(ray_result)

            logger.info(f"BatchRayTrace: {total_success} success, {total_failed} failed out of {rays_added}")

        except Exception as e:
            logger.error(f"BatchRayTrace FAILED: {type(e).__name__}: {e}", exc_info=True)
            raw_rays = []  # discard partial results on failure
        finally:
            if batch_trace is not None:
                try:
                    batch_trace.Close()
                except Exception:
                    pass
            ray_trace_elapsed_ms = (time.perf_counter() - ray_trace_start) * 1000
            log_timing(logger, "ray_trace_all", ray_trace_elapsed_ms)

        if not raw_rays:
            return {"success": False, "error": "BatchRayTrace produced no results"}

        result = {
            "paraxial": {
                "efl": paraxial.get("efl"),
                "bfl": paraxial.get("bfl"),
                "fno": paraxial.get("fno"),
                "total_track": paraxial.get("total_track"),
            },
            "num_surfaces": num_surfaces,
            "num_fields": num_fields,
            "raw_rays": raw_rays,
            "surface_semi_diameters": surface_semi_diameters,
        }
        _log_raw_output("/ray-trace-diagnostic", result)
        return result

