"""Ray-tracing mixin – batch ray diagnostic trace."""

import logging
import time
from typing import Any

from config import RAY_ERROR_CODES
from zospy_handler._base import _extract_value, _log_raw_output
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

        # Extract surface semi-diameters from LDE
        # Use _extract_value for UnitField objects
        surface_semi_diameters = []
        for i in range(1, lde.NumberOfSurfaces):
            surface = lde.GetSurfaceAt(i)
            surface_semi_diameters.append(_extract_value(surface.SemiDiameter))

        # Collect raw ray results using batch ray trace (single COM roundtrip)
        raw_rays = []

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
                hx = float(fx / max_field_x) if max_field_x > 1e-10 else 0.0
                hy = float(fy / max_field_y) if max_field_y > 1e-10 else 0.0
                for px, py in pupil_coords:
                    norm_unpol.AddRay(1, hx, hy, float(px), float(py), opd_none)
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
        finally:
            if batch_trace is not None:
                try:
                    batch_trace.Close()
                except Exception:
                    pass
            ray_trace_elapsed_ms = (time.perf_counter() - ray_trace_start) * 1000
            log_timing(logger, "ray_trace_all", ray_trace_elapsed_ms)

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

