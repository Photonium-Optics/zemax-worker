"""Ray-tracing mixin – single-ray diagnostic trace."""

import logging
import time
from typing import Any

import numpy as np

from config import RAY_ERROR_CODES
from zospy_handler._base import _extract_value, _log_raw_output, _safe_int
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
            distribution: Ray distribution type (currently uses square grid)

        Returns:
            Dict with:
                - paraxial: Basic paraxial data (efl, bfl, fno, total_track)
                - num_surfaces: Number of surfaces in system
                - num_fields: Number of fields
                - raw_rays: List of per-ray results with field, pupil coords, success/failure info
                - surface_semi_diameters: List of semi-diameters from LDE
        """
        # Note: 'distribution' parameter is accepted for API compatibility but only square grid
        # is currently implemented. Log warning if hexapolar is requested.
        if distribution != "square" and distribution != "grid":
            logger.warning(
                f"Distribution '{distribution}' requested but only square grid is implemented. "
                "Using square grid instead."
            )

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

        # Calculate grid size from num_rays
        grid_size = int(np.sqrt(num_rays))

        # Collect raw ray results
        raw_rays = []

        ray_trace_start = time.perf_counter()
        try:
            for fi in range(1, num_fields + 1):
                field = fields.GetField(fi)
                # Use _extract_value for UnitField objects
                field_x = _extract_value(field.X)
                field_y = _extract_value(field.Y)

                # Trace a grid of rays using ZosPy's single ray trace
                for px in np.linspace(-1, 1, grid_size):
                    for py in np.linspace(-1, 1, grid_size):
                        if px**2 + py**2 > 1:
                            continue  # Skip rays outside pupil

                        ray_result = {
                            "field_index": fi - 1,  # 0-indexed
                            "field_x": field_x,
                            "field_y": field_y,
                            "px": float(px),
                            "py": float(py),
                            "reached_image": False,
                            "failed_surface": None,
                            "failure_mode": None,
                        }

                        try:
                            # ZosPy SingleRayTrace: px/py are normalized pupil coordinates (-1 to 1)
                            # hx/hy are normalized field coordinates (set to 0 when using field index)
                            # CRITICAL: px/py must be Python float(), not numpy.float64, for COM interop
                            ray_trace = self._zp.analyses.raysandspots.SingleRayTrace(
                                hx=0.0,
                                hy=0.0,
                                px=float(px),
                                py=float(py),
                                wavelength=1,
                                field=fi,
                            )
                            result = ray_trace.run(self.oss)

                            # Check result - ZosPy returns data in result.data.real_ray_trace_data
                            if hasattr(result, 'data') and result.data is not None:
                                ray_data = result.data
                                # Access real_ray_trace_data if available (DataFrame)
                                if hasattr(ray_data, 'real_ray_trace_data'):
                                    df = ray_data.real_ray_trace_data
                                else:
                                    df = ray_data  # Fallback for older versions

                                # Check if ray reached image
                                if hasattr(df, '__len__') and len(df) > 0:
                                    # Check for error codes - vignetted rays have error_code != 0
                                    if hasattr(df, 'columns') and 'error_code' in df.columns:
                                        # Find first surface with error
                                        error_rows = df[df['error_code'] != 0]
                                        if len(error_rows) > 0:
                                            first_error = error_rows.iloc[0]
                                            ray_result["reached_image"] = False
                                            # Use _safe_int to handle NaN values from DataFrame
                                            ray_result["failed_surface"] = _safe_int(first_error.get('surface', 0), 0)
                                            # Map error code to failure mode string
                                            error_code = _safe_int(first_error.get('error_code', 0), 0)
                                            ray_result["failure_mode"] = self._error_code_to_mode(error_code)
                                        else:
                                            ray_result["reached_image"] = True
                                    else:
                                        # No error column in ZosPy 2.x - assume success if ray data exists
                                        ray_result["reached_image"] = True
                                else:
                                    ray_result["failure_mode"] = "NO_DATA"
                            else:
                                ray_result["failure_mode"] = "NO_RESULT"

                        except Exception as e:
                            logger.debug(f"Ray trace failed for field {fi}, pupil ({px:.2f}, {py:.2f}): {e}")
                            ray_result["failure_mode"] = "EXCEPTION"

                        raw_rays.append(ray_result)
        finally:
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

    def _error_code_to_mode(self, error_code: int) -> str:
        """
        Map ZosPy/OpticStudio error codes to human-readable failure modes.

        Common error codes (may vary by OpticStudio version):
            0 = No error (ray traced successfully)
            1 = Ray missed surface
            2 = TIR (Total Internal Reflection)
            3 = Ray reversed
            4 = Ray vignetted
            5+ = Other errors

        Args:
            error_code: Numeric error code from ray trace

        Returns:
            String describing the failure mode
        """
        return RAY_ERROR_CODES.get(error_code, f"ERROR_{error_code}")
