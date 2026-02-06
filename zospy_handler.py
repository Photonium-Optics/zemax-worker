"""
ZosPy Handler

Manages the connection to Zemax OpticStudio and executes ZosPy operations.
This module contains all the actual ZosPy/OpticStudio calls.

Note: This code runs on Windows only, where OpticStudio is installed.

Each uvicorn worker process gets its own OpticStudio connection (via ZOS singleton).
Multiple workers are supported — the constraint is license seats, not threading.
"""

import base64
import logging
import os
import tempfile
import time
from typing import Any, Optional

import numpy as np

from utils.timing import log_timing

# Configure module logger
logger = logging.getLogger(__name__)

# =============================================================================
# Lazy ZosPy Import
# =============================================================================
#
# CRITICAL: ZosPy imports are LAZY to prevent blocking during module load.
#
# When `import zospy` runs, it:
# 1. Imports pythonnet (.NET interop)
# 2. Loads ZOSAPI DLLs into the CLR
# 3. This can HANG on some systems (DLL loading issues, OpticStudio not found, etc.)
#
# By making imports lazy, the FastAPI server starts immediately and can respond
# to /health checks. ZosPy is only imported when the first real request comes in.
#
# The doubled "ZOSAPI imported to clr" log happens because uvicorn re-imports
# the module in its worker process. This is normal with string-based app references.
# =============================================================================

# Lazy-loaded module references
_zp = None  # zospy module
_OpticStudioSystem = None  # zospy.zpcore.OpticStudioSystem class
_ZOSPY_IMPORT_ATTEMPTED = False
_ZOSPY_AVAILABLE = False


def _ensure_zospy_imported() -> bool:
    """
    Lazily import ZosPy on first use.

    This prevents blocking during module load. The import only happens
    when ZosPyHandler is instantiated (typically in the lifespan function
    or on first request).

    Returns:
        True if ZosPy is available, False otherwise.
    """
    global _zp, _OpticStudioSystem, _ZOSPY_IMPORT_ATTEMPTED, _ZOSPY_AVAILABLE

    if _ZOSPY_IMPORT_ATTEMPTED:
        return _ZOSPY_AVAILABLE

    _ZOSPY_IMPORT_ATTEMPTED = True
    logger.info("Lazily importing ZosPy (this may take a moment)...")

    try:
        import zospy as zp_module
        from zospy.zpcore import OpticStudioSystem as OSS

        _zp = zp_module
        _OpticStudioSystem = OSS
        _ZOSPY_AVAILABLE = True
        logger.info(f"ZosPy {zp_module.__version__} imported successfully")
        return True
    except ImportError as e:
        logger.error(f"Failed to import ZosPy: {e}")
        _ZOSPY_AVAILABLE = False
        return False
    except Exception as e:
        logger.error(f"Unexpected error importing ZosPy: {e}")
        _ZOSPY_AVAILABLE = False
        return False


def get_zospy_module():
    """Get the zospy module, importing it lazily if needed."""
    _ensure_zospy_imported()
    return _zp


def is_zospy_available() -> bool:
    """Check if ZosPy is available (imports lazily if not yet attempted)."""
    return _ensure_zospy_imported()


# =============================================================================
# Constants
# =============================================================================

# Index Convention Notes:
# -----------------------
# OpticStudio/ZosPy uses 1-based indexing internally for surfaces, fields, wavelengths:
#   - LDE.GetSurfaceAt(1) returns the first optical surface (after object surface 0)
#   - Fields.GetField(1) returns the first field
#   - Wavelengths.GetWavelength(1) returns the first wavelength
#   - SingleRayTrace(field=1, wavelength=1) uses 1-based indices
#
# API Response Convention:
#   - Surface indices in responses are CONVERTED to 0-based for consistency with
#     the LLM JSON schema (surfaces[0] is the first surface after object)
#   - Field indices in responses are 0-based (field_index=0 is first field)
#   - Wavelength indices in responses are 0-based
#
# When calling ZosPy/OpticStudio methods, always use 1-based indices.
# When returning data in API responses, convert to 0-based indices.

# Analysis settings
DEFAULT_SAMPLING = "64x64"
DEFAULT_MAX_ZERNIKE_TERM = 37
DEFAULT_NUM_CROSS_SECTION_RAYS = 11

# Image export settings
CROSS_SECTION_IMAGE_SIZE = (1200, 800)
CROSS_SECTION_TEMP_FILENAME = "zemax_cross_section.png"
SEIDEL_TEMP_FILENAME = "seidel_native.txt"
SPOT_DIAGRAM_TEMP_FILENAME = "zemax_spot_diagram.png"

# OpticStudio version requirements
MIN_IMAGE_EXPORT_VERSION = (24, 1, 0)

# Ray trace error codes
RAY_ERROR_CODES = {
    0: "OK",
    1: "MISS",
    2: "TIR",
    3: "REVERSED",
    4: "VIGNETTE",
}


def _extract_value(obj: Any, default: float = 0.0) -> float:
    """
    Extract a numeric value from various ZosPy types.

    ZosPy 2.x returns UnitField objects for many values instead of plain floats.
    This helper handles both UnitField and plain numeric types.

    Args:
        obj: Value to extract (UnitField, float, int, etc.)
        default: Default value if extraction fails

    Returns:
        Float value extracted from the object
    """
    if obj is None:
        return default

    # Unwrap UnitField objects (have .value attribute)
    value = obj.value if hasattr(obj, 'value') else obj

    try:
        result = float(value)
        # Check for NaN
        if np.isnan(result):
            return default
        return result
    except (TypeError, ValueError):
        return default


def _get_column_value(row: Any, column_names: list[str], default: Any = None) -> Any:
    """
    Safely extract a value from a pandas Series or dict-like object.

    ZosPy 2.1.4 uses column names like 'X-coordinate', 'Y-coordinate', 'Z-coordinate'.
    This helper tries multiple column name patterns for backwards compatibility.

    Args:
        row: pandas Series or dict-like object
        column_names: List of column names to try in order
        default: Default value if no column found

    Returns:
        Column value or default
    """
    for col in column_names:
        try:
            if col in row:
                return row[col]
        except (KeyError, TypeError):
            # Some objects may not support 'in' operator
            pass

        # Fallback to .get() for dict-like objects
        if hasattr(row, 'get'):
            val = row.get(col)
            if val is not None:
                return val

    return default


def _safe_int(value: Any, default: int = 0) -> int:
    """
    Safely convert a value to int, handling None and NaN.

    Args:
        value: Value to convert
        default: Default value if conversion fails

    Returns:
        Integer value or default
    """
    if value is None:
        return default
    try:
        # Check for NaN (only floats can be NaN)
        if isinstance(value, float) and np.isnan(value):
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


# F/# aperture types
FNO_APERTURE_TYPES = ["ImageSpaceFNumber", "FloatByStopSize", "ParaxialWorkingFNumber"]


class ZosPyHandler:
    """
    Handler for ZosPy/OpticStudio operations.

    Manages the connection lifecycle and provides methods for
    optical analysis operations.

    Systems are loaded via load_zmx_file() only - no manual surface building.
    """

    def __init__(self):
        """Initialize ZosPy and connect to OpticStudio."""
        # Lazily import ZosPy - this is where the actual import happens
        if not is_zospy_available():
            raise ZosPyError("ZosPy is not available. Install it with: pip install zospy")

        self._zp = get_zospy_module()
        if self._zp is None:
            raise ZosPyError("ZosPy module not loaded")

        try:
            # Initialize ZOS connection
            self.zos = self._zp.ZOS()

            # Connect to OpticStudio (starts instance if needed)
            self.oss = self.zos.connect(mode="standalone")

            if self.oss is None:
                raise ZosPyError("Failed to connect to OpticStudio")

            logger.info(f"Connected to OpticStudio: {self.get_version()}")

        except Exception as e:
            raise ZosPyError(f"Failed to initialize ZosPy: {e}")

    def close(self) -> None:
        """
        Close the connection to OpticStudio.

        This should be called during application shutdown to cleanly
        disconnect from the OpticStudio COM server.
        """
        try:
            if hasattr(self, 'zos') and self.zos:
                self.zos.disconnect()
        except Exception as e:
            logger.warning(f"Error closing ZosPy connection: {e}")

    def get_version(self) -> str:
        """
        Get OpticStudio version string.

        Returns:
            Version string (e.g., "25.1.0") or "Unknown" if unavailable.
        """
        try:
            # ZosPy exposes version through the application object
            return str(self.oss.Application.ZemaxVersion) if self.oss else "Unknown"
        except Exception:
            return "Unknown"

    def get_status(self) -> dict[str, Any]:
        """
        Get current connection status.

        Returns:
            Dict with keys:
                - connected: bool - Whether OpticStudio is connected
                - opticstudio_version: str - OpticStudio version
                - zospy_version: str - ZosPy library version
        """
        try:
            zospy_version = self._zp.__version__
        except Exception:
            zospy_version = "Unknown"

        return {
            "connected": self.oss is not None,
            "opticstudio_version": self.get_version(),
            "zospy_version": zospy_version,
        }

    def load_zmx_file(self, file_path: str) -> dict[str, Any]:
        """
        Load an optical system from a .zmx file directly into OpticStudio.

        This is the preferred method for loading systems as it uses Zemax's
        native file format and avoids manual surface construction.

        Args:
            file_path: Absolute path to the .zmx file

        Returns:
            Dict with load status and system info
        """
        import os
        if not os.path.exists(file_path):
            raise ZosPyError(f"ZMX file not found: {file_path}")

        # Load the file directly using ZosPy's load method
        # This replaces all manual surface building with native file loading
        load_start = time.perf_counter()
        try:
            self.oss.load(file_path)
        finally:
            load_elapsed_ms = (time.perf_counter() - load_start) * 1000
            log_timing(logger, "oss.load", load_elapsed_ms)

        # Get system info after loading
        num_surfaces = self.oss.LDE.NumberOfSurfaces - 1  # Exclude object surface

        return {
            "num_surfaces": num_surfaces,
            "efl": self._get_efl(),
        }

    def _get_efl(self) -> Optional[float]:
        """
        Get effective focal length using ZosPy's SystemData analysis.

        Returns:
            Effective focal length in mm, or None if calculation fails.
        """
        try:
            # Use the SystemData analysis to get first-order properties
            result = self._zp.analyses.reports.SystemData().run(self.oss)
            # Use _extract_value for potential UnitField object
            return _extract_value(result.data.general_lens_data.effective_focal_length_air, None)
        except Exception:
            return None

    def get_paraxial_data(self) -> dict[str, Any]:
        """
        Get first-order (paraxial) optical properties.

        Delegates to _get_paraxial_from_lde() and adds F/# if available.

        Returns:
            Dict with paraxial properties (epd, max_field, total_track, fno, etc.)
        """
        paraxial = self._get_paraxial_from_lde()

        # Add F/# if available
        fno = self._get_fno()
        if fno is not None:
            paraxial["fno"] = fno

        return paraxial

    def get_cross_section(self) -> dict[str, Any]:
        """
        Generate cross-section diagram using ZosPy's CrossSection analysis.

        Requires ZosPy >= 1.3.0 and OpticStudio >= 24.1.0 for image export.
        Returns error on failure — no fallbacks.

        Note: System must be pre-loaded via load_zmx_file().
        """
        # Validate system state
        num_fields = self.oss.SystemData.Fields.NumberOfFields
        if num_fields == 0:
            return {"success": False, "error": "System has no fields defined"}

        zos_version = self.zos.version if hasattr(self.zos, 'version') else None
        if zos_version and zos_version < MIN_IMAGE_EXPORT_VERSION:
            return {"success": False, "error": f"OpticStudio {zos_version} < 24.1.0 — image export not supported"}

        temp_path = os.path.join(tempfile.gettempdir(), CROSS_SECTION_TEMP_FILENAME)

        try:
            from zospy.analyses.systemviewers.cross_section import CrossSection

            cross_section = CrossSection(
                number_of_rays=DEFAULT_NUM_CROSS_SECTION_RAYS,
                field="All",
                wavelength="All",
                color_rays_by="Fields",
                delete_vignetted=True,
                surface_line_thickness="Thick",
                rays_line_thickness="Standard",
                image_size=CROSS_SECTION_IMAGE_SIZE,
            )

            cs_start = time.perf_counter()
            try:
                result = cross_section.run(self.oss, image_output_file=temp_path)
            finally:
                cs_elapsed_ms = (time.perf_counter() - cs_start) * 1000
                log_timing(logger, "CrossSection.run", cs_elapsed_ms)

            if not os.path.exists(temp_path):
                return {"success": False, "error": "CrossSection analysis did not produce an image"}

            with open(temp_path, 'rb') as f:
                image_b64 = base64.b64encode(f.read()).decode('utf-8')
            logger.info(f"Successfully exported CrossSection image, size = {len(image_b64)}")

            # Get paraxial data and surface geometry
            paraxial = self._get_paraxial_from_lde()
            surfaces_data = self._get_surface_geometry()

            rays_total = DEFAULT_NUM_CROSS_SECTION_RAYS * max(1, num_fields)

            return {
                "success": True,
                "image": image_b64,
                "image_format": "png",
                "array_shape": None,
                "array_dtype": None,
                "paraxial": paraxial,
                "surfaces": surfaces_data,
                "rays_total": rays_total,
                "rays_through": rays_total,
            }

        except Exception as e:
            return {"success": False, "error": f"CrossSection analysis failed: {e}"}
        finally:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass

    def _get_paraxial_from_lde(self) -> dict[str, Any]:
        """
        Get paraxial data directly from LDE and SystemData.

        This method is more reliable than using analysis functions as it
        reads directly from the system data structures.

        Returns:
            Dict with keys: epd, max_field, field_type, field_unit, total_track
        """
        try:
            paraxial = {}

            # Get aperture info - use _extract_value for UnitField objects
            aperture = self.oss.SystemData.Aperture
            paraxial["epd"] = _extract_value(aperture.ApertureValue)

            # Get field info - use _extract_value for UnitField objects
            fields = self.oss.SystemData.Fields
            max_field = 0.0
            if fields.NumberOfFields > 0:
                for i in range(1, fields.NumberOfFields + 1):
                    f = fields.GetField(i)
                    max_field = max(max_field, abs(_extract_value(f.Y)), abs(_extract_value(f.X)))
            paraxial["max_field"] = max_field
            paraxial["field_type"] = "object_angle"
            paraxial["field_unit"] = "deg"

            # Calculate total track from LDE - use _extract_value for UnitField objects
            lde = self.oss.LDE
            total_track = 0.0
            for i in range(1, lde.NumberOfSurfaces):
                surface = lde.GetSurfaceAt(i)
                total_track += abs(_extract_value(surface.Thickness))
            paraxial["total_track"] = total_track

            return paraxial
        except Exception as e:
            logger.warning(f"Could not get paraxial data from LDE: {e}")
            return {}

    def _get_surface_geometry(self) -> list[dict[str, Any]]:
        """
        Extract surface geometry for client-side cross-section rendering.

        This method reads the Lens Data Editor (LDE) to extract geometric
        properties of each surface. Used as fallback data when image export fails.

        Returns:
            List of surface dicts with keys: index, z, radius, thickness,
            semi_diameter, conic, material, is_stop
        """
        surfaces = []
        lde = self.oss.LDE
        z_position = 0.0

        for i in range(1, lde.NumberOfSurfaces):
            surface = lde.GetSurfaceAt(i)

            # Radius of 0 in Zemax means infinity (flat surface)
            # We convert to None for client-side rendering
            # Use _extract_value for all UnitField properties
            radius = _extract_value(surface.Radius)
            thickness = _extract_value(surface.Thickness)
            semi_diameter = _extract_value(surface.SemiDiameter)
            conic = _extract_value(surface.Conic)
            surf_data = {
                "index": i,
                "z": z_position,
                "radius": radius if radius != 0 and abs(radius) < 1e10 else None,
                "thickness": thickness,
                "semi_diameter": semi_diameter,
                "conic": conic,
                "material": str(surface.Material) if surface.Material else None,
                "is_stop": surface.IsStop,
            }
            surfaces.append(surf_data)
            z_position += thickness

        return surfaces

    def calc_semi_diameters(self) -> dict[str, Any]:
        """
        Calculate semi-diameters by reading from surfaces after ray trace.

        Reads the SemiDiameter property from each surface in the LDE,
        which OpticStudio computes based on ray extent during tracing.

        Note: System must be pre-loaded via load_zmx_file().

        Returns:
            Dict with "semi_diameters" key containing list of
            {"index": int, "value": float} entries.
        """
        semi_diameters = []
        lde = self.oss.LDE

        # For each surface, get the computed semi-diameter
        # Use _extract_value for UnitField objects
        for i in range(1, lde.NumberOfSurfaces):
            surface = lde.GetSurfaceAt(i)
            sd = _extract_value(surface.SemiDiameter)

            semi_diameters.append({
                "index": i - 1,  # Convert to 0-indexed
                "value": sd,
            })

        return {"semi_diameters": semi_diameters}

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

        return {
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

    def get_seidel(self) -> dict[str, Any]:
        """
        Get raw Zernike coefficients from OpticStudio.

        This is a "dumb executor" - it tries the ZosPy method once and returns
        the result or an error. No fallback strategies. The Mac side handles
        retries and Zernike-to-Seidel conversion.

        Note: System must be pre-loaded via load_zmx_file().

        Returns:
            On success: {"success": True, "zernike_coefficients": [...], "wavelength_um": float, "num_surfaces": int}
            On error: {"success": False, "error": "..."}
        """
        # Try ZosPy Zernike analysis
        try:
            zernike_analysis = self._zp.analyses.wavefront.ZernikeStandardCoefficients(
                sampling=DEFAULT_SAMPLING,
                maximum_term=DEFAULT_MAX_ZERNIKE_TERM,
                wavelength=1,
                field=1,
                surface="Image",
            )
            zernike_start = time.perf_counter()
            try:
                result = zernike_analysis.run(self.oss)
            finally:
                zernike_elapsed_ms = (time.perf_counter() - zernike_start) * 1000
                log_timing(logger, "ZernikeStandardCoefficients.run (seidel)", zernike_elapsed_ms)

            # Extract coefficients from result
            if not hasattr(result, 'data') or result.data is None:
                return {"success": False, "error": "ZosPy analysis returned no data"}

            coeff_data = result.data
            if not hasattr(coeff_data, 'coefficients'):
                return {"success": False, "error": "ZosPy result has no coefficients attribute"}

            raw_coeffs = coeff_data.coefficients
            coefficients = []

            if isinstance(raw_coeffs, dict):
                # Dict keyed by term number
                if not raw_coeffs:
                    return {"success": False, "error": "ZosPy returned empty coefficients dict"}

                int_keys = [int(k) for k in raw_coeffs.keys()]
                max_term = max(int_keys) if int_keys else 0

                for i in range(1, min(max_term + 1, DEFAULT_MAX_ZERNIKE_TERM + 1)):
                    coeff = raw_coeffs.get(i) or raw_coeffs.get(str(i))
                    if coeff is not None:
                        if hasattr(coeff, 'value'):
                            coefficients.append(float(coeff.value))
                        else:
                            coefficients.append(float(coeff))
                    else:
                        coefficients.append(0.0)
            elif hasattr(raw_coeffs, '__iter__'):
                # Iterable (list-like)
                for coeff in raw_coeffs:
                    if hasattr(coeff, 'value'):
                        coefficients.append(float(coeff.value))
                    else:
                        coefficients.append(float(coeff) if coeff is not None else 0.0)
            else:
                return {"success": False, "error": f"Unknown coefficients type: {type(raw_coeffs)}"}

            if not coefficients:
                return {"success": False, "error": "No coefficients extracted from ZosPy result"}

            # Get system info - handle UnitField objects from ZosPy 2.x
            wl_obj = self.oss.SystemData.Wavelengths.GetWavelength(1).Wavelength
            wl_um = _extract_value(wl_obj, 0.5876)
            num_surfaces = self.oss.LDE.NumberOfSurfaces - 1

            return {
                "success": True,
                "zernike_coefficients": coefficients,
                "wavelength_um": wl_um,
                "num_surfaces": num_surfaces,
            }

        except Exception as e:
            return {"success": False, "error": f"ZosPy analysis failed: {e}"}

    def get_seidel_native(self) -> dict[str, Any]:
        """
        Get native Seidel text output using OpticStudio's SeidelCoefficients analysis.

        This is a "dumb executor" — runs the analysis, exports the text file,
        and returns the raw UTF-16 text content. All parsing happens on the
        Mac side (seidel_text_parser.py).

        Note: System must be pre-loaded via load_zmx_file().

        Returns:
            On success: {
                "success": True,
                "seidel_text": str (raw text from GetTextFile),
                "num_surfaces": int,
            }
            On error: {"success": False, "error": "..."}
        """
        analysis = None
        temp_path = os.path.join(tempfile.gettempdir(), SEIDEL_TEMP_FILENAME)

        try:
            idm = self._zp.constants.Analysis.AnalysisIDM

            # Create and run SeidelCoefficients analysis
            analysis = self._zp.analyses.new_analysis(
                self.oss,
                idm.SeidelCoefficients,
                settings_first=True
            )
            seidel_start = time.perf_counter()
            try:
                analysis.ApplyAndWaitForCompletion()
            finally:
                seidel_elapsed_ms = (time.perf_counter() - seidel_start) * 1000
                log_timing(logger, "SeidelCoefficients.ApplyAndWaitForCompletion", seidel_elapsed_ms)

            # Check for error messages in analysis results
            error_msg = self._check_analysis_errors(analysis)
            if error_msg:
                return {"success": False, "error": error_msg}

            # Export to text file
            analysis.Results.GetTextFile(temp_path)

            if not os.path.exists(temp_path):
                return {"success": False, "error": "GetTextFile did not create output file"}

            text_content = self._read_opticstudio_text_file(temp_path)
            if not text_content:
                return {"success": False, "error": "Seidel text output is empty"}

            num_surfaces = self.oss.LDE.NumberOfSurfaces - 1  # Exclude object surface

            return {
                "success": True,
                "seidel_text": text_content,
                "num_surfaces": num_surfaces,
            }

        except Exception as e:
            return {"success": False, "error": f"SeidelCoefficients analysis failed: {e}"}
        finally:
            self._cleanup_analysis(analysis, temp_path)

    def _check_analysis_errors(self, analysis: Any) -> Optional[str]:
        """
        Check if an OpticStudio analysis has error messages.

        Args:
            analysis: OpticStudio analysis object

        Returns:
            Error message string if errors found, None otherwise
        """
        if hasattr(analysis, 'messages') and analysis.messages:
            for msg in analysis.messages:
                if hasattr(msg, 'Message') and 'cannot' in str(msg.Message).lower():
                    return str(msg.Message)
        return None

    def _read_opticstudio_text_file(self, file_path: str) -> str:
        """
        Read a text file exported by OpticStudio.

        OpticStudio exports text files in UTF-16 encoding.

        Args:
            file_path: Path to the text file

        Returns:
            File content as string, or empty string if read fails
        """
        try:
            with open(file_path, 'r', encoding='utf-16') as f:
                return f.read().strip()
        except Exception as e:
            logger.warning(f"Failed to read OpticStudio text file: {e}")
            return ""

    def _cleanup_analysis(self, analysis: Any, temp_path: Optional[str] = None) -> None:
        """
        Clean up an OpticStudio analysis and its temporary files.

        Args:
            analysis: OpticStudio analysis object to close
            temp_path: Optional temp file path to delete
        """
        if analysis is not None:
            try:
                analysis.Close()
            except Exception as e:
                logger.warning(f"Failed to close analysis: {e}")

        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError as e:
                logger.warning(f"Failed to remove temp file {temp_path}: {e}")

    # NOTE: Seidel text parsing has been moved to the Mac side
    # (zemax-analysis-service/seidel_text_parser.py).
    # The worker now returns raw text from get_seidel_native().

    def trace_rays(
        self,
        num_rays: int = 7,
    ) -> dict[str, Any]:
        """
        Trace rays through the system and return positions at each surface.

        Note: System must be pre-loaded via load_zmx_file().

        Args:
            num_rays: Number of rays to trace
        """
        # Get system info
        num_surfaces = self.oss.LDE.NumberOfSurfaces
        num_fields = self.oss.SystemData.Fields.NumberOfFields
        num_wavelengths = self.oss.SystemData.Wavelengths.NumberOfWavelengths

        data = []

        trace_start = time.perf_counter()
        try:
            # Trace rays for each field and wavelength
            for fi in range(1, num_fields + 1):
                for wi in range(1, num_wavelengths + 1):
                    surfaces_data = []

                # Initialize arrays for each surface
                for si in range(num_surfaces):
                    surfaces_data.append({
                        "y": [],
                        "z": [],
                    })

                # Trace fan of rays across pupil
                for py in np.linspace(-1, 1, num_rays):
                    try:
                        # ZosPy SingleRayTrace: px/py are normalized pupil coordinates (-1 to 1)
                        # hx/hy are normalized field coordinates (not used when field index specified)
                        # CRITICAL: py must be Python float(), not numpy.float64, for COM interop
                        ray_trace = self._zp.analyses.raysandspots.SingleRayTrace(
                            hx=0.0,
                            hy=0.0,
                            px=0.0,  # Meridional fan (x=0)
                            py=float(py),   # Iterate over pupil Y - must be float() for COM
                            wavelength=wi,
                            field=fi,
                        )
                        result = ray_trace.run(self.oss)

                        # Extract ray positions from result.data.real_ray_trace_data
                        if hasattr(result, 'data') and result.data is not None:
                            ray_data = result.data
                            # Access real_ray_trace_data if available
                            if hasattr(ray_data, 'real_ray_trace_data'):
                                df = ray_data.real_ray_trace_data
                            else:
                                df = ray_data  # Fallback

                            if hasattr(df, 'iloc'):
                                for si in range(min(len(df), num_surfaces)):
                                    row = df.iloc[si]
                                    # ZosPy 2.1.4 uses 'Y-coordinate', 'Z-coordinate' column names
                                    # Use _get_column_value helper for safe access
                                    y_val = _get_column_value(row, ['Y-coordinate', 'Y', 'y'])
                                    z_val = _get_column_value(row, ['Z-coordinate', 'Z', 'z'])
                                    surfaces_data[si]["y"].append(y_val)
                                    surfaces_data[si]["z"].append(z_val)
                            else:
                                # Fallback - add None values
                                for si in range(num_surfaces):
                                    surfaces_data[si]["y"].append(None)
                                    surfaces_data[si]["z"].append(None)
                        else:
                            # Ray failed - add None values
                            for si in range(num_surfaces):
                                surfaces_data[si]["y"].append(None)
                                surfaces_data[si]["z"].append(None)

                    except Exception as e:
                        logger.debug(f"Ray trace failed for field {fi}, wavelength {wi}, py={py:.2f}: {e}")
                        for si in range(num_surfaces):
                            surfaces_data[si]["y"].append(None)
                            surfaces_data[si]["z"].append(None)

                    data.append({
                        "field_index": fi - 1,
                        "wavelength_index": wi - 1,
                        "surfaces": surfaces_data,
                    })
        finally:
            trace_elapsed_ms = (time.perf_counter() - trace_start) * 1000
            log_timing(logger, "trace_rays_all", trace_elapsed_ms)

        return {
            "num_surfaces": num_surfaces,
            "num_fields": num_fields,
            "num_wavelengths": num_wavelengths,
            "data": data,
        }

    def get_wavefront(
        self,
        field_index: int = 1,
        wavelength_index: int = 1,
        sampling: str = "64x64",
        remove_tilt: bool = False,
    ) -> dict[str, Any]:
        """
        Get wavefront error map and metrics using ZosPy's WavefrontMap
        and ZernikeStandardCoefficients analyses.

        This is a "dumb executor" - returns raw data only. Wavefront map
        image rendering happens on Mac side.

        Note: System must be pre-loaded via load_zmx_file().

        Args:
            field_index: Field index (1-indexed)
            wavelength_index: Wavelength index (1-indexed)
            sampling: Pupil sampling grid (e.g., '32x32', '64x64', '128x128')

        Returns:
            On success: {
                "success": True,
                "rms_waves": float,
                "pv_waves": float,
                "strehl_ratio": float or None,
                "wavelength_um": float,
                "field_x": float,
                "field_y": float,
                "image": str (base64 numpy array),
                "image_format": "numpy_array",
                "array_shape": [h, w],
                "array_dtype": str,
            }
            On error: {"success": False, "error": "..."}
        """
        try:
            # Get field coordinates for response
            fields = self.oss.SystemData.Fields
            if field_index > fields.NumberOfFields:
                return {"success": False, "error": f"Field index {field_index} out of range (max: {fields.NumberOfFields})"}

            field = fields.GetField(field_index)
            # Use _extract_value for UnitField objects
            field_x = _extract_value(field.X)
            field_y = _extract_value(field.Y)

            # Get wavelength for response
            wavelengths = self.oss.SystemData.Wavelengths
            if wavelength_index > wavelengths.NumberOfWavelengths:
                return {"success": False, "error": f"Wavelength index {wavelength_index} out of range (max: {wavelengths.NumberOfWavelengths})"}

            # Use _extract_value for UnitField objects
            wavelength_um = _extract_value(wavelengths.GetWavelength(wavelength_index).Wavelength, 0.5876)

            # Get wavefront metrics using ZernikeStandardCoefficients
            # This gives us RMS, P-V, and Strehl ratio
            rms_waves = None
            pv_waves = None
            strehl_ratio = None

            # Get wavefront metrics from ZernikeStandardCoefficients
            zernike_analysis = self._zp.analyses.wavefront.ZernikeStandardCoefficients(
                sampling=sampling,
                maximum_term=37,
                wavelength=wavelength_index,
                field=field_index,
                reference_opd_to_vertex=False,
                surface="Image",
            )
            zernike_start = time.perf_counter()
            try:
                zernike_result = zernike_analysis.run(self.oss)
            finally:
                zernike_elapsed_ms = (time.perf_counter() - zernike_start) * 1000
                log_timing(logger, "ZernikeStandardCoefficients.run", zernike_elapsed_ms)

            if hasattr(zernike_result, 'data') and zernike_result.data is not None:
                zdata = zernike_result.data

                # Get P-V wavefront error (in waves)
                if hasattr(zdata, 'peak_to_valley_to_chief'):
                    pv_waves = _extract_value(zdata.peak_to_valley_to_chief)
                elif hasattr(zdata, 'peak_to_valley_to_centroid'):
                    pv_waves = _extract_value(zdata.peak_to_valley_to_centroid)

                # Get RMS and Strehl from integration data
                if hasattr(zdata, 'from_integration_of_the_rays'):
                    integration = zdata.from_integration_of_the_rays
                    if hasattr(integration, 'rms_to_chief'):
                        rms_waves = _extract_value(integration.rms_to_chief)
                    elif hasattr(integration, 'rms_to_centroid'):
                        rms_waves = _extract_value(integration.rms_to_centroid)
                    if hasattr(integration, 'strehl_ratio'):
                        strehl_ratio = _extract_value(integration.strehl_ratio)

            if rms_waves is None and pv_waves is None:
                return {"success": False, "error": "ZernikeStandardCoefficients returned no metrics"}

            # Get wavefront map as numpy array
            image_b64 = None
            array_shape = None
            array_dtype = None

            try:
                wfm_start = time.perf_counter()
                try:
                    wavefront_map = self._zp.analyses.wavefront.WavefrontMap(
                        sampling=sampling,
                        wavelength=wavelength_index,
                        field=field_index,
                        surface="Image",
                        show_as="Surface",
                        rotation="Rotate_0",
                        scale=1,
                        polarization=None,
                        reference_to_primary=False,
                        remove_tilt=remove_tilt,
                        use_exit_pupil=True,
                    ).run(self.oss, oncomplete="Release")
                finally:
                    wfm_elapsed_ms = (time.perf_counter() - wfm_start) * 1000
                    log_timing(logger, "WavefrontMap.run", wfm_elapsed_ms)

                if hasattr(wavefront_map, 'data') and wavefront_map.data is not None:
                    wf_data = wavefront_map.data
                    if hasattr(wf_data, 'values'):
                        arr = np.array(wf_data.values, dtype=np.float64)
                    else:
                        arr = np.array(wf_data, dtype=np.float64)

                    if arr.ndim >= 2:
                        image_b64 = base64.b64encode(arr.tobytes()).decode('utf-8')
                        array_shape = list(arr.shape)
                        array_dtype = str(arr.dtype)
                        logger.info(f"Wavefront map generated: shape={arr.shape}, dtype={arr.dtype}")

            except Exception as e:
                logger.warning(f"WavefrontMap failed: {e} (metrics still available)")

            return {
                "success": True,
                "rms_waves": rms_waves,
                "pv_waves": pv_waves,
                "strehl_ratio": strehl_ratio,
                "wavelength_um": wavelength_um,
                "field_x": field_x,
                "field_y": field_y,
                "image": image_b64,
                "image_format": "numpy_array" if image_b64 else None,
                "array_shape": array_shape,
                "array_dtype": array_dtype,
            }

        except Exception as e:
            return {"success": False, "error": f"Wavefront analysis failed: {e}"}

    def get_spot_diagram(
        self,
        ray_density: int = 5,
        reference: str = "chief_ray",
    ) -> dict[str, Any]:
        """
        Generate spot diagram data using ZosPy's StandardSpot analysis for metrics
        and batch ray tracing for raw ray positions.

        ZOSAPI's StandardSpot analysis does NOT support direct image export (unlike
        CrossSection which uses a layout tool). Image rendering happens on the Mac
        side using the raw ray data returned here.

        This is a "dumb executor" — returns raw data or error. No fallbacks.

        Note: System must be pre-loaded via load_zmx_file().

        Args:
            ray_density: Rays per axis (determines grid density, 1-20)
            reference: Reference point: 'chief_ray' or 'centroid'

        Returns:
            On success: {
                "success": True,
                "image": None (not supported by ZOSAPI for StandardSpot),
                "image_format": None,
                "spot_data": [...] (per-field metrics: RMS, GEO radius, centroid),
                "spot_rays": [...] (raw ray X,Y positions for Mac-side rendering),
                "airy_radius": float,
            }
            On error: {"success": False, "error": "..."}
        """
        analysis = None

        # Get field info
        fields = self.oss.SystemData.Fields
        num_fields = fields.NumberOfFields

        if num_fields == 0:
            return {"success": False, "error": "System has no fields defined"}

        # Map reference parameter to OpticStudio constant (0 = Chief Ray, 1 = Centroid)
        reference_code = 0 if reference == "chief_ray" else 1

        try:
            # Use ZosPy's new_analysis to access StandardSpot for metrics
            analysis = self._zp.analyses.new_analysis(
                self.oss,
                self._zp.constants.Analysis.AnalysisIDM.StandardSpot,
                settings_first=True,
            )

            # Configure and run the analysis
            self._configure_spot_analysis(analysis.Settings, ray_density, reference_code)
            spot_start = time.perf_counter()
            try:
                analysis.ApplyAndWaitForCompletion()
            finally:
                spot_elapsed_ms = (time.perf_counter() - spot_start) * 1000
                log_timing(logger, "StandardSpot.ApplyAndWaitForCompletion", spot_elapsed_ms)

            # Extract spot metrics (RMS, GEO radius, centroid) from analysis results
            spot_data: list[dict[str, Any]] = []
            airy_radius: Optional[float] = None
            if analysis.Results is not None:
                airy_radius = self._extract_airy_radius(analysis.Results)
                spot_data = self._extract_spot_data_from_results(analysis.Results, fields, num_fields)

            # Close analysis before batch ray trace
            self._cleanup_analysis(analysis, None)
            analysis = None

            # Get raw ray X,Y positions using batch ray tracing
            # This is required because ZOSAPI doesn't expose raw ray data from StandardSpot
            ray_trace_start = time.perf_counter()
            try:
                spot_rays = self._get_spot_ray_data(ray_density)
            finally:
                ray_trace_elapsed_ms = (time.perf_counter() - ray_trace_start) * 1000
                log_timing(logger, "BatchRayTrace for spot diagram", ray_trace_elapsed_ms)

            # Note: image is None because ZOSAPI StandardSpot doesn't support image export
            # Mac side will render the spot diagram from spot_rays data
            return {
                "success": True,
                "image": None,
                "image_format": None,
                "array_shape": None,
                "array_dtype": None,
                "spot_data": spot_data,
                "spot_rays": spot_rays,
                "airy_radius": airy_radius,
            }

        except Exception as e:
            return {"success": False, "error": f"StandardSpot analysis failed: {e}"}
        finally:
            self._cleanup_analysis(analysis, None)

    def _configure_spot_analysis(self, settings: Any, ray_density: int, reference_code: int) -> None:
        """
        Configure StandardSpot analysis settings.

        Args:
            settings: OpticStudio analysis settings object
            ray_density: Rays per axis (1-20)
            reference_code: Reference point (0=Chief Ray, 1=Centroid)
        """
        # Set ray density
        if hasattr(settings, 'RayDensity'):
            settings.RayDensity = ray_density
        elif hasattr(settings, 'NumberOfRays'):
            settings.NumberOfRays = ray_density

        # Set reference point
        if hasattr(settings, 'ReferenceType'):
            settings.ReferenceType = reference_code
        elif hasattr(settings, 'Reference'):
            settings.Reference = reference_code

        # Set to show all fields and wavelengths
        if hasattr(settings, 'Field'):
            try:
                settings.Field.UseAllFields()
            except Exception:
                pass

        if hasattr(settings, 'Wavelength'):
            try:
                settings.Wavelength.UseAllWavelengths()
            except Exception:
                pass

    def _get_spot_ray_data(self, ray_density: int) -> list[dict[str, Any]]:
        """
        Get raw ray X,Y positions at image plane using batch ray tracing.

        ZOSAPI's StandardSpot analysis does not expose raw ray positions through
        its Results interface. The only way to get ray X,Y data for rendering
        spot diagrams is via batch ray tracing (IBatchRayTrace).

        Args:
            ray_density: Rays per axis (e.g., 5 means ~5x5 grid per field)

        Returns:
            List of dicts per field, each containing:
                - field_index: int (0-based)
                - field_x, field_y: float (field coordinates)
                - wavelength_index: int (0-based)
                - rays: list of {"x": float, "y": float} at image plane
        """
        spot_rays: list[dict[str, Any]] = []

        fields = self.oss.SystemData.Fields
        wavelengths = self.oss.SystemData.Wavelengths
        num_fields = fields.NumberOfFields
        num_wavelengths = wavelengths.NumberOfWavelengths

        # Generate pupil coordinates for ray grid
        # ray_density^2 rays in a circular pupil pattern
        pupil_coords = []
        for px in np.linspace(-1, 1, ray_density):
            for py in np.linspace(-1, 1, ray_density):
                if px**2 + py**2 <= 1.0:  # Inside circular pupil
                    pupil_coords.append((float(px), float(py)))

        # Calculate max field extent for normalization (avoid division by zero)
        max_field_x = 0.0
        max_field_y = 0.0
        for fi in range(1, num_fields + 1):
            max_field_x = max(max_field_x, abs(_extract_value(fields.GetField(fi).X)))
            max_field_y = max(max_field_y, abs(_extract_value(fields.GetField(fi).Y)))

        ray_trace = None
        try:
            # Use batch ray trace for efficiency
            ray_trace = self.oss.Tools.OpenBatchRayTrace()
            if ray_trace is None:
                logger.warning("Could not open BatchRayTrace tool")
                return spot_rays

            # Get the normalized unpolarized ray trace interface
            max_rays = num_fields * num_wavelengths * len(pupil_coords)
            norm_unpol = ray_trace.CreateNormUnpol(
                max_rays,
                self._zp.constants.Tools.RayTrace.RaysType.Real,
                self.oss.LDE.NumberOfSurfaces - 1,  # Image surface
            )

            if norm_unpol is None:
                logger.warning("Could not create NormUnpol ray trace")
                return spot_rays

            # AddRay signature: (WaveNumber, Hx, Hy, Px, Py, OPDMode)
            opd_none = self._zp.constants.Tools.RayTrace.OPDMode.None_

            # Add rays for all field/wavelength/pupil combinations
            for fi in range(1, num_fields + 1):
                field = fields.GetField(fi)
                field_x_val = _extract_value(field.X)
                field_y_val = _extract_value(field.Y)
                hx_norm = float(field_x_val / max_field_x) if max_field_x > 1e-10 else 0.0
                hy_norm = float(field_y_val / max_field_y) if max_field_y > 1e-10 else 0.0
                for wi in range(1, num_wavelengths + 1):
                    for px, py in pupil_coords:
                        norm_unpol.AddRay(wi, hx_norm, hy_norm, float(px), float(py), opd_none)

            # Run the ray trace
            ray_trace.RunAndWaitForCompletion()

            # Read results and organize by field/wavelength
            for fi in range(1, num_fields + 1):
                field = fields.GetField(fi)
                field_x = _extract_value(field.X)
                field_y = _extract_value(field.Y)

                for wi in range(1, num_wavelengths + 1):
                    field_rays = {
                        "field_index": fi - 1,
                        "field_x": field_x,
                        "field_y": field_y,
                        "wavelength_index": wi - 1,
                        "rays": [],
                    }

                    for _ in pupil_coords:
                        result = norm_unpol.ReadNextResult()
                        # ReadNextResult returns 15 values; we only need success(0), err_code(2), x(4), y(5)
                        success, err_code = result[0], result[2]
                        if success and err_code == 0:
                            field_rays["rays"].append({"x": float(result[4]), "y": float(result[5])})

                    spot_rays.append(field_rays)

        except Exception as e:
            logger.warning(f"Batch ray trace for spot diagram failed: {e}", exc_info=True)
        finally:
            if ray_trace is not None:
                try:
                    ray_trace.Close()
                except Exception:
                    pass  # Ignore cleanup errors

        return spot_rays

    def _extract_airy_radius(self, results: Any) -> Optional[float]:
        """
        Extract Airy disk radius from analysis results.

        Args:
            results: OpticStudio analysis results object

        Returns:
            Airy radius in lens units, or None if not available
        """
        try:
            # Use _extract_value() to handle ZosPy 2.x UnitField objects
            if hasattr(results, 'AiryRadius'):
                return _extract_value(results.AiryRadius)
            elif hasattr(results, 'GetAiryDiskRadius'):
                return _extract_value(results.GetAiryDiskRadius())
        except Exception as e:
            logger.debug(f"Could not extract Airy radius: {e}")
        return None

    def _extract_spot_data_from_results(
        self,
        results: Any,
        fields: Any,
        num_fields: int,
    ) -> list[dict[str, Any]]:
        """
        Extract per-field spot data from analysis results.

        Args:
            results: OpticStudio analysis results object
            fields: OpticStudio fields object
            num_fields: Number of fields in the system

        Returns:
            List of spot data dicts per field
        """
        spot_data: list[dict[str, Any]] = []

        try:
            for fi in range(num_fields):
                field = fields.GetField(fi + 1)  # 1-indexed
                # Use _extract_value for UnitField objects
                field_data = self._create_field_spot_data(fi, _extract_value(field.X), _extract_value(field.Y))

                # Try to get spot data for this field
                self._populate_spot_data_from_results(results, fi, field_data)
                spot_data.append(field_data)

        except Exception as e:
            logger.warning(f"Could not extract spot data from results: {e}")

        return spot_data

    def _create_field_spot_data(
        self,
        field_index: int,
        field_x: float,
        field_y: float,
    ) -> dict[str, Any]:
        """
        Create an empty spot data dict for a field.

        Args:
            field_index: 0-indexed field number
            field_x: Field X coordinate
            field_y: Field Y coordinate

        Returns:
            Dict with field info and None values for spot metrics
        """
        return {
            "field_index": field_index,
            "field_x": field_x,
            "field_y": field_y,
            "rms_radius": None,
            "geo_radius": None,
            "centroid_x": None,
            "centroid_y": None,
            "num_rays": None,
        }

    def _populate_spot_data_from_results(
        self,
        results: Any,
        field_index: int,
        field_data: dict[str, Any],
    ) -> None:
        """
        Populate spot data dict from analysis results.

        Args:
            results: OpticStudio analysis results object
            field_index: 0-indexed field number
            field_data: Dict to populate with spot metrics
        """
        try:
            # Use _extract_value() to handle ZosPy 2.x UnitField objects
            if hasattr(results, 'GetDataSeries'):
                series = results.GetDataSeries(field_index)
                if series:
                    if hasattr(series, 'RMS'):
                        field_data["rms_radius"] = _extract_value(series.RMS)
                    if hasattr(series, 'GEO'):
                        field_data["geo_radius"] = _extract_value(series.GEO)

            if hasattr(results, 'SpotData'):
                spot_info = results.SpotData
                if hasattr(spot_info, 'GetSpotDataAtField'):
                    fd = spot_info.GetSpotDataAtField(field_index + 1)
                    if fd:
                        if hasattr(fd, 'RMSSpotRadius'):
                            field_data["rms_radius"] = _extract_value(fd.RMSSpotRadius)
                        if hasattr(fd, 'GEOSpotRadius'):
                            field_data["geo_radius"] = _extract_value(fd.GEOSpotRadius)
                        if hasattr(fd, 'CentroidX'):
                            field_data["centroid_x"] = _extract_value(fd.CentroidX)
                        if hasattr(fd, 'CentroidY'):
                            field_data["centroid_y"] = _extract_value(fd.CentroidY)
                        if hasattr(fd, 'NumberOfRays'):
                            field_data["num_rays"] = int(_extract_value(fd.NumberOfRays))

        except Exception as e:
            logger.debug(f"Could not get spot data for field {field_index}: {e}")


    def _get_fno(self) -> Optional[float]:
        """
        Get the f-number of the optical system.

        Tries to get f/# directly from aperture settings if the aperture type
        is F/#-based, otherwise calculates from EPD and EFL.

        Returns:
            F-number, or None if it cannot be determined
        """
        try:
            aperture = self.oss.SystemData.Aperture
            # Handle enum ApertureType - try .name first, then string split
            aperture_type = ""
            if aperture.ApertureType:
                if hasattr(aperture.ApertureType, 'name'):
                    aperture_type = aperture.ApertureType.name
                else:
                    aperture_type = str(aperture.ApertureType).split(".")[-1]

            if aperture_type in FNO_APERTURE_TYPES:
                # Use _extract_value for UnitField objects
                return _extract_value(aperture.ApertureValue)

            # Calculate from EPD and EFL
            # Use _extract_value for UnitField objects
            epd = _extract_value(aperture.ApertureValue)
            efl = self._get_efl()
            if epd and efl and epd > 0:
                return efl / epd

        except Exception as e:
            logger.debug(f"Could not get f-number: {e}")

        return None


    def evaluate_merit_function(self, operand_rows: list[dict]) -> dict[str, Any]:
        """
        Evaluate a merit function by constructing operands in the MFE and computing.

        Args:
            operand_rows: List of dicts with keys:
                - operand_code: str (e.g. "EFFL")
                - params: list of up to 6 values (None for unused slots)
                - target: float
                - weight: float

        Returns:
            Dict with:
                - success: bool
                - total_merit: float or None
                - evaluated_rows: list of per-row results
                - row_errors: list of per-row error messages
        """
        zp = self._zp
        mfe = self.oss.MFE

        # Clear existing MFE
        mfe.DeleteAllRows()

        evaluated_rows = []
        row_errors = []
        valid_operand_indices = []  # (mfe_row_number, original_row_index)

        # MFE column constants for parameter cells
        try:
            mfe_cols = zp.constants.Editors.MFE.MeritColumn
        except AttributeError:
            return {
                "success": False,
                "error": "Cannot access MFE column constants",
                "total_merit": None,
                "evaluated_rows": [],
                "row_errors": [{"row_index": 0, "error": "Cannot access MFE column constants"}],
            }

        mfe_row_number = 0  # Tracks actual MFE row position (1-based)

        for row_index, row in enumerate(operand_rows):
            code = row.get("operand_code", "")
            params = row.get("params", [])
            target = float(row.get("target", 0))
            weight = float(row.get("weight", 1))

            # Resolve operand type enum
            try:
                op_type = getattr(zp.constants.Editors.MFE.MeritOperandType, code)
            except AttributeError:
                row_errors.append({
                    "row_index": row_index,
                    "error": f"Unknown operand code: {code}",
                })
                evaluated_rows.append({
                    "row_index": row_index,
                    "operand_code": code,
                    "value": None,
                    "target": target,
                    "weight": weight,
                    "contribution": None,
                    "error": f"Unknown operand code: {code}",
                })
                continue

            mfe_row_number += 1

            # After DeleteAllRows, MFE retains 1 empty row.
            # First operand uses GetOperandAt(1), subsequent use InsertNewOperandAt.
            if mfe_row_number == 1:
                op = mfe.GetOperandAt(1)
            else:
                op = mfe.InsertNewOperandAt(mfe_row_number)

            try:
                op.ChangeType(op_type)
                op.Target = float(target)
                op.Weight = float(weight)

                # Set parameter cells
                # Slots 0-1 (Int1, Int2) -> IntegerValue
                # Slots 2-5 (Hx, Hy, Px, Py) -> DoubleValue
                param_columns = [
                    mfe_cols.Param1, mfe_cols.Param2,
                    mfe_cols.Param3, mfe_cols.Param4,
                    mfe_cols.Param5, mfe_cols.Param6,
                ]
                for i, col in enumerate(param_columns):
                    if i < len(params) and params[i] is not None:
                        cell = op.GetOperandCell(col)
                        if i < 2:
                            cell.IntegerValue = int(float(params[i]))
                        else:
                            cell.DoubleValue = float(params[i])

                valid_operand_indices.append((mfe_row_number, row_index))

            except Exception as e:
                logger.warning(f"MFE row {mfe_row_number} ({code}): error setting params: {e}")
                row_errors.append({
                    "row_index": row_index,
                    "error": f"Error configuring {code}: {e}",
                })
                evaluated_rows.append({
                    "row_index": row_index,
                    "operand_code": code,
                    "value": None,
                    "target": target,
                    "weight": weight,
                    "contribution": None,
                    "error": f"Error configuring {code}: {e}",
                })
                continue

        if not valid_operand_indices:
            error_summary = f"All {len(operand_rows)} operand row(s) failed validation"
            return {
                "success": False,
                "error": error_summary,
                "total_merit": None,
                "evaluated_rows": evaluated_rows,
                "row_errors": row_errors,
            }

        # Calculate merit function
        try:
            # Use _extract_value() to handle ZosPy 2.x UnitField objects
            total_merit = _extract_value(mfe.CalculateMeritFunction())
        except Exception as e:
            logger.error(f"MFE CalculateMeritFunction failed: {e}")
            return {
                "success": False,
                "error": f"CalculateMeritFunction failed: {e}",
                "total_merit": None,
                "evaluated_rows": evaluated_rows,
                "row_errors": row_errors + [{"row_index": -1, "error": f"CalculateMeritFunction failed: {e}"}],
            }

        # Read back results for each valid row
        for mfe_row_num, orig_index in valid_operand_indices:
            row = operand_rows[orig_index]
            try:
                op = mfe.GetOperandAt(mfe_row_num)
                value = _extract_value(op.Value, None)
                contribution = _extract_value(op.Contribution, None)

                evaluated_rows.append({
                    "row_index": orig_index,
                    "operand_code": row.get("operand_code", ""),
                    "value": value,
                    "target": float(row.get("target", 0)),
                    "weight": float(row.get("weight", 1)),
                    "contribution": contribution,
                    "error": None,
                })
            except Exception as e:
                logger.warning(f"Error reading MFE row {mfe_row_num}: {e}")
                evaluated_rows.append({
                    "row_index": orig_index,
                    "operand_code": row.get("operand_code", ""),
                    "value": None,
                    "target": float(row.get("target", 0)),
                    "weight": float(row.get("weight", 1)),
                    "contribution": None,
                    "error": f"Error reading result: {e}",
                })

        # Sort evaluated_rows by row_index for consistent output
        evaluated_rows.sort(key=lambda r: r["row_index"])

        return {
            "success": True,
            "total_merit": total_merit,
            "evaluated_rows": evaluated_rows,
            "row_errors": row_errors,
        }


class ZosPyError(Exception):
    """Exception raised when ZosPy operations fail."""
    pass
