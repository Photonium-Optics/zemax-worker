"""
ZosPy Handler

Manages the connection to Zemax OpticStudio and executes ZosPy operations.
This module contains all the actual ZosPy/OpticStudio calls.

Note: This code runs on Windows only, where OpticStudio is installed.

IMPORTANT: This worker MUST run with a single uvicorn worker (--workers 1)
because ZosPy/COM requires single-threaded apartment (STA) semantics.
"""

import base64
import logging
import os
import tempfile
from typing import Any, Optional

import numpy as np

# Configure module logger
logger = logging.getLogger(__name__)

# ZosPy imports - these will fail on non-Windows or without OpticStudio
try:
    import zospy as zp
    from zospy.zpcore import OpticStudioSystem
    ZOSPY_AVAILABLE = True
except ImportError:
    ZOSPY_AVAILABLE = False
    zp = None
    OpticStudioSystem = None


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

# Seidel coefficient keys (in order)
SEIDEL_COEFFICIENT_KEYS = ["S1", "S2", "S3", "S4", "S5", "CLA", "CTR"]

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
    # Handle UnitField objects (have .value attribute)
    if hasattr(obj, 'value'):
        try:
            return float(obj.value)
        except (TypeError, ValueError):
            return default
    # Handle plain numeric types
    try:
        return float(obj)
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
        if not ZOSPY_AVAILABLE:
            raise ZosPyError("ZosPy is not available. Install it with: pip install zospy")

        try:
            # Initialize ZOS connection
            self.zos = zp.ZOS()

            # Connect to OpticStudio (starts instance if needed)
            self.oss: OpticStudioSystem = self.zos.connect(mode="standalone")

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
            zospy_version = zp.__version__
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
        self.oss.load(file_path)

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
            result = zp.analyses.reports.SystemData().run(self.oss)
            return result.data.general_lens_data.effective_focal_length_air
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
        Falls back to surface geometry if image export fails.

        Note: System must be pre-loaded via load_zmx_file().
        """

        image_b64 = None
        image_format = None
        # Metadata for numpy array reconstruction (only used when image_format="numpy_array")
        array_shape = None
        array_dtype = None
        error_msg = None

        # Validate system state before attempting CrossSection
        # The "Object reference not set" error in set_StartSurface() happens when
        # the system has no fields or invalid aperture configuration
        try:
            num_fields = self.oss.SystemData.Fields.NumberOfFields
            if num_fields == 0:
                logger.warning("CrossSection skipped: System has no fields defined (NumberOfFields=0)")
                error_msg = "System has no fields defined"
                # Skip to fallback - return surface geometry only
                paraxial = self._get_paraxial_from_lde()
                surfaces_data = self._get_surface_geometry()
                return {
                    "image": None,
                    "image_format": None,
                    "array_shape": None,
                    "array_dtype": None,
                    "paraxial": paraxial,
                    "surfaces": surfaces_data,
                    "rays_total": 0,
                    "rays_through": 0,
                    "error": error_msg,
                }
        except Exception as e:
            logger.warning(f"Could not validate system fields: {e}")
            # Continue anyway - let CrossSection fail with better error if needed

        # Check OpticStudio version - image export requires >= 24.1.0
        try:
            zos_version = self.zos.version if hasattr(self.zos, 'version') else None
            logger.debug(f"ZosPy zos.version = {zos_version}")
            logger.debug(f"ZosPy version = {zp.__version__ if hasattr(zp, '__version__') else 'unknown'}")

            if zos_version and zos_version < MIN_IMAGE_EXPORT_VERSION:
                logger.warning(f"OpticStudio {zos_version} < 24.1.0 - image export not supported, using fallback")
            else:
                # Use ZosPy's CrossSection wrapper with image_output_file parameter
                # Per ZosPy docs: export to file, then read it back
                import tempfile
                import os

                temp_path = os.path.join(tempfile.gettempdir(), CROSS_SECTION_TEMP_FILENAME)
                logger.debug(f"Trying ZosPy CrossSection with image_output_file={temp_path}")

                try:
                    from zospy.analyses.systemviewers.cross_section import CrossSection

                    # Log system state before attempting CrossSection
                    try:
                        mode = self.oss.Mode
                        num_fields = self.oss.SystemData.Fields.NumberOfFields
                        num_wavelengths = self.oss.SystemData.Wavelengths.NumberOfWavelengths
                        logger.info(f"CrossSection: System mode={mode}, fields={num_fields}, wavelengths={num_wavelengths}")
                    except Exception as diag_e:
                        logger.warning(f"CrossSection: Could not get system diagnostics: {diag_e}")

                    cross_section = CrossSection(
                        number_of_rays=DEFAULT_NUM_CROSS_SECTION_RAYS,
                        field="All",
                        wavelength="All",
                        color_rays_by="Fields",
                        delete_vignetted=True,
                        surface_line_thickness="Thick",  # Required to show lens surfaces
                        rays_line_thickness="Standard",
                        image_size=CROSS_SECTION_IMAGE_SIZE,
                    )

                    # Run with image_output_file to save to disk
                    result = cross_section.run(self.oss, image_output_file=temp_path)
                    logger.debug(f"CrossSection.run completed, result.data type = {type(result.data) if hasattr(result, 'data') else 'N/A'}")

                    # Check if file was created
                    if os.path.exists(temp_path):
                        with open(temp_path, 'rb') as f:
                            image_b64 = base64.b64encode(f.read()).decode('utf-8')
                        image_format = "png"
                        logger.info(f"Successfully exported CrossSection image, size = {len(image_b64)}")
                    elif result.data is not None:
                        # Fallback: return raw numpy array for Mac-side rendering
                        # This keeps matplotlib dependency on Mac side only
                        arr = np.array(result.data)
                        # Validate array is image-like (2D grayscale or 3D with channels)
                        if arr.ndim < 2 or arr.ndim > 3:
                            logger.warning(f"CrossSection: result.data has unexpected shape {arr.shape}, skipping numpy fallback")
                        else:
                            image_b64 = base64.b64encode(arr.tobytes()).decode('utf-8')
                            image_format = "numpy_array"
                            array_shape = list(arr.shape)
                            array_dtype = str(arr.dtype)
                            logger.info(f"Returning numpy array fallback for CrossSection, shape={arr.shape}, dtype={arr.dtype}")
                    else:
                        logger.warning("CrossSection: no file created and result.data is None")

                except ImportError as e:
                    logger.warning(f"CrossSection import failed: {e}")
                except Exception as e:
                    # Enhanced error logging for StartSurface and similar failures
                    # "Object reference not set" on StartSurface means OpenCrossSectionExport() returned NULL
                    error_str = str(e)
                    diag_info = []
                    try:
                        diag_info.append(f"Mode={self.oss.Mode}")
                        diag_info.append(f"Fields={self.oss.SystemData.Fields.NumberOfFields}")
                        diag_info.append(f"Wavelengths={self.oss.SystemData.Wavelengths.NumberOfWavelengths}")
                        diag_info.append(f"EPD={self.oss.SystemData.Aperture.ApertureValue}")
                        diag_info.append(f"Surfaces={self.oss.LDE.NumberOfSurfaces}")
                    except Exception as diag_e:
                        diag_info.append(f"DiagError={diag_e}")

                    if "StartSurface" in error_str or "Object reference" in error_str:
                        logger.warning(f"CrossSection export failed (likely OpenCrossSectionExport returned NULL): {error_str}")
                        logger.warning(f"System state: {', '.join(diag_info)}")
                    else:
                        logger.warning(f"CrossSection export failed: {error_str} | {', '.join(diag_info)}")
                finally:
                    # Clean up temp file if it exists
                    if os.path.exists(temp_path):
                        try:
                            os.remove(temp_path)
                        except OSError as e:
                            logger.warning(f"Failed to clean up temp file {temp_path}: {e}")

        except Exception as e:
            # Log but don't fail - we'll return surface geometry as fallback
            logger.warning(f"CrossSection image export failed: {e}")
            logger.debug(f"Full traceback:", exc_info=True)

        # Get paraxial data using direct LDE access (more reliable)
        paraxial = self._get_paraxial_from_lde()

        # Get surface geometry for client-side rendering (fallback or supplement)
        surfaces_data = self._get_surface_geometry()

        # Count rays
        num_fields = self.oss.SystemData.Fields.NumberOfFields
        rays_total = DEFAULT_NUM_CROSS_SECTION_RAYS * max(1, num_fields)
        rays_through = rays_total  # Assume success

        return {
            "image": image_b64,
            "image_format": image_format,
            "array_shape": array_shape,  # For numpy_array reconstruction
            "array_dtype": array_dtype,  # For numpy_array reconstruction
            "paraxial": paraxial,
            "surfaces": surfaces_data,  # Always include for fallback rendering
            "rays_total": rays_total,
            "rays_through": rays_through,
        }

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

            # Get aperture info
            aperture = self.oss.SystemData.Aperture
            paraxial["epd"] = aperture.ApertureValue

            # Get field info
            fields = self.oss.SystemData.Fields
            max_field = 0.0
            if fields.NumberOfFields > 0:
                for i in range(1, fields.NumberOfFields + 1):
                    f = fields.GetField(i)
                    max_field = max(max_field, abs(f.Y), abs(f.X))
            paraxial["max_field"] = max_field
            paraxial["field_type"] = "object_angle"
            paraxial["field_unit"] = "deg"

            # Calculate total track from LDE
            lde = self.oss.LDE
            total_track = 0.0
            for i in range(1, lde.NumberOfSurfaces):
                surface = lde.GetSurfaceAt(i)
                total_track += abs(surface.Thickness)
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
            radius = surface.Radius
            surf_data = {
                "index": i,
                "z": z_position,
                "radius": radius if radius != 0 and abs(radius) < 1e10 else None,
                "thickness": surface.Thickness,
                "semi_diameter": surface.SemiDiameter,
                "conic": surface.Conic,
                "material": str(surface.Material) if surface.Material else None,
                "is_stop": surface.IsStop,
            }
            surfaces.append(surf_data)
            z_position += surface.Thickness

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
        for i in range(1, lde.NumberOfSurfaces):
            surface = lde.GetSurfaceAt(i)
            sd = surface.SemiDiameter

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
        surface_semi_diameters = []
        for i in range(1, lde.NumberOfSurfaces):
            surface = lde.GetSurfaceAt(i)
            surface_semi_diameters.append(surface.SemiDiameter)

        # Calculate grid size from num_rays
        grid_size = int(np.sqrt(num_rays))

        # Collect raw ray results
        raw_rays = []

        for fi in range(1, num_fields + 1):
            field = fields.GetField(fi)
            field_x, field_y = field.X, field.Y

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
                        ray_trace = zp.analyses.raysandspots.SingleRayTrace(
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
                                        ray_result["failed_surface"] = int(first_error.get('surface', 0))
                                        # Map error code to failure mode string
                                        error_code = int(first_error.get('error_code', 0))
                                        ray_result["failure_mode"] = self._error_code_to_mode(error_code)
                                    else:
                                        ray_result["reached_image"] = True
                                else:
                                    # No error column - assume success
                                    ray_result["reached_image"] = True
                            else:
                                ray_result["failure_mode"] = "NO_DATA"
                        else:
                            ray_result["failure_mode"] = "NO_RESULT"

                    except Exception as e:
                        logger.debug(f"Ray trace failed for field {fi}, pupil ({px:.2f}, {py:.2f}): {e}")
                        ray_result["failure_mode"] = "EXCEPTION"

                    raw_rays.append(ray_result)

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
            zernike_analysis = zp.analyses.wavefront.ZernikeStandardCoefficients(
                sampling=DEFAULT_SAMPLING,
                maximum_term=DEFAULT_MAX_ZERNIKE_TERM,
                wavelength=1,
                field=1,
                surface="Image",
            )
            result = zernike_analysis.run(self.oss)

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
        Get native Seidel coefficients using OpticStudio's SeidelCoefficients analysis.

        This is a "dumb executor" - returns raw per-surface data only.
        Uses GetTextFile() to extract full Seidel data including:
        - Per-surface S1-S5 coefficients
        - Per-surface chromatic: CLA (axial), CTR (transverse)
        - Header data: Petzval radius, wavelength, optical invariant

        Note: System must be pre-loaded via load_zmx_file().

        Returns:
            On success: {
                "success": True,
                "per_surface": [
                    {"surface": 1, "S1": float, "S2": float, "S3": float,
                     "S4": float, "S5": float, "CLA": float, "CTR": float},
                    ...
                ],
                "totals": {"S1": float, "S2": float, "S3": float,
                           "S4": float, "S5": float, "CLA": float, "CTR": float},
                "header": {
                    "petzval_radius": float,
                    "wavelength_um": float,
                    "optical_invariant": float,
                },
                "num_surfaces": int,
            }
            On error: {"success": False, "error": "..."}
        """
        analysis = None
        temp_path = os.path.join(tempfile.gettempdir(), SEIDEL_TEMP_FILENAME)

        try:
            idm = zp.constants.Analysis.AnalysisIDM

            # Create and run SeidelCoefficients analysis
            analysis = zp.analyses.new_analysis(
                self.oss,
                idm.SeidelCoefficients,
                settings_first=True
            )
            analysis.ApplyAndWaitForCompletion()

            # Check for error messages in analysis results
            error_msg = self._check_analysis_errors(analysis)
            if error_msg:
                return {"success": False, "error": error_msg}

            # Export to text file and parse
            analysis.Results.GetTextFile(temp_path)

            if not os.path.exists(temp_path):
                return {"success": False, "error": "GetTextFile did not create output file"}

            text_content = self._read_opticstudio_text_file(temp_path)
            if not text_content:
                return {"success": False, "error": "Seidel text output is empty"}

            # Parse the text content
            parsed = self._parse_seidel_text(text_content)

            per_surface = parsed.get("per_surface", [])
            totals = parsed.get("totals", {})

            # Validate that we actually got data - empty results indicate analysis failure
            if not per_surface and not totals:
                logger.warning("Seidel text file parsed but no coefficient data found")
                logger.debug(f"Text content preview: {text_content[:500]}")
                return {
                    "success": False,
                    "error": "Seidel analysis produced no coefficient data (system may be invalid or have no optical power)"
                }

            # Also check if totals are all zeros (another sign of failure)
            if totals and all(abs(totals.get(key, 0.0)) < 1e-15 for key in SEIDEL_COEFFICIENT_KEYS[:5]):
                logger.warning("Seidel totals are all zeros - likely invalid system")

            return {
                "success": True,
                "per_surface": per_surface,
                "totals": totals,
                "header": parsed.get("header", {}),
                "num_surfaces": len(per_surface),
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

    def _parse_seidel_text(self, text: str) -> dict[str, Any]:
        """
        Parse Seidel text output from OpticStudio's SeidelCoefficients analysis.

        Expected format (from GetTextFile):
            Listing of Aberration Coefficient Data
            ...
            Wavelength                      :               0.5876 µm
            Petzval radius                  :            -623.2905
            Optical Invariant               :               4.1551
            ...
            Seidel Aberration Coefficients:

            Surf     SPHA  S1    COMA  S2    ASTI  S3    FCUR  S4    DIST  S5    CLA    CTR
              1      0.114175   -0.008176   0.000586   0.120467   -0.008669   -0.034   0.002
              2      0.000396   -0.005093   0.065502  -0.042772   -0.292349   -0.002   0.029
            ...
            Sum      0.xxxxx    ...

        Args:
            text: Raw text from GetTextFile()

        Returns:
            Dict with keys: header, per_surface, totals
        """
        lines = text.strip().split('\n')

        header: dict[str, Any] = {}
        per_surface: list[dict[str, Any]] = []
        totals: dict[str, float] = {}
        in_table = False
        found_totals = False

        for line in lines:
            line_stripped = line.strip()
            if not line_stripped:
                continue

            # Stop parsing after we find the totals row (TOT or Sum)
            # This prevents parsing the "Seidel Aberration Coefficients in Waves" table
            if found_totals:
                break

            # Parse header values (lines with colon before table)
            if ':' in line_stripped and not in_table:
                self._parse_header_line(line_stripped, header)

            # Detect table header row (contains SPHA or both S1 and S2)
            if 'SPHA' in line_stripped or ('S1' in line_stripped and 'S2' in line_stripped):
                in_table = True
                continue

            # Parse data rows
            if in_table:
                found_totals = self._parse_seidel_data_row(line_stripped, per_surface, totals)

        return {
            "header": header,
            "per_surface": per_surface,
            "totals": totals,
        }

    def _parse_header_line(self, line: str, header: dict[str, Any]) -> None:
        """
        Parse a header line from Seidel text output.

        Args:
            line: Line containing key: value format
            header: Dict to update with parsed value
        """
        parts = line.split(':', 1)
        if len(parts) != 2:
            return

        key = parts[0].strip().lower().replace(' ', '_')
        value = parts[1].strip()

        # Try to extract numeric value (handle units like "0.5876 µm")
        try:
            numeric_part = value.split()[0] if value else ''
            header[key] = float(numeric_part)
        except (ValueError, IndexError):
            header[key] = value

    def _parse_seidel_data_row(
        self,
        line: str,
        per_surface: list[dict[str, Any]],
        totals: dict[str, float],
    ) -> bool:
        """
        Parse a data row from Seidel coefficient table.

        Args:
            line: Line from the Seidel table
            per_surface: List to append surface data to
            totals: Dict to update with totals row

        Returns:
            True if this was the totals row (TOT/Sum), False otherwise
        """
        parts = line.split()
        if not parts:
            return False

        first_part = parts[0].upper()
        values = self._extract_floats(parts[1:])

        if first_part.isdigit() or first_part == 'STO':
            # Surface data row - handle both numeric surface numbers and "STO" (aperture stop)
            # STO is typically the aperture stop surface; we use surface number 0 as a marker
            surface_num = int(parts[0]) if first_part.isdigit() else 0
            surface_data = self._build_seidel_coefficients(surface_num, values)
            per_surface.append(surface_data)
            return False
        elif first_part == 'IMA':
            # Image surface - skip it (usually all zeros)
            return False
        elif first_part in ('TOT', 'SUM'):
            # Totals row - parse and signal that we're done with this table
            totals.update(self._build_seidel_totals(values))
            return True

        return False

    def _extract_floats(self, parts: list[str]) -> list[float]:
        """
        Extract float values from string parts, skipping non-numeric values.

        Args:
            parts: List of string parts to parse

        Returns:
            List of successfully parsed float values
        """
        values: list[float] = []
        for p in parts:
            try:
                values.append(float(p))
            except ValueError:
                continue
        return values

    def _build_seidel_coefficients(self, surface_num: int, values: list[float]) -> dict[str, Any]:
        """
        Build a per-surface Seidel coefficient dict.

        OpticStudio Seidel text output has paired columns:
        - 12 values: SPHA, S1, COMA, S2, ASTI, S3, FCUR, S4, DIST, S5, CLA, CTR
          Paired values are duplicates, so extract indices 1, 3, 5, 7, 9, 10, 11
        - 10 values: SPHA, S1, COMA, S2, ASTI, S3, FCUR, S4, DIST, S5 (no chromatic)
          Extract indices 1, 3, 5, 7, 9
        - 7 values: S1, S2, S3, S4, S5, CLA, CTR (already unpaired)
          Use directly

        Args:
            surface_num: Surface number (1-indexed)
            values: List of coefficient values from the text row

        Returns:
            Dict with surface number and coefficient keys
        """
        surface_data: dict[str, Any] = {"surface": surface_num}
        num_values = len(values)

        if num_values >= 12:
            # Paired format with chromatic: SPHA, S1, COMA, S2, ASTI, S3, FCUR, S4, DIST, S5, CLA, CTR
            # S1=values[1], S2=values[3], S3=values[5], S4=values[7], S5=values[9], CLA=values[10], CTR=values[11]
            extracted = [values[1], values[3], values[5], values[7], values[9], values[10], values[11]]
            logger.debug(f"Surface {surface_num}: 12+ values, extracted paired format: {extracted[:5]}")
        elif num_values >= 10:
            # Paired format without chromatic: SPHA, S1, COMA, S2, ASTI, S3, FCUR, S4, DIST, S5
            # S1=values[1], S2=values[3], S3=values[5], S4=values[7], S5=values[9]
            extracted = [values[1], values[3], values[5], values[7], values[9], 0.0, 0.0]
            logger.debug(f"Surface {surface_num}: 10-11 values, extracted paired format: {extracted[:5]}")
        elif num_values >= 7:
            # Already unpaired format: S1, S2, S3, S4, S5, CLA, CTR
            extracted = values[:7]
            logger.debug(f"Surface {surface_num}: 7-9 values, using direct format: {extracted[:5]}")
        elif num_values >= 5:
            # Minimal format: S1, S2, S3, S4, S5
            extracted = values[:5] + [0.0, 0.0]
            logger.debug(f"Surface {surface_num}: 5-6 values, using minimal format: {extracted[:5]}")
        else:
            # Insufficient data
            logger.warning(f"Surface {surface_num}: Only {num_values} values, padding with zeros")
            extracted = values + [0.0] * (7 - num_values)

        for i, key in enumerate(SEIDEL_COEFFICIENT_KEYS):
            surface_data[key] = extracted[i] if i < len(extracted) else 0.0

        return surface_data

    def _build_seidel_totals(self, values: list[float]) -> dict[str, float]:
        """
        Build a Seidel totals dict.

        Uses same extraction logic as _build_seidel_coefficients to handle
        paired column format from OpticStudio.

        Args:
            values: List of total values from the Sum row

        Returns:
            Dict with coefficient keys and total values
        """
        num_values = len(values)

        if num_values >= 12:
            # Paired format with chromatic
            extracted = [values[1], values[3], values[5], values[7], values[9], values[10], values[11]]
            logger.debug(f"Totals: 12+ values, extracted paired format: {extracted[:5]}")
        elif num_values >= 10:
            # Paired format without chromatic
            extracted = [values[1], values[3], values[5], values[7], values[9], 0.0, 0.0]
            logger.debug(f"Totals: 10-11 values, extracted paired format: {extracted[:5]}")
        elif num_values >= 7:
            # Already unpaired format
            extracted = values[:7]
            logger.debug(f"Totals: 7-9 values, using direct format: {extracted[:5]}")
        elif num_values >= 5:
            # Minimal format
            extracted = values[:5] + [0.0, 0.0]
            logger.debug(f"Totals: 5-6 values, using minimal format: {extracted[:5]}")
        else:
            logger.warning(f"Totals: Only {num_values} values, padding with zeros")
            extracted = values + [0.0] * (7 - num_values)

        totals: dict[str, float] = {}
        for i, key in enumerate(SEIDEL_COEFFICIENT_KEYS):
            totals[key] = extracted[i] if i < len(extracted) else 0.0
        return totals

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
                        ray_trace = zp.analyses.raysandspots.SingleRayTrace(
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
                                    # Column names may be 'Y' or 'y' depending on version
                                    y_val = row.get('Y', row.get('y', None))
                                    z_val = row.get('Z', row.get('z', None))
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
            field_x, field_y = field.X, field.Y

            # Get wavelength for response
            wavelengths = self.oss.SystemData.Wavelengths
            if wavelength_index > wavelengths.NumberOfWavelengths:
                return {"success": False, "error": f"Wavelength index {wavelength_index} out of range (max: {wavelengths.NumberOfWavelengths})"}

            wavelength_um = wavelengths.GetWavelength(wavelength_index).Wavelength

            # Get wavefront metrics using ZernikeStandardCoefficients
            # This gives us RMS, P-V, and Strehl ratio
            rms_waves = None
            pv_waves = None
            strehl_ratio = None

            try:
                zernike_analysis = zp.analyses.wavefront.ZernikeStandardCoefficients(
                    sampling=sampling,
                    maximum_term=37,
                    wavelength=wavelength_index,
                    field=field_index,
                    reference_opd_to_vertex=False,
                    surface="Image",
                )
                zernike_result = zernike_analysis.run(self.oss)

                if hasattr(zernike_result, 'data') and zernike_result.data is not None:
                    zdata = zernike_result.data

                    # Get P-V wavefront error (in waves)
                    if hasattr(zdata, 'peak_to_valley_to_chief'):
                        pv_waves = float(zdata.peak_to_valley_to_chief)
                    elif hasattr(zdata, 'peak_to_valley_to_centroid'):
                        pv_waves = float(zdata.peak_to_valley_to_centroid)

                    # Get RMS and Strehl from integration data
                    if hasattr(zdata, 'from_integration_of_the_rays'):
                        integration = zdata.from_integration_of_the_rays
                        if hasattr(integration, 'rms_to_chief'):
                            rms_waves = float(integration.rms_to_chief)
                        elif hasattr(integration, 'rms_to_centroid'):
                            rms_waves = float(integration.rms_to_centroid)
                        if hasattr(integration, 'strehl_ratio'):
                            strehl_ratio = float(integration.strehl_ratio)

                    # Fallback: try direct attributes
                    if rms_waves is None and hasattr(zdata, 'rms'):
                        rms_waves = float(zdata.rms)
                    if pv_waves is None and hasattr(zdata, 'peak_to_valley'):
                        pv_waves = float(zdata.peak_to_valley)

            except Exception as e:
                logger.warning(f"ZernikeStandardCoefficients failed: {e}, trying WavefrontMap only")

            # Get wavefront map as numpy array
            image_b64 = None
            array_shape = None
            array_dtype = None

            try:
                wavefront_map = zp.analyses.wavefront.WavefrontMap(
                    sampling=sampling,
                    wavelength=wavelength_index,
                    field=field_index,
                    surface="Image",
                    show_as="Surface",
                    rotation="Rotate_0",
                    scale=1,
                    polarization=None,
                    reference_to_primary=False,
                    remove_tilt=False,
                    use_exit_pupil=True,
                ).run(self.oss, oncomplete="Release")

                if hasattr(wavefront_map, 'data') and wavefront_map.data is not None:
                    # Convert DataFrame or array to numpy
                    wf_data = wavefront_map.data

                    if hasattr(wf_data, 'values'):
                        # It's a DataFrame - extract values
                        arr = np.array(wf_data.values, dtype=np.float64)
                    else:
                        # It's already array-like
                        arr = np.array(wf_data, dtype=np.float64)

                    # Validate array dimensions
                    if arr.ndim >= 2:
                        image_b64 = base64.b64encode(arr.tobytes()).decode('utf-8')
                        array_shape = list(arr.shape)
                        array_dtype = str(arr.dtype)
                        logger.info(f"Wavefront map generated: shape={arr.shape}, dtype={arr.dtype}")

                        # If we didn't get metrics from Zernike, compute from the map
                        if pv_waves is None:
                            valid_data = arr[~np.isnan(arr)]
                            if len(valid_data) > 0:
                                pv_waves = float(np.max(valid_data) - np.min(valid_data))

                        if rms_waves is None:
                            valid_data = arr[~np.isnan(arr)]
                            if len(valid_data) > 0:
                                rms_waves = float(np.std(valid_data))
                    else:
                        logger.warning(f"WavefrontMap: unexpected array dimensions {arr.ndim}")

            except Exception as e:
                logger.warning(f"WavefrontMap failed: {e}")

            # If we have no metrics at all, return error
            if rms_waves is None and pv_waves is None:
                return {"success": False, "error": "Could not compute wavefront metrics"}

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
        Generate spot diagram using ZosPy's StandardSpot analysis.

        This is a "dumb executor" - returns raw data only. Image rendering
        (if PNG export fails) happens on Mac side.

        Note: System must be pre-loaded via load_zmx_file().

        Args:
            ray_density: Rays per axis (determines grid density, 1-20)
            reference: Reference point: 'chief_ray' or 'centroid'

        Returns:
            On success: {
                "success": True,
                "image": str (base64 PNG or numpy array),
                "image_format": "png" or "numpy_array",
                "array_shape": [h, w, c] (for numpy_array only),
                "array_dtype": str (for numpy_array only),
                "spot_data": [
                    {"field_index": 0, "field_x": 0, "field_y": 0,
                     "rms_radius": float, "geo_radius": float,
                     "centroid_x": float, "centroid_y": float, "num_rays": int},
                    ...
                ],
                "airy_radius": float,
            }
            On error: {"success": False, "error": "..."}
        """
        image_b64: Optional[str] = None
        image_format: Optional[str] = None
        array_shape: Optional[list[int]] = None
        array_dtype: Optional[str] = None
        spot_data: list[dict[str, Any]] = []
        airy_radius: Optional[float] = None
        analysis = None
        temp_path = os.path.join(tempfile.gettempdir(), SPOT_DIAGRAM_TEMP_FILENAME)

        # Get field info
        fields = self.oss.SystemData.Fields
        num_fields = fields.NumberOfFields

        if num_fields == 0:
            return {"success": False, "error": "System has no fields defined"}

        # Map reference parameter to OpticStudio constant (0 = Chief Ray, 1 = Centroid)
        reference_code = 0 if reference == "chief_ray" else 1

        try:
            # Use ZosPy's new_analysis to access StandardSpot directly
            analysis = zp.analyses.new_analysis(
                self.oss,
                zp.constants.Analysis.AnalysisIDM.StandardSpot,
                settings_first=True,
            )

            # Configure and run the analysis
            self._configure_spot_analysis(analysis.Settings, ray_density, reference_code)
            analysis.ApplyAndWaitForCompletion()

            # Try to export image to PNG
            image_b64, image_format = self._export_analysis_image(analysis, temp_path)

            # Extract spot data and airy radius from results
            if analysis.Results is not None:
                airy_radius = self._extract_airy_radius(analysis.Results)
                spot_data = self._extract_spot_data_from_results(analysis.Results, fields, num_fields)

        except Exception as e:
            logger.warning(f"StandardSpot analysis failed: {e}, falling back to manual ray trace")
        finally:
            self._cleanup_analysis(analysis, temp_path)

        # If we couldn't get spot data from StandardSpot, compute manually via ray trace
        if not spot_data or all(sd.get("rms_radius") is None for sd in spot_data):
            logger.info("Computing spot data via manual ray trace")
            spot_data = self._compute_spot_data_manual(ray_density, reference)

        # If we still don't have an image, create numpy array from ray positions
        if image_b64 is None and spot_data:
            image_b64, image_format, array_shape, array_dtype = self._create_spot_array_fallback(
                spot_data, ray_density
            )

        # Get Airy radius if not already obtained
        if airy_radius is None:
            airy_radius = self._calculate_airy_radius()

        return {
            "success": True,
            "image": image_b64,
            "image_format": image_format,
            "array_shape": array_shape,
            "array_dtype": array_dtype,
            "spot_data": spot_data,
            "airy_radius": airy_radius,
        }

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

    def _export_analysis_image(
        self,
        analysis: Any,
        temp_path: str,
    ) -> tuple[Optional[str], Optional[str]]:
        """
        Export an analysis to PNG image.

        Args:
            analysis: OpticStudio analysis object
            temp_path: Path for temporary PNG file

        Returns:
            Tuple of (base64_image, image_format) or (None, None) if export fails
        """
        try:
            if hasattr(analysis, 'ExportGraphicAs'):
                analysis.ExportGraphicAs(temp_path)
            elif hasattr(analysis, 'Results') and hasattr(analysis.Results, 'ExportData'):
                analysis.Results.ExportData(temp_path)

            if os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
                with open(temp_path, 'rb') as f:
                    image_b64 = base64.b64encode(f.read()).decode('utf-8')
                logger.info(f"Exported analysis PNG, size={len(image_b64)}")
                return image_b64, "png"

        except Exception as e:
            logger.warning(f"Analysis PNG export failed: {e}")

        return None, None

    def _extract_airy_radius(self, results: Any) -> Optional[float]:
        """
        Extract Airy disk radius from analysis results.

        Args:
            results: OpticStudio analysis results object

        Returns:
            Airy radius in lens units, or None if not available
        """
        try:
            if hasattr(results, 'AiryRadius'):
                return float(results.AiryRadius)
            elif hasattr(results, 'GetAiryDiskRadius'):
                return float(results.GetAiryDiskRadius())
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
                field_data = self._create_field_spot_data(fi, field.X, field.Y)

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
            if hasattr(results, 'GetDataSeries'):
                series = results.GetDataSeries(field_index)
                if series:
                    if hasattr(series, 'RMS'):
                        field_data["rms_radius"] = float(series.RMS)
                    if hasattr(series, 'GEO'):
                        field_data["geo_radius"] = float(series.GEO)

            if hasattr(results, 'SpotData'):
                spot_info = results.SpotData
                if hasattr(spot_info, 'GetSpotDataAtField'):
                    fd = spot_info.GetSpotDataAtField(field_index + 1)
                    if fd:
                        if hasattr(fd, 'RMSSpotRadius'):
                            field_data["rms_radius"] = float(fd.RMSSpotRadius)
                        if hasattr(fd, 'GEOSpotRadius'):
                            field_data["geo_radius"] = float(fd.GEOSpotRadius)
                        if hasattr(fd, 'CentroidX'):
                            field_data["centroid_x"] = float(fd.CentroidX)
                        if hasattr(fd, 'CentroidY'):
                            field_data["centroid_y"] = float(fd.CentroidY)
                        if hasattr(fd, 'NumberOfRays'):
                            field_data["num_rays"] = int(fd.NumberOfRays)

        except Exception as e:
            logger.debug(f"Could not get spot data for field {field_index}: {e}")

    def _create_spot_array_fallback(
        self,
        spot_data: list[dict[str, Any]],
        ray_density: int,
    ) -> tuple[Optional[str], Optional[str], Optional[list[int]], Optional[str]]:
        """
        Create numpy array fallback for spot diagram.

        Args:
            spot_data: List of spot data dicts
            ray_density: Rays per axis

        Returns:
            Tuple of (image_b64, image_format, array_shape, array_dtype)
        """
        try:
            arr = self._create_spot_diagram_array(spot_data, ray_density)
            if arr is not None:
                image_b64 = base64.b64encode(arr.tobytes()).decode('utf-8')
                logger.info(f"Created spot diagram numpy array, shape={arr.shape}")
                return image_b64, "numpy_array", list(arr.shape), str(arr.dtype)
        except Exception as e:
            logger.warning(f"Could not create spot diagram array: {e}")

        return None, None, None, None

    def _compute_spot_data_manual(
        self,
        ray_density: int,
        reference: str,
    ) -> list[dict[str, Any]]:
        """
        Compute spot diagram data manually by tracing rays.

        This is a fallback when StandardSpot analysis doesn't provide data.
        """
        spot_data = []
        fields = self.oss.SystemData.Fields
        num_fields = fields.NumberOfFields

        # Grid of rays across the pupil
        grid_size = ray_density * 2 + 1  # e.g., ray_density=5 -> 11x11 grid

        for fi in range(1, num_fields + 1):
            field = fields.GetField(fi)
            field_x, field_y = field.X, field.Y

            ray_x_positions = []
            ray_y_positions = []
            chief_ray_x: Optional[float] = None
            chief_ray_y: Optional[float] = None

            # Trace rays across the pupil
            for px in np.linspace(-1, 1, grid_size):
                for py in np.linspace(-1, 1, grid_size):
                    if px**2 + py**2 > 1:
                        continue  # Skip rays outside circular pupil

                    try:
                        ray_trace = zp.analyses.raysandspots.SingleRayTrace(
                            hx=0.0,
                            hy=0.0,
                            px=float(px),
                            py=float(py),
                            wavelength=1,
                            field=fi,
                        )
                        result = ray_trace.run(self.oss)

                        if hasattr(result, 'data') and result.data is not None:
                            ray_data = result.data
                            if hasattr(ray_data, 'real_ray_trace_data'):
                                df = ray_data.real_ray_trace_data
                                if hasattr(df, 'iloc') and len(df) > 0:
                                    # Get last row (image surface)
                                    last_row = df.iloc[-1]
                                    # Check if ray reached image (no error)
                                    error_code = last_row.get('error_code', 0) if hasattr(last_row, 'get') else 0
                                    if error_code == 0:
                                        x_val = last_row.get('X', last_row.get('x', None))
                                        y_val = last_row.get('Y', last_row.get('y', None))
                                        if x_val is not None and y_val is not None:
                                            ray_x_positions.append(float(x_val))
                                            ray_y_positions.append(float(y_val))
                                            # Capture chief ray position (ray at pupil center px=0, py=0)
                                            if abs(px) < 0.01 and abs(py) < 0.01:
                                                chief_ray_x = float(x_val)
                                                chief_ray_y = float(y_val)
                    except Exception as e:
                        logger.debug(f"Ray trace failed at ({px:.2f}, {py:.2f}): {e}")
                        continue

            # Calculate spot metrics
            num_rays = len(ray_x_positions)
            field_result = {
                "field_index": fi - 1,
                "field_x": field_x,
                "field_y": field_y,
                "rms_radius": None,
                "geo_radius": None,
                "centroid_x": None,
                "centroid_y": None,
                "num_rays": num_rays,
            }

            if num_rays > 0:
                x_arr = np.array(ray_x_positions)
                y_arr = np.array(ray_y_positions)

                # Compute centroid
                centroid_x = np.mean(x_arr)
                centroid_y = np.mean(y_arr)
                field_result["centroid_x"] = float(centroid_x)
                field_result["centroid_y"] = float(centroid_y)

                # Reference point for radius calculation
                if reference == "centroid":
                    ref_x, ref_y = centroid_x, centroid_y
                else:
                    # Chief ray reference - use actual chief ray position (px=0, py=0)
                    # Fall back to centroid if chief ray wasn't captured (e.g., vignetted)
                    if chief_ray_x is not None and chief_ray_y is not None:
                        ref_x, ref_y = chief_ray_x, chief_ray_y
                    else:
                        logger.debug(f"Field {fi}: Chief ray not captured, using centroid as fallback")
                        ref_x, ref_y = centroid_x, centroid_y

                # Calculate radii from reference point
                distances = np.sqrt((x_arr - ref_x)**2 + (y_arr - ref_y)**2)
                field_result["rms_radius"] = float(np.sqrt(np.mean(distances**2)))
                field_result["geo_radius"] = float(np.max(distances))

            spot_data.append(field_result)

        return spot_data

    def _create_spot_diagram_array(
        self,
        spot_data: list[dict[str, Any]],
        ray_density: int,
    ) -> Optional[np.ndarray]:
        """
        Create a numpy array visualization of the spot diagram.

        Returns a grayscale image with spots plotted.
        This is a simple fallback - Mac side can render better with matplotlib.
        """
        # This is a minimal implementation - just return None
        # and let Mac side render from spot_data
        # A full implementation would trace rays and plot them
        return None

    def _calculate_airy_radius(self) -> Optional[float]:
        """
        Calculate the Airy disk radius for the system.

        Airy radius = 1.22 * wavelength * f_number

        Returns:
            Airy disk radius in lens units (typically mm), or None if calculation fails
        """
        try:
            # Get wavelength (primary wavelength in micrometers)
            wavelength_um = self.oss.SystemData.Wavelengths.GetWavelength(1).Wavelength

            # Get f-number
            fno = self._get_fno()

            if fno and wavelength_um:
                # Convert wavelength from micrometers to millimeters (lens units typically mm)
                wavelength_mm = wavelength_um / 1000.0
                airy_radius = 1.22 * wavelength_mm * fno
                return float(airy_radius)

        except Exception as e:
            logger.debug(f"Could not calculate Airy radius: {e}")

        return None

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
            aperture_type = str(aperture.ApertureType).split(".")[-1] if aperture.ApertureType else ""

            if aperture_type in FNO_APERTURE_TYPES:
                return aperture.ApertureValue

            # Calculate from EPD and EFL
            epd = aperture.ApertureValue
            efl = self._get_efl()
            if epd and efl and epd > 0:
                return efl / epd

        except Exception as e:
            logger.debug(f"Could not get f-number: {e}")

        return None


class ZosPyError(Exception):
    """Exception raised when ZosPy operations fail."""
    pass
