"""
ZosPy Handler

Manages the connection to Zemax OpticStudio and executes ZosPy operations.
This module contains all the actual ZosPy/OpticStudio calls.

Note: This code runs on Windows only, where OpticStudio is installed.

Each uvicorn worker process gets its own OpticStudio connection (via ZOS singleton).
Multiple workers are supported — the constraint is license seats, not threading.
"""

import base64
import json
import logging
import math
import os
import re
import tempfile
import time
from typing import Any, Literal, Optional

import numpy as np

from utils.timing import log_timing

# Configure module logger
logger = logging.getLogger(__name__)

# Dedicated logger for raw Zemax analysis output (filterable in dashboard)
logger_raw = logging.getLogger("zemax.raw")

# Maximum characters per raw output log message
_RAW_LOG_MAX_CHARS = 4000

# Fields to summarize (show first N elements + count)
_ARRAY_SUMMARY_FIELDS = {
    "spot_data", "spot_rays", "data", "evaluated_rows",
    "zernike_coefficients", "raw_rays", "generated_rows",
    "surface_semi_diameters", "surfaces", "frequency",
    "diffraction_limit", "tangential", "sagittal",
}
_ARRAY_SUMMARY_MAX = 5

# Fields containing binary/large base64 data to skip
_BINARY_FIELDS = {"image", "zmx_content"}

# Text fields to truncate
_TEXT_TRUNCATE_FIELDS = {"seidel_text": 200}


def _summarize_value(key: str, value: Any) -> Any:
    """Summarize a single result field for raw output logging.

    Returns a log-friendly representation: truncates binary/text fields,
    summarizes long arrays, and describes ndarrays by shape.
    """
    if key in _BINARY_FIELDS and isinstance(value, str) and len(value) > 100:
        return f"<base64 {len(value)} chars>"

    if key in _TEXT_TRUNCATE_FIELDS:
        max_len = _TEXT_TRUNCATE_FIELDS[key]
        if isinstance(value, str) and len(value) > max_len:
            return value[:max_len] + f"... ({len(value)} chars total)"

    if key in _ARRAY_SUMMARY_FIELDS and isinstance(value, (list, tuple)):
        if len(value) > _ARRAY_SUMMARY_MAX:
            return {
                "_summary": f"{len(value)} items (showing first {_ARRAY_SUMMARY_MAX})",
                "items": value[:_ARRAY_SUMMARY_MAX],
            }

    if isinstance(value, np.ndarray):
        return f"ndarray(shape={value.shape}, dtype={value.dtype})"

    return value


def _log_raw_output(operation: str, result: dict[str, Any]) -> None:
    """Log raw Zemax analysis output at DEBUG level on the zemax.raw logger.

    Filters out binary fields, summarizes arrays, and truncates long text
    to keep log entries readable in the dashboard.
    """
    if not logger_raw.isEnabledFor(logging.DEBUG):
        return

    try:
        filtered = {k: _summarize_value(k, v) for k, v in result.items()}
        msg = json.dumps(
            filtered, indent=2,
            default=lambda obj: f"<{type(obj).__name__}>",
        )
        if len(msg) > _RAW_LOG_MAX_CHARS:
            msg = msg[:_RAW_LOG_MAX_CHARS] + f"\n... (truncated at {_RAW_LOG_MAX_CHARS} chars)"

        logger_raw.debug(f"[RAW] {operation} output:\n{msg}")
    except Exception as e:
        logger_raw.debug(f"[RAW] {operation}: failed to serialize output: {e}")

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

# Mapping from sampling grid strings to OpticStudio SampleSize enum values
SAMPLING_ENUM_MAP = {
    "32x32": 1, "64x64": 2, "128x128": 3,
    "256x256": 4, "512x512": 5, "1024x1024": 6,
}

# Image export settings
CROSS_SECTION_IMAGE_SIZE = (1200, 800)
CROSS_SECTION_TEMP_FILENAME = "zemax_cross_section.png"
SEIDEL_TEMP_FILENAME = "seidel_native.txt"

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


def _extract_value(obj: Any, default: float = 0.0, allow_inf: bool = False) -> float:
    """
    Extract a numeric value from various ZosPy types.

    ZosPy 2.x returns UnitField objects for many values instead of plain floats.
    This helper handles both UnitField and plain numeric types.

    Args:
        obj: Value to extract (UnitField, float, int, etc.)
        default: Default value if extraction fails
        allow_inf: If True, allow Infinity through (e.g. flat surface radius)

    Returns:
        Float value extracted from the object
    """
    if obj is None:
        return default

    # Unwrap UnitField objects (have .value attribute)
    value = obj.value if hasattr(obj, 'value') else obj

    try:
        result = float(value)
        if np.isnan(result):
            return default
        if np.isinf(result) and not allow_inf:
            return default
        return result
    except (TypeError, ValueError):
        return default


def _extract_dataframe(result: Any, label: str = "Analysis") -> Optional["pd.DataFrame"]:
    """
    Extract a pandas DataFrame from a ZOSPy AnalysisResult or typed result object.

    ZOSPy run() returns an AnalysisResult wrapper whose .data holds a typed result
    (e.g. RayFanResult) with to_dataframe(), or sometimes a raw DataFrame directly.
    This helper walks the wrapper chain: result.data.to_dataframe() > result.data
    (if DataFrame) > result.data.data > result.to_dataframe().

    Returns None if no DataFrame can be extracted.
    """
    import pandas as pd

    # Unwrap AnalysisResult.data if present
    data = getattr(result, 'data', None)
    if data is not None:
        if hasattr(data, 'to_dataframe'):
            return data.to_dataframe()
        if isinstance(data, pd.DataFrame):
            return data
        # Some typed results nest the DataFrame one level deeper
        nested = getattr(data, 'data', None)
        if isinstance(nested, pd.DataFrame):
            return nested
        logger.debug(f"{label}: .data is {type(data).__name__}, not extractable")
        return None

    # Direct to_dataframe() on the result itself
    if hasattr(result, 'to_dataframe'):
        return result.to_dataframe()

    return None


def _parse_zernike_term_number(col: Any) -> Optional[int]:
    """
    Parse a Zernike term number from a column name.

    Handles pure integers ("1", "37"), Z-prefixed names ("Z1", "Z 4", "Z04"),
    and case-insensitive variants.

    Returns the term number as int, or None if the column is not a Zernike term.
    """
    try:
        return int(col)
    except (ValueError, TypeError):
        m = re.match(r'^Z\s*(\d+)$', str(col), re.IGNORECASE)
        if m:
            return int(m.group(1))
    return None


def _read_comment_cell(op: Any, comment_column: Any) -> Optional[str]:
    """Read the Comment cell from an MFE operand, returning stripped text or None."""
    try:
        raw = op.GetOperandCell(comment_column).Value
        if raw and str(raw).strip():
            return str(raw).strip()
    except Exception:
        pass
    return None


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
            # Clean up the OpticStudio process that ZOS() may have launched,
            # otherwise it becomes an orphan consuming a license seat.
            self.close()
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

    def get_paraxial(self) -> dict[str, Any]:
        """
        Get comprehensive first-order optical properties in a single call.

        Combines EFL, BFL, F/#, NA, EPD, total track, and FOV info from
        SystemData analysis and LDE.

        Returns:
            Dict with success flag and paraxial properties.
        """
        try:
            # Get EFL and BFL from SystemData analysis
            try:
                sys_data = self._zp.analyses.reports.SystemData().run(self.oss)
                gld = sys_data.data.general_lens_data
                efl = _extract_value(gld.effective_focal_length_air, None)
                bfl = _extract_value(
                    getattr(gld, "back_focal_length", None), None
                )
            except Exception as e:
                logger.warning(f"get_paraxial: SystemData failed: {e}")
                efl = self._get_efl()
                bfl = None

            fno = self._get_fno()
            na = 1.0 / (2.0 * fno) if fno is not None and fno > 0 else None

            # Get EPD, total track, field info from LDE
            lde_data = self._get_paraxial_from_lde()
            max_field = lde_data.get("max_field")
            field_type = lde_data.get("field_type")

            # Compute image height from EFL and max field angle
            image_height = None
            if (
                efl is not None
                and max_field is not None
                and max_field > 0
                and field_type == "object_angle"
            ):
                image_height = abs(efl) * math.tan(math.radians(max_field))

            result = {
                "success": True,
                "efl": efl,
                "bfl": bfl,
                "fno": fno,
                "na": na,
                "epd": lde_data.get("epd"),
                "total_track": lde_data.get("total_track"),
                "max_field": max_field,
                "field_type": field_type,
                "field_unit": lde_data.get("field_unit"),
                "image_height": image_height,
            }
            _log_raw_output("/paraxial", result)
            return result

        except Exception as e:
            logger.error(f"get_paraxial failed: {e}")
            return {"success": False, "error": str(e)}

    def get_cardinal_points(self) -> dict[str, Any]:
        """
        Get cardinal points of the optical system.

        Reports principal planes, nodal points, focal points, anti-principal
        planes, and anti-nodal planes for both object and image space.

        Returns:
            Dict with success flag and cardinal points data.
        """
        try:
            result_obj = self._zp.analyses.reports.CardinalPoints().run(self.oss)
            data = result_obj.data

            # Build a flat list of cardinal point entries
            cardinal_points = []
            spec = data.cardinal_points

            # Each attribute of CardinalPointSpecification is a CardinalPoint
            # with .object (Object Space) and .image (Image Space) values.
            point_names = [
                ("Focal Length", spec.focal_length),
                ("Focal Planes", spec.focal_planes),
                ("Principal Planes", spec.principal_planes),
                ("Anti-Principal Planes", spec.anti_principal_planes),
                ("Nodal Planes", spec.nodal_planes),
                ("Anti-Nodal Planes", spec.anti_nodal_planes),
            ]

            for name, point in point_names:
                cardinal_points.append({
                    "name": f"{name} (Object)",
                    "value": float(point.object) if point.object is not None else None,
                    "units": str(data.lens_units) if data.lens_units else "mm",
                })
                cardinal_points.append({
                    "name": f"{name} (Image)",
                    "value": float(point.image) if point.image is not None else None,
                    "units": str(data.lens_units) if data.lens_units else "mm",
                })

            result = {
                "success": True,
                "cardinal_points": cardinal_points,
                "starting_surface": data.starting_surface,
                "ending_surface": data.ending_surface,
                "wavelength": float(data.wavelength) if data.wavelength is not None else None,
                "orientation": str(data.orientation) if data.orientation else None,
                "lens_units": str(data.lens_units) if data.lens_units else "mm",
            }
            _log_raw_output("/cardinal-points", result)
            return result

        except Exception as e:
            logger.error(f"get_cardinal_points failed: {e}")
            return {"success": False, "error": str(e)}

    def get_cross_section(
        self,
        number_of_rays: int = DEFAULT_NUM_CROSS_SECTION_RAYS,
        color_rays_by: Literal["Fields", "Wavelengths", "None"] = "Fields",
    ) -> dict[str, Any]:
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
                number_of_rays=number_of_rays,
                field="All",
                wavelength="All",
                color_rays_by=color_rays_by,
                delete_vignetted=True,
                surface_line_thickness="Thick",
                rays_line_thickness="Standard",
                image_size=CROSS_SECTION_IMAGE_SIZE,
            )

            cs_start = time.perf_counter()
            try:
                cross_section.run(self.oss, image_output_file=temp_path)
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

            rays_total = number_of_rays * max(1, num_fields)

            result = {
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
            _log_raw_output("/cross-section", result)
            return result

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

            result = {
                "success": True,
                "zernike_coefficients": coefficients,
                "wavelength_um": wl_um,
                "num_surfaces": num_surfaces,
            }
            _log_raw_output("/seidel", result)
            return result

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

            result = {
                "success": True,
                "seidel_text": text_content,
                "num_surfaces": num_surfaces,
            }
            _log_raw_output("/seidel-native", result)
            return result

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

        result = {
            "num_surfaces": num_surfaces,
            "num_fields": num_fields,
            "num_wavelengths": num_wavelengths,
            "data": data,
        }
        _log_raw_output("/trace-rays", result)
        return result

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

            result = {
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
            _log_raw_output("/wavefront", result)
            return result

        except Exception as e:
            return {"success": False, "error": f"Wavefront analysis failed: {e}"}

    def get_spot_diagram(
        self,
        ray_density: int = 5,
        reference: str = "chief_ray",
        field_index: Optional[int] = None,
        wavelength_index: Optional[int] = None,
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
            field_index: Field index (1-indexed). None = all fields.
            wavelength_index: Wavelength index (1-indexed). None = all wavelengths.

        Returns:
            On success: {
                "success": True,
                "image": None (not supported by ZOSAPI for StandardSpot),
                "image_format": None,
                "spot_data": [...] (per-field metrics in µm: RMS, GEO radius, centroid),
                "spot_rays": [...] (raw ray X,Y positions in µm for Mac-side rendering),
                "airy_radius": float (µm),
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
            logger.info(f"[SPOT] Starting: ray_density={ray_density}, reference={reference}, num_fields={num_fields}, field_index={field_index}, wavelength_index={wavelength_index}")

            # Use ZosPy's new_analysis to access StandardSpot for metrics
            analysis = self._zp.analyses.new_analysis(
                self.oss,
                self._zp.constants.Analysis.AnalysisIDM.StandardSpot,
                settings_first=True,
            )

            # Configure and run the analysis
            self._configure_spot_analysis(analysis.Settings, ray_density, reference_code, field_index, wavelength_index)
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
                logger.info(f"[SPOT] StandardSpot results: airy_radius={airy_radius}, fields={len(spot_data)}")
            else:
                logger.warning("[SPOT] StandardSpot analysis.Results is None")

            # Close analysis before batch ray trace
            self._cleanup_analysis(analysis, None)
            analysis = None

            # Get raw ray X,Y positions using batch ray tracing
            # (ZOSAPI doesn't expose raw ray data from StandardSpot)
            ray_trace_start = time.perf_counter()
            try:
                spot_rays = self._get_spot_ray_data(ray_density, field_index, wavelength_index)
            finally:
                ray_trace_elapsed_ms = (time.perf_counter() - ray_trace_start) * 1000
                log_timing(logger, "BatchRayTrace for spot diagram", ray_trace_elapsed_ms)

            total_ray_count = sum(len(e.get("rays", [])) for e in spot_rays)
            if total_ray_count == 0:
                logger.warning("[SPOT] No rays traced - check pupil coords / field normalization")

            logger.info(
                f"[SPOT] Returning: fields={len(spot_data)}, ray_entries={len(spot_rays)}, "
                f"total_rays={total_ray_count}, airy_radius={airy_radius}"
            )

            # Image is None: ZOSAPI StandardSpot doesn't support image export.
            # Mac side renders the spot diagram from spot_rays data.
            result = {
                "success": True,
                "image": None,
                "image_format": None,
                "array_shape": None,
                "array_dtype": None,
                "spot_data": spot_data,
                "spot_rays": spot_rays,
                "airy_radius": airy_radius,
            }
            _log_raw_output("/spot-diagram", result)
            return result

        except Exception as e:
            logger.error(f"[SPOT] StandardSpot analysis FAILED: {type(e).__name__}: {e}", exc_info=True)
            return {"success": False, "error": f"StandardSpot analysis failed: {e}"}
        finally:
            self._cleanup_analysis(analysis, None)

    def _configure_spot_analysis(
        self,
        settings: Any,
        ray_density: int,
        reference_code: int,
        field_index: Optional[int] = None,
        wavelength_index: Optional[int] = None,
    ) -> None:
        """
        Configure StandardSpot analysis settings.

        Args:
            settings: OpticStudio analysis settings object
            ray_density: Rays per axis (1-20)
            reference_code: Reference point (0=Chief Ray, 1=Centroid)
            field_index: Field index (1-indexed). None = all fields.
            wavelength_index: Wavelength index (1-indexed). None = all wavelengths.
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

        # Set field selection
        if hasattr(settings, 'Field'):
            try:
                if field_index is not None:
                    settings.Field.SetFieldNumber(field_index)
                else:
                    settings.Field.UseAllFields()
            except Exception as e:
                logger.error(f"[SPOT] Failed to configure field selection (field_index={field_index}): {e}")
                raise ValueError(f"Cannot configure field selection: {e}") from e

        # Set wavelength selection
        if hasattr(settings, 'Wavelength'):
            try:
                if wavelength_index is not None:
                    settings.Wavelength.SetWavelengthNumber(wavelength_index)
                else:
                    settings.Wavelength.UseAllWavelengths()
            except Exception as e:
                logger.error(f"[SPOT] Failed to configure wavelength selection (wavelength_index={wavelength_index}): {e}")
                raise ValueError(f"Cannot configure wavelength selection: {e}") from e

    def _get_spot_ray_data(
        self,
        ray_density: int,
        field_index: Optional[int] = None,
        wavelength_index: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        """
        Get raw ray X,Y positions at image plane using batch ray tracing.

        ZOSAPI's StandardSpot analysis does not expose raw ray positions through
        its Results interface. The only way to get ray X,Y data for rendering
        spot diagrams is via batch ray tracing (IBatchRayTrace).

        Ray positions are converted from lens units (mm) to µm at the source.

        Args:
            ray_density: Rays per axis (e.g., 5 means ~5x5 grid per field)
            field_index: Field index (1-indexed). None = all fields.
            wavelength_index: Wavelength index (1-indexed). None = all wavelengths.

        Returns:
            List of dicts per field, each containing:
                - field_index: int (0-based)
                - field_x, field_y: float (field coordinates)
                - wavelength_index: int (0-based)
                - rays: list of {"x": float, "y": float} in µm at image plane
        """
        spot_rays: list[dict[str, Any]] = []

        fields = self.oss.SystemData.Fields
        wavelengths = self.oss.SystemData.Wavelengths
        num_fields = fields.NumberOfFields
        num_wavelengths = wavelengths.NumberOfWavelengths

        # Determine which fields and wavelengths to trace
        if field_index is not None:
            field_indices = [field_index]
        else:
            field_indices = list(range(1, num_fields + 1))

        if wavelength_index is not None:
            wl_indices = [wavelength_index]
        else:
            wl_indices = list(range(1, num_wavelengths + 1))

        logger.info(f"[SPOT] Ray trace: density={ray_density}, fields={field_indices}, wavelengths={wl_indices}")

        # Generate pupil coordinates for ray grid (circular pupil pattern)
        pupil_coords = []
        for px in np.linspace(-1, 1, ray_density):
            for py in np.linspace(-1, 1, ray_density):
                if px**2 + py**2 <= 1.0:
                    pupil_coords.append((float(px), float(py)))
        logger.debug(f"[SPOT] Pupil grid: {len(pupil_coords)} rays from {ray_density}x{ray_density} grid")

        # Calculate max field extent for normalization (must use ALL fields,
        # not just filtered ones, because Hx/Hy are normalized to full extent)
        max_field_x = 0.0
        max_field_y = 0.0
        for fi in range(1, num_fields + 1):
            max_field_x = max(max_field_x, abs(_extract_value(fields.GetField(fi).X)))
            max_field_y = max(max_field_y, abs(_extract_value(fields.GetField(fi).Y)))
        logger.debug(f"[SPOT] Field extents: max_x={max_field_x}, max_y={max_field_y}")

        ray_trace = None
        try:
            # Use batch ray trace for efficiency
            ray_trace = self.oss.Tools.OpenBatchRayTrace()
            if ray_trace is None:
                logger.warning("Could not open BatchRayTrace tool")
                return spot_rays

            # Get the normalized unpolarized ray trace interface
            max_rays = len(field_indices) * len(wl_indices) * len(pupil_coords)
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

            # Add rays for selected field/wavelength/pupil combinations
            rays_added = 0
            for fi in field_indices:
                field = fields.GetField(fi)
                field_x_val = _extract_value(field.X)
                field_y_val = _extract_value(field.Y)
                hx_norm = float(field_x_val / max_field_x) if max_field_x > 1e-10 else 0.0
                hy_norm = float(field_y_val / max_field_y) if max_field_y > 1e-10 else 0.0
                logger.debug(f"[SPOT] Field {fi}: raw=({field_x_val}, {field_y_val}), norm=({hx_norm}, {hy_norm})")
                for wi in wl_indices:
                    for px, py in pupil_coords:
                        norm_unpol.AddRay(wi, hx_norm, hy_norm, float(px), float(py), opd_none)
                        rays_added += 1
            logger.debug(f"[SPOT] Added {rays_added} rays to batch trace (expected {max_rays})")

            # Run the ray trace
            ray_trace.RunAndWaitForCompletion()

            # Read results and organize by field/wavelength
            total_success = 0
            total_failed = 0
            for fi in field_indices:
                field = fields.GetField(fi)
                field_x = _extract_value(field.X)
                field_y = _extract_value(field.Y)

                for wi in wl_indices:
                    field_rays = {
                        "field_index": fi - 1,
                        "field_x": field_x,
                        "field_y": field_y,
                        "wavelength_index": wi - 1,
                        "rays": [],
                    }

                    entry_failed = 0
                    for _ in pupil_coords:
                        result = norm_unpol.ReadNextResult()
                        # ReadNextResult returns 15 values; we only need success(0), err_code(2), x(4), y(5)
                        success, err_code = result[0], result[2]
                        if success and err_code == 0:
                            field_rays["rays"].append({"x": float(result[4]) * 1000, "y": float(result[5]) * 1000})
                            total_success += 1
                        else:
                            total_failed += 1
                            entry_failed += 1

                    if entry_failed > 0:
                        logger.debug(f"[SPOT] Field {fi} wl {wi}: {len(field_rays['rays'])} OK, {entry_failed} failed")
                    spot_rays.append(field_rays)

            logger.info(f"[SPOT] Ray trace: {total_success} success, {total_failed} failed out of {rays_added} total")

        except Exception as e:
            logger.error(f"[SPOT] Batch ray trace FAILED: {type(e).__name__}: {e}", exc_info=True)
        finally:
            if ray_trace is not None:
                try:
                    ray_trace.Close()
                except Exception:
                    pass

        return spot_rays

    def _extract_airy_radius(self, results: Any) -> Optional[float]:
        """
        Extract or compute Airy disk radius.

        ZOSAPI's StandardSpot IAR_ results object does NOT expose AiryRadius
        directly. We compute it: r_airy = 1.22 * wavelength * f_number.
        Result is returned in micrometers (µm).

        Args:
            results: OpticStudio analysis results object (checked first for
                     direct property, then falls back to computation)

        Returns:
            Airy radius in µm, or None if not available
        """
        # Try direct property first (future ZOSAPI versions may add it)
        # Note: direct properties likely return lens units (mm), so convert to µm
        for attr_name, call in [("AiryRadius", False), ("GetAiryDiskRadius", True)]:
            try:
                if hasattr(results, attr_name):
                    val = _extract_value(getattr(results, attr_name)() if call else getattr(results, attr_name))
                    val_um = val * 1000  # mm -> µm
                    logger.info(f"[SPOT] Airy radius from {attr_name}: {val} mm -> {val_um} µm")
                    return val_um
            except Exception as e:
                logger.debug(f"[SPOT] {attr_name} extraction failed: {e}")

        # Compute from F/# and primary wavelength: r_airy = 1.22 * lambda * F/#
        # Using µm directly: 1.22 * wl_um * fno gives result in µm
        try:
            fno = self._get_fno()
            primary_wl_um = _extract_value(
                self.oss.SystemData.Wavelengths.GetWavelength(1).Wavelength,
                0.5876,
            )
            if fno and fno > 0:
                airy_radius_um = 1.22 * primary_wl_um * fno
                logger.info(f"[SPOT] Computed airy_radius: 1.22 * {primary_wl_um:.4f}µm * F/{fno:.2f} = {airy_radius_um:.3f} µm")
                return airy_radius_um
            logger.warning(f"[SPOT] Cannot compute airy radius: fno={fno}")
        except Exception as e:
            logger.warning(f"[SPOT] Could not compute Airy radius: {type(e).__name__}: {e}")
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

        The ZOSAPI StandardSpot results expose an IAR_SpotDataResultMatrix via
        results.SpotData with methods like GetRMSSpotSizeFor(field, wavelength).
        Field and wavelength indices are 1-based in the ZOSAPI.

        Args:
            results: OpticStudio analysis results object
            field_index: 0-indexed field number
            field_data: Dict to populate with spot metrics
        """
        try:
            if not hasattr(results, 'SpotData'):
                if field_index == 0:
                    logger.warning("[SPOT] results has no SpotData attribute")
                return

            spot_data = results.SpotData
            if field_index == 0:
                spot_attrs = [a for a in dir(spot_data) if not a.startswith('_')]
                logger.debug(f"[SPOT] SpotData type={type(spot_data).__name__}, attrs={spot_attrs}")

            # ZOSAPI SpotData methods are 1-indexed for both field and wavelength.
            # Query primary wavelength (index 1).
            fi_1 = field_index + 1
            wi = 1

            # RMS/GEO spot sizes are already in µm from ZOSAPI
            if hasattr(spot_data, 'GetRMSSpotSizeFor'):
                field_data["rms_radius"] = _extract_value(spot_data.GetRMSSpotSizeFor(fi_1, wi))
            if hasattr(spot_data, 'GetGeoSpotSizeFor'):
                field_data["geo_radius"] = _extract_value(spot_data.GetGeoSpotSizeFor(fi_1, wi))

            # Centroid coordinates are in mm from ZOSAPI, convert to µm
            if hasattr(spot_data, 'GetReferenceCoordinate_X_For'):
                field_data["centroid_x"] = _extract_value(spot_data.GetReferenceCoordinate_X_For(fi_1, wi)) * 1000
            if hasattr(spot_data, 'GetReferenceCoordinate_Y_For'):
                field_data["centroid_y"] = _extract_value(spot_data.GetReferenceCoordinate_Y_For(fi_1, wi)) * 1000

            logger.info(
                f"[SPOT] field[{field_index}]: rms={field_data.get('rms_radius')} µm, "
                f"geo={field_data.get('geo_radius')} µm, "
                f"centroid=({field_data.get('centroid_x')}, {field_data.get('centroid_y')}) µm"
            )

        except Exception as e:
            logger.warning(f"[SPOT] Could not get spot data for field {field_index}: {type(e).__name__}: {e}", exc_info=True)


    def get_mtf(
        self,
        field_index: int = 0,
        wavelength_index: int = 1,
        sampling: str = "64x64",
        maximum_frequency: float = 0.0,
    ) -> dict[str, Any]:
        """
        Get MTF (Modulation Transfer Function) data using ZosPy's FFT MTF analysis.

        This is a "dumb executor" — returns raw frequency/modulation data.
        Image rendering happens on the Mac side.

        Args:
            field_index: Field index (0 = all fields, 1+ = specific field, 1-indexed)
            wavelength_index: Wavelength index (1-indexed)
            sampling: Pupil sampling grid (e.g., '64x64', '128x128')
            maximum_frequency: Maximum spatial frequency (cycles/mm). 0 = auto.

        Returns:
            On success: {
                "success": True,
                "frequency": [...],
                "fields": [{"field_index": int, "field_x": float, "field_y": float,
                            "tangential": [...], "sagittal": [...]}],
                "diffraction_limit": [...],
                "cutoff_frequency": float,
                "wavelength_um": float,
            }
            On error: {"success": False, "error": "..."}
        """
        try:
            fields = self.oss.SystemData.Fields
            num_fields = fields.NumberOfFields
            wavelengths = self.oss.SystemData.Wavelengths

            if wavelength_index > wavelengths.NumberOfWavelengths:
                return {"success": False, "error": f"Wavelength index {wavelength_index} out of range (max: {wavelengths.NumberOfWavelengths})"}

            wavelength_um = _extract_value(wavelengths.GetWavelength(wavelength_index).Wavelength, 0.5876)

            # Determine which fields to analyze
            if field_index == 0:
                field_indices = list(range(1, num_fields + 1))
            else:
                if field_index > num_fields:
                    return {"success": False, "error": f"Field index {field_index} out of range (max: {num_fields})"}
                field_indices = [field_index]

            # Get F/# for diffraction limit calculation
            fno = self._get_fno()
            if fno is None or fno <= 0:
                fno = 5.0  # Fallback

            # Calculate cutoff frequency: fc = 1 / (wavelength_mm * fno)
            wavelength_mm = wavelength_um / 1000.0
            cutoff_frequency = 1.0 / (wavelength_mm * fno)

            all_fields_data = []
            frequency = None

            for fi in field_indices:
                field = fields.GetField(fi)
                field_x = _extract_value(field.X)
                field_y = _extract_value(field.Y)

                analysis = None
                try:
                    # Use new_analysis with FFTMtf
                    idm = self._zp.constants.Analysis.AnalysisIDM
                    analysis = self._zp.analyses.new_analysis(
                        self.oss,
                        idm.FftMtf,
                        settings_first=True,
                    )

                    # Configure settings
                    settings = analysis.Settings
                    self._configure_analysis_settings(
                        settings,
                        field_index=fi,
                        wavelength_index=wavelength_index,
                        sampling=sampling,
                    )
                    if maximum_frequency > 0 and hasattr(settings, 'MaximumFrequency'):
                        try:
                            settings.MaximumFrequency = maximum_frequency
                        except Exception:
                            pass

                    mtf_start = time.perf_counter()
                    try:
                        analysis.ApplyAndWaitForCompletion()
                    finally:
                        mtf_elapsed_ms = (time.perf_counter() - mtf_start) * 1000
                        log_timing(logger, f"FFTMtf.run (field={fi})", mtf_elapsed_ms)

                    # Extract data from results
                    results = analysis.Results
                    tangential = []
                    sagittal = []
                    freq_data = []

                    if results is not None:
                        # Try to get data series
                        try:
                            num_series = results.NumberOfDataSeries
                            logger.info(f"MTF field {fi}: {num_series} data series")
                            unclassified = []
                            for si in range(num_series):
                                series = results.GetDataSeries(si)
                                if series is None:
                                    continue

                                desc = str(series.Description) if hasattr(series, 'Description') else ""
                                desc_lower = desc.lower()
                                n_points = series.NumberOfPoints if hasattr(series, 'NumberOfPoints') else 0
                                logger.info(f"MTF field {fi} series {si}: desc='{desc}', points={n_points}")

                                # Skip diffraction limit series before extracting points
                                if "diffrac" in desc_lower or "limit" in desc_lower:
                                    logger.debug(f"MTF field {fi} series {si}: skipping (diffraction limit)")
                                    continue

                                series_x = []
                                series_y = []
                                for pi in range(n_points):
                                    pt = series.GetDataPoint(pi)
                                    if pt is not None:
                                        x_val = _extract_value(pt.X if hasattr(pt, 'X') else pt[0])
                                        y_val = _extract_value(pt.Y if hasattr(pt, 'Y') else pt[1])
                                        series_x.append(x_val)
                                        series_y.append(y_val)

                                # Classify series by description — match flexibly:
                                # OpticStudio uses "TS ..." for tangential, "SS ..." for sagittal
                                # but may also use full words or other prefixes depending on version
                                is_tangential = (
                                    desc_lower.startswith(("ts ", "ts,", "ts."))
                                    or "tangential" in desc_lower
                                    or desc_lower.startswith("t ")
                                )
                                is_sagittal = (
                                    desc_lower.startswith(("ss ", "ss,", "ss."))
                                    or "sagittal" in desc_lower
                                    or desc_lower.startswith("s ")
                                )

                                if is_tangential and not tangential:
                                    tangential = series_y
                                    if not freq_data:
                                        freq_data = series_x
                                elif is_sagittal and not sagittal:
                                    sagittal = series_y
                                    if not freq_data:
                                        freq_data = series_x
                                else:
                                    unclassified.append((series_x, series_y))

                            # Fallback: assign unclassified series to missing slots
                            if unclassified and (not tangential or not sagittal):
                                logger.info(f"MTF field {fi}: using positional fallback for {len(unclassified)} unclassified series")
                                idx = 0
                                if not tangential and idx < len(unclassified):
                                    x, tangential = unclassified[idx]
                                    if not freq_data:
                                        freq_data = x
                                    idx += 1
                                if not sagittal and idx < len(unclassified):
                                    _, sagittal = unclassified[idx]
                        except Exception as e:
                            logger.warning(f"MTF: Could not extract data series for field {fi}: {e}")

                    if frequency is None and freq_data:
                        frequency = freq_data

                    all_fields_data.append({
                        "field_index": fi - 1,  # Convert to 0-indexed
                        "field_x": field_x,
                        "field_y": field_y,
                        "tangential": tangential,
                        "sagittal": sagittal,
                    })

                except Exception as e:
                    logger.warning(f"MTF analysis failed for field {fi}: {e}")
                    all_fields_data.append({
                        "field_index": fi - 1,
                        "field_x": field_x,
                        "field_y": field_y,
                        "tangential": [],
                        "sagittal": [],
                    })
                finally:
                    if analysis is not None:
                        try:
                            analysis.Close()
                        except Exception:
                            pass

            # Generate frequency array if not obtained from analysis
            if frequency is None or len(frequency) == 0:
                max_freq = maximum_frequency if maximum_frequency > 0 else cutoff_frequency
                frequency = list(np.linspace(0, max_freq, 64))

            # Compute diffraction limit: MTF_dl(f) = (2/pi)[arccos(f/fc) - (f/fc)*sqrt(1-(f/fc)^2)]
            freq_arr = np.array(frequency)
            fn = np.clip(freq_arr / cutoff_frequency, 0.0, 1.0)
            dl = np.where(
                fn >= 1.0,
                0.0,
                (2.0 / np.pi) * (np.arccos(fn) - fn * np.sqrt(1.0 - fn * fn)),
            )
            diffraction_limit = dl.tolist()

            result = {
                "success": True,
                "frequency": freq_arr.tolist(),
                "fields": all_fields_data,
                "diffraction_limit": diffraction_limit,
                "cutoff_frequency": float(cutoff_frequency),
                "wavelength_um": float(wavelength_um),
            }
            _log_raw_output("/mtf", result)
            return result

        except Exception as e:
            return {"success": False, "error": f"MTF analysis failed: {e}"}

    def get_huygens_mtf(
        self,
        field_index: int = 0,
        wavelength_index: int = 1,
        sampling: str = "64x64",
        maximum_frequency: float = 0.0,
    ) -> dict[str, Any]:
        """
        Get Huygens MTF data using ZosPy's Huygens MTF analysis.

        More accurate than FFT MTF for systems with significant aberrations
        or tilted/decentered elements.

        This is a "dumb executor" — returns raw frequency/modulation data.
        Image rendering happens on the Mac side.

        Args:
            field_index: Field index (0 = all fields, 1+ = specific field, 1-indexed)
            wavelength_index: Wavelength index (1-indexed)
            sampling: Pupil sampling grid (e.g., '64x64', '128x128')
            maximum_frequency: Maximum spatial frequency (cycles/mm). 0 = auto (150).

        Returns:
            On success: {
                "success": True,
                "frequency": [...],
                "fields": [{"field_index": int, "field_x": float, "field_y": float,
                            "tangential": [...], "sagittal": [...]}],
                "diffraction_limit": [...],
                "cutoff_frequency": float,
                "wavelength_um": float,
            }
            On error: {"success": False, "error": "..."}
        """
        try:
            fields = self.oss.SystemData.Fields
            num_fields = fields.NumberOfFields
            wavelengths = self.oss.SystemData.Wavelengths

            if wavelength_index > wavelengths.NumberOfWavelengths:
                return {"success": False, "error": f"Wavelength index {wavelength_index} out of range (max: {wavelengths.NumberOfWavelengths})"}

            wavelength_um = _extract_value(wavelengths.GetWavelength(wavelength_index).Wavelength, 0.5876)

            # Determine which fields to analyze
            if field_index == 0:
                field_indices = list(range(1, num_fields + 1))
            else:
                if field_index > num_fields:
                    return {"success": False, "error": f"Field index {field_index} out of range (max: {num_fields})"}
                field_indices = [field_index]

            # Get F/# for diffraction limit calculation
            fno = self._get_fno()
            if fno is None or fno <= 0:
                fno = 5.0  # Fallback

            # Calculate cutoff frequency: fc = 1 / (wavelength_mm * fno)
            wavelength_mm = wavelength_um / 1000.0
            cutoff_frequency = 1.0 / (wavelength_mm * fno)

            all_fields_data = []
            frequency = None

            for fi in field_indices:
                field = fields.GetField(fi)
                field_x = _extract_value(field.X)
                field_y = _extract_value(field.Y)

                analysis = None
                try:
                    # Use new_analysis with HuygensMtf
                    idm = self._zp.constants.Analysis.AnalysisIDM
                    analysis = self._zp.analyses.new_analysis(
                        self.oss,
                        idm.HuygensMtf,
                        settings_first=True,
                    )

                    # Configure settings
                    settings = analysis.Settings
                    self._configure_analysis_settings(
                        settings,
                        field_index=fi,
                        wavelength_index=wavelength_index,
                        sampling=sampling,
                    )
                    # Set maximum frequency if specified
                    if maximum_frequency > 0 and hasattr(settings, 'MaximumFrequency'):
                        try:
                            settings.MaximumFrequency = maximum_frequency
                        except Exception:
                            pass

                    mtf_start = time.perf_counter()
                    try:
                        analysis.ApplyAndWaitForCompletion()
                    finally:
                        mtf_elapsed_ms = (time.perf_counter() - mtf_start) * 1000
                        log_timing(logger, f"HuygensMtf.run (field={fi})", mtf_elapsed_ms)

                    # Extract data from results (same structure as FFT MTF)
                    results = analysis.Results
                    tangential = []
                    sagittal = []
                    freq_data = []

                    if results is not None:
                        try:
                            num_series = results.NumberOfDataSeries
                            logger.debug(f"Huygens MTF field {fi}: {num_series} data series")
                            unclassified = []
                            for si in range(num_series):
                                series = results.GetDataSeries(si)
                                if series is None:
                                    continue

                                desc = str(series.Description) if hasattr(series, 'Description') else ""
                                desc_lower = desc.lower()
                                n_points = series.NumberOfPoints if hasattr(series, 'NumberOfPoints') else 0
                                logger.debug(f"Huygens MTF field {fi} series {si}: desc='{desc}', points={n_points}")

                                # Skip diffraction limit series before extracting points
                                if "diffrac" in desc_lower or "limit" in desc_lower:
                                    logger.debug(f"Huygens MTF field {fi} series {si}: skipping (diffraction limit)")
                                    continue

                                series_x = []
                                series_y = []
                                for pi in range(n_points):
                                    pt = series.GetDataPoint(pi)
                                    if pt is not None:
                                        x_val = _extract_value(pt.X if hasattr(pt, 'X') else pt[0])
                                        y_val = _extract_value(pt.Y if hasattr(pt, 'Y') else pt[1])
                                        series_x.append(x_val)
                                        series_y.append(y_val)

                                # Classify series by description prefix
                                if desc_lower.startswith(("ts ", "ts,", "tangential")):
                                    tangential = series_y
                                    if not freq_data:
                                        freq_data = series_x
                                elif desc_lower.startswith(("ss ", "ss,", "sagittal")):
                                    sagittal = series_y
                                    if not freq_data:
                                        freq_data = series_x
                                else:
                                    unclassified.append((series_x, series_y))

                            # Fallback: assign unclassified series to missing slots
                            if unclassified and (not tangential or not sagittal):
                                logger.info(f"Huygens MTF field {fi}: using positional fallback for {len(unclassified)} unclassified series")
                                idx = 0
                                if not tangential and idx < len(unclassified):
                                    x, tangential = unclassified[idx]
                                    if not freq_data:
                                        freq_data = x
                                    idx += 1
                                if not sagittal and idx < len(unclassified):
                                    _, sagittal = unclassified[idx]
                        except Exception as e:
                            logger.warning(f"Huygens MTF: Could not extract data series for field {fi}: {e}")

                    if frequency is None and freq_data:
                        frequency = freq_data

                    all_fields_data.append({
                        "field_index": fi - 1,
                        "field_x": field_x,
                        "field_y": field_y,
                        "tangential": tangential,
                        "sagittal": sagittal,
                    })

                except Exception as e:
                    logger.warning(f"Huygens MTF analysis failed for field {fi}: {e}")
                    all_fields_data.append({
                        "field_index": fi - 1,
                        "field_x": field_x,
                        "field_y": field_y,
                        "tangential": [],
                        "sagittal": [],
                    })
                finally:
                    if analysis is not None:
                        try:
                            analysis.Close()
                        except Exception:
                            pass

            # Generate frequency array if not obtained from analysis
            if frequency is None or len(frequency) == 0:
                max_freq = maximum_frequency if maximum_frequency > 0 else cutoff_frequency
                frequency = list(np.linspace(0, max_freq, 64))

            # Compute diffraction limit
            freq_arr = np.array(frequency)
            fn = np.clip(freq_arr / cutoff_frequency, 0.0, 1.0)
            dl = np.where(
                fn >= 1.0,
                0.0,
                (2.0 / np.pi) * (np.arccos(fn) - fn * np.sqrt(1.0 - fn * fn)),
            )
            diffraction_limit = dl.tolist()

            result = {
                "success": True,
                "frequency": freq_arr.tolist(),
                "fields": all_fields_data,
                "diffraction_limit": diffraction_limit,
                "cutoff_frequency": float(cutoff_frequency),
                "wavelength_um": float(wavelength_um),
            }
            _log_raw_output("/huygens-mtf", result)
            return result

        except Exception as e:
            return {"success": False, "error": f"Huygens MTF analysis failed: {e}"}

    def get_through_focus_mtf(
        self,
        sampling: str = "64x64",
        delta_focus: float = 0.1,
        frequency: float = 0.0,
        number_of_steps: int = 5,
        field_index: int = 0,
        wavelength_index: int = 1,
    ) -> dict[str, Any]:
        """
        Get Through Focus MTF data using ZosPy's FFTThroughFocusMTF analysis.

        Shows how MTF varies at different focus positions. Critical for
        understanding depth of focus and finding best focus.

        Args:
            sampling: Pupil sampling grid (e.g., '64x64', '128x128')
            delta_focus: Focus step size in mm
            frequency: Spatial frequency in cycles/mm (0 = use default)
            number_of_steps: Number of steps in each direction from focus (total = 2*steps+1)
            field_index: Field index (0 = all fields, 1+ = specific field, 1-indexed)
            wavelength_index: Wavelength index (1-indexed)

        Returns:
            On success: {
                "success": True,
                "focus_positions": [...],
                "fields": [{"field_index": int, "field_x": float, "field_y": float,
                            "tangential": [...], "sagittal": [...]}],
                "best_focus": {"position": float, "mtf_value": float},
                "frequency": float,
                "wavelength_um": float,
                "delta_focus": float,
            }
            On error: {"success": False, "error": "..."}
        """
        try:
            fields = self.oss.SystemData.Fields
            num_fields = fields.NumberOfFields
            wavelengths = self.oss.SystemData.Wavelengths

            if wavelength_index > wavelengths.NumberOfWavelengths:
                return {"success": False, "error": f"Wavelength index {wavelength_index} out of range (max: {wavelengths.NumberOfWavelengths})"}

            wavelength_um = _extract_value(wavelengths.GetWavelength(wavelength_index).Wavelength, 0.5876)

            # Determine which fields to analyze
            if field_index == 0:
                field_indices = list(range(1, num_fields + 1))
            else:
                if field_index > num_fields:
                    return {"success": False, "error": f"Field index {field_index} out of range (max: {num_fields})"}
                field_indices = [field_index]

            all_fields_data = []
            focus_positions = None
            best_focus_pos = 0.0
            best_focus_mtf = 0.0

            for fi in field_indices:
                field = fields.GetField(fi)
                field_x = _extract_value(field.X)
                field_y = _extract_value(field.Y)

                analysis = None
                try:
                    # Use new_analysis with FftThroughFocusMtf
                    idm = self._zp.constants.Analysis.AnalysisIDM
                    analysis = self._zp.analyses.new_analysis(
                        self.oss,
                        idm.FftThroughFocusMtf,
                        settings_first=True,
                    )

                    # Configure settings
                    settings = analysis.Settings
                    self._configure_analysis_settings(
                        settings,
                        field_index=fi,
                        wavelength_index=wavelength_index,
                        sampling=sampling,
                    )

                    # Set through-focus specific settings
                    if hasattr(settings, 'DeltaFocus'):
                        try:
                            settings.DeltaFocus = delta_focus
                        except Exception:
                            pass

                    if frequency > 0 and hasattr(settings, 'Frequency'):
                        try:
                            settings.Frequency = frequency
                        except Exception:
                            pass

                    if hasattr(settings, 'NumberOfSteps'):
                        try:
                            settings.NumberOfSteps = number_of_steps
                        except Exception:
                            pass

                    mtf_start = time.perf_counter()
                    try:
                        analysis.ApplyAndWaitForCompletion()
                    finally:
                        mtf_elapsed_ms = (time.perf_counter() - mtf_start) * 1000
                        log_timing(logger, f"FFTThroughFocusMtf.run (field={fi})", mtf_elapsed_ms)

                    # Extract data from results
                    results = analysis.Results
                    tangential = []
                    sagittal = []
                    focus_data = []

                    if results is not None:
                        try:
                            num_series = results.NumberOfDataSeries
                            logger.debug(f"TF-MTF field {fi}: {num_series} data series")
                            unclassified = []
                            for si in range(num_series):
                                series = results.GetDataSeries(si)
                                if series is None:
                                    continue

                                desc = str(series.Description) if hasattr(series, 'Description') else ""
                                desc_lower = desc.lower()
                                n_points = series.NumberOfPoints if hasattr(series, 'NumberOfPoints') else 0
                                logger.debug(f"TF-MTF field {fi} series {si}: desc='{desc}', points={n_points}")

                                series_x = []
                                series_y = []
                                for pi in range(n_points):
                                    pt = series.GetDataPoint(pi)
                                    if pt is not None:
                                        x_val = _extract_value(pt.X if hasattr(pt, 'X') else pt[0])
                                        y_val = _extract_value(pt.Y if hasattr(pt, 'Y') else pt[1])
                                        series_x.append(x_val)
                                        series_y.append(y_val)

                                # Classify by description: TS=tangential, SS=sagittal
                                if desc_lower.startswith(("ts ", "ts,", "tangential")):
                                    tangential = series_y
                                    if not focus_data:
                                        focus_data = series_x
                                elif desc_lower.startswith(("ss ", "ss,", "sagittal")):
                                    sagittal = series_y
                                    if not focus_data:
                                        focus_data = series_x
                                else:
                                    unclassified.append((series_x, series_y))

                            # Fallback: assign unclassified series to missing slots
                            if unclassified and (not tangential or not sagittal):
                                logger.info(f"TF-MTF field {fi}: using positional fallback for {len(unclassified)} unclassified series")
                                idx = 0
                                if not tangential and idx < len(unclassified):
                                    x, tangential = unclassified[idx]
                                    if not focus_data:
                                        focus_data = x
                                    idx += 1
                                if not sagittal and idx < len(unclassified):
                                    _, sagittal = unclassified[idx]
                        except Exception as e:
                            logger.warning(f"TF-MTF: Could not extract data series for field {fi}: {e}")

                    if focus_positions is None and focus_data:
                        focus_positions = focus_data

                    # Track best focus (highest average of T+S)
                    if tangential and sagittal:
                        for i_pt in range(min(len(tangential), len(sagittal))):
                            avg_mtf = (tangential[i_pt] + sagittal[i_pt]) / 2.0
                            if avg_mtf > best_focus_mtf and focus_data:
                                best_focus_mtf = avg_mtf
                                best_focus_pos = focus_data[i_pt] if i_pt < len(focus_data) else 0.0
                    elif tangential:
                        for i_pt, val in enumerate(tangential):
                            if val > best_focus_mtf and focus_data:
                                best_focus_mtf = val
                                best_focus_pos = focus_data[i_pt] if i_pt < len(focus_data) else 0.0

                    all_fields_data.append({
                        "field_index": fi - 1,  # Convert to 0-indexed
                        "field_x": field_x,
                        "field_y": field_y,
                        "tangential": tangential,
                        "sagittal": sagittal,
                    })

                except Exception as e:
                    logger.warning(f"TF-MTF analysis failed for field {fi}: {e}")
                    all_fields_data.append({
                        "field_index": fi - 1,
                        "field_x": field_x,
                        "field_y": field_y,
                        "tangential": [],
                        "sagittal": [],
                    })
                finally:
                    if analysis is not None:
                        try:
                            analysis.Close()
                        except Exception:
                            pass

            # Generate focus positions if not obtained from analysis
            if focus_positions is None or len(focus_positions) == 0:
                focus_positions = list(np.linspace(
                    -number_of_steps * delta_focus,
                    number_of_steps * delta_focus,
                    2 * number_of_steps + 1,
                ))

            result = {
                "success": True,
                "focus_positions": [float(x) for x in focus_positions],
                "fields": all_fields_data,
                "best_focus": {
                    "position": float(best_focus_pos),
                    "mtf_value": float(best_focus_mtf),
                },
                "frequency": float(frequency),
                "wavelength_um": float(wavelength_um),
                "delta_focus": float(delta_focus),
            }
            _log_raw_output("/through-focus-mtf", result)
            return result

        except Exception as e:
            return {"success": False, "error": f"Through Focus MTF analysis failed: {e}"}

    def get_rms_vs_field(
        self,
        ray_density: int = 5,
        num_field_points: int = 20,
        reference: str = "centroid",
        wavelength_index: int = 1,
    ) -> dict[str, Any]:
        """
        Get RMS spot radius vs field using native RmsField analysis.

        Uses OpticStudio's AnalysisIDM.RmsField which auto-samples across
        the full field range, producing a smooth curve.

        Args:
            ray_density: Ray density (1-20, maps to RayDens_N enum)
            num_field_points: Number of field sample points (snapped to nearest FieldDensity enum: 5,10,...,100)
            reference: Reference point ('centroid' or 'chief_ray')
            wavelength_index: Wavelength index (1-indexed)

        Returns:
            On success: {
                "success": True,
                "data": [{"field_value": float, "rms_radius_um": float}, ...],
                "diffraction_limit": [{"field_value": float, "rms_radius_um": float}, ...],
                "wavelength_um": float,
                "field_unit": str,
            }
            On error: {"success": False, "error": "..."}
        """
        try:
            wavelengths = self.oss.SystemData.Wavelengths
            if wavelength_index > wavelengths.NumberOfWavelengths:
                return {"success": False, "error": f"Wavelength index {wavelength_index} out of range (max: {wavelengths.NumberOfWavelengths})"}
            wavelength_um = _extract_value(wavelengths.GetWavelength(wavelength_index).Wavelength, 0.5876)

            # Determine field unit from system
            fields = self.oss.SystemData.Fields
            try:
                ft = fields.GetFieldType()
                field_type_str = getattr(ft, 'name', str(ft).split(".")[-1])
            except Exception:
                field_type_str = ""
            field_unit = "deg" if "angle" in field_type_str.lower() else "mm"

            # Snap num_field_points to nearest FieldDensity enum value (multiples of 5, 5-100)
            snapped = max(5, min(100, round(num_field_points / 5) * 5))
            # Clamp ray_density to 1-20
            ray_density = max(1, min(20, ray_density))

            analysis = None
            try:
                idm = self._zp.constants.Analysis.AnalysisIDM
                analysis = self._zp.analyses.new_analysis(
                    self.oss,
                    idm.RMSField,
                    settings_first=True,
                )

                settings = analysis.Settings
                rms_consts = self._zp.constants.Analysis.Settings.RMS

                def _set_setting(attr: str, value: Any, label: str = "") -> None:
                    """Set a setting attribute, logging warnings on failure."""
                    if not hasattr(settings, attr):
                        logger.info(f"RmsField: settings has no attribute '{attr}'")
                        return
                    try:
                        setattr(settings, attr, value)
                        logger.info(f"RmsField: Set {label or attr} = {value}")
                    except Exception as e:
                        logger.warning(f"RmsField: Could not set {label or attr}: {e}")

                def _get_enum_value(enum_cls: Any, name: str) -> Any:
                    """Get an enum value by name, returning None if not found."""
                    value = getattr(enum_cls, name, None)
                    if value is None:
                        logger.warning(f"RmsField: Enum value '{name}' not found on {enum_cls}")
                    return value

                # Data = SpotRadius (RMS spot radius)
                # DataType lives under RMSField sub-namespace, not directly under RMS
                data_type = _get_enum_value(rms_consts.RMSField.DataType, 'SpotRadius')
                if data_type is not None:
                    _set_setting('Data', data_type, 'Data type')

                # FieldDensity
                fd_value = _get_enum_value(rms_consts.FieldDensities, f"FieldDens_{snapped}")
                if fd_value is not None:
                    _set_setting('FieldDensity', fd_value)

                # RayDensity
                rd_value = _get_enum_value(rms_consts.RayDensities, f"RayDens_{ray_density}")
                if rd_value is not None:
                    _set_setting('RayDensity', rd_value)

                # ReferTo
                refer_name = "ChiefRay" if reference == "chief_ray" else "Centroid"
                refer_value = _get_enum_value(rms_consts.ReferTo, refer_name)
                if refer_value is not None:
                    _set_setting('ReferTo', refer_value)

                # Wavelength
                self._configure_analysis_settings(settings, wavelength_index=wavelength_index)

                # ShowDiffractionLimit
                _set_setting('ShowDiffractionLimit', True)

                rms_start = time.perf_counter()
                try:
                    analysis.ApplyAndWaitForCompletion()
                finally:
                    rms_elapsed_ms = (time.perf_counter() - rms_start) * 1000
                    log_timing(logger, "RmsField.run", rms_elapsed_ms)

                # Extract data series using IAR_DataSeries interface:
                # series.XData.Data = field values (1D array)
                # series.YData.Data = RMS values (2D matrix: [num_points, num_curves])
                # series.NumSeries = number of sub-curves
                # series.SeriesLabels = labels for each sub-curve
                results = analysis.Results
                data_points = []
                diffraction_limit = []

                if results is not None:
                    try:
                        num_series = results.NumberOfDataSeries
                        logger.info(f"RmsField: {num_series} data series returned")

                        for si in range(num_series):
                            series = results.GetDataSeries(si)
                            if series is None:
                                continue

                            desc = str(series.Description) if hasattr(series, 'Description') else ""
                            logger.info(f"RmsField series {si}: desc='{desc}'")

                            # Get X data (field values) from IVectorData
                            x_data = series.XData
                            if x_data is None:
                                logger.warning(f"RmsField series {si}: XData is None")
                                continue

                            x_raw = x_data.Data
                            if x_raw is None:
                                logger.warning(f"RmsField series {si}: XData.Data is None")
                                continue

                            x_values = list(x_raw)
                            num_points = len(x_values)

                            # Get Y data (RMS values) from IMatrixData
                            y_data = series.YData
                            if y_data is None:
                                logger.warning(f"RmsField series {si}: YData is None")
                                continue

                            y_raw = y_data.Data
                            num_curves = series.NumSeries if hasattr(series, 'NumSeries') else 1

                            # Get labels to identify diffraction limit curve
                            labels = []
                            if hasattr(series, 'SeriesLabels') and series.SeriesLabels is not None:
                                labels = list(series.SeriesLabels)

                            logger.info(f"RmsField series {si}: {num_points} points, {num_curves} curves, labels={labels}")

                            for ci in range(num_curves):
                                label = labels[ci] if ci < len(labels) else None
                                label_str = str(label) if label is not None else ""
                                label_lower = label_str.lower()
                                # Diffraction limit curve: label is None (unlabeled second curve)
                                # or contains "diffrac"/"limit"
                                is_diffraction = (
                                    "diffrac" in label_lower
                                    or "limit" in label_lower
                                    or (ci > 0 and label is None)
                                )

                                curve_points = []
                                for pi in range(num_points):
                                    x_val = float(x_values[pi])
                                    # YData is 2D: [point_index, curve_index]
                                    try:
                                        y_val = float(y_raw[pi, ci])
                                    except (TypeError, IndexError):
                                        # Might be 1D if only one curve
                                        try:
                                            y_val = float(y_raw[pi])
                                        except Exception:
                                            continue
                                    curve_points.append({
                                        "field_value": x_val,
                                        "rms_radius_um": y_val,
                                    })

                                logger.info(f"RmsField series {si} curve {ci}: label='{label}', {len(curve_points)} points, is_diffraction={is_diffraction}")

                                if is_diffraction:
                                    diffraction_limit = curve_points
                                else:
                                    data_points.extend(curve_points)

                    except Exception as e:
                        logger.warning(f"RmsField: Could not extract data series: {e}", exc_info=True)

                # Fallback: parse text output
                if not data_points:
                    logger.warning("RmsField: No data extracted from DataSeries, attempting text fallback")
                    try:
                        import tempfile, os
                        tmp = tempfile.NamedTemporaryFile(suffix='.txt', delete=False)
                        tmp_path = tmp.name
                        tmp.close()
                        results.GetTextFile(tmp_path)
                        with open(tmp_path, 'r') as f:
                            text_content = f.read()
                        os.unlink(tmp_path)
                        if text_content:
                            logger.info(f"RmsField text output (first 500 chars): {text_content[:500]}")
                    except Exception as e:
                        logger.warning(f"RmsField: Text fallback also failed: {e}")
                    return {"success": False, "error": "RmsField analysis returned no extractable data"}

                result = {
                    "success": True,
                    "data": data_points,
                    "diffraction_limit": diffraction_limit,
                    "wavelength_um": float(wavelength_um),
                    "field_unit": field_unit,
                }
                _log_raw_output("/rms-vs-field", result)
                return result

            finally:
                if analysis is not None:
                    try:
                        analysis.Close()
                    except Exception:
                        pass

        except Exception as e:
            return {"success": False, "error": f"RmsField analysis failed: {e}"}

    def get_psf(
        self,
        field_index: int = 1,
        wavelength_index: int = 1,
        sampling: str = "64x64",
    ) -> dict[str, Any]:
        """
        Get PSF (Point Spread Function) data using ZosPy's FFT PSF analysis.

        This is a "dumb executor" — returns raw 2D intensity grid as base64 numpy.
        Image rendering happens on the Mac side.

        Args:
            field_index: Field index (1-indexed)
            wavelength_index: Wavelength index (1-indexed)
            sampling: Pupil sampling grid (e.g., '64x64', '128x128')

        Returns:
            On success: {
                "success": True,
                "image": str (base64 numpy array),
                "image_format": "numpy_array",
                "array_shape": [h, w],
                "array_dtype": str,
                "strehl_ratio": float or None,
                "psf_peak": float or None,
                "wavelength_um": float,
                "field_x": float,
                "field_y": float,
            }
            On error: {"success": False, "error": "..."}
        """
        try:
            # Validate field and wavelength indices
            fields = self.oss.SystemData.Fields
            if field_index > fields.NumberOfFields:
                return {"success": False, "error": f"Field index {field_index} out of range (max: {fields.NumberOfFields})"}

            wavelengths = self.oss.SystemData.Wavelengths
            if wavelength_index > wavelengths.NumberOfWavelengths:
                return {"success": False, "error": f"Wavelength index {wavelength_index} out of range (max: {wavelengths.NumberOfWavelengths})"}

            field = fields.GetField(field_index)
            field_x = _extract_value(field.X)
            field_y = _extract_value(field.Y)
            wavelength_um = _extract_value(wavelengths.GetWavelength(wavelength_index).Wavelength, 0.5876)

            # Run FFT PSF analysis
            idm = self._zp.constants.Analysis.AnalysisIDM
            analysis = self._zp.analyses.new_analysis(
                self.oss,
                idm.FftPsf,
                settings_first=True,
            )

            try:
                # Configure settings
                settings = analysis.Settings
                self._configure_analysis_settings(
                    settings,
                    field_index=field_index,
                    wavelength_index=wavelength_index,
                    sampling=sampling,
                )

                psf_start = time.perf_counter()
                try:
                    analysis.ApplyAndWaitForCompletion()
                finally:
                    psf_elapsed_ms = (time.perf_counter() - psf_start) * 1000
                    log_timing(logger, "FftPsf.run", psf_elapsed_ms)

                # Extract 2D PSF data
                results = analysis.Results
                image_b64 = None
                array_shape = None
                array_dtype = None
                psf_peak = None

                if results is not None:
                    try:
                        # Try to get the data grid
                        num_rows = results.NumberOfDataGrids
                        if num_rows > 0:
                            grid = results.GetDataGrid(0)
                            if grid is not None:
                                nx = grid.Nx if hasattr(grid, 'Nx') else 0
                                ny = grid.Ny if hasattr(grid, 'Ny') else 0

                                if nx > 0 and ny > 0:
                                    arr = np.zeros((ny, nx), dtype=np.float64)
                                    for yi in range(ny):
                                        for xi in range(nx):
                                            val = _extract_value(grid.Z(xi, yi))
                                            arr[yi, xi] = val

                                    psf_peak = float(np.max(arr))
                                    image_b64 = base64.b64encode(arr.tobytes()).decode('utf-8')
                                    array_shape = list(arr.shape)
                                    array_dtype = str(arr.dtype)
                                    logger.info(f"PSF data extracted: shape={arr.shape}, peak={psf_peak:.6f}")
                    except Exception as e:
                        logger.warning(f"PSF: Could not extract data grid: {e}")
            finally:
                try:
                    analysis.Close()
                except Exception:
                    pass

            if image_b64 is None:
                return {"success": False, "error": "FFT PSF analysis did not produce data"}

            # Try to get Strehl ratio from Huygens PSF
            strehl_ratio = None
            huygens = None
            try:
                huygens = self._zp.analyses.new_analysis(
                    self.oss,
                    idm.HuygensPsf,
                    settings_first=True,
                )
                h_settings = huygens.Settings
                self._configure_analysis_settings(
                    h_settings,
                    field_index=field_index,
                    wavelength_index=wavelength_index,
                )

                huygens_start = time.perf_counter()
                try:
                    huygens.ApplyAndWaitForCompletion()
                finally:
                    huygens_elapsed_ms = (time.perf_counter() - huygens_start) * 1000
                    log_timing(logger, "HuygensPSF.run (Strehl)", huygens_elapsed_ms)

                h_results = huygens.Results
                if h_results is not None:
                    # Try to extract Strehl from header text
                    try:
                        header_text = h_results.HeaderData.Lines if hasattr(h_results, 'HeaderData') else ""
                        if hasattr(header_text, '__iter__'):
                            for line in header_text:
                                line_str = str(line).lower()
                                if "strehl" in line_str:
                                    parts = line_str.split(":")
                                    if len(parts) > 1:
                                        val_str = parts[-1].strip()
                                        try:
                                            strehl_ratio = float(val_str)
                                        except ValueError:
                                            pass
                                    break
                    except Exception:
                        pass
            except Exception as e:
                logger.debug(f"PSF: Huygens Strehl ratio extraction failed (non-critical): {e}")
            finally:
                if huygens is not None:
                    try:
                        huygens.Close()
                    except Exception:
                        pass

            result = {
                "success": True,
                "image": image_b64,
                "image_format": "numpy_array",
                "array_shape": array_shape,
                "array_dtype": array_dtype,
                "strehl_ratio": strehl_ratio,
                "psf_peak": psf_peak,
                "wavelength_um": float(wavelength_um),
                "field_x": field_x,
                "field_y": field_y,
            }
            _log_raw_output("/psf", result)
            return result

        except Exception as e:
            return {"success": False, "error": f"PSF analysis failed: {e}"}

    def get_huygens_psf(
        self,
        field_index: int = 1,
        wavelength_index: int = 1,
        sampling: str = "64x64",
    ) -> dict[str, Any]:
        """
        Get Huygens PSF data using ZOS-API's HuygensPsf analysis.

        More accurate than FFT PSF for highly aberrated systems because it uses
        direct integration of the Huygens wavelet at each point on the image surface.

        Returns the same structure as get_psf() so the Mac side can reuse
        the same render_psf_to_png renderer.

        Args:
            field_index: Field index (1-indexed)
            wavelength_index: Wavelength index (1-indexed)
            sampling: Pupil sampling grid (e.g., '64x64', '128x128')

        Returns:
            On success: {
                "success": True,
                "image": str (base64 numpy array),
                "image_format": "numpy_array",
                "array_shape": [h, w],
                "array_dtype": str,
                "strehl_ratio": float or None,
                "psf_peak": float or None,
                "wavelength_um": float,
                "field_x": float,
                "field_y": float,
            }
            On error: {"success": False, "error": "..."}
        """
        try:
            # Validate field and wavelength indices
            fields = self.oss.SystemData.Fields
            if field_index > fields.NumberOfFields:
                return {"success": False, "error": f"Field index {field_index} out of range (max: {fields.NumberOfFields})"}

            wavelengths = self.oss.SystemData.Wavelengths
            if wavelength_index > wavelengths.NumberOfWavelengths:
                return {"success": False, "error": f"Wavelength index {wavelength_index} out of range (max: {wavelengths.NumberOfWavelengths})"}

            field = fields.GetField(field_index)
            field_x = _extract_value(field.X)
            field_y = _extract_value(field.Y)
            wavelength_um = _extract_value(wavelengths.GetWavelength(wavelength_index).Wavelength, 0.5876)

            # Run Huygens PSF analysis
            idm = self._zp.constants.Analysis.AnalysisIDM
            analysis = self._zp.analyses.new_analysis(
                self.oss,
                idm.HuygensPsf,
                settings_first=True,
            )

            try:
                # Configure settings
                settings = analysis.Settings
                self._configure_analysis_settings(
                    settings,
                    field_index=field_index,
                    wavelength_index=wavelength_index,
                    sampling=sampling,
                )

                huygens_start = time.perf_counter()
                try:
                    analysis.ApplyAndWaitForCompletion()
                finally:
                    huygens_elapsed_ms = (time.perf_counter() - huygens_start) * 1000
                    log_timing(logger, "HuygensPsf.run", huygens_elapsed_ms)

                # Extract results
                results = analysis.Results
                image_b64 = None
                array_shape = None
                array_dtype = None
                psf_peak = None
                strehl_ratio = None

                if results is not None:
                    # Extract 2D PSF data grid
                    try:
                        num_grids = results.NumberOfDataGrids
                        if num_grids > 0:
                            grid = results.GetDataGrid(0)
                            if grid is not None:
                                nx = grid.Nx if hasattr(grid, 'Nx') else 0
                                ny = grid.Ny if hasattr(grid, 'Ny') else 0

                                if nx > 0 and ny > 0:
                                    arr = np.zeros((ny, nx), dtype=np.float64)
                                    for yi in range(ny):
                                        for xi in range(nx):
                                            val = _extract_value(grid.Z(xi, yi))
                                            arr[yi, xi] = val

                                    psf_peak = float(np.max(arr))
                                    image_b64 = base64.b64encode(arr.tobytes()).decode('utf-8')
                                    array_shape = list(arr.shape)
                                    array_dtype = str(arr.dtype)
                                    logger.info(f"Huygens PSF data extracted: shape={arr.shape}, peak={psf_peak:.6f}")
                    except Exception as e:
                        logger.warning(f"Huygens PSF: Could not extract data grid: {e}")

                    # Extract Strehl ratio from header text
                    try:
                        header_text = results.HeaderData.Lines if hasattr(results, 'HeaderData') else ""
                        if hasattr(header_text, '__iter__'):
                            for line in header_text:
                                line_str = str(line).lower()
                                if "strehl" in line_str:
                                    parts = line_str.split(":")
                                    if len(parts) > 1:
                                        val_str = parts[-1].strip()
                                        try:
                                            strehl_ratio = float(val_str)
                                        except ValueError:
                                            pass
                                    break
                    except Exception as e:
                        logger.debug(f"Huygens PSF: Could not extract Strehl ratio from header: {e}")
            finally:
                try:
                    analysis.Close()
                except Exception:
                    pass

            if image_b64 is None:
                return {"success": False, "error": "Huygens PSF analysis did not produce data"}

            result = {
                "success": True,
                "image": image_b64,
                "image_format": "numpy_array",
                "array_shape": array_shape,
                "array_dtype": array_dtype,
                "strehl_ratio": strehl_ratio,
                "psf_peak": psf_peak,
                "wavelength_um": float(wavelength_um),
                "field_x": field_x,
                "field_y": field_y,
            }
            _log_raw_output("/huygens-psf", result)
            return result

        except Exception as e:
            return {"success": False, "error": f"Huygens PSF analysis failed: {e}"}

    def _configure_analysis_settings(
        self,
        settings,
        field_index: Optional[int] = None,
        wavelength_index: Optional[int] = None,
        sampling: Optional[str] = None,
    ) -> None:
        """Configure common analysis settings (Field, Wavelength, SampleSize).

        Silently ignores missing attributes or setter failures, since different
        analysis types expose different subsets of these settings.
        """
        if field_index is not None and hasattr(settings, 'Field'):
            try:
                settings.Field.SetFieldNumber(field_index)
            except Exception:
                pass
        if wavelength_index is not None and hasattr(settings, 'Wavelength'):
            try:
                settings.Wavelength.SetWavelengthNumber(wavelength_index)
            except Exception:
                pass
        if sampling is not None:
            sample_value = SAMPLING_ENUM_MAP.get(sampling, 2)
            # SampleSize is the standard attribute; Huygens analyses use
            # PupilSampleSize/ImageSampleSize instead.
            for attr in ('SampleSize', 'PupilSampleSize', 'ImageSampleSize'):
                if hasattr(settings, attr):
                    try:
                        setattr(settings, attr, sample_value)
                    except Exception:
                        pass

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
            params = list(row.get("params", []))
            target = float(row.get("target", 0))
            weight = float(row.get("weight", 1))

            # Resolve semantic pseudo-codes to real Zemax operands
            if code == "BFD_SEMANTIC":
                # Back focal distance = TTHI from last glass surface to image surface
                num_surf = self.oss.LDE.NumberOfSurfaces
                if num_surf < 3:
                    row_errors.append({
                        "row_index": row_index,
                        "error": "BFD requires at least one optical surface between object and image",
                    })
                    evaluated_rows.append({
                        "row_index": row_index,
                        "operand_code": code,
                        "value": None,
                        "target": target,
                        "weight": weight,
                        "contribution": None,
                        "error": "Insufficient surfaces for BFD calculation",
                    })
                    continue
                image_surf = num_surf - 1
                last_before_image = image_surf - 1
                code = "TTHI"
                params = [last_before_image, image_surf, None, None, None, None]

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

                # Set Comment cell for BLNK section headers
                comment = row.get("comment")
                if comment and code == "BLNK":
                    try:
                        comment_cell = op.GetOperandCell(mfe_cols.Comment)
                        comment_cell.Value = str(comment)
                    except Exception:
                        pass

                # Set parameter cells — use cell DataType to pick the right setter
                param_columns = [
                    mfe_cols.Param1, mfe_cols.Param2,
                    mfe_cols.Param3, mfe_cols.Param4,
                    mfe_cols.Param5, mfe_cols.Param6,
                    mfe_cols.Param7, mfe_cols.Param8,
                ]
                for i, col in enumerate(param_columns):
                    if i < len(params) and params[i] is not None:
                        cell = op.GetOperandCell(col)
                        dt = str(cell.DataType).split('.')[-1] if hasattr(cell, 'DataType') else ''
                        if dt == 'Integer':
                            cell.IntegerValue = int(float(params[i]))
                        elif dt == 'String':
                            cell.Value = str(params[i])
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
                    "comment": row.get("comment"),
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
                    "comment": row.get("comment"),
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

        result = {
            "success": True,
            "total_merit": total_merit,
            "evaluated_rows": evaluated_rows,
            "row_errors": row_errors,
        }
        _log_raw_output("/evaluate-merit-function", result)
        return result

    def apply_optimization_wizard(
        self,
        criterion: str = "Spot",
        reference: str = "Centroid",
        overall_weight: float = 1.0,
        rings: int = 3,
        arms: int = 6,
        use_gaussian_quadrature: bool = False,
        use_glass_boundary_values: bool = False,
        glass_min: float = 1.0,
        glass_max: float = 50.0,
        use_air_boundary_values: bool = False,
        air_min: float = 0.5,
        air_max: float = 1000.0,
        air_edge_thickness: float = 0.0,
        type: str = "RMS",
        spatial_frequency: float = 30.0,
        xs_weight: float = 1.0,
        yt_weight: float = 1.0,
        use_maximum_distortion: bool = False,
        max_distortion_pct: float = 1.0,
        ignore_lateral_color: bool = False,
        obscuration: float = 0.0,
        glass_edge_thickness: float = 0.0,
        optimization_goal: str = "nominal",
        manufacturing_yield_weight: float = 1.0,
        start_at: int = 1,
        use_all_configurations: bool = True,
        configuration_number: int = 1,
        use_all_fields: bool = True,
        field_number: int = 1,
        assume_axial_symmetry: bool = True,
        add_favorite_operands: bool = False,
        delete_vignetted: bool = True,
    ) -> dict[str, Any]:
        """
        Apply the SEQ Optimization Wizard to auto-generate merit function operands.

        Uses OpticStudio's SEQOptimizationWizard2 to populate the MFE based on
        image quality criteria (Spot, Wavefront, or Contrast).

        Returns:
            Dict with success, total_merit, generated_rows, num_rows_generated
        """
        def _wizard_error(error: str, total_merit=None) -> dict[str, Any]:
            return {
                "success": False, "error": error,
                "total_merit": total_merit, "generated_rows": [],
                "num_rows_generated": 0,
            }

        zp = self._zp
        mfe = self.oss.MFE

        # Cross-field validations (single-field constraints enforced by Pydantic)
        if use_glass_boundary_values and glass_min >= glass_max:
            return _wizard_error(f"glass_min ({glass_min}) must be < glass_max ({glass_max})")
        if use_air_boundary_values and air_min >= air_max:
            return _wizard_error(f"air_min ({air_min}) must be < air_max ({air_max})")

        # Check wizard availability (requires OpticStudio 18.5+)
        wizard = getattr(mfe, 'SEQOptimizationWizard2', None)
        if wizard is None:
            return _wizard_error("SEQOptimizationWizard2 not available (requires OpticStudio 18.5+)")

        def _set_wizard_prop(prop_name: str, value: Any) -> None:
            """Set a wizard property, logging a warning on failure."""
            try:
                setattr(wizard, prop_name, value)
            except Exception as e:
                logger.warning(f"Failed to set {prop_name}: {e}")

        try:
            # Resolve wizard enums namespace (ZOSAPI.Wizards)
            # ZosPy maps this to zp.constants.Wizards
            wizard_enums = getattr(zp.constants, 'Wizards', None)
            if wizard_enums is None:
                return _wizard_error(
                    "Wizard enums not found at zp.constants.Wizards. "
                    "Check ZosPy version compatibility."
                )

            # Set criterion (wizard.Criterion = ZOSAPI.Wizards.CriterionTypes.Spot)
            if not hasattr(wizard_enums.CriterionTypes, criterion):
                return _wizard_error(f"Unknown criterion type: {criterion}")
            wizard.Criterion = getattr(wizard_enums.CriterionTypes, criterion)

            # Set reference (wizard.Reference = ZOSAPI.Wizards.ReferenceTypes.Centroid)
            if not hasattr(wizard_enums.ReferenceTypes, reference):
                return _wizard_error(f"Unknown reference type: {reference}")
            wizard.Reference = getattr(wizard_enums.ReferenceTypes, reference)

            wizard.OverallWeight = float(overall_weight)
            wizard.Rings = int(rings)

            # Map arms count to ZOSAPI enum
            arms_enum_name = f"Arms_{arms}"
            if hasattr(wizard_enums.PupilArmsCount, arms_enum_name):
                wizard.Arms = getattr(wizard_enums.PupilArmsCount, arms_enum_name)
            else:
                logger.warning(f"Unknown arms count {arms}, using Arms_6")
                wizard.Arms = getattr(wizard_enums.PupilArmsCount, "Arms_6")

            wizard.IsGaussianQuadrature = bool(use_gaussian_quadrature)

            # Glass boundary values
            wizard.IsGlassBoundaryValues = bool(use_glass_boundary_values)
            if use_glass_boundary_values:
                wizard.GlassMin = float(glass_min)
                wizard.GlassMax = float(glass_max)
                _set_wizard_prop("GlassEdgeThickness", float(glass_edge_thickness))

            # Air boundary values
            wizard.IsAirBoundaryValues = bool(use_air_boundary_values)
            if use_air_boundary_values:
                wizard.AirMin = float(air_min)
                wizard.AirMax = float(air_max)
                wizard.AirEdgeThickness = float(air_edge_thickness)

            # Type (RMS/PTV) — wizard.Type = ZOSAPI.Wizards.OptimizationTypes.RMS
            if hasattr(wizard_enums, "OptimizationTypes"):
                opt_type = getattr(wizard_enums.OptimizationTypes, type, None)
                if opt_type is not None:
                    wizard.Type = opt_type
                else:
                    logger.warning(f"OptimizationTypes.{type} not found, skipping Type assignment")
            else:
                logger.warning("OptimizationTypes enum not found in Wizards namespace")

            # Optimization Function params
            _set_wizard_prop("SpatialFrequency", float(spatial_frequency))
            _set_wizard_prop("XSWeight", float(xs_weight))
            _set_wizard_prop("YTWeight", float(yt_weight))
            _set_wizard_prop("IsMaxDistortion", bool(use_maximum_distortion))
            if use_maximum_distortion:
                _set_wizard_prop("MaxDistortionPct", float(max_distortion_pct))
            _set_wizard_prop("IsIgnoreLateralColor", bool(ignore_lateral_color))

            # Pupil Integration
            _set_wizard_prop("Obscuration", float(obscuration))

            # Optimization Goal
            _set_wizard_prop("OptimizeForBestNominalPerformance", optimization_goal == "nominal")
            _set_wizard_prop("OptimizeForManufacturingYield", optimization_goal == "manufacturing_yield")
            if optimization_goal == "manufacturing_yield":
                _set_wizard_prop("ManufacturingYieldWeight", float(manufacturing_yield_weight))

            # Bottom bar params
            _set_wizard_prop("StartAt", int(start_at))
            _set_wizard_prop("UseAllConfigurations", bool(use_all_configurations))
            if not use_all_configurations:
                _set_wizard_prop("ConfigurationNumber", int(configuration_number))
            _set_wizard_prop("UseAllFields", bool(use_all_fields))
            if not use_all_fields:
                _set_wizard_prop("FieldNumber", int(field_number))
            _set_wizard_prop("AssumeAxialSymmetry", bool(assume_axial_symmetry))
            _set_wizard_prop("AddFavoriteOperands", bool(add_favorite_operands))
            _set_wizard_prop("DeleteVignetted", bool(delete_vignetted))

            logger.info(
                f"Applying optimization wizard: criterion={criterion}, type={type}, "
                f"reference={reference}, rings={rings}, arms={arms}, "
                f"gaussian_quadrature={use_gaussian_quadrature}, goal={optimization_goal}"
            )
            wizard.Apply()

        except Exception as e:
            logger.error(f"Optimization wizard Apply() failed: {e}")
            return _wizard_error(f"Wizard Apply() failed: {e}")

        # Calculate merit function to get per-row values
        try:
            total_merit = _extract_value(mfe.CalculateMeritFunction())
        except Exception as e:
            logger.error(f"CalculateMeritFunction after wizard failed: {e}")
            return _wizard_error(f"Merit function calculation failed after wizard: {e}")

        # Read all generated rows from MFE
        generated_rows = []
        try:
            mfe_cols = zp.constants.Editors.MFE.MeritColumn
            param_columns = [
                mfe_cols.Param1, mfe_cols.Param2,
                mfe_cols.Param3, mfe_cols.Param4,
                mfe_cols.Param5, mfe_cols.Param6,
                mfe_cols.Param7, mfe_cols.Param8,
            ]

            num_operands = mfe.NumberOfOperands
            logger.info(f"Wizard generated {num_operands} operand rows")

            for i in range(1, num_operands + 1):
                try:
                    op = mfe.GetOperandAt(i)

                    # Get operand code from enum
                    try:
                        op_code = str(op.Type).split('.')[-1]
                    except Exception:
                        op_code = f"UNK_{i}"

                    # Read 6 parameter cells
                    # Note: 0 is a valid value (e.g., surface index 0 = image surface),
                    # so we always include it rather than converting to None.
                    params = []
                    for j, col in enumerate(param_columns):
                        try:
                            cell = op.GetOperandCell(col)
                            dt = str(cell.DataType).split('.')[-1] if hasattr(cell, 'DataType') else ''
                            if dt == 'String':
                                val = cell.Value
                                params.append(str(val) if val else None)
                            else:
                                raw = float(cell.IntegerValue if dt == 'Integer' else cell.DoubleValue)
                                params.append(None if (math.isinf(raw) or math.isnan(raw)) else raw)
                        except Exception:
                            params.append(None)

                    generated_rows.append({
                        "row_index": i - 1,
                        "operand_code": op_code,
                        "params": params,
                        "target": _extract_value(op.Target, 0.0),
                        "weight": _extract_value(op.Weight, 0.0),
                        "value": _extract_value(op.Value, None),
                        "contribution": _extract_value(op.Contribution, None),
                        "comment": _read_comment_cell(op, mfe_cols.Comment),
                    })
                except Exception as e:
                    logger.warning(f"Error reading wizard MFE row {i}: {e}")
                    generated_rows.append({
                        "row_index": i - 1,
                        "operand_code": f"ERR_{i}",
                        "params": [None] * 8,
                        "target": 0.0,
                        "weight": 0.0,
                        "value": None,
                        "contribution": None,
                        "comment": None,
                    })

        except Exception as e:
            logger.error(f"Error reading wizard-generated MFE rows: {e}")
            return _wizard_error(f"Failed to read wizard-generated rows: {e}", total_merit=total_merit)

        result = {
            "success": True,
            "total_merit": total_merit,
            "generated_rows": generated_rows,
            "num_rows_generated": len(generated_rows),
        }
        _log_raw_output("/apply-optimization-wizard", result)
        return result


    def get_operand_catalog(self) -> dict[str, Any]:
        """
        Discover all supported MeritOperandType values and their parameter metadata.

        Iterates every operand type in the MeritOperandType enum, sets it on a
        temporary MFE row, and reads back cell metadata (Header, DataType,
        IsActive, IsReadOnly) for Comment + Param1-Param8 columns.

        Returns:
            Dict with success, operands list, total_count
        """
        mfe = self.oss.MFE
        mfe_constants = self._zp.constants.Editors.MFE
        MeritOperandType = mfe_constants.MeritOperandType
        mfe_cols = mfe_constants.MeritColumn

        # Column definitions: (name, enum_value)
        columns = [
            ("Comment", mfe_cols.Comment),
            *((f"Param{i}", getattr(mfe_cols, f"Param{i}")) for i in range(1, 9)),
        ]

        # Ensure at least 1 row exists in the MFE
        try:
            if mfe.NumberOfOperands < 1:
                mfe.AddOperand()
            op = mfe.GetOperandAt(1)
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to initialize MFE for operand catalog: {e}",
                "operands": [],
                "total_count": 0,
            }

        # Enumerate all operand type names from the namedtuple.
        # Use _fields (not dir()) to avoid namedtuple methods like count/index.
        if hasattr(MeritOperandType, '_fields'):
            type_names = list(MeritOperandType._fields)
        else:
            type_names = [name for name in dir(MeritOperandType) if not name.startswith('_')]
        logger.info(f"Enumerating {len(type_names)} operand types for catalog")

        operands = []
        skipped = 0
        start_time = time.time()

        for code in type_names:
            try:
                enum_val = getattr(MeritOperandType, code)
                op.ChangeType(enum_val)

                # Read the human-readable type name from OpticStudio
                type_name = ""
                try:
                    type_name = str(op.TypeName) if hasattr(op, 'TypeName') else ""
                except Exception:
                    pass

                parameters = [
                    self._read_operand_cell(op, col_name, col_enum, code)
                    for col_name, col_enum in columns
                ]

                operands.append({
                    "code": code,
                    "type_name": type_name,
                    "parameters": parameters,
                })
            except Exception as e:
                logger.warning(f"Skipping operand {code}: {e}")
                skipped += 1
                # Re-acquire the row reference in case ChangeType corrupted it
                try:
                    op = mfe.GetOperandAt(1)
                except Exception:
                    pass

        elapsed = time.time() - start_time
        logger.info(
            f"Operand catalog complete: {len(operands)} operands in {elapsed:.1f}s "
            f"(skipped {skipped})"
        )

        return {
            "success": True,
            "operands": operands,
            "total_count": len(operands),
        }

    @staticmethod
    def _read_cell_default(cell: Any, data_type: str) -> float | int | str | None:
        """Extract the default value from a COM operand cell.

        Returns None if the value cannot be read or is inf/NaN.
        """
        try:
            if data_type == "Integer" and hasattr(cell, 'IntegerValue'):
                raw = cell.IntegerValue
                if raw is not None and not (isinstance(raw, float) and not math.isfinite(raw)):
                    return int(raw)
            elif data_type == "Double" and hasattr(cell, 'DoubleValue'):
                raw = cell.DoubleValue
                if raw is not None and math.isfinite(raw):
                    return float(raw)
            elif data_type == "String" and hasattr(cell, 'Value'):
                raw = cell.Value
                if raw is not None:
                    return str(raw)
        except Exception:
            pass
        return None

    def _read_operand_cell(
        self, op: Any, col_name: str, col_enum: Any, operand_code: str,
    ) -> dict[str, Any]:
        """Read metadata from a single MFE operand cell.

        Returns a dict with column, header, data_type, is_active, is_read_only,
        and default_value. Falls back to safe defaults if the cell cannot be read.
        """
        try:
            cell = op.GetOperandCell(col_enum)
            data_type = str(cell.DataType).split('.')[-1] if hasattr(cell, 'DataType') else "Unknown"

            default_value = self._read_cell_default(cell, data_type)

            return {
                "column": col_name,
                "header": str(cell.Header) if hasattr(cell, 'Header') else col_name,
                "data_type": data_type,
                "is_active": bool(cell.IsActive) if hasattr(cell, 'IsActive') else False,
                "is_read_only": bool(cell.IsReadOnly) if hasattr(cell, 'IsReadOnly') else False,
                "default_value": default_value,
            }
        except Exception as e:
            logger.debug(f"Failed to read {col_name} for {operand_code}: {e}")
            return {
                "column": col_name,
                "header": col_name,
                "data_type": "Unknown",
                "is_active": False,
                "is_read_only": True,
                "default_value": None,
            }


    # ── Optimization enum helpers ──────────────────────────────────────

    @staticmethod
    def _resolve_enum(enum_obj, attr_name: str, fallback_name: str, label: str):
        """Resolve an enum member by name, falling back to a default with a warning."""
        resolved = getattr(enum_obj, attr_name, None)
        if resolved is not None:
            return resolved
        logger.warning(
            f"{label} '{attr_name}' not found in enum, falling back to {fallback_name}"
        )
        return getattr(enum_obj, fallback_name, None)

    @staticmethod
    def _resolve_algorithm(zp_module, algorithm: str):
        """Map algorithm string to OptimizationAlgorithm enum value."""
        alg_enum = zp_module.constants.Tools.Optimization.OptimizationAlgorithm
        aliases = {"DLS": "DampedLeastSquares"}
        attr_name = aliases.get(algorithm, algorithm)
        fallback = "DampedLeastSquares"
        return ZosPyHandler._resolve_enum(alg_enum, attr_name, fallback, "Algorithm")

    @staticmethod
    def _resolve_cycles(zp_module, cycles: int | None):
        """Map cycle count to OptimizationCycles enum value.

        The API only supports specific fixed counts (1, 5, 10, 50).
        Unrecognized values fall back to Automatic.
        """
        cycles_enum = zp_module.constants.Tools.Optimization.OptimizationCycles
        fallback = "Automatic"
        if cycles is None:
            return getattr(cycles_enum, fallback, None)
        mapping = {
            1: "Fixed_1_Cycle",
            5: "Fixed_5_Cycles",
            10: "Fixed_10_Cycles",
            50: "Fixed_50_Cycles",
        }
        attr_name = mapping.get(cycles, fallback)
        return ZosPyHandler._resolve_enum(cycles_enum, attr_name, fallback, "Cycles")

    @staticmethod
    def _resolve_save_count(zp_module, num_to_save: int | None):
        """Map save count to OptimizationSaveCount enum value.

        Picks the closest supported value (5, 10, 20, 50).
        """
        save_enum = zp_module.constants.Tools.Optimization.OptimizationSaveCount
        fallback = "Save_10"
        if num_to_save is None:
            return getattr(save_enum, fallback, None)
        mapping = {
            5: "Save_5",
            10: "Save_10",
            20: "Save_20",
            50: "Save_50",
        }
        closest = min(mapping.keys(), key=lambda k: abs(k - num_to_save))
        attr_name = mapping[closest]
        return ZosPyHandler._resolve_enum(save_enum, attr_name, fallback, "SaveCount")

    @staticmethod
    def _read_systems_evaluated(opt_tool) -> int | None:
        """Read systems evaluated count from an optimization tool, or None on failure."""
        try:
            return int(opt_tool.Systems) if hasattr(opt_tool, 'Systems') else None
        except Exception as e:
            logger.debug(f"Could not read systems_evaluated: {e}")
            return None

    def _read_best_solutions(self, opt_tool, mfe) -> list[float]:
        """Read the best merit value from a global optimizer, with MFE fallback."""
        solutions: list[float] = []
        try:
            mf_val = _extract_value(opt_tool.CurrentMeritFunction)
            if mf_val is not None and mf_val > 0:
                solutions.append(mf_val)
        except Exception as e:
            logger.warning(f"Failed to read CurrentMeritFunction from optimizer: {e}")
        if not solutions:
            try:
                mf_val = _extract_value(mfe.CalculateMeritFunction())
                if mf_val is not None:
                    solutions.append(mf_val)
            except Exception as e:
                logger.warning(f"Failed to read global best solutions: {e}")
        return solutions

    # ── Main optimization entry point ───────────────────────────────

    def run_optimization(
        self,
        method: str = "local",
        algorithm: str = "DLS",
        cycles: int | None = 5,
        timeout_seconds: float | None = 60,
        num_to_save: int | None = 10,
        operand_rows: list[dict] | None = None,
    ) -> dict[str, Any]:
        """
        Run OpticStudio optimization using Local, Global, or Hammer method.

        Args:
            method: "local" | "global" | "hammer"
            algorithm: "DLS" | "OrthogonalDescent" | "DLSX" | "PSD"
            cycles: Cycle count for local optimization (1, 5, 10, 50, or None=auto)
            timeout_seconds: Time limit for global/hammer (they run indefinitely)
            num_to_save: Number of best solutions to retain (global only)
            operand_rows: Explicit MFE operand rows

        Returns:
            Dict with merit_before, merit_after, cycles_completed,
            operand_results, variable_states, and (for global) best_solutions
        """
        zp = self._zp
        mfe = self.oss.MFE

        # Normalize method
        method = (method or "local").lower()

        # Step 1: Populate MFE
        if not operand_rows:
            return {"success": False, "error": "Must provide operand_rows"}

        mfe_result = self.evaluate_merit_function(operand_rows)
        if not mfe_result.get("success"):
            return {
                "success": False,
                "error": f"MFE setup failed: {mfe_result.get('error')}",
            }

        # Step 2: Read initial merit
        try:
            merit_before = _extract_value(mfe.CalculateMeritFunction())
        except Exception as e:
            return {"success": False, "error": f"Initial merit calculation failed: {e}"}

        # Step 3: Run optimization (method-specific)
        best_solutions: list[float] | None = None
        systems_evaluated: int | None = None

        try:
            tools = self.oss.Tools
            resolved_alg = self._resolve_algorithm(zp, algorithm or "DLS")

            if method in ("global", "hammer"):
                # Global and Hammer both use timeout-based execution
                if method == "global":
                    opt_tool = tools.OpenGlobalOptimization()
                    save_count = self._resolve_save_count(zp, num_to_save)
                    if save_count is not None:
                        opt_tool.NumberToSave = save_count
                else:
                    opt_tool = tools.OpenHammerOptimization()

                opt_tool.Algorithm = resolved_alg
                opt_tool.NumberOfCores = getattr(opt_tool, 'MaxCores', 8)
                timeout = max(10, min(timeout_seconds or 60, 600))

                try:
                    opt_tool.RunAndWaitWithTimeout(timeout)
                    opt_tool.Cancel()
                    opt_tool.WaitForCompletion()
                finally:
                    if method == "global":
                        best_solutions = self._read_best_solutions(opt_tool, mfe)
                    systems_evaluated = self._read_systems_evaluated(opt_tool)
                    opt_tool.Close()

            else:  # local (default)
                opt_tool = tools.OpenLocalOptimization()
                opt_tool.Algorithm = resolved_alg
                opt_tool.NumberOfCores = getattr(opt_tool, 'MaxCores', 8)

                resolved_cycles = self._resolve_cycles(zp, cycles)
                if resolved_cycles is not None and hasattr(opt_tool, 'Cycles'):
                    opt_tool.Cycles = resolved_cycles

                try:
                    opt_tool.RunAndWaitForCompletion()
                finally:
                    opt_tool.Close()

        except Exception as e:
            logger.error(f"Optimization run failed: {e}")
            return {"success": False, "error": f"Optimization failed: {e}"}

        # Step 4: Read final merit
        try:
            merit_after = _extract_value(mfe.CalculateMeritFunction())
        except Exception as e:
            merit_after = merit_before
            logger.warning(f"Post-optimization merit calculation failed: {e}")

        # Step 5: Read operand results from MFE
        operand_results = []
        try:
            mfe_cols = zp.constants.Editors.MFE.MeritColumn
            param_columns = [
                mfe_cols.Param1, mfe_cols.Param2,
                mfe_cols.Param3, mfe_cols.Param4,
                mfe_cols.Param5, mfe_cols.Param6,
                mfe_cols.Param7, mfe_cols.Param8,
            ]
            num_operands = mfe.NumberOfOperands
            for i in range(1, num_operands + 1):
                try:
                    op = mfe.GetOperandAt(i)
                    try:
                        op_code = str(op.Type).split('.')[-1]
                    except Exception:
                        op_code = f"UNK_{i}"

                    params = []
                    for j, col in enumerate(param_columns):
                        try:
                            cell = op.GetOperandCell(col)
                            dt = str(cell.DataType).split('.')[-1] if hasattr(cell, 'DataType') else ''
                            if dt == 'String':
                                val = cell.Value
                                params.append(str(val) if val else None)
                            else:
                                raw = float(cell.IntegerValue if dt == 'Integer' else cell.DoubleValue)
                                params.append(None if (math.isinf(raw) or math.isnan(raw)) else raw)
                        except Exception:
                            params.append(None)

                    operand_results.append({
                        "row_index": i - 1,
                        "operand_code": op_code,
                        "params": params,
                        "target": _extract_value(op.Target, 0.0),
                        "weight": _extract_value(op.Weight, 0.0),
                        "value": _extract_value(op.Value, None),
                        "contribution": _extract_value(op.Contribution, None),
                        "comment": _read_comment_cell(op, mfe_cols.Comment),
                    })
                except Exception as e:
                    logger.warning(f"Error reading post-opt MFE row {i}: {e}")
        except Exception as e:
            logger.warning(f"Error reading post-optimization MFE: {e}")

        # Step 6: Extract variable states from LDE
        variable_states = self._extract_variable_states()

        result: dict[str, Any] = {
            "success": True,
            "method": method,
            "algorithm": algorithm,
            "merit_before": merit_before,
            "merit_after": merit_after,
            "cycles_completed": cycles if method == "local" else None,
            "operand_results": operand_results,
            "variable_states": variable_states,
        }
        if best_solutions is not None:
            result["best_solutions"] = best_solutions
        if systems_evaluated is not None:
            result["systems_evaluated"] = systems_evaluated
        _log_raw_output("/run-optimization", result)
        return result

    # Maps parameter name -> (cell attribute, value attribute)
    _VARIABLE_PARAMS = {
        "radius": ("RadiusCell", "Radius"),
        "thickness": ("ThicknessCell", "Thickness"),
        "conic": ("ConicCell", "Conic"),
    }

    def _extract_variable_states(self) -> list[dict[str, Any]]:
        """
        Extract current values of all variable parameters from the LDE.

        Iterates all surfaces and checks if radius, thickness, conic, or
        aspheric parameters (Par1-Par12) are marked as variable.

        Returns:
            List of {surface_index, parameter, value, is_variable}
        """
        lde = self.oss.LDE
        variable_states: list[dict[str, Any]] = []

        try:
            num_surfaces = lde.NumberOfSurfaces
            surf_col = self._zp.constants.Editors.LDE.SurfaceColumn
        except Exception as e:
            logger.warning(f"Error accessing LDE for variable extraction: {e}")
            return variable_states

        # Start at surface 1 (skip object surface 0).
        # Return surf_idx as-is (Zemax LDE index) -- the analysis
        # service's surface_patcher uses the same indexing.
        for surf_idx in range(1, num_surfaces):
            try:
                surf = lde.GetSurfaceAt(surf_idx)
            except Exception as e:
                logger.debug(f"Error reading surface {surf_idx} variables: {e}")
                continue

            # Check radius, thickness, conic
            for param_name, (cell_attr, value_attr) in self._VARIABLE_PARAMS.items():
                try:
                    cell = getattr(surf, cell_attr)
                    if not hasattr(cell, 'GetSolveData'):
                        continue
                    solve = cell.GetSolveData()
                    solve_type = str(solve.Type).split('.')[-1] if solve else ""
                    if solve_type == "Variable":
                        # Radius and thickness can be Infinity in OpticStudio
                        # (flat surfaces, afocal systems / infinite conjugates)
                        val = _extract_value(
                            getattr(surf, value_attr), 0.0,
                            allow_inf=(param_name in ("radius", "thickness")),
                        )
                        variable_states.append({
                            "surface_index": surf_idx,
                            "parameter": param_name,
                            "value": val,
                            "is_variable": True,
                        })
                except Exception as e:
                    logger.warning(
                        f"Failed to read variable state for surface {surf_idx}, "
                        f"param '{param_name}': {type(e).__name__}: {e}"
                    )

            # Check aspheric parameters (PARM 1-12 via SurfaceColumn.Par1-Par12).
            # Not all surface types support all 12 params; stop at the first
            # missing column attribute.
            for par_idx in range(1, 13):
                par_attr = f'Par{par_idx}'
                if not hasattr(surf_col, par_attr):
                    break
                try:
                    cell = surf.GetSurfaceCell(getattr(surf_col, par_attr))
                    if cell is None or not hasattr(cell, 'GetSolveData'):
                        continue
                    solve = cell.GetSolveData()
                    solve_type = str(solve.Type).split('.')[-1] if solve else ""
                    if solve_type == "Variable":
                        variable_states.append({
                            "surface_index": surf_idx,
                            "parameter": f"param_{par_idx}",
                            "value": _extract_value(cell.DoubleValue, 0.0),
                            "is_variable": True,
                        })
                except (AttributeError, TypeError):
                    break  # Surface type does not support this parameter
                except Exception as e:
                    logger.debug(f"Error checking Par{par_idx} for surface {surf_idx}: {e}")

        return variable_states

    def _run_zernike_vs_field_fallback(
        self,
        maximum_term: int,
        wavelength_index: int,
        field_density: int,
    ) -> Optional["pd.DataFrame"]:
        """
        Run ZernikeCoefficientsVsField via the raw new_analysis API.

        Used as a fallback when the ZOSPy wrapper fails. Extracts data from
        either data series (graph-type) or data grids, and returns a DataFrame
        with field positions as the index and Zernike terms as columns.

        Returns None if no data could be extracted.
        """
        import pandas as pd

        idm = self._zp.constants.Analysis.AnalysisIDM
        analysis = self._zp.analyses.new_analysis(
            self.oss, idm.ZernikeCoefficientsVsField, settings_first=True,
        )
        try:
            settings = analysis.Settings
            if hasattr(settings, 'MaximumNumberOfTerms'):
                settings.MaximumNumberOfTerms = maximum_term
            elif hasattr(settings, 'MaximumTerm'):
                settings.MaximumTerm = maximum_term
            if hasattr(settings, 'Wavelength'):
                settings.Wavelength.SetWavelengthNumber(wavelength_index)
            if hasattr(settings, 'FieldDensity'):
                settings.FieldDensity = field_density

            zernike_start = time.perf_counter()
            try:
                analysis.ApplyAndWaitForCompletion()
            finally:
                elapsed_ms = (time.perf_counter() - zernike_start) * 1000
                log_timing(logger, "ZernikeCoefficientsVsField.run (fallback)", elapsed_ms)

            results = analysis.Results
            rows = self._extract_zernike_fallback_rows(results)
            if not rows:
                return None
            return pd.DataFrame(rows).set_index("field")
        finally:
            try:
                analysis.Close()
            except Exception:
                pass

    def _extract_zernike_fallback_rows(self, results: Any) -> list[dict]:
        """
        Extract row dicts from ZOS-API analysis results for ZernikeVsField.

        Tries data series first (graph-type result), then data grids.
        Each row dict has a "field" key plus Zernike term keys.
        """
        if results is None:
            return []

        # Try data series extraction (graph-type result)
        rows: list[dict] = []
        if hasattr(results, 'NumberOfDataSeries'):
            num_series = results.NumberOfDataSeries
            logger.info(f"ZernikeVsField fallback: {num_series} data series")
            for si in range(num_series):
                series = results.GetDataSeries(si)
                if series is None:
                    continue
                desc = str(series.Description) if hasattr(series, 'Description') else f"Z{si+1}"
                n_pts = getattr(series, 'NumberOfPoints', 0)
                for pi in range(n_pts):
                    pt = series.GetDataPoint(pi)
                    if pt is None:
                        continue
                    x = _extract_value(pt.X if hasattr(pt, 'X') else pt[0])
                    y = _extract_value(pt.Y if hasattr(pt, 'Y') else pt[1])
                    if pi >= len(rows):
                        rows.append({"field": x})
                    rows[pi][desc] = y

        # If no data series, try data grids
        if not rows and hasattr(results, 'NumberOfDataGrids'):
            num_grids = results.NumberOfDataGrids
            logger.info(f"ZernikeVsField fallback: {num_grids} data grids (no series)")
            for gi in range(num_grids):
                grid = results.GetDataGrid(gi)
                if grid is None:
                    continue
                n_rows = getattr(grid, 'NumberOfRows', 0)
                n_cols = getattr(grid, 'NumberOfCols', 0)
                for ri in range(n_rows):
                    row_data = {}
                    for ci in range(n_cols):
                        val = _extract_value(grid.Z(ri, ci))
                        if ci == 0:
                            row_data["field"] = val
                        else:
                            row_data[f"Z{ci}"] = val
                    if row_data:
                        rows.append(row_data)

        return rows

    def get_zernike_vs_field(
        self,
        maximum_term: int = 37,
        wavelength_index: int = 1,
        sampling: str = "64x64",
        field_density: int = 20,
    ) -> dict[str, Any]:
        """
        Get Zernike Coefficients vs Field using ZOSPy's ZernikeCoefficientsVsField analysis.

        Returns how each Zernike coefficient varies across field positions.

        Args:
            maximum_term: Maximum Zernike term number (default 37)
            wavelength_index: Wavelength index (1-indexed)
            sampling: Pupil sampling grid (e.g., '64x64')
            field_density: Number of field sample points (default 20)

        Returns:
            On success: {
                "success": True,
                "field_positions": [...],
                "coefficients": { "4": [values_per_field], "5": [...], ... },
                "wavelength_um": float,
                "field_unit": str,
            }
            On error: {"success": False, "error": "..."}
        """
        try:
            # Validate wavelength index
            wavelengths = self.oss.SystemData.Wavelengths
            if wavelength_index > wavelengths.NumberOfWavelengths:
                return {"success": False, "error": f"Wavelength index {wavelength_index} out of range (max: {wavelengths.NumberOfWavelengths})"}
            wavelength_um = _extract_value(wavelengths.GetWavelength(wavelength_index).Wavelength, 0.5876)

            # Determine field unit from system
            fields = self.oss.SystemData.Fields
            try:
                ft = fields.GetFieldType()
                field_type_str = getattr(ft, 'name', str(ft).split(".")[-1])
            except Exception:
                field_type_str = ""
            field_unit = "deg" if "angle" in field_type_str.lower() else "mm"

            # Build coefficients range string: "1-N"
            coefficients_str = f"1-{maximum_term}"

            # Run ZernikeCoefficientsVsField using the new-style ZOSPy API
            # ZosPy's parser can fail on certain input combinations in some versions,
            # so we fall back to the raw new_analysis API if needed.
            zernike_result = None
            try:
                zernike_vs_field = self._zp.analyses.wavefront.ZernikeCoefficientsVsField(
                    coefficients=coefficients_str,
                    wavelength=wavelength_index,
                    sampling=sampling,
                    field_density=field_density,
                )

                zernike_start = time.perf_counter()
                try:
                    zernike_result = zernike_vs_field.run(self.oss)
                finally:
                    zernike_elapsed_ms = (time.perf_counter() - zernike_start) * 1000
                    log_timing(logger, "ZernikeCoefficientsVsField.run", zernike_elapsed_ms)
            except Exception as wrapper_err:
                logger.warning(f"ZernikeCoefficientsVsField wrapper failed ({wrapper_err}), falling back to new_analysis API")
                zernike_result = self._run_zernike_vs_field_fallback(
                    maximum_term, wavelength_index, field_density,
                )
                if zernike_result is None:
                    return {"success": False, "error": "ZernikeCoefficientsVsField fallback returned no data"}

            # Extract DataFrame from the result (wrapper path yields AnalysisResult,
            # fallback path yields a DataFrame directly)
            import pandas as pd
            if isinstance(zernike_result, pd.DataFrame):
                df = zernike_result
            else:
                df = _extract_dataframe(zernike_result, "ZernikeCoefficientsVsField")
                if df is None:
                    return {"success": False, "error": "ZernikeCoefficientsVsField returned no extractable DataFrame"}

            logger.info(f"ZernikeVsField: DataFrame columns={list(df.columns)}, shape={df.shape}")

            if len(df) == 0:
                return {"success": False, "error": f"No Zernike vs field data extracted (cols={list(df.columns)[:10]})"}

            # Extract field positions from the DataFrame index
            try:
                field_positions = [float(v) for v in df.index.tolist()]
            except (ValueError, TypeError):
                field_positions = list(range(len(df)))

            # Extract coefficient columns - handle various naming formats:
            # Pure numbers: "1", "4", "37"
            # Z-prefixed: "Z1", "Z4", "Z 4", "Z04"
            coefficients_dict = {}
            for col in df.columns:
                term_num = _parse_zernike_term_number(col)
                if term_num is not None:
                    raw = df[col].tolist()
                    values = []
                    for v in raw:
                        try:
                            f = float(v)
                            values.append(0.0 if math.isnan(f) else f)
                        except (ValueError, TypeError):
                            values.append(0.0)
                    coefficients_dict[str(term_num)] = values
                else:
                    logger.debug(f"ZernikeVsField: Skipping non-Zernike column '{col}'")

            if not field_positions or not coefficients_dict:
                return {"success": False, "error": f"No Zernike vs field data extracted (cols={list(df.columns)[:10]})"}

            result = {
                "success": True,
                "field_positions": field_positions,
                "coefficients": coefficients_dict,
                "wavelength_um": float(wavelength_um),
                "field_unit": field_unit,
            }
            _log_raw_output("/zernike-vs-field", result)
            return result

        except Exception as e:
            logger.error(f"get_zernike_vs_field failed: {e}", exc_info=True)
            return {"success": False, "error": f"ZernikeCoefficientsVsField analysis failed: {e}"}

    def get_zernike_standard_coefficients(
        self,
        field_index: int = 1,
        wavelength_index: int = 1,
        sampling: str = "64x64",
        maximum_term: int = 37,
        surface: str = "Image",
    ) -> dict[str, Any]:
        """
        Get Zernike Standard Coefficients decomposition of the wavefront.

        Returns individual Zernike polynomial coefficients (Z1-Z37+), P-V wavefront
        error, RMS wavefront error, and Strehl ratio.

        Note: System must be pre-loaded via load_zmx_file().

        Args:
            field_index: Field index (1-indexed)
            wavelength_index: Wavelength index (1-indexed)
            sampling: Pupil sampling grid (e.g., '64x64', '128x128')
            maximum_term: Maximum Zernike term number (default 37)
            surface: Surface to analyze (default "Image")

        Returns:
            On success: dict with coefficients, P-V, RMS, Strehl, etc.
            On error: {"success": False, "error": "..."}
        """
        try:
            # Validate field index
            fields = self.oss.SystemData.Fields
            if field_index > fields.NumberOfFields:
                return {"success": False, "error": f"Field index {field_index} out of range (max: {fields.NumberOfFields})"}

            field = fields.GetField(field_index)
            field_x = _extract_value(field.X)
            field_y = _extract_value(field.Y)

            # Validate wavelength index
            wavelengths = self.oss.SystemData.Wavelengths
            if wavelength_index > wavelengths.NumberOfWavelengths:
                return {"success": False, "error": f"Wavelength index {wavelength_index} out of range (max: {wavelengths.NumberOfWavelengths})"}

            wavelength_um = _extract_value(wavelengths.GetWavelength(wavelength_index).Wavelength, 0.5876)

            # Run ZernikeStandardCoefficients analysis
            zernike_analysis = self._zp.analyses.wavefront.ZernikeStandardCoefficients(
                sampling=sampling,
                maximum_term=maximum_term,
                wavelength=wavelength_index,
                field=field_index,
                reference_opd_to_vertex=False,
                surface=surface,
            )

            zernike_start = time.perf_counter()
            try:
                zernike_result = zernike_analysis.run(self.oss)
            finally:
                zernike_elapsed_ms = (time.perf_counter() - zernike_start) * 1000
                log_timing(logger, "ZernikeStandardCoefficients.run (full)", zernike_elapsed_ms)

            if not hasattr(zernike_result, 'data') or zernike_result.data is None:
                return {"success": False, "error": "ZernikeStandardCoefficients returned no data"}

            zdata = zernike_result.data

            # Extract P-V wavefront error
            pv_to_chief = _extract_value(getattr(zdata, 'peak_to_valley_to_chief', None))
            pv_to_centroid = _extract_value(getattr(zdata, 'peak_to_valley_to_centroid', None))

            # Extract RMS and Strehl from integration data
            rms_to_chief = None
            rms_to_centroid = None
            strehl_ratio = None

            if hasattr(zdata, 'from_integration_of_the_rays'):
                integration = zdata.from_integration_of_the_rays
                rms_to_chief = _extract_value(getattr(integration, 'rms_to_chief', None))
                rms_to_centroid = _extract_value(getattr(integration, 'rms_to_centroid', None))
                strehl_ratio = _extract_value(getattr(integration, 'strehl_ratio', None))

            # Extract individual Zernike coefficients
            coefficients = []
            if hasattr(zdata, 'coefficients') and zdata.coefficients:
                for term, coeff in zdata.coefficients.items():
                    try:
                        coefficients.append({
                            "term": int(term),
                            "value": float(coeff.value),
                            "formula": str(coeff.formula) if hasattr(coeff, 'formula') else "",
                        })
                    except (ValueError, TypeError, AttributeError) as e:
                        logger.debug(f"Skipping Zernike term {term}: {e}")

            if not coefficients:
                return {"success": False, "error": "No Zernike coefficients extracted"}

            result = {
                "success": True,
                "coefficients": coefficients,
                "pv_to_chief": pv_to_chief,
                "pv_to_centroid": pv_to_centroid,
                "rms_to_chief": rms_to_chief,
                "rms_to_centroid": rms_to_centroid,
                "strehl_ratio": strehl_ratio,
                "surface": str(surface),
                "field_x": field_x,
                "field_y": field_y,
                "field_index": field_index,
                "wavelength_index": wavelength_index,
                "wavelength_um": wavelength_um,
                "maximum_term": maximum_term,
            }

            _log_raw_output("get_zernike_standard_coefficients", result)

            return result

        except Exception as e:
            logger.error(f"get_zernike_standard_coefficients failed: {e}")
            return {"success": False, "error": str(e)}


    def get_surface_data_report(self) -> dict[str, Any]:
        """
        Get Surface Data Report from OpticStudio.

        Iterates over every surface in the system and runs the ZosPy SurfaceData
        analysis to collect edge thickness, center thickness, material name,
        refractive index, and surface power for each surface.

        Note: System must be pre-loaded via load_zmx_file().

        Returns:
            On success: {
                "success": True,
                "surfaces": [...],
                "paraxial": {...},
            }
            On error: {"success": False, "error": "..."}
        """
        try:
            zp = self._zp
            num_surfaces = self.oss.LDE.NumberOfSurfaces
            surfaces = []

            for surf_idx in range(num_surfaces):
                entry: dict[str, Any] = {
                    "surface_number": surf_idx,
                    "surface_type": "Standard",
                    "radius": 0.0,
                    "thickness": 0.0,
                    "edge_thickness": 0.0,
                    "material": "",
                    "refractive_index": 0.0,
                    "surface_power": 0.0,
                }

                # Read basic LDE data for this surface
                try:
                    lde_surf = self.oss.LDE.GetSurfaceAt(surf_idx)
                    entry["radius"] = _extract_value(lde_surf.Radius, 0.0)
                    entry["thickness"] = _extract_value(lde_surf.Thickness, 0.0)

                    # Get surface type name
                    try:
                        type_name = str(lde_surf.TypeName) if hasattr(lde_surf, 'TypeName') else "Standard"
                        entry["surface_type"] = type_name
                    except Exception:
                        pass

                    # Get material name from LDE
                    try:
                        mat_cell = lde_surf.MaterialCell
                        if hasattr(mat_cell, 'Value') and mat_cell.Value:
                            entry["material"] = str(mat_cell.Value)
                    except Exception:
                        pass
                except Exception as e:
                    logger.debug(f"Surface {surf_idx} LDE read error: {e}")

                # Run ZosPy SurfaceData analysis for this surface
                try:
                    sd_analysis = zp.analyses.reports.SurfaceData(surface=surf_idx)
                    sd_start = time.perf_counter()
                    try:
                        sd_result = sd_analysis.run(self.oss)
                    finally:
                        sd_elapsed = (time.perf_counter() - sd_start) * 1000
                        if surf_idx == 0:
                            log_timing(logger, f"SurfaceData.run(surf={surf_idx})", sd_elapsed)

                    if sd_result and hasattr(sd_result, 'data') and sd_result.data is not None:
                        data = sd_result.data

                        # Edge thickness
                        if hasattr(data, 'edge_thickness') and data.edge_thickness is not None:
                            et = data.edge_thickness
                            # EdgeThickness has .y and .x — use .y (tangential plane)
                            if hasattr(et, 'y'):
                                entry["edge_thickness"] = _extract_value(et.y, 0.0)
                            elif hasattr(et, 'x'):
                                entry["edge_thickness"] = _extract_value(et.x, 0.0)

                        # Thickness override from analysis if available
                        if hasattr(data, 'thickness') and data.thickness is not None:
                            entry["thickness"] = _extract_value(data.thickness, entry["thickness"])

                        # Material and refractive index
                        if hasattr(data, 'material') and data.material is not None:
                            mat_data = data.material

                            # Glass name
                            if hasattr(mat_data, 'glass') and mat_data.glass is not None:
                                glass = mat_data.glass
                                if isinstance(glass, str):
                                    if glass and not entry["material"]:
                                        entry["material"] = glass
                                elif hasattr(glass, 'name'):
                                    entry["material"] = str(glass.name)
                                else:
                                    glass_str = str(glass)
                                    if glass_str and not entry["material"]:
                                        entry["material"] = glass_str

                            # Refractive index — take first wavelength's index
                            if hasattr(mat_data, 'indices') and mat_data.indices:
                                try:
                                    first_idx = mat_data.indices[0]
                                    if hasattr(first_idx, 'index'):
                                        entry["refractive_index"] = _extract_value(first_idx.index, 0.0)
                                    elif hasattr(first_idx, 'value'):
                                        entry["refractive_index"] = _extract_value(first_idx.value, 0.0)
                                except (IndexError, TypeError):
                                    pass

                        # Surface power (in air)
                        if hasattr(data, 'surface_powers') and data.surface_powers is not None:
                            sp = data.surface_powers
                            # Prefer "in_air" power
                            power_src = getattr(sp, 'in_air', None) or getattr(sp, 'as_situated', None)
                            if power_src is not None:
                                if hasattr(power_src, 'power') and power_src.power is not None:
                                    power_val = power_src.power
                                    if isinstance(power_val, dict):
                                        for v in power_val.values():
                                            entry["surface_power"] = _extract_value(v, 0.0)
                                            break
                                    else:
                                        entry["surface_power"] = _extract_value(power_val, 0.0)
                                elif hasattr(power_src, 'efl') and power_src.efl is not None:
                                    efl_val = power_src.efl
                                    if isinstance(efl_val, dict):
                                        for v in efl_val.values():
                                            efl_num = _extract_value(v, 0.0)
                                            if efl_num != 0.0:
                                                entry["surface_power"] = 1.0 / efl_num
                                            break
                                    else:
                                        efl_num = _extract_value(efl_val, 0.0)
                                        if efl_num != 0.0:
                                            entry["surface_power"] = 1.0 / efl_num

                except Exception as e:
                    logger.debug(f"SurfaceData analysis for surface {surf_idx} failed: {e}")

                surfaces.append(entry)

            # Get paraxial data
            paraxial = self.get_paraxial_data()

            result = {
                "success": True,
                "surfaces": surfaces,
                "paraxial": paraxial,
            }
            _log_raw_output("/surface-data-report", result)
            return result

        except Exception as e:
            logger.error(f"get_surface_data_report failed: {e}")
            return {"success": False, "error": str(e)}

    def get_surface_curvature(
        self,
        surface: int = 1,
        sampling: str = "65x65",
        show_as: str = "Surface",
        data: str = "TangentialCurvature",
        remove: str = "None_",
    ) -> dict[str, Any]:
        """
        Get surface curvature map using ZosPy's Curvature analysis.

        This is a "dumb executor" - returns raw numpy array data only.
        Curvature map image rendering happens on Mac side.

        Note: System must be pre-loaded via load_zmx_file().

        Args:
            surface: Surface number to analyze (1-indexed)
            sampling: Grid sampling resolution (e.g., '65x65', '129x129')
            show_as: Display format ('Surface', 'Contour', 'GreyScale', etc.)
            data: Curvature data type ('TangentialCurvature', 'SagittalCurvature',
                  'X_Curvature', 'Y_Curvature')
            remove: Removal option ('None_', 'BaseROC', 'BestFitSphere')

        Returns:
            On success: {
                "success": True,
                "image": str (base64 numpy array),
                "image_format": "numpy_array",
                "array_shape": [h, w],
                "array_dtype": str,
                "min_curvature": float,
                "max_curvature": float,
                "mean_curvature": float,
                "surface_number": int,
                "data_type": str,
            }
            On error: {"success": False, "error": "..."}
        """
        try:
            # Validate surface number
            num_surfaces = self.oss.LDE.NumberOfSurfaces
            if surface < 1 or surface >= num_surfaces:
                return {
                    "success": False,
                    "error": f"Surface {surface} out of range (1 to {num_surfaces - 1})",
                }

            # Run Curvature analysis
            curvature_analysis = self._zp.analyses.surface.Curvature(
                surface=surface,
                sampling=sampling,
                show_as=show_as,
                data=data,
                remove=remove,
            )

            curv_start = time.perf_counter()
            try:
                curvature_result = curvature_analysis.run(self.oss, oncomplete="Release")
            finally:
                curv_elapsed_ms = (time.perf_counter() - curv_start) * 1000
                log_timing(logger, "Curvature.run", curv_elapsed_ms)

            # Extract grid data as numpy array
            image_b64 = None
            array_shape = None
            array_dtype = None
            min_curvature = None
            max_curvature = None
            mean_curvature = None

            if hasattr(curvature_result, 'data') and curvature_result.data is not None:
                curv_data = curvature_result.data
                # Curvature.run() returns AnalysisResult where .data is CurvatureResult
                # (or list[CurvatureResult]). The actual grid is in CurvatureResult.data.
                if hasattr(curv_data, 'data') and curv_data.data is not None:
                    curv_data = curv_data.data
                elif isinstance(curv_data, list) and len(curv_data) > 0:
                    curv_data = curv_data[0].data if hasattr(curv_data[0], 'data') else curv_data[0]
                if hasattr(curv_data, 'values'):
                    arr = np.array(curv_data.values, dtype=np.float64)
                else:
                    arr = np.array(curv_data, dtype=np.float64)

                if arr.ndim >= 2:
                    # Compute statistics on valid (non-NaN) values
                    valid_mask = ~np.isnan(arr)
                    if np.any(valid_mask):
                        min_curvature = float(np.nanmin(arr))
                        max_curvature = float(np.nanmax(arr))
                        mean_curvature = float(np.nanmean(arr))

                    image_b64 = base64.b64encode(arr.tobytes()).decode('utf-8')
                    array_shape = list(arr.shape)
                    array_dtype = str(arr.dtype)
                    logger.info(
                        f"Curvature map generated: shape={arr.shape}, "
                        f"dtype={arr.dtype}, min={min_curvature}, max={max_curvature}"
                    )

            if image_b64 is None:
                return {
                    "success": False,
                    "error": "Curvature analysis returned no grid data",
                }

            result = {
                "success": True,
                "image": image_b64,
                "image_format": "numpy_array",
                "array_shape": array_shape,
                "array_dtype": array_dtype,
                "min_curvature": min_curvature,
                "max_curvature": max_curvature,
                "mean_curvature": mean_curvature,
                "surface_number": surface,
                "data_type": data,
            }
            _log_raw_output("/surface-curvature", result)
            return result

        except Exception as e:
            logger.error(f"get_surface_curvature failed: {e}")
            return {"success": False, "error": f"Surface curvature analysis failed: {e}"}

    # =========================================================================
    # Geometric Image Analysis
    # =========================================================================

    def get_geometric_image_analysis(
        self,
        field_size: float = 0.0,
        image_size: float = 50.0,
        rays_x_1000: int = 10,
        number_of_pixels: int = 100,
        field: int = 1,
        wavelength: str | int = "All",
    ) -> dict[str, Any]:
        """
        Run Geometric Image Analysis to simulate how an extended scene looks
        through the optical system.

        Uses ZosPy's GeometricImageAnalysis from the extendedscene module.
        Returns the simulated image as a base64-encoded numpy array, plus
        paraxial metadata.

        This is a "dumb executor" -- returns raw 2D intensity grid as base64 numpy.
        Image rendering happens on the Mac side.

        Note: System must be pre-loaded via load_zmx_file().

        Args:
            field_size: Image width in field coordinates (0 = auto)
            image_size: Detector size in lens units
            rays_x_1000: Approximate ray count in thousands (1-100)
            number_of_pixels: Pixels across image width (10-1000)
            field: Field number (1-indexed)
            wavelength: Wavelength selection ('All' or wavelength number)

        Returns:
            On success: {
                "success": True,
                "image": str (base64 numpy array),
                "image_format": "numpy_array",
                "array_shape": [h, w],
                "array_dtype": str,
                "field_size": float,
                "image_size": float,
                "rays_x_1000": int,
                "number_of_pixels": int,
                "paraxial": dict,
            }
            On error: {"success": False, "error": "..."}
        """
        try:
            from zospy.analyses.extendedscene.geometric_image_analysis import GeometricImageAnalysis

            logger.info(
                f"[GEO_IMAGE] Starting: field_size={field_size}, image_size={image_size}, "
                f"rays_x_1000={rays_x_1000}, number_of_pixels={number_of_pixels}, "
                f"field={field}, wavelength={wavelength}"
            )

            # Create and run the analysis
            analysis = GeometricImageAnalysis(
                field_size=field_size,
                image_size=image_size,
                rays_x_1000=rays_x_1000,
                number_of_pixels=number_of_pixels,
                field=field,
                wavelength=wavelength,
                show_as="Surface",
                source="Uniform",
                file="LETTERF.IMA",
                remove_vignetting_factors=True,
            )

            geo_start = time.perf_counter()
            try:
                result = analysis.run(self.oss)
            finally:
                geo_elapsed_ms = (time.perf_counter() - geo_start) * 1000
                log_timing(logger, "GeometricImageAnalysis.run", geo_elapsed_ms)

            # Extract the image data grid
            image_b64 = None
            array_shape = None
            array_dtype = None

            if result is not None:
                try:
                    # run() returns AnalysisResult where .data is a DataFrame.
                    # Extract the DataFrame from the AnalysisResult wrapper.
                    actual_data = result.data if hasattr(result, 'data') else result

                    if actual_data is None:
                        return {"success": False, "error": "Geometric Image Analysis returned no data"}

                    if hasattr(actual_data, 'values'):
                        arr = np.array(actual_data.values, dtype=np.float64)
                    elif isinstance(actual_data, np.ndarray):
                        arr = actual_data.astype(np.float64)
                    else:
                        logger.warning(f"[GEO_IMAGE] Unexpected result type: {type(actual_data)}")
                        return {"success": False, "error": f"Unexpected result type: {type(actual_data)}"}

                    if arr.size == 0:
                        return {"success": False, "error": "Geometric Image Analysis returned empty data"}

                    logger.info(f"[GEO_IMAGE] Extracted image: shape={arr.shape}, min={arr.min():.4f}, max={arr.max():.4f}")

                    image_b64 = base64.b64encode(arr.tobytes()).decode('utf-8')
                    array_shape = list(arr.shape)
                    array_dtype = str(arr.dtype)

                except Exception as e:
                    logger.warning(f"[GEO_IMAGE] Could not extract data grid: {e}")
                    return {"success": False, "error": f"Failed to extract image data: {e}"}
            else:
                return {"success": False, "error": "Geometric Image Analysis returned None"}

            # Get paraxial data
            paraxial = self._get_paraxial_from_lde()

            result_dict = {
                "success": True,
                "image": image_b64,
                "image_format": "numpy_array",
                "array_shape": array_shape,
                "array_dtype": array_dtype,
                "field_size": field_size,
                "image_size": image_size,
                "rays_x_1000": rays_x_1000,
                "number_of_pixels": number_of_pixels,
                "paraxial": paraxial,
            }
            _log_raw_output("/geometric-image-analysis", result_dict)
            return result_dict

        except ImportError as e:
            logger.error(f"[GEO_IMAGE] GeometricImageAnalysis not available: {e}")
            return {
                "success": False,
                "error": "GeometricImageAnalysis requires ZosPy >= 1.3.0 with extendedscene support",
            }
        except Exception as e:
            logger.error(f"[GEO_IMAGE] Failed: {e}")
            return {"success": False, "error": f"Geometric Image Analysis failed: {e}"}


    # =========================================================================
    # Physical Optics Propagation
    # =========================================================================

    def get_physical_optics_propagation(
        self,
        field_index: int = 1,
        wavelength_index: int = 1,
        beam_type: str = "GaussianWaist",
        waist_x: Optional[float] = None,
        waist_y: Optional[float] = None,
        x_sampling: int = 64,
        y_sampling: int = 64,
        x_width: float = 4.0,
        y_width: float = 4.0,
        start_surface: int = 1,
        end_surface: str = "Image",
        use_polarization: bool = False,
        data_type: str = "Irradiance",
    ) -> dict[str, Any]:
        """
        Run Physical Optics Propagation (POP) analysis.

        This is a "dumb executor" -- returns raw 2D beam profile grid as base64 numpy
        plus beam parameters. Image rendering happens on the Mac side.
        """
        try:
            # Validate field and wavelength indices
            fields = self.oss.SystemData.Fields
            if field_index > fields.NumberOfFields:
                return {"success": False, "error": f"Field index {field_index} out of range (max: {fields.NumberOfFields})"}

            wavelengths = self.oss.SystemData.Wavelengths
            if wavelength_index > wavelengths.NumberOfWavelengths:
                return {"success": False, "error": f"Wavelength index {wavelength_index} out of range (max: {wavelengths.NumberOfWavelengths})"}

            field = fields.GetField(field_index)
            field_x = _extract_value(field.X)
            field_y = _extract_value(field.Y)
            wavelength_um = _extract_value(wavelengths.GetWavelength(wavelength_index).Wavelength, 0.5876)

            # Build beam parameter dict via ZosPy helper
            beam_params = {}
            try:
                beam_params = self._zp.analyses.physicaloptics.physical_optics_propagation.create_beam_parameter_dict(
                    self.oss, beam_type=beam_type,
                )
            except Exception as e:
                logger.warning(f"create_beam_parameter_dict failed (beam_type={beam_type}): {e}")

            # Override waist sizes if provided
            if waist_x is not None:
                matched = False
                for key in list(beam_params.keys()):
                    if "waist" in key.lower() and ("x" in key.lower() or "size" in key.lower()):
                        beam_params[key] = waist_x
                        matched = True
                        break
                if not matched and "Waist X" in beam_params:
                    beam_params["Waist X"] = waist_x

            if waist_y is not None:
                matched = False
                for key in list(beam_params.keys()):
                    if "waist" in key.lower() and "y" in key.lower():
                        beam_params[key] = waist_y
                        matched = True
                        break
                if not matched and "Waist Y" in beam_params:
                    beam_params["Waist Y"] = waist_y

            logger.info(f"POP beam_params: {beam_params}")

            # Build POP settings
            pop_kwargs = {
                "field": field_index,
                "wavelength": wavelength_index,
                "beam_type": beam_type,
                "x_sampling": x_sampling,
                "y_sampling": y_sampling,
                "x_width": x_width,
                "y_width": y_width,
                "start_surface": start_surface,
                "end_surface": end_surface,
                "use_polarization": use_polarization,
                "data_type": data_type,
                "show_as": "FalseColor",
            }

            if beam_params:
                pop_kwargs["beam_parameters"] = beam_params

            logger.info(
                f"POP settings: field={field_index}, wl={wavelength_index}, beam={beam_type}, "
                f"sampling={x_sampling}x{y_sampling}, width={x_width}x{y_width}"
            )

            # Try the high-level ZosPy wrapper first
            image_b64 = None
            array_shape = None
            array_dtype = None

            try:
                pop_analysis = self._zp.analyses.physicaloptics.physical_optics_propagation.PhysicalOpticsPropagation(
                    **pop_kwargs,
                )

                pop_start = time.perf_counter()
                try:
                    pop_result = pop_analysis.run(self.oss)
                finally:
                    pop_elapsed_ms = (time.perf_counter() - pop_start) * 1000
                    log_timing(logger, "PhysicalOpticsPropagation.run", pop_elapsed_ms)

                if pop_result is not None and hasattr(pop_result, 'data') and pop_result.data is not None and pop_result.data.size > 0:
                    # ZosPy wrapper returns AnalysisResult; .data is the DataFrame
                    arr = pop_result.data.values.astype(np.float64)
                    image_b64 = base64.b64encode(arr.tobytes()).decode('utf-8')
                    array_shape = list(arr.shape)
                    array_dtype = str(arr.dtype)
                    logger.info(f"POP: Extracted data array shape={arr.shape}")

            except Exception as e:
                logger.warning(f"POP high-level wrapper failed: {e}, trying raw API")

            # Fallback: use raw ZOSAPI DataGrid approach
            if image_b64 is None:
                try:
                    idm = self._zp.constants.Analysis.AnalysisIDM
                    analysis = self._zp.analyses.new_analysis(
                        self.oss,
                        idm.PhysicalOpticsPropagation,
                        settings_first=True,
                    )

                    try:
                        settings = analysis.Settings
                        self._configure_analysis_settings(
                            settings,
                            field_index=field_index,
                            wavelength_index=wavelength_index,
                        )
                        # POP uses XSampling/YSampling instead of the generic SampleSize
                        for attr, val in (('XSampling', x_sampling), ('YSampling', y_sampling)):
                            if hasattr(settings, attr):
                                try:
                                    setattr(settings, attr, val)
                                except Exception:
                                    pass

                        pop_start2 = time.perf_counter()
                        try:
                            analysis.ApplyAndWaitForCompletion()
                        finally:
                            pop_elapsed2_ms = (time.perf_counter() - pop_start2) * 1000
                            log_timing(logger, "POP.raw_api.run", pop_elapsed2_ms)

                        results = analysis.Results
                        if results is not None:
                            num_grids = results.NumberOfDataGrids
                            if num_grids > 0:
                                grid = results.GetDataGrid(0)
                                if grid is not None:
                                    nx = grid.Nx if hasattr(grid, 'Nx') else 0
                                    ny = grid.Ny if hasattr(grid, 'Ny') else 0
                                    if nx > 0 and ny > 0:
                                        arr = np.zeros((ny, nx), dtype=np.float64)
                                        for yi in range(ny):
                                            for xi in range(nx):
                                                val = _extract_value(grid.Z(xi, yi))
                                                arr[yi, xi] = val

                                        image_b64 = base64.b64encode(arr.tobytes()).decode('utf-8')
                                        array_shape = list(arr.shape)
                                        array_dtype = str(arr.dtype)
                                        logger.info(f"POP: Extracted from DataGrid shape={arr.shape}")
                    finally:
                        try:
                            analysis.Close()
                        except Exception:
                            pass
                except Exception as e:
                    logger.warning(f"POP: Raw API fallback failed: {e}")

            if image_b64 is None:
                return {"success": False, "error": "Physical Optics Propagation returned no beam profile data"}

            result = {
                "success": True,
                "image": image_b64,
                "image_format": "numpy_array",
                "array_shape": array_shape,
                "array_dtype": array_dtype,
                "beam_params": beam_params,
                "wavelength_um": wavelength_um,
                "field_x": field_x,
                "field_y": field_y,
                "data_type": data_type,
            }

            _log_raw_output("get_physical_optics_propagation", result)

            return result

        except Exception as e:
            logger.error(f"get_physical_optics_propagation failed: {e}")
            return {"success": False, "error": str(e)}


    def get_ray_fan(
        self,
        field_index: int = 0,
        wavelength_index: int = 0,
        plot_scale: float = 0.0,
        number_of_rays: int = 20,
    ) -> dict[str, Any]:
        """
        Get Ray Fan (Ray Aberration) data using ZosPy's RayFan analysis.

        Returns raw pupil/aberration data per field. Image rendering on Mac side.

        Args:
            field_index: 0 = all fields, 1+ = specific field (1-indexed)
            wavelength_index: 0 = all wavelengths, 1+ = specific (1-indexed)
            plot_scale: Max vertical scale; 0 = auto
            number_of_rays: Rays traced on each side of origin
        """
        try:
            sys_fields = self.oss.SystemData.Fields
            num_fields = sys_fields.NumberOfFields
            sys_wl = self.oss.SystemData.Wavelengths
            num_wl = sys_wl.NumberOfWavelengths

            field_arg = "All" if field_index == 0 else field_index
            if field_index > 0 and field_index > num_fields:
                return {"success": False, "error": f"Field index {field_index} out of range (max: {num_fields})"}

            wl_arg = "All" if wavelength_index == 0 else wavelength_index
            if wavelength_index > 0 and wavelength_index > num_wl:
                return {"success": False, "error": f"Wavelength index {wavelength_index} out of range (max: {num_wl})"}

            logger.info(f"RayFan: field={field_arg}, wl={wl_arg}, rays={number_of_rays}")

            from zospy.analyses.raysandspots import RayFan
            analysis = RayFan(
                plot_scale=plot_scale,
                number_of_rays=number_of_rays,
                field=field_arg,
                wavelength=wl_arg,
            )

            t0 = time.perf_counter()
            try:
                rfr = analysis.run(self.oss)
            finally:
                log_timing(logger, "RayFan.run", (time.perf_counter() - t0) * 1000)

            # ZOSPy run() returns AnalysisResult wrapper whose .data holds the
            # typed result (RayFanResult) with to_dataframe(). Handle both patterns.
            df = _extract_dataframe(rfr, "RayFan")
            if df is None:
                return {"success": False, "error": "RayFan result has no extractable DataFrame"}
            logger.debug(f"RayFan df cols={list(df.columns)}, shape={df.shape}")
            if df.empty:
                return {"success": False, "error": "RayFan returned empty data"}

            all_fans = []
            max_ab = 0.0
            meta = {"Direction", "Field Number", "FieldX", "FieldY", "Pupil", "Wavelength"}
            vcols = [c for c in df.columns if c not in meta]
            vcol = vcols[0] if vcols else df.columns[-1]

            gcols = []
            if "Field Number" in df.columns:
                gcols.append("Field Number")
            if "Wavelength" in df.columns:
                gcols.append("Wavelength")
            if not gcols:
                return {"success": False, "error": "RayFan DataFrame missing expected columns"}

            for gk, gdf in df.groupby(gcols, sort=True):
                if isinstance(gk, tuple):
                    fnum = int(gk[0]) if len(gk) > 0 else 1
                    wval = float(gk[1]) if len(gk) > 1 else 0.0
                else:
                    fnum = int(gk)
                    wval = 0.0

                fx = float(gdf["FieldX"].iloc[0]) if "FieldX" in gdf.columns else 0.0
                fy = float(gdf["FieldY"].iloc[0]) if "FieldY" in gdf.columns else 0.0

                wi = 0
                if wval > 0:
                    for k in range(1, num_wl + 1):
                        if abs(_extract_value(sys_wl.GetWavelength(k).Wavelength, 0.0) - wval) < 1e-6:
                            wi = k
                            break

                hd = "Direction" in gdf.columns
                tdf = gdf[gdf["Direction"] == "Tangential"] if hd else gdf
                sdf = gdf[gdf["Direction"] == "Sagittal"] if hd else gdf

                tpy = tdf["Pupil"].tolist() if not tdf.empty and "Pupil" in tdf.columns else []
                tey = tdf[vcol].tolist() if not tdf.empty and vcol in tdf.columns else []
                spx = sdf["Pupil"].tolist() if not sdf.empty and "Pupil" in sdf.columns else []
                sex = sdf[vcol].tolist() if not sdf.empty and vcol in sdf.columns else []

                for arr in (tey, sex):
                    vld = [abs(v) for v in arr if not math.isnan(v)]
                    if vld:
                        max_ab = max(max_ab, max(vld))

                tey = [0.0 if math.isnan(v) else v for v in tey]
                sex = [0.0 if math.isnan(v) else v for v in sex]

                all_fans.append({
                    "field_index": fnum - 1,
                    "field_x": fx, "field_y": fy,
                    "wavelength_um": wval, "wavelength_index": wi,
                    "tangential_py": tpy, "tangential_ey": tey,
                    "sagittal_px": spx, "sagittal_ex": sex,
                })

            result = {
                "success": True, "fans": all_fans,
                "max_aberration": float(max_ab),
                "num_fields": num_fields, "num_wavelengths": num_wl,
            }
            _log_raw_output("/ray-fan", result)
            return result

        except Exception as e:
            logger.error(f"RayFan failed: {e}", exc_info=True)
            return {"success": False, "error": f"Ray Fan analysis failed: {e}"}

    # =========================================================================
    # Polarization Analyses
    # =========================================================================

    def get_polarization_pupil_map(
        self,
        field_index: int = 1,
        wavelength_index: int = 1,
        surface: str = "Image",
        sampling: str = "11x11",
        jx: float = 1.0,
        jy: float = 0.0,
        x_phase: float = 0.0,
        y_phase: float = 0.0,
    ) -> dict[str, Any]:
        """Get Polarization Pupil Map data via ZosPy PolarizationPupilMap wrapper."""
        try:
            fields = self.oss.SystemData.Fields
            if field_index > fields.NumberOfFields:
                return {"success": False, "error": f"Field index {field_index} out of range (max: {fields.NumberOfFields})"}
            field = fields.GetField(field_index)
            field_x = _extract_value(field.X)
            field_y = _extract_value(field.Y)

            wavelengths = self.oss.SystemData.Wavelengths
            if wavelength_index > wavelengths.NumberOfWavelengths:
                return {"success": False, "error": f"Wavelength index {wavelength_index} out of range (max: {wavelengths.NumberOfWavelengths})"}
            wavelength_um = _extract_value(wavelengths.GetWavelength(wavelength_index).Wavelength, 0.5876)

            ppm_analysis = self._zp.analyses.polarization.PolarizationPupilMap(
                jx=jx, jy=jy, x_phase=x_phase, y_phase=y_phase,
                wavelength=wavelength_index, field=field_index,
                surface=surface, sampling=sampling,
            )
            ppm_start = time.perf_counter()
            try:
                ppm_result = ppm_analysis.run(self.oss)
            finally:
                log_timing(logger, "PolarizationPupilMap.run", (time.perf_counter() - ppm_start) * 1000)

            if ppm_result is None or not hasattr(ppm_result, 'data') or ppm_result.data is None:
                return {"success": False, "error": "PolarizationPupilMap returned no data"}
            data = ppm_result.data

            transmission = _extract_value(data.transmission) if hasattr(data, 'transmission') else None

            pupil_map_data, pupil_map_columns, pupil_map_shape = [], [], []
            if hasattr(data, 'pupil_map') and data.pupil_map is not None:
                df = data.pupil_map
                try:
                    pupil_map_data = df.values.tolist()
                    pupil_map_columns = list(df.columns)
                    pupil_map_shape = list(df.shape)
                    logger.info(f"PolarizationPupilMap: columns={pupil_map_columns}, shape={df.shape}")
                except Exception as e:
                    logger.warning(f"PolarizationPupilMap: Could not extract DataFrame: {e}")

            result = {
                "success": True,
                "pupil_map": pupil_map_data,
                "pupil_map_columns": pupil_map_columns,
                "pupil_map_shape": pupil_map_shape,
                "transmission": transmission,
                "x_field": _extract_value(getattr(data, 'x_field', None)),
                "y_field": _extract_value(getattr(data, 'y_field', None)),
                "x_phase": _extract_value(getattr(data, 'x_phase', None)),
                "y_phase": _extract_value(getattr(data, 'y_phase', None)),
                "field_x": field_x, "field_y": field_y,
                "field_index": field_index, "wavelength_index": wavelength_index,
                "wavelength_um": wavelength_um, "surface": str(surface),
                "sampling": sampling, "jx": jx, "jy": jy,
                "input_x_phase": x_phase, "input_y_phase": y_phase,
            }
            _log_raw_output("get_polarization_pupil_map", result)
            return result
        except Exception as e:
            logger.error(f"get_polarization_pupil_map failed: {e}")
            return {"success": False, "error": str(e)}

    def get_polarization_transmission(
        self,
        sampling: str = "32x32",
        unpolarized: bool = False,
        jx: float = 1.0,
        jy: float = 0.0,
        x_phase: float = 0.0,
        y_phase: float = 0.0,
    ) -> dict[str, Any]:
        """Get Polarization Transmission data via ZosPy PolarizationTransmission wrapper."""
        try:
            fields = self.oss.SystemData.Fields
            num_fields = fields.NumberOfFields
            wavelengths = self.oss.SystemData.Wavelengths
            num_wavelengths = wavelengths.NumberOfWavelengths

            field_info = []
            for i in range(1, num_fields + 1):
                f = fields.GetField(i)
                field_info.append({"index": i, "x": _extract_value(f.X), "y": _extract_value(f.Y)})

            wavelength_info = []
            for i in range(1, num_wavelengths + 1):
                wl = wavelengths.GetWavelength(i)
                wavelength_info.append({"index": i, "um": _extract_value(wl.Wavelength, 0.5876)})

            pt_analysis = self._zp.analyses.polarization.PolarizationTransmission(
                sampling=sampling, unpolarized=unpolarized,
                jx=jx, jy=jy, x_phase=x_phase, y_phase=y_phase,
            )
            pt_start = time.perf_counter()
            try:
                pt_result = pt_analysis.run(self.oss)
            finally:
                log_timing(logger, "PolarizationTransmission.run", (time.perf_counter() - pt_start) * 1000)

            if pt_result is None or not hasattr(pt_result, 'data') or pt_result.data is None:
                return {"success": False, "error": "PolarizationTransmission returned no data"}
            data = pt_result.data

            field_transmissions = []
            if hasattr(data, 'field_transmissions') and data.field_transmissions:
                for ft in data.field_transmissions:
                    ft_entry = {
                        "field_pos": _extract_value(getattr(ft, 'field_pos', None)),
                        "total_transmission": _extract_value(getattr(ft, 'total_transmission', None)),
                    }
                    if hasattr(ft, 'transmissions') and ft.transmissions:
                        try:
                            ft_entry["transmissions"] = {str(k): float(v) for k, v in ft.transmissions.items()}
                        except Exception as e:
                            logger.warning(f"Could not extract field transmission details: {e}")
                    field_transmissions.append(ft_entry)

            chief_ray_transmissions = []
            if hasattr(data, 'chief_ray_transmissions') and data.chief_ray_transmissions:
                for crt in data.chief_ray_transmissions:
                    crt_entry = {"field_pos": _extract_value(getattr(crt, 'field_pos', None))}
                    if hasattr(crt, 'wavelength') and crt.wavelength:
                        crt_entry["wavelength"] = {str(k): _extract_value(v) for k, v in crt.wavelength.items()}
                    if hasattr(crt, 'transmissions') and crt.transmissions is not None:
                        try:
                            df = crt.transmissions
                            crt_entry["transmissions_data"] = df.values.tolist()
                            crt_entry["transmissions_columns"] = list(df.columns)
                        except Exception as e:
                            logger.warning(f"Could not extract chief ray transmission DataFrame: {e}")
                    chief_ray_transmissions.append(crt_entry)

            result = {
                "success": True,
                "field_transmissions": field_transmissions,
                "chief_ray_transmissions": chief_ray_transmissions,
                "x_field": float(data.x_field) if getattr(data, 'x_field', None) is not None else None,
                "y_field": float(data.y_field) if getattr(data, 'y_field', None) is not None else None,
                "x_phase": float(data.x_phase) if getattr(data, 'x_phase', None) is not None else None,
                "y_phase": float(data.y_phase) if getattr(data, 'y_phase', None) is not None else None,
                "grid_size": str(getattr(data, 'grid_size', sampling)),
                "num_fields": num_fields, "num_wavelengths": num_wavelengths,
                "field_info": field_info, "wavelength_info": wavelength_info,
                "unpolarized": unpolarized, "jx": jx, "jy": jy,
                "input_x_phase": x_phase, "input_y_phase": y_phase,
            }
            _log_raw_output("get_polarization_transmission", result)
            return result
        except Exception as e:
            logger.error(f"get_polarization_transmission failed: {e}")
            return {"success": False, "error": str(e)}


class ZosPyError(Exception):
    """Exception raised when ZosPy operations fail."""
    pass
