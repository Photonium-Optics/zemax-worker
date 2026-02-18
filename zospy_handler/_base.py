"""
ZosPy Handler

Manages the connection to Zemax OpticStudio and executes ZosPy operations.
This module contains all the actual ZosPy/OpticStudio calls.

Note: This code runs on Windows only, where OpticStudio is installed.

Each uvicorn worker process gets its own OpticStudio connection (via ZOS singleton).
Multiple workers are supported — the constraint is license seats, not threading.
"""

import json
import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from utils.timing import log_timing
from config import (
    _RAW_LOG_MAX_CHARS, _ARRAY_SUMMARY_FIELDS, _ARRAY_SUMMARY_MAX,
    _BINARY_FIELDS, _TEXT_TRUNCATE_FIELDS,
    DEFAULT_NUM_CROSS_SECTION_RAYS, SAMPLING_INT_FALLBACK, FIELD_TYPE_MAP,
    CROSS_SECTION_IMAGE_SIZE, CROSS_SECTION_TEMP_FILENAME, SEIDEL_TEMP_FILENAME,
    MIN_IMAGE_EXPORT_VERSION, FNO_APERTURE_TYPES,
)

# Configure module logger
logger = logging.getLogger(__name__)

# Dedicated logger for raw Zemax analysis output (filterable in dashboard)
logger_raw = logging.getLogger("zemax.raw")


@dataclass
class GridWithMetadata:
    """Data grid with optional spatial extent metadata from ZOS-API IGrid."""
    data: np.ndarray
    min_x: Optional[float] = None
    max_x: Optional[float] = None
    min_y: Optional[float] = None
    max_y: Optional[float] = None
    dx: Optional[float] = None
    dy: Optional[float] = None

    @property
    def extent_x(self) -> Optional[float]:
        """Total grid extent in X (MaxX - MinX), or None if bounds are unavailable."""
        if self.min_x is not None and self.max_x is not None:
            return self.max_x - self.min_x
        return None

    @property
    def extent_y(self) -> Optional[float]:
        """Total grid extent in Y (MaxY - MinY), or None if bounds are unavailable."""
        if self.min_y is not None and self.max_y is not None:
            return self.max_y - self.min_y
        return None


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
    global _zp, _ZOSPY_IMPORT_ATTEMPTED, _ZOSPY_AVAILABLE

    if _ZOSPY_IMPORT_ATTEMPTED:
        return _ZOSPY_AVAILABLE

    _ZOSPY_IMPORT_ATTEMPTED = True
    logger.info("Lazily importing ZosPy (this may take a moment)...")

    try:
        import zospy as zp_module
        _zp = zp_module
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

# Constants imported from config.py


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


class ZosPyError(Exception):
    """Exception raised when ZosPy operations fail."""
    pass


class ZosPyHandlerBase:
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
        Get effective focal length via ZOS-API GetOperandValue (read-only, no MFE modification).

        Returns:
            Effective focal length in mm, or None if calculation fails.
        """
        try:
            effl_type = self._zp.constants.Editors.MFE.MeritOperandType.EFFL
            efl = float(self.oss.MFE.GetOperandValue(effl_type, 0, 0, 0, 0, 0, 0, 0, 0))
            if np.isnan(efl) or np.isinf(efl):
                return None
            return efl
        except Exception as e:
            logger.warning(f"_get_efl failed: {e}")
            return None

    def _get_bfl(self) -> Optional[float]:
        """
        Get back focal length (distance from last surface to image plane).

        Returns:
            Back focal length in lens units, or None if calculation fails.
        """
        try:
            last_surf_idx = self.oss.LDE.NumberOfSurfaces - 1  # Image surface index
            if last_surf_idx < 2:
                return None
            # BFL = thickness of the surface before the image plane
            bfl = _extract_value(self.oss.LDE.GetSurfaceAt(last_surf_idx - 1).Thickness, None)
            return bfl
        except Exception as e:
            logger.warning(f"_get_bfl failed: {e}")
            return None

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

    @staticmethod
    def _extract_data_grid(grid) -> Optional[np.ndarray]:
        """Extract a 2D data grid from an OpticStudio analysis result.

        Uses tiered bulk .Values extraction when available (single COM call),
        falling back to per-pixel grid.Z(xi, yi) if needed.

        Tiers for .Values path:
          1. np.asarray — instant if pythonnet supports buffer protocol
          2. List comprehension — builds array in one numpy call
          3. Per-element loop — guaranteed fallback
        """
        if hasattr(grid, 'Values'):
            try:
                raw_values = grid.Values
                ny = raw_values.GetLength(0)
                nx = raw_values.GetLength(1)
                if nx > 0 and ny > 0:
                    # Tier 1: direct numpy conversion (zero-copy if supported)
                    try:
                        arr = np.asarray(raw_values, dtype=np.float64)
                        if arr.shape == (ny, nx):
                            logger.debug("_extract_data_grid: Tier 1 (np.asarray) succeeded")
                            return arr
                    except Exception:
                        pass

                    # Tier 2: list comprehension → single numpy call
                    try:
                        arr = np.array(
                            [[raw_values[yi, xi] for xi in range(nx)] for yi in range(ny)],
                            dtype=np.float64,
                        )
                        logger.debug("_extract_data_grid: Tier 2 (list comprehension) succeeded")
                        return arr
                    except Exception:
                        pass

                    # Tier 3: per-element loop (original fallback)
                    arr = np.zeros((ny, nx), dtype=np.float64)
                    for yi in range(ny):
                        for xi in range(nx):
                            arr[yi, xi] = raw_values[yi, xi]
                    logger.debug("_extract_data_grid: Tier 3 (per-element loop) succeeded")
                    return arr
            except Exception:
                pass  # Fall through to per-pixel extraction

        # Fallback: per-pixel extraction via grid.Z()
        nx = grid.Nx if hasattr(grid, 'Nx') else 0
        ny = grid.Ny if hasattr(grid, 'Ny') else 0
        if nx <= 0 or ny <= 0:
            return None
        arr = np.zeros((ny, nx), dtype=np.float64)
        for yi in range(ny):
            for xi in range(nx):
                arr[yi, xi] = _extract_value(grid.Z(xi, yi))
        return arr

    def _extract_grid_with_metadata(self, grid) -> Optional["GridWithMetadata"]:
        """Extract data grid with spatial metadata from a ZOS-API IGrid object.

        Probes the grid for MinX/MaxX/MinY/MaxY/Dx/Dy properties (IGrid interface).
        Safe if properties are absent.
        """
        data = self._extract_data_grid(grid)
        if data is None:
            return None
        meta = GridWithMetadata(data=data)
        for attr, field_name in [
            ('MinX', 'min_x'), ('MaxX', 'max_x'),
            ('MinY', 'min_y'), ('MaxY', 'max_y'),
            ('Dx', 'dx'), ('Dy', 'dy'),
        ]:
            if hasattr(grid, attr):
                try:
                    setattr(meta, field_name, float(getattr(grid, attr)))
                except Exception:
                    pass
        return meta

    def _resolve_sample_size(self, sampling: str):
        """Resolve a sampling string like '64x64' to the ZOSAPI SampleSizes enum value."""
        try:
            sample_sizes = self._zp.constants.Analysis.SampleSizes
            return getattr(sample_sizes, f"S_{sampling}")
        except Exception:
            logger.warning(f"_resolve_sample_size: Could not resolve '{sampling}' via ZOSAPI enum, using integer fallback")
            return SAMPLING_INT_FALLBACK.get(sampling, 2)

    def _configure_analysis_settings(
        self,
        settings,
        field_index: Optional[int] = None,
        wavelength_index: Optional[int] = None,
        sampling: Optional[str] = None,
    ) -> None:
        """Configure common analysis settings (Field, Wavelength, SampleSize).

        Logs warnings on setter failures, since different analysis types
        expose different subsets of these settings.
        """
        if field_index is not None and hasattr(settings, 'Field'):
            try:
                settings.Field.SetFieldNumber(field_index)
            except Exception as e:
                logger.warning(f"Failed to set Field={field_index}: {e}")
        if wavelength_index is not None and hasattr(settings, 'Wavelength'):
            try:
                settings.Wavelength.SetWavelengthNumber(wavelength_index)
            except Exception as e:
                logger.warning(f"Failed to set Wavelength={wavelength_index}: {e}")
        if sampling is not None:
            sample_value = self._resolve_sample_size(sampling)
            # SampleSize is the standard attribute; Huygens analyses use
            # PupilSampleSize/ImageSampleSize instead.
            for attr in ('SampleSize', 'PupilSampleSize', 'ImageSampleSize'):
                if hasattr(settings, attr):
                    try:
                        setattr(settings, attr, sample_value)
                    except Exception as e:
                        logger.warning(f"Failed to set {attr}={sampling} ({sample_value}): {e}")

    def _get_fno(self) -> Optional[float]:
        """
        Get the working f-number of the optical system.

        Uses the WFNO merit function operand (paraxial working F/#) as the
        primary method — this is reliable regardless of aperture type.
        Falls back to aperture settings or EFL/EPD if the operand fails.

        Returns:
            F-number, or None if it cannot be determined.
            Returns None for afocal systems (F/# is meaningless when EFL is infinite).
        """
        try:
            aperture = self.oss.SystemData.Aperture

            # Afocal systems have no meaningful F/# (EFL is infinite)
            if hasattr(aperture, 'AFocalImageSpace') and aperture.AFocalImageSpace:
                return None
        except Exception:
            pass

        # Primary: use WFNO operand (works for all aperture types)
        try:
            wfno_type = self._zp.constants.Editors.MFE.MeritOperandType.WFNO
            fno = float(self.oss.MFE.GetOperandValue(wfno_type, 0, 0, 0, 0, 0, 0, 0, 0))
            if not np.isnan(fno) and not np.isinf(fno) and fno > 0:
                return fno
        except Exception as e:
            logger.debug(f"WFNO operand failed: {e}")

        # Fallback: aperture settings
        try:
            aperture = self.oss.SystemData.Aperture
            aperture_type = ""
            if aperture.ApertureType:
                if hasattr(aperture.ApertureType, 'name'):
                    aperture_type = aperture.ApertureType.name
                else:
                    aperture_type = str(aperture.ApertureType).split(".")[-1]

            if aperture_type in FNO_APERTURE_TYPES:
                val = _extract_value(aperture.ApertureValue)
                if val > 0:
                    return val

            # Calculate from EPD and EFL
            epd = _extract_value(aperture.ApertureValue)
            efl = self._get_efl()
            if epd and efl and epd > 0:
                return efl / epd

        except Exception as e:
            logger.debug(f"Could not get f-number from aperture: {e}")

        return None
