"""Aberrations mixin – Seidel, wavefront, Zernike, RMS vs field."""
from __future__ import annotations

import base64
import logging
import math
import os
import re
import tempfile
import time

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

from config import SEIDEL_TEMP_FILENAME
from zospy_handler._base import _extract_value, _log_raw_output, _parse_zernike_term_number
from utils.timing import log_timing

logger = logging.getLogger(__name__)


def _parse_zernike_full_text(text: str) -> dict:
    """Parse full ZernikeStandardCoefficients text output.

    Extracts individual coefficients plus RMS, P-V, and Strehl metrics.

    The OpticStudio text output has coefficient lines like:
        Z   1      0.12345678     1
        Z   2     -0.01234567     4ρ·Cos(A)

    And metric lines like:
        RMS (to chief)     :   0.12345678
        P-V (to chief)     :   0.56789012
        Strehl Ratio       :   0.90123456

    Returns:
        dict with keys: coefficients, rms_to_chief, rms_to_centroid,
        pv_to_chief, pv_to_centroid, strehl_ratio
    """
    result = {
        "coefficients": [],
        "rms_to_chief": None,
        "rms_to_centroid": None,
        "pv_to_chief": None,
        "pv_to_centroid": None,
        "strehl_ratio": None,
    }

    for line in text.splitlines():
        line_stripped = line.strip()

        # Match coefficient lines: Z   N      value     formula
        m = re.match(r'Z\s+(\d+)\s+([-\d.eE+]+)\s*(.*)', line_stripped)
        if m:
            try:
                result["coefficients"].append({
                    "term": int(m.group(1)),
                    "value": float(m.group(2)),
                    "formula": m.group(3).strip(),
                })
            except (ValueError, TypeError):
                pass
            continue

        # Match RMS lines
        m = re.match(
            r'RMS\s*\(to\s+(chief|centroid)\)\s*:\s*([\d.eE+-]+)',
            line_stripped, re.IGNORECASE,
        )
        if m:
            key = f"rms_to_{m.group(1).lower()}"
            try:
                result[key] = float(m.group(2))
            except ValueError:
                pass
            continue

        # Match P-V lines
        m = re.match(
            r'P-V\s*\(to\s+(chief|centroid)\)\s*:\s*([\d.eE+-]+)',
            line_stripped, re.IGNORECASE,
        )
        if m:
            key = f"pv_to_{m.group(1).lower()}"
            try:
                result[key] = float(m.group(2))
            except ValueError:
                pass
            continue

        # Match Strehl Ratio
        m = re.match(
            r'Strehl\s+Ratio\s*:\s*([\d.eE+-]+)',
            line_stripped, re.IGNORECASE,
        )
        if m:
            try:
                result["strehl_ratio"] = float(m.group(1))
            except ValueError:
                pass

    return result


class AberrationsMixin:

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

    def get_wavefront(
        self,
        field_index: int = 1,
        wavelength_index: int = 1,
        sampling: str = "64x64",
        remove_tilt: bool = False,
    ) -> dict[str, Any]:
        """
        Get wavefront error map and metrics using raw ZOS-API calls.

        Uses ZernikeStandardCoefficients (text output) for RMS/P-V/Strehl,
        and WavefrontMap (data grid) for the 2D wavefront array. Both use
        the raw new_analysis() pattern to avoid ZosPy temp-file issues.

        Note: System must be pre-loaded via load_zmx_file().

        Args:
            field_index: Field index (1-indexed)
            wavelength_index: Wavelength index (1-indexed)
            sampling: Pupil sampling grid (e.g., '32x32', '64x64', '128x128')
            remove_tilt: Whether to remove tilt from wavefront map

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
        zernike_analysis = None
        wfm_analysis = None
        zernike_temp_path = os.path.join(tempfile.gettempdir(), "zospy_zernike_wavefront.txt")

        try:
            # Get field coordinates for response
            fields = self.oss.SystemData.Fields
            if field_index > fields.NumberOfFields:
                return {"success": False, "error": f"Field index {field_index} out of range (max: {fields.NumberOfFields})"}

            field = fields.GetField(field_index)
            field_x = _extract_value(field.X)
            field_y = _extract_value(field.Y)

            # Get wavelength for response
            wavelengths = self.oss.SystemData.Wavelengths
            if wavelength_index > wavelengths.NumberOfWavelengths:
                return {"success": False, "error": f"Wavelength index {wavelength_index} out of range (max: {wavelengths.NumberOfWavelengths})"}

            wavelength_um = _extract_value(wavelengths.GetWavelength(wavelength_index).Wavelength, 0.5876)

            idm = self._zp.constants.Analysis.AnalysisIDM
            image_surf = self.oss.LDE.NumberOfSurfaces - 1

            # -----------------------------------------------------------------
            # Part A: ZernikeStandardCoefficients → RMS, P-V, Strehl
            # Uses GetTextFile + text parsing (same pattern as get_seidel_native)
            # -----------------------------------------------------------------
            rms_waves = None
            pv_waves = None
            strehl_ratio = None

            try:
                zernike_analysis = self._zp.analyses.new_analysis(
                    self.oss, idm.ZernikeStandardCoefficients, settings_first=True,
                )
                settings = zernike_analysis.Settings
                self._configure_analysis_settings(settings, field_index, wavelength_index, sampling)

                # Zernike-specific settings (hasattr guard pattern from get_rms_vs_field)
                if hasattr(settings, 'MaximumNumberOfTerms'):
                    settings.MaximumNumberOfTerms = 37
                elif hasattr(settings, 'MaximumTerm'):
                    settings.MaximumTerm = 37

                if hasattr(settings, 'Surface'):
                    try:
                        settings.Surface.SetSurfaceNumber(image_surf)
                    except Exception as e:
                        logger.warning(f"ZernikeStandardCoefficients: Could not set Surface: {e}")

                # ZOS-API uses "OBD" not "OPD" (naming quirk)
                if hasattr(settings, 'ReferenceOBDToVertex'):
                    settings.ReferenceOBDToVertex = False
                elif hasattr(settings, 'ReferenceOPDToVertex'):
                    settings.ReferenceOPDToVertex = False

                zernike_start = time.perf_counter()
                try:
                    zernike_analysis.ApplyAndWaitForCompletion()
                finally:
                    elapsed = (time.perf_counter() - zernike_start) * 1000
                    log_timing(logger, "ZernikeStandardCoefficients.ApplyAndWaitForCompletion", elapsed)

                # Check for analysis errors
                error_msg = self._check_analysis_errors(zernike_analysis)
                if error_msg:
                    logger.warning(f"ZernikeStandardCoefficients error: {error_msg}")
                else:
                    # Extract metrics from text output
                    zernike_analysis.Results.GetTextFile(zernike_temp_path)
                    if os.path.exists(zernike_temp_path):
                        text = self._read_opticstudio_text_file(zernike_temp_path)
                        if text:
                            metrics = _parse_zernike_full_text(text)
                            rms_waves = metrics["rms_to_chief"]
                            pv_waves = metrics["pv_to_chief"]
                            strehl_ratio = metrics["strehl_ratio"]
                    else:
                        logger.warning("ZernikeStandardCoefficients: GetTextFile produced no output")

            except Exception as e:
                logger.warning(f"ZernikeStandardCoefficients failed: {e}")
            finally:
                self._cleanup_analysis(zernike_analysis, zernike_temp_path)
                zernike_analysis = None

            if rms_waves is None and pv_waves is None:
                return {"success": False, "error": "ZernikeStandardCoefficients returned no metrics"}

            # -----------------------------------------------------------------
            # Part B: WavefrontMap → 2D wavefront array
            # Uses GetDataGrid + _extract_data_grid (same pattern as spot/MTF)
            # -----------------------------------------------------------------
            image_b64 = None
            array_shape = None
            array_dtype = None

            try:
                wfm_analysis = self._zp.analyses.new_analysis(
                    self.oss, idm.WavefrontMap, settings_first=True,
                )
                settings = wfm_analysis.Settings
                self._configure_analysis_settings(settings, field_index, wavelength_index, sampling)

                # WavefrontMap-specific settings
                if hasattr(settings, 'Surface'):
                    try:
                        settings.Surface.SetSurfaceNumber(image_surf)
                    except Exception as e:
                        logger.warning(f"WavefrontMap: Could not set Surface: {e}")

                if hasattr(settings, 'RemoveTilt'):
                    settings.RemoveTilt = remove_tilt

                if hasattr(settings, 'UseExitPupil'):
                    settings.UseExitPupil = True

                wfm_start = time.perf_counter()
                try:
                    wfm_analysis.ApplyAndWaitForCompletion()
                finally:
                    elapsed = (time.perf_counter() - wfm_start) * 1000
                    log_timing(logger, "WavefrontMap.ApplyAndWaitForCompletion", elapsed)

                results = wfm_analysis.Results
                if results and hasattr(results, 'NumberOfDataGrids') and results.NumberOfDataGrids > 0:
                    grid = results.GetDataGrid(0)
                    arr = self._extract_data_grid(grid)
                    if arr is not None and arr.ndim >= 2:
                        image_b64 = base64.b64encode(arr.tobytes()).decode('utf-8')
                        array_shape = list(arr.shape)
                        array_dtype = str(arr.dtype)
                        logger.info(f"Wavefront map generated: shape={arr.shape}, dtype={arr.dtype}")

            except Exception as e:
                logger.warning(f"WavefrontMap failed: {e} (metrics still available)")
            finally:
                self._cleanup_analysis(wfm_analysis)
                wfm_analysis = None

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
                        # Verify the setting took effect by reading it back
                        readback = getattr(settings, attr, "?")
                        logger.info(f"RmsField: Set {label or attr} = {value}, readback = {readback}")
                    except Exception as e:
                        logger.warning(f"RmsField: Could not set {label or attr}: {e}")

                # Log all available settings attributes for diagnostics
                settings_attrs = [a for a in dir(settings) if not a.startswith('_')]
                logger.info(f"RmsField: Available settings: {settings_attrs}")

                def _get_enum_value(enum_cls: Any, name: str) -> Any:
                    """Get an enum value by name, returning None if not found."""
                    value = getattr(enum_cls, name, None)
                    if value is None:
                        available = [a for a in dir(enum_cls) if not a.startswith('_')]
                        logger.warning(f"RmsField: Enum value '{name}' not found on {enum_cls}. Available: {available}")
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
                else:
                    logger.warning(f"RmsField: FieldDensity not set — analysis will use default (likely 5 points)")

                # RayDensity
                rd_value = _get_enum_value(rms_consts.RayDensities, f"RayDens_{ray_density}")
                if rd_value is not None:
                    _set_setting('RayDensity', rd_value)
                else:
                    logger.warning(f"RmsField: RayDensity not set — analysis will use default")

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
                                        "rms_radius_um": y_val * 1000,  # mm → µm
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
                    tmp = tempfile.NamedTemporaryFile(suffix='.txt', delete=False)
                    tmp_path = tmp.name
                    tmp.close()
                    try:
                        results.GetTextFile(tmp_path)
                        with open(tmp_path, 'r') as f:
                            text_content = f.read()
                        if text_content:
                            logger.info(f"RmsField text output (first 500 chars): {text_content[:500]}")
                    except Exception as e:
                        logger.warning(f"RmsField: Text fallback also failed: {e}")
                    finally:
                        try:
                            os.unlink(tmp_path)
                        except OSError:
                            pass
                    return {"success": False, "error": "RmsField analysis returned no extractable data"}

                logger.info(f"RmsField: Returning {len(data_points)} data points, {len(diffraction_limit)} diffraction limit points (requested FieldDens_{snapped})")
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
                self._cleanup_analysis(analysis)

        except Exception as e:
            return {"success": False, "error": f"RmsField analysis failed: {e}"}

    def _run_zernike_vs_field_fallback(
        self,
        maximum_term: int,
        wavelength_index: int,
        field_density: int,
        sampling: str = "64x64",
    ) -> pd.DataFrame | None:
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
            self._configure_analysis_settings(settings, wavelength_index=wavelength_index, sampling=sampling)
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
            self._cleanup_analysis(analysis)

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

            # Run ZernikeCoefficientsVsField using raw ZOS-API (no ZosPy temp files)
            df = self._run_zernike_vs_field_fallback(
                maximum_term, wavelength_index, field_density, sampling,
            )
            if df is None:
                return {"success": False, "error": "ZernikeCoefficientsVsField returned no data"}

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
        analysis = None
        temp_path = os.path.join(tempfile.gettempdir(), "zospy_zernike_standard.txt")

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

            # Determine surface number
            if surface == "Image" or surface is None:
                surf_num = self.oss.LDE.NumberOfSurfaces - 1
            else:
                try:
                    surf_num = int(surface)
                except (ValueError, TypeError):
                    surf_num = self.oss.LDE.NumberOfSurfaces - 1

            # Run ZernikeStandardCoefficients via raw ZOS-API (no ZosPy temp files)
            idm = self._zp.constants.Analysis.AnalysisIDM
            analysis = self._zp.analyses.new_analysis(
                self.oss, idm.ZernikeStandardCoefficients, settings_first=True,
            )
            settings = analysis.Settings
            self._configure_analysis_settings(settings, field_index, wavelength_index, sampling)

            # Zernike-specific settings
            if hasattr(settings, 'MaximumNumberOfTerms'):
                settings.MaximumNumberOfTerms = maximum_term
            elif hasattr(settings, 'MaximumTerm'):
                settings.MaximumTerm = maximum_term

            if hasattr(settings, 'Surface'):
                try:
                    settings.Surface.SetSurfaceNumber(surf_num)
                except Exception as e:
                    logger.warning(f"ZernikeStandardCoefficients: Could not set Surface: {e}")

            # ZOS-API uses "OBD" not "OPD" (naming quirk)
            if hasattr(settings, 'ReferenceOBDToVertex'):
                settings.ReferenceOBDToVertex = False
            elif hasattr(settings, 'ReferenceOPDToVertex'):
                settings.ReferenceOPDToVertex = False

            zernike_start = time.perf_counter()
            try:
                analysis.ApplyAndWaitForCompletion()
            finally:
                elapsed = (time.perf_counter() - zernike_start) * 1000
                log_timing(logger, "ZernikeStandardCoefficients.run (full)", elapsed)

            # Check for analysis errors
            error_msg = self._check_analysis_errors(analysis)
            if error_msg:
                return {"success": False, "error": f"ZernikeStandardCoefficients error: {error_msg}"}

            # Extract text output and parse coefficients + metrics
            analysis.Results.GetTextFile(temp_path)
            if not os.path.exists(temp_path):
                return {"success": False, "error": "ZernikeStandardCoefficients: GetTextFile produced no output"}

            text = self._read_opticstudio_text_file(temp_path)
            if not text:
                return {"success": False, "error": "ZernikeStandardCoefficients text output is empty"}

            parsed = _parse_zernike_full_text(text)

            if not parsed["coefficients"]:
                return {"success": False, "error": "No Zernike coefficients extracted from text output"}

            result = {
                "success": True,
                "coefficients": parsed["coefficients"],
                "pv_to_chief": parsed["pv_to_chief"],
                "pv_to_centroid": parsed["pv_to_centroid"],
                "rms_to_chief": parsed["rms_to_chief"],
                "rms_to_centroid": parsed["rms_to_centroid"],
                "strehl_ratio": parsed["strehl_ratio"],
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
        finally:
            self._cleanup_analysis(analysis, temp_path)
