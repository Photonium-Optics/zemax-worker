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
from zospy_handler._base import _enum_name, _extract_value, _log_raw_output, _parse_zernike_term_number
from utils.timing import log_timing

logger = logging.getLogger(__name__)


def _parse_zernike_full_text(text: str) -> dict:
    """Parse ZernikeStandardCoefficients text output into coefficients and metrics."""
    result = {
        "coefficients": [],
        "rms_to_chief": None,
        "rms_to_centroid": None,
        "strehl_ratio": None,
    }

    lines = text.splitlines()
    logger.debug(f"_parse_zernike_full_text: parsing {len(lines)} lines")

    for line in lines:
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
                logger.debug(f"_parse_zernike_full_text: matched {key}={result[key]}")
            except ValueError:
                pass
            continue

        # Match Strehl Ratio — OpticStudio may output "Strehl Ratio (Est)" with tab separators
        m = re.match(
            r'Strehl\s+Ratio[^:]*:\s*([\d.eE+-]+)',
            line_stripped, re.IGNORECASE,
        )
        if m:
            try:
                result["strehl_ratio"] = float(m.group(1))
                logger.debug(f"_parse_zernike_full_text: matched strehl_ratio={result['strehl_ratio']}")
            except ValueError:
                pass

        # Log unmatched lines that look like they SHOULD be metrics (helps debug regex mismatches)
        elif 'strehl' in line_stripped.lower():
            logger.warning(f"_parse_zernike_full_text: UNMATCHED metric-like line: {line_stripped!r} (bytes: {line_stripped.encode('utf-8')!r})")

    logger.debug(
        f"_parse_zernike_full_text: result — "
        f"rms_chief={result['rms_to_chief']}, rms_centroid={result['rms_to_centroid']}, "
        f"strehl={result['strehl_ratio']}, coeffs={len(result['coefficients'])}"
    )
    return result


def _parse_zernike_vs_field_text(text: str) -> list[dict]:
    """Parse ZernikeCoefficientsVsField tab-delimited text into rows of {field, Z1, Z2, ...}."""
    rows: list[dict] = []
    lines = text.splitlines()
    term_numbers: list[int] = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        # Header line: "Field: \t 1\t 2\t ..."
        if stripped.lower().startswith("field:") or stripped.lower().startswith("field\t"):
            parts = stripped.split("\t")
            for p in parts[1:]:
                p = p.strip()
                if p:
                    try:
                        term_numbers.append(int(p))
                    except ValueError:
                        pass
            if term_numbers:
                logger.debug(f"_parse_zernike_vs_field_text: found {len(term_numbers)} terms in header")
            continue

        # Data rows: first column is field value, rest are Zernike coefficients
        if not term_numbers:
            continue

        parts = stripped.split("\t")
        if len(parts) < 2:
            continue

        try:
            field_val = float(parts[0].strip())
        except (ValueError, TypeError):
            continue

        row: dict = {"field": field_val}
        for i, p in enumerate(parts[1:]):
            p = p.strip()
            if not p or i >= len(term_numbers):
                continue
            try:
                row[f"Z{term_numbers[i]}"] = float(p)
            except (ValueError, TypeError):
                pass

        if len(row) > 1:  # Must have at least field + one coefficient
            rows.append(row)

    logger.debug(f"_parse_zernike_vs_field_text: parsed {len(rows)} data rows")
    return rows


_ANGULAR_FIELD_TYPES = {"Angle", "TheodoliteAngle"}


def _get_field_unit(fields) -> str:
    """Return 'deg' for angular field types, 'mm' for spatial."""
    field_type_name = _enum_name(fields.GetFieldType())
    return "deg" if field_type_name in _ANGULAR_FIELD_TYPES else "mm"


class AberrationsMixin:

    def get_seidel_native(self) -> dict[str, Any]:
        """Run SeidelCoefficients analysis and return raw text output for Mac-side parsing."""
        analysis = None
        temp_path = os.path.join(tempfile.gettempdir(), SEIDEL_TEMP_FILENAME)

        try:
            idm = self._zp.constants.Analysis.AnalysisIDM
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

            error_msg = self._check_analysis_errors(analysis)
            if error_msg:
                return {"success": False, "error": error_msg}

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
        """Get wavefront error map and metrics (RMS, Strehl, 2D array) via ZOS-API."""
        zernike_analysis = None
        wfm_analysis = None
        zernike_temp_path = os.path.join(tempfile.gettempdir(), "zospy_zernike_wavefront.txt")

        try:
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
            # Part A: ZernikeStandardCoefficients → RMS, Strehl
            # Uses GetTextFile + text parsing (same pattern as get_seidel_native)
            # -----------------------------------------------------------------
            rms_waves = None
            strehl_ratio = None

            try:
                zernike_analysis = self._zp.analyses.new_analysis(
                    self.oss, idm.ZernikeStandardCoefficients, settings_first=True,
                )
                settings = zernike_analysis.Settings
                self._configure_analysis_settings(settings, field_index, wavelength_index, sampling)

                # Zernike-specific settings (ZOSAPI: IAS_ZernikeStandardCoefficients.MaximumNumberOfTerms)
                try:
                    settings.MaximumNumberOfTerms = 37
                except Exception as e:
                    logger.warning(f"ZernikeStandardCoefficients: MaximumNumberOfTerms not settable ({type(e).__name__}: {e}). Continuing with default.")

                try:
                    settings.Surface.SetSurfaceNumber(image_surf)
                except Exception as e:
                    logger.warning(f"ZernikeStandardCoefficients: Could not set Surface: {e}")

                settings.ReferenceOBDToVertex = False

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
                    zernike_analysis.Results.GetTextFile(zernike_temp_path)
                    if os.path.exists(zernike_temp_path):
                        file_size = os.path.getsize(zernike_temp_path)
                        logger.debug(f"ZernikeStandardCoefficients: GetTextFile wrote {file_size} bytes to {zernike_temp_path}")
                        text = self._read_opticstudio_text_file(zernike_temp_path)
                        if text:
                            logger.debug(f"ZernikeStandardCoefficients: text file read OK, {len(text)} chars, {len(text.splitlines())} lines")
                            # Log ALL non-empty, non-coefficient lines so we can see
                            # the actual text format OpticStudio produces
                            for dbg_line in text.splitlines():
                                ls = dbg_line.strip()
                                if ls and not ls.startswith('Z '):
                                    logger.debug(f"ZernikeText line: {ls!r}")
                            metrics = _parse_zernike_full_text(text)
                            rms_waves = metrics["rms_to_chief"]
                            strehl_ratio = metrics["strehl_ratio"]
                            if strehl_ratio is None:
                                logger.warning(f"ZernikeStandardCoefficients: parsed metrics incomplete — rms={rms_waves}, strehl={strehl_ratio}")
                        else:
                            logger.warning(f"ZernikeStandardCoefficients: _read_opticstudio_text_file returned empty string (file was {file_size} bytes)")
                    else:
                        logger.warning("ZernikeStandardCoefficients: GetTextFile produced no output file")

            except Exception as e:
                logger.warning(f"ZernikeStandardCoefficients failed: {e}")
            finally:
                self._cleanup_analysis(zernike_analysis, zernike_temp_path)
                zernike_analysis = None

            if rms_waves is None:
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
                try:
                    settings.Surface.SetSurfaceNumber(image_surf)
                except Exception as e:
                    logger.warning(f"WavefrontMap: Could not set Surface: {e}")

                settings.RemoveTilt = remove_tilt
                settings.UseExitPupil = True

                wfm_start = time.perf_counter()
                try:
                    wfm_analysis.ApplyAndWaitForCompletion()
                finally:
                    elapsed = (time.perf_counter() - wfm_start) * 1000
                    log_timing(logger, "WavefrontMap.ApplyAndWaitForCompletion", elapsed)

                results = wfm_analysis.Results
                if results and results.NumberOfDataGrids > 0:
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

            if strehl_ratio is None:
                logger.warning("ZernikeStandardCoefficients: Strehl extraction failed — text parser did not match Strehl line from GetTextFile output")

            result = {
                "success": True,
                "rms_waves": rms_waves,
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
        wavelength_index: int | None = None,
    ) -> dict[str, Any]:
        """
        Get RMS spot radius vs field using native RmsField analysis.

        Uses OpticStudio's AnalysisIDM.RmsField which auto-samples across
        the full field range, producing a smooth curve.

        Args:
            ray_density: Ray density (1-20, maps to RayDens_N enum)
            num_field_points: Number of field sample points (snapped to nearest FieldDensity enum: 5,10,...,100)
            reference: Reference point ('centroid' or 'chief_ray')
            wavelength_index: Wavelength index (1-indexed). None = use OpticStudio primary wavelength.

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
            num_wavelengths = int(wavelengths.NumberOfWavelengths)

            # IWavelength.IsPrimary is a reliable bool property per ZOS-API docs.
            resolved_wavelength_index = wavelength_index
            if resolved_wavelength_index is None:
                for wi in range(1, num_wavelengths + 1):
                    wl = wavelengths.GetWavelength(wi)
                    if wl is not None and wl.IsPrimary:
                        resolved_wavelength_index = wi
                        break

                if resolved_wavelength_index is None:
                    resolved_wavelength_index = 1
                    logger.warning("RmsField: No primary wavelength found; defaulting to wavelength #1")

            if resolved_wavelength_index > num_wavelengths:
                return {
                    "success": False,
                    "error": f"Wavelength index {resolved_wavelength_index} out of range (max: {num_wavelengths})",
                }

            wavelength_um = _extract_value(
                wavelengths.GetWavelength(resolved_wavelength_index).Wavelength, 0.5876,
            )

            fields = self.oss.SystemData.Fields
            field_unit = _get_field_unit(fields)

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
                    if not hasattr(settings, attr):
                        return
                    try:
                        setattr(settings, attr, value)
                    except Exception as e:
                        logger.warning(f"RmsField: Could not set {label or attr}: {e}")

                _set_setting('Data', rms_consts.RMSField.DataType.SpotRadius, 'Data type')

                field_dens_name = f"FieldDens_{snapped}"
                field_dens_val = getattr(rms_consts.FieldDensities, field_dens_name, None)
                if field_dens_val is None:
                    raise ValueError(f"Unknown FieldDensity enum: {field_dens_name}")
                _set_setting('FieldDensity', field_dens_val)

                ray_dens_name = f"RayDens_{ray_density}"
                ray_dens_val = getattr(rms_consts.RayDensities, ray_dens_name, None)
                if ray_dens_val is None:
                    raise ValueError(f"Unknown RayDensity enum: {ray_dens_name}")
                _set_setting('RayDensity', ray_dens_val)

                refer_name = "ChiefRay" if reference == "chief_ray" else "Centroid"
                refer_val = getattr(rms_consts.ReferTo, refer_name, None)
                if refer_val is None:
                    raise ValueError(f"Unknown ReferTo enum: {refer_name}")
                _set_setting('ReferTo', refer_val)

                # Wavelength
                self._configure_analysis_settings(
                    settings, wavelength_index=resolved_wavelength_index,
                )

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

                            desc = str(series.Description)
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
                            num_curves = series.NumSeries

                            # Get labels to identify diffraction limit curve
                            labels = []
                            if series.SeriesLabels is not None:
                                labels = list(series.SeriesLabels)

                            logger.info(f"RmsField series {si}: {num_points} points, {num_curves} curves, labels={labels}")

                            for ci in range(num_curves):
                                raw_label = labels[ci] if ci < len(labels) else None
                                label = str(raw_label).lower() if raw_label is not None else ""
                                # Diffraction limit curve is identified by label text OR by
                                # being the last curve with a None/empty label when
                                # ShowDiffractionLimit is True (ZOS-API returns null label
                                # for the diffraction limit series).
                                is_diffraction = (
                                    "diffrac" in label
                                    or "limit" in label
                                    or (raw_label is None and ci == num_curves - 1)
                                )

                                curve_points = []
                                for pi in range(num_points):
                                    x_val = float(x_values[pi])
                                    # IMatrixData.Data is always 2D: [row, col]
                                    y_val = float(y_raw[pi, ci])
                                    curve_points.append({
                                        "field_value": x_val,
                                        "rms_radius_um": y_val,  # API already returns µm
                                    })

                                logger.info(f"RmsField series {si} curve {ci}: label='{label}', {len(curve_points)} points, is_diffraction={is_diffraction}")

                                if is_diffraction:
                                    diffraction_limit = curve_points
                                else:
                                    data_points.extend(curve_points)

                    except Exception as e:
                        logger.warning(f"RmsField: Could not extract data series: {e}", exc_info=True)

                if not data_points:
                    return {"success": False, "error": "RmsField analysis returned no extractable data from DataSeries. Check that the analysis ran successfully and the system has valid fields/wavelengths."}

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

    def _run_zernike_vs_field(
        self,
        maximum_term: int,
        wavelength_index: int,
        field_density: int,
        sampling: str = "64x64",
    ) -> pd.DataFrame | None:
        """
        Run ZernikeCoefficientsVsField via the ZOS-API.

        ZernikeCoefficientsVsField returns text-only output (no data series
        or data grids). This method exports the text via GetTextFile and
        parses the tab-delimited table.

        Returns a DataFrame with field positions as the index and
        Zernike term columns, or None if no data could be extracted.
        """
        import pandas as pd

        idm = self._zp.constants.Analysis.AnalysisIDM
        analysis = self._zp.analyses.new_analysis(
            self.oss, idm.ZernikeCoefficientsVsField, settings_first=True,
        )
        temp_path = os.path.join(tempfile.gettempdir(), "zospy_zernike_vs_field.txt")
        try:
            settings = analysis.Settings
            # IAS_ZernikeCoefficientsVsField uses Coefficients (comma-separated string),
            # not MaximumNumberOfTerms (which only exists on IAS_ZernikeStandardCoefficients)
            settings.Coefficients = ",".join(str(i) for i in range(1, maximum_term + 1))
            self._configure_analysis_settings(settings, wavelength_index=wavelength_index, sampling=sampling)
            settings.FieldDensity = field_density

            zernike_start = time.perf_counter()
            try:
                analysis.ApplyAndWaitForCompletion()
            finally:
                elapsed_ms = (time.perf_counter() - zernike_start) * 1000
                log_timing(logger, "ZernikeCoefficientsVsField.run", elapsed_ms)

            # ZernikeCoefficientsVsField produces text-only output (NumberOfDataSeries
            # and NumberOfDataGrids are both 0). Parse the tab-delimited text output.
            analysis.Results.GetTextFile(temp_path)
            if not os.path.exists(temp_path):
                logger.warning("ZernikeCoefficientsVsField: GetTextFile produced no output")
                return None

            text = self._read_opticstudio_text_file(temp_path)
            if not text:
                logger.warning("ZernikeCoefficientsVsField: text output is empty")
                return None

            rows = _parse_zernike_vs_field_text(text)
            if not rows:
                logger.warning("ZernikeCoefficientsVsField: no data rows parsed from text")
                return None

            return pd.DataFrame(rows).set_index("field")
        finally:
            self._cleanup_analysis(analysis, temp_path)

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
            wavelengths = self.oss.SystemData.Wavelengths
            if wavelength_index > wavelengths.NumberOfWavelengths:
                return {"success": False, "error": f"Wavelength index {wavelength_index} out of range (max: {wavelengths.NumberOfWavelengths})"}
            wavelength_um = _extract_value(wavelengths.GetWavelength(wavelength_index).Wavelength, 0.5876)

            fields = self.oss.SystemData.Fields
            field_unit = _get_field_unit(fields)

            df = self._run_zernike_vs_field(
                maximum_term, wavelength_index, field_density, sampling,
            )
            if df is None:
                return {"success": False, "error": "ZernikeCoefficientsVsField returned no data"}

            logger.info(f"ZernikeVsField: DataFrame columns={list(df.columns)}, shape={df.shape}")

            if len(df) == 0:
                return {"success": False, "error": f"No Zernike vs field data extracted (cols={list(df.columns)[:10]})"}

            try:
                field_positions = [float(v) for v in df.index.tolist()]
            except (ValueError, TypeError):
                field_positions = list(range(len(df)))

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

            # Zernike-specific settings (ZOSAPI: IAS_ZernikeStandardCoefficients.MaximumNumberOfTerms)
            try:
                settings.MaximumNumberOfTerms = maximum_term
            except Exception as e:
                logger.error(f"ZernikeStandardCoefficients: Failed to set MaximumNumberOfTerms={maximum_term}: {type(e).__name__}: {e}")
                return {"success": False, "error": f"Cannot set Zernike MaximumNumberOfTerms={maximum_term}: {e}"}

            try:
                settings.Surface.SetSurfaceNumber(surf_num)
            except Exception as e:
                logger.warning(f"ZernikeStandardCoefficients: Could not set Surface: {e}")

            settings.ReferenceOBDToVertex = False

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
            logger.error(f"get_zernike_standard_coefficients failed: {e}", exc_info=True)
            return {"success": False, "error": str(e)}
        finally:
            self._cleanup_analysis(analysis, temp_path)
