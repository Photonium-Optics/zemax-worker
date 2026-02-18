"""Aberrations mixin – Seidel, wavefront, Zernike, RMS vs field."""

import base64
import logging
import math
import os
import tempfile
import time
from typing import Any

import numpy as np

from config import SEIDEL_TEMP_FILENAME
from zospy_handler._base import _extract_value, _log_raw_output, _extract_dataframe, _parse_zernike_term_number
from utils.timing import log_timing

logger = logging.getLogger(__name__)


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
                    try:
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
                if analysis is not None:
                    try:
                        analysis.Close()
                    except Exception:
                        pass

        except Exception as e:
            return {"success": False, "error": f"RmsField analysis failed: {e}"}

    def _run_zernike_vs_field_fallback(
        self,
        maximum_term: int,
        wavelength_index: int,
        field_density: int,
    ) -> "pd.DataFrame | None":
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
