"""Performance mixin – MTF, PSF, ray fan, standard spot metrics."""

import base64
import logging
import math
import re
import time
from typing import Any, Optional

import numpy as np

from zospy_handler._base import _extract_value, _log_raw_output, GridWithMetadata, _compute_field_normalization, _normalize_field
from zospy_handler.pupil import generate_random_coords
from utils.timing import log_timing

logger = logging.getLogger(__name__)

# He-d line — standard optical reference wavelength fallback
DEFAULT_WAVELENGTH_UM = 0.5876

_STREHL_RE = re.compile(
    r"strehl\s*(?:ratio)?\s*[:=]\s*([\d.]+(?:e[+-]?\d+)?)",
    re.IGNORECASE,
)


def _cutoff_frequency(wavelength_um: float, fno: Optional[float]) -> Optional[float]:
    """Calculate diffraction cutoff frequency: fc = 1 / (wavelength_mm * f/#)."""
    if not fno or fno <= 0:
        return None
    wavelength_mm = wavelength_um / 1000.0
    return 1.0 / (wavelength_mm * fno)


def _parse_strehl_from_header(header_lines) -> Optional[float]:
    """Parse Strehl ratio from analysis header lines using regex.

    IAR_HeaderData.Lines returns String[] (always iterable).
    Returns None if no Strehl value is found.

    Handles formats like:
      "Strehl Ratio : 0.8532"
      "Strehl Ratio : 0.8532 (0.550 um)"
      "Strehl = 8.532e-01"
    """
    if not header_lines:
        return None
    # IAR_HeaderData.Lines is String[] — materialize .NET IList once
    lines_iter = list(header_lines)
    for line in lines_iter:
        m = _STREHL_RE.search(str(line))
        if m:
            try:
                return float(m.group(1))
            except ValueError as e:
                logger.warning(f"Failed to parse Strehl value from '{line}': {e}")
    return None


def _grid_meta_to_psf_fields(grid_meta: GridWithMetadata) -> dict[str, Any]:
    """Convert a GridWithMetadata into the common PSF result dict fields.

    Returns a dict with keys: image (base64), image_format, array_shape,
    array_dtype, psf_peak, delta_x, delta_y, extent_x, extent_y.
    """
    arr = grid_meta.data
    return {
        "image": base64.b64encode(arr.tobytes()).decode('utf-8'),
        "image_format": "numpy_array",
        "array_shape": list(arr.shape),
        "array_dtype": str(arr.dtype),
        "psf_peak": float(np.max(arr)) if arr.size > 0 else None,
        "delta_x": grid_meta.dx,
        "delta_y": grid_meta.dy,
        "extent_x": grid_meta.extent_x,
        "extent_y": grid_meta.extent_y,
    }


def _extract_fan_series(ds):
    """Extract pupil (x) and 2D aberration (y) arrays from a fan data series.

    Works for both Ray Fan and OPD Fan analyses. Uses explicit indexed access
    on .NET arrays to avoid pythonnet interop issues with tuple() on 2D
    double[,] arrays.
    """
    x_raw = ds.XData.Data
    y_raw = ds.YData.Data
    nrows = y_raw.GetLength(0)
    ncols = y_raw.GetLength(1)
    x = np.array([float(x_raw[i]) for i in range(nrows)])
    y_2d = np.array([[float(y_raw[r, c]) for c in range(ncols)] for r in range(nrows)])
    return x, y_2d


def _sanitize_fan_values(values: list[float]) -> list[float]:
    """Replace NaN values with 0.0 in a fan data array."""
    return [0.0 if math.isnan(v) else v for v in values]


class PerformanceMixin:

    @staticmethod
    def _extract_mtf_series(series) -> tuple[list, list, list]:
        """Extract x, tangential, and sagittal data from an MTF data series.

        IAR_DataSeries has XData (IVectorData, 1D) and YData (IMatrixData, 2D).
        IMatrixData layout: GetLength(0)=Rows=freq points, GetLength(1)=Cols=NumSeries.
        For standard T/S MTF, NumSeries=2 (col 0=tangential, col 1=sagittal).

        Returns (x_values, tangential_values, sagittal_values).
        sagittal_values is empty if the series is single-column.
        """
        x_raw = series.XData.Data    # IVectorData.Data -> double[]
        y_raw = series.YData.Data    # IMatrixData.Data -> double[,]
        n_pts = y_raw.GetLength(0)   # Rows = frequency points
        n_cols = y_raw.GetLength(1)  # Cols = NumSeries (tang/sag)

        x_vals = [float(x_raw[i]) for i in range(n_pts)]
        tang = [float(y_raw[i, 0]) for i in range(n_pts)]
        sag = [float(y_raw[i, 1]) for i in range(n_pts)] if n_cols >= 2 else []
        logger.debug(f"MTF extraction: {n_pts} pts x {n_cols} cols")
        return x_vals, tang, sag

    @staticmethod
    def _classify_mtf_series(desc_lower: str) -> Optional[str]:
        """Classify an MTF series description as 'tangential', 'sagittal', or None."""
        if (desc_lower.startswith(("ts ", "ts,", "ts."))
                or "tangential" in desc_lower
                or desc_lower.startswith("t ")):
            return "tangential"
        if (desc_lower.startswith(("ss ", "ss,", "ss."))
                or "sagittal" in desc_lower
                or desc_lower.startswith("s ")):
            return "sagittal"
        return None

    def _extract_tf_mtf_field_data(
        self,
        results,
        fi: int,
    ) -> tuple[list[float], list[float], list[float]]:
        """Extract tangential, sagittal, and focus data from Through Focus MTF results."""
        tangential: list[float] = []
        sagittal: list[float] = []
        focus_data: list[float] = []

        try:
            num_series = results.NumberOfDataSeries
            logger.debug(f"TF-MTF field {fi}: {num_series} data series")
            unclassified: list[tuple[list[float], list[float]]] = []
            for si in range(num_series):
                series = results.GetDataSeries(si)
                if series is None:
                    continue

                desc_lower = str(series.Description).lower()
                series_x, series_y, series_sag = self._extract_mtf_series(series)

                # Multi-column: _extract_mtf_series returns both tang and sag
                if series_sag:
                    if not tangential:
                        tangential = series_y
                    if not sagittal:
                        sagittal = series_sag
                    if not focus_data:
                        focus_data = series_x
                    continue

                # Single-column: classify by description
                category = self._classify_mtf_series(desc_lower)
                if category == "tangential" and not tangential:
                    tangential = series_y
                    if not focus_data:
                        focus_data = series_x
                elif category == "sagittal" and not sagittal:
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

        return tangential, sagittal, focus_data

    def _extract_mtf_field_data(
        self,
        results,
        fi: int,
        label: str,
        diffraction_limit_from_api: list[float],
    ) -> tuple[list[float], list[float], list[float], list[float]]:
        """Extract tangential, sagittal, frequency, and diffraction limit data from MTF results.

        Iterates over all data series in the analysis results, classifying each as
        diffraction limit, tangential, sagittal, or unclassified. Unclassified series
        are assigned to missing slots as a positional fallback.

        Args:
            results: The analysis Results object from ZOS-API.
            fi: 1-indexed field number (for log messages).
            label: Log prefix, e.g. "MTF" or "Huygens MTF".
            diffraction_limit_from_api: Mutable list; populated in-place if a
                diffraction limit series is found and the list is still empty.

        Returns:
            (tangential, sagittal, freq_data, diffraction_limit_from_api)
        """
        tangential: list[float] = []
        sagittal: list[float] = []
        freq_data: list[float] = []

        if results is None:
            return tangential, sagittal, freq_data, diffraction_limit_from_api

        try:
            num_series = results.NumberOfDataSeries
            logger.info(f"{label} field {fi}: {num_series} data series")
            unclassified = []
            for si in range(num_series):
                series = results.GetDataSeries(si)
                if series is None:
                    continue

                desc = str(series.Description)
                desc_lower = desc.lower()

                if "diffrac" in desc_lower or "limit" in desc_lower:
                    dl_x, dl_y, _ = self._extract_mtf_series(series)
                    if dl_y and not diffraction_limit_from_api:
                        diffraction_limit_from_api = dl_y
                        if not freq_data:
                            freq_data = dl_x
                        logger.debug(f"{label} field {fi}: extracted diffraction limit from API ({len(dl_y)} pts)")
                    continue

                series_x, series_tang, series_sag = self._extract_mtf_series(series)

                # 2D bulk extraction yields both tang and sag from one series
                if series_sag:
                    if not tangential:
                        tangential = series_tang
                    if not sagittal:
                        sagittal = series_sag
                    if not freq_data:
                        freq_data = series_x
                    continue

                # Single-column: classify by description
                category = self._classify_mtf_series(desc_lower)
                if category == "tangential" and not tangential:
                    tangential = series_tang
                elif category == "sagittal" and not sagittal:
                    sagittal = series_tang
                else:
                    unclassified.append((series_x, series_tang))
                    continue
                if not freq_data:
                    freq_data = series_x

            # Fallback: assign unclassified series to missing slots
            if unclassified and (not tangential or not sagittal):
                logger.info(f"{label} field {fi}: using positional fallback for {len(unclassified)} unclassified series")
                idx = 0
                if not tangential and idx < len(unclassified):
                    x, tangential = unclassified[idx]
                    if not freq_data:
                        freq_data = x
                    idx += 1
                if not sagittal and idx < len(unclassified):
                    _, sagittal = unclassified[idx]
        except Exception as e:
            logger.warning(f"{label}: Could not extract data series for field {fi}: {e}")

        return tangential, sagittal, freq_data, diffraction_limit_from_api

    @staticmethod
    def _validate_mtf_results(
        label: str,
        frequency: Optional[list[float]],
        all_fields_data: list[dict],
    ) -> Optional[dict[str, Any]]:
        """Validate MTF extraction results, returning an error dict if validation fails.

        Returns None if validation passes, or a {"success": False, "error": ...} dict.
        """
        if not all_fields_data:
            return {
                "success": False,
                "error": f"{label}: no field data collected. Field iteration may have failed.",
            }

        if not frequency:
            return {
                "success": False,
                "error": f"{label}: no frequency data extracted from analysis results. "
                         "ZOS-API data series may use an unexpected format.",
            }

        has_any_data = any(f["tangential"] or f["sagittal"] for f in all_fields_data)
        if not has_any_data:
            return {
                "success": False,
                "error": f"{label}: analysis ran but returned no tangential/sagittal data "
                         "for any field. Data extraction from ZOS-API series failed.",
            }

        return None

    def get_standard_spot_metrics(
        self,
        ray_density: int = 20,
        reference: str = "centroid",
        field_index: Optional[int] = None,
        wavelength_index: Optional[int] = None,
    ) -> dict[str, Any]:
        """
        Run StandardSpot analysis for official ZOS-API RMS/GEO metrics only.

        Runs the StandardSpot analysis to extract official RMS radius,
        GEO radius, and centroid per field. No batch ray trace.

        Args:
            ray_density: Controls StandardSpot ray density (default 20).
            reference: Reference point: 'chief_ray' or 'centroid'
            field_index: Field index (1-indexed). None = all fields.
            wavelength_index: Wavelength index (1-indexed). None = all wavelengths.

        Returns:
            On success: {"success": True, "spot_data": [...]}
            On error: {"success": False, "error": "..."}
        """
        analysis = None

        fields = self.oss.SystemData.Fields
        num_fields = fields.NumberOfFields

        if num_fields == 0:
            return {"success": False, "error": "System has no fields defined"}

        try:
            reference_value = self._resolve_spot_reference(reference)
        except ValueError as e:
            return {"success": False, "error": str(e)}

        try:
            logger.info(f"[STDSPOT] Starting: ray_density={ray_density}, reference={reference}, field_index={field_index}, wavelength_index={wavelength_index}")

            analysis = self._zp.analyses.new_analysis(
                self.oss,
                self._zp.constants.Analysis.AnalysisIDM.StandardSpot,
                settings_first=True,
            )

            self._configure_spot_analysis(analysis.Settings, ray_density, reference_value, field_index, wavelength_index)
            analysis.ApplyAndWaitForCompletion()

            spot_data: list[dict[str, Any]] = []
            if analysis.Results is not None:
                spot_data = self._extract_spot_data_from_results(
                    analysis.Results, fields, num_fields,
                    field_index=field_index, wavelength_index=wavelength_index,
                )
                logger.info(f"[STDSPOT] Extracted {len(spot_data)} field entries")
            else:
                logger.warning("[STDSPOT] analysis.Results is None")

            return {"success": True, "spot_data": spot_data}

        except Exception as e:
            logger.error(f"[STDSPOT] StandardSpot metrics FAILED: {type(e).__name__}: {e}", exc_info=True)
            return {"success": False, "error": f"StandardSpot metrics failed: {e}"}
        finally:
            self._cleanup_analysis(analysis, None)

    def _resolve_spot_reference(self, reference: str) -> Any:
        """Resolve a reference string ('chief_ray' or 'centroid') to the ZOS-API enum value.

        Raises:
            ValueError: If the reference string cannot be resolved.
        """
        refer_name = "ChiefRay" if reference == "chief_ray" else "Centroid"
        try:
            return getattr(
                self._zp.constants.Analysis.Settings.Spot.Reference, refer_name
            )
        except Exception as e:
            raise ValueError(f"Could not resolve Spot.Reference.{refer_name}: {e}") from e

    def _configure_spot_analysis(
        self,
        settings: Any,
        ray_density: int,
        reference_value: Any,
        field_index: Optional[int] = None,
        wavelength_index: Optional[int] = None,
    ) -> None:
        """
        Configure StandardSpot analysis settings.

        Args:
            settings: OpticStudio analysis settings object
            ray_density: Rays per axis (controls ray count: (ray_density+1)^2 per field/wl)
            reference_value: Spot.Reference enum value (ChiefRay or Centroid)
            field_index: Field index (1-indexed). None = all fields.
            wavelength_index: Wavelength index (1-indexed). None = all wavelengths.
        """
        settings.RayDensity = ray_density
        settings.ReferTo = reference_value

        # 0 = all fields/wavelengths in ZOS-API
        settings.Field.SetFieldNumber(field_index or 0)
        settings.Wavelength.SetWavelengthNumber(wavelength_index or 0)

    def _extract_airy_radius(self, wavelength_index: int = 0) -> Optional[float]:
        """Compute Airy disk radius: r_airy = 1.22 * wavelength * f_number.

        Result is returned in micrometers (µm).

        Args:
            wavelength_index: 1-based wavelength index. 0 = use primary wavelength.

        Returns:
            Airy radius in µm, or None if F/# or wavelength unavailable
        """
        fno = self._get_fno()
        if not fno or fno <= 0:
            logger.warning(f"[SPOT] Cannot compute airy radius: fno={fno}")
            return None

        # Try specific wavelength first, then fall back to primary
        wl_um = None
        if wavelength_index is not None and wavelength_index > 0:
            wl_obj = self.oss.SystemData.Wavelengths.GetWavelength(wavelength_index)
            val = _extract_value(wl_obj.Wavelength, 0.0) if wl_obj else 0.0
            if val > 0:
                wl_um = val

        if wl_um is None:
            wl_um = self._get_primary_wavelength_um()

        if wl_um is None:
            return None

        airy_radius_um = 1.22 * wl_um * fno
        logger.info(f"[SPOT] Computed airy_radius: 1.22 * {wl_um:.4f}µm * F/{fno:.2f} = {airy_radius_um:.3f} µm")
        return airy_radius_um

    def _get_primary_wavelength_um(self) -> Optional[float]:
        """Get the primary wavelength in µm from the system data.

        IWavelength.IsPrimary is a reliable bool property per ZOS-API docs.
        IWavelength.Wavelength is a double in µm.
        """
        wavelengths = self.oss.SystemData.Wavelengths
        for wi in range(1, wavelengths.NumberOfWavelengths + 1):
            wl = wavelengths.GetWavelength(wi)
            if wl is None:
                continue
            if wl.IsPrimary:
                val = _extract_value(wl.Wavelength, DEFAULT_WAVELENGTH_UM)
                if val > 0:
                    return val
                logger.warning(f"[SPOT] Primary wavelength at index {wi} has unusable value ({val!r})")
                return None
        logger.warning("[SPOT] No primary wavelength found in system")
        return None

    def _extract_spot_data_from_results(
        self,
        results: Any,
        fields: Any,
        num_fields: int,
        field_index: Optional[int] = None,
        wavelength_index: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        """
        Extract per-field spot data from analysis results.

        Args:
            results: OpticStudio analysis results object
            fields: OpticStudio fields object
            num_fields: Number of fields in the system
            field_index: If set (1-indexed), only extract data for this field.
                         When a single field is analyzed, ZOSAPI returns 0 for
                         non-analyzed fields, so we must limit iteration.
            wavelength_index: If set (1-indexed), query metrics for this wavelength.
                              None = primary wavelength (1).

        Returns:
            List of spot data dicts per field
        """
        spot_data: list[dict[str, Any]] = []

        # Determine which fields to iterate
        if field_index is not None:
            field_indices_0based = [field_index - 1]  # Convert 1-based to 0-based
        else:
            field_indices_0based = list(range(num_fields))

        try:
            for fi in field_indices_0based:
                field = fields.GetField(fi + 1)
                field_data = {
                    "field_index": fi,
                    "field_x": _extract_value(field.X),
                    "field_y": _extract_value(field.Y),
                    "rms_radius": None,
                    "geo_radius": None,
                    "centroid_x": None,
                    "centroid_y": None,
                    "num_rays": None,
                }
                self._populate_spot_data_from_results(results, fi, field_data, wavelength_index=wavelength_index)
                spot_data.append(field_data)

        except Exception as e:
            logger.warning(f"Could not extract spot data from results: {e}")

        return spot_data

    def _populate_spot_data_from_results(
        self,
        results: Any,
        field_index: int,
        field_data: dict[str, Any],
        wavelength_index: Optional[int] = None,
    ) -> None:
        """Populate spot data dict from SpotData result matrix (1-based indices)."""
        try:
            spot_data = results.SpotData
            if spot_data is None:
                if field_index == 0:
                    logger.warning("[SPOT] results.SpotData is None")
                return

            fi_1 = field_index + 1
            wi = wavelength_index if wavelength_index else 1

            if field_index == 0:
                logger.debug(f"[SPOT] SpotData matrix: NumberOfFields={spot_data.NumberOfFields}, NumberOfWavelengths={spot_data.NumberOfWavelengths}")

            # SpotData returns RMS/GEO in µm; centroid in mm (convert to µm)
            field_data["rms_radius"] = _extract_value(spot_data.GetRMSSpotSizeFor(fi_1, wi))
            field_data["geo_radius"] = _extract_value(spot_data.GetGeoSpotSizeFor(fi_1, wi))
            field_data["centroid_x"] = _extract_value(spot_data.GetReferenceCoordinate_X_For(fi_1, wi)) * 1000
            field_data["centroid_y"] = _extract_value(spot_data.GetReferenceCoordinate_Y_For(fi_1, wi)) * 1000

            logger.info(
                f"[SPOT] field[{field_index}]: rms={field_data.get('rms_radius')} µm, "
                f"geo={field_data.get('geo_radius')} µm, "
                f"centroid=({field_data.get('centroid_x')}, {field_data.get('centroid_y')}) µm"
            )

        except Exception as e:
            logger.warning(f"[SPOT] Could not get spot data for field {field_index}: {type(e).__name__}: {e}", exc_info=True)

    def _run_mtf_analysis(
        self,
        analysis_idm,
        label: str,
        endpoint: str,
        field_index: int,
        wavelength_index: int,
        sampling: str,
        maximum_frequency: float,
    ) -> dict[str, Any]:
        """Run an MTF analysis (FFT or Huygens) and return frequency/modulation data."""
        try:
            fields = self.oss.SystemData.Fields
            num_fields = fields.NumberOfFields
            wavelengths = self.oss.SystemData.Wavelengths

            if wavelength_index > wavelengths.NumberOfWavelengths:
                return {"success": False, "error": f"Wavelength index {wavelength_index} out of range (max: {wavelengths.NumberOfWavelengths})"}

            wavelength_um = _extract_value(wavelengths.GetWavelength(wavelength_index).Wavelength, DEFAULT_WAVELENGTH_UM)

            if field_index == 0:
                field_indices = list(range(1, num_fields + 1))
            else:
                if field_index > num_fields:
                    return {"success": False, "error": f"Field index {field_index} out of range (max: {num_fields})"}
                field_indices = [field_index]

            fno = self._get_fno()
            if fno is None or fno <= 0:
                logger.warning(f"{label}: Could not determine F/# (got {fno}), diffraction limit and cutoff will be omitted")

            cutoff_frequency = _cutoff_frequency(wavelength_um, fno)

            all_fields_data = []
            frequency = None
            diffraction_limit_from_api: list[float] = []

            for fi in field_indices:
                field = fields.GetField(fi)
                field_x = _extract_value(field.X)
                field_y = _extract_value(field.Y)

                analysis = None
                try:
                    analysis = self._zp.analyses.new_analysis(
                        self.oss, analysis_idm, settings_first=True,
                    )

                    settings = analysis.Settings
                    self._configure_analysis_settings(
                        settings,
                        field_index=fi,
                        wavelength_index=wavelength_index,
                        sampling=sampling,
                    )
                    if maximum_frequency > 0:
                        settings.MaximumFrequency = maximum_frequency
                    # IAS_FftMtf always has ShowDiffractionLimit; IAS_HuygensMtf may not
                    if hasattr(settings, 'ShowDiffractionLimit'):
                        settings.ShowDiffractionLimit = True

                    mtf_start = time.perf_counter()
                    try:
                        analysis.ApplyAndWaitForCompletion()
                    finally:
                        mtf_elapsed_ms = (time.perf_counter() - mtf_start) * 1000
                        log_timing(logger, f"{label}.run (field={fi})", mtf_elapsed_ms)

                    tangential, sagittal, freq_data, diffraction_limit_from_api = \
                        self._extract_mtf_field_data(
                            analysis.Results, fi, label, diffraction_limit_from_api,
                        )

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
                    logger.warning(f"{label} analysis failed for field {fi}: {e}")
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

            validation_error = self._validate_mtf_results(label, frequency, all_fields_data)
            if validation_error:
                return validation_error

            result = {
                "success": True,
                "frequency": np.array(frequency).tolist(),
                "fields": all_fields_data,
                "diffraction_limit": diffraction_limit_from_api,
                "cutoff_frequency": float(cutoff_frequency) if cutoff_frequency else None,
                "wavelength_um": float(wavelength_um),
            }
            _log_raw_output(endpoint, result)
            return result

        except Exception as e:
            return {"success": False, "error": f"{label} analysis failed: {e}"}

    def get_mtf(
        self,
        field_index: int = 0,
        wavelength_index: int = 1,
        sampling: str = "64x64",
        maximum_frequency: float = 0.0,
    ) -> dict[str, Any]:
        """Get FFT MTF data. Returns raw frequency/modulation data per field."""
        idm = self._zp.constants.Analysis.AnalysisIDM
        return self._run_mtf_analysis(
            analysis_idm=idm.FftMtf,
            label="FFT MTF",
            endpoint="/mtf",
            field_index=field_index,
            wavelength_index=wavelength_index,
            sampling=sampling,
            maximum_frequency=maximum_frequency,
        )

    def get_huygens_mtf(
        self,
        field_index: int = 0,
        wavelength_index: int = 1,
        sampling: str = "64x64",
        maximum_frequency: float = 0.0,
    ) -> dict[str, Any]:
        """Get Huygens MTF data. More accurate than FFT for highly aberrated systems."""
        idm = self._zp.constants.Analysis.AnalysisIDM
        return self._run_mtf_analysis(
            analysis_idm=idm.HuygensMtf,
            label="Huygens MTF",
            endpoint="/huygens-mtf",
            field_index=field_index,
            wavelength_index=wavelength_index,
            sampling=sampling,
            maximum_frequency=maximum_frequency,
        )

    def get_through_focus_mtf(
        self,
        sampling: str = "64x64",
        delta_focus: float = 0.1,
        frequency: float = 0.0,
        number_of_steps: int = 5,
        field_index: int = 0,
        wavelength_index: int = 1,
    ) -> dict[str, Any]:
        """Get Through Focus MTF data showing how MTF varies across defocus positions."""
        try:
            fields = self.oss.SystemData.Fields
            num_fields = fields.NumberOfFields
            wavelengths = self.oss.SystemData.Wavelengths

            if wavelength_index > wavelengths.NumberOfWavelengths:
                return {"success": False, "error": f"Wavelength index {wavelength_index} out of range (max: {wavelengths.NumberOfWavelengths})"}

            wavelength_um = _extract_value(wavelengths.GetWavelength(wavelength_index).Wavelength, DEFAULT_WAVELENGTH_UM)

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
                    settings.DeltaFocus = delta_focus
                    if frequency > 0:
                        settings.Frequency = frequency
                    settings.NumberOfSteps = number_of_steps

                    mtf_start = time.perf_counter()
                    try:
                        analysis.ApplyAndWaitForCompletion()
                    finally:
                        mtf_elapsed_ms = (time.perf_counter() - mtf_start) * 1000
                        log_timing(logger, f"FFTThroughFocusMtf.run (field={fi})", mtf_elapsed_ms)

                    # Extract data from results
                    results = analysis.Results
                    if results is not None:
                        tangential, sagittal, focus_data = \
                            self._extract_tf_mtf_field_data(results, fi)
                    else:
                        tangential, sagittal, focus_data = [], [], []

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

    def get_psf(
        self,
        field_index: int = 1,
        wavelength_index: int = 1,
        sampling: str = "64x64",
    ) -> dict[str, Any]:
        """
        Get FFT PSF data. Returns raw 2D intensity grid as base64 numpy array.
        """
        try:
            fields = self.oss.SystemData.Fields
            if field_index > fields.NumberOfFields:
                return {"success": False, "error": f"Field index {field_index} out of range (max: {fields.NumberOfFields})"}

            wavelengths = self.oss.SystemData.Wavelengths
            if wavelength_index > wavelengths.NumberOfWavelengths:
                return {"success": False, "error": f"Wavelength index {wavelength_index} out of range (max: {wavelengths.NumberOfWavelengths})"}

            field = fields.GetField(field_index)
            field_x = _extract_value(field.X)
            field_y = _extract_value(field.Y)
            wavelength_um = _extract_value(wavelengths.GetWavelength(wavelength_index).Wavelength, DEFAULT_WAVELENGTH_UM)

            idm = self._zp.constants.Analysis.AnalysisIDM
            analysis = self._zp.analyses.new_analysis(
                self.oss,
                idm.FftPsf,
                settings_first=True,
            )

            try:
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

                # Extract 2D PSF data with spatial metadata
                results = analysis.Results
                psf_fields: dict[str, Any] = {}
                fft_header_lines = []

                if results is not None:
                    try:
                        num_grids = results.NumberOfDataGrids
                        if num_grids > 0:
                            grid = results.GetDataGrid(0)
                            if grid is not None:
                                grid_meta = self._extract_grid_with_metadata(grid)
                                if grid_meta is not None:
                                    psf_fields = _grid_meta_to_psf_fields(grid_meta)
                                    logger.info(f"PSF data extracted: shape={grid_meta.data.shape}, peak={psf_fields.get('psf_peak')}")
                    except Exception as e:
                        logger.warning(f"PSF: Could not extract data grid: {e}")

                    # Grab header text BEFORE closing the analysis (COM object
                    # becomes invalid after Close).
                    try:
                        header_data = results.HeaderData
                        if header_data is not None and header_data.Lines:
                            fft_header_lines = list(header_data.Lines)
                            for i, hl in enumerate(fft_header_lines):
                                logger.debug(f"FFT PSF: header[{i}]: {str(hl)!r}")
                        else:
                            logger.debug("FFT PSF: HeaderData or Lines is empty")
                    except Exception as e:
                        logger.warning(f"FFT PSF: HeaderData extraction failed: {e}")
            finally:
                try:
                    analysis.Close()
                except Exception:
                    pass

            if not psf_fields:
                return {"success": False, "error": "FFT PSF analysis did not produce data"}

            strehl_ratio = _parse_strehl_from_header(fft_header_lines)
            if strehl_ratio is not None:
                logger.debug(f"FFT PSF: Strehl parsed from FFT header: {strehl_ratio}")
            else:
                logger.warning("FFT PSF: Strehl extraction failed — not found in HeaderData.Lines")

            result = {
                "success": True,
                **psf_fields,
                "strehl_ratio": strehl_ratio,
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
        """Get Huygens PSF data. More accurate than FFT for highly aberrated systems.

        Returns the same structure as get_psf().
        """
        try:
            fields = self.oss.SystemData.Fields
            if field_index > fields.NumberOfFields:
                return {"success": False, "error": f"Field index {field_index} out of range (max: {fields.NumberOfFields})"}

            wavelengths = self.oss.SystemData.Wavelengths
            if wavelength_index > wavelengths.NumberOfWavelengths:
                return {"success": False, "error": f"Wavelength index {wavelength_index} out of range (max: {wavelengths.NumberOfWavelengths})"}

            field = fields.GetField(field_index)
            field_x = _extract_value(field.X)
            field_y = _extract_value(field.Y)
            wavelength_um = _extract_value(wavelengths.GetWavelength(wavelength_index).Wavelength, DEFAULT_WAVELENGTH_UM)

            idm = self._zp.constants.Analysis.AnalysisIDM
            analysis = self._zp.analyses.new_analysis(
                self.oss,
                idm.HuygensPsf,
                settings_first=True,
            )

            try:
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
                psf_fields: dict[str, Any] = {}
                strehl_ratio = None

                if results is not None:
                    # Extract 2D PSF data grid with spatial metadata
                    num_grids = results.NumberOfDataGrids
                    if num_grids > 0:
                        grid = results.GetDataGrid(0)
                        if grid is not None:
                            grid_meta = self._extract_grid_with_metadata(grid)
                            if grid_meta is not None:
                                psf_fields = _grid_meta_to_psf_fields(grid_meta)
                                logger.info(f"Huygens PSF data extracted: shape={grid_meta.data.shape}, peak={psf_fields.get('psf_peak')}")

                    # Extract Strehl ratio from header text.
                    # Materialize .NET IList before analysis.Close() invalidates COM objects.
                    header_data = results.HeaderData
                    if header_data is not None and header_data.Lines:
                        header_lines = list(header_data.Lines)
                        for i, hl in enumerate(header_lines):
                            logger.debug(f"Huygens PSF: header[{i}]: {str(hl)!r}")
                        strehl_ratio = _parse_strehl_from_header(header_lines)
                        if strehl_ratio is not None:
                            logger.debug(f"Huygens PSF: Strehl parsed from header: {strehl_ratio}")
                    else:
                        logger.debug("Huygens PSF: HeaderData or Lines is empty")

                if strehl_ratio is None:
                    logger.warning("Huygens PSF: Strehl extraction failed — not found in HeaderData.Lines")
            finally:
                try:
                    analysis.Close()
                except Exception:
                    pass

            if not psf_fields:
                return {"success": False, "error": "Huygens PSF analysis did not produce data"}

            result = {
                "success": True,
                **psf_fields,
                "strehl_ratio": strehl_ratio,
                "wavelength_um": float(wavelength_um),
                "field_x": field_x,
                "field_y": field_y,
            }
            _log_raw_output("/huygens-psf", result)
            return result

        except Exception as e:
            return {"success": False, "error": f"Huygens PSF analysis failed: {e}"}

    def _run_fan_analysis(
        self,
        analysis_idm,
        label: str,
        endpoint: str,
        field_index: int,
        wavelength_index: int,
        plot_scale: float,
        number_of_rays: int,
    ) -> dict[str, Any]:
        """Run a fan analysis (Ray Fan or OPD Fan) and return T/S data per field."""
        try:
            sys_fields = self.oss.SystemData.Fields
            num_fields = sys_fields.NumberOfFields
            sys_wl = self.oss.SystemData.Wavelengths
            num_wl = sys_wl.NumberOfWavelengths

            if field_index > 0 and field_index > num_fields:
                return {"success": False, "error": f"Field index {field_index} out of range (max: {num_fields})"}
            if wavelength_index > 0 and wavelength_index > num_wl:
                return {"success": False, "error": f"Wavelength index {wavelength_index} out of range (max: {num_wl})"}

            logger.info(f"{label}: field={field_index}, wl={wavelength_index}, rays={number_of_rays}")

            analysis = self._zp.analyses.new_analysis(
                self.oss, analysis_idm, settings_first=True
            )
            try:
                settings = analysis.Settings

                if field_index == 0:
                    settings.Field.UseAllFields()
                else:
                    settings.Field.SetFieldNumber(field_index)

                if wavelength_index == 0:
                    settings.Wavelength.UseAllWavelengths()
                else:
                    settings.Wavelength.SetWavelengthNumber(wavelength_index)

                settings.NumberOfRays = number_of_rays
                if plot_scale > 0:
                    settings.PlotScale = plot_scale

                t0 = time.perf_counter()
                try:
                    analysis.ApplyAndWaitForCompletion()
                finally:
                    log_timing(logger, f"{label}.ApplyAndWaitForCompletion", (time.perf_counter() - t0) * 1000)

                results = analysis.Results
                actual_fields = num_fields if field_index == 0 else 1
                actual_wl = num_wl if wavelength_index == 0 else 1

                all_fans = []
                max_ab = 0.0

                for fi in range(actual_fields):
                    field_num = fi + 1 if field_index == 0 else field_index
                    field_obj = sys_fields.GetField(field_num)
                    fx = _extract_value(field_obj.X, 0.0)
                    fy = _extract_value(field_obj.Y, 0.0)

                    tan_x, tan_y_2d = _extract_fan_series(results.GetDataSeries(fi * 2))
                    sag_x, sag_y_2d = _extract_fan_series(results.GetDataSeries(fi * 2 + 1))

                    tpy = tan_x.tolist()
                    spx = sag_x.tolist()

                    for wi in range(actual_wl):
                        wl_num = wi + 1 if wavelength_index == 0 else wavelength_index
                        wl_um = _extract_value(sys_wl.GetWavelength(wl_num).Wavelength, DEFAULT_WAVELENGTH_UM)

                        tey = _sanitize_fan_values(tan_y_2d[:, wi].tolist())
                        sex = _sanitize_fan_values(sag_y_2d[:, wi].tolist())

                        for arr in (tey, sex):
                            if arr:
                                max_ab = max(max_ab, max(abs(v) for v in arr))

                        all_fans.append({
                            "field_index": field_num - 1,
                            "field_x": fx, "field_y": fy,
                            "wavelength_um": wl_um,
                            "wavelength_index": wl_num - 1,
                            "tangential_py": tpy, "tangential_ey": tey,
                            "sagittal_px": spx, "sagittal_ex": sex,
                        })
            finally:
                analysis.Close()

            result = {
                "success": True, "fans": all_fans,
                "max_aberration": float(max_ab),
                "num_fields": num_fields, "num_wavelengths": num_wl,
            }
            _log_raw_output(endpoint, result)
            return result

        except Exception as e:
            logger.error(f"{label} failed: {e}", exc_info=True)
            return {"success": False, "error": f"{label} analysis failed: {e}"}

    def get_ray_fan(
        self,
        field_index: int = 0,
        wavelength_index: int = 0,
        plot_scale: float = 0.0,
        number_of_rays: int = 20,
    ) -> dict[str, Any]:
        """Get Ray Fan (Ray Aberration) data using raw ZOS-API.

        Bypasses ZOSPy's RayFan wrapper which has a bug where SeriesLabels
        can contain None entries. Returns raw pupil/aberration data per field.
        """
        idm = self._zp.constants.Analysis.AnalysisIDM
        return self._run_fan_analysis(
            analysis_idm=idm.RayFan,
            label="RayFan",
            endpoint="/ray-fan",
            field_index=field_index,
            wavelength_index=wavelength_index,
            plot_scale=plot_scale,
            number_of_rays=number_of_rays,
        )

    def get_optical_path_fan(
        self,
        field_index: int = 0,
        wavelength_index: int = 0,
        plot_scale: float = 0.0,
        number_of_rays: int = 20,
    ) -> dict[str, Any]:
        """Get OPD Fan (Optical Path Difference Fan) data using raw ZOS-API.

        Returns OPD in waves per field. Same data shape as get_ray_fan().
        """
        idm = self._zp.constants.Analysis.AnalysisIDM
        return self._run_fan_analysis(
            analysis_idm=idm.OpticalPathFan,
            label="OPDFan",
            endpoint="/optical-path-fan",
            field_index=field_index,
            wavelength_index=wavelength_index,
            plot_scale=plot_scale,
            number_of_rays=number_of_rays,
        )

    def get_through_focus_spot(
        self,
        delta_focus: float = 0.1,
        number_of_steps: int = 5,
        ray_density: int = 15,
        field_index: int = 0,
        wavelength_index: int = 0,
    ) -> dict[str, Any]:
        """Get Through Focus Spot Diagram by batch ray tracing at multiple defocus positions."""
        try:
            fields = self.oss.SystemData.Fields
            num_fields = fields.NumberOfFields
            wavelengths = self.oss.SystemData.Wavelengths
            num_wavelengths = wavelengths.NumberOfWavelengths

            if field_index == 0:
                field_indices = list(range(1, num_fields + 1))
            else:
                if field_index > num_fields:
                    return {"success": False, "error": f"Field index {field_index} out of range (max: {num_fields})"}
                field_indices = [field_index]

            if wavelength_index == 0:
                wl_indices = list(range(1, num_wavelengths + 1))
            else:
                if wavelength_index > num_wavelengths:
                    return {"success": False, "error": f"Wavelength index {wavelength_index} out of range (max: {num_wavelengths})"}
                wl_indices = [wavelength_index]

            focus_positions = list(np.linspace(
                -number_of_steps * delta_focus,
                number_of_steps * delta_focus,
                2 * number_of_steps + 1,
            ))

            primary_wl_um = self._get_primary_wavelength_um() or DEFAULT_WAVELENGTH_UM
            airy_radius_um = self._extract_airy_radius(wavelength_index=wavelength_index)

            lde = self.oss.LDE
            image_surf_idx = lde.NumberOfSurfaces - 1
            last_surf_idx = image_surf_idx - 1
            original_thickness = _extract_value(lde.GetSurfaceAt(last_surf_idx).Thickness, 0.0)
            if original_thickness == 0.0:
                logger.warning(f"[TF-SPOT] Surface {last_surf_idx} thickness is 0.0 — may be solver-controlled or unreadable")

            logger.info(
                f"[TF-SPOT] Starting: delta={delta_focus}mm, steps={number_of_steps}, "
                f"density={ray_density}, fields={field_indices}, wavelengths={wl_indices}, "
                f"surface={last_surf_idx}, orig_thickness={original_thickness}"
            )

            field_coords: dict[int, tuple[float, float]] = {}
            for fi in field_indices:
                field = fields.GetField(fi)
                field_coords[fi] = (_extract_value(field.X), _extract_value(field.Y))

            num_rays = (ray_density + 1) ** 2
            pupil_coords = generate_random_coords(num_rays, seed=42)
            is_radial, max_field_x, max_field_y, max_field_r = _compute_field_normalization(fields, num_fields)

            fields_result: dict[int, dict[str, Any]] = {}
            for fi in field_indices:
                fx, fy = field_coords[fi]
                fields_result[fi] = {
                    "field_index": fi - 1,
                    "field_x": fx,
                    "field_y": fy,
                    "focus_spots": [],
                }

            def _empty_spot(fp: float) -> dict:
                """Build an empty spot entry for a failed or skipped focus position."""
                return {
                    "focus_position": float(fp),
                    "rms_radius_um": 0.0,
                    "geo_radius_um": 0.0,
                    "centroid_x": 0.0,
                    "centroid_y": 0.0,
                    "rays": [],
                }

            try:
                for focus_pos in focus_positions:
                    new_thickness = original_thickness + focus_pos
                    try:
                        lde.GetSurfaceAt(last_surf_idx).Thickness = new_thickness
                    except Exception as e:
                        logger.warning(f"[TF-SPOT] Could not set thickness for focus={focus_pos}: {e}")
                        for fi in field_indices:
                            fields_result[fi]["focus_spots"].append(_empty_spot(focus_pos))
                        continue

                    ray_trace = None
                    fields_done: set[int] = set()
                    try:
                        ray_trace = self.oss.Tools.OpenBatchRayTrace()
                        if ray_trace is None:
                            raise RuntimeError("OpenBatchRayTrace returned None")

                        max_rays_total = len(field_indices) * len(wl_indices) * len(pupil_coords)
                        norm_unpol = ray_trace.CreateNormUnpol(
                            max_rays_total,
                            self._zp.constants.Tools.RayTrace.RaysType.Real,
                            self.oss.LDE.NumberOfSurfaces,
                        )
                        if norm_unpol is None:
                            raise RuntimeError("CreateNormUnpol returned None")

                        opd_none = self._zp.constants.Tools.RayTrace.OPDMode.None_

                        for fi in field_indices:
                            fx, fy = field_coords[fi]
                            hx_norm, hy_norm = _normalize_field(fx, fy, is_radial, max_field_x, max_field_y, max_field_r)
                            for wi in wl_indices:
                                for px, py in pupil_coords:
                                    norm_unpol.AddRay(wi, hx_norm, hy_norm, float(px), float(py), opd_none)

                        ray_trace.RunAndWaitForCompletion()
                        norm_unpol.StartReadingResults()

                        for fi in field_indices:
                            rays_x: list[float] = []
                            rays_y: list[float] = []

                            for wi in wl_indices:
                                for _ in pupil_coords:
                                    result = norm_unpol.ReadNextResult()
                                    success_flag, err_code, vignette_code = result[0], result[2], result[3]
                                    if success_flag and err_code == 0 and vignette_code == 0:
                                        rays_x.append(float(result[4]) * 1000)  # mm to µm
                                        rays_y.append(float(result[5]) * 1000)

                            if rays_x:
                                ax = np.array(rays_x)
                                ay = np.array(rays_y)
                                cx = float(ax.mean())
                                cy = float(ay.mean())
                                dists = np.sqrt((ax - cx) ** 2 + (ay - cy) ** 2)
                                rms_r = float(np.sqrt(np.mean(dists ** 2)))
                                geo_r = float(dists.max())
                                rays_list = np.column_stack([ax, ay]).tolist()
                            else:
                                cx, cy, rms_r, geo_r = 0.0, 0.0, 0.0, 0.0
                                rays_list = []

                            fields_result[fi]["focus_spots"].append({
                                "focus_position": float(focus_pos),
                                "rms_radius_um": rms_r,
                                "geo_radius_um": geo_r,
                                "centroid_x": cx,
                                "centroid_y": cy,
                                "rays": rays_list,
                            })
                            fields_done.add(fi)

                    except Exception as e:
                        logger.warning(f"[TF-SPOT] Ray trace failed at focus={focus_pos}: {e}")
                        # Only fill empty spots for fields not already processed
                        for fi in field_indices:
                            if fi not in fields_done:
                                fields_result[fi]["focus_spots"].append(_empty_spot(focus_pos))
                    finally:
                        if ray_trace is not None:
                            try:
                                ray_trace.Close()
                            except Exception:
                                pass
            finally:
                # Always restore original thickness, even on unexpected exceptions
                try:
                    lde.GetSurfaceAt(last_surf_idx).Thickness = original_thickness
                except Exception as e:
                    logger.error(f"[TF-SPOT] CRITICAL: Could not restore original thickness: {e}")

            # Find best focus (minimum average RMS across all fields)
            best_focus_pos = 0.0
            best_focus_rms = float('inf')
            for i, fp in enumerate(focus_positions):
                rms_values = [
                    fields_result[fi]["focus_spots"][i]["rms_radius_um"]
                    for fi in field_indices
                    if i < len(fields_result[fi]["focus_spots"])
                    and fields_result[fi]["focus_spots"][i]["rms_radius_um"] > 0
                ]
                if rms_values:
                    avg_rms = sum(rms_values) / len(rms_values)
                    if avg_rms < best_focus_rms:
                        best_focus_rms = avg_rms
                        best_focus_pos = fp

            fields_list = [fields_result[fi] for fi in field_indices]

            result = {
                "success": True,
                "focus_positions": [float(x) for x in focus_positions],
                "fields": fields_list,
                "best_focus": {
                    "position": float(best_focus_pos),
                    "rms_radius_um": float(best_focus_rms) if math.isfinite(best_focus_rms) else 0.0,
                },
                "airy_radius_um": float(airy_radius_um) if airy_radius_um else None,
                "wavelength_um": float(primary_wl_um),
            }

            logger.info(
                f"[TF-SPOT] Done: {len(focus_positions)} positions, {len(fields_list)} fields, "
                f"best_focus={best_focus_pos:.4f}mm (rms={best_focus_rms:.2f}µm)"
            )
            _log_raw_output("/through-focus-spot", result)
            return result

        except Exception as e:
            logger.error(f"[TF-SPOT] Through Focus Spot failed: {e}", exc_info=True)
            return {"success": False, "error": f"Through Focus Spot analysis failed: {e}"}
