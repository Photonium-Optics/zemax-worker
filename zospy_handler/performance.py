"""Performance mixin – spot diagram, MTF, PSF, ray fan."""

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

    Accepts a list of line strings, a single multi-line string, or a .NET
    IList<string>.  Returns None if no Strehl value is found.

    Handles formats like:
      "Strehl Ratio : 0.8532"
      "Strehl Ratio : 0.8532 (0.550 um)"
      "Strehl = 8.532e-01"
    """
    if not header_lines:
        logger.debug(f"_parse_strehl_from_header: input is falsy ({type(header_lines).__name__}: {header_lines!r})")
        return None
    # Handle a single multi-line string (some ZOS-API versions)
    if isinstance(header_lines, str):
        lines_iter = header_lines.splitlines()
        logger.debug(f"_parse_strehl_from_header: str input, {len(lines_iter)} lines")
    elif hasattr(header_lines, '__iter__'):
        lines_iter = list(header_lines)  # materialize .NET IList once
        logger.debug(f"_parse_strehl_from_header: iterable input ({type(header_lines).__name__}), {len(lines_iter)} items")
    else:
        logger.debug(f"_parse_strehl_from_header: non-iterable input ({type(header_lines).__name__}), returning None")
        return None
    for line in lines_iter:
        line_str = str(line)
        m = _STREHL_RE.search(line_str)
        if m:
            try:
                val = float(m.group(1))
                logger.debug(f"_parse_strehl_from_header: matched Strehl={val} in line: {line_str!r}")
                return val
            except ValueError:
                pass
        # Log lines that contain "strehl" but didn't match the regex
        elif 'strehl' in line_str.lower():
            logger.warning(f"_parse_strehl_from_header: UNMATCHED Strehl-like line: {line_str!r}")
    logger.debug(f"_parse_strehl_from_header: no Strehl found in {len(lines_iter)} lines")
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

                desc = str(series.Description) if hasattr(series, 'Description') else ""
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

    def get_spot_diagram(
        self,
        ray_density: int = 20,
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
            ray_density: Controls number of rays: (ray_density+1)^2 per field/wavelength.
                         Default 20 → 441 rays. Range 5-200.
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

        # Read wavelength info from system data
        wavelengths_obj = self.oss.SystemData.Wavelengths
        num_wavelengths = wavelengths_obj.NumberOfWavelengths
        wavelength_info = []
        for wi in range(1, num_wavelengths + 1):
            wl = wavelengths_obj.GetWavelength(wi)
            wavelength_info.append({
                "index": wi,
                "um": _extract_value(wl.Wavelength, 0.0),
                "weight": _extract_value(wl.Weight, 1.0),
            })

        # Map reference parameter to OpticStudio Spot.Reference enum
        refer_name = "ChiefRay" if reference == "chief_ray" else "Centroid"
        try:
            reference_value = getattr(
                self._zp.constants.Analysis.Settings.Spot.Reference, refer_name
            )
        except Exception as e:
            return {"success": False, "error": f"Could not resolve Spot.Reference.{refer_name} (input: {reference!r}): {type(e).__name__}: {e}"}

        try:
            logger.info(f"[SPOT] Starting: ray_density={ray_density}, reference={reference}, num_fields={num_fields}, field_index={field_index}, wavelength_index={wavelength_index}")

            # Use ZosPy's new_analysis to access StandardSpot for metrics
            analysis = self._zp.analyses.new_analysis(
                self.oss,
                self._zp.constants.Analysis.AnalysisIDM.StandardSpot,
                settings_first=True,
            )

            # Configure and run the analysis
            self._configure_spot_analysis(analysis.Settings, ray_density, reference_value, field_index, wavelength_index)
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
                airy_radius = self._extract_airy_radius()
                spot_data = self._extract_spot_data_from_results(analysis.Results, fields, num_fields, field_index=field_index, wavelength_index=wavelength_index)
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
                "num_fields": num_fields,
                "num_wavelengths": num_wavelengths,
                "wavelength_info": wavelength_info,
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
            ray_density: Controls ray count: (ray_density+1)^2 random rays per field/wavelength.
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

        # Generate random pupil coordinates within the unit circle.
        # Per official ZOSAPI example (PythonStandalone_22), spot diagrams use
        # random sampling — not a grid — to avoid grid-like artifacts.
        # ray_density maps to num_rays = (ray_density + 1)^2 per field/wavelength.
        num_rays = (ray_density + 1) ** 2
        pupil_coords = generate_random_coords(num_rays, seed=42)
        logger.debug(f"[SPOT] Random pupil sampling: {len(pupil_coords)} rays (density={ray_density})")

        # Compute field normalization (must use ALL fields, respects Radial vs Rectangular)
        is_radial, max_field_x, max_field_y, max_field_r = _compute_field_normalization(fields, num_fields)
        logger.debug(f"[SPOT] Field extents: max_x={max_field_x}, max_y={max_field_y}, max_r={max_field_r}, radial={is_radial}")

        ray_trace = None
        try:
            # Use batch ray trace for efficiency
            ray_trace = self.oss.Tools.OpenBatchRayTrace()
            if ray_trace is None:
                logger.error("[SPOT] Could not open BatchRayTrace tool — OpticStudio connection may be degraded")
                raise RuntimeError("OpenBatchRayTrace returned None — OpticStudio tool API unavailable")

            # Get the normalized unpolarized ray trace interface
            max_rays = len(field_indices) * len(wl_indices) * len(pupil_coords)
            # Per official ZOSAPI examples, toSurface = NumberOfSurfaces (not -1)
            norm_unpol = ray_trace.CreateNormUnpol(
                max_rays,
                self._zp.constants.Tools.RayTrace.RaysType.Real,
                self.oss.LDE.NumberOfSurfaces,
            )

            if norm_unpol is None:
                logger.error("[SPOT] Could not create NormUnpol ray trace")
                raise RuntimeError("CreateNormUnpol returned None — ray trace initialization failed")

            # AddRay signature: (WaveNumber, Hx, Hy, Px, Py, OPDMode)
            opd_none = self._zp.constants.Tools.RayTrace.OPDMode.None_

            # Cache field coordinates to avoid redundant GetField COM calls
            field_coords: dict[int, tuple[float, float]] = {}
            rays_added = 0
            for fi in field_indices:
                field = fields.GetField(fi)
                field_x = _extract_value(field.X)
                field_y = _extract_value(field.Y)
                field_coords[fi] = (field_x, field_y)
                hx_norm, hy_norm = _normalize_field(field_x, field_y, is_radial, max_field_x, max_field_y, max_field_r)
                logger.debug(f"[SPOT] Field {fi}: raw=({field_x}, {field_y}), norm=({hx_norm}, {hy_norm})")
                for wi in wl_indices:
                    for px, py in pupil_coords:
                        norm_unpol.AddRay(wi, hx_norm, hy_norm, float(px), float(py), opd_none)
                        rays_added += 1
            logger.debug(f"[SPOT] Added {rays_added} rays to batch trace (expected {max_rays})")

            ray_trace.RunAndWaitForCompletion()

            # Initialize the read cursor before reading results (required by ZOSAPI)
            norm_unpol.StartReadingResults()

            total_success = 0
            total_vignetted = 0
            total_failed = 0
            for fi in field_indices:
                field_x, field_y = field_coords[fi]

                for wi in wl_indices:
                    wl_um = _extract_value(wavelengths.GetWavelength(wi).Wavelength, 0.0)
                    field_rays = {
                        "field_index": fi - 1,
                        "field_x": field_x,
                        "field_y": field_y,
                        "wavelength_index": wi - 1,
                        "wavelength_um": wl_um,
                        "rays": [],
                    }

                    entry_failed = 0
                    entry_vignetted = 0
                    for _ in pupil_coords:
                        result = norm_unpol.ReadNextResult()
                        # ReadNextResult returns 15 values; we need success(0), err_code(2), vignette_code(3), x(4), y(5)
                        success, err_code, vignette_code = result[0], result[2], result[3]
                        if success and err_code == 0 and vignette_code == 0:
                            field_rays["rays"].append({"x": float(result[4]) * 1000, "y": float(result[5]) * 1000})
                            total_success += 1
                        elif success and err_code == 0:
                            total_vignetted += 1
                            entry_vignetted += 1
                        else:
                            total_failed += 1
                            entry_failed += 1

                    if entry_failed > 0 or entry_vignetted > 0:
                        logger.debug(f"[SPOT] Field {fi} wl {wi}: {len(field_rays['rays'])} OK, {entry_vignetted} vignetted, {entry_failed} failed")
                    spot_rays.append(field_rays)

            logger.info(f"[SPOT] Ray trace: {total_success} success, {total_vignetted} vignetted, {total_failed} failed out of {rays_added}")

        except Exception as e:
            logger.error(f"[SPOT] Batch ray trace FAILED: {type(e).__name__}: {e}", exc_info=True)
        finally:
            if ray_trace is not None:
                try:
                    ray_trace.Close()
                except Exception:
                    pass

        return spot_rays

    def _extract_airy_radius(self) -> Optional[float]:
        """Compute Airy disk radius: r_airy = 1.22 * wavelength * f_number.

        Result is returned in micrometers (µm).

        Returns:
            Airy radius in µm, or None if F/# or primary wavelength unavailable
        """
        fno = self._get_fno()
        if not fno or fno <= 0:
            logger.warning(f"[SPOT] Cannot compute airy radius: fno={fno}")
            return None

        primary_wl_um = self._get_primary_wavelength_um()
        if primary_wl_um is None:
            return None

        airy_radius_um = 1.22 * primary_wl_um * fno
        logger.info(f"[SPOT] Computed airy_radius: 1.22 * {primary_wl_um:.4f}µm * F/{fno:.2f} = {airy_radius_um:.3f} µm")
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
                val = _extract_value(wl.Wavelength, 0.0)
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
                field = fields.GetField(fi + 1)  # 1-indexed
                # Use _extract_value for UnitField objects
                field_data = self._create_field_spot_data(fi, _extract_value(field.X), _extract_value(field.Y))

                # Try to get spot data for this field
                self._populate_spot_data_from_results(results, fi, field_data, wavelength_index=wavelength_index)
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
        wavelength_index: Optional[int] = None,
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
            wavelength_index: 1-indexed wavelength number. None = primary wavelength (1).
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
            fi_1 = field_index + 1
            wi = wavelength_index if wavelength_index else 1

            # Log matrix dimensions for debugging (only for first field to avoid noise)
            if field_index == 0:
                if hasattr(spot_data, 'NumberOfFields') and hasattr(spot_data, 'NumberOfWavelengths'):
                    logger.debug(f"[SPOT] SpotData matrix: NumberOfFields={spot_data.NumberOfFields}, NumberOfWavelengths={spot_data.NumberOfWavelengths}")
                else:
                    logger.debug("[SPOT] SpotData matrix dimensions unknown")

            # StandardSpot SpotData methods return values in µm (not lens units).
            # The ZOSAPI example (PythonStandalone_22) prints these raw with no conversion.
            if hasattr(spot_data, 'GetRMSSpotSizeFor'):
                field_data["rms_radius"] = _extract_value(spot_data.GetRMSSpotSizeFor(fi_1, wi))
            if hasattr(spot_data, 'GetGeoSpotSizeFor'):
                field_data["geo_radius"] = _extract_value(spot_data.GetGeoSpotSizeFor(fi_1, wi))

            # Centroid coordinates from SpotData are in lens units (mm);
            # convert to µm to match batch ray trace positions (* 1000 at line ~471).
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
                logger.warning(f"MTF: Could not determine F/# (got {fno}), diffraction limit and cutoff will be omitted")

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
                    # Ask ZOS-API to include diffraction limit as a data series
                    if hasattr(settings, 'ShowDiffractionLimit'):
                        settings.ShowDiffractionLimit = True

                    mtf_start = time.perf_counter()
                    try:
                        analysis.ApplyAndWaitForCompletion()
                    finally:
                        mtf_elapsed_ms = (time.perf_counter() - mtf_start) * 1000
                        log_timing(logger, f"FFTMtf.run (field={fi})", mtf_elapsed_ms)

                    tangential, sagittal, freq_data, diffraction_limit_from_api = \
                        self._extract_mtf_field_data(
                            analysis.Results, fi, "MTF", diffraction_limit_from_api,
                        )

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

            validation_error = self._validate_mtf_results("FFT MTF", frequency, all_fields_data)
            if validation_error:
                return validation_error

            freq_arr = np.array(frequency)

            result = {
                "success": True,
                "frequency": freq_arr.tolist(),
                "fields": all_fields_data,
                "diffraction_limit": diffraction_limit_from_api,
                "cutoff_frequency": float(cutoff_frequency) if cutoff_frequency else None,
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
                logger.warning(f"Huygens MTF: Could not determine F/# (got {fno}), diffraction limit and cutoff will be omitted")

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
                    if hasattr(settings, 'ShowDiffractionLimit'):
                        settings.ShowDiffractionLimit = True

                    mtf_start = time.perf_counter()
                    try:
                        analysis.ApplyAndWaitForCompletion()
                    finally:
                        mtf_elapsed_ms = (time.perf_counter() - mtf_start) * 1000
                        log_timing(logger, f"HuygensMtf.run (field={fi})", mtf_elapsed_ms)

                    tangential, sagittal, freq_data, diffraction_limit_from_api = \
                        self._extract_mtf_field_data(
                            analysis.Results, fi, "Huygens MTF", diffraction_limit_from_api,
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

            validation_error = self._validate_mtf_results("Huygens MTF", frequency, all_fields_data)
            if validation_error:
                return validation_error

            freq_arr = np.array(frequency)

            result = {
                "success": True,
                "frequency": freq_arr.tolist(),
                "fields": all_fields_data,
                "diffraction_limit": diffraction_limit_from_api,
                "cutoff_frequency": float(cutoff_frequency) if cutoff_frequency else None,
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
                        if hasattr(results, 'HeaderData'):
                            hd = results.HeaderData
                            if hd is not None and hasattr(hd, 'Lines'):
                                fft_header_lines = list(hd.Lines)  # materialize .NET IEnumerable once
                                logger.debug(f"FFT PSF: HeaderData.Lines returned {len(fft_header_lines)} lines")
                                for i, hl in enumerate(fft_header_lines):
                                    logger.debug(f"FFT PSF: header[{i}]: {str(hl)!r}")
                            else:
                                logger.debug(f"FFT PSF: HeaderData is None or has no Lines (HeaderData={hd})")
                        else:
                            logger.debug("FFT PSF: results has no HeaderData attribute")
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
        """Run a fan analysis (Ray Fan or OPD Fan) using raw ZOS-API.

        Both fan types use the same ZOS-API pattern: configure field/wavelength
        settings, run the analysis, then extract tangential/sagittal data series
        per field. Each field produces 2 data series (tangential, sagittal),
        each with XData (1D pupil coords) and YData (2D: num_rays x num_wavelengths).

        Args:
            analysis_idm: The AnalysisIDM enum value (e.g., RayFan, OpticalPathFan)
            label: Human-readable name for logging (e.g., "RayFan", "OPDFan")
            endpoint: API endpoint for _log_raw_output (e.g., "/ray-fan")
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

            if field_index > 0 and field_index > num_fields:
                return {"success": False, "error": f"Field index {field_index} out of range (max: {num_fields})"}
            if wavelength_index > 0 and wavelength_index > num_wl:
                return {"success": False, "error": f"Wavelength index {wavelength_index} out of range (max: {num_wl})"}

            logger.info(f"{label}: field={field_index}, wl={wavelength_index}, rays={number_of_rays}")

            analysis = self._zp.analyses.new_analysis(
                self.oss, analysis_idm, settings_first=True
            )
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

            try:
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
                        wl_um = _extract_value(sys_wl.GetWavelength(wl_num).Wavelength, 0.0)

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
        """
        Get Through Focus Spot Diagram data by tracing spot rays at multiple
        defocus positions.

        Loops through focus positions by temporarily adjusting the image surface
        thickness, then uses batch ray tracing at each position to collect raw
        ray (x,y) data — the same approach the standard spot diagram uses.

        Args:
            delta_focus: Focus step size in mm.
            number_of_steps: Steps in each direction from nominal (total = 2*steps+1).
            ray_density: Rays per axis ((ray_density+1)^2 total per field/wavelength).
            field_index: Field index (0 = all fields, 1+ = specific field, 1-indexed).
            wavelength_index: Wavelength index (0 = all wavelengths, 1+ = specific, 1-indexed).

        Returns:
            {
                "success": True,
                "focus_positions": [...],
                "fields": [{field_index, field_x, field_y, focus_spots: [{...}]}],
                "airy_radius_um": float,
                "wavelength_um": float,
            }
        """
        try:
            fields = self.oss.SystemData.Fields
            num_fields = fields.NumberOfFields
            wavelengths = self.oss.SystemData.Wavelengths
            num_wavelengths = wavelengths.NumberOfWavelengths

            # Determine which fields to trace
            if field_index == 0:
                field_indices = list(range(1, num_fields + 1))
            else:
                if field_index > num_fields:
                    return {"success": False, "error": f"Field index {field_index} out of range (max: {num_fields})"}
                field_indices = [field_index]

            # Determine which wavelengths to trace
            if wavelength_index == 0:
                wl_indices = list(range(1, num_wavelengths + 1))
            else:
                if wavelength_index > num_wavelengths:
                    return {"success": False, "error": f"Wavelength index {wavelength_index} out of range (max: {num_wavelengths})"}
                wl_indices = [wavelength_index]

            # Compute focus positions
            focus_positions = list(np.linspace(
                -number_of_steps * delta_focus,
                number_of_steps * delta_focus,
                2 * number_of_steps + 1,
            ))

            # Reuse shared primary wavelength + Airy radius helpers
            primary_wl_um = self._get_primary_wavelength_um() or 0.5876
            airy_radius_um = self._extract_airy_radius()

            # Get the image surface index and its current thickness
            lde = self.oss.LDE
            image_surf_idx = lde.NumberOfSurfaces - 1
            # The surface before image is the last one with thickness to image
            last_surf_idx = image_surf_idx - 1
            original_thickness = _extract_value(lde.GetSurfaceAt(last_surf_idx).Thickness, 0.0)
            if original_thickness == 0.0:
                logger.warning(f"[TF-SPOT] Surface {last_surf_idx} thickness is 0.0 — may be solver-controlled or unreadable")

            logger.info(
                f"[TF-SPOT] Starting: delta={delta_focus}mm, steps={number_of_steps}, "
                f"density={ray_density}, fields={field_indices}, wavelengths={wl_indices}, "
                f"surface={last_surf_idx}, orig_thickness={original_thickness}"
            )

            # Build field coordinate cache
            field_coords: dict[int, tuple[float, float]] = {}
            for fi in field_indices:
                field = fields.GetField(fi)
                field_coords[fi] = (_extract_value(field.X), _extract_value(field.Y))

            # Generate pupil coordinates (same approach as standard spot)
            num_rays = (ray_density + 1) ** 2
            pupil_coords = generate_random_coords(num_rays, seed=42)

            # Compute field normalization
            is_radial, max_field_x, max_field_y, max_field_r = _compute_field_normalization(fields, num_fields)

            # Initialize per-field result structure
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

            # Trace rays at each focus position (wrapped in try/finally to guarantee thickness restore)
            try:
                for focus_pos in focus_positions:
                    # Adjust thickness to shift focus
                    new_thickness = original_thickness + focus_pos
                    try:
                        lde.GetSurfaceAt(last_surf_idx).Thickness = new_thickness
                    except Exception as e:
                        logger.warning(f"[TF-SPOT] Could not set thickness for focus={focus_pos}: {e}")
                        for fi in field_indices:
                            fields_result[fi]["focus_spots"].append(_empty_spot(focus_pos))
                        continue

                    # Batch ray trace at this focus position
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

                        # Read results per field, per wavelength
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

                            # Compute centroid and RMS/GEO using numpy
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
                    "rms_radius_um": float(best_focus_rms) if best_focus_rms != float('inf') else 0.0,
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
