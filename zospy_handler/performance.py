"""Performance mixin – spot diagram, MTF, PSF, ray fan."""

import base64
import logging
import math
import re
import time
from typing import Any, Optional

import numpy as np

from zospy_handler._base import _extract_value, _log_raw_output, _extract_dataframe, GridWithMetadata, _compute_field_normalization, _normalize_field
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
        return None
    # Handle a single multi-line string (some ZOS-API versions)
    if isinstance(header_lines, str):
        lines_iter = header_lines.splitlines()
    elif hasattr(header_lines, '__iter__'):
        lines_iter = header_lines
    else:
        return None
    for line in lines_iter:
        m = _STREHL_RE.search(str(line))
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                pass
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


class PerformanceMixin:

    @staticmethod
    def _extract_mtf_series(series) -> tuple[list, list, list]:
        """Extract x, tangential, and sagittal data from an MTF data series.

        Tries bulk XData.Data/YData.Data first (single COM call).
        IMatrixData layout: GetLength(0)=Rows=freq points, GetLength(1)=Cols=NumSeries(tang/sag).
        Access pattern: y_raw[row, col] i.e. y_raw[freq_index, series_index].
        Falls back to per-point GetDataPoint() extraction.

        Returns (x_values, tangential_values, sagittal_values).
        sagittal_values is empty if the series is single-column.
        """
        if (hasattr(series, 'XData') and hasattr(series.XData, 'Data')
                and hasattr(series, 'YData') and hasattr(series.YData, 'Data')):
            try:
                x_raw = series.XData.Data
                y_raw = series.YData.Data
                n_pts = y_raw.GetLength(0)   # Rows = frequency points
                n_cols = y_raw.GetLength(1)  # Cols = NumSeries (tang/sag)
                if n_pts > 0:
                    if n_cols > 2:
                        logger.warning(f"MTF bulk: unexpected n_cols={n_cols} (expected 1 or 2); using cols 0 and 1 only")
                    x_len = x_raw.GetLength(0)
                    if x_len != n_pts:
                        logger.warning(f"MTF bulk: x length {x_len} != y points {n_pts}; truncating to shorter")
                        n_pts = min(x_len, n_pts)
                    x_vals = [float(x_raw[i]) for i in range(n_pts)]
                    tang = [float(y_raw[i, 0]) for i in range(n_pts)]
                    sag = [float(y_raw[i, 1]) for i in range(n_pts)] if n_cols >= 2 else []
                    logger.debug(f"MTF bulk extraction: {n_pts} pts x {n_cols} cols")
                    return x_vals, tang, sag
            except Exception as e:
                logger.warning(f"MTF bulk extraction failed, falling back to per-point: {type(e).__name__}: {e}")

        # Fallback: per-point extraction
        n_points = series.NumberOfPoints if hasattr(series, 'NumberOfPoints') else 0
        x_vals, tang = [], []
        for pi in range(n_points):
            pt = series.GetDataPoint(pi)
            if pt is not None:
                x_vals.append(_extract_value(pt.X if hasattr(pt, 'X') else pt[0]))
                tang.append(_extract_value(pt.Y if hasattr(pt, 'Y') else pt[1]))
        logger.debug(f"MTF per-point extraction: {len(x_vals)} pts")
        return x_vals, tang, []

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
                         Default 20 → 441 rays. Range 5-40.
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
            logger.warning(f"[SPOT] Could not resolve Spot.Reference.{refer_name}, falling back to integer: {e}")
            reference_value = 0 if reference == "chief_ray" else 1

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
                airy_radius = self._extract_airy_radius(analysis.Results)
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
            ray_density: Rays per axis (1-20)
            reference_value: Spot.Reference enum value (ChiefRay or Centroid)
            field_index: Field index (1-indexed). None = all fields.
            wavelength_index: Wavelength index (1-indexed). None = all wavelengths.
        """
        # Set ray density
        if hasattr(settings, 'RayDensity'):
            settings.RayDensity = ray_density
        elif hasattr(settings, 'NumberOfRays'):
            settings.NumberOfRays = ray_density

        # Set reference point (ReferTo is the correct ZOSAPI property name)
        try:
            settings.ReferTo = reference_value
        except Exception as e:
            logger.warning(f"[SPOT] Failed to set ReferTo={reference_value}: {e}")

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
                logger.warning("Could not open BatchRayTrace tool")
                return spot_rays

            # Get the normalized unpolarized ray trace interface
            max_rays = len(field_indices) * len(wl_indices) * len(pupil_coords)
            # Per official ZOSAPI examples, toSurface = NumberOfSurfaces (not -1)
            norm_unpol = ray_trace.CreateNormUnpol(
                max_rays,
                self._zp.constants.Tools.RayTrace.RaysType.Real,
                self.oss.LDE.NumberOfSurfaces,
            )

            if norm_unpol is None:
                logger.warning("Could not create NormUnpol ray trace")
                return spot_rays

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
            # Find the actual primary wavelength (not always index 1)
            wavelengths = self.oss.SystemData.Wavelengths
            primary_wl_um = 0.5876  # fallback
            for wi in range(1, wavelengths.NumberOfWavelengths + 1):
                wl = wavelengths.GetWavelength(wi)
                if wl.IsPrimary:
                    primary_wl_um = _extract_value(wl.Wavelength, 0.5876)
                    break
            else:
                # No primary found, use wavelength 1
                primary_wl_um = _extract_value(
                    wavelengths.GetWavelength(1).Wavelength, 0.5876,
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

                    # Extract data from results
                    results = analysis.Results
                    tangential = []
                    sagittal = []
                    freq_data = []

                    if results is not None:
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

                                if "diffrac" in desc_lower or "limit" in desc_lower:
                                    dl_x, dl_y, _ = self._extract_mtf_series(series)
                                    if dl_y and not diffraction_limit_from_api:
                                        diffraction_limit_from_api = dl_y
                                        if not freq_data:
                                            freq_data = dl_x
                                        logger.debug(f"MTF field {fi}: extracted diffraction limit from API ({len(dl_y)} pts)")
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
                max_freq = maximum_frequency if maximum_frequency > 0 else (cutoff_frequency or 100.0)
                # Match grid size from sampling (e.g. "128x128" → 128 points)
                grid_size = int(sampling.split('x')[0]) if 'x' in sampling else 64
                frequency = list(np.linspace(0, max_freq, grid_size))

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

                                # Extract diffraction limit from API series
                                if "diffrac" in desc_lower or "limit" in desc_lower:
                                    dl_x, dl_y, _ = self._extract_mtf_series(series)
                                    if dl_y and not diffraction_limit_from_api:
                                        diffraction_limit_from_api = dl_y
                                        if not freq_data:
                                            freq_data = dl_x
                                        logger.debug(f"Huygens MTF field {fi}: extracted diffraction limit from API ({len(dl_y)} pts)")
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
                max_freq = maximum_frequency if maximum_frequency > 0 else (cutoff_frequency or 100.0)
                grid_size = int(sampling.split('x')[0]) if 'x' in sampling else 64
                frequency = list(np.linspace(0, max_freq, grid_size))

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
                                    logger.info(f"PSF data extracted: shape={grid_meta.data.shape}, peak={psf_fields['psf_peak']:.6f}")
                    except Exception as e:
                        logger.warning(f"PSF: Could not extract data grid: {e}")

                    # Grab header text BEFORE closing the analysis (COM object
                    # becomes invalid after Close).
                    try:
                        if hasattr(results, 'HeaderData'):
                            fft_header_lines = results.HeaderData.Lines
                    except Exception:
                        pass
            finally:
                try:
                    analysis.Close()
                except Exception:
                    pass

            if not psf_fields:
                return {"success": False, "error": "FFT PSF analysis did not produce data"}

            # Try to get Strehl ratio — check FFT header first (cheap),
            # then fall back to a minimal 32x32 Huygens run.
            strehl_ratio = _parse_strehl_from_header(fft_header_lines)

            if strehl_ratio is None:
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
                        sampling="32x32",
                    )

                    huygens_start = time.perf_counter()
                    try:
                        huygens.ApplyAndWaitForCompletion()
                    finally:
                        huygens_elapsed_ms = (time.perf_counter() - huygens_start) * 1000
                        log_timing(logger, "HuygensPSF.run (Strehl-only, 32x32)", huygens_elapsed_ms)

                    h_results = huygens.Results
                    if h_results is not None:
                        try:
                            header_text = h_results.HeaderData.Lines if hasattr(h_results, 'HeaderData') else ""
                            strehl_ratio = _parse_strehl_from_header(header_text)
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
                    try:
                        num_grids = results.NumberOfDataGrids
                        if num_grids > 0:
                            grid = results.GetDataGrid(0)
                            if grid is not None:
                                grid_meta = self._extract_grid_with_metadata(grid)
                                if grid_meta is not None:
                                    psf_fields = _grid_meta_to_psf_fields(grid_meta)
                                    logger.info(f"Huygens PSF data extracted: shape={grid_meta.data.shape}, peak={psf_fields['psf_peak']:.6f}")
                    except Exception as e:
                        logger.warning(f"Huygens PSF: Could not extract data grid: {e}")

                    # Extract Strehl ratio from header text
                    try:
                        header_text = results.HeaderData.Lines if hasattr(results, 'HeaderData') else ""
                        strehl_ratio = _parse_strehl_from_header(header_text)
                    except Exception as e:
                        logger.debug(f"Huygens PSF: Could not extract Strehl ratio from header: {e}")
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
