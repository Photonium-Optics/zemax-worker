"""Physical-optics mixin – POP, geometric image analysis."""

import base64
import logging
import time
from typing import Any, Callable, Optional

import numpy as np

from zospy_handler._base import _extract_value, _log_raw_output
from utils.timing import log_timing

logger = logging.getLogger(__name__)


def _override_beam_param(
    beam_params: dict,
    value: float,
    fallback_key: str,
    match_fn: Callable[[str], bool],
) -> None:
    """Override a beam parameter by fuzzy-matching the key name.

    Searches beam_params keys (case-insensitive) using match_fn.
    Falls back to fallback_key if no fuzzy match is found.
    """
    for key in beam_params:
        if match_fn(key.lower()):
            beam_params[key] = value
            return
    if fallback_key in beam_params:
        beam_params[fallback_key] = value


class PhysicalOpticsMixin:

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
                    # run() returns AnalysisResult where .data is a DataFrame
                    # with .values (numpy array). ZosPy wrapper always returns this shape.
                    actual_data = result.data

                    if actual_data is None:
                        return {"success": False, "error": "Geometric Image Analysis returned no data"}

                    arr = np.array(actual_data.values, dtype=np.float64)

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
            paraxial = self.get_paraxial_data()

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
                _override_beam_param(beam_params, waist_x, "Waist X",
                                     lambda k: "waist" in k and ("x" in k or "size" in k) and "y" not in k)
            if waist_y is not None:
                _override_beam_param(beam_params, waist_y, "Waist Y",
                                     lambda k: "waist" in k and "y" in k and "x" not in k)

            logger.info(
                f"POP: field={field_index}, wl={wavelength_index}, beam={beam_type}, "
                f"sampling={x_sampling}x{y_sampling}, width={x_width}x{y_width}, "
                f"beam_params={beam_params}"
            )

            image_b64 = None
            array_shape = None
            array_dtype = None

            # Use raw ZOSAPI DataGrid approach (reliable)
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
                    settings.XSampling = x_sampling
                    settings.YSampling = y_sampling

                    pop_start = time.perf_counter()
                    try:
                        analysis.ApplyAndWaitForCompletion()
                    finally:
                        pop_elapsed_ms = (time.perf_counter() - pop_start) * 1000
                        log_timing(logger, "POP.raw_api.run", pop_elapsed_ms)

                    results = analysis.Results
                    if results is not None and results.NumberOfDataGrids > 0:
                        grid = results.GetDataGrid(0)
                        if grid is not None:
                            arr = self._extract_data_grid(grid)
                            if arr is not None:
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
                logger.warning(f"POP: Raw API failed: {e}")

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
