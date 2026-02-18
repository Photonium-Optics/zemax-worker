"""Polarization mixin – pupil map, transmission."""

import logging
import time
from typing import Any

from zospy_handler._base import _extract_value, _log_raw_output
from utils.timing import log_timing

logger = logging.getLogger(__name__)


class PolarizationMixin:
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
