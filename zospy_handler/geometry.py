"""Geometry mixin – cross-section, paraxial data, cardinal points, surface data, curvature."""

import base64
import logging
import math
import os
import tempfile
import time
from typing import Any, Literal, Optional

import numpy as np

from config import (
    DEFAULT_NUM_CROSS_SECTION_RAYS, CROSS_SECTION_IMAGE_SIZE,
    CROSS_SECTION_TEMP_FILENAME, MIN_IMAGE_EXPORT_VERSION, FIELD_TYPE_MAP,
)
from zospy_handler._base import _enum_name, _extract_value, _log_raw_output
from utils.timing import log_timing

logger = logging.getLogger(__name__)


class GeometryMixin:

    def get_paraxial_data(self) -> dict[str, Any]:
        """Get first-order (paraxial) optical properties including EFL, F/#, NA, and image height."""
        paraxial = self._get_paraxial_from_lde()

        efl = self._get_efl()
        if efl is not None:
            paraxial["efl"] = efl

        bfl = self._get_bfl()
        if bfl is not None:
            paraxial["bfl"] = bfl

        fno = self._get_fno()
        if fno is not None:
            paraxial["fno"] = fno
            paraxial["na"] = 1.0 / (2.0 * fno) if fno > 0 else None

        # Compute image height from EFL and max field angle
        max_field = paraxial.get("max_field")
        if (
            efl is not None
            and max_field is not None
            and max_field > 0
            and paraxial.get("field_type") == "object_angle"
        ):
            paraxial["image_height"] = abs(efl) * math.tan(math.radians(max_field))

        return paraxial

    def get_cardinal_points(self) -> dict[str, Any]:
        """Get cardinal points (principal, nodal, focal planes) for object and image space."""
        try:
            result_obj = self._zp.analyses.reports.CardinalPoints().run(self.oss)
            data = result_obj.data

            cardinal_points = []
            spec = data.cardinal_points
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
        """Generate cross-section diagram via ZosPy's CrossSection analysis."""
        # Validate system state
        num_fields = self.oss.SystemData.Fields.NumberOfFields
        if num_fields == 0:
            return {"success": False, "error": "System has no fields defined"}

        zos_version = self.zos.version if hasattr(self.zos, 'version') else None
        if zos_version and zos_version < MIN_IMAGE_EXPORT_VERSION:
            return {"success": False, "error": f"OpticStudio {zos_version} < 24.1.0 — image export not supported"}

        temp_path = os.path.join(tempfile.gettempdir(), CROSS_SECTION_TEMP_FILENAME)

        # Always collect paraxial data and surface geometry — these don't depend
        # on the image export succeeding.
        paraxial = self.get_paraxial_data()
        surfaces_data = self._get_surface_geometry()
        rays_total = number_of_rays * max(1, num_fields)

        image_b64: Optional[str] = None
        image_error: Optional[str] = None

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
                image_error = "CrossSection analysis did not produce an image"
            else:
                with open(temp_path, 'rb') as f:
                    image_b64 = base64.b64encode(f.read()).decode('utf-8')
                logger.info(f"Successfully exported CrossSection image, size = {len(image_b64)}")

        except Exception as e:
            logger.warning(f"CrossSection image export failed: {e}")
            # ZosPy raises a generic error; try to get the real message from OpticStudio.
            image_error = self._get_cross_section_error(str(e))
        finally:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass

        result = {
            "success": image_b64 is not None,
            "image": image_b64,
            "image_format": "png" if image_b64 else None,
            "array_shape": None,
            "array_dtype": None,
            "paraxial": paraxial,
            "surfaces": surfaces_data,
            "rays_total": rays_total,
            "rays_through": rays_total,
        }
        if image_error:
            result["error"] = image_error
        _log_raw_output("/cross-section", result)
        return result

    def _get_cross_section_error(self, fallback: str) -> str:
        """Extract the OpticStudio error message via the cross-section export tool, or return fallback."""
        tool = None
        try:
            tool = self.oss.Tools.Layouts.OpenCrossSectionExport()
            tool.RunAndWaitForCompletion()
            error_msg = tool.ErrorMessage  # ISystemTool.ErrorMessage (string property)
            if error_msg:
                error_msg = str(error_msg).strip()
                logger.info(f"[CROSS] OpticStudio error detail: {error_msg}")
                return error_msg
        except Exception as e2:
            logger.debug(f"[CROSS] Could not extract OpticStudio error: {e2}")
        finally:
            if tool is not None:
                try:
                    tool.Close()
                except Exception as e:
                    logger.warning(f"Failed to close tool: {e}")
        return fallback

    def _get_paraxial_from_lde(self) -> dict[str, Any]:
        """Get paraxial data directly from LDE and SystemData."""
        try:
            paraxial = {}

            aperture = self.oss.SystemData.Aperture
            paraxial["epd"] = _extract_value(aperture.ApertureValue)

            fields = self.oss.SystemData.Fields
            max_field = 0.0
            if fields.NumberOfFields > 0:
                for i in range(1, fields.NumberOfFields + 1):
                    f = fields.GetField(i)
                    max_field = max(max_field, abs(_extract_value(f.Y)), abs(_extract_value(f.X)))
            paraxial["max_field"] = max_field

            # IFields.GetFieldType() returns FieldType enum (always present)
            field_type_str = _enum_name(fields.GetFieldType())

            ft_type, ft_unit = FIELD_TYPE_MAP.get(
                field_type_str, (field_type_str.lower(), "mm")
            )
            paraxial["field_type"] = ft_type
            paraxial["field_unit"] = ft_unit

            # Calculate total track (algebraic sum of thicknesses, not abs)
            lde = self.oss.LDE
            total_track = 0.0
            for i in range(1, lde.NumberOfSurfaces):
                surface = lde.GetSurfaceAt(i)
                total_track += _extract_value(surface.Thickness)
            paraxial["total_track"] = total_track

            return paraxial
        except Exception as e:
            logger.warning(f"Could not get paraxial data from LDE: {e}")
            return {}

    def _get_surface_geometry(self) -> list[dict[str, Any]]:
        """Extract surface geometry from the LDE for client-side rendering."""
        surfaces = []
        lde = self.oss.LDE
        z_position = 0.0

        for i in range(1, lde.NumberOfSurfaces):
            surface = lde.GetSurfaceAt(i)

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
        """Read computed semi-diameters from all surfaces in the LDE."""
        semi_diameters = []
        lde = self.oss.LDE

        for i in range(1, lde.NumberOfSurfaces):
            surface = lde.GetSurfaceAt(i)
            sd = _extract_value(surface.SemiDiameter)

            semi_diameters.append({
                "index": i - 1,  # Convert to 0-indexed
                "value": sd,
            })

        return {"semi_diameters": semi_diameters}

    def get_surface_data_report(self) -> dict[str, Any]:
        """Get per-surface data (thickness, material, refractive index, power) via ZosPy SurfaceData."""
        try:
            zp = self._zp
            num_surfaces = self.oss.LDE.NumberOfSurfaces
            surfaces = []

            for surf_idx in range(1, num_surfaces):
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

                try:
                    lde_surf = self.oss.LDE.GetSurfaceAt(surf_idx)
                    entry["radius"] = _extract_value(lde_surf.Radius, 0.0)
                    entry["thickness"] = _extract_value(lde_surf.Thickness, 0.0)
                    entry["surface_type"] = str(lde_surf.TypeName)
                    mat_value = lde_surf.MaterialCell.Value
                    if mat_value:
                        entry["material"] = str(mat_value)
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
                        if surf_idx == 1:
                            log_timing(logger, f"SurfaceData.run(surf={surf_idx})", sd_elapsed)

                    if sd_result and sd_result.data is not None:
                        data = sd_result.data

                        entry["edge_thickness"] = _extract_value(data.edge_thickness.y, 0.0)
                        entry["thickness"] = _extract_value(data.thickness, entry["thickness"])

                        mat_data = data.material
                        glass = mat_data.glass
                        if glass is not None and not entry["material"]:
                            entry["material"] = glass if isinstance(glass, str) else str(glass)

                        if mat_data.indices:
                            entry["refractive_index"] = _extract_value(mat_data.indices[0].index, 0.0)

                        power_src = data.surface_powers.in_air
                        if power_src.power is not None:
                            for v in power_src.power.values():
                                entry["surface_power"] = _extract_value(v, 0.0)
                                break
                        elif power_src.efl is not None:
                            for v in power_src.efl.values():
                                efl_num = _extract_value(v, 0.0)
                                if efl_num != 0.0:
                                    entry["surface_power"] = 1.0 / efl_num
                                break

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
        """Get surface curvature map as a raw numpy array via ZosPy's Curvature analysis."""
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

            # Curvature.run() returns AnalysisResult where .data is
            # list[CurvatureResult] or CurvatureResult.
            # CurvatureResult.data is ValidatedDataFrame (pandas-like, has .values).
            if curvature_result.data is not None:
                curv_data = curvature_result.data
                if isinstance(curv_data, list):
                    curv_data = curv_data[0].data  # CurvatureResult.data → DataFrame
                elif hasattr(curv_data, 'data'):
                    curv_data = curv_data.data  # Single CurvatureResult → DataFrame
                arr = np.array(curv_data.values, dtype=np.float64)

                if arr.ndim >= 2:
                    # Compute statistics on valid (non-NaN) values
                    if not np.all(np.isnan(arr)):
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
