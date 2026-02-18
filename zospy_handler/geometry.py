"""Geometry mixin – cross-section, paraxial data, cardinal points, surface data, curvature."""

import base64
import logging
import math
import os
import tempfile
import time
from typing import Any, Literal

import numpy as np

from config import (
    DEFAULT_NUM_CROSS_SECTION_RAYS, CROSS_SECTION_IMAGE_SIZE,
    CROSS_SECTION_TEMP_FILENAME, MIN_IMAGE_EXPORT_VERSION, FIELD_TYPE_MAP,
)
from zospy_handler._base import _extract_value, _log_raw_output
from utils.timing import log_timing

logger = logging.getLogger(__name__)


class GeometryMixin:

    def get_paraxial_data(self) -> dict[str, Any]:
        """
        Get first-order (paraxial) optical properties.

        Delegates to _get_paraxial_from_lde() and adds F/# if available.

        Returns:
            Dict with paraxial properties (epd, max_field, total_track, fno, etc.)
        """
        paraxial = self._get_paraxial_from_lde()

        # Add F/# if available
        fno = self._get_fno()
        if fno is not None:
            paraxial["fno"] = fno

        return paraxial

    def get_paraxial(self) -> dict[str, Any]:
        """
        Get comprehensive first-order optical properties in a single call.

        Combines EFL, BFL, F/#, NA, EPD, total track, and FOV info from
        SystemData analysis and LDE.

        Returns:
            Dict with success flag and paraxial properties.
        """
        try:
            # Get EFL and BFL
            efl = self._get_efl()
            bfl = None

            fno = self._get_fno()
            na = 1.0 / (2.0 * fno) if fno is not None and fno > 0 else None

            # Get EPD, total track, field info from LDE
            lde_data = self._get_paraxial_from_lde()
            max_field = lde_data.get("max_field")
            field_type = lde_data.get("field_type")

            # Compute image height from EFL and max field angle
            image_height = None
            if (
                efl is not None
                and max_field is not None
                and max_field > 0
                and field_type == "object_angle"
            ):
                image_height = abs(efl) * math.tan(math.radians(max_field))

            result = {
                "success": True,
                "efl": efl,
                "bfl": bfl,
                "fno": fno,
                "na": na,
                "epd": lde_data.get("epd"),
                "total_track": lde_data.get("total_track"),
                "max_field": max_field,
                "field_type": field_type,
                "field_unit": lde_data.get("field_unit"),
                "image_height": image_height,
            }
            _log_raw_output("/paraxial", result)
            return result

        except Exception as e:
            logger.error(f"get_paraxial failed: {e}")
            return {"success": False, "error": str(e)}

    def get_cardinal_points(self) -> dict[str, Any]:
        """
        Get cardinal points of the optical system.

        Reports principal planes, nodal points, focal points, anti-principal
        planes, and anti-nodal planes for both object and image space.

        Returns:
            Dict with success flag and cardinal points data.
        """
        try:
            result_obj = self._zp.analyses.reports.CardinalPoints().run(self.oss)
            data = result_obj.data

            # Build a flat list of cardinal point entries
            cardinal_points = []
            spec = data.cardinal_points

            # Each attribute of CardinalPointSpecification is a CardinalPoint
            # with .object (Object Space) and .image (Image Space) values.
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
        """
        Generate cross-section diagram using ZosPy's CrossSection analysis.

        Requires ZosPy >= 1.3.0 and OpticStudio >= 24.1.0 for image export.
        Returns error on failure — no fallbacks.

        Note: System must be pre-loaded via load_zmx_file().
        """
        # Validate system state
        num_fields = self.oss.SystemData.Fields.NumberOfFields
        if num_fields == 0:
            return {"success": False, "error": "System has no fields defined"}

        zos_version = self.zos.version if hasattr(self.zos, 'version') else None
        if zos_version and zos_version < MIN_IMAGE_EXPORT_VERSION:
            return {"success": False, "error": f"OpticStudio {zos_version} < 24.1.0 — image export not supported"}

        temp_path = os.path.join(tempfile.gettempdir(), CROSS_SECTION_TEMP_FILENAME)

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
                return {"success": False, "error": "CrossSection analysis did not produce an image"}

            with open(temp_path, 'rb') as f:
                image_b64 = base64.b64encode(f.read()).decode('utf-8')
            logger.info(f"Successfully exported CrossSection image, size = {len(image_b64)}")

            # Get paraxial data and surface geometry
            paraxial = self._get_paraxial_from_lde()
            efl = self._get_efl()
            if efl is not None:
                paraxial["efl"] = efl
            surfaces_data = self._get_surface_geometry()

            rays_total = number_of_rays * max(1, num_fields)

            result = {
                "success": True,
                "image": image_b64,
                "image_format": "png",
                "array_shape": None,
                "array_dtype": None,
                "paraxial": paraxial,
                "surfaces": surfaces_data,
                "rays_total": rays_total,
                "rays_through": rays_total,
            }
            _log_raw_output("/cross-section", result)
            return result

        except Exception as e:
            return {"success": False, "error": f"CrossSection analysis failed: {e}"}
        finally:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass

    def _get_paraxial_from_lde(self) -> dict[str, Any]:
        """
        Get paraxial data directly from LDE and SystemData.

        This method is more reliable than using analysis functions as it
        reads directly from the system data structures.

        Returns:
            Dict with keys: epd, max_field, field_type, field_unit, total_track
        """
        try:
            paraxial = {}

            # Get aperture info - use _extract_value for UnitField objects
            aperture = self.oss.SystemData.Aperture
            paraxial["epd"] = _extract_value(aperture.ApertureValue)

            # Get field info - use _extract_value for UnitField objects
            fields = self.oss.SystemData.Fields
            max_field = 0.0
            if fields.NumberOfFields > 0:
                for i in range(1, fields.NumberOfFields + 1):
                    f = fields.GetField(i)
                    max_field = max(max_field, abs(_extract_value(f.Y)), abs(_extract_value(f.X)))
            paraxial["max_field"] = max_field

            # Read actual field type from system (ZOSAPI FieldType enum)
            try:
                ft = fields.GetFieldType()
                field_type_str = getattr(ft, 'name', str(ft).split(".")[-1])
            except Exception:
                field_type_str = "Angle"

            ft_type, ft_unit = FIELD_TYPE_MAP.get(
                field_type_str, (field_type_str.lower(), "mm")
            )
            paraxial["field_type"] = ft_type
            paraxial["field_unit"] = ft_unit

            # Calculate total track from LDE - use _extract_value for UnitField objects
            lde = self.oss.LDE
            total_track = 0.0
            for i in range(1, lde.NumberOfSurfaces):
                surface = lde.GetSurfaceAt(i)
                total_track += abs(_extract_value(surface.Thickness))
            paraxial["total_track"] = total_track

            return paraxial
        except Exception as e:
            logger.warning(f"Could not get paraxial data from LDE: {e}")
            return {}

    def _get_surface_geometry(self) -> list[dict[str, Any]]:
        """
        Extract surface geometry for client-side cross-section rendering.

        This method reads the Lens Data Editor (LDE) to extract geometric
        properties of each surface. Used as fallback data when image export fails.

        Returns:
            List of surface dicts with keys: index, z, radius, thickness,
            semi_diameter, conic, material, is_stop
        """
        surfaces = []
        lde = self.oss.LDE
        z_position = 0.0

        for i in range(1, lde.NumberOfSurfaces):
            surface = lde.GetSurfaceAt(i)

            # Radius of 0 in Zemax means infinity (flat surface)
            # We convert to None for client-side rendering
            # Use _extract_value for all UnitField properties
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
        """
        Calculate semi-diameters by reading from surfaces after ray trace.

        Reads the SemiDiameter property from each surface in the LDE,
        which OpticStudio computes based on ray extent during tracing.

        Note: System must be pre-loaded via load_zmx_file().

        Returns:
            Dict with "semi_diameters" key containing list of
            {"index": int, "value": float} entries.
        """
        semi_diameters = []
        lde = self.oss.LDE

        # For each surface, get the computed semi-diameter
        # Use _extract_value for UnitField objects
        for i in range(1, lde.NumberOfSurfaces):
            surface = lde.GetSurfaceAt(i)
            sd = _extract_value(surface.SemiDiameter)

            semi_diameters.append({
                "index": i - 1,  # Convert to 0-indexed
                "value": sd,
            })

        return {"semi_diameters": semi_diameters}

    def get_surface_data_report(self) -> dict[str, Any]:
        """
        Get Surface Data Report from OpticStudio.

        Iterates over every surface in the system and runs the ZosPy SurfaceData
        analysis to collect edge thickness, center thickness, material name,
        refractive index, and surface power for each surface.

        Note: System must be pre-loaded via load_zmx_file().

        Returns:
            On success: {
                "success": True,
                "surfaces": [...],
                "paraxial": {...},
            }
            On error: {"success": False, "error": "..."}
        """
        try:
            zp = self._zp
            num_surfaces = self.oss.LDE.NumberOfSurfaces
            surfaces = []

            for surf_idx in range(num_surfaces):
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

                # Read basic LDE data for this surface
                try:
                    lde_surf = self.oss.LDE.GetSurfaceAt(surf_idx)
                    entry["radius"] = _extract_value(lde_surf.Radius, 0.0)
                    entry["thickness"] = _extract_value(lde_surf.Thickness, 0.0)

                    # Get surface type name
                    try:
                        type_name = str(lde_surf.TypeName) if hasattr(lde_surf, 'TypeName') else "Standard"
                        entry["surface_type"] = type_name
                    except Exception:
                        pass

                    # Get material name from LDE
                    try:
                        mat_cell = lde_surf.MaterialCell
                        if hasattr(mat_cell, 'Value') and mat_cell.Value:
                            entry["material"] = str(mat_cell.Value)
                    except Exception:
                        pass
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
                        if surf_idx == 0:
                            log_timing(logger, f"SurfaceData.run(surf={surf_idx})", sd_elapsed)

                    if sd_result and hasattr(sd_result, 'data') and sd_result.data is not None:
                        data = sd_result.data

                        # Edge thickness
                        if hasattr(data, 'edge_thickness') and data.edge_thickness is not None:
                            et = data.edge_thickness
                            # EdgeThickness has .y and .x — use .y (tangential plane)
                            if hasattr(et, 'y'):
                                entry["edge_thickness"] = _extract_value(et.y, 0.0)
                            elif hasattr(et, 'x'):
                                entry["edge_thickness"] = _extract_value(et.x, 0.0)

                        # Thickness override from analysis if available
                        if hasattr(data, 'thickness') and data.thickness is not None:
                            entry["thickness"] = _extract_value(data.thickness, entry["thickness"])

                        # Material and refractive index
                        if hasattr(data, 'material') and data.material is not None:
                            mat_data = data.material

                            # Glass name
                            if hasattr(mat_data, 'glass') and mat_data.glass is not None:
                                glass = mat_data.glass
                                if isinstance(glass, str):
                                    if glass and not entry["material"]:
                                        entry["material"] = glass
                                elif hasattr(glass, 'name'):
                                    entry["material"] = str(glass.name)
                                else:
                                    glass_str = str(glass)
                                    if glass_str and not entry["material"]:
                                        entry["material"] = glass_str

                            # Refractive index — take first wavelength's index
                            if hasattr(mat_data, 'indices') and mat_data.indices:
                                try:
                                    first_idx = mat_data.indices[0]
                                    if hasattr(first_idx, 'index'):
                                        entry["refractive_index"] = _extract_value(first_idx.index, 0.0)
                                    elif hasattr(first_idx, 'value'):
                                        entry["refractive_index"] = _extract_value(first_idx.value, 0.0)
                                except (IndexError, TypeError):
                                    pass

                        # Surface power (in air)
                        if hasattr(data, 'surface_powers') and data.surface_powers is not None:
                            sp = data.surface_powers
                            # Prefer "in_air" power
                            power_src = getattr(sp, 'in_air', None) or getattr(sp, 'as_situated', None)
                            if power_src is not None:
                                if hasattr(power_src, 'power') and power_src.power is not None:
                                    power_val = power_src.power
                                    if isinstance(power_val, dict):
                                        for v in power_val.values():
                                            entry["surface_power"] = _extract_value(v, 0.0)
                                            break
                                    else:
                                        entry["surface_power"] = _extract_value(power_val, 0.0)
                                elif hasattr(power_src, 'efl') and power_src.efl is not None:
                                    efl_val = power_src.efl
                                    if isinstance(efl_val, dict):
                                        for v in efl_val.values():
                                            efl_num = _extract_value(v, 0.0)
                                            if efl_num != 0.0:
                                                entry["surface_power"] = 1.0 / efl_num
                                            break
                                    else:
                                        efl_num = _extract_value(efl_val, 0.0)
                                        if efl_num != 0.0:
                                            entry["surface_power"] = 1.0 / efl_num

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
        """
        Get surface curvature map using ZosPy's Curvature analysis.

        This is a "dumb executor" - returns raw numpy array data only.
        Curvature map image rendering happens on Mac side.

        Note: System must be pre-loaded via load_zmx_file().

        Args:
            surface: Surface number to analyze (1-indexed)
            sampling: Grid sampling resolution (e.g., '65x65', '129x129')
            show_as: Display format ('Surface', 'Contour', 'GreyScale', etc.)
            data: Curvature data type ('TangentialCurvature', 'SagittalCurvature',
                  'X_Curvature', 'Y_Curvature')
            remove: Removal option ('None_', 'BaseROC', 'BestFitSphere')

        Returns:
            On success: {
                "success": True,
                "image": str (base64 numpy array),
                "image_format": "numpy_array",
                "array_shape": [h, w],
                "array_dtype": str,
                "min_curvature": float,
                "max_curvature": float,
                "mean_curvature": float,
                "surface_number": int,
                "data_type": str,
            }
            On error: {"success": False, "error": "..."}
        """
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

            if hasattr(curvature_result, 'data') and curvature_result.data is not None:
                curv_data = curvature_result.data
                # Curvature.run() returns AnalysisResult where .data is CurvatureResult
                # (or list[CurvatureResult]). The actual grid is in CurvatureResult.data.
                if hasattr(curv_data, 'data') and curv_data.data is not None:
                    curv_data = curv_data.data
                elif isinstance(curv_data, list) and len(curv_data) > 0:
                    curv_data = curv_data[0].data if hasattr(curv_data[0], 'data') else curv_data[0]
                if hasattr(curv_data, 'values'):
                    arr = np.array(curv_data.values, dtype=np.float64)
                else:
                    arr = np.array(curv_data, dtype=np.float64)

                if arr.ndim >= 2:
                    # Compute statistics on valid (non-NaN) values
                    valid_mask = ~np.isnan(arr)
                    if np.any(valid_mask):
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
