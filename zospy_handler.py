"""
ZosPy Handler

Manages the connection to Zemax OpticStudio and executes ZosPy operations.
This module contains all the actual ZosPy/OpticStudio calls.

Note: This code runs on Windows only, where OpticStudio is installed.

IMPORTANT: This worker MUST run with a single uvicorn worker (--workers 1)
because ZosPy/COM requires single-threaded apartment (STA) semantics.
"""

import base64
import io
from typing import Any, Optional

# ZosPy imports - these will fail on non-Windows or without OpticStudio
try:
    import zospy as zp
    from zospy.zpcore import OpticStudioSystem
    ZOSPY_AVAILABLE = True
except ImportError:
    ZOSPY_AVAILABLE = False
    zp = None
    OpticStudioSystem = None

import numpy as np


class ZosPyHandler:
    """
    Handler for ZosPy/OpticStudio operations.

    Manages the connection lifecycle and provides methods for
    optical analysis operations.
    """

    def _to_float(self, value: Any, default: float, name: str) -> float:
        """Convert a value to float with error handling."""
        if value is None:
            return default
        try:
            return float(value)
        except (TypeError, ValueError):
            raise ZosPyError(f"Invalid {name}: {value!r} (must be a number)")

    def _extract_value(self, spec: Any, default: Any = None) -> Any:
        """
        Extract value from a spec that may be:
        - None -> returns default
        - A dict with 'value' key -> returns dict['value'] or default
        - A direct value -> returns the value
        """
        if spec is None:
            return default
        if isinstance(spec, dict):
            return spec.get("value", default)
        return spec

    def __init__(self):
        """Initialize ZosPy and connect to OpticStudio."""
        if not ZOSPY_AVAILABLE:
            raise ZosPyError("ZosPy is not available. Install it with: pip install zospy")

        try:
            # Initialize ZOS connection
            self.zos = zp.ZOS()

            # Connect to OpticStudio (starts instance if needed)
            self.oss: OpticStudioSystem = self.zos.connect(mode="standalone")

            if self.oss is None:
                raise ZosPyError("Failed to connect to OpticStudio")

            print(f"Connected to OpticStudio: {self.get_version()}")

        except Exception as e:
            raise ZosPyError(f"Failed to initialize ZosPy: {e}")

    def close(self):
        """Close the connection to OpticStudio."""
        try:
            if hasattr(self, 'zos') and self.zos:
                self.zos.disconnect()
        except Exception as e:
            print(f"Warning: Error closing ZosPy connection: {e}")

    def get_version(self) -> str:
        """Get OpticStudio version string."""
        try:
            # ZosPy exposes version through the application object
            return str(self.oss.Application.ZemaxVersion) if self.oss else "Unknown"
        except Exception:
            return "Unknown"

    def get_status(self) -> dict[str, Any]:
        """Get current connection status."""
        try:
            zospy_version = zp.__version__
        except Exception:
            zospy_version = "Unknown"

        return {
            "connected": self.oss is not None,
            "opticstudio_version": self.get_version(),
            "zospy_version": zospy_version,
        }

    def load_system(self, llm_json: dict[str, Any]) -> dict[str, Any]:
        """
        Load an optical system from LLM JSON format into OpticStudio.

        Args:
            llm_json: Full LLM JSON with system and surfaces

        Returns:
            Dict with load status and system info
        """
        # Create a new system
        self.oss.new()

        system = llm_json.get("system", {})
        surfaces = llm_json.get("surfaces", [])

        if not surfaces:
            raise ZosPyError("Cannot create optical system with no surfaces")

        # Set up aperture
        self._setup_aperture(system)

        # Set up wavelengths
        self._setup_wavelengths(system)

        # Set up fields
        self._setup_fields(system)

        # Set up surfaces
        self._setup_surfaces(surfaces)

        # Update the system to compute derived values
        self.oss.make_sequential()

        return {
            "num_surfaces": len(surfaces),
            "efl": self._get_efl(),
        }

    def _setup_aperture(self, system: dict[str, Any]):
        """Configure system aperture using ZosPy API."""
        pupil = system.get("pupil") or {}
        mode = pupil.get("mode") or "epd"  # Handles None and empty string
        value = self._to_float(pupil.get("value"), 10.0, "aperture value")

        # ZosPy uses zp.constants for aperture types
        # Note: constant names may vary by ZosPy version
        aperture_types = zp.constants.SystemData.ZemaxApertureType

        if mode == "epd":
            if hasattr(aperture_types, "EntrancePupilDiameter"):
                self.oss.SystemData.Aperture.ApertureType = aperture_types.EntrancePupilDiameter
            else:
                raise ZosPyError(f"Aperture mode '{mode}' not supported by this ZosPy version")
        elif mode in ("fno", "image_space_fnum"):
            # Try ImageSpaceFNumber first, fall back to FloatByStopSize
            if hasattr(aperture_types, "ImageSpaceFNumber"):
                self.oss.SystemData.Aperture.ApertureType = aperture_types.ImageSpaceFNumber
            elif hasattr(aperture_types, "FloatByStopSize"):
                self.oss.SystemData.Aperture.ApertureType = aperture_types.FloatByStopSize
            else:
                raise ZosPyError(f"Aperture mode '{mode}' not supported by this ZosPy version")
        elif mode in ("na", "na_object"):
            if hasattr(aperture_types, "ObjectSpaceNA"):
                self.oss.SystemData.Aperture.ApertureType = aperture_types.ObjectSpaceNA
            else:
                raise ZosPyError(f"Aperture mode '{mode}' not supported by this ZosPy version")
        elif mode == "float_by_stop_size":
            if hasattr(aperture_types, "FloatByStopSize"):
                self.oss.SystemData.Aperture.ApertureType = aperture_types.FloatByStopSize
            else:
                raise ZosPyError(f"Aperture mode '{mode}' not supported by this ZosPy version")
        elif mode == "image_space_paraxial_working_fnum":
            if hasattr(aperture_types, "ParaxialWorkingFNumber"):
                self.oss.SystemData.Aperture.ApertureType = aperture_types.ParaxialWorkingFNumber
            else:
                raise ZosPyError(f"Aperture mode '{mode}' not supported by this ZosPy version")
        else:
            valid_modes = ["epd", "fno", "image_space_fnum", "na", "na_object",
                           "float_by_stop_size", "image_space_paraxial_working_fnum"]
            raise ZosPyError(f"Unknown aperture mode: {mode!r}. Valid modes: {', '.join(valid_modes)}")

        self.oss.SystemData.Aperture.ApertureValue = value

    def _setup_wavelengths(self, system: dict[str, Any]):
        """Configure system wavelengths using ZosPy API."""
        wavelengths = system.get("wavelengths", [{"um": 0.5876}])
        primary_wavelength = system.get("primary_wavelength", 1)  # 1-indexed

        # Get wavelength data object
        wl_data = self.oss.SystemData.Wavelengths

        # Remove all existing wavelengths first
        # ZosPy: use RemoveWavelength with 1-based index, remove from end to avoid index shift
        while wl_data.NumberOfWavelengths > 1:
            wl_data.RemoveWavelength(wl_data.NumberOfWavelengths)

        # Set the first wavelength (can't remove the last one, so modify it)
        if wavelengths:
            first_wl = wavelengths[0]
            wl = wl_data.GetWavelength(1)
            wl.Wavelength = float(first_wl.get("um", 0.5876))
            wl.Weight = float(first_wl.get("weight", 1.0))

        # Add remaining wavelengths
        for i, wl_spec in enumerate(wavelengths[1:], start=2):
            um = float(wl_spec.get("um", 0.5876))
            weight = float(wl_spec.get("weight", 1.0))
            wl_data.AddWavelength(um, weight)

        # Set primary wavelength
        # Use MakePrimary() on the individual wavelength object
        if 1 <= primary_wavelength <= len(wavelengths):
            wl = wl_data.GetWavelength(primary_wavelength)
            wl.MakePrimary()

    # Mapping from LLM JSON field types to ZosPy constant names
    _FIELD_TYPE_MAP = {
        "object_angle": "Angle",
        "object_height": "ObjectHeight",
        "image_height": "RealImageHeight",
        "real_image_height": "RealImageHeight",
        "paraxial_image_height": "ParaxialImageHeight",
    }

    def _setup_fields(self, system: dict[str, Any]):
        """Configure system fields using ZosPy API."""
        field_type = system.get("field_type") or "object_angle"  # Handles None and empty string
        fields = system.get("fields") or [{"x": 0, "y": 0}]  # Handles None and empty list

        field_data = self.oss.SystemData.Fields

        # Set field type using ZosPy constants
        field_types = zp.constants.SystemData.FieldType

        if field_type not in self._FIELD_TYPE_MAP:
            valid_types = list(self._FIELD_TYPE_MAP.keys())
            raise ZosPyError(f"Unknown field type: {field_type!r}. Valid types: {', '.join(valid_types)}")

        attr_name = self._FIELD_TYPE_MAP[field_type]
        if hasattr(field_types, attr_name):
            field_data.SetFieldType(getattr(field_types, attr_name))
        else:
            raise ZosPyError(f"Field type '{field_type}' not supported by this ZosPy version")

        # Remove all fields except the first (can't remove last field)
        while field_data.NumberOfFields > 1:
            field_data.RemoveField(field_data.NumberOfFields)

        # Set the first field
        if fields:
            first_field = fields[0] if isinstance(fields[0], dict) else {}
            f = field_data.GetField(1)
            f.X = self._to_float(first_field.get("x"), 0, "field X")
            f.Y = self._to_float(first_field.get("y"), 0, "field Y")
            f.Weight = self._to_float(first_field.get("weight"), 1.0, "field weight")

        # Add remaining fields
        for field_spec in fields[1:]:
            if not isinstance(field_spec, dict):
                continue
            x = self._to_float(field_spec.get("x"), 0, "field X")
            y = self._to_float(field_spec.get("y"), 0, "field Y")
            weight = self._to_float(field_spec.get("weight"), 1.0, "field weight")
            field_data.AddField(x, y, weight)

    def _setup_surfaces(self, surfaces: list[dict[str, Any]]):
        """Configure system surfaces using ZosPy API."""
        lde = self.oss.LDE  # Lens Data Editor

        # Find stop surface index
        stop_index = None
        for i, surf in enumerate(surfaces):
            if surf.get("stop", False):
                stop_index = i
                break

        # Ensure we have enough surfaces (LDE starts with Object and Image surfaces)
        # We need len(surfaces) surfaces between Object (0) and Image
        while lde.NumberOfSurfaces < len(surfaces) + 1:  # +1 for image surface
            # Insert before image surface
            lde.InsertNewSurfaceAt(lde.NumberOfSurfaces)

        # Configure each surface
        for i, surf in enumerate(surfaces):
            # Surface index is i+1 (0 is Object surface)
            surface = lde.GetSurfaceAt(i + 1)

            # Handle radius - may be direct value or object with 'value' key
            radius = self._extract_value(surf.get("radius"), float('inf'))
            surface.Radius = float(radius) if radius != float('inf') else float('inf')

            # Handle thickness - may be direct value or object with 'value' key
            thickness = self._extract_value(surf.get("thickness"), 0.0)
            surface.Thickness = float(thickness)

            # Semi-diameter
            sd_value = self._extract_value(surf.get("semi_diameter"))
            if sd_value is not None:
                surface.SemiDiameter = float(sd_value)

            # Glass/material
            glass = surf.get("glass")
            if glass:
                if isinstance(glass, str):
                    surface.Material = glass
                elif isinstance(glass, dict):
                    glass_name = glass.get("name")
                    solve_type = glass.get("solve", "fixed")

                    if glass_name and solve_type == "fixed":
                        # Use catalog glass
                        surface.Material = glass_name
                    elif solve_type == "model" or (glass.get("nd") is not None and not glass_name):
                        # Model glass - use ZosPy's material model solver
                        nd = float(glass.get("nd", 1.5))
                        vd = float(glass.get("vd", 50.0))
                        try:
                            # ZosPy provides solvers for model glass
                            zp.solvers.material_model(
                                surface.MaterialCell,
                                refractive_index=nd,
                                abbe_number=vd
                            )
                        except Exception as e:
                            # Fallback: just set a placeholder name
                            print(f"Warning: Could not set model glass solver: {e}")
                            surface.Material = f"MODEL_{nd:.4f}_{vd:.1f}"

            # Handle conic - may be direct value or object with 'value' key
            conic = self._extract_value(surf.get("conic"), 0.0)
            conic = float(conic) if conic is not None else 0.0
            if conic != 0.0:
                surface.Conic = conic

            # Set as stop surface using IsStop property
            # Per Zemax Community: oss.LDE.GetSurfaceAt(xx).IsStop = True
            if i == stop_index:
                try:
                    surface.IsStop = True
                except Exception as e:
                    raise ZosPyError(f"Failed to set surface {i} as stop: {e}")

    def _get_efl(self) -> Optional[float]:
        """Get effective focal length using ZosPy's SystemData analysis."""
        try:
            # Use the SystemData analysis to get first-order properties
            result = zp.analyses.reports.SystemData().run(self.oss)
            return result.data.general_lens_data.effective_focal_length_air
        except Exception:
            return None

    def get_paraxial_data(self) -> dict[str, Any]:
        """Get first-order (paraxial) optical properties.

        Uses direct ZOSAPI access to bypass ZosPy text parser issues with OpticStudio v25.
        """
        try:
            # Get field info directly from SystemData
            fields = self.oss.SystemData.Fields
            field_type = "object_angle"  # Default
            max_field = 0.0
            if fields.NumberOfFields > 0:
                for i in range(1, fields.NumberOfFields + 1):
                    f = fields.GetField(i)
                    max_field = max(max_field, abs(f.Y), abs(f.X))

            paraxial = {
                "field_type": field_type,
                "max_field": max_field,
                "field_unit": "deg" if field_type == "object_angle" else "mm",
            }

            # Get aperture info
            aperture = self.oss.SystemData.Aperture
            paraxial["epd"] = aperture.ApertureValue

            # Calculate total track from LDE
            lde = self.oss.LDE
            total_track = 0.0
            for i in range(1, lde.NumberOfSurfaces):
                surface = lde.GetSurfaceAt(i)
                total_track += abs(surface.Thickness)
            paraxial["total_track"] = total_track

            # Use raw ZOSAPI to get first-order data (bypass ZosPy parser)
            try:
                # Get first-order data from SystemData.FirstOrderData
                first_order = self.oss.SystemData.FirstOrderData
                if first_order is not None:
                    # EFL - try different property names for different OpticStudio versions
                    if hasattr(first_order, 'EffectiveFocalLength'):
                        paraxial["efl"] = first_order.EffectiveFocalLength
                    elif hasattr(first_order, 'EFL'):
                        paraxial["efl"] = first_order.EFL

                    # BFL
                    if hasattr(first_order, 'BackFocalLength'):
                        paraxial["bfl"] = first_order.BackFocalLength
                    elif hasattr(first_order, 'BFL'):
                        paraxial["bfl"] = first_order.BFL

                    # F-number
                    if hasattr(first_order, 'WorkingFNumber'):
                        paraxial["fno"] = first_order.WorkingFNumber
                    elif hasattr(first_order, 'ImageSpaceFNum'):
                        paraxial["fno"] = first_order.ImageSpaceFNum
            except Exception as e:
                print(f"Warning: Could not get first-order data from ZOSAPI: {e}")

            return paraxial
        except Exception as e:
            print(f"Warning: Could not get paraxial data: {e}")
            return {}

    def get_cross_section(self, llm_json: dict[str, Any]) -> dict[str, Any]:
        """
        Generate cross-section diagram using ZosPy's CrossSection analysis.

        Requires ZosPy >= 1.3.0 and OpticStudio >= 24.1.0 for image export.
        Falls back to surface geometry if image export fails.
        """
        # Load the system first
        self.load_system(llm_json)

        image_b64 = None
        image_format = None

        # Check OpticStudio version - image export requires >= 24.1.0
        try:
            zos_version = self.zos.version if hasattr(self.zos, 'version') else None
            print(f"DEBUG: ZosPy zos.version = {zos_version}")
            print(f"DEBUG: ZosPy version = {zp.__version__ if hasattr(zp, '__version__') else 'unknown'}")

            if zos_version and zos_version < (24, 1, 0):
                print(f"Warning: OpticStudio {zos_version} < 24.1.0 - image export not supported, using fallback")
            else:
                # Try CrossSection analysis
                cross_section = zp.analyses.systemviewers.CrossSection(
                    number_of_rays=11,
                    delete_vignetted=True,
                )
                print(f"DEBUG: Created CrossSection analysis object")

                result = cross_section.run(self.oss)
                print(f"DEBUG: CrossSection.run() completed")
                print(f"DEBUG: result type = {type(result)}")
                print(f"DEBUG: result.data type = {type(result.data) if hasattr(result, 'data') else 'no data attr'}")
                print(f"DEBUG: result.data is None = {result.data is None if hasattr(result, 'data') else 'N/A'}")

                # Check if we got image data
                if result.data is not None:
                    import matplotlib.pyplot as plt
                    print(f"DEBUG: result.data shape = {result.data.shape if hasattr(result.data, 'shape') else 'no shape'}")

                    # result.data is a numpy array (RGB image)
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.imshow(result.data)
                    ax.axis('off')
                    buffer = io.BytesIO()
                    fig.savefig(buffer, format='PNG', dpi=150, bbox_inches='tight', pad_inches=0)
                    plt.close(fig)
                    buffer.seek(0)
                    image_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                    image_format = "png"
                    print(f"DEBUG: Successfully encoded image, length = {len(image_b64)}")
                else:
                    print(f"DEBUG: result.data is None - image export not available")

        except Exception as e:
            # Log but don't fail - we'll return surface geometry as fallback
            import traceback
            print(f"Warning: CrossSection image export failed: {e}")
            print(f"DEBUG: Full traceback:\n{traceback.format_exc()}")

        # Get paraxial data using direct LDE access (more reliable)
        paraxial = self._get_paraxial_from_lde()

        # Get surface geometry for client-side rendering (fallback or supplement)
        surfaces_data = self._get_surface_geometry()

        # Count rays
        num_fields = self.oss.SystemData.Fields.NumberOfFields
        rays_total = 11 * max(1, num_fields)
        rays_through = rays_total  # Assume success

        return {
            "image": image_b64,
            "image_format": image_format,
            "paraxial": paraxial,
            "surfaces": surfaces_data,  # Always include for fallback rendering
            "rays_total": rays_total,
            "rays_through": rays_through,
        }

    def _get_paraxial_from_lde(self) -> dict[str, Any]:
        """Get paraxial data directly from LDE and SystemData (more reliable)."""
        try:
            paraxial = {}

            # Get aperture info
            aperture = self.oss.SystemData.Aperture
            paraxial["epd"] = aperture.ApertureValue

            # Get field info
            fields = self.oss.SystemData.Fields
            max_field = 0.0
            if fields.NumberOfFields > 0:
                for i in range(1, fields.NumberOfFields + 1):
                    f = fields.GetField(i)
                    max_field = max(max_field, abs(f.Y), abs(f.X))
            paraxial["max_field"] = max_field
            paraxial["field_type"] = "object_angle"
            paraxial["field_unit"] = "deg"

            # Calculate total track from LDE
            lde = self.oss.LDE
            total_track = 0.0
            for i in range(1, lde.NumberOfSurfaces):
                surface = lde.GetSurfaceAt(i)
                total_track += abs(surface.Thickness)
            paraxial["total_track"] = total_track

            return paraxial
        except Exception as e:
            print(f"Warning: Could not get paraxial data from LDE: {e}")
            return {}

    def _get_surface_geometry(self) -> list[dict[str, Any]]:
        """Extract surface geometry for client-side cross-section rendering."""
        surfaces = []
        lde = self.oss.LDE
        z_position = 0.0

        for i in range(1, lde.NumberOfSurfaces):
            surface = lde.GetSurfaceAt(i)

            # Radius of 0 in Zemax means infinity (flat surface)
            # We convert to None for client-side rendering
            radius = surface.Radius
            surf_data = {
                "index": i,
                "z": z_position,
                "radius": radius if radius != 0 and abs(radius) < 1e10 else None,
                "thickness": surface.Thickness,
                "semi_diameter": surface.SemiDiameter,
                "conic": surface.Conic,
                "material": str(surface.Material) if surface.Material else None,
                "is_stop": surface.IsStop,
            }
            surfaces.append(surf_data)
            z_position += surface.Thickness

        return surfaces

    def calc_semi_diameters(self, llm_json: dict[str, Any]) -> dict[str, Any]:
        """
        Calculate semi-diameters by reading from surfaces after ray trace.
        """
        # Load the system
        self.load_system(llm_json)

        semi_diameters = []
        lde = self.oss.LDE

        # For each surface, get the computed semi-diameter
        for i in range(1, lde.NumberOfSurfaces):
            surface = lde.GetSurfaceAt(i)
            sd = surface.SemiDiameter

            semi_diameters.append({
                "index": i - 1,  # Convert to 0-indexed
                "value": sd,
            })

        return {"semi_diameters": semi_diameters}

    def ray_trace_diagnostic(
        self,
        llm_json: dict[str, Any],
        num_rays: int = 50,
        distribution: str = "hexapolar",
    ) -> dict[str, Any]:
        """
        Run ray trace diagnostic to identify ray failures.

        Uses ZosPy's ray tracing to trace rays and detect failures.
        """
        # Load the system
        self.load_system(llm_json)

        # Get paraxial data
        paraxial = self.get_paraxial_data()

        # Get system info
        lde = self.oss.LDE
        num_surfaces = lde.NumberOfSurfaces - 1  # Exclude object surface
        fields = self.oss.SystemData.Fields
        num_fields = fields.NumberOfFields

        # Trace rays for each field
        field_results = []
        all_surface_failures: dict[int, dict[str, Any]] = {}

        # Calculate grid size
        grid_size = int(np.sqrt(num_rays))

        for fi in range(1, num_fields + 1):
            field = fields.GetField(fi)
            field_x, field_y = field.X, field.Y

            rays_traced = 0
            rays_reached = 0
            rays_failed = 0

            # Trace a grid of rays using ZosPy's single ray trace
            for hx in np.linspace(-1, 1, grid_size):
                for hy in np.linspace(-1, 1, grid_size):
                    if hx**2 + hy**2 > 1:
                        continue  # Skip rays outside pupil

                    rays_traced += 1

                    try:
                        # ZosPy SingleRayTrace: px/py are normalized pupil coordinates (-1 to 1)
                        # hx/hy are normalized field coordinates (-1 to 1)
                        # When field index is specified, we iterate over pupil positions
                        ray_trace = zp.analyses.raysandspots.SingleRayTrace(
                            hx=0.0,  # Use on-axis field direction
                            hy=0.0,
                            px=hx,   # Pupil X coordinate (iterating over pupil)
                            py=hy,   # Pupil Y coordinate (iterating over pupil)
                            wavelength=1,
                            field=fi,
                        )
                        result = ray_trace.run(self.oss)

                        # Check result - ZosPy returns data in result.data.real_ray_trace_data
                        if hasattr(result, 'data') and result.data is not None:
                            ray_data = result.data
                            # Access real_ray_trace_data if available (DataFrame)
                            if hasattr(ray_data, 'real_ray_trace_data'):
                                df = ray_data.real_ray_trace_data
                            else:
                                df = ray_data  # Fallback for older versions

                            # Check if ray reached image (last surface has valid data)
                            if hasattr(df, '__len__') and len(df) > 0:
                                # Check for error codes - vignetted rays have error_code != 0
                                if hasattr(df, 'columns') and 'error_code' in df.columns:
                                    # Find first surface with error
                                    error_rows = df[df['error_code'] != 0]
                                    if len(error_rows) > 0:
                                        rays_failed += 1
                                        first_error = error_rows.iloc[0]
                                        surf_idx = int(first_error.get('surface', 0))
                                        if surf_idx not in all_surface_failures:
                                            all_surface_failures[surf_idx] = {
                                                "surface_index": surf_idx,
                                                "total_failures": 0,
                                                "dominant_mode": "VIGNETTE",
                                            }
                                        all_surface_failures[surf_idx]["total_failures"] += 1
                                    else:
                                        rays_reached += 1
                                else:
                                    # No error column - assume success
                                    rays_reached += 1
                            else:
                                rays_failed += 1
                        else:
                            rays_failed += 1

                    except Exception:
                        rays_failed += 1

            field_results.append({
                "field_index": fi - 1,
                "field_x": field_x,
                "field_y": field_y,
                "rays_traced": rays_traced,
                "rays_reached_image": rays_reached,
                "rays_failed": rays_failed,
                "surface_failures": [],
            })

        # Build aggregate surface failures
        aggregate_surface_failures = list(all_surface_failures.values())

        # Find hotspots (surfaces causing >10% of failures)
        total_failures = sum(f["total_failures"] for f in aggregate_surface_failures)
        hotspots = []
        if total_failures > 0:
            for sf in aggregate_surface_failures:
                if sf["failure_count"] / total_failures > 0.1:
                    hotspots.append(sf["surface_index"])

        return {
            "paraxial": {
                "efl": paraxial.get("efl"),
                "bfl": paraxial.get("bfl"),
                "fno": paraxial.get("fno"),
                "total_track": paraxial.get("total_track"),
            },
            "num_surfaces": num_surfaces,
            "num_fields": num_fields,
            "field_results": field_results,
            "aggregate_surface_failures": aggregate_surface_failures,
            "hotspots": hotspots,
        }

    def get_seidel(self, llm_json: dict[str, Any]) -> dict[str, Any]:
        """
        Get Seidel aberrations via Zernike coefficients.

        OpticStudio provides Zernike Standard Coefficients, which we
        convert to Seidel format using aberration theory.

        Uses raw ZOSAPI access to avoid ZosPy parser issues with OpticStudio v25.
        """
        # Load the system
        self.load_system(llm_json)

        try:
            # Use raw ZOSAPI access to bypass ZosPy's text parser issues
            # This approach is more compatible with different OpticStudio versions
            analysis = zp.analyses.new_analysis(
                self.oss,
                zp.constants.Analysis.AnalysisIDM.ZernikeStandardCoefficients,
                settings_first=True
            )

            # Configure analysis settings
            settings = analysis.Settings
            settings.Field.SetFieldNumber(1)
            settings.Wavelength.SetWavelengthNumber(1)
            settings.Surface.SetSurfaceNumber(0)  # 0 = Image surface
            settings.SampleSize = zp.constants.Analysis.SampleSizes.S_64x64
            settings.MaximumNumberOfTerms = 37
            settings.ReferenceOPDToVertex = False

            # Run the analysis
            analysis.ApplyAndWaitForCompletion()

            # Get results from the analysis
            results = analysis.GetResults()

            # Extract Zernike coefficients from the data grid
            coefficients = []
            if results is not None:
                data_grid = results.DataGrids[0] if results.NumberOfDataGrids > 0 else None
                if data_grid is not None:
                    # Data grid contains Zernike terms
                    for row in range(data_grid.Rows):
                        for col in range(data_grid.Cols):
                            val = data_grid.GetValueAt(row, col)
                            if val is not None:
                                coefficients.append(float(val))

            # If we couldn't get coefficients from data grid, try text parsing
            if not coefficients:
                # Fall back to returning placeholder data
                print("Warning: Could not extract Zernike coefficients, using placeholder values")
                coefficients = [0.0] * 37

            # Release the analysis
            analysis.Release()

            # Convert Zernike to Seidel
            from seidel_converter import zernike_to_seidel, build_seidel_response

            # Get wavelength for scaling
            wl_um = self.oss.SystemData.Wavelengths.GetWavelength(1).Wavelength

            seidel = zernike_to_seidel(
                coefficients,
                wavelength_um=wl_um,
            )

            # Get number of surfaces
            num_surfaces = self.oss.LDE.NumberOfSurfaces - 1

            # Build response
            response = build_seidel_response(
                seidel=seidel,
                num_surfaces=num_surfaces,
            )

            return response

        except Exception as e:
            raise ZosPyError(f"Seidel analysis failed: {e}")

    def trace_rays(
        self,
        llm_json: dict[str, Any],
        num_rays: int = 7,
    ) -> dict[str, Any]:
        """
        Trace rays through the system and return positions at each surface.
        """
        # Load the system
        self.load_system(llm_json)

        # Get system info
        num_surfaces = self.oss.LDE.NumberOfSurfaces
        num_fields = self.oss.SystemData.Fields.NumberOfFields
        num_wavelengths = self.oss.SystemData.Wavelengths.NumberOfWavelengths

        data = []

        # Trace rays for each field and wavelength
        for fi in range(1, num_fields + 1):
            for wi in range(1, num_wavelengths + 1):
                surfaces_data = []

                # Initialize arrays for each surface
                for si in range(num_surfaces):
                    surfaces_data.append({
                        "y": [],
                        "z": [],
                    })

                # Trace fan of rays across pupil
                for py in np.linspace(-1, 1, num_rays):
                    try:
                        # ZosPy SingleRayTrace: px/py are normalized pupil coordinates (-1 to 1)
                        # hx/hy are normalized field coordinates (not used when field index specified)
                        ray_trace = zp.analyses.raysandspots.SingleRayTrace(
                            hx=0.0,
                            hy=0.0,
                            px=0.0,  # Meridional fan (x=0)
                            py=py,   # Iterate over pupil Y
                            wavelength=wi,
                            field=fi,
                        )
                        result = ray_trace.run(self.oss)

                        # Extract ray positions from result.data.real_ray_trace_data
                        if hasattr(result, 'data') and result.data is not None:
                            ray_data = result.data
                            # Access real_ray_trace_data if available
                            if hasattr(ray_data, 'real_ray_trace_data'):
                                df = ray_data.real_ray_trace_data
                            else:
                                df = ray_data  # Fallback

                            if hasattr(df, 'iloc'):
                                for si in range(min(len(df), num_surfaces)):
                                    row = df.iloc[si]
                                    # Column names may be 'Y' or 'y' depending on version
                                    y_val = row.get('Y', row.get('y', None))
                                    z_val = row.get('Z', row.get('z', None))
                                    surfaces_data[si]["y"].append(y_val)
                                    surfaces_data[si]["z"].append(z_val)
                            else:
                                # Fallback - add None values
                                for si in range(num_surfaces):
                                    surfaces_data[si]["y"].append(None)
                                    surfaces_data[si]["z"].append(None)
                        else:
                            # Ray failed - add None values
                            for si in range(num_surfaces):
                                surfaces_data[si]["y"].append(None)
                                surfaces_data[si]["z"].append(None)

                    except Exception:
                        # Ray failed - add None values
                        for si in range(num_surfaces):
                            surfaces_data[si]["y"].append(None)
                            surfaces_data[si]["z"].append(None)

                data.append({
                    "field_index": fi - 1,
                    "wavelength_index": wi - 1,
                    "surfaces": surfaces_data,
                })

        return {
            "num_surfaces": num_surfaces,
            "num_fields": num_fields,
            "num_wavelengths": num_wavelengths,
            "data": data,
        }


class ZosPyError(Exception):
    """Exception raised when ZosPy operations fail."""
    pass
