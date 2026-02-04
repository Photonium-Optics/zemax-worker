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
        pupil = system.get("pupil", {})
        mode = pupil.get("mode", "epd")
        value = pupil.get("value", 10.0)

        # ZosPy uses zp.constants for aperture types
        if mode == "epd":
            self.oss.SystemData.Aperture.ApertureType = zp.constants.SystemData.ZemaxApertureType.EntrancePupilDiameter
        elif mode in ("fno", "image_space_fnum"):
            self.oss.SystemData.Aperture.ApertureType = zp.constants.SystemData.ZemaxApertureType.ImageSpaceFNumber
        elif mode in ("na", "na_object"):
            self.oss.SystemData.Aperture.ApertureType = zp.constants.SystemData.ZemaxApertureType.ObjectSpaceNA

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
            wl.Wavelength = first_wl.get("um", 0.5876)
            wl.Weight = first_wl.get("weight", 1.0)

        # Add remaining wavelengths
        for i, wl_spec in enumerate(wavelengths[1:], start=2):
            um = wl_spec.get("um", 0.5876)
            weight = wl_spec.get("weight", 1.0)
            wl_data.AddWavelength(um, weight)

        # Set primary wavelength
        if 1 <= primary_wavelength <= len(wavelengths):
            wl_data.SelectWavelength(primary_wavelength)

    def _setup_fields(self, system: dict[str, Any]):
        """Configure system fields using ZosPy API."""
        field_type = system.get("field_type", "object_angle")
        fields = system.get("fields", [{"x": 0, "y": 0}])

        field_data = self.oss.SystemData.Fields

        # Set field type using ZosPy constants
        if field_type == "object_angle":
            field_data.SetFieldType(zp.constants.SystemData.FieldType.Angle)
        elif field_type == "object_height":
            field_data.SetFieldType(zp.constants.SystemData.FieldType.ObjectHeight)
        elif field_type in ("image_height", "real_image_height"):
            field_data.SetFieldType(zp.constants.SystemData.FieldType.RealImageHeight)

        # Remove all fields except the first (can't remove last field)
        while field_data.NumberOfFields > 1:
            field_data.RemoveField(field_data.NumberOfFields)

        # Set the first field
        if fields:
            first_field = fields[0]
            f = field_data.GetField(1)
            f.X = first_field.get("x", 0)
            f.Y = first_field.get("y", 0)
            f.Weight = first_field.get("weight", 1.0)

        # Add remaining fields
        for field_spec in fields[1:]:
            x = field_spec.get("x", 0)
            y = field_spec.get("y", 0)
            weight = field_spec.get("weight", 1.0)
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
            radius_spec = surf.get("radius")
            if radius_spec is None:
                radius = float('inf')  # Flat surface
            elif isinstance(radius_spec, dict):
                radius = radius_spec.get("value", float('inf'))
            else:
                radius = radius_spec
            surface.Radius = radius

            # Handle thickness - may be direct value or object with 'value' key
            thickness_spec = surf.get("thickness", 0.0)
            if isinstance(thickness_spec, dict):
                thickness = thickness_spec.get("value", 0.0)
            else:
                thickness = thickness_spec
            surface.Thickness = thickness

            # Semi-diameter
            sd_spec = surf.get("semi_diameter")
            if sd_spec is not None:
                if isinstance(sd_spec, dict):
                    sd_value = sd_spec.get("value")
                else:
                    sd_value = sd_spec
                if sd_value is not None:
                    surface.SemiDiameter = sd_value

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
                        nd = glass.get("nd", 1.5)
                        vd = glass.get("vd", 50.0)
                        try:
                            # ZosPy provides solvers for model glass
                            zp.solvers.material_model(
                                surface.MaterialCell,
                                refractive_index=nd,
                                abbe_number=vd
                            )
                        except Exception:
                            # Fallback: just set a placeholder name
                            surface.Material = f"MODEL_{nd:.4f}_{vd:.1f}"

            # Handle conic - may be direct value or object with 'value' key
            conic_spec = surf.get("conic", 0.0)
            if isinstance(conic_spec, dict):
                conic = conic_spec.get("value", 0.0)
            else:
                conic = conic_spec
            if conic != 0.0:
                surface.Conic = conic

            # Set as stop surface
            if i == stop_index:
                surface.MakeThisSurfaceTheStop()

    def _get_efl(self) -> Optional[float]:
        """Get effective focal length using ZosPy's SystemData analysis."""
        try:
            # Use the SystemData analysis to get first-order properties
            result = zp.analyses.reports.SystemData().run(self.oss)
            return result.data.general_lens_data.effective_focal_length_air
        except Exception:
            return None

    def get_paraxial_data(self) -> dict[str, Any]:
        """Get first-order (paraxial) optical properties using ZosPy API."""
        try:
            # Use SystemData analysis for paraxial properties
            result = zp.analyses.reports.SystemData().run(self.oss)
            gld = result.data.general_lens_data

            # Get field info
            fields = self.oss.SystemData.Fields
            field_type = "object_angle"  # Default
            max_field = 0.0
            if fields.NumberOfFields > 0:
                for i in range(1, fields.NumberOfFields + 1):
                    f = fields.GetField(i)
                    max_field = max(max_field, abs(f.Y), abs(f.X))

            return {
                "efl": gld.effective_focal_length_air,
                "bfl": gld.back_focal_length,
                "ffl": gld.front_focal_length,
                "fno": gld.working_f_number,
                "total_track": gld.total_track,
                "epd": gld.entrance_pupil_diameter,
                "exp": gld.exit_pupil_diameter,
                "epl": gld.entrance_pupil_position,
                "exl": gld.exit_pupil_position,
                "field_type": field_type,
                "max_field": max_field,
                "field_unit": "deg" if field_type == "object_angle" else "mm",
            }
        except Exception as e:
            print(f"Warning: Could not get paraxial data: {e}")
            return {}

    def get_cross_section(self, llm_json: dict[str, Any]) -> dict[str, Any]:
        """
        Generate cross-section diagram using ZosPy's CrossSection analysis.
        """
        # Load the system first
        self.load_system(llm_json)

        try:
            # Run CrossSection analysis - ZosPy pattern: ClassName(params).run(oss)
            cross_section = zp.analyses.systemviewers.CrossSection(
                number_of_rays=11,
                y_stretch=1.0,
                delete_vignetted=True,
            )
            result = cross_section.run(self.oss)

            # ZosPy returns the image in result.figure (matplotlib figure)
            # or result.data depending on version
            image_b64 = None
            if hasattr(result, 'figure') and result.figure is not None:
                # result.figure is a matplotlib figure - save to PNG
                buffer = io.BytesIO()
                result.figure.savefig(buffer, format='PNG', dpi=150, bbox_inches='tight')
                buffer.seek(0)
                image_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            elif hasattr(result, 'data') and result.data is not None:
                # Fallback: result.data might be PIL Image
                buffer = io.BytesIO()
                result.data.save(buffer, format='PNG')
                image_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

            # Get paraxial data
            paraxial = self.get_paraxial_data()

            # Count rays (from analysis settings)
            rays_total = 11 * 3  # rays_per_field * typical_fields
            rays_through = rays_total  # Assume success unless we trace

            return {
                "image": image_b64,
                "image_format": "png",
                "paraxial": paraxial,
                "rays_total": rays_total,
                "rays_through": rays_through,
            }

        except Exception as e:
            raise ZosPyError(f"CrossSection analysis failed: {e}")

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
                        # ZosPy single ray trace - pattern: ClassName(params).run(oss)
                        ray_trace = zp.analyses.raysandspots.SingleRayTrace(
                            Hx=hx,
                            Hy=hy,
                            Px=0.0,
                            Py=0.0,
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
                                                "failure_count": 0,
                                                "failure_mode": "VIGNETTE",
                                            }
                                        all_surface_failures[surf_idx]["failure_count"] += 1
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
        total_failures = sum(f["failure_count"] for f in aggregate_surface_failures)
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
        """
        # Load the system
        self.load_system(llm_json)

        try:
            # Run Zernike analysis - ZosPy pattern: ClassName(params).run(oss)
            zernike_analysis = zp.analyses.wavefront.ZernikeStandardCoefficients(
                sampling='64x64',
                maximum_term=37,
                wavelength=1,
                field=1,
            )
            result = zernike_analysis.run(self.oss)

            # Extract Zernike coefficients from result.data
            # ZosPy returns coefficients as objects with .value attribute
            coefficients = []
            if hasattr(result, 'data') and result.data is not None:
                coeff_data = result.data
                # Handle different ZosPy data structures
                if hasattr(coeff_data, 'coefficients'):
                    # Newer ZosPy: object with coefficients list/dict
                    raw_coeffs = coeff_data.coefficients
                    if isinstance(raw_coeffs, dict):
                        # Dict keyed by term number - coefficients have .value
                        max_term = max(raw_coeffs.keys()) if raw_coeffs else 0
                        for i in range(1, max_term + 1):
                            coeff = raw_coeffs.get(i)
                            if coeff is not None and hasattr(coeff, 'value'):
                                coefficients.append(coeff.value)
                            elif coeff is not None:
                                coefficients.append(float(coeff))
                            else:
                                coefficients.append(0.0)
                    elif hasattr(raw_coeffs, '__iter__'):
                        # List-like - each element may have .value
                        for coeff in raw_coeffs:
                            if hasattr(coeff, 'value'):
                                coefficients.append(coeff.value)
                            else:
                                coefficients.append(float(coeff) if coeff is not None else 0.0)
                elif hasattr(coeff_data, '__iter__') and not isinstance(coeff_data, str):
                    # Direct iterable
                    for coeff in coeff_data:
                        if hasattr(coeff, 'value'):
                            coefficients.append(coeff.value)
                        else:
                            coefficients.append(float(coeff) if coeff is not None else 0.0)

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

                # Trace fan of rays
                for hy in np.linspace(-1, 1, num_rays):
                    try:
                        # ZosPy single ray trace - pattern: ClassName(params).run(oss)
                        ray_trace = zp.analyses.raysandspots.SingleRayTrace(
                            Hx=0.0,
                            Hy=hy,
                            Px=0.0,
                            Py=0.0,
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
