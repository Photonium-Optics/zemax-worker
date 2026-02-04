"""
Seidel Converter

Converts Zernike Standard Coefficients (from ZosPy) to Seidel aberrations.

Zemax provides Zernike polynomials via the ZernikeStandardCoefficients analysis.
This module converts them to the Seidel aberration format expected by the UI:
- S1: Spherical Aberration
- S2: Coma
- S3: Astigmatism
- S4: Field Curvature (Petzval)
- S5: Distortion
- C1: Axial Chromatic Aberration
- C2: Lateral Chromatic Aberration
"""

import math
from typing import Any, Optional


# Zernike Standard indices (Noll ordering)
# See: https://en.wikipedia.org/wiki/Zernike_polynomials
ZERNIKE_INDICES = {
    "piston": 1,          # Z1
    "tilt_x": 2,          # Z2
    "tilt_y": 3,          # Z3
    "defocus": 4,         # Z4 (power)
    "astig_45": 5,        # Z5
    "astig_0": 6,         # Z6
    "coma_y": 7,          # Z7
    "coma_x": 8,          # Z8
    "trefoil_y": 9,       # Z9
    "trefoil_x": 10,      # Z10
    "spherical": 11,      # Z11 (primary spherical)
    "astig_2nd_45": 12,   # Z12
    "astig_2nd_0": 13,    # Z13
    "coma_2nd_y": 14,     # Z14
    "coma_2nd_x": 15,     # Z15
    "tetrafoil_y": 16,    # Z16
    "tetrafoil_x": 17,    # Z17
    "spherical_2nd": 22,  # Z22 (secondary spherical)
}


def zernike_to_seidel(
    zernike_coefficients: list[float],
    wavelength_um: float = 0.5876,
    field_angle_deg: float = 0.0,
) -> dict[str, Any]:
    """
    Convert Zernike Standard coefficients to Seidel aberrations.

    The conversion uses the relationship between Zernike polynomials
    and classical aberration theory. This is an approximation that
    works well for systems with moderate aberrations.

    Note: Zernike coefficients from Zemax are typically in waves, normalized
    to the actual pupil radius. The wavelength is used to convert from
    waves to physical units (mm).

    Args:
        zernike_coefficients: List of Zernike coefficients (Z1, Z2, ..., Z37) in waves
        wavelength_um: Wavelength in micrometers (for converting waves to mm)
        field_angle_deg: Field angle in degrees (for field-dependent terms like distortion)

    Returns:
        Dict with Seidel coefficients S1-S5 in mm
    """
    # Ensure we have enough coefficients (pad with zeros if needed)
    z = zernike_coefficients + [0.0] * (37 - len(zernike_coefficients))

    # Convert to 1-indexed for clarity (Z1 = z[1], etc.)
    z = [0.0] + z  # Now z[1] is the first Zernike term

    # Scale factor: convert from waves to mm
    # Zernike coefficients are often in waves, need to convert
    scale = wavelength_um / 1000.0  # um to mm

    # ==========================================================================
    # Seidel Aberrations (S1-S5)
    # ==========================================================================
    #
    # The mapping from Zernike to Seidel is approximate. The exact relationship
    # depends on the normalization and the optical system. These formulas are
    # based on standard aberration theory.
    #
    # Reference: Mahajan, V.N., "Aberration Theory Made Simple"

    # S1: Spherical Aberration
    # Primary contribution from Z11 (primary spherical)
    # Secondary contribution from Z22 (secondary spherical)
    s1 = _safe_get(z, 11) * scale * 6.0 * math.sqrt(5)

    # S2: Coma
    # From Z7 and Z8 (primary coma)
    coma_y = _safe_get(z, 7)
    coma_x = _safe_get(z, 8)
    s2 = math.sqrt(coma_y**2 + coma_x**2) * scale * 3.0 * math.sqrt(8)

    # S3: Astigmatism
    # From Z5 and Z6 (primary astigmatism)
    # Note: The standard conversion factor is sqrt(6), not 2*sqrt(6)
    # Reference: Noll (1976), "Zernike polynomials and atmospheric turbulence"
    astig_45 = _safe_get(z, 5)
    astig_0 = _safe_get(z, 6)
    s3 = math.sqrt(astig_45**2 + astig_0**2) * scale * math.sqrt(6)

    # S4: Field Curvature (Petzval)
    # ⚠️ FUNDAMENTALLY LIMITED: True Petzval sum CANNOT be computed from Zernike defocus.
    # Z4 (defocus) contains contributions from:
    #   1. Field curvature (Petzval)
    #   2. Astigmatism
    #   3. Defocus error
    # The true Petzval sum requires: Σ(φᵢ / nᵢ·nᵢ') across all surfaces
    # This proxy value is for display purposes only - DO NOT use for optimization.
    s4 = _safe_get(z, 4) * scale * 2.0 * math.sqrt(3)

    # S5: Distortion
    # ⚠️ FUNDAMENTALLY LIMITED: True distortion CANNOT be computed from Zernike tilt.
    # Zernike Z2/Z3 (tilt) represents wavefront slope, NOT image displacement.
    # True distortion (S5) requires:
    #   - Ray tracing at multiple field heights
    #   - Comparing actual vs. ideal (paraxial) image positions
    #   - Computing: (y_actual - y_paraxial) / y_paraxial
    # This proxy value is for display purposes only - DO NOT use for optimization.
    if field_angle_deg > 0:
        tilt_y = _safe_get(z, 3)
        tilt_x = _safe_get(z, 2)
        s5 = math.sqrt(tilt_y**2 + tilt_x**2) * scale * 2.0
    else:
        s5 = 0.0  # No distortion on-axis

    return {
        "S1": s1,
        "S2": s2,
        "S3": s3,
        "S4": s4,
        "S5": s5,
    }


def _safe_get(arr: list[float], idx: int, default: float = 0.0) -> float:
    """Safely get array element with default."""
    try:
        return arr[idx] if idx < len(arr) else default
    except (IndexError, TypeError):
        return default


def build_seidel_response(
    seidel: dict[str, float],
    chromatic: Optional[dict[str, float]] = None,
    per_surface: Optional[list[dict[str, float]]] = None,
    num_surfaces: int = 0,
) -> dict[str, Any]:
    """
    Build the response structure for the /seidel endpoint.

    Args:
        seidel: Dict with S1-S5 values
        chromatic: Dict with C1, C2 values (optional)
        per_surface: List of per-surface coefficient dicts (optional)
        num_surfaces: Number of optical surfaces

    Returns:
        Dict matching the SeidelDiagramResponse model
    """
    # Build coefficient entries
    coefficient_names = {
        "S1": "Spherical",
        "S2": "Coma",
        "S3": "Astigmatism",
        "S4": "Petzval",
        "S5": "Distortion",
    }

    coefficients = []
    raw_values = []

    for key in ["S1", "S2", "S3", "S4", "S5"]:
        value = seidel.get(key, 0.0)
        coefficients.append({
            "key": key,
            "name": coefficient_names[key],
            "value": value,
        })
        raw_values.append(value)

    response = {
        "success": True,
        "seidel_coefficients": {
            "coefficients": coefficients,
            "raw_values": raw_values,
            "units": "mm",
        },
        "num_surfaces": num_surfaces,
    }

    # Add per-surface breakdown if available
    if per_surface:
        response["per_surface"] = {
            "TSC": [s.get("S1", 0.0) for s in per_surface],
            "CC": [s.get("S2", 0.0) for s in per_surface],
            "TAC": [s.get("S3", 0.0) for s in per_surface],
            "TPC": [s.get("S4", 0.0) for s in per_surface],
            "DC": [s.get("S5", 0.0) for s in per_surface],
        }

    # Add chromatic aberrations if available
    if chromatic:
        response["chromatic"] = {
            "axial_color": {
                "name": "Axial Chromatic Aberration",
                "total": chromatic.get("C1", 0.0),
                "per_surface": chromatic.get("C1_per_surface", []),
                "units": "mm",
            },
            "lateral_color": {
                "name": "Lateral Chromatic Aberration",
                "total": chromatic.get("C2", 0.0),
                "per_surface": chromatic.get("C2_per_surface", []),
                "units": "mm",
            },
        }

    return response


def estimate_chromatic_aberration(
    primary_wavelength_um: float,
    short_wavelength_um: float,
    long_wavelength_um: float,
    focus_shift_short_mm: float,
    focus_shift_long_mm: float,
    lateral_shift_short_mm: float = 0.0,
    lateral_shift_long_mm: float = 0.0,
) -> dict[str, float]:
    """
    Estimate chromatic aberrations from focus shifts at different wavelengths.

    In Zemax/ZosPy, chromatic aberration can be computed by comparing
    the focus position at different wavelengths.

    Args:
        primary_wavelength_um: Primary wavelength (reference)
        short_wavelength_um: Short wavelength (e.g., F-line 486.1nm)
        long_wavelength_um: Long wavelength (e.g., C-line 656.3nm)
        focus_shift_short_mm: Axial focus shift at short wavelength
        focus_shift_long_mm: Axial focus shift at long wavelength
        lateral_shift_short_mm: Lateral shift at short wavelength
        lateral_shift_long_mm: Lateral shift at long wavelength

    Returns:
        Dict with C1 (axial) and C2 (lateral) chromatic aberrations
    """
    # Axial chromatic (longitudinal): difference in focus position
    c1 = focus_shift_long_mm - focus_shift_short_mm

    # Lateral chromatic (transverse): difference in image height
    c2 = lateral_shift_long_mm - lateral_shift_short_mm

    return {
        "C1": c1,
        "C2": c2,
    }


# =============================================================================
# Alternative: Direct Seidel from Zemax
# =============================================================================

def parse_zemax_seidel_data(zemax_data: dict[str, Any]) -> dict[str, Any]:
    """
    Parse Seidel data directly from Zemax's Seidel Coefficients analysis.

    If the Windows worker can access Zemax's built-in Seidel analysis
    (not available in all ZosPy versions), use this to parse the results.

    Args:
        zemax_data: Raw output from Zemax Seidel analysis

    Returns:
        Dict matching our Seidel response format
    """
    # Extract total Seidel coefficients
    seidel = {
        "S1": zemax_data.get("SPHA", 0.0),  # Spherical
        "S2": zemax_data.get("COMA", 0.0),  # Coma
        "S3": zemax_data.get("ASTI", 0.0),  # Astigmatism
        "S4": zemax_data.get("FCUR", 0.0),  # Field curvature
        "S5": zemax_data.get("DIST", 0.0),  # Distortion
    }

    # Extract chromatic if available
    chromatic = None
    if "AXCL" in zemax_data or "LACL" in zemax_data:
        chromatic = {
            "C1": zemax_data.get("AXCL", 0.0),  # Axial chromatic
            "C2": zemax_data.get("LACL", 0.0),  # Lateral chromatic
        }

    # Extract per-surface breakdown if available
    per_surface = zemax_data.get("per_surface", None)

    return build_seidel_response(
        seidel=seidel,
        chromatic=chromatic,
        per_surface=per_surface,
        num_surfaces=zemax_data.get("num_surfaces", 0),
    )
