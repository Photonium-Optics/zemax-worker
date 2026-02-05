#!/usr/bin/env python3
"""
Comprehensive Seidel Analysis Diagnostic Script

This script tests both Seidel methods independently to isolate issues:
1. Native SeidelCoefficients analysis (/seidel-native endpoint)
2. ZernikeStandardCoefficients analysis (/seidel endpoint)

Run on Windows where OpticStudio is installed:
    python test_seidel_diagnostic.py

Output includes detailed diagnostics for troubleshooting.
"""

import os
import sys
import tempfile
import traceback
from typing import Any, Optional

# Add some visual separators for readability
DIVIDER = "=" * 70
SUBDIV = "-" * 50

# Seidel coefficient keys (same as in zospy_handler.py)
SEIDEL_COEFFICIENT_KEYS = ["S1", "S2", "S3", "S4", "S5", "CLA", "CTR"]


def print_header(title: str) -> None:
    """Print a section header."""
    print(f"\n{DIVIDER}")
    print(f" {title}")
    print(DIVIDER)


def print_subheader(title: str) -> None:
    """Print a subsection header."""
    print(f"\n{SUBDIV}")
    print(f" {title}")
    print(SUBDIV)


def find_sample_file() -> Optional[str]:
    """
    Find a sample .zmx file from OpticStudio installation.

    Returns the first path found, or None if no sample file exists.
    """
    possible_paths = [
        # Public documents (most common)
        r"C:\Users\Public\Documents\Zemax\Samples\Sequential\Objectives\Double Gauss 28 degree field.zmx",
        # User documents
        os.path.expanduser(r"~\Documents\Zemax\Samples\Sequential\Objectives\Double Gauss 28 degree field.zmx"),
        # Alternative Ansys location
        r"C:\ProgramData\Ansys\Zemax OpticStudio\Samples\Sequential\Objectives\Double Gauss 28 degree field.zmx",
        # Alternative: Cooke Triplet (simpler system)
        r"C:\Users\Public\Documents\Zemax\Samples\Sequential\Objectives\Cooke 40 degree field.zmx",
    ]

    for path in possible_paths:
        if os.path.exists(path):
            return path

    return None


def get_system_state(oss: Any) -> dict[str, Any]:
    """
    Get detailed system state for diagnostics.

    Args:
        oss: OpticStudioSystem object

    Returns:
        Dict with system state information
    """
    state = {}

    try:
        # Basic system info
        state["num_surfaces"] = oss.LDE.NumberOfSurfaces
        state["num_fields"] = oss.SystemData.Fields.NumberOfFields
        state["num_wavelengths"] = oss.SystemData.Wavelengths.NumberOfWavelengths

        # Aperture info
        aperture = oss.SystemData.Aperture
        state["aperture_type"] = str(aperture.ApertureType)
        state["aperture_value"] = aperture.ApertureValue

        # Wavelengths
        wavelengths = []
        for i in range(1, oss.SystemData.Wavelengths.NumberOfWavelengths + 1):
            wl = oss.SystemData.Wavelengths.GetWavelength(i)
            wavelengths.append({
                "index": i,
                "wavelength_um": wl.Wavelength,
                "is_primary": wl.IsPrimary if hasattr(wl, 'IsPrimary') else None,
            })
        state["wavelengths"] = wavelengths

        # Fields
        fields = []
        for i in range(1, oss.SystemData.Fields.NumberOfFields + 1):
            f = oss.SystemData.Fields.GetField(i)
            fields.append({
                "index": i,
                "x": f.X,
                "y": f.Y,
                "weight": f.Weight if hasattr(f, 'Weight') else None,
            })
        state["fields"] = fields

        # Try to get EFL
        try:
            import zospy as zp
            result = zp.analyses.reports.SystemData().run(oss)
            state["efl"] = result.data.general_lens_data.effective_focal_length_air
        except Exception as e:
            state["efl"] = f"Error: {e}"

    except Exception as e:
        state["error"] = str(e)

    return state


def print_system_state(state: dict[str, Any]) -> None:
    """Print system state in readable format."""
    print_subheader("System State")

    if "error" in state:
        print(f"  ERROR: {state['error']}")
        return

    print(f"  Surfaces:    {state.get('num_surfaces', 'N/A')}")
    print(f"  Fields:      {state.get('num_fields', 'N/A')}")
    print(f"  Wavelengths: {state.get('num_wavelengths', 'N/A')}")
    print(f"  Aperture:    {state.get('aperture_type', 'N/A')} = {state.get('aperture_value', 'N/A')}")
    print(f"  EFL:         {state.get('efl', 'N/A')}")

    if state.get("wavelengths"):
        print("\n  Wavelengths:")
        for wl in state["wavelengths"]:
            primary = " (primary)" if wl.get("is_primary") else ""
            print(f"    [{wl['index']}] {wl['wavelength_um']:.4f} um{primary}")

    if state.get("fields"):
        print("\n  Fields:")
        for f in state["fields"]:
            print(f"    [{f['index']}] X={f['x']:.2f}, Y={f['y']:.2f}")


def test_native_seidel(oss: Any, zp: Any) -> dict[str, Any]:
    """
    Test native SeidelCoefficients analysis.

    This mimics what the /seidel-native endpoint does.

    Args:
        oss: OpticStudioSystem object
        zp: ZosPy module

    Returns:
        Dict with test results
    """
    print_header("Native SeidelCoefficients Analysis")

    results = {
        "success": False,
        "raw_text": None,
        "parsed_data": None,
        "errors": [],
        "warnings": [],
    }

    analysis = None
    temp_path = os.path.join(tempfile.gettempdir(), "seidel_diagnostic.txt")

    try:
        # Create analysis
        print("\n[1] Creating SeidelCoefficients analysis...")
        idm = zp.constants.Analysis.AnalysisIDM
        analysis = zp.analyses.new_analysis(
            oss,
            idm.SeidelCoefficients,
            settings_first=True
        )
        print("    Analysis created successfully")

        # Check settings
        print("\n[2] Checking analysis settings...")
        settings = analysis.Settings
        settings_attrs = [a for a in dir(settings) if not a.startswith("_")]
        print(f"    Settings type: {type(settings)}")
        print(f"    Available settings ({len(settings_attrs)}): {settings_attrs[:10]}...")

        # Run analysis
        print("\n[3] Running analysis (ApplyAndWaitForCompletion)...")
        analysis.ApplyAndWaitForCompletion()
        print("    Analysis completed")

        # Check for messages
        if hasattr(analysis, 'messages') and analysis.messages:
            print("\n    Analysis messages:")
            for msg in analysis.messages:
                msg_text = str(msg.Message) if hasattr(msg, 'Message') else str(msg)
                print(f"      - {msg_text}")
                results["warnings"].append(msg_text)

        # Export to text file
        print("\n[4] Exporting to text file (GetTextFile)...")
        res = analysis.Results
        res.GetTextFile(temp_path)

        if not os.path.exists(temp_path):
            results["errors"].append("GetTextFile did not create output file")
            print("    ERROR: Text file was not created!")
            return results

        file_size = os.path.getsize(temp_path)
        print(f"    File created: {temp_path}")
        print(f"    File size: {file_size} bytes")

        # Read with different encodings
        print("\n[5] Reading text file content...")
        raw_text = None

        # Try UTF-16 first (OpticStudio default)
        try:
            with open(temp_path, 'r', encoding='utf-16') as f:
                raw_text = f.read()
            print(f"    Read with UTF-16 encoding: {len(raw_text)} chars")
        except Exception as e:
            print(f"    UTF-16 failed: {e}")
            results["warnings"].append(f"UTF-16 read failed: {e}")

        # Try UTF-8 as fallback
        if not raw_text:
            try:
                with open(temp_path, 'r', encoding='utf-8') as f:
                    raw_text = f.read()
                print(f"    Read with UTF-8 encoding: {len(raw_text)} chars")
            except Exception as e:
                print(f"    UTF-8 failed: {e}")
                results["warnings"].append(f"UTF-8 read failed: {e}")

        # Try latin-1 as last resort
        if not raw_text:
            try:
                with open(temp_path, 'r', encoding='latin-1') as f:
                    raw_text = f.read()
                print(f"    Read with latin-1 encoding: {len(raw_text)} chars")
            except Exception as e:
                print(f"    latin-1 failed: {e}")
                results["errors"].append(f"All encoding attempts failed: {e}")
                return results

        results["raw_text"] = raw_text

        # Check for empty content
        if not raw_text or raw_text.strip() == "":
            results["errors"].append("Text file is empty")
            print("    ERROR: Text content is empty!")
            return results

        # Display raw text
        print_subheader("Raw Text File Content")
        print(raw_text[:3000])
        if len(raw_text) > 3000:
            print(f"\n... truncated ({len(raw_text)} total chars)")

        # Parse the text
        print("\n[6] Parsing text content...")
        parsed = parse_seidel_text(raw_text)
        results["parsed_data"] = parsed

        # Display parsed results
        print_subheader("Parsed Results")

        # Header info
        if parsed.get("header"):
            print("\n  Header data:")
            for key, value in parsed["header"].items():
                print(f"    {key}: {value}")
        else:
            print("  No header data parsed")
            results["warnings"].append("No header data found")

        # Per-surface data
        if parsed.get("per_surface"):
            print(f"\n  Per-surface coefficients ({len(parsed['per_surface'])} surfaces):")
            print(f"    {'Surf':>5} {'S1':>12} {'S2':>12} {'S3':>12} {'S4':>12} {'S5':>12} {'CLA':>10} {'CTR':>10}")
            print("    " + "-" * 90)
            for surf in parsed["per_surface"]:
                print(f"    {surf['surface']:>5} "
                      f"{surf.get('S1', 0):>12.6f} "
                      f"{surf.get('S2', 0):>12.6f} "
                      f"{surf.get('S3', 0):>12.6f} "
                      f"{surf.get('S4', 0):>12.6f} "
                      f"{surf.get('S5', 0):>12.6f} "
                      f"{surf.get('CLA', 0):>10.4f} "
                      f"{surf.get('CTR', 0):>10.4f}")
        else:
            print("  No per-surface data parsed")
            results["errors"].append("No per-surface coefficients found")

        # Totals
        if parsed.get("totals"):
            totals = parsed["totals"]
            print(f"\n  Totals:")
            print(f"    S1: {totals.get('S1', 0):.6f}")
            print(f"    S2: {totals.get('S2', 0):.6f}")
            print(f"    S3: {totals.get('S3', 0):.6f}")
            print(f"    S4: {totals.get('S4', 0):.6f}")
            print(f"    S5: {totals.get('S5', 0):.6f}")
            print(f"    CLA: {totals.get('CLA', 0):.4f}")
            print(f"    CTR: {totals.get('CTR', 0):.4f}")

            # Check for all zeros
            if all(abs(totals.get(key, 0)) < 1e-15 for key in SEIDEL_COEFFICIENT_KEYS[:5]):
                print("\n    WARNING: All Seidel totals are zero!")
                results["warnings"].append("All Seidel totals are zero")
        else:
            print("  No totals data parsed")
            results["warnings"].append("No totals row found")

        results["success"] = bool(parsed.get("per_surface")) or bool(parsed.get("totals"))

    except Exception as e:
        results["errors"].append(f"Exception: {e}")
        print(f"\n  ERROR: {e}")
        traceback.print_exc()

    finally:
        # Cleanup
        if analysis:
            try:
                analysis.Close()
            except Exception:
                pass
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass

    return results


def test_zernike_seidel(oss: Any, zp: Any) -> dict[str, Any]:
    """
    Test ZernikeStandardCoefficients analysis.

    This mimics what the /seidel endpoint does.

    Args:
        oss: OpticStudioSystem object
        zp: ZosPy module

    Returns:
        Dict with test results
    """
    print_header("ZernikeStandardCoefficients Analysis")

    results = {
        "success": False,
        "coefficients": None,
        "metadata": None,
        "errors": [],
        "warnings": [],
    }

    try:
        # Create and run Zernike analysis
        print("\n[1] Creating ZernikeStandardCoefficients analysis...")
        zernike_analysis = zp.analyses.wavefront.ZernikeStandardCoefficients(
            sampling='64x64',
            maximum_term=37,
            wavelength=1,
            field=1,
            surface="Image",
        )
        print("    Analysis created")

        print("\n[2] Running analysis...")
        result = zernike_analysis.run(oss)
        print("    Analysis completed")

        # Check result object
        print("\n[3] Checking result object...")
        print(f"    Result type: {type(result)}")

        if not hasattr(result, 'data') or result.data is None:
            results["errors"].append("Analysis returned no data attribute")
            print("    ERROR: result.data is None or missing!")
            return results

        data = result.data
        print(f"    Data type: {type(data)}")

        # List available attributes
        data_attrs = [a for a in dir(data) if not a.startswith("_")]
        print(f"    Data attributes ({len(data_attrs)}): {data_attrs[:15]}...")

        # Get metadata
        metadata = {}

        if hasattr(data, 'surface'):
            metadata["surface"] = str(data.surface)
            print(f"\n    Surface: {metadata['surface']}")

        if hasattr(data, 'field'):
            metadata["field"] = str(data.field)
            print(f"    Field: {metadata['field']}")

        if hasattr(data, 'wavelength'):
            metadata["wavelength"] = str(data.wavelength)
            print(f"    Wavelength: {metadata['wavelength']}")

        # Get P-V and RMS
        if hasattr(data, 'peak_to_valley_to_chief'):
            metadata["pv_to_chief"] = float(data.peak_to_valley_to_chief)
            print(f"    P-V to chief: {metadata['pv_to_chief']:.6f} waves")

        if hasattr(data, 'peak_to_valley_to_centroid'):
            metadata["pv_to_centroid"] = float(data.peak_to_valley_to_centroid)
            print(f"    P-V to centroid: {metadata['pv_to_centroid']:.6f} waves")

        # Get integration data
        if hasattr(data, 'from_integration_of_the_rays'):
            integration = data.from_integration_of_the_rays
            if hasattr(integration, 'rms_to_chief'):
                metadata["rms_to_chief"] = float(integration.rms_to_chief)
                print(f"    RMS to chief: {metadata['rms_to_chief']:.6f} waves")
            if hasattr(integration, 'strehl_ratio'):
                metadata["strehl_ratio"] = float(integration.strehl_ratio)
                print(f"    Strehl ratio: {metadata['strehl_ratio']:.6f}")

        results["metadata"] = metadata

        # Get coefficients
        print("\n[4] Extracting Zernike coefficients...")

        if not hasattr(data, 'coefficients'):
            results["errors"].append("No coefficients attribute on result.data")
            print("    ERROR: result.data.coefficients not found!")
            return results

        raw_coeffs = data.coefficients
        print(f"    Coefficients type: {type(raw_coeffs)}")

        if raw_coeffs is None:
            results["errors"].append("Coefficients is None")
            print("    ERROR: Coefficients is None!")
            return results

        # Parse coefficients
        coefficients = []

        if isinstance(raw_coeffs, dict):
            print(f"    Dict with {len(raw_coeffs)} entries")

            if not raw_coeffs:
                results["errors"].append("Empty coefficients dict")
                print("    ERROR: Coefficients dict is empty!")
                return results

            # Show sample entries
            sample_keys = list(raw_coeffs.keys())[:5]
            print(f"    Sample keys: {sample_keys}")

            for key in sample_keys:
                val = raw_coeffs[key]
                print(f"      [{key}] type={type(val)}, value={val}")

            # Extract values
            int_keys = [int(k) for k in raw_coeffs.keys()]
            max_term = max(int_keys) if int_keys else 0
            print(f"    Max term: {max_term}")

            for i in range(1, min(max_term + 1, 38)):
                coeff = raw_coeffs.get(i) or raw_coeffs.get(str(i))
                if coeff is not None:
                    if hasattr(coeff, 'value'):
                        coefficients.append(float(coeff.value))
                    else:
                        coefficients.append(float(coeff))
                else:
                    coefficients.append(0.0)

        elif hasattr(raw_coeffs, '__iter__'):
            print(f"    Iterable type")
            for coeff in raw_coeffs:
                if hasattr(coeff, 'value'):
                    coefficients.append(float(coeff.value))
                else:
                    coefficients.append(float(coeff) if coeff is not None else 0.0)
        else:
            results["errors"].append(f"Unknown coefficients type: {type(raw_coeffs)}")
            print(f"    ERROR: Unknown type: {type(raw_coeffs)}")
            return results

        results["coefficients"] = coefficients

        # Display coefficients
        print_subheader("Zernike Coefficients (First 16)")
        print(f"    {'Term':>5} {'Value (waves)':>15}")
        print("    " + "-" * 25)

        # Zernike term names (standard ordering)
        zernike_names = {
            1: "Piston",
            2: "Tilt X",
            3: "Tilt Y",
            4: "Defocus",
            5: "Astig (0)",
            6: "Astig (45)",
            7: "Coma X",
            8: "Coma Y",
            9: "Trefoil X",
            10: "Trefoil Y",
            11: "Spherical",
            12: "2nd Astig",
            13: "2nd Astig",
            14: "Quadrafoil",
            15: "Quadrafoil",
            16: "2nd Coma",
        }

        for i, val in enumerate(coefficients[:16], 1):
            name = zernike_names.get(i, "")
            print(f"    Z{i:>3}: {val:>12.6f}  {name}")

        # Check for all zeros
        if all(abs(c) < 1e-15 for c in coefficients):
            print("\n    WARNING: All Zernike coefficients are zero!")
            results["warnings"].append("All Zernike coefficients are zero")

        # Show key aberration terms
        print_subheader("Key Aberration Terms")

        if len(coefficients) >= 11:
            print(f"    Z4  (Defocus):    {coefficients[3]:>12.6f} waves")
            print(f"    Z5  (Astig 0):    {coefficients[4]:>12.6f} waves")
            print(f"    Z6  (Astig 45):   {coefficients[5]:>12.6f} waves")
            print(f"    Z7  (Coma X):     {coefficients[6]:>12.6f} waves")
            print(f"    Z8  (Coma Y):     {coefficients[7]:>12.6f} waves")
            print(f"    Z11 (Spherical):  {coefficients[10]:>12.6f} waves")

        results["success"] = len(coefficients) > 0 and not all(abs(c) < 1e-15 for c in coefficients)

    except Exception as e:
        results["errors"].append(f"Exception: {e}")
        print(f"\n  ERROR: {e}")
        traceback.print_exc()

    return results


def parse_seidel_text(text: str) -> dict[str, Any]:
    """
    Parse Seidel text output from OpticStudio.

    This is a copy of the parsing logic from zospy_handler.py for diagnostic purposes.
    """
    lines = text.strip().split('\n')

    header: dict[str, Any] = {}
    per_surface: list[dict[str, Any]] = []
    totals: dict[str, float] = {}
    in_table = False

    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            continue

        # Parse header values
        if ':' in line_stripped and not in_table:
            parts = line_stripped.split(':', 1)
            if len(parts) == 2:
                key = parts[0].strip().lower().replace(' ', '_')
                value = parts[1].strip()
                try:
                    numeric_part = value.split()[0] if value else ''
                    header[key] = float(numeric_part)
                except (ValueError, IndexError):
                    header[key] = value

        # Detect table header
        if 'SPHA' in line_stripped or ('S1' in line_stripped and 'S2' in line_stripped):
            in_table = True
            continue

        # Parse data rows
        if in_table:
            parts = line_stripped.split()
            if not parts:
                continue

            first_part = parts[0]
            values = []
            for p in parts[1:]:
                try:
                    values.append(float(p))
                except ValueError:
                    continue

            if first_part.isdigit() or first_part.upper() == 'STO':
                surface_num = int(first_part) if first_part.isdigit() else 0
                surface_data = build_seidel_coefficients(surface_num, values)
                per_surface.append(surface_data)
            elif first_part.lower() == 'sum':
                totals.update(build_seidel_totals(values))

    return {
        "header": header,
        "per_surface": per_surface,
        "totals": totals,
    }


def build_seidel_coefficients(surface_num: int, values: list[float]) -> dict[str, Any]:
    """Build per-surface Seidel coefficient dict."""
    surface_data: dict[str, Any] = {"surface": surface_num}
    num_values = len(values)

    if num_values >= 12:
        # Paired format with chromatic
        extracted = [values[1], values[3], values[5], values[7], values[9], values[10], values[11]]
    elif num_values >= 10:
        # Paired format without chromatic
        extracted = [values[1], values[3], values[5], values[7], values[9], 0.0, 0.0]
    elif num_values >= 7:
        # Already unpaired
        extracted = values[:7]
    elif num_values >= 5:
        # Minimal
        extracted = values[:5] + [0.0, 0.0]
    else:
        extracted = values + [0.0] * (7 - num_values)

    for i, key in enumerate(SEIDEL_COEFFICIENT_KEYS):
        surface_data[key] = extracted[i] if i < len(extracted) else 0.0

    return surface_data


def build_seidel_totals(values: list[float]) -> dict[str, float]:
    """Build Seidel totals dict."""
    num_values = len(values)

    if num_values >= 12:
        extracted = [values[1], values[3], values[5], values[7], values[9], values[10], values[11]]
    elif num_values >= 10:
        extracted = [values[1], values[3], values[5], values[7], values[9], 0.0, 0.0]
    elif num_values >= 7:
        extracted = values[:7]
    elif num_values >= 5:
        extracted = values[:5] + [0.0, 0.0]
    else:
        extracted = values + [0.0] * (7 - num_values)

    totals: dict[str, float] = {}
    for i, key in enumerate(SEIDEL_COEFFICIENT_KEYS):
        totals[key] = extracted[i] if i < len(extracted) else 0.0
    return totals


def compare_results(native_results: dict[str, Any], zernike_results: dict[str, Any]) -> None:
    """Compare results from both methods."""
    print_header("Results Comparison")

    print("\n  Summary:")
    print(f"    Native Seidel:  {'SUCCESS' if native_results['success'] else 'FAILED'}")
    print(f"    Zernike-based:  {'SUCCESS' if zernike_results['success'] else 'FAILED'}")

    # Show errors
    if native_results.get("errors"):
        print(f"\n  Native errors: {native_results['errors']}")
    if zernike_results.get("errors"):
        print(f"\n  Zernike errors: {zernike_results['errors']}")

    # Compare values if both succeeded
    if native_results["success"] and zernike_results["success"]:
        print("\n  Note: Native Seidel returns S1-S5 in waves directly.")
        print("        Zernike method requires conversion on Mac side.")
        print("        Direct comparison of values is not meaningful here.")

    # Failure mode analysis
    print_subheader("Failure Mode Analysis")

    failure_modes = {
        "empty_text": "GetTextFile produced empty output",
        "encoding": "Text file encoding issues (UTF-16 vs UTF-8)",
        "no_data": "Analysis returned no data attribute",
        "all_zeros": "All coefficient values are zero",
        "parse_error": "Could not parse text file format",
        "no_coefficients": "Coefficients attribute missing or empty",
    }

    print("\n  Checking for common failure modes:")

    # Check native
    if not native_results["success"]:
        for error in native_results.get("errors", []):
            for mode, desc in failure_modes.items():
                if mode in error.lower() or any(word in error.lower() for word in desc.lower().split()):
                    print(f"    [Native] {desc}")

    # Check Zernike
    if not zernike_results["success"]:
        for error in zernike_results.get("errors", []):
            for mode, desc in failure_modes.items():
                if mode in error.lower() or any(word in error.lower() for word in desc.lower().split()):
                    print(f"    [Zernike] {desc}")

    if native_results["success"] and zernike_results["success"]:
        print("    No failures detected in either method.")


def main() -> int:
    """Main diagnostic function."""
    print_header("Seidel Analysis Diagnostic Tool")
    print(f"  Working directory: {os.getcwd()}")

    # Import ZosPy
    print("\n[1] Importing ZosPy...")
    try:
        import zospy as zp
        zospy_version = getattr(zp, '__version__', 'unknown')
        print(f"    ZosPy version: {zospy_version}")
    except ImportError as e:
        print(f"    ERROR: Could not import ZosPy: {e}")
        print("    Make sure ZosPy is installed: pip install zospy")
        return 1

    # Connect to OpticStudio
    print("\n[2] Connecting to OpticStudio...")
    try:
        zos = zp.ZOS()
        oss = zos.connect(mode="standalone")
        if oss is None:
            print("    ERROR: Could not connect to OpticStudio")
            return 1
        print("    Connected successfully")

        # Get OpticStudio version
        try:
            version = str(oss.Application.ZemaxVersion) if oss else "Unknown"
            print(f"    OpticStudio version: {version}")
        except Exception:
            print("    OpticStudio version: Unknown")

    except Exception as e:
        print(f"    ERROR: {e}")
        traceback.print_exc()
        return 1

    # Find and load sample file
    print("\n[3] Loading sample optical system...")
    sample_file = find_sample_file()

    if not sample_file:
        print("    ERROR: No sample file found!")
        print("    Searched locations:")
        print("      - C:\\Users\\Public\\Documents\\Zemax\\Samples\\...")
        print("      - %USERPROFILE%\\Documents\\Zemax\\Samples\\...")
        print("    Please specify a valid .zmx file path in the script.")
        zos.disconnect()
        return 1

    print(f"    Found: {sample_file}")

    try:
        oss.load(sample_file)
        print("    Loaded successfully")
    except Exception as e:
        print(f"    ERROR loading file: {e}")
        zos.disconnect()
        return 1

    # Get and print system state
    state = get_system_state(oss)
    print_system_state(state)

    # Run diagnostics
    native_results = test_native_seidel(oss, zp)
    zernike_results = test_zernike_seidel(oss, zp)

    # Compare results
    compare_results(native_results, zernike_results)

    # Final summary
    print_header("Final Summary")

    if native_results["success"] and zernike_results["success"]:
        print("\n  RESULT: Both methods working correctly")
        print("  The Seidel analysis pipeline should function properly.")
    elif native_results["success"]:
        print("\n  RESULT: Native Seidel works, Zernike has issues")
        print("  The /seidel-native endpoint should work.")
        print("  Check Zernike analysis setup for /seidel endpoint.")
    elif zernike_results["success"]:
        print("\n  RESULT: Zernike works, Native Seidel has issues")
        print("  The /seidel endpoint (with Zernike conversion) should work.")
        print("  Check SeidelCoefficients analysis for /seidel-native endpoint.")
    else:
        print("\n  RESULT: Both methods failing")
        print("  Check OpticStudio installation and license.")
        print("  Verify the optical system is valid and has optical power.")

    # Disconnect
    print("\n[4] Disconnecting from OpticStudio...")
    try:
        zos.disconnect()
        print("    Disconnected successfully")
    except Exception as e:
        print(f"    Warning: {e}")

    print(f"\n{DIVIDER}")
    print(" Diagnostic complete")
    print(DIVIDER)

    return 0 if (native_results["success"] or zernike_results["success"]) else 1


if __name__ == "__main__":
    sys.exit(main())
