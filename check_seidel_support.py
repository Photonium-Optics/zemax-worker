#!/usr/bin/env python3
"""
Diagnostic script to check if SeidelCoefficients analysis is available in ZOSPy.

Run this on the Windows machine with OpticStudio installed:
    python check_seidel_support.py

This will tell us whether we can use the native Seidel analysis instead of
deriving Seidel from Zernike coefficients.

Based on GPT's suggestion - includes version info for debugging.
"""

import zospy as zp


def main():
    print("=" * 70)
    print("ZOSPy Seidel Support Diagnostic")
    print("=" * 70)

    # Connect to OpticStudio
    print("\n[1] Connecting to OpticStudio...")
    zos = zp.ZOS()
    oss = zos.connect(mode="standalone")

    # Print version info (helps debug missing enum members)
    print(f"\n[2] Version Info:")
    print(f"    ZOSPy version: {getattr(zp, '__version__', 'unknown')}")

    # Get OpticStudio version - use direct attribute access (correct ZOSPy API)
    try:
        version = str(oss.Application.ZemaxVersion)
        print(f"    OpticStudio version: {version}")
    except Exception as e:
        print(f"    OpticStudio version: (error: {e})")

    # Force-load constants (they're dynamic)
    print("\n[3] Loading ZOSPy constants...")
    _ = zp.constants

    # Check AnalysisIDM for Seidel-related entries
    print("\n[4] Checking AnalysisIDM for Seidel analyses...")
    idm = zp.constants.Analysis.AnalysisIDM
    names = [n for n in dir(idm) if not n.startswith("_")]

    print(f"    Total analysis types available: {len(names)}")

    # Find Seidel-related entries (case-insensitive)
    seidelish = [n for n in names if "seidel" in n.lower()]
    print(f"    AnalysisIDM entries containing 'Seidel': {seidelish if seidelish else 'NONE FOUND'}")

    # Also check for related aberration analyses
    aberration_related = [n for n in names if any(
        kw in n.lower() for kw in ['aberr', 'third', 'petzval', 'coma', 'astig']
    )]
    if aberration_related:
        print(f"    Other aberration-related entries: {aberration_related}")

    # Try to run Seidel analyses if present
    print("\n[5] Attempting to run Seidel analyses...")

    for candidate in ["SeidelCoefficients", "SeidelDiagram", "SeidelAberrations"]:
        if candidate in names:
            print(f"\n    Found '{candidate}' - attempting to run...")
            try:
                an = zp.analyses.new_analysis(
                    oss,
                    getattr(idm, candidate),
                    settings_first=True
                )

                # Check available settings using hasattr pattern
                if hasattr(an, 'Settings'):
                    settings = an.Settings
                    settings_attrs = [a for a in dir(settings) if not a.startswith("_")]
                    print(f"    Settings attributes: {settings_attrs[:15]}...")

                # Run with defaults
                an.ApplyAndWaitForCompletion()

                # Access results safely using hasattr
                if hasattr(an, 'Results'):
                    res = an.Results

                    # Try to get text output (most reliable across versions)
                    if hasattr(res, 'GetTextFile'):
                        try:
                            txt = res.GetTextFile()
                            print(f"    Text output (first 800 chars):")
                            print("-" * 50)
                            print(txt[:800] if txt else "    (empty)")
                            print("-" * 50)
                        except Exception as e:
                            print(f"    GetTextFile failed: {e}")

                    # Try to get data grids
                    if hasattr(res, 'GetDataGrid'):
                        try:
                            for i in range(3):  # Try first 3 grids
                                try:
                                    grid = res.GetDataGrid(i)
                                    if grid:
                                        print(f"    DataGrid[{i}]: {grid}")
                                except:
                                    break
                        except Exception as e:
                            print(f"    GetDataGrid failed: {e}")

                    # Check results attributes
                    results_attrs = [a for a in dir(res) if not a.startswith("_")]
                    print(f"    Results attributes: {results_attrs[:15]}...")

                an.Close()
                print(f"    SUCCESS: '{candidate}' analysis works!")

            except Exception as e:
                print(f"    ERROR running '{candidate}': {e}")
        else:
            print(f"    '{candidate}' not found in AnalysisIDM")

    # If no Seidel found, show what IS available for reference
    if not seidelish:
        print("\n[6] No Seidel analyses found. Showing all available analyses for reference:")
        print(f"    {names[:30]}...")
        if len(names) > 30:
            print(f"    ... and {len(names) - 30} more")

    # Cleanup
    print("\n[7] Cleanup...")
    zos.disconnect()
    print("    Disconnected from OpticStudio.")

    print("\n" + "=" * 70)
    print("Diagnostic complete.")
    print("Share this output to determine if native Seidel analysis is available.")
    print("=" * 70)


if __name__ == "__main__":
    main()
