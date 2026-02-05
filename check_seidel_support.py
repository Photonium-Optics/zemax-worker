#!/usr/bin/env python3
"""
Diagnostic script to check if SeidelCoefficients analysis is available in ZOSPy.
"""

import zospy as zp
import os


def find_sample_file(oss):
    """Find a sample .zmx file from OpticStudio installation."""
    possible_paths = [
        r"C:\Users\Public\Documents\Zemax\Samples\Sequential\Objectives\Double Gauss 28 degree field.zmx",
        os.path.expanduser(r"~\Documents\Zemax\Samples\Sequential\Objectives\Double Gauss 28 degree field.zmx"),
    ]
    for path in possible_paths:
        if os.path.exists(path):
            return path
    return None


def main():
    print("=" * 70)
    print("ZOSPy Seidel Support Diagnostic")
    print("=" * 70)

    zos = zp.ZOS()
    oss = zos.connect(mode="standalone")

    print(f"\nZOSPy version: {getattr(zp, '__version__', 'unknown')}")

    _ = zp.constants

    # Load sample file
    print("\nLoading sample lens file...")
    sample_file = find_sample_file(oss)
    if sample_file:
        oss.load(sample_file)
        print(f"Loaded: {sample_file}")
    else:
        print("No sample file found!")
        zos.disconnect()
        return

    idm = zp.constants.Analysis.AnalysisIDM

    # Run SeidelCoefficients
    print("\n" + "=" * 70)
    print("Running SeidelCoefficients analysis...")
    print("=" * 70)

    an = zp.analyses.new_analysis(
        oss,
        idm.SeidelCoefficients,
        settings_first=True
    )

    # Check Settings before running
    print("\n--- SETTINGS ---")
    settings = an.Settings
    print(f"Settings type: {type(settings)}")
    settings_attrs = [a for a in dir(settings) if not a.startswith("_")]
    print(f"Settings attributes: {settings_attrs}")

    # Try to read each setting
    for attr in settings_attrs:
        try:
            val = getattr(settings, attr)
            if not callable(val):
                print(f"  {attr}: {val}")
        except Exception as e:
            print(f"  {attr}: ERROR - {e}")

    # Run analysis
    print("\n--- RUNNING ANALYSIS ---")
    an.ApplyAndWaitForCompletion()

    # Check all result properties
    print("\n--- RESULTS ---")
    res = an.Results
    print(f"Results type: {type(res)}")

    # Print ALL attributes and their values
    res_attrs = [a for a in dir(res) if not a.startswith("_")]
    print(f"\nAll Results attributes ({len(res_attrs)}):")

    for attr in res_attrs:
        try:
            val = getattr(res, attr)
            if callable(val):
                print(f"  {attr}(): <method>")
            else:
                print(f"  {attr}: {val}")
        except Exception as e:
            print(f"  {attr}: ERROR - {e}")

    # Try GetTextFile with a temp file path
    print("\n--- TEXT FILE OUTPUT ---")
    try:
        import tempfile
        temp_path = os.path.join(tempfile.gettempdir(), "seidel_output.txt")
        res.GetTextFile(temp_path)
        if os.path.exists(temp_path):
            with open(temp_path, 'r') as f:
                content = f.read()
            print(f"Text file content ({len(content)} chars):")
            print("-" * 50)
            print(content[:2000])
            print("-" * 50)
            if len(content) > 2000:
                print(f"... truncated ({len(content)} total chars)")
    except Exception as e:
        print(f"GetTextFile error: {e}")

    # Try HeaderData
    print("\n--- HEADER DATA ---")
    try:
        hdr = res.HeaderData
        print(f"HeaderData type: {type(hdr)}")
        if hdr:
            hdr_attrs = [a for a in dir(hdr) if not a.startswith("_")]
            print(f"HeaderData attributes: {hdr_attrs}")
            for attr in hdr_attrs[:20]:
                try:
                    val = getattr(hdr, attr)
                    if not callable(val):
                        print(f"  {attr}: {val}")
                except:
                    pass
    except Exception as e:
        print(f"HeaderData error: {e}")

    # Try MetaData
    print("\n--- METADATA ---")
    try:
        meta = res.MetaData
        print(f"MetaData type: {type(meta)}")
        if meta:
            meta_attrs = [a for a in dir(meta) if not a.startswith("_")]
            for attr in meta_attrs[:20]:
                try:
                    val = getattr(meta, attr)
                    if not callable(val):
                        print(f"  {attr}: {val}")
                except:
                    pass
    except Exception as e:
        print(f"MetaData error: {e}")

    # Check wrapper attributes too
    print("\n--- WRAPPER ATTRIBUTES ---")
    print(f"header_data: {an.header_data}")
    print(f"messages: {an.messages}")
    print(f"metadata: {an.metadata}")

    an.Close()

    # Also try SeidelDiagram
    print("\n" + "=" * 70)
    print("Running SeidelDiagram analysis...")
    print("=" * 70)

    an2 = zp.analyses.new_analysis(
        oss,
        idm.SeidelDiagram,
        settings_first=True
    )

    an2.ApplyAndWaitForCompletion()
    res2 = an2.Results

    print(f"\nNumberOfDataGrids: {res2.NumberOfDataGrids}")
    print(f"NumberOfDataSeries: {res2.NumberOfDataSeries}")

    # Try text file for diagram too
    try:
        import tempfile
        temp_path = os.path.join(tempfile.gettempdir(), "seidel_diagram.txt")
        res2.GetTextFile(temp_path)
        if os.path.exists(temp_path):
            with open(temp_path, 'r') as f:
                content = f.read()
            print(f"\nSeidelDiagram text ({len(content)} chars):")
            print("-" * 50)
            print(content[:2000])
            print("-" * 50)
    except Exception as e:
        print(f"GetTextFile error: {e}")

    an2.Close()

    # Cleanup
    print("\nDisconnecting...")
    zos.disconnect()
    print("Done.")


if __name__ == "__main__":
    main()
