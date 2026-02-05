#!/usr/bin/env python3
"""
Diagnostic script for spot diagram ray tracing issues.
Run this on Windows with an active OpticStudio connection.

Usage:
    python diagnose_spot_diagram.py [path_to_zmx_file]

If no file is provided, uses a simple test system.
"""

import sys
import numpy as np

try:
    import zospy as zp
    print(f"✓ ZosPy version: {zp.__version__}")
except ImportError:
    print("✗ ZosPy not installed. Run: pip install zospy")
    sys.exit(1)


def diagnose_single_ray_trace(oss):
    """Diagnose what SingleRayTrace returns."""
    print("\n" + "="*60)
    print("DIAGNOSING SingleRayTrace OUTPUT STRUCTURE")
    print("="*60)

    try:
        # Try to trace a single ray at pupil center
        print("\n1. Creating SingleRayTrace analysis...")
        ray_trace = zp.analyses.raysandspots.single_ray_trace(
            oss,
            Hx=0.0,
            Hy=0.0,
            Px=0.0,
            Py=0.0,
            wavelength=1,
        )

        print(f"   Result type: {type(ray_trace)}")
        print(f"   Result dir: {[a for a in dir(ray_trace) if not a.startswith('_')]}")

        # Check for data attribute
        if hasattr(ray_trace, 'data'):
            data = ray_trace.data
            print(f"\n2. ray_trace.data exists:")
            print(f"   Type: {type(data)}")
            print(f"   Attributes: {[a for a in dir(data) if not a.startswith('_')]}")

            # Check for real_ray_trace_data
            if hasattr(data, 'real_ray_trace_data'):
                df = data.real_ray_trace_data
                print(f"\n3. data.real_ray_trace_data exists:")
                print(f"   Type: {type(df)}")
                print(f"   Shape: {df.shape if hasattr(df, 'shape') else 'N/A'}")
                print(f"   Columns: {list(df.columns) if hasattr(df, 'columns') else 'N/A'}")

                if len(df) > 0:
                    print(f"\n4. DataFrame content (last row = image surface):")
                    last_row = df.iloc[-1]
                    print(f"   last_row type: {type(last_row)}")
                    print(f"   last_row index: {last_row.index.tolist() if hasattr(last_row, 'index') else 'N/A'}")
                    print(f"\n   Full last row data:")
                    for col in df.columns:
                        val = last_row[col] if col in last_row.index else 'MISSING'
                        print(f"      {col}: {val}")

                    # Test .get() method
                    print(f"\n5. Testing .get() method on last_row:")
                    print(f"   hasattr(last_row, 'get'): {hasattr(last_row, 'get')}")

                    # Try various column name patterns
                    x_patterns = ['X', 'x', 'X-coordinate', 'x-coordinate', 'Position X', 'position_x']
                    y_patterns = ['Y', 'y', 'Y-coordinate', 'y-coordinate', 'Position Y', 'position_y']

                    print(f"\n6. Searching for X coordinate column:")
                    for pattern in x_patterns:
                        val = last_row.get(pattern, 'NOT_FOUND')
                        if val != 'NOT_FOUND':
                            print(f"   ✓ Found '{pattern}': {val}")
                        else:
                            print(f"   ✗ '{pattern}' not found")

                    print(f"\n7. Searching for Y coordinate column:")
                    for pattern in y_patterns:
                        val = last_row.get(pattern, 'NOT_FOUND')
                        if val != 'NOT_FOUND':
                            print(f"   ✓ Found '{pattern}': {val}")
                        else:
                            print(f"   ✗ '{pattern}' not found")

                    # Check for error code
                    print(f"\n8. Searching for error code:")
                    error_patterns = ['error_code', 'Error', 'error', 'ErrorCode', 'Vignetted', 'vignetted']
                    for pattern in error_patterns:
                        val = last_row.get(pattern, 'NOT_FOUND')
                        if val != 'NOT_FOUND':
                            print(f"   ✓ Found '{pattern}': {val}")
                        else:
                            print(f"   ✗ '{pattern}' not found")
                else:
                    print("   ✗ DataFrame is empty!")
            else:
                print("\n3. ✗ data.real_ray_trace_data does NOT exist")
                print(f"   Available attributes: {[a for a in dir(data) if not a.startswith('_')]}")
        else:
            print("\n2. ✗ ray_trace.data does NOT exist")

        # Also try the new-style API
        print("\n" + "-"*60)
        print("TRYING ALTERNATIVE API: zp.analyses.new_analysis")
        print("-"*60)

        try:
            from zospy.api.codecs import AnalysisIDM
            analysis = zp.analyses.new_analysis(oss, AnalysisIDM.SingleRayTrace)
            analysis.settings.Hx = 0.0
            analysis.settings.Hy = 0.0
            analysis.settings.Px = 0.0
            analysis.settings.Py = 0.0
            analysis.settings.Wavelength = 1
            analysis.apply_and_wait_for_completion()

            print(f"   New-style analysis type: {type(analysis)}")
            if hasattr(analysis, 'results'):
                print(f"   results type: {type(analysis.results)}")
                print(f"   results dir: {[a for a in dir(analysis.results) if not a.startswith('_')]}")
        except Exception as e:
            print(f"   ✗ New-style API failed: {e}")

    except Exception as e:
        print(f"\n✗ SingleRayTrace failed: {e}")
        import traceback
        traceback.print_exc()


def diagnose_standard_spot(oss):
    """Diagnose what StandardSpot returns."""
    print("\n" + "="*60)
    print("DIAGNOSING StandardSpot OUTPUT STRUCTURE")
    print("="*60)

    try:
        from zospy.api.codecs import AnalysisIDM

        print("\n1. Creating StandardSpot analysis...")
        analysis = zp.analyses.new_analysis(oss, AnalysisIDM.StandardSpot)
        analysis.apply_and_wait_for_completion()

        print(f"   Analysis type: {type(analysis)}")
        print(f"   Analysis dir: {[a for a in dir(analysis) if not a.startswith('_')]}")

        if hasattr(analysis, 'results'):
            results = analysis.results
            print(f"\n2. analysis.results exists:")
            print(f"   Type: {type(results)}")
            print(f"   Attributes: {[a for a in dir(results) if not a.startswith('_')]}")

            # Try to get spot data
            if hasattr(results, 'data'):
                data = results.data
                print(f"\n3. results.data:")
                print(f"   Type: {type(data)}")
                if hasattr(data, 'columns'):
                    print(f"   Columns: {list(data.columns)}")
                if hasattr(data, 'shape'):
                    print(f"   Shape: {data.shape}")
                if hasattr(data, 'head'):
                    print(f"\n   First few rows:")
                    print(data.head())

            # Try various attribute names for spot data
            spot_attrs = ['spot_data', 'SpotData', 'data', 'Data', 'spot_diagram_data']
            for attr in spot_attrs:
                if hasattr(results, attr):
                    print(f"\n   Found results.{attr}: {type(getattr(results, attr))}")

    except Exception as e:
        print(f"\n✗ StandardSpot failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    print("="*60)
    print("ZEMAX SPOT DIAGRAM DIAGNOSTIC TOOL")
    print("="*60)

    # Connect to OpticStudio
    print("\n1. Connecting to OpticStudio...")
    try:
        zos = zp.ZOS()
        print(f"   ZOS instance created: {type(zos)}")

        oss = zos.connect(mode="extension")
        print(f"   Connected in extension mode: {type(oss)}")
    except Exception as e:
        print(f"   ✗ Extension mode failed: {e}")
        print("   Trying standalone mode...")
        try:
            oss = zos.connect(mode="standalone")
            print(f"   Connected in standalone mode: {type(oss)}")
        except Exception as e2:
            print(f"   ✗ Standalone mode also failed: {e2}")
            sys.exit(1)

    # Load system
    if len(sys.argv) > 1:
        zmx_path = sys.argv[1]
        print(f"\n2. Loading system from: {zmx_path}")
        try:
            oss.load(zmx_path)
            print("   ✓ System loaded")
        except Exception as e:
            print(f"   ✗ Failed to load: {e}")
            sys.exit(1)
    else:
        print("\n2. Creating simple test system...")
        try:
            # Create a simple singlet lens
            oss.new()
            lde = oss.LDE

            # Set up aperture and field
            oss.SystemData.Aperture.ApertureType = 1  # EPD
            oss.SystemData.Aperture.ApertureValue = 10.0

            # Add a field
            oss.SystemData.Fields.AddField(0, 5, 1.0)

            # Add surfaces for a simple lens
            # Surface 1: Front of lens
            lde.InsertNewSurfaceAt(1)
            lde.GetSurfaceAt(1).Radius = 50.0
            lde.GetSurfaceAt(1).Thickness = 5.0
            lde.GetSurfaceAt(1).Material = "N-BK7"

            # Surface 2: Back of lens
            lde.GetSurfaceAt(2).Radius = -50.0
            lde.GetSurfaceAt(2).Thickness = 95.0

            print("   ✓ Test system created (simple singlet)")
        except Exception as e:
            print(f"   ✗ Failed to create test system: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    # Show system info
    print(f"\n3. System info:")
    try:
        print(f"   Number of surfaces: {oss.LDE.NumberOfSurfaces}")
        print(f"   Number of fields: {oss.SystemData.Fields.NumberOfFields}")
        print(f"   Number of wavelengths: {oss.SystemData.Wavelengths.NumberOfWavelengths}")
    except Exception as e:
        print(f"   ✗ Could not get system info: {e}")

    # Run diagnostics
    diagnose_single_ray_trace(oss)
    diagnose_standard_spot(oss)

    print("\n" + "="*60)
    print("DIAGNOSTIC COMPLETE")
    print("="*60)
    print("\nNext steps:")
    print("1. Check the column names printed above")
    print("2. Update zospy_handler.py to use the correct column names")
    print("3. If StandardSpot works, prefer that over manual ray tracing")


if __name__ == "__main__":
    main()
