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
        # ZosPy 2.x: Use SingleRayTrace class (not function), lowercase params
        ray_trace = zp.analyses.raysandspots.SingleRayTrace(
            hx=0.0,
            hy=0.0,
            px=0.0,
            py=0.0,
            wavelength=1,
            field=1,
        )
        result = ray_trace.run(oss)

        print(f"   ray_trace type: {type(ray_trace)}")
        print(f"   result type: {type(result)}")
        print(f"   result dir: {[a for a in dir(result) if not a.startswith('_')]}")

        # Check for data attribute on result (not ray_trace)
        if hasattr(result, 'data'):
            data = result.data
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
            print("\n2. ✗ result.data does NOT exist")
            print(f"   Available attributes on result: {[a for a in dir(result) if not a.startswith('_')]}")

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
        # ZosPy 2.x: Use zp.constants.Analysis.AnalysisIDM
        idm = zp.constants.Analysis.AnalysisIDM

        print("\n1. Creating StandardSpot analysis...")
        analysis = zp.analyses.new_analysis(oss, idm.StandardSpot, settings_first=True)
        analysis.ApplyAndWaitForCompletion()

        print(f"   Analysis type: {type(analysis)}")
        print(f"   Analysis dir: {[a for a in dir(analysis) if not a.startswith('_')]}")

        # Try both lowercase and uppercase Results attribute
        results = getattr(analysis, 'Results', None) or getattr(analysis, 'results', None)
        if results is not None:
            print(f"\n2. analysis.Results exists:")
            print(f"   Type: {type(results)}")
            print(f"   Attributes: {[a for a in dir(results) if not a.startswith('_')]}")

            # Try to get spot data - check both Data and data
            data = getattr(results, 'Data', None) or getattr(results, 'data', None)
            if data is not None:
                print(f"\n3. results.Data:")
                print(f"   Type: {type(data)}")
                if hasattr(data, 'columns'):
                    print(f"   Columns: {list(data.columns)}")
                if hasattr(data, 'shape'):
                    print(f"   Shape: {data.shape}")
                if hasattr(data, 'head'):
                    print(f"\n   First few rows:")
                    print(data.head())

            # Try various attribute names for spot data
            spot_attrs = ['spot_data', 'SpotData', 'data', 'Data', 'spot_diagram_data', 'AiryRadius']
            for attr in spot_attrs:
                if hasattr(results, attr):
                    print(f"\n   Found results.{attr}: {getattr(results, attr)}")

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

            # Set up aperture and field (use enum constants for Python.NET 3.0 compatibility)
            oss.SystemData.Aperture.ApertureType = zp.constants.SystemData.ZemaxApertureType.EntrancePupilDiameter
            oss.SystemData.Aperture.ApertureValue = float(10.0)

            # Add a field
            oss.SystemData.Fields.AddField(0, 5, 1.0)

            # Add surfaces for a simple lens
            # Surface 1: Front of lens
            lde.InsertNewSurfaceAt(1)
            lde.GetSurfaceAt(1).Radius = float(50.0)
            lde.GetSurfaceAt(1).Thickness = float(5.0)
            lde.GetSurfaceAt(1).Material = "N-BK7"

            # Surface 2: Back of lens
            lde.GetSurfaceAt(2).Radius = float(-50.0)
            lde.GetSurfaceAt(2).Thickness = float(95.0)

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
