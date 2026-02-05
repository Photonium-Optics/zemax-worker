#!/usr/bin/env python3
"""
Diagnostic script for wavefront analysis issues.
Run this on Windows with an active OpticStudio connection.

Usage:
    python diagnose_wavefront.py [path_to_zmx_file]

If no file is provided, uses a simple test system.
"""

import sys

try:
    import zospy as zp
    print(f"✓ ZosPy version: {zp.__version__}")
except ImportError:
    print("✗ ZosPy not installed. Run: pip install zospy")
    sys.exit(1)


def diagnose_zernike_analysis(oss):
    """Diagnose what ZernikeStandardCoefficients returns."""
    print("\n" + "="*60)
    print("DIAGNOSING ZernikeStandardCoefficients OUTPUT")
    print("="*60)

    try:
        from zospy.api.codecs import AnalysisIDM

        print("\n1. Creating ZernikeStandardCoefficients analysis...")
        analysis = zp.analyses.new_analysis(oss, AnalysisIDM.ZernikeStandardCoefficients)

        # Configure for field 1, wavelength 1
        if hasattr(analysis, 'settings') or hasattr(analysis, 'Settings'):
            settings = getattr(analysis, 'settings', None) or getattr(analysis, 'Settings', None)
            print(f"   Settings type: {type(settings)}")
            print(f"   Settings dir: {[a for a in dir(settings) if not a.startswith('_')]}")

            # Try to set field and wavelength
            field_attrs = ['Field', 'field', 'FieldNumber', 'field_number']
            wave_attrs = ['Wavelength', 'wavelength', 'WavelengthNumber', 'wavelength_number']

            for attr in field_attrs:
                if hasattr(settings, attr):
                    print(f"   Setting {attr} = 1")
                    setattr(settings, attr, 1)
                    break

            for attr in wave_attrs:
                if hasattr(settings, attr):
                    print(f"   Setting {attr} = 1")
                    setattr(settings, attr, 1)
                    break

        # Run analysis
        print("\n2. Running analysis...")
        if hasattr(analysis, 'apply_and_wait_for_completion'):
            analysis.apply_and_wait_for_completion()
        elif hasattr(analysis, 'ApplyAndWaitForCompletion'):
            analysis.ApplyAndWaitForCompletion()
        else:
            print("   ✗ No apply method found!")
            return

        # Check results
        print("\n3. Checking results...")
        results = getattr(analysis, 'results', None) or getattr(analysis, 'Results', None)

        if results is None:
            print("   ✗ No results returned!")
            return

        print(f"   Results type: {type(results)}")
        print(f"   Results dir: {[a for a in dir(results) if not a.startswith('_')]}")

        # Check for data
        if hasattr(results, 'data') or hasattr(results, 'Data'):
            data = getattr(results, 'data', None) or getattr(results, 'Data', None)
            print(f"\n4. Results.data:")
            print(f"   Type: {type(data)}")
            print(f"   Dir: {[a for a in dir(data) if not a.startswith('_')]}")

            # Look for P-V and RMS
            pv_attrs = ['peak_to_valley_to_chief', 'peak_to_valley_to_centroid',
                       'peak_to_valley', 'PeakToValley', 'pv', 'P-V']
            rms_attrs = ['rms', 'RMS', 'rms_to_chief', 'rms_to_centroid',
                        'from_integration_of_the_rays']

            print(f"\n5. Looking for P-V value:")
            for attr in pv_attrs:
                if hasattr(data, attr):
                    val = getattr(data, attr)
                    print(f"   ✓ Found data.{attr}: {val}")

            print(f"\n6. Looking for RMS value:")
            for attr in rms_attrs:
                if hasattr(data, attr):
                    val = getattr(data, attr)
                    print(f"   ✓ Found data.{attr}: {val}")
                    # If it's nested, explore
                    if hasattr(val, '__dict__'):
                        print(f"      Nested dir: {[a for a in dir(val) if not a.startswith('_')]}")

            # Print all numeric attributes
            print(f"\n7. All data attributes with values:")
            for attr in dir(data):
                if not attr.startswith('_'):
                    try:
                        val = getattr(data, attr)
                        if isinstance(val, (int, float)):
                            print(f"      {attr}: {val}")
                    except:
                        pass

    except Exception as e:
        print(f"\n✗ ZernikeStandardCoefficients failed: {e}")
        import traceback
        traceback.print_exc()


def diagnose_wavefront_map(oss):
    """Diagnose what WavefrontMap returns."""
    print("\n" + "="*60)
    print("DIAGNOSING WavefrontMap OUTPUT")
    print("="*60)

    try:
        from zospy.api.codecs import AnalysisIDM

        print("\n1. Creating WavefrontMap analysis...")
        analysis = zp.analyses.new_analysis(oss, AnalysisIDM.WavefrontMap)

        # Configure
        if hasattr(analysis, 'settings') or hasattr(analysis, 'Settings'):
            settings = getattr(analysis, 'settings', None) or getattr(analysis, 'Settings', None)
            print(f"   Settings dir: {[a for a in dir(settings) if not a.startswith('_')]}")

        # Run analysis
        print("\n2. Running analysis...")
        if hasattr(analysis, 'apply_and_wait_for_completion'):
            analysis.apply_and_wait_for_completion()
        elif hasattr(analysis, 'ApplyAndWaitForCompletion'):
            analysis.ApplyAndWaitForCompletion()

        # Check results
        print("\n3. Checking results...")
        results = getattr(analysis, 'results', None) or getattr(analysis, 'Results', None)

        if results is None:
            print("   ✗ No results returned!")
            return

        print(f"   Results type: {type(results)}")
        print(f"   Results dir: {[a for a in dir(results) if not a.startswith('_')]}")

        # Check for wavefront data
        if hasattr(results, 'data') or hasattr(results, 'Data'):
            data = getattr(results, 'data', None) or getattr(results, 'Data', None)
            print(f"\n4. Results.data:")
            print(f"   Type: {type(data)}")

            if hasattr(data, 'shape'):
                print(f"   Shape: {data.shape}")

            if hasattr(data, 'columns'):
                print(f"   Columns: {list(data.columns)}")

            # Look for wavefront map array
            wavefront_attrs = ['wavefront', 'Wavefront', 'map', 'Map', 'data', 'Data', 'values']
            print(f"\n5. Looking for wavefront map:")
            for attr in wavefront_attrs:
                if hasattr(data, attr):
                    val = getattr(data, attr)
                    print(f"   ✓ Found data.{attr}: type={type(val)}")
                    if hasattr(val, 'shape'):
                        print(f"      Shape: {val.shape}")

    except Exception as e:
        print(f"\n✗ WavefrontMap failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    print("="*60)
    print("ZEMAX WAVEFRONT DIAGNOSTIC TOOL")
    print("="*60)

    # Connect to OpticStudio
    print("\n1. Connecting to OpticStudio...")
    try:
        zos = zp.ZOS()
        oss = zos.connect(mode="extension")
        print(f"   Connected in extension mode")
    except Exception as e:
        print(f"   Extension mode failed: {e}")
        print("   Trying standalone mode...")
        try:
            oss = zos.connect(mode="standalone")
            print(f"   Connected in standalone mode")
        except Exception as e2:
            print(f"   Standalone mode also failed: {e2}")
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
            oss.new()
            lde = oss.LDE

            # Use enum constants for Python.NET 3.0 compatibility
            oss.SystemData.Aperture.ApertureType = zp.constants.SystemData.ZemaxApertureType.EntrancePupilDiameter
            oss.SystemData.Aperture.ApertureValue = float(10.0)

            oss.SystemData.Fields.AddField(0, 5, 1.0)

            lde.InsertNewSurfaceAt(1)
            lde.GetSurfaceAt(1).Radius = float(50.0)
            lde.GetSurfaceAt(1).Thickness = float(5.0)
            lde.GetSurfaceAt(1).Material = "N-BK7"

            lde.GetSurfaceAt(2).Radius = float(-50.0)
            lde.GetSurfaceAt(2).Thickness = float(95.0)

            print("   ✓ Test system created")
        except Exception as e:
            print(f"   ✗ Failed to create test system: {e}")
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
    diagnose_zernike_analysis(oss)
    diagnose_wavefront_map(oss)

    print("\n" + "="*60)
    print("DIAGNOSTIC COMPLETE")
    print("="*60)
    print("\nNext steps:")
    print("1. Check the attribute names printed above")
    print("2. Update zospy_handler.py get_wavefront() to use correct paths")


if __name__ == "__main__":
    main()
