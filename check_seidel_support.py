#!/usr/bin/env python3
"""
Diagnostic script to check if SeidelCoefficients analysis is available in ZOSPy.

Run this on the Windows machine with OpticStudio installed:
    python check_seidel_support.py
"""

import zospy as zp


def setup_simple_lens(oss):
    """Create a simple singlet lens for testing Seidel analysis."""
    print("    Creating simple singlet lens for testing...")

    # Create new system
    oss.new()

    # Set up system basics
    oss.SystemData.Aperture.ApertureValue = 10.0  # EPD = 10mm
    oss.SystemData.Wavelengths.GetWavelength(1).Wavelength = 0.55  # 550nm

    # Add a field point
    oss.SystemData.Fields.AddField(0, 5, 1.0)  # 5 degree field

    # Get the lens data editor
    lde = oss.LDE

    # Insert surfaces for a simple singlet
    lde.InsertNewSurfaceAt(1)  # Front surface
    lde.InsertNewSurfaceAt(2)  # Back surface

    # Configure front surface (Surface 1)
    surf1 = lde.GetSurfaceAt(1)
    surf1.Radius = float(50.0)
    surf1.Thickness = float(5.0)
    surf1.Material = "N-BK7"
    surf1.SemiDiameter = float(10.0)

    # Configure back surface (Surface 2)
    surf2 = lde.GetSurfaceAt(2)
    surf2.Radius = float(-50.0)
    surf2.Thickness = float(45.0)
    surf2.SemiDiameter = float(10.0)

    print("    Simple singlet lens created (biconvex, f~50mm)")


def safe_len(obj):
    """Safely get length of .NET array or Python object."""
    if obj is None:
        return 0
    try:
        # Try Python len first
        return len(obj)
    except TypeError:
        pass
    try:
        # Try .NET Length property
        return obj.Length
    except:
        pass
    try:
        # Try Count property
        return obj.Count
    except:
        pass
    return 0


def main():
    print("=" * 70)
    print("ZOSPy Seidel Support Diagnostic")
    print("=" * 70)

    # Connect to OpticStudio
    print("\n[1] Connecting to OpticStudio...")
    zos = zp.ZOS()
    oss = zos.connect(mode="standalone")

    print(f"\n[2] Version Info:")
    print(f"    ZOSPy version: {getattr(zp, '__version__', 'unknown')}")

    print("\n[3] Loading ZOSPy constants...")
    _ = zp.constants

    print("\n[4] Setting up test lens system...")
    try:
        setup_simple_lens(oss)
    except Exception as e:
        print(f"    ERROR setting up lens: {e}")
        import traceback
        traceback.print_exc()
        zos.disconnect()
        return

    print("\n[5] Checking AnalysisIDM for Seidel analyses...")
    idm = zp.constants.Analysis.AnalysisIDM
    names = [n for n in dir(idm) if not n.startswith("_")]
    print(f"    Total analysis types: {len(names)}")

    seidelish = [n for n in names if "seidel" in n.lower()]
    print(f"    Seidel entries: {seidelish}")

    # Run SeidelCoefficients
    print("\n[6] Running SeidelCoefficients analysis...")

    if "SeidelCoefficients" in names:
        try:
            an = zp.analyses.new_analysis(
                oss,
                idm.SeidelCoefficients,
                settings_first=True
            )

            an.ApplyAndWaitForCompletion()
            res = an.Results

            # Check all result attributes
            print(f"\n    Results object type: {type(res)}")
            res_attrs = [a for a in dir(res) if not a.startswith("_")]
            print(f"    Results attributes: {res_attrs}")

            # Try NumberOfDataGrids if it exists
            if hasattr(res, 'NumberOfDataGrids'):
                print(f"    NumberOfDataGrids: {res.NumberOfDataGrids}")

            # Explore DataGrids - it might be an array
            print(f"\n    DataGrids raw value: {res.DataGrids}")
            print(f"    DataGrids type: {type(res.DataGrids)}")

            num_grids = safe_len(res.DataGrids)
            print(f"    DataGrids count: {num_grids}")

            # Try to iterate DataGrids if it's an array
            if num_grids > 0:
                for i in range(num_grids):
                    try:
                        grid = res.GetDataGrid(i)
                        print(f"\n    --- DataGrid[{i}] ---")
                        rows = getattr(grid, 'Rows', None)
                        cols = getattr(grid, 'Cols', None)
                        print(f"    Dimensions: {rows}x{cols}")

                        if rows and cols and rows > 0 and cols > 0:
                            print(f"    Data:")
                            for row in range(min(rows, 12)):
                                row_data = []
                                for col in range(min(cols, 6)):
                                    try:
                                        val = grid.GetDouble(row, col)
                                        row_data.append(f"{val:12.6f}")
                                    except:
                                        try:
                                            val = grid.GetString(row, col)
                                            row_data.append(f"{str(val):>12}")
                                        except:
                                            row_data.append("     ?      ")
                                print(f"      [{row:2d}]: {' '.join(row_data)}")
                    except Exception as e:
                        print(f"    DataGrid[{i}] error: {e}")

            # Explore DataSeries
            print(f"\n    DataSeries raw value: {res.DataSeries}")
            print(f"    DataSeries type: {type(res.DataSeries)}")

            num_series = safe_len(res.DataSeries)
            print(f"    DataSeries count: {num_series}")

            if num_series > 0:
                for i in range(min(num_series, 10)):
                    try:
                        series = res.GetDataSeries(i)
                        print(f"\n    --- DataSeries[{i}] ---")
                        if hasattr(series, 'Description'):
                            print(f"    Description: {series.Description}")
                        if hasattr(series, 'SeriesLabel'):
                            print(f"    SeriesLabel: {series.SeriesLabel}")
                    except Exception as e:
                        print(f"    DataSeries[{i}] error: {e}")

            # Try alternative: maybe data is in header_data or other attributes
            print("\n    Checking analysis wrapper attributes...")
            an_attrs = [a for a in dir(an) if not a.startswith("_")]
            print(f"    Analysis attributes: {an_attrs}")

            if hasattr(an, 'header_data'):
                print(f"    header_data: {an.header_data}")
            if hasattr(an, 'metadata'):
                print(f"    metadata: {an.metadata}")
            if hasattr(an, 'messages'):
                print(f"    messages: {an.messages}")

            an.Close()
            print("\n    SeidelCoefficients completed!")

        except Exception as e:
            print(f"    ERROR: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("    SeidelCoefficients not found!")

    # Cleanup
    print("\n[7] Cleanup...")
    zos.disconnect()
    print("    Disconnected.")

    print("\n" + "=" * 70)
    print("Done. Share output to see the Seidel data structure.")
    print("=" * 70)


if __name__ == "__main__":
    main()
