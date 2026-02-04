#!/usr/bin/env python3
"""
Diagnostic script to check if SeidelCoefficients analysis is available in ZOSPy.

Run this on the Windows machine with OpticStudio installed:
    python check_seidel_support.py

This will tell us whether we can use the native Seidel analysis instead of
deriving Sernike from Zernike coefficients.
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

    # Surface 0 is object (already exists)
    # Surface 1 will be front of lens
    # Surface 2 will be back of lens
    # Surface 3 is image (already exists as surface 1, we'll insert before it)

    # Insert surfaces for a simple singlet
    lde.InsertNewSurfaceAt(1)  # Front surface
    lde.InsertNewSurfaceAt(2)  # Back surface

    # Configure front surface (Surface 1)
    surf1 = lde.GetSurfaceAt(1)
    surf1.Radius = float(50.0)  # 50mm radius
    surf1.Thickness = float(5.0)  # 5mm thick
    surf1.Material = "N-BK7"
    surf1.SemiDiameter = float(10.0)

    # Configure back surface (Surface 2)
    surf2 = lde.GetSurfaceAt(2)
    surf2.Radius = float(-50.0)  # -50mm radius (symmetric biconvex)
    surf2.Thickness = float(45.0)  # Distance to image
    surf2.SemiDiameter = float(10.0)

    # Image surface is Surface 3
    print("    Simple singlet lens created (biconvex, f~50mm)")


def main():
    print("=" * 70)
    print("ZOSPy Seidel Support Diagnostic")
    print("=" * 70)

    # Connect to OpticStudio
    print("\n[1] Connecting to OpticStudio...")
    zos = zp.ZOS()
    oss = zos.connect(mode="standalone")

    # Print version info
    print(f"\n[2] Version Info:")
    print(f"    ZOSPy version: {getattr(zp, '__version__', 'unknown')}")

    # Force-load constants (they're dynamic)
    print("\n[3] Loading ZOSPy constants...")
    _ = zp.constants

    # Set up a simple lens system
    print("\n[4] Setting up test lens system...")
    try:
        setup_simple_lens(oss)
    except Exception as e:
        print(f"    ERROR setting up lens: {e}")
        import traceback
        traceback.print_exc()
        zos.disconnect()
        return

    # Check AnalysisIDM for Seidel-related entries
    print("\n[5] Checking AnalysisIDM for Seidel analyses...")
    idm = zp.constants.Analysis.AnalysisIDM
    names = [n for n in dir(idm) if not n.startswith("_")]

    print(f"    Total analysis types available: {len(names)}")

    seidelish = [n for n in names if "seidel" in n.lower()]
    print(f"    Seidel entries: {seidelish}")

    # Try SeidelCoefficients and extract actual data
    print("\n[6] Running SeidelCoefficients analysis...")

    if "SeidelCoefficients" in names:
        try:
            an = zp.analyses.new_analysis(
                oss,
                idm.SeidelCoefficients,
                settings_first=True
            )

            # Run the analysis
            an.ApplyAndWaitForCompletion()
            res = an.Results

            # Explore DataGrids
            num_grids = res.DataGrids if res.DataGrids is not None else 0
            print(f"\n    Number of DataGrids: {num_grids}")

            for i in range(num_grids):
                try:
                    grid = res.GetDataGrid(i)
                    print(f"\n    --- DataGrid[{i}] ---")

                    # Check grid attributes
                    grid_attrs = [a for a in dir(grid) if not a.startswith("_")]
                    print(f"    Attributes: {grid_attrs[:20]}")

                    # Try to get dimensions
                    rows = getattr(grid, 'Rows', None)
                    cols = getattr(grid, 'Cols', None)
                    print(f"    Dimensions: {rows} rows x {cols} cols")

                    # Try to iterate values
                    if rows and cols:
                        print(f"    Grid data:")
                        for row in range(min(rows, 15)):  # First 15 rows
                            row_data = []
                            for col in range(min(cols, 6)):  # First 6 cols
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
            num_series = res.DataSeries if res.DataSeries is not None else 0
            print(f"\n    Number of DataSeries: {num_series}")

            for i in range(min(num_series, 10)):
                try:
                    series = res.GetDataSeries(i)
                    print(f"\n    --- DataSeries[{i}] ---")

                    series_attrs = [a for a in dir(series) if not a.startswith("_")]
                    print(f"    Attributes: {series_attrs[:15]}")

                    if hasattr(series, 'Description'):
                        print(f"    Description: {series.Description}")
                    if hasattr(series, 'SeriesLabel'):
                        print(f"    SeriesLabel: {series.SeriesLabel}")
                    if hasattr(series, 'NumData'):
                        num_data = series.NumData
                        print(f"    NumData: {num_data}")
                        # Try to get values
                        if num_data and num_data > 0:
                            print(f"    Values:")
                            for j in range(min(num_data, 10)):
                                try:
                                    if hasattr(series, 'XData') and hasattr(series, 'YData'):
                                        x = series.XData.Data(j)
                                        y = series.YData.Data(j)
                                        print(f"      [{j}]: x={x:.6f}, y={y:.6f}")
                                except Exception as e:
                                    print(f"      [{j}]: error - {e}")
                except Exception as e:
                    print(f"    DataSeries[{i}] error: {e}")

            an.Close()
            print("\n    SeidelCoefficients analysis completed!")

        except Exception as e:
            print(f"    ERROR: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("    SeidelCoefficients not found!")

    # Also try SeidelDiagram
    print("\n[7] Running SeidelDiagram analysis...")
    if "SeidelDiagram" in names:
        try:
            an = zp.analyses.new_analysis(
                oss,
                idm.SeidelDiagram,
                settings_first=True
            )

            an.ApplyAndWaitForCompletion()
            res = an.Results

            num_grids = res.DataGrids if res.DataGrids is not None else 0
            num_series = res.DataSeries if res.DataSeries is not None else 0
            print(f"    DataGrids: {num_grids}, DataSeries: {num_series}")

            # Show first grid if available
            if num_grids > 0:
                grid = res.GetDataGrid(0)
                rows = getattr(grid, 'Rows', 0)
                cols = getattr(grid, 'Cols', 0)
                print(f"    Grid[0]: {rows}x{cols}")

            an.Close()
        except Exception as e:
            print(f"    ERROR: {e}")

    # Cleanup
    print("\n[8] Cleanup...")
    zos.disconnect()
    print("    Disconnected from OpticStudio.")

    print("\n" + "=" * 70)
    print("Diagnostic complete. Share this output to see the Seidel data structure.")
    print("=" * 70)


if __name__ == "__main__":
    main()
