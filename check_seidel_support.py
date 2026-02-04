#!/usr/bin/env python3
"""
Diagnostic script to check if SeidelCoefficients analysis is available in ZOSPy.

Run this on the Windows machine with OpticStudio installed:
    python check_seidel_support.py

This will tell us whether we can use the native Seidel analysis instead of
deriving Seidel from Zernike coefficients.
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

    # Print version info
    print(f"\n[2] Version Info:")
    print(f"    ZOSPy version: {getattr(zp, '__version__', 'unknown')}")

    # Force-load constants (they're dynamic)
    print("\n[3] Loading ZOSPy constants...")
    _ = zp.constants

    # Check AnalysisIDM for Seidel-related entries
    print("\n[4] Checking AnalysisIDM for Seidel analyses...")
    idm = zp.constants.Analysis.AnalysisIDM
    names = [n for n in dir(idm) if not n.startswith("_")]

    print(f"    Total analysis types available: {len(names)}")

    seidelish = [n for n in names if "seidel" in n.lower()]
    print(f"    AnalysisIDM entries containing 'Seidel': {seidelish if seidelish else 'NONE FOUND'}")

    # Try SeidelCoefficients and extract actual data
    print("\n[5] Running SeidelCoefficients analysis and extracting data...")

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
            print(f"\n    Number of DataGrids: {res.DataGrids}")
            for i in range(res.DataGrids):
                try:
                    grid = res.GetDataGrid(i)
                    print(f"\n    --- DataGrid[{i}] ---")
                    print(f"    Type: {type(grid)}")

                    # Check grid attributes
                    grid_attrs = [a for a in dir(grid) if not a.startswith("_")]
                    print(f"    Attributes: {grid_attrs}")

                    # Try to get dimensions
                    if hasattr(grid, 'Rows'):
                        print(f"    Rows: {grid.Rows}")
                    if hasattr(grid, 'Cols'):
                        print(f"    Cols: {grid.Cols}")
                    if hasattr(grid, 'Values'):
                        print(f"    Values: {grid.Values}")

                    # Try to iterate values
                    if hasattr(grid, 'Rows') and hasattr(grid, 'Cols'):
                        print(f"    Grid data ({grid.Rows}x{grid.Cols}):")
                        for row in range(min(grid.Rows, 10)):  # First 10 rows
                            row_data = []
                            for col in range(min(grid.Cols, 8)):  # First 8 cols
                                try:
                                    val = grid.GetDouble(row, col)
                                    row_data.append(f"{val:.6f}")
                                except:
                                    try:
                                        val = grid.GetString(row, col)
                                        row_data.append(str(val)[:12])
                                    except:
                                        row_data.append("?")
                            print(f"      Row {row}: {row_data}")
                except Exception as e:
                    print(f"    DataGrid[{i}] error: {e}")

            # Explore DataSeries
            print(f"\n    Number of DataSeries: {res.DataSeries}")
            for i in range(min(res.DataSeries, 5)):  # First 5 series
                try:
                    series = res.GetDataSeries(i)
                    print(f"\n    --- DataSeries[{i}] ---")
                    print(f"    Type: {type(series)}")

                    series_attrs = [a for a in dir(series) if not a.startswith("_")]
                    print(f"    Attributes: {series_attrs}")

                    if hasattr(series, 'Description'):
                        print(f"    Description: {series.Description}")
                    if hasattr(series, 'NumData'):
                        print(f"    NumData: {series.NumData}")
                        # Try to get values
                        values = []
                        for j in range(min(series.NumData, 10)):
                            try:
                                x = series.XData.Data(j)
                                y = series.YData.Data(j)
                                values.append(f"({x:.4f}, {y:.4f})")
                            except:
                                pass
                        if values:
                            print(f"    Values: {values}")
                except Exception as e:
                    print(f"    DataSeries[{i}] error: {e}")

            an.Close()
            print("\n    SeidelCoefficients analysis completed successfully!")

        except Exception as e:
            print(f"    ERROR: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("    SeidelCoefficients not found!")

    # Cleanup
    print("\n[6] Cleanup...")
    zos.disconnect()
    print("    Disconnected from OpticStudio.")

    print("\n" + "=" * 70)
    print("Diagnostic complete. Share this output to see the Seidel data structure.")
    print("=" * 70)


if __name__ == "__main__":
    main()
