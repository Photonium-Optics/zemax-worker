#!/usr/bin/env python3
"""
Diagnostic script to check if SeidelCoefficients analysis is available in ZOSPy.

Run this on the Windows machine with OpticStudio installed:
    python check_seidel_support.py
"""

import zospy as zp
import os


def find_sample_file(oss):
    """Find a sample .zmx file from OpticStudio installation."""
    # Common sample file locations
    possible_paths = [
        r"C:\Users\Public\Documents\Zemax\Samples\Sequential\Objectives\Double Gauss 28 degree field.zmx",
        r"C:\Users\Public\Documents\Zemax\Samples\Sequential\Objectives\Cooke 40 degree field.zmx",
        r"C:\Users\Public\Documents\Zemax\Samples\Sequential\Objectives\Petzval.zmx",
        # User-specific paths
        os.path.expanduser(r"~\Documents\Zemax\Samples\Sequential\Objectives\Double Gauss 28 degree field.zmx"),
        os.path.expanduser(r"~\Documents\Zemax\Samples\Sequential\Objectives\Cooke 40 degree field.zmx"),
    ]

    for path in possible_paths:
        if os.path.exists(path):
            return path

    # Try to find any .zmx file in Zemax samples
    search_dirs = [
        r"C:\Users\Public\Documents\Zemax\Samples",
        os.path.expanduser(r"~\Documents\Zemax\Samples"),
    ]

    for search_dir in search_dirs:
        if os.path.exists(search_dir):
            for root, dirs, files in os.walk(search_dir):
                for f in files:
                    if f.endswith(".zmx") and "Objective" in root:
                        return os.path.join(root, f)

    return None


def safe_len(obj):
    """Safely get length of .NET array or Python object."""
    if obj is None:
        return 0
    try:
        return len(obj)
    except TypeError:
        pass
    try:
        return obj.Length
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

    # Try to load a sample file
    print("\n[4] Loading sample lens file...")
    sample_file = find_sample_file(oss)

    if sample_file:
        print(f"    Found: {sample_file}")
        try:
            oss.load(sample_file)
            print(f"    Loaded successfully!")
            print(f"    Surfaces: {oss.LDE.NumberOfSurfaces}")
            print(f"    Fields: {oss.SystemData.Fields.NumberOfFields}")
        except Exception as e:
            print(f"    Load error: {e}")
            sample_file = None

    if not sample_file:
        print("    No sample file found. Creating simple lens...")
        oss.new()

        # Minimal working lens
        oss.SystemData.Aperture.ApertureValue = float(10.0)
        oss.SystemData.Wavelengths.GetWavelength(1).Wavelength = float(0.55)

        # Keep only on-axis field for now
        field1 = oss.SystemData.Fields.GetField(1)
        field1.Y = float(0.0)

        lde = oss.LDE

        # Object at infinity
        lde.GetSurfaceAt(0).Thickness = float(1e10)

        # Insert lens
        lde.InsertNewSurfaceAt(1)
        lde.InsertNewSurfaceAt(2)

        surf1 = lde.GetSurfaceAt(1)
        surf1.Radius = float(100.0)
        surf1.Thickness = float(5.0)
        surf1.Material = "N-BK7"

        surf2 = lde.GetSurfaceAt(2)
        surf2.Radius = float(-100.0)
        surf2.Thickness = float(100.0)

        print(f"    Created simple lens. Surfaces: {lde.NumberOfSurfaces}")

    print("\n[5] Checking AnalysisIDM...")
    idm = zp.constants.Analysis.AnalysisIDM
    names = [n for n in dir(idm) if not n.startswith("_")]
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

            # Check messages
            print(f"\n    Messages: {an.messages}")

            num_grids = res.NumberOfDataGrids if hasattr(res, 'NumberOfDataGrids') else 0
            num_series = res.NumberOfDataSeries if hasattr(res, 'NumberOfDataSeries') else 0

            print(f"    NumberOfDataGrids: {num_grids}")
            print(f"    NumberOfDataSeries: {num_series}")

            # Extract DataGrids
            if num_grids > 0:
                for i in range(num_grids):
                    try:
                        grid = res.GetDataGrid(i)
                        print(f"\n    --- DataGrid[{i}] ---")
                        rows = getattr(grid, 'Rows', 0) or 0
                        cols = getattr(grid, 'Cols', 0) or 0
                        print(f"    Size: {rows}x{cols}")

                        if rows > 0 and cols > 0:
                            for row in range(min(rows, 20)):
                                row_data = []
                                for col in range(min(cols, 10)):
                                    try:
                                        val = grid.GetDouble(row, col)
                                        row_data.append(f"{val:12.6g}")
                                    except:
                                        try:
                                            val = grid.GetString(row, col)
                                            row_data.append(f"{str(val):>12}")
                                        except:
                                            row_data.append("      -     ")
                                print(f"    [{row:2d}]: {' '.join(row_data)}")
                    except Exception as e:
                        print(f"    DataGrid[{i}] error: {e}")

            # Extract DataSeries
            if num_series > 0:
                for i in range(min(num_series, 15)):
                    try:
                        series = res.GetDataSeries(i)
                        print(f"\n    --- DataSeries[{i}] ---")

                        desc = getattr(series, 'Description', '')
                        xlabel = getattr(series, 'XLabel', '')
                        print(f"    Description: '{desc}', XLabel: '{xlabel}'")

                        # Get series labels if available
                        if hasattr(series, 'SeriesLabels'):
                            labels = series.SeriesLabels
                            if labels:
                                print(f"    SeriesLabels type: {type(labels)}")
                                try:
                                    label_list = list(labels) if hasattr(labels, '__iter__') else [str(labels)]
                                    print(f"    SeriesLabels: {label_list[:10]}")
                                except:
                                    print(f"    SeriesLabels: {labels}")

                        if hasattr(series, 'NumSeries'):
                            print(f"    NumSeries: {series.NumSeries}")

                        # Get XData and YData
                        if hasattr(series, 'XData') and hasattr(series, 'YData'):
                            xdata = series.XData
                            ydata = series.YData

                            print(f"    XData type: {type(xdata)}")
                            print(f"    YData type: {type(ydata)}")

                            if xdata is not None and ydata is not None:
                                # Try different ways to get length
                                num_pts = 0
                                if hasattr(xdata, 'Length'):
                                    num_pts = xdata.Length
                                elif hasattr(xdata, 'Count'):
                                    num_pts = xdata.Count
                                else:
                                    num_pts = safe_len(xdata)

                                print(f"    NumPoints: {num_pts}")

                                # Try to access data
                                for j in range(min(num_pts, 25)):
                                    try:
                                        # Try Data(j) method
                                        if hasattr(xdata, 'Data'):
                                            x = xdata.Data(j)
                                            y = ydata.Data(j)
                                        else:
                                            x = xdata[j]
                                            y = ydata[j]
                                        print(f"      [{j:2d}]: x={x:12.6g}, y={y:12.6g}")
                                    except Exception as e:
                                        print(f"      [{j:2d}]: error - {e}")
                                        break
                    except Exception as e:
                        print(f"    DataSeries[{i}] error: {e}")
                        import traceback
                        traceback.print_exc()

            an.Close()
            print("\n    SeidelCoefficients completed!")

        except Exception as e:
            print(f"    ERROR: {e}")
            import traceback
            traceback.print_exc()

    # Cleanup
    print("\n[7] Cleanup...")
    zos.disconnect()
    print("    Disconnected.")

    print("\n" + "=" * 70)
    print("Done. Share output to see the Seidel data structure.")
    print("=" * 70)


if __name__ == "__main__":
    main()
