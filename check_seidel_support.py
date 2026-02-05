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

    # Set up aperture - use float() for COM interop
    oss.SystemData.Aperture.ApertureType = zp.constants.SystemData.ZemaxApertureType.EntrancePupilDiameter
    oss.SystemData.Aperture.ApertureValue = float(10.0)  # EPD = 10mm

    # Set up wavelength
    wl = oss.SystemData.Wavelengths.GetWavelength(1)
    wl.Wavelength = float(0.55)  # 550nm

    # Set up fields - use smaller field angle to ensure chief ray traces
    oss.SystemData.Fields.SetFieldType(zp.constants.SystemData.FieldType.Angle)
    field1 = oss.SystemData.Fields.GetField(1)
    field1.Y = float(0.0)  # On-axis field

    # Add off-axis field
    oss.SystemData.Fields.AddField(float(0.0), float(2.0), float(1.0))  # 2 degree field

    # Get the lens data editor
    lde = oss.LDE

    # Set object to infinity
    obj_surf = lde.GetSurfaceAt(0)
    obj_surf.Thickness = float(1e10)  # Infinity

    # Insert surfaces for a simple singlet
    lde.InsertNewSurfaceAt(1)  # Front surface
    lde.InsertNewSurfaceAt(2)  # Back surface

    # Configure front surface (Surface 1)
    surf1 = lde.GetSurfaceAt(1)
    surf1.Radius = float(100.0)  # 100mm radius - gentler curve
    surf1.Thickness = float(6.0)  # 6mm thick
    surf1.Material = "N-BK7"
    surf1.SemiDiameter = float(15.0)

    # Configure back surface (Surface 2)
    surf2 = lde.GetSurfaceAt(2)
    surf2.Radius = float(-100.0)  # Symmetric biconvex
    surf2.Thickness = float(95.0)  # Distance to image (~focal length)
    surf2.SemiDiameter = float(15.0)

    # Image surface is Surface 3
    img_surf = lde.GetSurfaceAt(3)
    img_surf.SemiDiameter = float(20.0)

    print("    Singlet lens created: biconvex 100/-100mm, 6mm thick, EPD=10mm")
    print(f"    Number of surfaces: {lde.NumberOfSurfaces}")
    print(f"    Number of fields: {oss.SystemData.Fields.NumberOfFields}")


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
    try:
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

            # Check messages first
            print(f"\n    Messages: {an.messages}")

            # Check counts using proper attributes
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
                            # Get header row if present
                            if hasattr(grid, 'GetLabel'):
                                labels = []
                                for c in range(min(cols, 8)):
                                    try:
                                        labels.append(grid.GetLabel(c))
                                    except:
                                        labels.append(f"Col{c}")
                                print(f"    Labels: {labels}")

                            # Get data rows
                            for row in range(min(rows, 15)):
                                row_data = []
                                for col in range(min(cols, 8)):
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
                for i in range(min(num_series, 10)):
                    try:
                        series = res.GetDataSeries(i)
                        print(f"\n    --- DataSeries[{i}] ---")

                        # Print all series attributes
                        s_attrs = [a for a in dir(series) if not a.startswith("_") and not a.startswith("get_")]
                        print(f"    Attributes: {s_attrs}")

                        desc = getattr(series, 'Description', '')
                        label = getattr(series, 'SeriesLabel', '')
                        print(f"    Description: '{desc}', Label: '{label}'")

                        # Try to get data
                        if hasattr(series, 'XData') and hasattr(series, 'YData'):
                            xdata = series.XData
                            ydata = series.YData
                            if xdata and ydata:
                                num_pts = getattr(xdata, 'Length', 0) or safe_len(xdata)
                                print(f"    NumPoints: {num_pts}")
                                for j in range(min(num_pts, 20)):
                                    try:
                                        x = xdata.Data(j) if hasattr(xdata, 'Data') else xdata[j]
                                        y = ydata.Data(j) if hasattr(ydata, 'Data') else ydata[j]
                                        print(f"      [{j:2d}]: x={x:12.6g}, y={y:12.6g}")
                                    except Exception as e:
                                        print(f"      [{j:2d}]: error - {e}")
                    except Exception as e:
                        print(f"    DataSeries[{i}] error: {e}")

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
