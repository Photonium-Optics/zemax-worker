"""
Diagnostic: batch ray trace for spot diagram.
Run on Windows where OpticStudio is available.

Usage:
    python diagnose_batch_ray_trace.py [path_to_zmx]

If no ZMX path given, creates a simple singlet lens.
"""
import sys
import os
import time
import tempfile

# Lazy import zospy
import zospy as zp
from zospy_handler import _extract_value


def create_simple_singlet(oss):
    """Create a minimal singlet lens for testing."""
    oss.new()
    oss.SystemData.Aperture.ApertureValue = 10.0
    # Add a surface (singlet)
    lde = oss.LDE
    surf1 = lde.InsertNewSurfaceAt(1)
    surf1.Radius = 50.0
    surf1.Thickness = 5.0
    surf1.Material = "BK7"
    surf2 = lde.InsertNewSurfaceAt(2)
    surf2.Radius = -50.0
    surf2.Thickness = 45.0
    # Image plane is auto
    print(f"Created singlet: {lde.NumberOfSurfaces} surfaces")


def run_diagnostic(oss):
    fields = oss.SystemData.Fields
    wavelengths = oss.SystemData.Wavelengths
    num_fields = fields.NumberOfFields
    num_wavelengths = wavelengths.NumberOfWavelengths

    print(f"\n=== System Info ===")
    print(f"Surfaces: {oss.LDE.NumberOfSurfaces}")
    print(f"Fields: {num_fields}")
    print(f"Wavelengths: {num_wavelengths}")

    for fi in range(1, num_fields + 1):
        f = fields.GetField(fi)
        print(f"  Field {fi}: X={_extract_value(f.X)}, Y={_extract_value(f.Y)}")

    for wi in range(1, num_wavelengths + 1):
        w = wavelengths.GetWavelength(wi)
        print(f"  Wavelength {wi}: {_extract_value(w.Value)} um")

    # Max field for normalization
    max_field_x = 0.0
    max_field_y = 0.0
    for fi in range(1, num_fields + 1):
        max_field_x = max(max_field_x, abs(_extract_value(fields.GetField(fi).X)))
        max_field_y = max(max_field_y, abs(_extract_value(fields.GetField(fi).Y)))
    print(f"  max_field_x={max_field_x}, max_field_y={max_field_y}")

    # Small pupil grid for testing
    import numpy as np
    ray_density = 3
    pupil_coords = []
    for px in np.linspace(-1, 1, ray_density):
        for py in np.linspace(-1, 1, ray_density):
            if px**2 + py**2 <= 1.0:
                pupil_coords.append((float(px), float(py)))
    print(f"\nPupil coords ({len(pupil_coords)} rays per field/wavelength):")
    for pc in pupil_coords:
        print(f"  px={pc[0]:.3f}, py={pc[1]:.3f}")

    # Open batch ray trace
    print(f"\n=== Batch Ray Trace ===")
    ray_trace = oss.Tools.OpenBatchRayTrace()
    print(f"ray_trace type: {type(ray_trace)}")
    print(f"ray_trace is None: {ray_trace is None}")

    max_rays = num_fields * num_wavelengths * len(pupil_coords)
    print(f"max_rays: {max_rays}")

    image_surface = oss.LDE.NumberOfSurfaces - 1
    print(f"image_surface index: {image_surface}")

    norm_unpol = ray_trace.CreateNormUnpol(
        max_rays,
        zp.constants.Tools.RayTrace.RaysType.Real,
        image_surface,
    )
    print(f"norm_unpol type: {type(norm_unpol)}")
    print(f"norm_unpol is None: {norm_unpol is None}")

    opd_none = zp.constants.Tools.RayTrace.OPDMode.None_
    print(f"OPDMode.None_: {opd_none} (type={type(opd_none)})")

    # Add rays
    rays_added = 0
    for fi in range(1, num_fields + 1):
        field = fields.GetField(fi)
        fx = _extract_value(field.X)
        fy = _extract_value(field.Y)
        hx = float(fx / max_field_x) if max_field_x > 1e-10 else 0.0
        hy = float(fy / max_field_y) if max_field_y > 1e-10 else 0.0
        for wi in range(1, num_wavelengths + 1):
            for px, py in pupil_coords:
                norm_unpol.AddRay(wi, hx, hy, float(px), float(py), opd_none)
                rays_added += 1
    print(f"Rays added: {rays_added}")

    # Run
    t0 = time.perf_counter()
    ray_trace.RunAndWaitForCompletion()
    elapsed = (time.perf_counter() - t0) * 1000
    print(f"RunAndWaitForCompletion: {elapsed:.1f}ms")

    # Read results one at a time, dump everything
    print(f"\n=== Reading Results ===")
    total_ok = 0
    total_err = 0
    for i in range(rays_added):
        result = norm_unpol.ReadNextResult()
        result_len = len(result) if hasattr(result, '__len__') else 'N/A'
        if i == 0:
            print(f"ReadNextResult return type: {type(result)}")
            print(f"ReadNextResult length: {result_len}")
            print(f"Full first result: {result}")
            print(f"Individual values:")
            for idx, val in enumerate(result):
                print(f"  [{idx}] = {val!r} (type={type(val).__name__})")

        success = result[0]
        err_code = result[2]
        if success and err_code == 0:
            total_ok += 1
            if i < 5:
                print(f"  Ray {i}: OK  x={float(result[4]):.6f}, y={float(result[5]):.6f}")
        else:
            total_err += 1
            if total_err <= 5:
                print(f"  Ray {i}: FAIL success={success}, err_code={err_code}")

    print(f"\n=== Summary ===")
    print(f"Total rays: {rays_added}")
    print(f"OK: {total_ok}")
    print(f"Errors: {total_err}")

    ray_trace.Close()
    print("Done.")


def main():
    zmx_path = sys.argv[1] if len(sys.argv) > 1 else None

    zos = zp.ZOS()
    oss = zos.connect(mode="extension")
    print(f"Connected to OpticStudio")

    if zmx_path:
        oss.load(zmx_path)
        print(f"Loaded: {zmx_path}")
    else:
        create_simple_singlet(oss)

    run_diagnostic(oss)


if __name__ == "__main__":
    main()
