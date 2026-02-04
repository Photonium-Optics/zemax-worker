# Zemax Worker Development Notes

**Read this before editing zemax-worker files.**

## Goal: Keep This Worker Thin

Only OpticStudio/ZosPy operations belong here. All other logic lives in `zemax-analysis-service/` (Mac side).

**What stays on Windows:**
- System loading (`load_system`, `_setup_*` methods)
- Ray tracing (`ray_trace_diagnostic`, `trace_rays`)
- Analysis execution (`get_cross_section`, `get_seidel`, `calc_semi_diameters`)
- Raw data extraction from LDE/SystemData
- `_parse_zernike_text()` - Parses OpticStudio text output to extract raw coefficients

**What lives on Mac:**
- `seidel_converter.py` - Zernike→Seidel conversion (pure math)
- Response formatting and aggregation logic
- Business logic (margin calculations, clamping, etc.)

## ZosPy Docs

- GitHub: https://github.com/MREYE-LUMC/ZOSPy
- Docs: https://zospy.readthedocs.io/
- Context7 library ID: `/mreye-lumc/zospy`

## Critical Rules

### 1. Type Conversion
All numeric values to ZosPy MUST be `float()`:
```python
surface.Radius = float(radius)  # NOT just radius
```

### 2. Single Worker
Run with `workers=1` - COM requires single-threaded apartment.

### 3. Wavelength Selection
```python
# WRONG: wl_data.SelectWavelength(index)
# RIGHT:
wl_data.GetWavelength(index).MakePrimary()
```

### 4. OpticStudio v25 Text Parser Bug
ZosPy text parsers fail. Use raw ZOSAPI:
```python
analysis = zp.analyses.new_analysis(
    self.oss,
    zp.constants.Analysis.AnalysisIDM.ZernikeStandardCoefficients,
    settings_first=True
)
```

### 5. CrossSection Image Export
Use `image_output_file` parameter AND `surface_line_thickness` to show lenses:
```python
result = CrossSection(
    number_of_rays=11,
    surface_line_thickness="Thick",  # REQUIRED to show lens elements!
    rays_line_thickness="Standard",
    ...
).run(oss, image_output_file="/tmp/output.png")
```
Without `surface_line_thickness`, you only see rays - no lens elements.
Always include surface geometry as fallback for client-side SVG rendering.

### 6. SingleRayTrace Parameters
- `px, py`: Pupil coordinates (iterate these)
- `hx, hy`: Field coordinates (set to 0 when using `field=fi`)

### 7. Attribute Checks
Use `hasattr()` - properties vary by version:
```python
if hasattr(data, 'front_focal_length'):
    ffl = data.front_focal_length
```

## Known Issues

- CrossSection export often fails - fallback to surface geometry works
- Seidel S4/S5 from Zernike are approximations only
- Ray trace header warnings are cosmetic
- `distribution="hexapolar"` in ray_trace_diagnostic uses square grid (not actual hexapolar)

## Code Review Notes (2026-02-04)

**Addressed:**
- Consolidated `get_paraxial_data` and `_get_paraxial_from_lde` - now delegates to single method
- Added debug logging to silent ray trace exceptions (was `except Exception: pass`)

**Technical Debt:**
- `get_seidel()` is 150+ lines - could extract `_extract_zernike_from_zospy_wrapper()` and `_extract_zernike_from_raw_api()`
- `ray_trace_diagnostic()` is 130+ lines - could extract `_trace_ray_grid()` and `_aggregate_surface_failures()`
- Hotspot detection (>10% failures) could move to Mac side
- `distribution` parameter accepted but not used (always uses square grid)
- Surface index conventions inconsistent (some 0-indexed, some 1-indexed)

## Changelog

**2026-02-04**
- Fixed `SelectWavelength` → `MakePrimary()`
- Added `float()` conversions for COM interop
- Raw ZOSAPI fallback for Zernike analysis
- Added `_parse_zernike_text()` for text-based coefficient extraction
- Standardized response field names (`success`, `total_failures`, `dominant_mode`)
- **ARCHITECTURE**: Moved Seidel conversion to Mac side
  - `/seidel` endpoint now returns raw Zernike coefficients
  - Removed `seidel_converter.py` from Windows worker
  - Mac does Zernike→Seidel conversion
