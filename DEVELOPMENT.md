# Zemax Worker Development Notes

**Read this before editing zemax-worker files.**

## Goal: Keep This Worker Thin

Only OpticStudio/ZosPy operations belong here. All other logic lives in `zemax-analysis-service/` (Mac side).

## Architecture: ZMX-Based Loading

All system loading uses .zmx files only:

```
LLM JSON → Node.js /convert-to-zmx → .zmx content (base64) → Worker → oss.load(file)
```

The worker uses the existing `zemax-converter` (TypeScript) to produce .zmx files,
which are then loaded natively by OpticStudio. This approach:
- Uses native file format (more reliable)
- Single source of truth for conversion (zemax-converter)
- Minimal code in the worker

All endpoints require `zmx_content` (base64-encoded .zmx file) in the request body.

**What stays on Windows:**
- System loading from .zmx files (`load_zmx_file`)
- Ray tracing (`ray_trace_diagnostic`, `trace_rays`)
- Analysis execution (`get_cross_section`, `get_seidel`, `calc_semi_diameters`)
- Raw data extraction from LDE/SystemData

**What lives on Mac:**
- LLM JSON → ZMX conversion (via Node.js backend `/convert-to-zmx`)
- `seidel_converter.py` - Zernike→Seidel conversion (pure math)
- Response formatting and aggregation logic
- Business logic (margin calculations, clamping, etc.)
- Fallback strategies and retry logic for failed analysis calls
- Matplotlib rendering (cross-section numpy array fallback)

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

### 4. Dumb Executor Pattern
Analysis methods (`get_seidel`, `ray_trace_diagnostic`) follow the "dumb executor" pattern:
- Try the primary ZosPy method once
- Return raw results on success, error on failure
- NO fallback strategies - Mac side handles retries
- NO aggregation or post-processing - Mac side handles that

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
- ~~`get_seidel()` is 150+ lines~~ - FIXED: simplified to ~80 lines, no fallbacks
- ~~`ray_trace_diagnostic()` is 130+ lines~~ - FIXED: now simplified to raw data return
- ~~Hotspot detection (>10% failures) could move to Mac side~~ - FIXED: moved to zemax-analysis-service
- `distribution` parameter accepted but not used (always uses square grid)
- Surface index conventions inconsistent (some 0-indexed, some 1-indexed)

## Changelog

**2026-02-04 (Part 4) - Dumb Executor Refactor for get_seidel**
- Simplified `get_seidel()` from ~160 lines to ~80 lines
- **REMOVED** from worker:
  - Raw ZOSAPI fallback (GetDataGrid, GetDataSeries, GetTextFile exploration)
  - `_parse_zernike_text()` helper method
  - Placeholder coefficient generation on failure
  - Complex debug logging for fallback strategies
- **NEW** behavior:
  - Tries ZosPy `ZernikeStandardCoefficients` once
  - On success: returns `{"success": True, "zernike_coefficients": [...], "wavelength_um": float, "num_surfaces": int}`
  - On failure: returns `{"success": False, "error": "ZosPy analysis failed: <message>"}`
  - Mac side handles fallbacks and retries

**2026-02-04 (Part 3) - Dumb Executor Refactor for ray_trace_diagnostic**
- Refactored `ray_trace_diagnostic()` to be a "dumb executor"
- **REMOVED** from worker:
  - Field-level aggregation (rays_traced, rays_reached, rays_failed)
  - `aggregate_surface_failures` dictionary building
  - Hotspot detection (>10% threshold calculation)
  - `field_results` nested structure
- **NEW** raw response format:
  ```python
  {
      "paraxial": {"efl": ..., "bfl": ..., "fno": ..., "total_track": ...},
      "num_surfaces": int,
      "num_fields": int,
      "raw_rays": [
          {"field_index": 0, "field_x": 0, "field_y": 0, "px": 0.5, "py": 0.5,
           "reached_image": True, "failed_surface": None, "failure_mode": None},
          ...
      ],
      "surface_semi_diameters": [sd1, sd2, ...]
  }
  ```
- Added `_error_code_to_mode()` helper for mapping OpticStudio error codes to strings
- Added `RawRay` Pydantic model for type safety
- All aggregation logic now belongs in zemax-analysis-service (Mac side)

**2026-02-04 (Part 5) - Remove Legacy LLM JSON Loading**
- **REMOVED** `load_system()` method and all `_setup_*` methods
- **REMOVED** `_extract_value()` and `_to_float()` helper methods
- **REMOVED** `skip_load` parameter from all analysis methods
- All endpoints now require `zmx_content` (no `system` field)
- Simplified request models: `SystemRequest(zmx_content: str)`, `RayTraceDiagnosticRequest(zmx_content: str, ...)`

**2026-02-04 (Part 2) - ZMX-Based Loading**
- Added `load_zmx_file()` method for native .zmx file loading
- zemax-analysis-service uses Node.js `/convert-to-zmx` for conversion

**2026-02-04 (Part 1)**
- Fixed `SelectWavelength` → `MakePrimary()`
- Added `float()` conversions for COM interop
- Standardized response field names (`success`, `total_failures`, `dominant_mode`)
- **ARCHITECTURE**: Moved Seidel conversion to Mac side
  - `/seidel` endpoint now returns raw Zernike coefficients
  - Removed `seidel_converter.py` from Windows worker
  - Mac does Zernike→Seidel conversion
