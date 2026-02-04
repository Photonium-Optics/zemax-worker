# Zemax Worker Development Notes

**Read this before editing zemax-worker files.**

## Goal: Keep This Worker Thin

Only OpticStudio/ZosPy operations belong here. All other logic lives in `zemax-analysis-service/` (Mac side).

## Architecture: ZMX-Based Loading

All system loading uses .zmx files only:

```
LLM JSON --> Node.js /convert-to-zmx --> .zmx content (base64) --> Worker --> oss.load(file)
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
- Analysis execution (`get_cross_section`, `get_seidel`, `get_wavefront`, `get_spot_diagram`, `calc_semi_diameters`)
- Raw data extraction from LDE/SystemData

**What lives on Mac:**
- LLM JSON to ZMX conversion (via Node.js backend `/convert-to-zmx`)
- `seidel_converter.py` - Zernike to Seidel conversion (pure math)
- Response formatting and aggregation logic
- Business logic (margin calculations, clamping, etc.)
- Fallback strategies and retry logic for failed analysis calls
- Matplotlib rendering (cross-section, wavefront map, spot diagram from numpy arrays)

## Project Structure

```
zemax-worker/
|-- main.py               # FastAPI app, endpoints, request/response models
|-- zospy_handler.py      # ZosPy wrapper, all OpticStudio operations
|-- check_seidel_support.py  # Diagnostic script for Seidel support
|-- DEVELOPMENT.md        # This file
|-- requirements.txt      # Python dependencies
```

## Endpoints

| Endpoint | Purpose | Request Fields |
|----------|---------|----------------|
| `GET /health` | Health check | - |
| `POST /load-system` | Load ZMX into OpticStudio | `zmx_content` |
| `POST /cross-section` | Cross-section diagram + paraxial | `zmx_content` |
| `POST /calc-semi-diameters` | Surface aperture calculation | `zmx_content` |
| `POST /ray-trace-diagnostic` | Ray failure analysis | `zmx_content`, `num_rays`, `distribution` |
| `POST /seidel` | Zernike coefficients (raw) | `zmx_content` |
| `POST /trace-rays` | Ray positions at surfaces | `zmx_content`, `num_rays` |
| `POST /wavefront` | Wavefront error map + metrics | `zmx_content`, `field_index`, `wavelength_index`, `sampling` |
| `POST /spot-diagram` | Spot diagram + spot radii | `zmx_content`, `ray_density`, `reference` |

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
Run with `workers=1` - COM requires single-threaded apartment (STA):
```bash
uvicorn main:app --host 0.0.0.0 --port 8787 --workers 1
```

### 3. Wavelength Selection
```python
# WRONG: wl_data.SelectWavelength(index)
# RIGHT:
wl_data.GetWavelength(index).MakePrimary()
```

### 4. Dumb Executor Pattern
Analysis methods follow the "dumb executor" pattern:
- Try the primary ZosPy method once
- Return raw results on success, error on failure
- NO fallback strategies - Mac side handles retries
- NO aggregation or post-processing - Mac side handles that

### 5. CrossSection Image Export
Use `image_output_file` parameter AND `surface_line_thickness` to show lenses:
```python
from zospy.analyses.systemviewers.cross_section import CrossSection

result = CrossSection(
    number_of_rays=11,
    surface_line_thickness="Thick",  # REQUIRED to show lens elements!
    rays_line_thickness="Standard",
    field="All",
    wavelength="All",
    color_rays_by="Fields",
    delete_vignetted=True,
    image_size=(1200, 800),
).run(oss, image_output_file="/tmp/output.png")
```
Without `surface_line_thickness`, you only see rays - no lens elements.

**Numpy array fallback**: If PNG file export fails but `result.data` (numpy array) exists:
- Return `image_format="numpy_array"`
- Return `image`: base64-encoded raw array bytes
- Return `array_shape=[height, width, channels]`
- Return `array_dtype="uint8"` (or actual dtype)

Mac side reconstructs and renders using matplotlib.

### 6. SingleRayTrace Parameters
```python
ray_trace = zp.analyses.raysandspots.SingleRayTrace(
    hx=0.0,      # Normalized field X (0 when using field index)
    hy=0.0,      # Normalized field Y (0 when using field index)
    px=px,       # Normalized pupil X (-1 to 1) - ITERATE THESE
    py=py,       # Normalized pupil Y (-1 to 1) - ITERATE THESE
    wavelength=1,  # Wavelength index (1-indexed)
    field=fi,      # Field index (1-indexed)
)
result = ray_trace.run(oss)
```

### 7. Attribute Checks
Use `hasattr()` - properties vary by version:
```python
if hasattr(data, 'front_focal_length'):
    ffl = data.front_focal_length
```

## ZosPy Patterns That Work

### StandardSpot Analysis (Spot Diagram)
```python
# Use new_analysis for native OpticStudio access
analysis = zp.analyses.new_analysis(
    self.oss,
    zp.constants.Analysis.AnalysisIDM.StandardSpot,
    settings_first=True,
)

# Configure settings
settings = analysis.Settings
if hasattr(settings, 'RayDensity'):
    settings.RayDensity = ray_density

# Run and wait
analysis.ApplyAndWaitForCompletion()

# Try PNG export
if hasattr(analysis, 'ExportGraphicAs'):
    analysis.ExportGraphicAs("/tmp/spot.png")

# Get results
results = analysis.Results
if results and hasattr(results, 'AiryRadius'):
    airy_radius = float(results.AiryRadius)

# Close when done
analysis.Close()
```

### ZernikeStandardCoefficients Analysis
```python
zernike_analysis = zp.analyses.wavefront.ZernikeStandardCoefficients(
    sampling='64x64',
    maximum_term=37,
    wavelength=1,          # 1-indexed
    field=1,               # 1-indexed
    surface="Image",
)
result = zernike_analysis.run(oss)

# Access coefficients
if hasattr(result.data, 'coefficients'):
    coeffs = result.data.coefficients
    # May be dict keyed by term number or list
    if isinstance(coeffs, dict):
        for i in range(1, 38):
            coeff = coeffs.get(i) or coeffs.get(str(i))
            if hasattr(coeff, 'value'):
                value = float(coeff.value)
            else:
                value = float(coeff) if coeff else 0.0
```

### WavefrontMap Analysis
```python
wavefront_map = zp.analyses.wavefront.WavefrontMap(
    sampling="64x64",
    wavelength=wavelength_index,  # 1-indexed
    field=field_index,            # 1-indexed
    surface="Image",
    show_as="Surface",
    rotation="Rotate_0",
    scale=1,
    reference_to_primary=False,
    remove_tilt=False,
    use_exit_pupil=True,
).run(oss, oncomplete="Release")

# Access data (DataFrame or array)
if hasattr(wavefront_map.data, 'values'):
    arr = np.array(wavefront_map.data.values, dtype=np.float64)
else:
    arr = np.array(wavefront_map.data, dtype=np.float64)

# Return as numpy array for Mac-side rendering
image_b64 = base64.b64encode(arr.tobytes()).decode('utf-8')
```

### Manual Spot Data via Ray Trace
When StandardSpot doesn't provide data, compute manually:
```python
ray_x_positions = []
ray_y_positions = []

for px in np.linspace(-1, 1, grid_size):
    for py in np.linspace(-1, 1, grid_size):
        if px**2 + py**2 > 1:
            continue  # Skip rays outside circular pupil

        ray_trace = zp.analyses.raysandspots.SingleRayTrace(
            hx=0.0, hy=0.0,
            px=float(px), py=float(py),
            wavelength=1, field=fi,
        )
        result = ray_trace.run(oss)

        if hasattr(result.data, 'real_ray_trace_data'):
            df = result.data.real_ray_trace_data
            last_row = df.iloc[-1]  # Image surface
            error_code = last_row.get('error_code', 0)
            if error_code == 0:
                ray_x_positions.append(float(last_row['X']))
                ray_y_positions.append(float(last_row['Y']))

# Calculate metrics
x_arr = np.array(ray_x_positions)
y_arr = np.array(ray_y_positions)
centroid_x = np.mean(x_arr)
centroid_y = np.mean(y_arr)
distances = np.sqrt((x_arr - centroid_x)**2 + (y_arr - centroid_y)**2)
rms_radius = float(np.sqrt(np.mean(distances**2)))
geo_radius = float(np.max(distances))
```

### Getting Paraxial Data from LDE
```python
def _get_paraxial_from_lde(self) -> dict[str, Any]:
    paraxial = {}

    # Aperture info
    aperture = self.oss.SystemData.Aperture
    paraxial["epd"] = aperture.ApertureValue

    # Field info
    fields = self.oss.SystemData.Fields
    max_field = 0.0
    for i in range(1, fields.NumberOfFields + 1):
        f = fields.GetField(i)
        max_field = max(max_field, abs(f.Y), abs(f.X))
    paraxial["max_field"] = max_field

    # Total track from LDE
    lde = self.oss.LDE
    total_track = 0.0
    for i in range(1, lde.NumberOfSurfaces):
        surface = lde.GetSurfaceAt(i)
        total_track += abs(surface.Thickness)
    paraxial["total_track"] = total_track

    return paraxial
```

### Surface Geometry Extraction
```python
def _get_surface_geometry(self) -> list[dict]:
    surfaces = []
    lde = self.oss.LDE
    z_position = 0.0

    for i in range(1, lde.NumberOfSurfaces):
        surface = lde.GetSurfaceAt(i)
        radius = surface.Radius

        surf_data = {
            "index": i,
            "z": z_position,
            "radius": radius if radius != 0 and abs(radius) < 1e10 else None,
            "thickness": surface.Thickness,
            "semi_diameter": surface.SemiDiameter,
            "conic": surface.Conic,
            "material": str(surface.Material) if surface.Material else None,
            "is_stop": surface.IsStop,
        }
        surfaces.append(surf_data)
        z_position += surface.Thickness

    return surfaces
```

## Known Issues

### CrossSection "Object reference not set" Error
The error `Object reference not set to an instance of an object` in `set_StartSurface()` means:
- System has no fields defined (`NumberOfFields = 0`)
- OR system has invalid aperture/pupil configuration
- Often happens with ZMX files that have incomplete optical definitions

**Diagnostic clue:** `EFL=None` in load logs indicates invalid system state.

**Fix (2026-02-04):** Added pre-flight validation in `get_cross_section()` to check `NumberOfFields > 0` before attempting analysis. Returns surface geometry fallback if validation fails.

### Ray Trace Error Codes
OpticStudio returns numeric error codes for failed rays:

| Code | Meaning | Failure Mode String |
|------|---------|---------------------|
| 0 | No error | "OK" |
| 1 | Ray missed surface | "MISS" |
| 2 | Total Internal Reflection | "TIR" |
| 3 | Ray reversed | "REVERSED" |
| 4 | Ray vignetted | "VIGNETTE" |
| 5+ | Other errors | "ERROR_N" |

### Other Known Issues
- Seidel S4/S5 from Zernike are approximations only
- Ray trace header warnings are cosmetic
- `distribution="hexapolar"` uses square grid (not actual hexapolar)
- ZosPy version differences may require different attribute names

## Technical Debt

**Addressed:**
- Consolidated `get_paraxial_data` and `_get_paraxial_from_lde` - now delegates to single method
- Added debug logging to silent ray trace exceptions (was `except Exception: pass`)
- Simplified `get_seidel()` from ~160 lines to ~80 lines (no fallbacks)
- Simplified `ray_trace_diagnostic()` - returns raw data only, aggregation on Mac side
- Hotspot detection moved to zemax-analysis-service

**Remaining:**
- `distribution` parameter accepted but not used (always uses square grid)
- Surface index conventions inconsistent (some 0-indexed, some 1-indexed)

## Changelog

**2026-02-04 (Part 8) - Add Spot Diagram Endpoint**
- **NEW** `/spot-diagram` endpoint for spot diagram analysis
- Uses ZosPy `new_analysis` with `AnalysisIDM.StandardSpot` for native OpticStudio access
- Parameters: `ray_density` (1-20), `reference` ('chief_ray' or 'centroid')
- Returns PNG image (or numpy array fallback) + per-field spot data
- Per-field data includes: `rms_radius`, `geo_radius`, `centroid_x`, `centroid_y`, `num_rays`
- Also returns `airy_radius` for diffraction limit comparison
- **Fallback**: If StandardSpot doesn't provide data, manually traces rays via `SingleRayTrace`
- Follows "dumb executor" pattern - no retries, Mac side handles rendering if needed

**2026-02-04 (Part 7) - Add Wavefront Endpoint**
- **NEW** `/wavefront` endpoint for wavefront error analysis
- Uses ZosPy `ZernikeStandardCoefficients` for RMS, P-V, and Strehl ratio
- Uses ZosPy `WavefrontMap` for OPD map visualization
- Returns raw numpy array for wavefront map (Mac side renders to PNG)
- Response includes: `rms_waves`, `pv_waves`, `strehl_ratio`, `wavelength_um`, `field_x`, `field_y`
- Follows "dumb executor" pattern - no fallbacks or retries on worker side

**2026-02-04 (Part 6) - Move matplotlib fallback to Mac**
- **REMOVED** from worker:
  - matplotlib import and rendering in `get_cross_section()`
- **NEW** cross-section fallback behavior:
  - If PNG file export fails but `result.data` (numpy array) exists:
    - Return `image_format="numpy_array"`
    - Return `image` as base64-encoded raw array bytes
    - Return `array_shape=[height, width, channels]` for array dimensions
    - Return `array_dtype="uint8"` (or actual dtype) for element type
  - Mac side (`zemax-analysis-service`) reconstructs the numpy array and renders to PNG using matplotlib
- **BENEFIT**: matplotlib dependency removed from Windows worker, making it thinner

**2026-02-04 (Part 5) - Remove Legacy LLM JSON Loading**
- **REMOVED** `load_system()` method and all `_setup_*` methods
- **REMOVED** `_extract_value()` and `_to_float()` helper methods
- **REMOVED** `skip_load` parameter from all analysis methods
- All endpoints now require `zmx_content` (no `system` field)
- Simplified request models: `SystemRequest(zmx_content: str)`, `RayTraceDiagnosticRequest(zmx_content: str, ...)`

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

**2026-02-04 (Part 2) - ZMX-Based Loading**
- Added `load_zmx_file()` method for native .zmx file loading
- zemax-analysis-service uses Node.js `/convert-to-zmx` for conversion

**2026-02-04 (Part 1) - Initial Wave**
- Fixed `SelectWavelength` to `MakePrimary()`
- Added `float()` conversions for COM interop
- Standardized response field names (`success`, `total_failures`, `dominant_mode`)
- **ARCHITECTURE**: Moved Seidel conversion to Mac side
  - `/seidel` endpoint now returns raw Zernike coefficients
  - Removed `seidel_converter.py` from Windows worker
  - Mac does Zernike to Seidel conversion

## Adding a New Endpoint

### 1. Add Request/Response Models in `main.py`

```python
class MyAnalysisRequest(BaseModel):
    """Request for my analysis."""
    zmx_content: str = Field(description="Base64-encoded .zmx file content")
    my_param: int = Field(default=5, description="My parameter")

class MyAnalysisResponse(BaseModel):
    """Response from my analysis."""
    success: bool = Field(description="Whether the operation succeeded")
    my_result: Optional[float] = Field(default=None)
    error: Optional[str] = Field(default=None)
```

### 2. Add Handler Method in `zospy_handler.py`

```python
def get_my_analysis(self, my_param: int = 5) -> dict[str, Any]:
    """
    Description of what this analysis does.

    This is a "dumb executor" - returns raw data only.

    Note: System must be pre-loaded via load_zmx_file().

    Args:
        my_param: Description

    Returns:
        On success: {"success": True, "my_result": ...}
        On error: {"success": False, "error": "..."}
    """
    try:
        # Your ZosPy analysis code here
        result = some_analysis.run(self.oss)

        return {
            "success": True,
            "my_result": float(result.data.some_value),
        }
    except Exception as e:
        return {"success": False, "error": f"Analysis failed: {e}"}
```

### 3. Add Endpoint in `main.py`

```python
@app.post("/my-analysis", response_model=MyAnalysisResponse)
async def get_my_analysis(
    request: MyAnalysisRequest,
    _: None = Depends(verify_api_key),
) -> MyAnalysisResponse:
    """
    Endpoint description.

    This is a "dumb executor" endpoint - returns raw data only.
    """
    async with _zospy_lock:
        if _ensure_connected() is None:
            return MyAnalysisResponse(success=False, error=NOT_CONNECTED_ERROR)

        try:
            # Load system from request
            _load_system_from_request(request)

            # Run analysis (system already loaded)
            result = zospy_handler.get_my_analysis(
                my_param=request.my_param,
            )

            if not result.get("success", False):
                return MyAnalysisResponse(
                    success=False,
                    error=result.get("error", "Analysis failed"),
                )

            return MyAnalysisResponse(
                success=True,
                my_result=result.get("my_result"),
            )
        except Exception as e:
            _handle_zospy_error("My analysis", e)
            return MyAnalysisResponse(success=False, error=str(e))
```

### 4. Add Client Method in `zemax-analysis-service/zemax_client.py`

See zemax-analysis-service/DEVELOPMENT.md for client-side implementation.

## Debugging Tips

### Testing Worker Locally
```bash
# Start the worker
python main.py

# Test health
curl http://localhost:8787/health

# Test with zmx_content
curl -X POST http://localhost:8787/cross-section \
  -H "Content-Type: application/json" \
  -d '{"zmx_content": "..."}'

# Interactive docs
open http://localhost:8787/docs
```

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| COM error / STA violation | Running with multiple workers | Use `--workers 1` |
| OpticStudio not connected | License issue or OpticStudio not installed | Check OpticStudio installation |
| Attribute not found | ZosPy version mismatch | Check ZosPy version, use `hasattr()` |
| Analysis returns None | Invalid system state | Check NumberOfFields, aperture config |

### Logging
```python
import logging
logger = logging.getLogger(__name__)

# Info level for normal operations
logger.info(f"Loaded system: {result.get('num_surfaces')} surfaces")

# Debug level for detailed troubleshooting
logger.debug(f"Ray trace failed at ({px:.2f}, {py:.2f}): {e}")

# Warning for non-fatal issues
logger.warning(f"CrossSection export failed: {e}")
```
