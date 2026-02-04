# Zemax Worker Development Notes

**Read this before editing zemax-worker files.**

## Goal: Keep This Worker Thin

This worker follows the **"dumb executor" pattern**:
- Only OpticStudio/ZosPy operations belong here
- No business logic, aggregation, or fallback strategies
- Mac side (zemax-analysis-service) handles everything else

## Quick Start

**Prerequisites:**
- Windows 10/11
- Zemax OpticStudio (Professional or Premium license for API access)
- Python 3.9-3.11
- ZosPy >= 1.2.0

**Start:**
```bash
cd zemax-worker
pip install -r requirements.txt
python main.py
```

**Interactive API Docs:** http://localhost:8787/docs

**CRITICAL:** Always run with single worker for COM/STA compatibility:
```bash
uvicorn main:app --host 0.0.0.0 --port 8787 --workers 1
```

## Architecture: ZMX-Based Loading

```
LLM JSON --> Node.js /convert-to-zmx --> .zmx content (base64) --> Worker --> oss.load(file)
```

All endpoints require `zmx_content` (base64-encoded .zmx file) in the request body.

**What stays on Windows (this worker):**
- System loading from .zmx files (`load_zmx_file`)
- Analysis execution (cross-section, seidel, wavefront, spot diagram)
- Ray tracing (ray trace diagnostic, trace rays)
- Raw data extraction from LDE/SystemData

**What lives on Mac (zemax-analysis-service):**
- LLM JSON to ZMX conversion (via Node.js backend)
- Zernike to Seidel conversion (`seidel_converter.py`)
- Response aggregation and hotspot detection
- Business logic (margins, clamping, thresholds)
- Matplotlib rendering (for numpy array fallback)

## Project Structure

```
zemax-worker/
|-- main.py               # FastAPI app, endpoints, request/response models
|-- zospy_handler.py      # ZosPy wrapper, all OpticStudio operations
|-- check_seidel_support.py  # Diagnostic script for Seidel support
|-- DEVELOPMENT.md        # This file
|-- requirements.txt      # Python dependencies
```

---

## Endpoints Reference

| Endpoint | Purpose | Request Fields |
|----------|---------|----------------|
| `GET /health` | Health check + OpticStudio status | - |
| `POST /load-system` | Load ZMX into OpticStudio | `zmx_content` |
| `POST /cross-section` | Cross-section diagram + paraxial | `zmx_content` |
| `POST /calc-semi-diameters` | Surface aperture calculation | `zmx_content` |
| `POST /ray-trace-diagnostic` | Ray failure analysis (raw data) | `zmx_content`, `num_rays`, `distribution` |
| `POST /seidel` | Zernike coefficients (raw) | `zmx_content` |
| `POST /trace-rays` | Ray positions at surfaces | `zmx_content`, `num_rays` |
| `POST /wavefront` | Wavefront error map + metrics | `zmx_content`, `field_index`, `wavelength_index`, `sampling` |
| `POST /spot-diagram` | Spot diagram + spot radii | `zmx_content`, `ray_density`, `reference` |

### Response Format

All endpoints return:
```json
{
  "success": true,
  ...analysis data...
}
```

Or on error:
```json
{
  "success": false,
  "error": "Error message describing what went wrong"
}
```

---

## The "Dumb Executor" Pattern

Every analysis method follows this pattern:

1. **Try the ZosPy operation once**
2. **Return raw results on success**
3. **Return error on failure** - no fallbacks, no retries
4. **No post-processing** - Mac side handles aggregation

**Example from `get_seidel()`:**
```python
def get_seidel(self) -> dict[str, Any]:
    """
    This is a "dumb executor" - tries once, returns result or error.
    Mac side handles retries and Zernike-to-Seidel conversion.
    """
    try:
        zernike_analysis = zp.analyses.wavefront.ZernikeStandardCoefficients(
            sampling='64x64',
            maximum_term=37,
            wavelength=1,
            field=1,
            surface="Image",
        )
        result = zernike_analysis.run(self.oss)

        if not hasattr(result, 'data') or result.data is None:
            return {"success": False, "error": "ZosPy analysis returned no data"}

        # Extract coefficients...
        return {
            "success": True,
            "zernike_coefficients": coefficients,
            "wavelength_um": wl_um,
            "num_surfaces": num_surfaces,
        }

    except Exception as e:
        return {"success": False, "error": f"ZosPy analysis failed: {e}"}
```

**Why this pattern?**
- Keeps Windows code simple and maintainable
- All complex logic (retries, fallbacks, error handling) in one place (Mac)
- Easier to debug - clear separation of concerns
- Faster worker startup and operation

---

## Critical Rules for ZosPy

### 1. Type Conversion for COM Interop

**All numeric values to ZosPy MUST be `float()`:**
```python
# WRONG - will cause COM errors
surface.Radius = radius

# RIGHT
surface.Radius = float(radius)
```

### 2. Single Worker Required

ZosPy/COM requires single-threaded apartment (STA):
```bash
uvicorn main:app --host 0.0.0.0 --port 8787 --workers 1
```

Multiple workers will cause race conditions and unpredictable failures.

### 3. Wavelength Selection

```python
# WRONG - method doesn't exist
wl_data.SelectWavelength(index)

# RIGHT
wl_data.GetWavelength(index).MakePrimary()
```

### 4. Attribute Checks

Properties vary by ZosPy/OpticStudio version. Always use `hasattr()`:
```python
if hasattr(data, 'front_focal_length'):
    ffl = data.front_focal_length
```

### 5. CrossSection Image Export

Use `surface_line_thickness` to show lens elements:
```python
from zospy.analyses.systemviewers.cross_section import CrossSection

cross_section = CrossSection(
    number_of_rays=11,
    surface_line_thickness="Thick",  # REQUIRED to show lenses!
    rays_line_thickness="Standard",
    field="All",
    wavelength="All",
    color_rays_by="Fields",
    delete_vignetted=True,
    image_size=(1200, 800),
)
result = cross_section.run(oss, image_output_file="/tmp/output.png")
```

Without `surface_line_thickness`, you only see rays - no lens elements.

### 6. Numpy Array Fallback

If PNG export fails but `result.data` (numpy array) exists:
```python
if os.path.exists(temp_path):
    # PNG export succeeded
    with open(temp_path, 'rb') as f:
        image_b64 = base64.b64encode(f.read()).decode('utf-8')
    image_format = "png"
elif result.data is not None:
    # Fallback: return numpy array
    arr = np.array(result.data)
    if arr.ndim >= 2:
        image_b64 = base64.b64encode(arr.tobytes()).decode('utf-8')
        image_format = "numpy_array"
        array_shape = list(arr.shape)
        array_dtype = str(arr.dtype)
```

Mac side reconstructs and renders using matplotlib.

---

## ZosPy Patterns That Work

### Loading a System from ZMX

```python
def load_zmx_file(self, file_path: str) -> dict[str, Any]:
    """Load .zmx file directly into OpticStudio."""
    if not os.path.exists(file_path):
        raise ZosPyError(f"ZMX file not found: {file_path}")

    self.oss.load(file_path)
    num_surfaces = self.oss.LDE.NumberOfSurfaces - 1  # Exclude object

    return {
        "num_surfaces": num_surfaces,
        "efl": self._get_efl(),
    }
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

### SingleRayTrace for Ray Positions

```python
ray_trace = zp.analyses.raysandspots.SingleRayTrace(
    hx=0.0,      # Normalized field X (0 when using field index)
    hy=0.0,      # Normalized field Y (0 when using field index)
    px=float(px), # Normalized pupil X (-1 to 1)
    py=float(py), # Normalized pupil Y (-1 to 1)
    wavelength=1, # Wavelength index (1-indexed)
    field=fi,     # Field index (1-indexed)
)
result = ray_trace.run(self.oss)

# Access ray data
if hasattr(result.data, 'real_ray_trace_data'):
    df = result.data.real_ray_trace_data
    last_row = df.iloc[-1]  # Image surface
    x_val = last_row.get('X', last_row.get('x', None))
    y_val = last_row.get('Y', last_row.get('y', None))
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
result = zernike_analysis.run(self.oss)

# Access coefficients
if hasattr(result.data, 'coefficients'):
    coeffs = result.data.coefficients
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
).run(self.oss, oncomplete="Release")

# Access data
if hasattr(wavefront_map.data, 'values'):
    arr = np.array(wavefront_map.data.values, dtype=np.float64)
else:
    arr = np.array(wavefront_map.data, dtype=np.float64)

# Return as numpy array (Mac renders to PNG)
image_b64 = base64.b64encode(arr.tobytes()).decode('utf-8')
```

### StandardSpot Analysis via new_analysis

```python
analysis = zp.analyses.new_analysis(
    self.oss,
    zp.constants.Analysis.AnalysisIDM.StandardSpot,
    settings_first=True,
)

# Configure
settings = analysis.Settings
if hasattr(settings, 'RayDensity'):
    settings.RayDensity = ray_density

# Run
analysis.ApplyAndWaitForCompletion()

# Try PNG export
if hasattr(analysis, 'ExportGraphicAs'):
    analysis.ExportGraphicAs("/tmp/spot.png")

# Get Airy radius
results = analysis.Results
if hasattr(results, 'AiryRadius'):
    airy_radius = float(results.AiryRadius)

# Close when done
analysis.Close()
```

### Surface Geometry Extraction

```python
def _get_surface_geometry(self) -> list[dict[str, Any]]:
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

---

## How to Add a New Endpoint

### Step 1: Add Request/Response Models in `main.py`

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

### Step 2: Add Handler Method in `zospy_handler.py`

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

        if not hasattr(result, 'data') or result.data is None:
            return {"success": False, "error": "Analysis returned no data"}

        return {
            "success": True,
            "my_result": float(result.data.some_value),
        }
    except Exception as e:
        return {"success": False, "error": f"Analysis failed: {e}"}
```

### Step 3: Add Endpoint in `main.py`

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

### Step 4: Add Client Method in zemax-analysis-service

See `zemax-analysis-service/DEVELOPMENT.md` for client-side implementation.

---

## Known Issues

### CrossSection "Object reference not set" Error

The error `Object reference not set to an instance of an object` in `set_StartSurface()` means:
- System has no fields defined (`NumberOfFields = 0`)
- OR system has invalid aperture/pupil configuration

**Diagnostic clue:** `EFL=None` in load logs indicates invalid system state.

**Fix (2026-02-04):** Added pre-flight validation in `get_cross_section()` to check `NumberOfFields > 0` before attempting analysis.

### Ray Trace Error Codes

| Code | Meaning | Failure Mode String |
|------|---------|---------------------|
| 0 | No error | "OK" |
| 1 | Ray missed surface | "MISS" |
| 2 | Total Internal Reflection | "TIR" |
| 3 | Ray reversed | "REVERSED" |
| 4 | Ray vignetted | "VIGNETTE" |
| 5+ | Other errors | "ERROR_N" |

### Other Known Issues

| Issue | Notes |
|-------|-------|
| `distribution="hexapolar"` | Uses square grid (not actual hexapolar) |
| Surface index conventions | Some 0-indexed, some 1-indexed |
| ZosPy version differences | Attribute names may vary |
| Seidel S4/S5 from Zernike | Approximations only |

---

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

### Common Issues and Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| COM error / STA violation | Multiple workers | Use `--workers 1` |
| OpticStudio not connected | License issue or not installed | Check OpticStudio installation |
| Attribute not found | ZosPy version mismatch | Use `hasattr()`, check version |
| Analysis returns None | Invalid system state | Check NumberOfFields, aperture |
| "Object reference not set" | System has no fields | Validate system before analysis |

### Logging Best Practices

```python
import logging
logger = logging.getLogger(__name__)

# Info level for normal operations
logger.info(f"Loaded system: {result.get('num_surfaces')} surfaces")

# Debug level for detailed troubleshooting
logger.debug(f"Ray trace at ({px:.2f}, {py:.2f}): {result}")

# Warning for non-fatal issues
logger.warning(f"CrossSection export failed: {e}")

# Error for failures
logger.error(f"Analysis failed: {e}")
```

### Diagnosing System State Issues

```python
# Add diagnostic logging before analysis
try:
    mode = self.oss.Mode
    num_fields = self.oss.SystemData.Fields.NumberOfFields
    num_wavelengths = self.oss.SystemData.Wavelengths.NumberOfWavelengths
    epd = self.oss.SystemData.Aperture.ApertureValue
    logger.info(f"System state: mode={mode}, fields={num_fields}, "
                f"wavelengths={num_wavelengths}, EPD={epd}")
except Exception as e:
    logger.warning(f"Could not get system diagnostics: {e}")
```

---

## ZosPy Documentation Links

- GitHub: https://github.com/MREYE-LUMC/ZOSPy
- Docs: https://zospy.readthedocs.io/
- Context7 library ID: `/mreye-lumc/zospy`

---

## Changelog

### 2026-02-04 (Part 8) - Add Spot Diagram Endpoint
- **NEW** `/spot-diagram` endpoint for spot diagram analysis
- Uses ZosPy `new_analysis` with `AnalysisIDM.StandardSpot`
- Parameters: `ray_density` (1-20), `reference` ('chief_ray' or 'centroid')
- Returns PNG image (or numpy array fallback) + per-field spot data
- Per-field data: `rms_radius`, `geo_radius`, `centroid_x`, `centroid_y`, `num_rays`
- Also returns `airy_radius` for diffraction limit comparison
- **Fallback**: Manual ray trace via `SingleRayTrace` if StandardSpot doesn't provide data

### 2026-02-04 (Part 7) - Add Wavefront Endpoint
- **NEW** `/wavefront` endpoint for wavefront error analysis
- Uses ZosPy `ZernikeStandardCoefficients` for RMS, P-V, and Strehl ratio
- Uses ZosPy `WavefrontMap` for OPD map visualization
- Returns raw numpy array (Mac side renders to PNG)
- Response: `rms_waves`, `pv_waves`, `strehl_ratio`, `wavelength_um`, `field_x`, `field_y`

### 2026-02-04 (Part 6) - Move matplotlib to Mac
- **REMOVED** matplotlib from worker
- Cross-section returns `image_format="numpy_array"` when PNG export fails
- Mac side reconstructs and renders using matplotlib
- **BENEFIT**: Worker is thinner, no matplotlib dependency on Windows

### 2026-02-04 (Part 5) - Remove Legacy LLM JSON Loading
- **REMOVED** `load_system()` and all `_setup_*` methods
- **REMOVED** `_extract_value()` and `_to_float()` helpers
- **REMOVED** `skip_load` parameter from all methods
- All endpoints now require `zmx_content` only

### 2026-02-04 (Part 4) - Dumb Executor for get_seidel
- Simplified from ~160 lines to ~80 lines
- **REMOVED**: ZOSAPI fallback, `_parse_zernike_text()`, placeholder generation
- **NEW**: Single try, return result or error
- Mac side handles Zernike to Seidel conversion

### 2026-02-04 (Part 3) - Dumb Executor for ray_trace_diagnostic
- **REMOVED**: Field-level aggregation, hotspot detection
- **NEW**: Returns raw per-ray data
- Added `_error_code_to_mode()` helper
- Mac side handles all aggregation

### 2026-02-04 (Part 2) - ZMX-Based Loading
- Added `load_zmx_file()` for native .zmx loading
- All conversion via Node.js `/convert-to-zmx`

### 2026-02-04 (Part 1) - Initial Wave
- Fixed `SelectWavelength` to `MakePrimary()`
- Added `float()` conversions for COM interop
- Moved Seidel conversion to Mac side

---

## Technical Debt

### Addressed
- Consolidated `get_paraxial_data` and `_get_paraxial_from_lde` - now delegates to single method
- Added debug logging to silent ray trace exceptions
- Simplified `get_seidel()` from ~160 to ~80 lines
- Simplified `ray_trace_diagnostic()` - raw data only
- Hotspot detection moved to Mac side
- Removed matplotlib dependency

### Remaining
- `distribution` parameter accepted but not used (always square grid)
- Surface index conventions inconsistent (some 0-indexed, some 1-indexed)
- Some ZosPy version compatibility issues may exist
