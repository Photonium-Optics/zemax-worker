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
| `POST /seidel-native` | Native Seidel coefficients (per-surface + chromatic) | `zmx_content` |
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

# Access ray data - ZosPy 2.1.4 uses 'X-coordinate', 'Y-coordinate' column names
if hasattr(result.data, 'real_ray_trace_data'):
    df = result.data.real_ray_trace_data
    # DataFrame columns: ['Surf', 'X-coordinate', 'Y-coordinate', 'Z-coordinate',
    #                     'X-cosine', 'Y-cosine', 'Z-cosine', 'X-normal', 'Y-normal',
    #                     'Z-normal', 'Angle in', 'Path length', 'Comment']
    last_row = df.iloc[-1]  # Image surface
    x_val = last_row['X-coordinate']  # NOT 'X' or 'x'
    y_val = last_row['Y-coordinate']  # NOT 'Y' or 'y'
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

### Native SeidelCoefficients Analysis

Native Seidel analysis provides accurate per-surface S1-S5 and chromatic aberrations (CLA, CTR) directly from OpticStudio.

```python
def get_seidel_native(self) -> dict[str, Any]:
    """Get native Seidel coefficients using OpticStudio's SeidelCoefficients analysis."""
    import tempfile
    import os

    try:
        idm = zp.constants.Analysis.AnalysisIDM
        analysis = zp.analyses.new_analysis(
            self.oss,
            idm.SeidelCoefficients,
            settings_first=True
        )
        analysis.ApplyAndWaitForCompletion()

        # Export results to text file (required - no direct data access)
        temp_path = os.path.join(tempfile.gettempdir(), "seidel_native.txt")
        analysis.Results.GetTextFile(temp_path)

        # Parse the text file (UTF-16 encoding!)
        with open(temp_path, 'r', encoding='utf-16') as f:
            text_content = f.read()

        parsed = self._parse_seidel_text(text_content)

        return {
            "success": True,
            "per_surface": parsed.get("per_surface", []),
            "totals": parsed.get("totals", {}),
            "header": parsed.get("header", {}),
        }
    except Exception as e:
        return {"success": False, "error": f"SeidelCoefficients failed: {e}"}
    finally:
        analysis.Close()
```

**Key points:**
- Uses `zp.analyses.new_analysis()` with `AnalysisIDM.SeidelCoefficients`
- Data accessed via `GetTextFile()` (no direct `.data` attribute)
- Text file is UTF-16 encoded (common for OpticStudio exports)
- Returns per-surface S1-S5 + CLA + CTR chromatic aberrations
- Sum row provides system totals

---

## Native Seidel Text File Format (Deep Dive)

The native Seidel text file from `GetTextFile()` has specific formatting that the parser must handle:

### File Encoding

**CRITICAL: File is UTF-16 encoded, not UTF-8!**
```python
# WRONG - will produce garbage or decode errors
with open(temp_path, 'r') as f:
    text = f.read()

# RIGHT
with open(temp_path, 'r', encoding='utf-16') as f:
    text = f.read()
```

### Header Section

The header contains metadata before the coefficient table:

```
Listing of Aberration Coefficient Data

File: C:\Users\...\system.zmx
Title: My Optical System
Date: Wednesday, February 05, 2026

Wavelength                      :               0.5876 µm
Petzval radius                  :            -623.2905
Optical Invariant               :               4.1551

Seidel Aberration Coefficients:
```

**Parsing header values:**
- Lines with `:` contain key-value pairs
- Key is before the colon, value is after
- Values may have units (e.g., `0.5876 µm`) - extract numeric part only
- Normalize keys: lowercase, replace spaces with underscores

### Column Header Row

The column header defines what data follows:

```
Surf     SPHA  S1    COMA  S2    ASTI  S3    FCUR  S4    DIST  S5    CLA    CTR
```

**Column meanings:**
| Column | Full Name | Seidel Coefficient |
|--------|-----------|-------------------|
| SPHA/S1 | Spherical | S1 (third-order spherical) |
| COMA/S2 | Coma | S2 (third-order coma) |
| ASTI/S3 | Astigmatism | S3 (third-order astigmatism) |
| FCUR/S4 | Field Curvature | S4 (Petzval + astigmatism) |
| DIST/S5 | Distortion | S5 (third-order distortion) |
| CLA | Axial Chromatic | Longitudinal chromatic aberration |
| CTR | Transverse Chromatic | Lateral chromatic aberration |

**IMPORTANT:** The paired columns (SPHA/S1, COMA/S2, etc.) show the same value. Only need to extract one of each pair.

### Data Rows

```
  1      0.114175   -0.008176   0.000586   0.120467   -0.008669   -0.034   0.002
  2      0.000396   -0.005093   0.065502  -0.042772   -0.292349   -0.002   0.029
STO     -0.000000    0.000000  -0.000000   0.000000    0.000000    0.000    0.000
  4      0.023456   -0.001234   0.005678  -0.012345   -0.067890   -0.001   0.003
Sum      0.114571   -0.013269   0.066088   0.077695   -0.301018   -0.036   0.031
```

**Row identification:**
- Surface numbers are right-aligned (may have leading spaces)
- `STO` = aperture stop surface
- `Sum` = totals row (sum of all surfaces)

**Parsing gotchas:**
1. Surface number detection: `parts[0].isdigit()` works for numbered surfaces, but need special handling for `STO`
2. Negative values: Scientific notation (`-1.234E-05`) may appear for very small values
3. Column count may vary: CLA/CTR columns are optional (only present for multi-wavelength systems)
4. Values are space-separated, not fixed-width

### Handling Missing Chromatic Columns

Some systems won't have CLA/CTR columns (single wavelength systems):

```python
coef_keys = ['S1', 'S2', 'S3', 'S4', 'S5', 'CLA', 'CTR']
for i, key in enumerate(coef_keys):
    if i < len(values):
        surface_data[key] = values[i]
    else:
        surface_data[key] = 0.0  # Default missing chromatic to 0
```

---

## Native Seidel vs Zernike-Based Comparison

| Aspect | Native Seidel (`/seidel-native`) | Zernike-Based (`/seidel`) |
|--------|----------------------------------|---------------------------|
| **Source** | OpticStudio SeidelCoefficients analysis | ZernikeStandardCoefficients + conversion |
| **Accuracy** | Exact third-order coefficients | S1-S3 accurate; S4, S5 are approximations |
| **Per-surface** | Yes - full breakdown | No - totals only |
| **Chromatic** | CLA + CTR per surface | Not available |
| **Units** | Waves | mm (wavelength-scaled) |
| **Requirements** | OpticStudio with analysis capability | Any OpticStudio version |
| **Failure modes** | May fail for non-sequential systems | More robust |
| **Mac-side processing** | Minimal (just parsing) | Zernike-to-Seidel conversion |

**When each is used:**
- Mac service tries native first (`/seidel-native`)
- Falls back to Zernike if native fails
- Response indicates which method was used via `units` field and presence of `per_surface`

---

## Diagnostic Script: check_seidel_support.py

Use this script to diagnose Seidel analysis capabilities on your OpticStudio installation.

### Running the Diagnostic

```bash
cd zemax-worker
python check_seidel_support.py
```

### What It Tests

1. **ZosPy version** - Confirms ZosPy is installed
2. **Sample file loading** - Loads a Double Gauss from OpticStudio samples
3. **SeidelCoefficients analysis** - Tests the native analysis we use
4. **SeidelDiagram analysis** - Tests the diagram variant
5. **GetTextFile output** - Shows the actual text file format

### Interpreting Results

**Successful output looks like:**
```
ZOSPy version: 1.3.0
Loaded: C:\...\Double Gauss 28 degree field.zmx

--- TEXT FILE OUTPUT ---
Text file content (1234 chars):
--------------------------------------------------
Listing of Aberration Coefficient Data
...
Surf     SPHA  S1    COMA  S2    ...
  1      0.114175   -0.008176   ...
--------------------------------------------------
```

**If GetTextFile fails:**
- Check OpticStudio license level (Premium/Professional required)
- Check ZosPy version (need >= 1.2.0)
- Check if sample file exists at expected path

### Modifying Sample File Path

If the default sample file isn't found:
```python
possible_paths = [
    r"C:\Users\Public\Documents\Zemax\Samples\Sequential\Objectives\Double Gauss 28 degree field.zmx",
    os.path.expanduser(r"~\Documents\Zemax\Samples\Sequential\Objectives\Double Gauss 28 degree field.zmx"),
    # Add your custom path here:
    r"D:\OpticStudio\Samples\YourFile.zmx",
]
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
| `distribution="hexapolar"` | Uses square grid (logs warning if hexapolar requested) |
| ZosPy version differences | Attribute names may vary |
| Seidel S4/S5 from Zernike | Approximations only (use `/seidel-native` for accurate values) |

### Surface Index Convention

**OpticStudio/ZosPy internally uses 1-based indices:**
- `LDE.GetSurfaceAt(1)` = first optical surface (after object surface 0)
- `Fields.GetField(1)` = first field
- `Wavelengths.GetWavelength(1)` = first wavelength
- `SingleRayTrace(field=1, wavelength=1)` uses 1-based indices

**API responses use 0-based indices for consistency with LLM JSON schema:**
- `field_index=0` = first field
- `surface_index=0` = first optical surface (surfaces[0] in LLM JSON)
- `wavelength_index=0` = first wavelength

**Rule of thumb:**
- When calling ZosPy/OpticStudio methods, use 1-based indices
- When returning data in API responses, convert to 0-based indices

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
- PyPI: https://pypi.org/project/zospy/

---

## Comprehensive ZosPy API Reference

This section provides detailed ZosPy patterns and best practices based on official documentation and proven implementations.

### Connection Modes

ZosPy supports two connection modes to OpticStudio:

#### Standalone Mode (Default)
Starts a new, invisible OpticStudio instance. Most performant for automated workflows.

```python
import zospy as zp

# Initialize the ZOS-API connection
zos = zp.ZOS()

# Connect in standalone mode (default)
oss = zos.connect()

# Explicitly specify standalone mode
oss = zos.connect("standalone")

# Connect to specific OpticStudio version
zos = zp.ZOS(opticstudio_directory="C:/Program Files/Zemax OpticStudio")
oss = zos.connect()
```

#### Extension Mode
Connects to an already-running OpticStudio instance. Useful for debugging.

```python
import zospy as zp

zos = zp.ZOS()
# Requires "Interactive Extension" mode enabled in OpticStudio
oss = zos.connect("extension")
```

**Key differences:**
| Aspect | Standalone | Extension |
|--------|------------|-----------|
| OpticStudio visibility | Hidden | Visible (user's session) |
| Performance | Fastest | Slower (shared with UI) |
| Debugging | Limited | Full UI access |
| Use case | Production automation | Development/debugging |

### Index Conventions (Critical!)

**ZosPy uses 1-based indexing for most operations:**

| Entity | Index Range | Notes |
|--------|-------------|-------|
| Surfaces | 0 to N | Surface 0 = Object, Surface N = Image |
| Fields | 1 to NumberOfFields | `GetField(1)` = first field |
| Wavelengths | 1 to NumberOfWavelengths | `GetWavelength(1)` = first wavelength |
| Zernike terms | 1 to max_term | Z1 = piston, Z4 = defocus |
| LDE rows | 0 to NumberOfSurfaces-1 | `GetSurfaceAt(0)` = Object surface |

**Common mistakes:**
```python
# WRONG - Fields are 1-indexed
field = oss.SystemData.Fields.GetField(0)  # Error!

# RIGHT
field = oss.SystemData.Fields.GetField(1)  # First field

# WRONG - Wavelengths are 1-indexed
wl = oss.SystemData.Wavelengths.GetWavelength(0)  # Error!

# RIGHT
wl = oss.SystemData.Wavelengths.GetWavelength(1)  # First wavelength

# Surface iteration (LDE is 0-indexed but has object surface at 0)
for i in range(1, oss.LDE.NumberOfSurfaces):  # Skip object surface
    surface = oss.LDE.GetSurfaceAt(i)
```

### Lens Data Editor (LDE) Operations

#### Accessing and Modifying Surfaces

```python
# Get surface count (includes object and image)
num_surfaces = oss.LDE.NumberOfSurfaces

# Access surfaces
object_surface = oss.LDE.GetSurfaceAt(0)  # Object at infinity
first_surface = oss.LDE.GetSurfaceAt(1)   # First optical surface
image_surface = oss.LDE.GetSurfaceAt(num_surfaces - 1)  # Image plane

# Set surface properties (ALWAYS use float()!)
surface.Radius = float(50.0)
surface.Thickness = float(5.0)
surface.Conic = float(-0.5)
surface.SemiDiameter = float(12.5)
surface.MechanicalSemiDiameter = float(15.0)

# Set material using solver
zp.solvers.material_model(
    surface.MaterialCell,
    refractive_index=1.5168,
    abbe_number=64.17
)

# Or use material name
surface.Material = "N-BK7"

# Add surface comment
surface.Comment = "Front lens surface"
```

#### Inserting New Surfaces

```python
# Insert surface at position 1 (after object)
input_beam = oss.LDE.InsertNewSurfaceAt(1)
input_beam.Comment = "Input beam"
input_beam.Thickness = float(10.0)

# Insert lens surfaces
lens_front = oss.LDE.InsertNewSurfaceAt(2)
lens_front.Radius = float(30.0)
lens_front.Thickness = float(5.0)
zp.solvers.material_model(lens_front.MaterialCell, refractive_index=1.5)

lens_back = oss.LDE.InsertNewSurfaceAt(3)
lens_back.Radius = float(-30.0)
lens_back.Thickness = float(50.0)  # Distance to image
```

#### Changing Surface Types

```python
# Change to aspheric surface
new_surface = oss.LDE.InsertNewSurfaceAt(3)
zp.functions.lde.surface_change_type(
    new_surface,
    zp.constants.Editors.LDE.SurfaceType.EvenAsphere
)
```

### System Data Configuration

#### Aperture Settings

```python
# Access aperture settings
aperture = oss.SystemData.Aperture

# Get current aperture value
epd = aperture.ApertureValue

# Set aperture type and value
# Common aperture types:
# - EntrancePupilDiameter
# - ImageSpaceFNum
# - ObjectSpaceNA
aperture.ApertureType = zp.constants.SystemData.ZemaxApertureType.EntrancePupilDiameter
aperture.ApertureValue = float(10.0)  # 10mm EPD
```

#### Field Configuration

```python
fields = oss.SystemData.Fields

# Get number of fields
num_fields = fields.NumberOfFields

# Configure first field (must exist)
field1 = fields.GetField(1)
field1.X = float(0.0)
field1.Y = float(0.0)
field1.Weight = float(1.0)

# Add additional fields
field2 = fields.AddField(X=0, Y=5.0, Weight=1)   # +5 degrees
field3 = fields.AddField(X=0, Y=-5.0, Weight=1)  # -5 degrees
field4 = fields.AddField(X=5.0, Y=0, Weight=1)   # +5 degrees X

# Find maximum field angle
max_field = 0.0
for i in range(1, fields.NumberOfFields + 1):
    f = fields.GetField(i)
    max_field = max(max_field, abs(f.Y), abs(f.X))
```

#### Wavelength Configuration

```python
wavelengths = oss.SystemData.Wavelengths

# Get number of wavelengths
num_wls = wavelengths.NumberOfWavelengths

# Access wavelength
wl = wavelengths.GetWavelength(1)  # 1-indexed!
wavelength_um = wl.Wavelength  # In micrometers

# Set primary wavelength
wl.MakePrimary()

# NOTE: Do NOT use SelectWavelength() - it doesn't exist!
# WRONG: wavelengths.SelectWavelength(1)
# RIGHT: wavelengths.GetWavelength(1).MakePrimary()
```

### Analysis Execution Patterns

#### Pattern 1: Direct Analysis Classes (Preferred)

```python
# ZosPy provides high-level analysis classes
result = zp.analyses.raysandspots.SingleRayTrace(
    hx=0.0, hy=0.0,      # Normalized field (-1 to 1)
    px=0.0, py=1.0,      # Normalized pupil (-1 to 1)
    wavelength=1,         # 1-indexed
    field=1,              # 1-indexed
    global_coordinates=True
).run(oss)

# Access results
if result.data.real_ray_trace_data is not None:
    df = result.data.real_ray_trace_data
    print(df)
```

#### Pattern 2: new_analysis() for Generic Access

Use when ZosPy doesn't have a dedicated analysis class:

```python
# Create analysis
analysis = zp.analyses.new_analysis(
    oss,
    zp.constants.Analysis.AnalysisIDM.SeidelCoefficients,
    settings_first=True  # Configure settings before running
)

# Run and wait
analysis.ApplyAndWaitForCompletion()

# Export results to file (some analyses require this)
import tempfile
temp_path = os.path.join(tempfile.gettempdir(), "results.txt")
analysis.Results.GetTextFile(temp_path)

# IMPORTANT: Text files are UTF-16 encoded!
with open(temp_path, 'r', encoding='utf-16') as f:
    content = f.read()

# Always close when done
analysis.Close()
```

#### Pattern 3: Analysis with Image Export

```python
cross_section = zp.analyses.systemviewers.cross_section.CrossSection(
    number_of_rays=11,
    surface_line_thickness="Thick",  # Required to show lens elements!
    rays_line_thickness="Standard",
    field="All",
    wavelength="All",
    color_rays_by="Fields",
    delete_vignetted=True,
    image_size=(1200, 800),
)
result = cross_section.run(oss, image_output_file="/tmp/cross_section.png")
```

### SingleRayTrace Deep Dive

SingleRayTrace is critical for ray failure analysis and spot diagram computation.

```python
# Normalized coordinates:
# hx, hy: Field coordinates (-1 to 1), ignored when field index is specified
# px, py: Pupil coordinates (-1 to 1)

# Marginal ray (full aperture, on-axis)
marginal = zp.analyses.raysandspots.SingleRayTrace(
    hx=0.0, hy=0.0,
    px=0.0, py=1.0,    # Top of pupil
    wavelength=1,
    field=1,           # Use field index, not hx/hy
    global_coordinates=True
).run(oss)

# Chief ray (center of pupil, off-axis)
chief = zp.analyses.raysandspots.SingleRayTrace(
    hx=0.0, hy=0.7,    # Used when field=0 (arbitrary field)
    px=0.0, py=0.0,    # Center of pupil
    wavelength=1,
    field=0,           # 0 = use hx, hy coordinates
    global_coordinates=True
).run(oss)

# Accessing ray data
if marginal.data.real_ray_trace_data is not None:
    df = marginal.data.real_ray_trace_data

    # DataFrame columns vary by ZosPy version
    # Common columns: X, Y, Z (or lowercase x, y, z)
    last_surface = df.iloc[-1]
    x_pos = last_surface.get('X', last_surface.get('x', None))
    y_pos = last_surface.get('Y', last_surface.get('y', None))
    z_pos = last_surface.get('Z-coordinate', last_surface.get('z', None))
```

**Pupil coordinate patterns for ray grids:**
```python
import numpy as np

# Square grid
num_rays_per_axis = 11
coords = np.linspace(-1, 1, num_rays_per_axis)
for px in coords:
    for py in coords:
        if px*px + py*py <= 1:  # Inside circular pupil
            trace_ray(float(px), float(py))  # ALWAYS float()!

# Polar grid
num_rings = 5
rays_per_ring = 12
for ring in range(num_rings):
    r = (ring + 1) / num_rings
    for angle in range(rays_per_ring):
        theta = 2 * np.pi * angle / rays_per_ring
        px = r * np.cos(theta)
        py = r * np.sin(theta)
        trace_ray(float(px), float(py))
```

### Wavefront Analysis

#### Zernike Standard Coefficients

```python
result = zp.analyses.wavefront.ZernikeStandardCoefficients(
    sampling="64x64",      # Pupil sampling: "32x32", "64x64", "128x128"
    maximum_term=37,       # Zernike terms 1-37 (standard)
    wavelength=1,          # 1-indexed
    field=1,               # 1-indexed
    surface="Image",       # Usually "Image"
    reference_opd_to_vertex=False,
    sx=0.0, sy=0.0, sr=0.0  # Subaperture (optional)
).run(oss)

# Access RMS and P-V
if hasattr(result.data, 'from_integration_of_the_rays'):
    integration = result.data.from_integration_of_the_rays
    rms_to_chief = integration.rms_to_chief
    strehl_ratio = integration.strehl_ratio

pv_to_chief = result.data.peak_to_valley_to_chief

# Access individual coefficients
coeffs = result.data.coefficients
for term, coeff in coeffs.items():
    if hasattr(coeff, 'value'):
        value = float(coeff.value)
    else:
        value = float(coeff)
    print(f"Z{term}: {value:.6f}")
```

#### Wavefront Map (OPD)

```python
wavefront = zp.analyses.wavefront.WavefrontMap(
    sampling="64x64",
    wavelength=1,
    field=1,
    surface="Image",
    show_as="Surface",     # "Surface", "Contour", "GreyScale"
    rotation="Rotate_0",   # "Rotate_0", "Rotate_90", "Rotate_180", "Rotate_270"
    scale=1,
    reference_to_primary=False,
    remove_tilt=False,
    use_exit_pupil=True,
).run(oss, oncomplete="Release")

# Data is a pandas DataFrame (2D grid)
if wavefront.data is not None:
    import numpy as np
    arr = np.array(wavefront.data.values, dtype=np.float64)
    # arr.shape is typically (64, 64) for 64x64 sampling
```

### FFT PSF Analysis

```python
result = zp.analyses.psf.FFTPSF(
    sampling="64x64",         # Pupil sampling
    display="64x64",          # Display grid (up to 2x sampling)
    rotation=0,               # 0, 90, 180, 270 degrees
    wavelength="All",         # Wavelength number or "All"
    field=1,                  # Field number
    psf_type="Linear",        # "Linear", "Log", "Phase", "Real", "Imaginary"
    use_polarization=False,
    image_delta=0,            # Delta in micrometers
    normalize=False,
    surface="Image"
).run(oss)

# Result is a 2D pandas DataFrame
import matplotlib.pyplot as plt
if result.data is not None:
    plt.imshow(result.data.values, cmap='hot', origin='lower')
    plt.colorbar(label='Intensity')
    plt.show()
```

### Optimization via ZosPy

#### Setting Up Variables

```python
# Mark parameters as variable for optimization
zp.solvers.variable(surface.RadiusCell)     # Radius
zp.solvers.variable(surface.ThicknessCell)  # Thickness
zp.solvers.variable(surface.ConicCell)      # Conic constant

# For extended parameters (aspheric coefficients, etc.)
zp.solvers.variable(surface.GetCellAt(12))  # Access by column index
```

#### Merit Function Editor (MFE)

```python
# Show editor (optional, for debugging)
oss.MFE.ShowEditor()

# Clear existing operands
oss.MFE.DeleteAllRows()

# Configure operand
op1 = oss.MFE.GetOperandAt(1)
op1.ChangeType(zp.constants.Editors.MFE.MeritOperandType.ZERN)
op1.GetOperandCell(zp.constants.Editors.MFE.MeritColumn.Param1).IntegerValue = 4
op1.Target = 0
op1.Weight = 1

# Add more operands
op2 = oss.MFE.AddOperand()
op2.ChangeType(zp.constants.Editors.MFE.MeritOperandType.EFFL)
op2.Target = 50.0
op2.Weight = 1
```

#### Running Local Optimization

```python
local_optimization = oss.Tools.OpenLocalOptimization()
local_optimization.Algorithm = zp.constants.Tools.Optimization.OptimizationAlgorithm.DampedLeastSquares
local_optimization.Cycles = zp.constants.Tools.Optimization.OptimizationCycles.Automatic
local_optimization.NumberOfCores = 8
local_optimization.RunAndWaitForCompletion()
local_optimization.Close()
```

### Python.NET and COM Interop Gotchas

#### Type Conversion (Critical!)

```python
# WRONG - numpy types cause COM errors
import numpy as np
coords = np.linspace(-1, 1, 11)
px = coords[5]  # numpy.float64
ray_trace = SingleRayTrace(px=px, ...)  # COM ERROR!

# RIGHT - always convert to Python float
px = float(coords[5])
ray_trace = SingleRayTrace(px=px, ...)  # Works!

# Pattern: always wrap numpy values
for px in np.linspace(-1, 1, 11):
    for py in np.linspace(-1, 1, 11):
        trace = SingleRayTrace(px=float(px), py=float(py), ...)
```

#### Interface Downcasting

Python.NET 3+ requires explicit downcasting for specific interface access:

```python
# Generic interface returned
settings = analysis.GetSettings()  # IAS_ (generic)

# Access specific property via __implementation__
normalize = settings.__implementation__.Normalize

# ZosPy's OpticStudioInterfaceEncoder usually handles this automatically
# but if you see AttributeError, try __implementation__
```

#### UTF-16 Text Files

OpticStudio exports text files in UTF-16 encoding:

```python
# WRONG - default encoding fails
with open(temp_path, 'r') as f:
    text = f.read()  # Garbage or decode error!

# RIGHT
with open(temp_path, 'r', encoding='utf-16') as f:
    text = f.read()
```

### Error Handling Best Practices

```python
def safe_analysis(oss):
    """Template for robust analysis execution."""
    try:
        # Validate system state first
        if oss.SystemData.Fields.NumberOfFields == 0:
            return {"success": False, "error": "System has no fields defined"}

        if oss.SystemData.Wavelengths.NumberOfWavelengths == 0:
            return {"success": False, "error": "System has no wavelengths defined"}

        # Run analysis
        result = some_analysis.run(oss)

        # Check for valid data
        if not hasattr(result, 'data') or result.data is None:
            return {"success": False, "error": "Analysis returned no data"}

        # Process and return
        return {
            "success": True,
            "data": process_result(result.data)
        }

    except Exception as e:
        # Log full traceback for debugging
        import traceback
        traceback.print_exc()
        return {"success": False, "error": f"Analysis failed: {e}"}
```

### Handling ZosPy 2.x UnitField Objects

ZosPy 2.x returns `UnitField` objects for values with units instead of plain floats:

```python
# ZosPy 2.x returns objects like:
# UnitField(value=0.5876, unit='µm')
# UnitField(value=50.0, unit='mm')

# WRONG - will fail with TypeError
wavelength = result.data.wavelength  # UnitField object, not float!
calculation = wavelength * 2  # TypeError!

# RIGHT - use helper to extract value
def _extract_value(obj: Any, default: float = 0.0) -> float:
    """Safely extract numeric value from ZosPy objects."""
    if obj is None:
        return default
    if hasattr(obj, 'value'):
        try:
            return float(obj.value)
        except (TypeError, ValueError):
            return default
    try:
        return float(obj)
    except (TypeError, ValueError):
        return default

wavelength = _extract_value(result.data.wavelength, 0.5876)
```

**When to use `_extract_value()`:**
- Extracting wavelength from Zernike analysis results
- Any value that might have units (lengths, angles, etc.)
- When unsure if ZosPy version returns plain float or UnitField

### Version Compatibility

ZosPy works with OpticStudio 20.1+ and Python 3.9-3.11.

```python
# Check ZosPy version
import zospy as zp
print(f"ZosPy version: {zp.__version__}")

# Check OpticStudio connection status
zos = zp.ZOS()
oss = zos.connect()
print(f"Connected: {oss is not None}")

# Get OpticStudio version (if available)
if hasattr(oss, 'SystemData'):
    # Version info may be in different places depending on ZosPy version
    pass
```

**Version-specific notes:**
- ZosPy 1.2.0+: Required for `new_analysis()` pattern
- ZosPy 1.3.0+: Improved DataFrame column consistency
- OpticStudio 23.1+: Some analysis settings may differ

---

## Changelog

### 2026-02-05 - Critical Bug Fixes (Seidel Parsing - Multiple Tables, TOT Row, UnitField)

**zospy_handler.py - Seidel Text Parser:**
- **FIXED** Parser reading multiple tables (44 surfaces instead of 13)
  - OpticStudio's Seidel text output contains multiple tables: "Seidel Aberration Coefficients" followed by "Seidel Aberration Coefficients in Waves"
  - Parser now stops after finding the TOT/Sum row instead of continuing into subsequent tables
  - Added `break` after `found_totals = True` in `_parse_seidel_text()`
- **FIXED** "No totals data parsed" error
  - OpticStudio uses "TOT" for the totals row, but parser only checked for "sum"
  - `_parse_seidel_data_row()` now checks for both "TOT" and "SUM" (case-insensitive)
- **FIXED** ZosPy 2.x UnitField objects causing failures
  - ZosPy 2.x returns `UnitField(value=0.4861, unit='µm')` objects instead of plain floats
  - Added `_extract_value()` helper to safely extract numeric values:
  ```python
  def _extract_value(obj: Any, default: float = 0.0) -> float:
      if hasattr(obj, 'value'):
          return float(obj.value)
      return float(obj) if obj is not None else default
  ```
  - Applied to wavelength extraction in `get_seidel()` (Zernike method)
- **FIXED** `/seidel` endpoint always returning `success=True`
  - Endpoint now checks `result.get("success", False)` before returning success response
  - Returns proper error message when analysis fails

**main.py - /seidel endpoint:**
- Added success check: `if not result.get("success", False): return error response`
- Added logging: `logger.info("seidel-native: Starting native Seidel analysis")`

### 2026-02-05 - Medium Severity Bug Fixes (Seidel Parsing, Spot Diagram)

**zospy_handler.py - Seidel Parsing:**
- **FIXED** STO row skipped in Seidel parsing - `isdigit()` returns False for "STO" (aperture stop), causing that row to be skipped
- `_parse_seidel_data_row()` now handles both numeric surface numbers and "STO" string
- STO surface is assigned surface number 0 as a marker in the per-surface results

**zospy_handler.py - Spot Diagram:**
- **FIXED** Chief ray reference not actually computed - comment said "chief ray reference" but code used centroid for both cases
- `_compute_spot_data_manual()` now traces and captures the actual chief ray position at `px=0, py=0`
- Chief ray position stored in `chief_ray_x`, `chief_ray_y` during ray tracing loop
- When `reference="chief_ray"`, uses actual chief ray position instead of centroid
- Falls back to centroid if chief ray wasn't captured (e.g., vignetted central ray)

### 2026-02-05 - Documentation and Low Severity Bug Fixes
- **IMPROVED** Added index convention documentation at top of zospy_handler.py constants section
- **IMPROVED** Added warning log when `distribution="hexapolar"` is requested (only square grid supported)
- **UPDATED** Known Issues section with detailed index convention explanation

### 2026-02-05 - Fix numpy.float64 COM Interop Bug
- **FIXED** Missing `float()` conversion for `px` and `py` in `SingleRayTrace` calls
- `np.linspace()` returns `numpy.float64`, but ZosPy COM interop requires Python `float()`
- Fixed in `ray_trace_diagnostic()` and `trace_rays()` methods
- `_compute_spot_data_manual()` already had correct `float()` conversion (good reference)
- **Pattern**: Always wrap numpy values in `float()` before passing to ZosPy analyses

### 2026-02-05 - Fix Seidel Coefficients Returning Zeros (Part 2)
- **FIXED** Paired column parsing in `_build_seidel_coefficients()` and `_build_seidel_totals()`
- OpticStudio text output has **12 columns** (paired: SPHA/S1, COMA/S2, etc.)
- Parser now correctly extracts indices 1, 3, 5, 7, 9, 10, 11 for S1-S5, CLA, CTR
- Added validation in `get_seidel_native()` to fail if no data extracted
- **Mac-side validation**: analysis.py now validates native results have non-zero data
- **Mac-side validation**: analysis.py now validates Zernike coefficients are not empty
- Added debug logging to trace coefficient extraction

### 2026-02-05 - Native Seidel Refactoring
- **REFACTORED** `get_seidel_native()` for improved modularity
- Extracted helper methods: `_check_analysis_errors()`, `_read_opticstudio_text_file()`, `_cleanup_analysis()`
- Extracted parsing helpers: `_parse_header_line()`, `_parse_seidel_data_row()`, `_extract_floats()`, `_build_seidel_coefficients()`, `_build_seidel_totals()`
- Added constants: `SEIDEL_COEFFICIENT_KEYS`, `SEIDEL_TEMP_FILENAME`, `RAY_ERROR_CODES`, `FNO_APERTURE_TYPES`
- Added constants: `DEFAULT_SAMPLING`, `DEFAULT_MAX_ZERNIKE_TERM`, `DEFAULT_NUM_CROSS_SECTION_RAYS`, `MIN_IMAGE_EXPORT_VERSION`
- Extracted `_get_fno()` helper used by both `get_paraxial_data()` and `_calculate_airy_radius()`
- **REFACTORED** `get_spot_diagram()` for improved modularity
- Extracted helper methods: `_configure_spot_analysis()`, `_export_analysis_image()`, `_extract_airy_radius()`, `_extract_spot_data_from_results()`, `_create_field_spot_data()`, `_populate_spot_data_from_results()`, `_create_spot_array_fallback()`
- Improved type hints throughout

### 2026-02-05 - Native Seidel Analysis
- **NEW** `/seidel-native` endpoint for accurate Seidel coefficients
- Uses native OpticStudio `SeidelCoefficients` analysis (not Zernike approximation)
- Returns per-surface S1-S5 coefficients
- Returns chromatic aberrations (CLA, CTR) per surface
- Returns totals (sum) for all coefficients
- Added `get_seidel_native()` and `_parse_seidel_text()` to `zospy_handler.py`
- Text file export required (UTF-16 encoding) - no direct `.data` access for this analysis
- Mac side tries native first, falls back to Zernike method if native fails

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

### 2026-02-05 - ZosPy 2.1.4 Comprehensive Fix
- **FIXED**: Column names in `real_ray_trace_data` DataFrame
  - Use `X-coordinate`, `Y-coordinate`, `Z-coordinate` (NOT `X`, `x`, `Y`, `y`)
- **REMOVED**: `error_code` column check (doesn't exist in ZosPy 2.x)
- **ADDED**: `_get_column_value()` helper for safe DataFrame/Series column access
- **FIXED**: UnitField extraction - ZosPy 2.x returns UnitField objects, not floats
  - Applied `_extract_value()` to: `ApertureValue`, `field.X/Y`, `surface.Radius/Thickness/SemiDiameter/Conic`, `Wavelength`, EFL
  - Locations: `get_paraxial_data()`, `get_surfaces()`, `calc_semi_diameters()`, `ray_trace_diagnostic()`, `trace_rays()`, `get_wavefront()`, `_compute_spot_data_manual()`, `calc_fno()`
- **FIXED**: Enum handling in `calc_fno()` - use `.name` attribute when available
- **UPDATED**: `_compute_spot_data_manual()` and cross-section ray tracing
- Spot diagram now works correctly with ZosPy 2.1.4

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
- Extracted magic strings/numbers into constants
- Extracted reusable helpers for analysis cleanup, image export, etc.
- Improved type hints throughout

### Remaining
- `distribution` parameter accepted but not used (always square grid)
- Surface index conventions inconsistent (some 0-indexed, some 1-indexed)
- Some ZosPy version compatibility issues may exist

---

## Diagnostic Scripts

### diagnose_spot_diagram.py

Use when spot diagram returns `success: true` but `image: null` and `num_rays: 0`.

```bash
cd zemax-worker
python diagnose_spot_diagram.py [optional_zmx_file]
```

**What it tests:**
1. SingleRayTrace API - checks `result.data.real_ray_trace_data` structure
2. DataFrame column names - finds actual X/Y column names
3. StandardSpot analysis - checks if this works as primary method

**RESOLVED (2026-02-05):** Fixed column name issue. ZosPy 2.1.4 uses:
- `X-coordinate`, `Y-coordinate`, `Z-coordinate` (NOT `X`, `x`, `Y`, `y`)
- No `error_code` column exists - rays that reach the image surface are valid
- DataFrame columns: `['Surf', 'X-coordinate', 'Y-coordinate', 'Z-coordinate', 'X-cosine', 'Y-cosine', 'Z-cosine', 'X-normal', 'Y-normal', 'Z-normal', 'Angle in', 'Path length', 'Comment']`

### diagnose_wavefront.py

Use when wavefront analysis returns `success: false` with error "Could not compute wavefront metrics".

```bash
cd zemax-worker
python diagnose_wavefront.py [optional_zmx_file]
```

**What it tests:**
1. ZernikeStandardCoefficients - P-V and RMS attribute paths
2. WavefrontMap - wavefront array extraction
3. Settings/Results attribute naming (lowercase vs PascalCase)

**Known issue (2026-02):** Both Zernike and WavefrontMap analyses may fail to extract metrics because:
- ZosPy version changes attribute paths
- `from_integration_of_the_rays` nesting may not exist
- `peak_to_valley_to_chief` vs `peak_to_valley` naming

---

## Cross-Reference: zemax-analysis-service DEVELOPMENT.md

For Mac-side implementation details, SSE streaming, and optimization architecture, see:
`zemax-analysis-service/DEVELOPMENT.md`

Key sections:
- "Optimization System Deep Dive" - Merit function, scipy integration
- "SSE Streaming Pattern" - Server-sent events for progress
- "Seidel Analysis Strategy" - Native-first fallback approach
- "Common Patterns" - Building LLM data, parallel analysis
