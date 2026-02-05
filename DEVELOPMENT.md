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

**Multiple Workers (Parallelism):**
```bash
# Single worker (default)
uvicorn main:app --host 0.0.0.0 --port 8787 --workers 1

# Multiple workers for parallel processing
uvicorn main:app --host 0.0.0.0 --port 8787 --workers 3
```

Each uvicorn worker is a separate process with its own OpticStudio connection. This allows true parallel processing but **each worker consumes a license seat**.

**License limits (per Ansys):**
| License Type | Max Simultaneous Instances |
|--------------|---------------------------|
| Professional (subscription) | 4 |
| Premium (subscription) | 8 |
| Perpetual (legacy 19.4+) | 2 |

Set `--workers N` where N ≤ your license limit. On macOS, set `TASK_QUEUE_WORKERS=N` to match.

## Service Ports

| Service | Port |
|---------|------|
| Quadoa Analysis Service | 8000 |
| Optiland Analysis Service | 8001 |
| Zemax Analysis Service | **8002** |
| Zemax Worker (this service) | 8787 |

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
- Seidel text parsing (`seidel_text_parser.py`) — moved from worker
- Response aggregation and hotspot detection
- Business logic (margins, clamping, thresholds)
- Matplotlib rendering (for wavefront numpy arrays)

## Project Structure

```
zemax-worker/
|-- main.py               # FastAPI app, endpoints, request/response models
|-- zospy_handler.py      # ZosPy wrapper, all OpticStudio operations
|-- check_seidel_support.py  # Diagnostic script for Seidel support
|-- diagnose_spot_diagram.py # Diagnostic for spot diagram issues
|-- diagnose_wavefront.py    # Diagnostic for wavefront issues
|-- DEVELOPMENT.md        # This file
|-- requirements.txt      # Python dependencies
```

---

## Endpoints Reference

| Endpoint | Purpose | Key Request Fields |
|----------|---------|-------------------|
| `GET /health` | Health check + OpticStudio status | - |
| `POST /load-system` | Load ZMX into OpticStudio | `zmx_content` |
| `POST /cross-section` | Cross-section diagram + paraxial | `zmx_content` |
| `POST /calc-semi-diameters` | Surface aperture calculation | `zmx_content` |
| `POST /ray-trace-diagnostic` | Ray failure analysis (raw data) | `zmx_content`, `num_rays` |
| `POST /seidel` | Zernike coefficients (raw) | `zmx_content` |
| `POST /seidel-native` | Native Seidel raw text (parsing on Mac side) | `zmx_content` |
| `POST /trace-rays` | Ray positions at surfaces | `zmx_content`, `num_rays` |
| `POST /wavefront` | Wavefront error map + metrics | `zmx_content`, `field_index`, `wavelength_index` |
| `POST /spot-diagram` | Spot diagram + spot radii | `zmx_content`, `ray_density` |
| `POST /evaluate-merit-function` | Evaluate merit function operands | `zmx_content`, `operand_rows` |

All endpoints return `{"success": true, ...}` or `{"success": false, "error": "..."}`.

---

## The "Dumb Executor" Pattern

Every analysis method follows this pattern:

1. **Try the ZosPy operation once**
2. **Return raw results on success**
3. **Return error on failure** - no fallbacks, no retries
4. **No post-processing** - Mac side handles aggregation

**Why:** Keeps Windows code simple. All complex logic (retries, fallbacks, error handling) in one place (Mac side). Easier to debug.

---

## Critical Rules for ZosPy

### 1. Type Conversion for COM Interop

**All numeric values to ZosPy MUST be Python `float()`, not numpy types:**
```python
# WRONG - numpy.float64 causes COM errors
surface.Radius = radius            # might be numpy type
ray_trace = SingleRayTrace(px=px)  # if px from np.linspace

# RIGHT
surface.Radius = float(radius)
ray_trace = SingleRayTrace(px=float(px))
```

This applies everywhere: `np.linspace()` results, values from DataFrames, anything from numpy.

### 2. Multiple Workers and Parallelism

**Each uvicorn worker process gets its own ZOS connection and OpticStudio instance.** This is because:
- ZOSPy's `ZOS` class is a singleton per process (only one instance allowed)
- The ZOS-API only supports a single connection per process
- But separate processes can each have their own connection

**This means `--workers N` works for parallelism**, with each worker handling requests independently. The `asyncio.Lock()` serializes requests within a single process only.

**Constraint:** Each worker consumes one OpticStudio license seat. Premium allows 8 instances, Professional allows 4, perpetual allows 2. Don't exceed your license limit.

**On macOS (zemax-analysis-service):** Set `TASK_QUEUE_WORKERS=N` to match your Windows worker count, so the task queue sends N concurrent requests.

### 3. Index Conventions

**ZosPy uses 1-based indexing for fields, wavelengths, Zernike terms:**
```python
# WRONG
field = oss.SystemData.Fields.GetField(0)    # Error!
wl = oss.SystemData.Wavelengths.GetWavelength(0)  # Error!

# RIGHT
field = oss.SystemData.Fields.GetField(1)    # First field
wl = oss.SystemData.Wavelengths.GetWavelength(1)  # First wavelength
```

**Surfaces are 0-based in LDE** (surface 0 = object, surface N = image):
```python
for i in range(1, oss.LDE.NumberOfSurfaces):  # Skip object surface
    surface = oss.LDE.GetSurfaceAt(i)
```

**API responses use 0-based** for consistency with LLM JSON schema.

### 4. Wavelength Selection

```python
# WRONG - method doesn't exist
wl_data.SelectWavelength(index)

# RIGHT
wl_data.GetWavelength(index).MakePrimary()
```

### 5. ZosPy 2.x UnitField Objects

ZosPy 2.x returns `UnitField` objects (not plain floats) for values with units:
```python
# WRONG
wavelength = result.data.wavelength  # UnitField(value=0.5876, unit='µm')
calculation = wavelength * 2         # TypeError!

# RIGHT - use _extract_value() helper
def _extract_value(obj, default=0.0):
    if obj is None: return default
    if hasattr(obj, 'value'):
        try: return float(obj.value)
        except (TypeError, ValueError): return default
    try: return float(obj)
    except (TypeError, ValueError): return default
```

Apply `_extract_value()` to: `ApertureValue`, `field.X/Y`, `surface.Radius/Thickness/SemiDiameter/Conic`, `Wavelength`, EFL, and any value that might have units.

### 6. DataFrame Column Names (ZosPy 2.1.4)

`SingleRayTrace` result DataFrame uses hyphenated column names:
```python
df = result.data.real_ray_trace_data
# Columns: ['Surf', 'X-coordinate', 'Y-coordinate', 'Z-coordinate',
#            'X-cosine', 'Y-cosine', 'Z-cosine', 'X-normal', 'Y-normal',
#            'Z-normal', 'Angle in', 'Path length', 'Comment']

# WRONG
x = last_row['X']  # KeyError!

# RIGHT
x = last_row['X-coordinate']
```

No `error_code` column exists — rays that reach the image surface are valid.

### 7. Attribute Checks

Properties vary by ZosPy/OpticStudio version. Always use `hasattr()`:
```python
if hasattr(data, 'front_focal_length'):
    ffl = data.front_focal_length
```

### 8. OpticStudio Text File Encoding

**UTF-16, not UTF-8:**
```python
# WRONG
with open(temp_path, 'r') as f: text = f.read()

# RIGHT
with open(temp_path, 'r', encoding='utf-16') as f: text = f.read()
```

---

## ZosPy Patterns That Work

### Loading a System from ZMX

```python
self.oss.load(file_path)
num_surfaces = self.oss.LDE.NumberOfSurfaces - 1  # Exclude object
```

### CrossSection Image Export

Use `surface_line_thickness` to show lens elements:
```python
cross_section = CrossSection(
    number_of_rays=11,
    surface_line_thickness="Thick",  # REQUIRED to show lenses!
    rays_line_thickness="Standard",
    field="All", wavelength="All",
    color_rays_by="Fields",
    delete_vignetted=True,
    image_size=(1200, 800),
)
result = cross_section.run(oss, image_output_file="/tmp/output.png")
```

### Native Seidel Analysis

```python
analysis = zp.analyses.new_analysis(
    self.oss,
    zp.constants.Analysis.AnalysisIDM.SeidelCoefficients,
    settings_first=True
)
analysis.ApplyAndWaitForCompletion()

# Export to text file (required - no direct data access)
temp_path = os.path.join(tempfile.gettempdir(), "seidel_native.txt")
analysis.Results.GetTextFile(temp_path)

# Parse (UTF-16!)
with open(temp_path, 'r', encoding='utf-16') as f:
    text_content = f.read()

analysis.Close()  # Always close!
```

### SingleRayTrace

```python
ray_trace = zp.analyses.raysandspots.SingleRayTrace(
    hx=0.0, hy=0.0,
    px=float(px), py=float(py),  # ALWAYS float()!
    wavelength=1,  # 1-indexed
    field=fi,      # 1-indexed
)
result = ray_trace.run(self.oss)

if hasattr(result.data, 'real_ray_trace_data'):
    df = result.data.real_ray_trace_data
    last_row = df.iloc[-1]
    x_val = last_row['X-coordinate']
    y_val = last_row['Y-coordinate']
```

### ZernikeStandardCoefficients

```python
result = zp.analyses.wavefront.ZernikeStandardCoefficients(
    sampling='64x64', maximum_term=37,
    wavelength=1, field=1, surface="Image",
).run(self.oss)

coeffs = result.data.coefficients
for term, coeff in coeffs.items():
    value = float(coeff.value) if hasattr(coeff, 'value') else float(coeff)
```

### WavefrontMap

```python
wavefront = zp.analyses.wavefront.WavefrontMap(
    sampling="64x64", wavelength=1, field=1,
    surface="Image", show_as="Surface",
    rotation="Rotate_0", use_exit_pupil=True,
).run(self.oss, oncomplete="Release")

arr = np.array(wavefront.data.values, dtype=np.float64)
# Return as numpy array (Mac renders to PNG)
```

### Spot Diagram via new_analysis

```python
analysis = zp.analyses.new_analysis(
    self.oss,
    zp.constants.Analysis.AnalysisIDM.StandardSpot,
    settings_first=True,
)
settings = analysis.Settings
if hasattr(settings, 'RayDensity'):
    settings.RayDensity = ray_density
analysis.ApplyAndWaitForCompletion()
analysis.Close()  # Always close!
```

---

## Native Seidel Text File Format

### Key Parsing Details

The text file from `GetTextFile()` has these characteristics:

**Multiple tables:** The file contains both "Seidel Aberration Coefficients" and "Seidel Aberration Coefficients in Waves". Parser must stop after the first table's TOT/Sum row.

**TOT vs Sum:** OpticStudio uses "TOT" for the totals row (not "Sum"). Check both case-insensitively.

**Paired columns:** The data has 12+ columns (SPHA/S1, COMA/S2, etc. — each pair shows the same value). Extract indices 1, 3, 5, 7, 9 for S1-S5, then 10, 11 for CLA, CTR.

**STO row:** The aperture stop surface appears as "STO" (not a number). `isdigit()` returns False — handle specially.

**Missing chromatic columns:** Single-wavelength systems may not have CLA/CTR columns. Default to 0.0.

### Native vs Zernike-Based Comparison

| Aspect | Native (`/seidel-native`) | Zernike-Based (`/seidel`) |
|--------|--------------------------|---------------------------|
| Per-surface | Yes | No (totals only) |
| Chromatic (CLA, CTR) | Yes | No |
| Accuracy | Exact third-order | S1-S3 accurate; S4, S5 approximate |
| Failure modes | May fail for non-sequential | More robust |

Mac service tries native first, falls back to Zernike.

---

## How to Add a New Endpoint

### Step 1: Add Models in `main.py`

```python
class MyAnalysisRequest(BaseModel):
    zmx_content: str = Field(description="Base64-encoded .zmx file content")
    my_param: int = Field(default=5)

class MyAnalysisResponse(BaseModel):
    success: bool
    my_result: Optional[float] = None
    error: Optional[str] = None
```

### Step 2: Add Handler in `zospy_handler.py`

```python
def get_my_analysis(self, my_param=5):
    """Dumb executor - returns raw data only."""
    try:
        result = some_analysis.run(self.oss)
        if not hasattr(result, 'data') or result.data is None:
            return {"success": False, "error": "Analysis returned no data"}
        return {"success": True, "my_result": float(result.data.some_value)}
    except Exception as e:
        return {"success": False, "error": f"Analysis failed: {e}"}
```

### Step 3: Add Endpoint in `main.py`

Use the `_run_endpoint` helper, which handles timed logging, lock acquisition, connection check, system loading, error handling, and response building:

```python
@app.post("/my-analysis", response_model=MyAnalysisResponse)
async def get_my_analysis(request: MyAnalysisRequest, _=Depends(verify_api_key)):
    return await _run_endpoint(
        "/my-analysis", MyAnalysisResponse, request,
        lambda: zospy_handler.get_my_analysis(my_param=request.my_param),
    )
```

For endpoints that need custom response mapping (e.g. converting dicts to Pydantic models), pass a `build_response` function:

```python
@app.post("/my-analysis", response_model=MyAnalysisResponse)
async def get_my_analysis(request: MyAnalysisRequest, _=Depends(verify_api_key)):
    def _build_response(result: dict) -> MyAnalysisResponse:
        if not result.get("success", False):
            return MyAnalysisResponse(success=False, error=result.get("error"))
        return MyAnalysisResponse(success=True, items=[MyItem(**i) for i in result["items"]])

    return await _run_endpoint(
        "/my-analysis", MyAnalysisResponse, request,
        lambda: zospy_handler.get_my_analysis(my_param=request.my_param),
        build_response=_build_response,
    )
```

### Step 4: Add Client Method in zemax-analysis-service

See `zemax-analysis-service/DEVELOPMENT.md` "How to Add a New Endpoint" section.

---

## Optimization via ZosPy (Future Reference)

Native Zemax optimization has not been implemented yet, but these are the proven ZosPy patterns:

### Setting Up Variables

```python
zp.solvers.variable(surface.RadiusCell)
zp.solvers.variable(surface.ThicknessCell)
zp.solvers.variable(surface.ConicCell)
```

### Merit Function Editor (MFE)

```python
oss.MFE.DeleteAllRows()
op1 = oss.MFE.GetOperandAt(1)
op1.ChangeType(zp.constants.Editors.MFE.MeritOperandType.EFFL)
op1.Target = 50.0
op1.Weight = 1
```

### Merit Function Evaluation (evaluate_merit_function)

```python
mfe = oss.MFE
mfe.DeleteAllRows()

# After DeleteAllRows, MFE retains 1 empty row
# First operand: GetOperandAt(1), subsequent: InsertNewOperandAt(n)
op1 = mfe.GetOperandAt(1)
op1.ChangeType(zp.constants.Editors.MFE.MeritOperandType.EFFL)
op1.Target = 100.0
op1.Weight = 1.0

# Set parameter cells (Int1/Int2 = IntegerValue, Hx-Py = DoubleValue)
mfe_cols = zp.constants.Editors.MFE.MeritColumn
cell = op1.GetOperandCell(mfe_cols.Param1)
cell.IntegerValue = 0  # Surface number

# Calculate and read back
total_merit = float(mfe.CalculateMeritFunction())
value = _extract_value(op1.Value)
contribution = _extract_value(op1.Contribution)
```

**Key quirks:**
- `DeleteAllRows()` leaves 1 empty row — use `GetOperandAt(1)` for first, `InsertNewOperandAt(n)` for rest
- Params 1-2 (Int1, Int2) use `cell.IntegerValue = int(val)`
- Params 3-6 (Hx, Hy, Px, Py) use `cell.DoubleValue = float(val)`
- Operand type resolution: `getattr(zp.constants.Editors.MFE.MeritOperandType, "EFFL")`
- Unknown operand codes raise `AttributeError` — catch per-row and continue

### Running Local Optimization

```python
opt = oss.Tools.OpenLocalOptimization()
opt.Algorithm = zp.constants.Tools.Optimization.OptimizationAlgorithm.DampedLeastSquares
opt.Cycles = zp.constants.Tools.Optimization.OptimizationCycles.Automatic
opt.NumberOfCores = 8
opt.RunAndWaitForCompletion()
opt.Close()
```

---

## Known Issues

### CrossSection "Object reference not set" Error

Means system has no fields defined (`NumberOfFields = 0`) or invalid aperture. Check `EFL=None` in load logs — indicates invalid system state. Pre-flight validation checks `NumberOfFields > 0`.

### Ray Trace Error Codes

| Code | Meaning |
|------|---------|
| 0 | No error (OK) |
| 1 | Ray missed surface (MISS) |
| 2 | Total Internal Reflection (TIR) |
| 3 | Ray reversed |
| 4 | Ray vignetted |

### Startup Hang (ZOSAPI imported to clr, then nothing)

**Symptom:** Logs show "ZOSAPI imported to clr" but server never starts.

**Cause:** ZosPy import loads ZOSAPI DLLs into the .NET CLR at module import time. On some systems, this DLL loading hangs due to:
- OpticStudio not installed or corrupt installation
- ZOSAPI DLL registration issues
- .NET Framework compatibility problems
- DLL trying to connect to OpticStudio immediately

**Fix:** ZosPy is now lazily imported in `zospy_handler.py`. The import only happens when `ZosPyHandler()` is first instantiated (in the lifespan function), not at module load time. This allows:
- FastAPI server to start immediately
- `/health` endpoint to respond (showing `opticstudio_connected: false`)
- The actual ZosPy import to happen in the background

If you still see hangs, check:
1. OpticStudio is properly installed
2. Python version matches OpticStudio requirements (3.9-3.11)
3. Try running OpticStudio manually first to ensure it works

### Doubled "ZOSAPI imported to clr" Log

Normal when running with `uvicorn main:app` (string reference). Uvicorn re-imports the module in its worker process. Each import triggers ZosPy's DLL loading. The second import is a no-op (DLLs already loaded).

### Other Known Issues

| Issue | Notes |
|-------|-------|
| `distribution="hexapolar"` | Uses square grid (logs warning) |
| ZosPy version differences | Attribute names may vary — use `hasattr()` |
| Enum handling | Use `.name` attribute when available for string comparison |

---

## Debugging Tips

### Testing Worker Locally

```bash
python main.py
curl http://localhost:8787/health
curl -X POST http://localhost:8787/cross-section -H "Content-Type: application/json" -d '{"zmx_content": "..."}'
```

### Diagnosing System State

```python
mode = self.oss.Mode
num_fields = self.oss.SystemData.Fields.NumberOfFields
num_wavelengths = self.oss.SystemData.Wavelengths.NumberOfWavelengths
epd = self.oss.SystemData.Aperture.ApertureValue
logger.info(f"System state: mode={mode}, fields={num_fields}, wls={num_wavelengths}, EPD={epd}")
```

### Diagnostic Scripts

| Script | When to Use |
|--------|-------------|
| `check_seidel_support.py` | Diagnose Seidel analysis capability on your OpticStudio installation |
| `diagnose_spot_diagram.py` | Spot diagram returns `success: true` but `image: null` and `num_rays: 0` |
| `diagnose_wavefront.py` | Wavefront returns `success: false` with "Could not compute wavefront metrics" |

---

## ZosPy Documentation Links

- GitHub: https://github.com/MREYE-LUMC/ZOSPy
- Docs: https://zospy.readthedocs.io/
- Context7 library ID: `/mreye-lumc/zospy`

---

## Changelog

### 2026-02-05: Move Seidel Text Parsing to Mac Side

**Change:** Moved ~230 lines of Seidel text parsing from `zospy_handler.py` to `zemax-analysis-service/seidel_text_parser.py`.

**Why:** The parsing logic is pure Python string manipulation with no OpticStudio/ZosPy dependency. It belongs on the Mac side per the "dumb executor" pattern.

**What changed:**
- `zospy_handler.py`: `get_seidel_native()` now returns raw UTF-16 text (`seidel_text` field) instead of parsed data
- `main.py`: `NativeSeidelResponse` simplified — just `success`, `seidel_text`, `num_surfaces`, `error`
- Removed from worker: `_parse_seidel_text()`, `_parse_header_line()`, `_parse_seidel_data_row()`, `_extract_floats()`, `_build_seidel_coefficients()`, `_build_seidel_totals()`, `SEIDEL_COEFFICIENT_KEYS`
- Created on Mac: `seidel_text_parser.py` with unified `_extract_paired_coefficients()` (merged the old duplicate `_build_seidel_coefficients`/`_build_seidel_totals`)

**API contract change:** `/seidel-native` response now contains `seidel_text` (raw string) instead of `per_surface`/`totals`/`header` objects. Mac-side `routers/analysis.py` updated to parse the text.

### 2026-02-05: DRY Up Endpoint Boilerplate

**Change:** Extracted `_run_endpoint()` helper in `main.py` to eliminate repeated boilerplate across all POST endpoints.

**Before (every endpoint):**
```python
with timed_operation(...):
    async with timed_lock_acquire(...):
        if _ensure_connected() is None: return error
        try: load system, call handler, build response
        except: handle error, return error
```

**After:**
```python
return await _run_endpoint("/name", ResponseClass, request, lambda: handler())
```

**Impact:** Reduced `main.py` from ~856 to ~738 lines. All POST endpoints except `/load-system` and `/health` now use the helper. The `build_response` parameter handles custom result-to-response mapping (used by spot-diagram and evaluate-merit-function).

### 2026-02-05: Corrected Multi-Worker Documentation

**Previous docs (WRONG):** Claimed single worker required due to "COM/STA single-threaded apartment" limitations.

**Actual truth (VERIFIED):** Multiple uvicorn workers work fine. Each process gets its own ZOS singleton and OpticStudio connection. The constraint is **license seats**, not threading.

**Sources:**
- https://optics.ansys.com/hc/en-us/articles/42712696418835-FAQ-on-opening-multiple-OpticStudio-instances
- https://community.zemax.com/zos-api-12/python-multiprocessing-licensing-error-3117
- ZOSPy FAQ: "The ZOS-API only supports a single connection per process"

**Changes:**
- Updated module docstring in `main.py`
- Updated "Multiple Workers" section in DEVELOPMENT.md
- Added `WORKERS` env var support to `main.py`
- Removed incorrect "single worker required" warnings

### 2026-02-05: Remove Manual Fallbacks — Enforce Dumb Executor Pattern

**Change:** Removed all manual fallback implementations from analysis methods. The worker now returns `success: false` with an error message when a ZosPy analysis fails, instead of silently falling back to slow manual computations.

**Removed:**
- `_compute_spot_data_manual()` — traced individual rays via SingleRayTrace (~89s for ~291 rays)
- `_create_spot_array_fallback()` — created numpy array from spot data
- `_create_spot_diagram_array()` — stub that always returned None
- `_calculate_airy_radius()` — manual Airy radius from wavelength/f-number

**Methods simplified:**
- `get_spot_diagram()` — returns error if StandardSpot analysis fails (no manual ray trace fallback)
- `get_cross_section()` — returns error if PNG export fails (no numpy array fallback, no "always success" pattern)
- `get_wavefront()` — returns error if ZernikeStandardCoefficients fails (no computing metrics from WavefrontMap array)

**Why:** The old fallbacks violated the dumb executor pattern, were extremely slow, and masked real failures. If a native ZosPy analysis fails, the Mac side should see the error and handle it appropriately.

### 2026-02-05: Fix UnitField Errors in Wavefront and Spot Diagram

**Bug:** `float() argument must be a string or a real number, not 'UnitField'`

**Root cause:** Multiple methods used `float()` directly on ZosPy 2.x `UnitField` objects instead of `_extract_value()`.

**Fixes applied:**

1. **`get_wavefront()`** - ZernikeStandardCoefficients metrics:
   - `peak_to_valley_to_chief` / `peak_to_valley_to_centroid`
   - `rms_to_chief` / `rms_to_centroid`
   - `strehl_ratio`, `rms`, `peak_to_valley`

2. **`_extract_airy_radius()`** - Spot diagram Airy radius:
   - `results.AiryRadius`
   - `results.GetAiryDiskRadius()`

3. **`_populate_spot_data_from_results()`** - Spot diagram metrics:
   - `series.RMS`, `series.GEO`
   - `fd.RMSSpotRadius`, `fd.GEOSpotRadius`
   - `fd.CentroidX`, `fd.CentroidY`
   - `fd.NumberOfRays` (wrapped in `int(_extract_value(...))`)

4. **`evaluate_merit_function()`** - Merit function total:
   - `mfe.CalculateMeritFunction()` result

**Lesson:** Always use `_extract_value()` for any ZosPy result attribute that might have units. See "Critical Rules for ZosPy" section 5.

### 2026-02-05: Merit Function Evaluation Endpoint

**Change:** Added `POST /evaluate-merit-function` endpoint and `evaluate_merit_function()` handler method.

**What it does:**
- Accepts ZMX content + list of operand rows (code, params, target, weight)
- Loads system, constructs MFE operands, calls `CalculateMeritFunction()`
- Returns per-row `value` and `contribution`, plus `total_merit`
- Per-row error handling: invalid operand codes are skipped, valid rows still evaluate

**Files modified:**
- `zospy_handler.py` — Added `evaluate_merit_function()` method
- `main.py` — Added Pydantic models (`MeritFunctionRequest`, `MeritFunctionResponse`, etc.) and endpoint

### 2026-02-05: Lazy Connection on Startup

**Change:** Removed eager `_init_zospy()` call from the `lifespan()` startup function.

**Why:**
- Server now starts instantly without waiting for ZosPy import and OpticStudio connection
- ZosPy import can take several seconds (loads .NET CLR and ZOSAPI DLLs)
- OpticStudio connection can hang on some systems
- `/health` endpoint was already designed to handle `zospy_handler = None`
- All other endpoints use `_ensure_connected()` which lazily connects on first request

**Behavior change:**
- Before: Server waits for OpticStudio connection on startup, then responds to requests
- After: Server starts immediately, `/health` returns `opticstudio_connected: false`, first real request triggers connection

**Connection mode:** Always uses `zos.connect(mode="standalone")` - no extension mode is attempted.

---

## Cross-Reference: zemax-analysis-service DEVELOPMENT.md

For Mac-side implementation details, parallel analysis patterns, and endpoint orchestration:
`zemax-analysis-service/DEVELOPMENT.md`
