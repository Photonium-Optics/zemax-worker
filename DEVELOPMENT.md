# Zemax Worker Development Notes

**Read this before editing zemax-worker files.**

## Goal: Keep This Worker Thin

This worker follows the **"dumb executor" pattern**:
- Only OpticStudio/ZosPy operations belong here
- No business logic, aggregation, or fallback strategies
- Mac side (zemax-analysis-service) handles everything else

## Quick Start

**Prerequisites:** Windows 10/11, OpticStudio (Professional or Premium), Python 3.9-3.11, ZosPy >= 1.2.0

```bash
cd zemax-worker
pip install -r requirements.txt
python main.py          # Single worker
# OR: uvicorn main:app --host 0.0.0.0 --port 8787 --workers 3
```

**Interactive API Docs:** http://localhost:8787/docs

**Multiple Workers:** Each uvicorn worker is a separate process with its own OpticStudio connection. Each consumes a license seat:

| License Type | Max Instances |
|--------------|---------------|
| Professional (subscription) | 4 |
| Premium (subscription) | 8 |
| Perpetual (legacy 19.4+) | 2 |

On macOS, set `TASK_QUEUE_WORKERS=N` to match.

## Architecture

```
LLM JSON --> Node.js /convert-to-zmx --> .zmx content (base64) --> Worker --> oss.load(file)
```

**What stays on Windows (this worker):**
- System loading from .zmx files
- Analysis execution (cross-section, seidel, wavefront, spot diagram)
- Ray tracing, raw data extraction from LDE/SystemData

**What lives on Mac (zemax-analysis-service):**
- LLM JSON to ZMX conversion, Zernike-to-Seidel conversion
- Seidel text parsing (`seidel_text_parser.py`)
- Response aggregation, hotspot detection, matplotlib rendering
- Business logic (margins, clamping, thresholds)

## Project Structure

```
zemax-worker/
|-- main.py               # FastAPI app, endpoints, Pydantic models, _run_endpoint helper
|-- zospy_handler.py       # ZosPy wrapper — all OpticStudio COM operations
|-- utils/
|   |-- timing.py          # [TIMING] profiling context managers
|-- check_seidel_support.py   # Diagnostic: Seidel analysis capability
|-- diagnose_spot_diagram.py  # Diagnostic: spot diagram issues
|-- diagnose_wavefront.py     # Diagnostic: wavefront issues
|-- test_seidel_diagnostic.py # Diagnostic: both Seidel methods comparison
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
| `POST /seidel-native` | Native Seidel raw text (parsed on Mac) | `zmx_content` |
| `POST /trace-rays` | Ray positions at surfaces | `zmx_content`, `num_rays` |
| `POST /wavefront` | Wavefront error map + metrics | `zmx_content`, `field_index`, `wavelength_index` |
| `POST /spot-diagram` | Spot diagram + spot radii | `zmx_content`, `ray_density` |
| `POST /evaluate-merit-function` | Merit function evaluation | `zmx_content`, `operand_rows` |

All POST endpoints use `Depends(verify_api_key)` and return `{"success": true/false, ...}`.

---

## The `_run_endpoint` Helper

All POST endpoints except `/load-system` and `/health` use `_run_endpoint()` in `main.py`, which handles:

1. Timed logging (`[TIMING] /endpoint START/COMPLETE`)
2. Async lock acquisition (serializes ZosPy calls within a process)
3. Lazy connection check (`_ensure_connected()`)
4. System loading from base64 ZMX
5. Handler invocation and response building
6. Error handling with auto-reconnect on `ZosPyError`

**Basic usage:**
```python
@app.post("/my-analysis", response_model=MyResponse)
async def get_my_analysis(request: MyRequest, _=Depends(verify_api_key)):
    return await _run_endpoint(
        "/my-analysis", MyResponse, request,
        lambda: zospy_handler.get_my_analysis(param=request.param),
    )
```

**Custom response mapping** (for endpoints needing manual dict-to-model conversion):
```python
    return await _run_endpoint(
        "/my-analysis", MyResponse, request,
        lambda: zospy_handler.get_my_analysis(),
        build_response=lambda result: MyResponse(
            success=result.get("success", False),
            items=[MyItem(**i) for i in result.get("items", [])],
        ),
    )
```

**Default dict-splat path:** When no `build_response` is provided, `_run_endpoint` filters the handler result to known `model_fields` and splats into the response class. The `success` and `error` keys are excluded from the splat to avoid duplicate keyword args.

---

## Critical Rules for ZosPy

### 1. COM Type Conversion
**All numerics to ZosPy MUST be Python `float()`, not numpy types:**
```python
# WRONG: numpy.float64 causes COM errors
ray_trace = SingleRayTrace(px=px)  # px from np.linspace

# RIGHT
ray_trace = SingleRayTrace(px=float(px))
```

### 2. ZosPy 2.x UnitField Objects
ZosPy 2.x returns `UnitField` objects (not plain floats) for values with units. Always use `_extract_value()`:
```python
def _extract_value(obj, default=0.0):
    if obj is None: return default
    if hasattr(obj, 'value'):
        try: return float(obj.value)
        except (TypeError, ValueError): return default
    try: return float(obj)
    except (TypeError, ValueError): return default
```

Apply to: `ApertureValue`, `field.X/Y`, `surface.Radius/Thickness/SemiDiameter/Conic`, `Wavelength`, EFL, any analysis metric (RMS, P-V, Strehl, Airy radius, spot radii, MFE values).

### 3. Index Conventions
- **Fields, wavelengths, Zernike terms:** 1-based (`GetField(1)`, `GetWavelength(1)`)
- **LDE surfaces:** 0-based (surface 0 = object, skip it: `for i in range(1, oss.LDE.NumberOfSurfaces)`)
- **API responses:** 0-based for consistency with LLM JSON schema

### 4. Wavelength Selection
```python
# WRONG: method doesn't exist
wl_data.SelectWavelength(index)
# RIGHT
wl_data.GetWavelength(index).MakePrimary()
```

### 5. DataFrame Column Names (ZosPy 2.1.4)
`SingleRayTrace` uses hyphenated columns: `'X-coordinate'`, `'Y-coordinate'`, etc. No `error_code` column exists.

### 6. UTF-16 Text Files
OpticStudio exports UTF-16, not UTF-8:
```python
with open(temp_path, 'r', encoding='utf-16') as f: text = f.read()
```

### 7. Attribute Checks
Properties vary by version. Always use `hasattr()`:
```python
if hasattr(data, 'front_focal_length'):
    ffl = data.front_focal_length
```

---

## ZosPy Patterns That Work

### Loading a System
```python
self.oss.load(file_path)
num_surfaces = self.oss.LDE.NumberOfSurfaces - 1  # Exclude object
```

### CrossSection Image Export
```python
cross_section = CrossSection(
    number_of_rays=11, surface_line_thickness="Thick",
    rays_line_thickness="Standard", field="All", wavelength="All",
    color_rays_by="Fields", delete_vignetted=True, image_size=(1200, 800),
)
result = cross_section.run(oss, image_output_file="/tmp/output.png")
```

### Native Seidel Analysis
```python
analysis = zp.analyses.new_analysis(
    self.oss, zp.constants.Analysis.AnalysisIDM.SeidelCoefficients,
    settings_first=True,
)
analysis.ApplyAndWaitForCompletion()
temp_path = os.path.join(tempfile.gettempdir(), "seidel_native.txt")
analysis.Results.GetTextFile(temp_path)
with open(temp_path, 'r', encoding='utf-16') as f:
    text_content = f.read()
analysis.Close()  # Always close!
```

### Spot Diagram — Why No Image Export

**ZOSAPI's `StandardSpot` does NOT support image export.** Unlike `CrossSection` which uses a layout tool with `SaveImageAsFile` + `OutputFileName`, `StandardSpot` is a regular analysis created via `new_analysis()`. The raw ZOSAPI analysis object has no `ExportGraphicAs` method.

**Available data from `StandardSpot.Results`:**
- SpotData: RMS radius, GEO radius, centroid X/Y via `GetRMSSpotSizeFor()`, `GetGeoSpotSizeFor()`
- Airy radius via `AiryRadius` property
- **NOT available:** Raw ray X,Y positions

**Solution: Batch ray tracing for raw ray data**
```python
ray_trace = oss.Tools.OpenBatchRayTrace()
norm_unpol = ray_trace.CreateNormUnpol(
    max_rays,
    zp.constants.Tools.RayTrace.RaysType.Real,
    oss.LDE.NumberOfSurfaces - 1,  # Image surface
)
# AddRay signature: (WaveNumber, Hx, Hy, Px, Py, OPDMode)
# WaveNumber is 1-based, Hx/Hy are normalized field coords, Px/Py are normalized pupil coords
opd_none = zp.constants.Tools.RayTrace.OPDMode.None_
for px, py in pupil_coords:
    norm_unpol.AddRay(wavelength_index, float(hx_norm), float(hy_norm), float(px), float(py), opd_none)
ray_trace.RunAndWaitForCompletion()
# Read results
success, ray_num, err_code, vig_code, x, y, z, l, m, n, l2, m2, intensity = (
    norm_unpol.ReadNextResult()
)
ray_trace.Close()
```

**Mac-side rendering:** The worker returns `spot_rays` (list of X,Y positions per field/wavelength). zemax-analysis-service renders the plot using matplotlib.

### Merit Function Evaluation
```python
mfe = oss.MFE
mfe.DeleteAllRows()
# DeleteAllRows leaves 1 empty row — use GetOperandAt(1) for first
op1 = mfe.GetOperandAt(1)
op1.ChangeType(zp.constants.Editors.MFE.MeritOperandType.EFFL)
op1.Target = 100.0
op1.Weight = 1.0
# Params 1-2: IntegerValue, Params 3-6: DoubleValue
mfe_cols = zp.constants.Editors.MFE.MeritColumn
cell = op1.GetOperandCell(mfe_cols.Param1)
cell.IntegerValue = 0
total_merit = _extract_value(mfe.CalculateMeritFunction())
```

---

## Native Seidel Text Format

The text file from `GetTextFile()` contains two tables — parser must stop after the first TOT/Sum row. Key details:
- **Paired columns:** 12+ values (SPHA/S1, COMA/S2, ..., CLA, CTR). Extract indices 1,3,5,7,9 for S1-S5, then 10,11 for CLA/CTR.
- **TOT vs Sum:** Both appear; check case-insensitively.
- **STO row:** Aperture stop is "STO" not a number.
- **Missing chromatic:** Single-wavelength systems may omit CLA/CTR. Default to 0.0.

| Aspect | Native (`/seidel-native`) | Zernike-Based (`/seidel`) |
|--------|--------------------------|---------------------------|
| Per-surface | Yes | No (totals only) |
| Chromatic (CLA, CTR) | Yes | No |
| Accuracy | Exact third-order | S1-S3 accurate; S4, S5 approximate |

---

## Known Issues

| Issue | Notes |
|-------|-------|
| CrossSection "Object reference not set" | System has no fields or invalid aperture |
| `distribution="hexapolar"` | Uses square grid (logs warning) |
| Startup hang at "ZOSAPI imported to clr" | ZosPy import loads .NET CLR DLLs; mitigated by lazy import in `zospy_handler.py` |
| Doubled "ZOSAPI imported to clr" log | Normal with `uvicorn main:app` — re-import in worker process, no-op |
| StandardSpot no image export | ZOSAPI limitation — use batch ray trace for raw data, render on Mac side |

### Ray Trace Error Codes
| Code | Meaning |
|------|---------|
| 0 | OK |
| 1 | MISS (ray missed surface) |
| 2 | TIR (total internal reflection) |
| 3 | Reversed |
| 4 | Vignetted |

---

## Debugging

**Test locally:**
```bash
python main.py
curl http://localhost:8787/health
```

**Diagnostic scripts:**
| Script | When to Use |
|--------|-------------|
| `check_seidel_support.py` | Seidel analysis not working |
| `diagnose_spot_diagram.py` | Spot diagram returns null image/zero rays |
| `diagnose_wavefront.py` | Wavefront returns "Could not compute metrics" |
| `test_seidel_diagnostic.py` | Compare native vs Zernike Seidel methods |

## ZosPy Documentation

- GitHub: https://github.com/MREYE-LUMC/ZOSPy
- Docs: https://zospy.readthedocs.io/
- Context7 library ID: `/mreye-lumc/zospy`

---

## Changelog

### 2026-02-05
- **Fix spot diagram image export** — ZOSAPI's `StandardSpot` analysis does NOT support image export (no `ExportGraphicAs` method). Unlike `CrossSection` which uses a layout tool with `SaveImageAsFile`, `StandardSpot` is a regular analysis without image export. Solution: use batch ray tracing (`IBatchRayTrace`) to get raw ray X,Y positions, return them in `spot_rays` field for Mac-side matplotlib rendering. The `spot_data` field still contains metrics (RMS, GEO radius) from `StandardSpot.Results`.
- **Move Seidel text parsing to Mac side** — `get_seidel_native()` now returns raw UTF-16 text; parsing in `zemax-analysis-service/seidel_text_parser.py`
- **DRY up endpoint boilerplate** — Extracted `_run_endpoint()` helper; reduced `main.py` from ~856 to ~738 lines
- **Multiple workers verified** — Each uvicorn worker process gets its own ZOS singleton; constraint is license seats, not threading
- **Remove manual fallbacks** — No more slow SingleRayTrace fallbacks for spot diagram/cross-section/wavefront
- **Fix batch ray trace AddRay call** — `IRayTraceNormUnpolData.AddRay` requires 6 params: `(WaveNumber, Hx, Hy, Px, Py, OPDMode)`. Was passing only 5 with wrong parameter mapping (missing OPDMode, WaveNumber in wrong position). Also added Hx normalization (was hardcoded to 0).
- **Fix UnitField errors** — Applied `_extract_value()` to all wavefront, spot diagram, and MFE metrics
- **Fix merit function NoneType** — Empty `row_errors` list no longer converted to `None`
- **Merit function endpoint** — Added `POST /evaluate-merit-function` with MFE operand construction
- **Lazy startup** — Server starts instantly; ZosPy import and OpticStudio connection deferred to first request

---

## Cross-Reference

Mac-side orchestration, caching, task queue, rendering: `zemax-analysis-service/DEVELOPMENT.md`
