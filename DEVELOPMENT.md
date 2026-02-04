# Zemax Worker Development Notes

**IMPORTANT: Read this file before making changes to zemax-worker!**

> **Cross-reference**: If editing the Mac-side proxy (`zemax-analysis-service/`), see `zemax-analysis-service/DEVELOPMENT.md` instead.

## ZosPy Documentation

**Official ZosPy (Zemax API Python Wrapper) Resources:**
- GitHub: https://github.com/MREYE-LUMC/ZOSPy
- Documentation: https://zospy.readthedocs.io/
- PyPI: https://pypi.org/project/zospy/

**Context7 for AI-assisted lookup:**
- Library ID: `/mreye-lumc/zospy`
- Use Context7 MCP tools to query current API documentation

## Key Lessons Learned

### 1. ZosPy API Quirks

#### Wavelength Selection
- **WRONG**: `wl_data.SelectWavelength(index)` - Does NOT exist on `IWavelengths`
- **RIGHT**: `wl_data.GetWavelength(index).MakePrimary()` - Call on individual wavelength object

#### Type Conversion for .NET/COM
All numeric values passed to ZosPy/OpticStudio MUST be explicitly converted to Python `float()`:
```python
# WRONG - may pass strings from JSON
surface.Radius = radius

# RIGHT - explicit float conversion
surface.Radius = float(radius)
```

This applies to: radius, thickness, semi_diameter, conic, wavelength (um), weight, nd, vd, field x/y

#### CrossSection / System Viewer Analysis
- **REQUIRES OpticStudio >= 24.1.0** for image export (check `zos.version`)
- **USE `image_output_file` parameter** - This is the reliable way to export images:
```python
from zospy.analyses.systemviewers.cross_section import CrossSection

cross_section = CrossSection(
    number_of_rays=11,
    field="All",
    wavelength="All",
    color_rays_by="Fields",
    delete_vignetted=True,
    image_size=(1200, 800),
)
result = cross_section.run(oss, image_output_file="/path/to/output.png")
```
- If `image_output_file` is provided, ZosPy saves directly to that path
- `result.data` may also contain a numpy array as fallback
- Always provide surface geometry as fallback for client-side SVG rendering
- Raw ZOSAPI `Layouts.OpenCrossSectionExport()` often fails - prefer ZosPy wrapper

#### Zernike/Seidel Analysis
- ZosPy's `ZernikeStandardCoefficients().run()` has parsing bugs with OpticStudio v25.x
- Use raw ZOSAPI access via `zp.analyses.new_analysis()` to bypass parser:
```python
analysis = zp.analyses.new_analysis(
    self.oss,
    zp.constants.Analysis.AnalysisIDM.ZernikeStandardCoefficients,
    settings_first=True
)
```
- **Settings property names vary between versions** - wrap each setting in try/except
- `ReferenceOPDToVertex` may not exist on raw API settings object
- Use `dir(settings)` to debug available properties
- Results may be in `DataGrids` or `DataSeries` - check both

#### GeneralLensData Attributes
Not all attributes exist in all ZosPy versions. Use `hasattr()` checks:
- `effective_focal_length_air` - Usually exists
- `back_focal_length` - Usually exists
- `front_focal_length` - **MAY NOT EXIST** in some versions
- `total_track` - May not exist
- `exit_pupil_diameter` - May not exist

### 2. Threading Constraints

**CRITICAL**: Must run with `workers=1` due to COM single-threaded apartment (STA) requirements.
All ZosPy operations are serialized via `asyncio.Lock()`.

### 3. OpticStudio Version Compatibility

Current target: **OpticStudio v25.2** (Ansys 2025 R2)

Known issues:
- Text output parsing in ZosPy may fail - use raw API access
- Some analysis result structures differ from older versions

### 4. Error Handling

- ZosPy errors often indicate reconnection needed
- Implement `_reconnect_zospy()` fallback
- Log warnings but don't fail on non-critical data (paraxial properties)

### 5. Code Architecture Patterns

#### Logging
Use the Python `logging` module consistently throughout:
```python
import logging
logger = logging.getLogger(__name__)

# Use appropriate log levels:
logger.debug("Verbose debugging info")      # Development only
logger.info("Normal operational messages")  # Startup, connections
logger.warning("Non-fatal issues")          # Fallbacks, missing optional data
logger.error("Operation failures")          # Errors that affect results
```

**Never use `print()` statements** - they bypass log configuration and cannot be filtered.

#### Connection Management
Use helper functions to reduce boilerplate in endpoints:
```python
# In main.py
def _ensure_connected() -> Optional[ZosPyHandler]:
    """Ensure ZosPy is connected, attempting reconnection if needed."""
    global zospy_handler
    if zospy_handler is None:
        zospy_handler = _reconnect_zospy()
    return zospy_handler

def _handle_zospy_error(operation_name: str, error: Exception) -> None:
    """Handle ZosPy errors by logging and attempting reconnection."""
    global zospy_handler
    if isinstance(error, ZosPyError):
        logger.error(f"{operation_name} failed: {error}")
        zospy_handler = _reconnect_zospy()
    else:
        logger.error(f"{operation_name} unexpected error: {error}")
```

#### LLM JSON Value Extraction
Surface properties in LLM JSON can be direct values or objects with solve info:
```python
def _extract_value(self, spec: Any, default: Any = None) -> Any:
    """Extract value from spec that may be direct or {'value': ..., 'solve': ...}"""
    if spec is None:
        return default
    if isinstance(spec, dict):
        return spec.get("value", default)
    return spec
```

## Changelog

### 2026-02-04
- Fixed `SelectWavelength` -> `MakePrimary()` for setting primary wavelength
- Added explicit `float()` conversions for all numeric values passed to ZosPy
- Fixed CrossSection to handle numpy array result instead of PIL Image
- Made paraxial data extraction resilient with `hasattr()` checks
- Switched Seidel analysis to raw ZOSAPI access to avoid parser bugs

### 2026-02-04 (zemax-analysis-service)
- Fixed response field name mismatch: worker returns `success`, not `ok`
- zemax-analysis-service was checking `result.get("ok")` which was always None
- Also fixed: `result.get("error")` can be None even on failure, use `or` fallback

### 2026-02-04 (continued)
- Added debug logging to CrossSection and Seidel analysis
- Fixed Seidel: `ReferenceOPDToVertex` doesn't exist on raw ZOSAPI settings
- Wrapped each settings property in try/except for version compatibility
- Added fallback to DataSeries if DataGrids is empty

### 2026-02-04 (API consistency)
- Standardized all response models to use `success` instead of mixed `ok`/`success`
- Fixed field name mismatch in ray_trace_diagnostic: `failure_count` → `total_failures`, `failure_mode` → `dominant_mode`
- Fixed SingleRayTrace parameter confusion: px/py are pupil coords, hx/hy are field coords
- Fixed radius check: Zemax uses 0 for infinity, check `radius != 0 and abs(radius) < 1e10`
- Fixed Pydantic validation: `errors: list[dict[str, str]]` requires non-None strings - always use `str(error_msg)`
- `ISystemData.FirstOrderData` doesn't exist - calculate from LDE directly

### 2026-02-04 (Code Quality Review)
- Fixed bug in `ray_trace_diagnostic`: accessing `sf["failure_count"]` instead of `sf["total_failures"]`
- Converted all `print()` statements to proper `logging` calls
- Added proper docstrings to helper methods `_to_float()` and `_extract_value()`
- Refactored endpoint boilerplate into `_ensure_connected()` and `_handle_zospy_error()` helpers
- Reduced code duplication across all 6 API endpoints

## Common AI Mistakes to Avoid

1. **Don't assume ZOSAPI property names** - They vary between OpticStudio versions. Always use try/except.

2. **Don't use ZosPy text parsers with OpticStudio v25** - They fail with "Unexpected end-of-input". Use raw ZOSAPI.

3. **Don't confuse hx/hy with px/py** in SingleRayTrace:
   - `hx, hy`: Normalized field coordinates
   - `px, py`: Normalized pupil coordinates
   - When iterating over pupil with `field=fi`, set hx=hy=0 and vary px/py

4. **Don't return `None` in `dict[str, str]`** - Pydantic will fail. Always cast to `str()`.

5. **Don't assume `result.get("error") or "fallback"` handles None** - Empty string `""` passes through. Check explicitly.

6. **Don't use magic numbers for infinity** - Zemax uses 0 for flat surfaces, check `radius != 0 and abs(radius) < 1e10`.

### 2026-02-04 (Seidel Text Parsing)
- Added `_parse_zernike_text()` helper method to parse Zernike coefficients from raw ZOSAPI text output
- Improved dict key handling in Seidel - keys can be int or str depending on ZosPy version
- Added limit of 37 terms to avoid processing unnecessary data

## TODO / Known Issues

- [ ] CrossSection image export fails ("system viewer export tool failed") - using fallback surface geometry
- [ ] Ray trace header mismatch warnings (cosmetic, doesn't affect functionality)
- [ ] Consider caching loaded systems to avoid reloading on every request
- [ ] Seidel S4 (Petzval) and S5 (Distortion) are approximations - cannot compute true values from Zernike
- [ ] Test `_parse_zernike_text()` with actual OpticStudio text output format
