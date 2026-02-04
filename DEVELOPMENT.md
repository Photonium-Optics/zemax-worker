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
- **USE `Viewer3D` instead of `CrossSection`** - CrossSection has export tool issues
- `CrossSection` may fail with "system viewer export tool failed to run" even on v25.2
- `Viewer3D` works reliably for image export per ZosPy examples
- `result.data` is a **numpy array**, NOT a PIL Image
- Use `plt.imshow(result.data)` then save the figure to PNG
- Always provide surface geometry as fallback for client-side rendering
- Use `oncomplete=OnComplete.Release` to clean up analysis window

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

## TODO / Known Issues

- [ ] CrossSection image export fails ("system viewer export tool failed") - using fallback surface geometry
- [ ] Ray trace header mismatch warnings (cosmetic, doesn't affect functionality)
- [ ] Consider caching loaded systems to avoid reloading on every request
- [ ] Seidel S4 (Petzval) and S5 (Distortion) are approximations - cannot compute true values from Zernike
