from typing import Any, Literal, Optional
from pydantic import BaseModel, Field


class SpotDiagramRequest(BaseModel):
    """Spot diagram analysis request."""
    zmx_content: str = Field(description="Base64-encoded .zmx file content")
    ray_density: int = Field(default=20, ge=5, le=40, description="Ray density for pupil sampling")
    reference: str = Field(default="centroid", description="Reference point: 'chief_ray' or 'centroid'")
    field_index: Optional[int] = Field(default=None, ge=1, description="Field index (1-indexed). None = all fields.")
    wavelength_index: Optional[int] = Field(default=None, ge=1, description="Wavelength index (1-indexed). None = all wavelengths.")


class SpotFieldData(BaseModel):
    """Spot diagram data for a single field point."""
    field_index: int = Field(description="0-indexed field number")
    field_x: float = Field(description="Field X coordinate")
    field_y: float = Field(description="Field Y coordinate")
    rms_radius: Optional[float] = Field(default=None, description="RMS spot radius in lens units")
    geo_radius: Optional[float] = Field(default=None, description="GEO (max) spot radius in lens units")
    centroid_x: Optional[float] = Field(default=None, description="Centroid X coordinate on image plane")
    centroid_y: Optional[float] = Field(default=None, description="Centroid Y coordinate on image plane")
    num_rays: Optional[int] = Field(default=None, description="Number of rays traced for this field")


class SpotRayPoint(BaseModel):
    """A single ray hit point on the image plane."""
    x: float = Field(description="X coordinate on image plane")
    y: float = Field(description="Y coordinate on image plane")


class SpotRayData(BaseModel):
    """Raw ray data for a single field/wavelength combination."""
    field_index: int = Field(description="0-based field index")
    field_x: float = Field(description="Field X coordinate")
    field_y: float = Field(description="Field Y coordinate")
    wavelength_index: int = Field(description="0-based wavelength index")
    wavelength_um: float = Field(default=0.0, description="Wavelength in micrometers")
    rays: list[SpotRayPoint] = Field(default_factory=list, description="Ray hit points on image plane")


class SpotDiagramResponse(BaseModel):
    """
    Spot diagram analysis response.

    Returns spot metrics (RMS, GEO radius) and raw ray data for Mac-side rendering.
    ZOSAPI's StandardSpot does NOT support image export - use spot_rays for rendering.
    """
    success: bool = Field(description="Whether the operation succeeded")
    image: Optional[str] = Field(default=None, description="Always None - ZOSAPI StandardSpot doesn't support image export")
    image_format: Optional[str] = Field(default=None, description="Always None")
    array_shape: Optional[list[int]] = Field(default=None, description="Always None")
    array_dtype: Optional[str] = Field(default=None, description="Always None")
    spot_data: Optional[list[SpotFieldData]] = Field(default=None, description="Per-field spot metrics (RMS, GEO radius, centroid)")
    spot_rays: Optional[list[SpotRayData]] = Field(default=None, description="Raw ray X,Y positions for Mac-side rendering")
    airy_radius: Optional[float] = Field(default=None, description="Airy disk radius in lens units")
    wavelength_info: Optional[list[dict]] = Field(default=None, description="Wavelength info [{index, um}, ...]")
    num_fields: Optional[int] = Field(default=None, description="Number of fields in the system")
    num_wavelengths: Optional[int] = Field(default=None, description="Number of wavelengths in the system")
    error: Optional[str] = Field(default=None, description="Error message if operation failed")


class MTFRequest(BaseModel):
    """MTF analysis request."""
    zmx_content: str = Field(description="Base64-encoded .zmx file content")
    field_index: int = Field(default=0, ge=0, description="Field index (0 = all fields, 1+ = specific field, 1-indexed)")
    wavelength_index: int = Field(default=1, ge=1, description="Wavelength index (1-indexed)")
    sampling: str = Field(default="64x64", description="Pupil sampling grid")
    maximum_frequency: float = Field(default=0.0, ge=0, description="Maximum spatial frequency (cycles/mm). 0 = auto.")


class MTFFieldData(BaseModel):
    """MTF data for a single field point."""
    field_index: int = Field(description="0-indexed field number")
    field_x: float = Field(description="Field X coordinate")
    field_y: float = Field(description="Field Y coordinate")
    tangential: list[float] = Field(default_factory=list, description="Tangential MTF values")
    sagittal: list[float] = Field(default_factory=list, description="Sagittal MTF values")


class MTFResponse(BaseModel):
    """MTF analysis response."""
    success: bool = Field(description="Whether the operation succeeded")
    frequency: Optional[list[float]] = Field(default=None, description="Spatial frequency array (cycles/mm)")
    fields: Optional[list[MTFFieldData]] = Field(default=None, description="Per-field MTF data")
    diffraction_limit: Optional[list[float]] = Field(default=None, description="Diffraction-limited MTF curve")
    cutoff_frequency: Optional[float] = Field(default=None, description="Cutoff frequency (cycles/mm)")
    wavelength_um: Optional[float] = Field(default=None, description="Wavelength in micrometers")
    error: Optional[str] = Field(default=None, description="Error message if operation failed")


class HuygensMTFRequest(BaseModel):
    """Huygens MTF analysis request."""
    zmx_content: str = Field(description="Base64-encoded .zmx file content")
    field_index: int = Field(default=0, ge=0, description="Field index (0 = all fields, 1+ = specific field, 1-indexed)")
    wavelength_index: int = Field(default=1, ge=1, description="Wavelength index (1-indexed)")
    sampling: str = Field(default="64x64", description="Pupil sampling grid")
    maximum_frequency: float = Field(default=0.0, ge=0, description="Maximum spatial frequency (cycles/mm). 0 = auto.")


class RayFanRequest(BaseModel):
    """Ray Fan analysis request."""
    zmx_content: str = Field(description="Base64-encoded .zmx file content")
    field_index: int = Field(default=0, ge=0, description="Field index (0 = all fields, 1+ = specific field, 1-indexed)")
    wavelength_index: int = Field(default=0, ge=0, description="Wavelength index (0 = all wavelengths, 1+ = specific, 1-indexed)")
    plot_scale: float = Field(default=0.0, ge=0, description="Maximum vertical scale for plots; 0 = auto")
    number_of_rays: int = Field(default=20, ge=5, le=100, description="Number of rays traced on each side of origin")


class RayFanFieldData(BaseModel):
    """Ray fan data for a single field/wavelength combination."""
    field_index: int = Field(description="0-indexed field number")
    field_x: float = Field(description="Field X coordinate")
    field_y: float = Field(description="Field Y coordinate")
    wavelength_um: float = Field(default=0.0, description="Wavelength in micrometers")
    wavelength_index: int = Field(default=0, description="Wavelength index")
    tangential_py: list[float] = Field(default_factory=list, description="Tangential pupil Y coordinates")
    tangential_ey: list[float] = Field(default_factory=list, description="Tangential aberration EY values")
    sagittal_px: list[float] = Field(default_factory=list, description="Sagittal pupil X coordinates")
    sagittal_ex: list[float] = Field(default_factory=list, description="Sagittal aberration EX values")


class RayFanResponse(BaseModel):
    """Ray Fan analysis response."""
    success: bool = Field(description="Whether the operation succeeded")
    fans: Optional[list[RayFanFieldData]] = Field(default=None, description="Per-field/wavelength fan data")
    max_aberration: Optional[float] = Field(default=None, description="Maximum aberration value")
    num_fields: int = Field(default=0, description="Number of fields in the system")
    num_wavelengths: int = Field(default=0, description="Number of wavelengths")
    error: Optional[str] = Field(default=None, description="Error message if operation failed")


class PSFRequest(BaseModel):
    """PSF analysis request."""
    zmx_content: str = Field(description="Base64-encoded .zmx file content")
    field_index: int = Field(default=1, ge=1, description="Field index (1-indexed)")
    wavelength_index: int = Field(default=1, ge=1, description="Wavelength index (1-indexed)")
    sampling: str = Field(default="64x64", description="Pupil sampling grid")


class PSFResponse(BaseModel):
    """PSF analysis response."""
    success: bool = Field(description="Whether the operation succeeded")
    image: Optional[str] = Field(default=None, description="Base64-encoded numpy array bytes")
    image_format: Optional[str] = Field(default=None, description="Image format: 'numpy_array'")
    array_shape: Optional[list[int]] = Field(default=None, description="Shape for numpy array reconstruction")
    array_dtype: Optional[str] = Field(default=None, description="Dtype for numpy array reconstruction")
    strehl_ratio: Optional[float] = Field(default=None, description="Strehl ratio (0-1)")
    psf_peak: Optional[float] = Field(default=None, description="Peak PSF intensity")
    wavelength_um: Optional[float] = Field(default=None, description="Wavelength in micrometers")
    field_x: Optional[float] = Field(default=None, description="Field X coordinate")
    field_y: Optional[float] = Field(default=None, description="Field Y coordinate")
    error: Optional[str] = Field(default=None, description="Error message if operation failed")


class HuygensPSFRequest(BaseModel):
    """Huygens PSF analysis request."""
    zmx_content: str = Field(description="Base64-encoded .zmx file content")
    field_index: int = Field(default=1, ge=1, description="Field index (1-indexed)")
    wavelength_index: int = Field(default=1, ge=1, description="Wavelength index (1-indexed)")
    sampling: str = Field(default="64x64", description="Pupil sampling grid")


class ThroughFocusMTFRequest(BaseModel):
    """Through Focus MTF analysis request."""
    zmx_content: str = Field(description="Base64-encoded .zmx file content")
    field_index: int = Field(default=0, ge=0, description="Field index (0 = all fields, 1+ = specific field, 1-indexed)")
    wavelength_index: int = Field(default=1, ge=1, description="Wavelength index (1-indexed)")
    sampling: str = Field(default="64x64", description="Pupil sampling grid")
    delta_focus: float = Field(default=0.1, gt=0, description="Focus step size in mm")
    frequency: float = Field(default=0.0, ge=0, description="Spatial frequency (cycles/mm). 0 = default.")
    number_of_steps: int = Field(default=5, ge=1, le=50, description="Number of steps in each direction from focus")


class ThroughFocusMTFFieldData(BaseModel):
    """Through Focus MTF data for a single field point."""
    field_index: int = Field(description="0-indexed field number")
    field_x: float = Field(description="Field X coordinate")
    field_y: float = Field(description="Field Y coordinate")
    tangential: list[float] = Field(default_factory=list, description="Tangential MTF values at each focus position")
    sagittal: list[float] = Field(default_factory=list, description="Sagittal MTF values at each focus position")


class BestFocusData(BaseModel):
    """Best focus position data."""
    position: float = Field(description="Best focus position (mm)")
    mtf_value: float = Field(description="MTF value at best focus")


class ThroughFocusMTFResponse(BaseModel):
    """Through Focus MTF analysis response."""
    success: bool = Field(description="Whether the operation succeeded")
    focus_positions: Optional[list[float]] = Field(default=None, description="Defocus positions (mm)")
    fields: Optional[list[ThroughFocusMTFFieldData]] = Field(default=None, description="Per-field MTF data")
    best_focus: Optional[BestFocusData] = Field(default=None, description="Best focus position and MTF value")
    frequency: Optional[float] = Field(default=None, description="Spatial frequency (cycles/mm)")
    wavelength_um: Optional[float] = Field(default=None, description="Wavelength in micrometers")
    delta_focus: Optional[float] = Field(default=None, description="Focus step size (mm)")
    error: Optional[str] = Field(default=None, description="Error message if operation failed")
