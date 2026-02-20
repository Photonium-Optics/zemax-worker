"""Models for the unified ray analysis endpoint."""

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class RawAnalysisRay(BaseModel):
    """Raw ray trace result with both position and diagnostic data."""
    field_index: int = Field(description="0-indexed field number")
    field_x: float = Field(description="Field X coordinate")
    field_y: float = Field(description="Field Y coordinate")
    wavelength_index: int = Field(description="0-indexed wavelength number")
    wavelength_um: float = Field(description="Wavelength in micrometers")
    px: float = Field(description="Normalized pupil X coordinate (-1 to 1)")
    py: float = Field(description="Normalized pupil Y coordinate (-1 to 1)")
    reached_image: bool = Field(description="Whether the ray reached the image surface")
    failed_surface: Optional[int] = Field(default=None, description="Surface index where ray failed")
    failure_mode: Optional[str] = Field(default=None, description="Failure mode: MISS, TIR, VIGNETTE, etc.")
    x_um: Optional[float] = Field(default=None, description="X position at image plane (µm)")
    y_um: Optional[float] = Field(default=None, description="Y position at image plane (µm)")


class WavelengthInfo(BaseModel):
    """Wavelength metadata."""
    index: int = Field(description="0-indexed wavelength number")
    wavelength_um: float = Field(description="Wavelength in micrometers")


class RayAnalysisRequest(BaseModel):
    """Unified ray analysis request."""
    zmx_content: str = Field(description="Base64-encoded .zmx file content")
    num_rays: int = Field(default=50, description="Number of rays per field/wavelength")
    distribution: str = Field(default="hexapolar", description="Ray distribution: hexapolar, grid, or random")
    field_index: Optional[int] = Field(default=None, description="0-based field index. None = all fields.")
    wavelength_index: Optional[int] = Field(default=None, description="0-based wavelength index. None = all wavelengths.")


class RayAnalysisResponse(BaseModel):
    """Unified ray analysis response with both spot positions and diagnostic data."""
    success: bool = Field(description="Whether the operation succeeded")
    paraxial: Optional[dict[str, Any]] = Field(default=None, description="Basic paraxial data")
    num_surfaces: Optional[int] = Field(default=None, description="Number of surfaces")
    num_fields: Optional[int] = Field(default=None, description="Number of fields")
    num_wavelengths: Optional[int] = Field(default=None, description="Number of wavelengths")
    wavelength_info: Optional[list[WavelengthInfo]] = Field(default=None, description="Wavelength metadata")
    raw_rays: Optional[list[RawAnalysisRay]] = Field(default=None, description="Per-ray results with position and diagnostic data")
    surface_semi_diameters: Optional[list[float]] = Field(default=None, description="Semi-diameters from LDE")
    error: Optional[str] = Field(default=None, description="Error message if operation failed")
