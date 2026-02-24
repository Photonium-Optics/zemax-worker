from typing import Any, Literal, Optional
from pydantic import BaseModel, Field


class RmsVsFieldRequest(BaseModel):
    """RMS vs Field analysis request."""
    zmx_content: str = Field(description="Base64-encoded .zmx file content")
    ray_density: int = Field(default=5, ge=1, le=20, description="Ray density (RayDens_1 to RayDens_20)")
    num_field_points: int = Field(default=20, ge=5, le=100, description="Number of field points (FieldDens_5 to FieldDens_100, step 5)")
    reference: str = Field(default="centroid", description="Reference point: 'centroid' or 'chief_ray'")
    wavelength_index: Optional[int] = Field(
        default=None,
        ge=1,
        description="Wavelength index (1-indexed). None = use OpticStudio primary wavelength.",
    )


class RmsVsFieldDataPoint(BaseModel):
    """Single data point in RMS vs Field result."""
    field_value: float = Field(description="Field coordinate value")
    rms_radius_um: float = Field(description="RMS spot radius in micrometers")


class RmsVsFieldResponse(BaseModel):
    """RMS vs Field analysis response."""
    success: bool = Field(description="Whether the operation succeeded")
    data: Optional[list[RmsVsFieldDataPoint]] = Field(default=None, description="RMS vs field data points")
    diffraction_limit: Optional[list[RmsVsFieldDataPoint]] = Field(default=None, description="Diffraction limit curve")
    wavelength_um: Optional[float] = Field(default=None, description="Wavelength in micrometers")
    field_unit: Optional[str] = Field(default=None, description="Field coordinate unit (e.g. deg, mm)")
    error: Optional[str] = Field(default=None, description="Error message if operation failed")
