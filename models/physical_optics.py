from typing import Any, Literal, Optional
from pydantic import BaseModel, Field


class PhysicalOpticsPropagationRequest(BaseModel):
    """Physical Optics Propagation (POP) analysis request."""
    zmx_content: str = Field(description="Base64-encoded .zmx file content")
    field_index: int = Field(default=1, ge=1, description="Field index (1-indexed)")
    wavelength_index: int = Field(default=1, ge=1, description="Wavelength index (1-indexed)")
    beam_type: str = Field(default="GaussianWaist", description="Beam type (GaussianWaist, GaussianAngle, TopHat, etc.)")
    waist_x: Optional[float] = Field(default=None, description="Beam waist X (mm)")
    waist_y: Optional[float] = Field(default=None, description="Beam waist Y (mm)")
    x_sampling: int = Field(default=64, description="X sampling points (power of 2)")
    y_sampling: int = Field(default=64, description="Y sampling points (power of 2)")
    x_width: float = Field(default=4.0, description="X display width (mm)")
    y_width: float = Field(default=4.0, description="Y display width (mm)")
    start_surface: int = Field(default=1, description="Start surface (1-indexed)")
    end_surface: str = Field(default="Image", description="End surface ('Image' or surface index)")
    use_polarization: bool = Field(default=False, description="Use polarization")
    data_type: str = Field(default="Irradiance", description="Data type: Irradiance or Phase")


class PhysicalOpticsPropagationResponse(BaseModel):
    """Physical Optics Propagation (POP) analysis response."""
    success: bool = Field(description="Whether the operation succeeded")
    image: Optional[str] = Field(default=None, description="Base64-encoded numpy array bytes")
    image_format: Optional[str] = Field(default=None, description="Image format: 'numpy_array'")
    array_shape: Optional[list[int]] = Field(default=None, description="Shape for numpy array reconstruction")
    array_dtype: Optional[str] = Field(default=None, description="Dtype for numpy array reconstruction")
    beam_params: Optional[dict[str, Any]] = Field(default=None, description="Propagated beam parameters")
    wavelength_um: Optional[float] = Field(default=None, description="Wavelength in micrometers")
    field_x: Optional[float] = Field(default=None, description="Field X coordinate")
    field_y: Optional[float] = Field(default=None, description="Field Y coordinate")
    data_type: Optional[str] = Field(default=None, description="Data type used (Irradiance, Phase)")
    error: Optional[str] = Field(default=None, description="Error message if operation failed")
