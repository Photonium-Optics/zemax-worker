from typing import Any, Literal, Optional
from pydantic import BaseModel, Field


class PolarizationPupilMapRequest(BaseModel):
    """Polarization Pupil Map analysis request."""
    zmx_content: str = Field(description="Base64-encoded .zmx file content")
    field_index: int = Field(default=1, ge=1, description="Field index (1-indexed)")
    wavelength_index: int = Field(default=1, ge=1, description="Wavelength index (1-indexed)")
    surface: str = Field(default="Image", description="Target surface ('Image' or integer)")
    sampling: str = Field(default="11x11", description="Pupil sampling grid")
    jx: float = Field(default=1.0, description="Jones vector X-component")
    jy: float = Field(default=0.0, description="Jones vector Y-component")
    x_phase: float = Field(default=0.0, description="X-component phase (degrees)")
    y_phase: float = Field(default=0.0, description="Y-component phase (degrees)")


class PolarizationPupilMapResponse(BaseModel):
    """Polarization Pupil Map analysis response."""
    success: bool = Field(description="Whether the operation succeeded")
    pupil_map: Optional[list[list[float]]] = Field(default=None, description="2D pupil map data rows")
    pupil_map_columns: Optional[list[str]] = Field(default=None, description="Column names for the pupil map")
    pupil_map_shape: Optional[list[int]] = Field(default=None, description="Shape of the pupil map [rows, cols]")
    transmission: Optional[float] = Field(default=None, description="Total transmission")
    x_field: Optional[float] = Field(default=None, description="Resulting X electric field")
    y_field: Optional[float] = Field(default=None, description="Resulting Y electric field")
    x_phase: Optional[float] = Field(default=None, description="Resulting X phase (degrees)")
    y_phase: Optional[float] = Field(default=None, description="Resulting Y phase (degrees)")
    field_x: Optional[float] = Field(default=None, description="Field X coordinate")
    field_y: Optional[float] = Field(default=None, description="Field Y coordinate")
    field_index: Optional[int] = Field(default=None, description="Field index used")
    wavelength_index: Optional[int] = Field(default=None, description="Wavelength index used")
    wavelength_um: Optional[float] = Field(default=None, description="Wavelength in micrometers")
    surface: Optional[str] = Field(default=None, description="Surface analyzed")
    sampling: Optional[str] = Field(default=None, description="Sampling grid used")
    jx: Optional[float] = Field(default=None, description="Input Jones vector X")
    jy: Optional[float] = Field(default=None, description="Input Jones vector Y")
    input_x_phase: Optional[float] = Field(default=None, description="Input X phase")
    input_y_phase: Optional[float] = Field(default=None, description="Input Y phase")
    error: Optional[str] = Field(default=None, description="Error message if operation failed")


class PolarizationTransmissionRequest(BaseModel):
    """Polarization Transmission analysis request."""
    zmx_content: str = Field(description="Base64-encoded .zmx file content")
    sampling: str = Field(default="32x32", description="Pupil sampling grid")
    unpolarized: bool = Field(default=False, description="Use unpolarized light")
    jx: float = Field(default=1.0, description="Jones vector X-component")
    jy: float = Field(default=0.0, description="Jones vector Y-component")
    x_phase: float = Field(default=0.0, description="X-component phase (degrees)")
    y_phase: float = Field(default=0.0, description="Y-component phase (degrees)")


class PolarizationTransmissionResponse(BaseModel):
    """Polarization Transmission analysis response."""
    success: bool = Field(description="Whether the operation succeeded")
    field_transmissions: Optional[list[dict]] = Field(default=None, description="Per-field transmission data")
    chief_ray_transmissions: Optional[list[dict]] = Field(default=None, description="Per-field chief ray transmission data")
    x_field: Optional[float] = Field(default=None, description="Resulting X electric field")
    y_field: Optional[float] = Field(default=None, description="Resulting Y electric field")
    x_phase: Optional[float] = Field(default=None, description="Resulting X phase")
    y_phase: Optional[float] = Field(default=None, description="Resulting Y phase")
    grid_size: Optional[str] = Field(default=None, description="Grid size used")
    num_fields: Optional[int] = Field(default=None, description="Number of fields in system")
    num_wavelengths: Optional[int] = Field(default=None, description="Number of wavelengths in system")
    field_info: Optional[list[dict]] = Field(default=None, description="Field position info")
    wavelength_info: Optional[list[dict]] = Field(default=None, description="Wavelength info")
    unpolarized: Optional[bool] = Field(default=None, description="Whether unpolarized mode was used")
    jx: Optional[float] = Field(default=None, description="Input Jones vector X")
    jy: Optional[float] = Field(default=None, description="Input Jones vector Y")
    input_x_phase: Optional[float] = Field(default=None, description="Input X phase")
    input_y_phase: Optional[float] = Field(default=None, description="Input Y phase")
    error: Optional[str] = Field(default=None, description="Error message if operation failed")
