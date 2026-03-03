from typing import Any, Optional
from pydantic import BaseModel, Field, field_validator


class GeometricImageRequest(BaseModel):
    """Geometric Image Analysis request."""
    zmx_content: str = Field(description="Base64-encoded .zmx file content")
    field_size: float = Field(default=0.0, ge=0, description="Image width in field coordinates (0 = auto)")
    image_size: float = Field(default=50.0, gt=0, description="Detector size in lens units")
    rays_x_1000: int = Field(default=10, ge=2, le=100, description="Approximate ray count in thousands")
    number_of_pixels: int = Field(default=100, ge=10, le=1000, description="Pixels across image width")
    field: int = Field(default=1, ge=1, description="Field number (1-indexed)")
    wavelength: str | int = Field(default="All", description="Wavelength: 'All' or wavelength number")

    @field_validator('wavelength', mode='before')
    @classmethod
    def coerce_wavelength(cls, v):
        if isinstance(v, int):
            return v
        if isinstance(v, str):
            try:
                return int(v)
            except ValueError:
                return v  # "All" or other string sentinel
        return v


class GeometricImageResponse(BaseModel):
    """Geometric Image Analysis response."""
    success: bool = Field(description="Whether the operation succeeded")
    image: Optional[str] = Field(default=None, description="Base64-encoded numpy array bytes")
    image_format: Optional[str] = Field(default=None, description="Image format: 'numpy_array'")
    array_shape: Optional[list[int]] = Field(default=None, description="Shape for numpy array reconstruction")
    array_dtype: Optional[str] = Field(default=None, description="Dtype for numpy array reconstruction")
    field_size: Optional[float] = Field(default=None, description="Field size used")
    image_size: Optional[float] = Field(default=None, description="Image size used")
    rays_x_1000: Optional[int] = Field(default=None, description="Rays x 1000 used")
    number_of_pixels: Optional[int] = Field(default=None, description="Number of pixels used")
    paraxial: Optional[dict[str, Any]] = Field(default=None, description="Paraxial properties")
    error: Optional[str] = Field(default=None, description="Error message if operation failed")
