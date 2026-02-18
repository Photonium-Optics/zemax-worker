from typing import Any, Literal, Optional
from pydantic import BaseModel, Field


class ParaxialResponse(BaseModel):
    """First-order (paraxial) optical properties response."""
    success: bool = Field(description="Whether the operation succeeded")
    efl: Optional[float] = Field(default=None, description="Effective focal length (mm)")
    bfl: Optional[float] = Field(default=None, description="Back focal length (mm)")
    fno: Optional[float] = Field(default=None, description="F-number")
    na: Optional[float] = Field(default=None, description="Numerical aperture")
    epd: Optional[float] = Field(default=None, description="Entrance pupil diameter (mm)")
    total_track: Optional[float] = Field(default=None, description="Total track length (mm)")
    max_field: Optional[float] = Field(default=None, description="Maximum field value")
    field_type: Optional[str] = Field(default=None, description="Field type (e.g. object_angle)")
    field_unit: Optional[str] = Field(default=None, description="Field unit (e.g. deg)")
    image_height: Optional[float] = Field(default=None, description="Paraxial image height (mm)")
    error: Optional[str] = Field(default=None, description="Error message if failed")


class SurfaceDataReportResponse(BaseModel):
    """Surface Data Report response."""
    success: bool = Field(description="Whether the operation succeeded")
    surfaces: Optional[list[dict[str, Any]]] = Field(default=None, description="Per-surface data (edge thickness, material, refractive index, power)")
    paraxial: Optional[dict[str, Any]] = Field(default=None, description="Paraxial properties")
    error: Optional[str] = Field(default=None, description="Error message if operation failed")


class CardinalPointEntry(BaseModel):
    """Single cardinal point entry."""
    name: str = Field(description="Cardinal point name (e.g. 'Focal Length (Object)')")
    value: Optional[float] = Field(default=None, description="Cardinal point value")
    units: str = Field(default="mm", description="Units")


class CardinalPointsResponse(BaseModel):
    """Cardinal points analysis response."""
    success: bool = Field(description="Whether the operation succeeded")
    cardinal_points: Optional[list[CardinalPointEntry]] = Field(default=None, description="List of cardinal point entries")
    starting_surface: Optional[int] = Field(default=None, description="Starting surface number")
    ending_surface: Optional[int] = Field(default=None, description="Ending surface number")
    wavelength: Optional[float] = Field(default=None, description="Analysis wavelength")
    orientation: Optional[str] = Field(default=None, description="Analysis orientation (Y-Z or X-Z)")
    lens_units: Optional[str] = Field(default=None, description="Lens units (e.g. mm)")
    error: Optional[str] = Field(default=None, description="Error message if failed")


class SurfaceCurvatureRequest(BaseModel):
    """Surface Curvature analysis request."""
    zmx_content: str = Field(description="Base64-encoded .zmx file content")
    surface: int = Field(default=1, ge=1, description="Surface number (1-indexed)")
    sampling: str = Field(default="65x65", description="Grid sampling resolution (e.g., '65x65', '129x129')")
    show_as: str = Field(default="Surface", description="Display format: Surface, Contour, GreyScale, etc.")
    data: str = Field(default="TangentialCurvature", description="Curvature data type: TangentialCurvature, SagittalCurvature, X_Curvature, Y_Curvature")
    remove: str = Field(default="None_", description="Removal option: None_, BaseROC, BestFitSphere")


class SurfaceCurvatureResponse(BaseModel):
    """Surface Curvature analysis response."""
    success: bool = Field(description="Whether the operation succeeded")
    image: Optional[str] = Field(default=None, description="Base64-encoded numpy array bytes")
    image_format: Optional[str] = Field(default=None, description="Image format: 'numpy_array'")
    array_shape: Optional[list[int]] = Field(default=None, description="Shape for numpy array reconstruction")
    array_dtype: Optional[str] = Field(default=None, description="Dtype for numpy array reconstruction")
    min_curvature: Optional[float] = Field(default=None, description="Minimum curvature value")
    max_curvature: Optional[float] = Field(default=None, description="Maximum curvature value")
    mean_curvature: Optional[float] = Field(default=None, description="Mean curvature value")
    surface_number: Optional[int] = Field(default=None, description="Surface number analyzed")
    data_type: Optional[str] = Field(default=None, description="Curvature data type used")
    error: Optional[str] = Field(default=None, description="Error message if operation failed")
