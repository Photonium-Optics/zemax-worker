from typing import Any, Literal, Optional
from pydantic import BaseModel, Field


class WavefrontRequest(BaseModel):
    """Wavefront analysis request."""
    zmx_content: str = Field(description="Base64-encoded .zmx file content")
    field_index: int = Field(default=1, ge=1, description="Field index (1-indexed)")
    wavelength_index: int = Field(default=1, ge=1, description="Wavelength index (1-indexed)")
    sampling: str = Field(default="64x64", description="Pupil sampling grid (e.g., '32x32', '64x64', '128x128')")
    remove_tilt: bool = Field(default=False, description="Remove tilt from wavefront map")


class WavefrontResponse(BaseModel):
    """
    Wavefront analysis response.

    Returns raw wavefront data including:
    - RMS and P-V wavefront error in waves
    - Strehl ratio (if available)
    - Wavefront map as numpy array (Mac side renders to PNG)
    """
    success: bool = Field(description="Whether the operation succeeded")
    rms_waves: Optional[float] = Field(default=None, description="RMS wavefront error in waves")
    pv_waves: Optional[float] = Field(default=None, description="Peak-to-valley wavefront error in waves")
    strehl_ratio: Optional[float] = Field(default=None, description="Strehl ratio (0-1)")
    wavelength_um: Optional[float] = Field(default=None, description="Wavelength in micrometers")
    field_x: Optional[float] = Field(default=None, description="Field X coordinate")
    field_y: Optional[float] = Field(default=None, description="Field Y coordinate")
    image: Optional[str] = Field(default=None, description="Base64-encoded numpy array bytes")
    image_format: Optional[str] = Field(default=None, description="Image format: 'numpy_array'")
    array_shape: Optional[list[int]] = Field(default=None, description="Shape for numpy array reconstruction")
    array_dtype: Optional[str] = Field(default=None, description="Dtype for numpy array reconstruction")
    error: Optional[str] = Field(default=None, description="Error message if operation failed")


class NativeSeidelResponse(BaseModel):
    """
    Native Seidel raw text response from OpticStudio SeidelCoefficients analysis.

    Returns the raw UTF-16 text output from OpticStudio's GetTextFile().
    Parsing happens on the Mac side (seidel_text_parser.py).
    """
    success: bool = Field(description="Whether the operation succeeded")
    seidel_text: Optional[str] = Field(default=None, description="Raw text from OpticStudio GetTextFile()")
    num_surfaces: Optional[int] = Field(default=None, description="Number of optical surfaces")
    error: Optional[str] = Field(default=None, description="Error message if operation failed")


class ZernikeCoefficientsRequest(BaseModel):
    """Zernike Standard Coefficients analysis request."""
    zmx_content: str = Field(description="Base64-encoded .zmx file content")
    field_index: int = Field(default=1, ge=1, description="Field index (1-indexed)")
    wavelength_index: int = Field(default=1, ge=1, description="Wavelength index (1-indexed)")
    sampling: str = Field(default="64x64", description="Pupil sampling grid (e.g., '64x64', '128x128')")
    maximum_term: int = Field(default=37, ge=1, le=231, description="Maximum Zernike term number")


class ZernikeCoefficientsDetailResponse(BaseModel):
    """Zernike Standard Coefficients response."""
    success: bool = Field(description="Whether the operation succeeded")
    coefficients: Optional[list[dict[str, Any]]] = Field(default=None, description="List of {term, value, formula} dicts")
    pv_to_chief: Optional[float] = Field(default=None, description="P-V wavefront error to chief ray (waves)")
    pv_to_centroid: Optional[float] = Field(default=None, description="P-V wavefront error to centroid (waves)")
    rms_to_chief: Optional[float] = Field(default=None, description="RMS wavefront error to chief ray (waves)")
    rms_to_centroid: Optional[float] = Field(default=None, description="RMS wavefront error to centroid (waves)")
    strehl_ratio: Optional[float] = Field(default=None, description="Strehl ratio (0-1)")
    surface: Optional[str] = Field(default=None, description="Analysis surface")
    field_x: Optional[float] = Field(default=None, description="Field X coordinate")
    field_y: Optional[float] = Field(default=None, description="Field Y coordinate")
    field_index: Optional[int] = Field(default=None, description="Field index used")
    wavelength_index: Optional[int] = Field(default=None, description="Wavelength index used")
    wavelength_um: Optional[float] = Field(default=None, description="Wavelength in micrometers")
    maximum_term: Optional[int] = Field(default=None, description="Maximum Zernike term computed")
    error: Optional[str] = Field(default=None, description="Error message if operation failed")


class ZernikeVsFieldRequest(BaseModel):
    """Zernike Coefficients vs Field analysis request."""
    zmx_content: str = Field(description="Base64-encoded .zmx file content")
    maximum_term: int = Field(default=37, ge=1, le=231, description="Maximum Zernike term number")
    wavelength_index: int = Field(default=1, ge=1, description="Wavelength index (1-indexed)")
    sampling: str = Field(default="64x64", description="Pupil sampling grid (e.g., '64x64', '128x128')")
    field_density: int = Field(default=20, ge=5, le=100, description="Number of field sample points")


class ZernikeVsFieldResponse(BaseModel):
    """Zernike Coefficients vs Field response."""
    success: bool = Field(description="Whether the operation succeeded")
    field_positions: Optional[list[float]] = Field(default=None, description="Field position values")
    coefficients: Optional[dict[str, list[float]]] = Field(default=None, description="Dict mapping term number (str) to list of coefficient values per field")
    wavelength_um: Optional[float] = Field(default=None, description="Wavelength in micrometers")
    field_unit: Optional[str] = Field(default=None, description="Field coordinate unit (e.g. deg, mm)")
    error: Optional[str] = Field(default=None, description="Error message if operation failed")
