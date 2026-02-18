from typing import Any, Literal, Optional
from pydantic import BaseModel, Field


class SystemRequest(BaseModel):
    """
    Request containing an optical system.

    Requires zmx_content: Base64-encoded .zmx file from zemax-converter.
    """
    zmx_content: str = Field(description="Base64-encoded .zmx file content")


class HealthResponse(BaseModel):
    """Health check response."""
    success: bool = Field(description="Whether the worker is healthy")
    opticstudio_connected: bool = Field(description="Whether OpticStudio is connected")
    version: Optional[str] = Field(default=None, description="OpticStudio version")
    zospy_version: Optional[str] = Field(default=None, description="ZosPy version")
    worker_count: int = Field(description="Number of uvicorn worker processes serving this URL")
    connection_error: Optional[str] = Field(default=None, description="Error detail when opticstudio_connected is False")
