"""Pydantic models for custom script execution."""

from typing import Any, Optional

from pydantic import BaseModel, Field

from models.base import SystemRequest


class RunScriptRequest(SystemRequest):
    """Request to execute a custom Python script against OpticStudio."""

    zmx_content: str = Field(default="", description="Base64-encoded .zmx file (empty when no_system=True)")
    script: str = Field(description="Python source code to execute")
    no_system: bool = Field(
        default=False,
        description="Skip system loading (zmx_content can be empty)",
    )
    may_modify_system: bool = Field(
        default=False,
        description="If True and script sets system_modified=True, save modified ZMX",
    )
    working_directory: str = Field(
        default="",
        description="Path to a directory of ZMX files on the worker machine",
    )


class RunScriptResponse(BaseModel):
    """Response from custom script execution."""

    success: bool = Field(description="Whether the script executed without errors")
    result: Optional[dict[str, Any]] = Field(default=None, description="Script result dict")
    stdout: str = Field(default="", description="Captured stdout from the script")
    stderr: str = Field(default="", description="Captured stderr from the script")
    modified_zmx_content: Optional[str] = Field(
        default=None,
        description="Base64-encoded modified ZMX if system was modified",
    )
    execution_time_ms: Optional[float] = Field(
        default=None,
        description="Script execution time in milliseconds",
    )
    error: Optional[str] = Field(default=None, description="Error message if success=False")
