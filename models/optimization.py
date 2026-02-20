from typing import Any, Literal, Optional
from pydantic import BaseModel, Field

from models.base import SystemRequest
from models.merit import MeritFunctionOperandRow


class RunOptimizationRequest(BaseModel):
    """Request to run OpticStudio optimization."""
    zmx_content: str = Field(description="Base64-encoded .zmx file content")
    method: Literal["local", "global", "hammer"] = Field(default="local", description="Optimization method: local (gradient descent), global (full search), hammer (perturb+refine)")
    algorithm: Literal["DLS", "OrthogonalDescent", "DLSX", "PSD"] = Field(default="DLS", description="Optimization algorithm (Hammer is a method, not an algorithm)")
    cycles: Optional[int] = Field(default=5, ge=1, le=50, description="Cycles for local optimization (ignored for global/hammer)")
    timeout_seconds: Optional[float] = Field(default=60, ge=5, le=600, description="Time limit in seconds for global/hammer optimization")
    num_to_save: Optional[int] = Field(default=10, ge=1, le=50, description="Number of best solutions to retain (global only)")
    operand_rows: Optional[list[MeritFunctionOperandRow]] = Field(default=None, description="Explicit MFE operand rows to populate the merit function")


class VariableState(BaseModel):
    """State of a single variable parameter after optimization."""
    surface_index: int = Field(description="1-based surface index in OpticStudio LDE (surface 1 = first optical surface)")
    parameter: str = Field(description="Parameter name: radius, thickness, or conic")
    value: float = Field(description="Current value after optimization")
    is_variable: bool = Field(default=True, description="Whether the parameter is marked as variable")


class RunOptimizationResponse(BaseModel):
    """Response from optimization run."""
    success: bool = Field(description="Whether the optimization succeeded")
    method: Optional[str] = Field(default=None, description="Optimization method used: local, global, or hammer")
    algorithm: Optional[str] = Field(default=None, description="Optimization algorithm used")
    merit_before: Optional[float] = Field(default=None, description="Merit function value before optimization")
    merit_after: Optional[float] = Field(default=None, description="Merit function value after optimization")
    cycles_completed: Optional[int] = Field(default=None, description="Number of optimization cycles completed (local only)")
    operand_results: Optional[list[dict[str, Any]]] = Field(default=None, description="Per-operand results after optimization")
    variable_states: Optional[list[VariableState]] = Field(default=None, description="Variable parameter states after optimization")
    best_solutions: Optional[list[float]] = Field(default=None, description="Best merit function values from global optimization")
    systems_evaluated: Optional[int] = Field(default=None, description="Number of systems evaluated (global/hammer)")
    modified_zmx_content: Optional[str] = Field(default=None, description="Base64-encoded ZMX of modified system for full LDE readback")
    error: Optional[str] = Field(default=None, description="Error message if failed")


class ScaleLensRequest(SystemRequest):
    """Request for native Scale Lens tool."""
    mode: Literal["factor", "units"] = Field(description="Scaling mode: 'factor' to scale by a numeric factor, 'units' to convert between unit systems")
    scale_factor: Optional[float] = Field(default=None, ge=0.001, le=1000, description="Scale factor (required for mode='factor')")
    target_unit: Optional[Literal["mm", "cm", "inches", "meters"]] = Field(default=None, description="Target unit system (required for mode='units')")


class ScaleLensResponse(BaseModel):
    """Response from Scale Lens tool."""
    success: bool = Field(description="Whether scaling succeeded")
    mode: Optional[str] = Field(default=None, description="Scaling mode used")
    scale_factor: Optional[float] = Field(default=None, description="Effective scale factor applied")
    efl_before: Optional[float] = Field(default=None, description="EFL before scaling")
    efl_after: Optional[float] = Field(default=None, description="EFL after scaling")
    total_track_before: Optional[float] = Field(default=None, description="Total track before scaling")
    total_track_after: Optional[float] = Field(default=None, description="Total track after scaling")
    original_unit: Optional[str] = Field(default=None, description="Original unit system (for mode='units')")
    target_unit: Optional[str] = Field(default=None, description="Target unit system (for mode='units')")
    modified_zmx_content: Optional[str] = Field(default=None, description="Base64-encoded ZMX of modified system for full LDE readback")
    error: Optional[str] = Field(default=None, description="Error message if failed")


class QuickFocusRequest(SystemRequest):
    """Request for native QuickFocus tool."""
    criterion: str = Field(default="SpotSizeRadial", description="Focus criterion: SpotSizeRadial, RMSSpotSizeRadial, SpotSizeX, SpotSizeY")
    use_centroid: bool = Field(default=True, description="Use centroid reference (True) or chief ray (False)")


class QuickFocusResponse(BaseModel):
    """Response from QuickFocus tool."""
    success: bool = Field(description="Whether QuickFocus succeeded")
    surface_index: Optional[int] = Field(default=None, description="1-based LDE index of adjusted surface")
    thickness_before: Optional[float] = Field(default=None, description="Back focal distance before adjustment")
    thickness_after: Optional[float] = Field(default=None, description="Back focal distance after adjustment")
    delta_thickness: Optional[float] = Field(default=None, description="Change in thickness")
    criterion: Optional[str] = Field(default=None, description="Criterion used")
    modified_zmx_content: Optional[str] = Field(default=None, description="Base64-encoded ZMX of modified system for full LDE readback")
    error: Optional[str] = Field(default=None, description="Error message if failed")
