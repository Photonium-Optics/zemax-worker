from typing import Any, Literal, Optional
from pydantic import BaseModel, Field


class MeritFunctionOperandRow(BaseModel):
    """A single merit function operand row."""
    operand_code: str = Field(description="Zemax operand code (e.g. EFFL, MTFA)")
    params: list[Optional[float]] = Field(default_factory=list, max_length=8, description="Up to 8 parameter values [Param1..Param8]")
    target: float = Field(default=0, description="Target value")
    weight: float = Field(default=1, description="Weight")
    comment: Optional[str] = Field(default=None, description="Comment text (for BLNK section header rows)")


class MeritFunctionRequest(BaseModel):
    """Request to evaluate a merit function."""
    zmx_content: str = Field(description="Base64-encoded .zmx file content")
    operand_rows: list[MeritFunctionOperandRow] = Field(max_length=200, description="Merit function operand rows")


class EvaluatedOperandRow(BaseModel):
    """Result for a single evaluated operand row."""
    row_index: int = Field(description="0-based index in the original request")
    operand_code: str = Field(description="Zemax operand code")
    value: Optional[float] = Field(default=None, description="Computed value")
    target: float = Field(description="Target value")
    weight: float = Field(description="Weight")
    contribution: Optional[float] = Field(default=None, description="Contribution to total merit")
    error: Optional[str] = Field(default=None, description="Per-row error message")
    comment: Optional[str] = Field(default=None, description="Comment text (for BLNK section header rows)")


class MeritFunctionResponse(BaseModel):
    """Response from merit function evaluation."""
    success: bool = Field(description="Whether the evaluation succeeded")
    total_merit: Optional[float] = Field(default=None, description="Total merit function value")
    evaluated_rows: Optional[list[EvaluatedOperandRow]] = Field(default=None, description="Per-row results")
    row_errors: Optional[list[dict[str, Any]]] = Field(default=None, description="Per-row errors for invalid/failed operands")
    error: Optional[str] = Field(default=None, description="Error message if failed")


class OptimizationWizardRequest(BaseModel):
    """Request to apply the SEQ Optimization Wizard (ISEQOptimizationWizard2)."""
    zmx_content: str = Field(description="Base64-encoded .zmx file content")
    criterion: Literal["Spot", "Wavefront", "Angular", "Contrast"] = Field(default="Spot", description="Optimization criterion (CriterionTypes)")
    reference: Literal["Centroid", "ChiefRay", "Unreferenced"] = Field(default="Centroid", description="Reference type (ReferenceTypes)")
    overall_weight: float = Field(default=1.0, ge=0, description="Overall weight for wizard operands")
    rings: int = Field(default=3, ge=1, le=20, description="Number of pupil rings")
    arms: Literal[6, 8, 10, 12] = Field(default=6, description="Number of pupil arms (PupilArmsCount)")
    # Pupil sampling mode
    use_gaussian_quadrature: bool = Field(default=False, description="Use Gaussian quadrature sampling")
    use_rectangular_array: bool = Field(default=False, description="Use rectangular NxN grid instead of ring/arm sampling")
    grid_size_nxn: int = Field(default=32, ge=4, le=204, description="Grid size NxN when use_rectangular_array is true (must be even, 4-204)")
    # Glass boundary values
    use_glass_boundary_values: bool = Field(default=True, description="Apply glass thickness constraints (MNEG/MXEG)")
    glass_min: float = Field(default=1.0, ge=0, description="Minimum glass center thickness (mm)")
    glass_max: float = Field(default=50.0, ge=0, description="Maximum glass center thickness (mm)")
    glass_edge_thickness: float = Field(default=0.0, ge=0, description="Minimum glass edge thickness (mm)")
    # Air boundary values
    use_air_boundary_values: bool = Field(default=True, description="Apply air spacing constraints (MNCA/MXCA)")
    air_min: float = Field(default=0.5, ge=0, description="Minimum air spacing (mm)")
    air_max: float = Field(default=1000.0, ge=0, description="Maximum air spacing (mm)")
    air_edge_thickness: float = Field(default=0.0, ge=0, description="Minimum air edge thickness (mm)")
    # Optimization Function
    type: Literal["RMS", "PTV"] = Field(default="RMS", description="Optimization type (OptimizationTypes)")
    spatial_frequency: float = Field(default=30.0, gt=0, description="Spatial frequency (cycles/mm) for Contrast criterion")
    xs_weight: float = Field(default=1.0, ge=0, description="Sagittal (X) weight for MTF")
    yt_weight: float = Field(default=1.0, ge=0, description="Tangential (Y) weight for MTF")
    use_maximum_distortion: bool = Field(default=False, description="Enable maximum distortion constraint")
    max_distortion_pct: float = Field(default=1.0, ge=0, description="Maximum distortion percentage")
    ignore_lateral_color: bool = Field(default=False, description="Ignore lateral color")
    # Pupil Integration
    obscuration: float = Field(default=0.0, ge=0, le=1, description="Pupil obscuration ratio 0-1 (for reflective systems)")
    # Optimization Goal
    optimization_goal: Literal["nominal", "manufacturing_yield"] = Field(default="nominal", description="Goal: nominal or manufacturing_yield")
    manufacturing_yield_weight: float = Field(default=1.0, ge=0, description="Manufacturing yield weight")
    # Bottom bar
    start_at: int = Field(default=1, ge=1, description="Starting surface index")
    use_all_configurations: bool = Field(default=True, description="Use all configurations")
    configuration_number: int = Field(default=1, ge=1, description="Specific configuration number")
    use_all_fields: bool = Field(default=True, description="Use all fields")
    field_number: int = Field(default=1, ge=1, description="Specific field number")
    assume_axial_symmetry: bool = Field(default=True, description="Assume axial symmetry")
    add_favorite_operands: bool = Field(default=False, description="Add favorite operands")
    delete_vignetted: bool = Field(default=True, description="Delete vignetted rays")


class WizardGeneratedRow(BaseModel):
    """A single merit function row generated by the optimization wizard."""
    row_index: int = Field(description="0-based row index")
    operand_code: str = Field(description="Zemax operand code (e.g. BLNK, DMFS, OPDX)")
    params: list[Optional[float]] = Field(default_factory=list, description="Up to 8 parameter values [Param1..Param8]")
    target: float = Field(default=0, description="Target value")
    weight: float = Field(default=0, description="Weight")
    value: Optional[float] = Field(default=None, description="Computed value after wizard apply")
    contribution: Optional[float] = Field(default=None, description="Contribution to total merit")
    comment: Optional[str] = Field(default=None, description="Comment text (for BLNK section header rows)")


class OptimizationWizardResponse(BaseModel):
    """Response from the optimization wizard."""
    success: bool = Field(description="Whether the wizard succeeded")
    total_merit: Optional[float] = Field(default=None, description="Total merit function value after wizard")
    generated_rows: Optional[list[WizardGeneratedRow]] = Field(default=None, description="All generated operand rows")
    num_rows_generated: int = Field(default=0, description="Number of rows generated")
    error: Optional[str] = Field(default=None, description="Error message if failed")


class OperandParameterInfo(BaseModel):
    """Metadata for a single operand parameter column."""
    column: str = Field(description="Column name: Comment, Param1-Param8")
    header: str = Field(description="Column header label from OpticStudio")
    data_type: str = Field(description="Cell data type as string")
    is_active: bool = Field(description="Whether this parameter column is active")
    is_read_only: bool = Field(description="Whether this parameter column is read-only")
    default_value: Optional[float | int | str] = Field(default=None, description="Default value after ChangeType()")


class OperandCatalogEntry(BaseModel):
    """Metadata for a single operand type."""
    code: str = Field(description="Operand code (e.g. 'EFFL')")
    type_name: str = Field(default="", description="Human-readable operand description")
    parameters: list[OperandParameterInfo] = Field(default_factory=list)


class OperandCatalogResponse(BaseModel):
    """Response from operand catalog discovery."""
    success: bool = Field(description="Whether the operation succeeded")
    operands: list[OperandCatalogEntry] = Field(default_factory=list)
    total_count: int = Field(default=0, description="Total number of operands discovered")
    error: Optional[str] = Field(default=None, description="Error message if failed")
