"""Merit function router – evaluate, optimization wizard, operand catalog."""

import logging

from fastapi import APIRouter, Depends

import main
from models import (
    MeritFunctionRequest, EvaluatedOperandRow, MeritFunctionResponse,
    OptimizationWizardRequest, WizardGeneratedRow, OptimizationWizardResponse,
    OperandCatalogResponse,
)
from utils.timing import timed_operation, timed_lock_acquire

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/evaluate-merit-function", response_model=MeritFunctionResponse)
async def evaluate_merit_function(
    request: MeritFunctionRequest,
    _: None = Depends(main.verify_api_key),
) -> MeritFunctionResponse:
    """
    Evaluate a merit function: construct operands in the MFE and compute.

    Loads the system from zmx_content, populates the Merit Function Editor
    with the provided operand rows, and returns computed values and contributions.
    """
    def _build_merit_response(result: dict) -> MeritFunctionResponse:
        raw_rows = result.get("evaluated_rows", [])
        evaluated = [EvaluatedOperandRow(**r) for r in raw_rows] if raw_rows else None
        row_errors = result.get("row_errors", []) if result.get("row_errors") else None
        return MeritFunctionResponse(
            success=result.get("success", False),
            error=result.get("error") if not result.get("success", False) else None,
            total_merit=result.get("total_merit"),
            evaluated_rows=evaluated,
            row_errors=row_errors,
        )

    def _call_handler():
        operand_dicts = [row.model_dump() for row in request.operand_rows]
        return main.zospy_handler.evaluate_merit_function(operand_dicts)

    return await main._run_endpoint(
        "/evaluate-merit-function", MeritFunctionResponse, request,
        _call_handler,
        build_response=_build_merit_response,
    )


@router.post("/apply-optimization-wizard", response_model=OptimizationWizardResponse)
async def apply_optimization_wizard(
    request: OptimizationWizardRequest,
    _: None = Depends(main.verify_api_key),
) -> OptimizationWizardResponse:
    """
    Apply the SEQ Optimization Wizard to auto-generate merit function operands.

    Uses OpticStudio's SEQOptimizationWizard2 to populate the MFE based on
    image quality criteria, accounting for all active fields, wavelengths,
    and pupil sampling.
    """
    def _build_wizard_response(result: dict) -> OptimizationWizardResponse:
        raw_rows = result.get("generated_rows", [])
        generated = [WizardGeneratedRow(**r) for r in raw_rows] if raw_rows else None
        return OptimizationWizardResponse(
            success=result.get("success", False),
            error=result.get("error") if not result.get("success", False) else None,
            total_merit=result.get("total_merit"),
            generated_rows=generated,
            num_rows_generated=result.get("num_rows_generated", 0),
        )

    # Pass all wizard params except zmx_content (already loaded by _run_endpoint)
    wizard_params = request.model_dump(exclude={"zmx_content"})

    return await main._run_endpoint(
        "/apply-optimization-wizard", OptimizationWizardResponse, request,
        lambda: main.zospy_handler.apply_optimization_wizard(**wizard_params),
        build_response=_build_wizard_response,
    )


@router.post("/operand-catalog", response_model=OperandCatalogResponse)
async def get_operand_catalog(
    _: None = Depends(main.verify_api_key),
) -> OperandCatalogResponse:
    """
    Discover all supported merit function operand types and their parameter metadata.

    No ZMX file needed -- only requires an OpticStudio connection.
    """
    with timed_operation(logger, "/operand-catalog"):
        async with timed_lock_acquire(main._zospy_lock, logger, name="zospy"):
            if await main._ensure_connected() is None:
                return OperandCatalogResponse(success=False, error=main._not_connected_error())

            try:
                result = main.zospy_handler.get_operand_catalog()

                if not result.get("success"):
                    return OperandCatalogResponse(
                        success=False,
                        error=result.get("error", "/operand-catalog failed"),
                    )

                # Filter to known model fields, matching _run_endpoint pattern
                model_fields = set(OperandCatalogResponse.model_fields.keys())
                return OperandCatalogResponse(success=True, **{
                    k: v for k, v in result.items()
                    if k not in ("success", "error") and k in model_fields
                })
            except Exception as e:
                await main._handle_zospy_error("/operand-catalog", e)
                return OperandCatalogResponse(success=False, error=str(e))
