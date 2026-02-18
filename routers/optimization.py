"""Optimization router – run optimization, quick focus, scale lens."""

from fastapi import APIRouter, Depends

import main
from models import (
    RunOptimizationRequest, VariableState, RunOptimizationResponse,
    QuickFocusRequest, QuickFocusResponse,
    ScaleLensRequest, ScaleLensResponse,
)

router = APIRouter()


@router.post("/run-optimization", response_model=RunOptimizationResponse)
async def run_optimization(
    request: RunOptimizationRequest,
    _: None = Depends(main.verify_api_key),
) -> RunOptimizationResponse:
    """
    Run OpticStudio optimization using Local, Global, or Hammer method.

    Loads the system from zmx_content, populates the MFE with operand rows,
    runs the optimizer, and returns before/after merit values plus variable
    states from the LDE.
    """
    def _build_response(result: dict) -> RunOptimizationResponse:
        if not result.get("success"):
            return RunOptimizationResponse(
                success=False,
                error=result.get("error", "Optimization failed"),
            )

        raw_states = result.get("variable_states") or []
        variable_states = [VariableState(**vs) for vs in raw_states]

        return RunOptimizationResponse(
            success=True,
            method=result.get("method"),
            algorithm=result.get("algorithm"),
            merit_before=result.get("merit_before"),
            merit_after=result.get("merit_after"),
            cycles_completed=result.get("cycles_completed"),
            operand_results=result.get("operand_results"),
            variable_states=variable_states,
            best_solutions=result.get("best_solutions"),
            systems_evaluated=result.get("systems_evaluated"),
        )

    def _call_handler():
        operand_dicts = None
        if request.operand_rows:
            operand_dicts = [row.model_dump() for row in request.operand_rows]
        return main.zospy_handler.run_optimization(
            method=request.method,
            algorithm=request.algorithm,
            cycles=request.cycles,
            timeout_seconds=request.timeout_seconds,
            num_to_save=request.num_to_save,
            operand_rows=operand_dicts,
        )

    return await main._run_endpoint(
        "/run-optimization", RunOptimizationResponse, request,
        _call_handler,
        build_response=_build_response,
    )


@router.post("/quick-focus", response_model=QuickFocusResponse)
async def quick_focus(
    request: QuickFocusRequest,
    _: None = Depends(main.verify_api_key),
) -> QuickFocusResponse:
    """Run OpticStudio's native QuickFocus tool to find best focus position."""
    return await main._run_endpoint(
        "/quick-focus", QuickFocusResponse, request,
        lambda: main.zospy_handler.run_quick_focus(
            criterion=request.criterion,
            use_centroid=request.use_centroid,
        ),
    )


@router.post("/scale-lens", response_model=ScaleLensResponse)
async def scale_lens(
    request: ScaleLensRequest,
    _: None = Depends(main.verify_api_key),
) -> ScaleLensResponse:
    """Run OpticStudio's native Scale Lens tool to uniformly scale all dimensions."""
    return await main._run_endpoint(
        "/scale-lens", ScaleLensResponse, request,
        lambda: main.zospy_handler.run_scale_lens(
            mode=request.mode,
            scale_factor=request.scale_factor,
            target_unit=request.target_unit,
        ),
    )
