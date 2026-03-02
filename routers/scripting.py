"""Scripting router -- execute arbitrary Python scripts against OpticStudio."""

import io
import math
import sys
import time
import traceback as _tb_mod

import numpy as np
from fastapi import APIRouter, Depends

import main
from models import RunScriptRequest, RunScriptResponse
from utils.timing import timed_operation, timed_lock_acquire
from diagnostics.connection_diagnostics import (
    record_operation_start, record_operation_success, record_operation_error,
)

router = APIRouter()


def _make_result_serializable(obj: object) -> object:
    if isinstance(obj, np.bool_):
        return bool(obj)
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, (float, np.floating)):
        v = float(obj)
        return None if math.isnan(v) or math.isinf(v) else v
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): _make_result_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_result_serializable(v) for v in obj]
    return str(obj)


@router.post("/run-script", response_model=RunScriptResponse)
async def run_script(
    request: RunScriptRequest,
    _: None = Depends(main.verify_api_key),
) -> RunScriptResponse:
    """Execute a custom Python script with access to the loaded OpticStudio system."""
    with timed_operation(main.logger, "/run-script"):
        async with timed_lock_acquire(main._zospy_lock, main.logger, name="zospy"):
            if await main._ensure_connected() is None:
                return RunScriptResponse(
                    success=False,
                    error=main._not_connected_error(),
                )

            if not request.no_system:
                try:
                    main._load_system_from_request(request)
                except Exception as e:
                    return RunScriptResponse(
                        success=False,
                        error=f"Failed to load system: {e}",
                    )

            handler = main.zospy_handler

            namespace = {
                "oss": handler.oss,
                "zp": handler._zp,
                "np": np,
                "math": math,
                "handler": handler,
                "result": {},
                "system_modified": False,
            }

            old_stdout, old_stderr = sys.stdout, sys.stderr
            capture_out = io.StringIO()
            capture_err = io.StringIO()

            record_operation_start(endpoint="/run-script")
            op_t0 = time.monotonic()
            t0 = time.perf_counter()
            try:
                sys.stdout = capture_out
                sys.stderr = capture_err

                # Internal dev tool -- the script needs the live COM connection to
                # OpticStudio which cannot be serialized to a subprocess.
                compiled = compile(request.script, "<run-script>", "exec")
                exec(compiled, namespace)  # noqa: S102 -- intentional dynamic execution

                elapsed_ms = (time.perf_counter() - t0) * 1000

                # Restore before save so diagnostics appear in server logs
                sys.stdout = old_stdout
                sys.stderr = old_stderr

                raw_result = namespace.get("result", {})
                if not isinstance(raw_result, dict):
                    raw_result = {"value": raw_result}
                raw_result = _make_result_serializable(raw_result)

                modified_zmx = None
                if request.may_modify_system and namespace.get("system_modified"):
                    modified_zmx = handler._try_save_modified_system("run-script")

                dur_ms = (time.monotonic() - op_t0) * 1000
                record_operation_success(endpoint="/run-script", duration_ms=dur_ms)

                return RunScriptResponse(
                    success=True,
                    result=raw_result,
                    stdout=capture_out.getvalue(),
                    stderr=capture_err.getvalue(),
                    modified_zmx_content=modified_zmx,
                    execution_time_ms=round(elapsed_ms, 2),
                )

            except Exception as e:
                elapsed_ms = (time.perf_counter() - t0) * 1000
                dur_ms = (time.monotonic() - op_t0) * 1000
                record_operation_error(
                    endpoint="/run-script",
                    error=str(e),
                    error_type=type(e).__name__,
                    tb=_tb_mod.format_exc(),
                    duration_ms=dur_ms,
                )
                await main._handle_zospy_error("/run-script", e)
                return RunScriptResponse(
                    success=False,
                    error=f"{type(e).__name__}: {e}",
                    stdout=capture_out.getvalue(),
                    stderr=capture_err.getvalue() + _tb_mod.format_exc(),
                    execution_time_ms=round(elapsed_ms, 2),
                )

            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr
