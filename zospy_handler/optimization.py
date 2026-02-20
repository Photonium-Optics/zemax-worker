"""Optimization mixin – merit function, wizard, optimization, quick focus."""

import logging
import math
import time
from typing import Any

from zospy_handler._base import _extract_value, _log_raw_output, _read_comment_cell

logger = logging.getLogger(__name__)


class OptimizationMixin:

    def evaluate_merit_function(self, operand_rows: list[dict]) -> dict[str, Any]:
        """
        Evaluate a merit function by constructing operands in the MFE and computing.

        Args:
            operand_rows: List of dicts with keys:
                - operand_code: str (e.g. "EFFL")
                - params: list of up to 8 values (None for unused slots)
                - target: float
                - weight: float

        Returns:
            Dict with:
                - success: bool
                - total_merit: float or None
                - evaluated_rows: list of per-row results
                - row_errors: list of per-row error messages
        """
        zp = self._zp
        mfe = self.oss.MFE

        # Clear existing MFE
        mfe.DeleteAllRows()

        evaluated_rows = []
        row_errors = []
        valid_operand_indices = []  # (mfe_row_number, original_row_index)

        # MFE column constants for parameter cells
        try:
            mfe_cols = zp.constants.Editors.MFE.MeritColumn
        except AttributeError:
            return {
                "success": False,
                "error": "Cannot access MFE column constants",
                "total_merit": None,
                "evaluated_rows": [],
                "row_errors": [{"row_index": 0, "error": "Cannot access MFE column constants"}],
            }

        mfe_row_number = 0  # Tracks actual MFE row position (1-based)

        for row_index, row in enumerate(operand_rows):
            code = row.get("operand_code", "")
            params = list(row.get("params", []))
            target = float(row.get("target", 0))
            weight = float(row.get("weight", 1))

            # Resolve operand type enum
            try:
                op_type = getattr(zp.constants.Editors.MFE.MeritOperandType, code)
            except AttributeError:
                row_errors.append({
                    "row_index": row_index,
                    "error": f"Unknown operand code: {code}",
                })
                evaluated_rows.append({
                    "row_index": row_index,
                    "operand_code": code,
                    "value": None,
                    "target": target,
                    "weight": weight,
                    "contribution": None,
                    "error": f"Unknown operand code: {code}",
                })
                continue

            mfe_row_number += 1

            # After DeleteAllRows, MFE retains 1 empty row.
            # First operand uses GetOperandAt(1), subsequent use InsertNewOperandAt.
            if mfe_row_number == 1:
                op = mfe.GetOperandAt(1)
            else:
                op = mfe.InsertNewOperandAt(mfe_row_number)

            try:
                op.ChangeType(op_type)
                op.Target = float(target)
                op.Weight = float(weight)

                # Set Comment cell for BLNK section headers
                comment = row.get("comment")
                if comment and code == "BLNK":
                    try:
                        comment_cell = op.GetOperandCell(mfe_cols.Comment)
                        comment_cell.Value = str(comment)
                    except Exception:
                        pass

                # Set parameter cells — use cell DataType to pick the right setter
                param_columns = [
                    mfe_cols.Param1, mfe_cols.Param2,
                    mfe_cols.Param3, mfe_cols.Param4,
                    mfe_cols.Param5, mfe_cols.Param6,
                    mfe_cols.Param7, mfe_cols.Param8,
                ]
                for i, col in enumerate(param_columns):
                    if i < len(params) and params[i] is not None:
                        cell = op.GetOperandCell(col)
                        dt = str(cell.DataType).split('.')[-1] if hasattr(cell, 'DataType') else ''
                        if dt == 'Integer':
                            cell.IntegerValue = int(float(params[i]))
                        elif dt == 'String':
                            cell.Value = str(params[i])
                        else:
                            cell.DoubleValue = float(params[i])

                valid_operand_indices.append((mfe_row_number, row_index))

            except Exception as e:
                logger.warning(f"MFE row {mfe_row_number} ({code}): error setting params: {e}")
                row_errors.append({
                    "row_index": row_index,
                    "error": f"Error configuring {code}: {e}",
                })
                evaluated_rows.append({
                    "row_index": row_index,
                    "operand_code": code,
                    "value": None,
                    "target": target,
                    "weight": weight,
                    "contribution": None,
                    "error": f"Error configuring {code}: {e}",
                    "comment": row.get("comment"),
                })
                continue

        if not valid_operand_indices:
            error_summary = f"All {len(operand_rows)} operand row(s) failed validation"
            return {
                "success": False,
                "error": error_summary,
                "total_merit": None,
                "evaluated_rows": evaluated_rows,
                "row_errors": row_errors,
            }

        # Calculate merit function
        try:
            # Use _extract_value() to handle ZosPy 2.x UnitField objects
            total_merit = _extract_value(mfe.CalculateMeritFunction())
        except Exception as e:
            logger.error(f"MFE CalculateMeritFunction failed: {e}")
            return {
                "success": False,
                "error": f"CalculateMeritFunction failed: {e}",
                "total_merit": None,
                "evaluated_rows": evaluated_rows,
                "row_errors": row_errors + [{"row_index": -1, "error": f"CalculateMeritFunction failed: {e}"}],
            }

        # Read back results for each valid row
        for mfe_row_num, orig_index in valid_operand_indices:
            row = operand_rows[orig_index]
            try:
                op = mfe.GetOperandAt(mfe_row_num)
                value = _extract_value(op.Value, None)
                contribution = _extract_value(op.Contribution, None)

                evaluated_rows.append({
                    "row_index": orig_index,
                    "operand_code": row.get("operand_code", ""),
                    "value": value,
                    "target": float(row.get("target", 0)),
                    "weight": float(row.get("weight", 1)),
                    "contribution": contribution,
                    "error": None,
                    "comment": row.get("comment"),
                })
            except Exception as e:
                logger.warning(f"Error reading MFE row {mfe_row_num}: {e}")
                evaluated_rows.append({
                    "row_index": orig_index,
                    "operand_code": row.get("operand_code", ""),
                    "value": None,
                    "target": float(row.get("target", 0)),
                    "weight": float(row.get("weight", 1)),
                    "contribution": None,
                    "error": f"Error reading result: {e}",
                })

        # Sort evaluated_rows by row_index for consistent output
        evaluated_rows.sort(key=lambda r: r["row_index"])

        result = {
            "success": True,
            "total_merit": total_merit,
            "evaluated_rows": evaluated_rows,
            "row_errors": row_errors,
        }
        _log_raw_output("/evaluate-merit-function", result)
        return result

    def apply_optimization_wizard(
        self,
        criterion: str = "Spot",
        reference: str = "Centroid",
        overall_weight: float = 1.0,
        rings: int = 3,
        arms: int = 6,
        use_gaussian_quadrature: bool = False,
        use_rectangular_array: bool = False,
        grid_size_nxn: int = 32,
        use_glass_boundary_values: bool = True,
        glass_min: float = 1.0,
        glass_max: float = 50.0,
        glass_edge_thickness: float = 0.0,
        use_air_boundary_values: bool = True,
        air_min: float = 0.5,
        air_max: float = 1000.0,
        air_edge_thickness: float = 0.0,
        type: str = "RMS",
        spatial_frequency: float = 30.0,
        xs_weight: float = 1.0,
        yt_weight: float = 1.0,
        use_maximum_distortion: bool = False,
        max_distortion_pct: float = 1.0,
        ignore_lateral_color: bool = False,
        obscuration: float = 0.0,
        optimization_goal: str = "nominal",
        manufacturing_yield_weight: float = 1.0,
        start_at: int = 1,
        use_all_configurations: bool = True,
        configuration_number: int = 1,
        use_all_fields: bool = True,
        field_number: int = 1,
        assume_axial_symmetry: bool = True,
        add_favorite_operands: bool = False,
        delete_vignetted: bool = True,
    ) -> dict[str, Any]:
        """
        Apply the SEQ Optimization Wizard to auto-generate merit function operands.

        Uses OpticStudio's ISEQOptimizationWizard2 to populate the MFE based on
        image quality criteria (Spot, Wavefront, Contrast, or Angular).

        Returns:
            Dict with success, total_merit, generated_rows, num_rows_generated
        """
        def _wizard_error(error: str, total_merit=None) -> dict[str, Any]:
            return {
                "success": False, "error": error,
                "total_merit": total_merit, "generated_rows": [],
                "num_rows_generated": 0,
            }

        zp = self._zp
        mfe = self.oss.MFE

        # Cross-field validations (single-field constraints enforced by Pydantic)
        if use_glass_boundary_values and glass_min >= glass_max:
            return _wizard_error(f"glass_min ({glass_min}) must be < glass_max ({glass_max})")
        if use_air_boundary_values and air_min >= air_max:
            return _wizard_error(f"air_min ({air_min}) must be < air_max ({air_max})")

        # Check wizard availability (requires OpticStudio 18.5+)
        wizard = getattr(mfe, 'SEQOptimizationWizard2', None)
        if wizard is None:
            return _wizard_error("SEQOptimizationWizard2 not available (requires OpticStudio 18.5+)")

        def _set_wizard_prop(prop_name: str, value: Any) -> None:
            """Set a wizard property, logging a warning on failure."""
            try:
                setattr(wizard, prop_name, value)
            except Exception as e:
                logger.warning(f"Failed to set {prop_name}: {e}")

        try:
            # Resolve wizard enums namespace (ZOSAPI.Wizards)
            # ZosPy maps this to zp.constants.Wizards
            wizard_enums = getattr(zp.constants, 'Wizards', None)
            if wizard_enums is None:
                return _wizard_error(
                    "Wizard enums not found at zp.constants.Wizards. "
                    "Check ZosPy version compatibility."
                )

            # Criterion — ZOSAPI.Wizards.CriterionTypes: Wavefront=0, Contrast=1, Spot=2, Angular=3
            if not hasattr(wizard_enums.CriterionTypes, criterion):
                return _wizard_error(f"Unknown criterion type: {criterion}")
            wizard.Criterion = getattr(wizard_enums.CriterionTypes, criterion)

            # Reference — ZOSAPI.Wizards.ReferenceTypes: Centroid=0, ChiefRay=1, Unreferenced=2
            if not hasattr(wizard_enums.ReferenceTypes, reference):
                return _wizard_error(f"Unknown reference type: {reference}")
            wizard.Reference = getattr(wizard_enums.ReferenceTypes, reference)

            wizard.OverallWeight = float(overall_weight)
            wizard.Rings = int(rings)

            # Arms — ZOSAPI.Wizards.PupilArmsCount: Arms_6=0, Arms_8=1, Arms_10=2, Arms_12=3
            arms_enum_name = f"Arms_{arms}"
            if hasattr(wizard_enums.PupilArmsCount, arms_enum_name):
                wizard.Arms = getattr(wizard_enums.PupilArmsCount, arms_enum_name)
            else:
                logger.warning(f"Unknown arms count {arms}, using Arms_6")
                wizard.Arms = getattr(wizard_enums.PupilArmsCount, "Arms_6")

            # Pupil sampling mode
            wizard.UseGaussianQuadrature = bool(use_gaussian_quadrature)
            _set_wizard_prop("UseRectangularArray", bool(use_rectangular_array))
            if use_rectangular_array:
                _set_wizard_prop("GridSizeNxN", int(grid_size_nxn))

            # Glass boundary values
            wizard.UseGlassBoundaryValues = bool(use_glass_boundary_values)
            if use_glass_boundary_values:
                wizard.GlassMin = float(glass_min)
                wizard.GlassMax = float(glass_max)
                _set_wizard_prop("GlassEdgeThickness", float(glass_edge_thickness))

            # Air boundary values
            wizard.UseAirBoundaryValues = bool(use_air_boundary_values)
            if use_air_boundary_values:
                wizard.AirMin = float(air_min)
                wizard.AirMax = float(air_max)
                wizard.AirEdgeThickness = float(air_edge_thickness)

            # Type — ZOSAPI.Wizards.OptimizationTypes: RMS=0, PTV=1
            if hasattr(wizard_enums, "OptimizationTypes"):
                opt_type = getattr(wizard_enums.OptimizationTypes, type, None)
                if opt_type is not None:
                    wizard.Type = opt_type
                else:
                    logger.warning(f"OptimizationTypes.{type} not found, skipping Type assignment")
            else:
                logger.warning("OptimizationTypes enum not found in Wizards namespace")

            # Optimization Function params
            _set_wizard_prop("SpatialFrequency", float(spatial_frequency))
            _set_wizard_prop("XSWeight", float(xs_weight))
            _set_wizard_prop("YTWeight", float(yt_weight))
            _set_wizard_prop("UseMaximumDistortion", bool(use_maximum_distortion))
            if use_maximum_distortion:
                _set_wizard_prop("MaxDistortionPct", float(max_distortion_pct))
            _set_wizard_prop("IgnoreLateralColor", bool(ignore_lateral_color))

            # Pupil Integration
            _set_wizard_prop("Obscuration", float(obscuration))

            # Optimization Goal — check availability before enabling manufacturing yield
            if optimization_goal == "manufacturing_yield":
                is_available = getattr(wizard, "IsHighManufacturingYieldAvailable", None)
                if is_available is False:
                    logger.warning(
                        "Manufacturing yield optimization not available in this license; "
                        "falling back to nominal"
                    )
                    _set_wizard_prop("OptimizeForBestNominalPerformance", True)
                    _set_wizard_prop("OptimizeForManufacturingYield", False)
                else:
                    _set_wizard_prop("OptimizeForBestNominalPerformance", False)
                    _set_wizard_prop("OptimizeForManufacturingYield", True)
                    _set_wizard_prop("ManufacturingYieldWeight", float(manufacturing_yield_weight))
            else:
                _set_wizard_prop("OptimizeForBestNominalPerformance", True)
                _set_wizard_prop("OptimizeForManufacturingYield", False)

            # Bottom bar params
            _set_wizard_prop("StartAt", int(start_at))
            _set_wizard_prop("UseAllConfigurations", bool(use_all_configurations))
            if not use_all_configurations:
                _set_wizard_prop("ConfigurationNumber", int(configuration_number))
            _set_wizard_prop("UseAllFields", bool(use_all_fields))
            if not use_all_fields:
                _set_wizard_prop("FieldNumber", int(field_number))
            _set_wizard_prop("AssumeAxialSymmetry", bool(assume_axial_symmetry))
            _set_wizard_prop("AddFavoriteOperands", bool(add_favorite_operands))
            _set_wizard_prop("DeleteVignetted", bool(delete_vignetted))

            logger.info(
                f"Applying optimization wizard: criterion={criterion}, type={type}, "
                f"reference={reference}, rings={rings}, arms={arms}, "
                f"gaussian_quadrature={use_gaussian_quadrature}, "
                f"rectangular_array={use_rectangular_array}, goal={optimization_goal}"
            )
            wizard.Apply()

        except Exception as e:
            logger.error(f"Optimization wizard Apply() failed: {e}")
            return _wizard_error(f"Wizard Apply() failed: {e}")

        # Calculate merit function to get per-row values
        try:
            total_merit = _extract_value(mfe.CalculateMeritFunction())
        except Exception as e:
            logger.error(f"CalculateMeritFunction after wizard failed: {e}")
            return _wizard_error(f"Merit function calculation failed after wizard: {e}")

        # Read all generated rows from MFE
        try:
            mfe_cols = zp.constants.Editors.MFE.MeritColumn
            param_columns = [
                mfe_cols.Param1, mfe_cols.Param2,
                mfe_cols.Param3, mfe_cols.Param4,
                mfe_cols.Param5, mfe_cols.Param6,
                mfe_cols.Param7, mfe_cols.Param8,
            ]
            logger.info(f"Wizard generated {mfe.NumberOfOperands} operand rows")
            generated_rows = self._read_mfe_rows(mfe, mfe_cols, param_columns, "wizard")
        except Exception as e:
            logger.error(f"Error reading wizard-generated MFE rows: {e}")
            return _wizard_error(f"Failed to read wizard-generated rows: {e}", total_merit=total_merit)

        result = {
            "success": True,
            "total_merit": total_merit,
            "generated_rows": generated_rows,
            "num_rows_generated": len(generated_rows),
        }
        _log_raw_output("/apply-optimization-wizard", result)
        return result


    def get_operand_catalog(self) -> dict[str, Any]:
        """
        Discover all supported MeritOperandType values and their parameter metadata.

        Iterates every operand type in the MeritOperandType enum, sets it on a
        temporary MFE row, and reads back cell metadata (Header, DataType,
        IsActive, IsReadOnly) for Comment + Param1-Param8 columns.

        Returns:
            Dict with success, operands list, total_count
        """
        mfe = self.oss.MFE
        mfe_constants = self._zp.constants.Editors.MFE
        MeritOperandType = mfe_constants.MeritOperandType
        mfe_cols = mfe_constants.MeritColumn

        # Column definitions: (name, enum_value)
        columns = [
            ("Comment", mfe_cols.Comment),
            *((f"Param{i}", getattr(mfe_cols, f"Param{i}")) for i in range(1, 9)),
        ]

        # Ensure at least 1 row exists in the MFE
        try:
            if mfe.NumberOfOperands < 1:
                mfe.AddOperand()
            op = mfe.GetOperandAt(1)
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to initialize MFE for operand catalog: {e}",
                "operands": [],
                "total_count": 0,
            }

        # Enumerate all operand type names from the namedtuple.
        # Use _fields (not dir()) to avoid namedtuple methods like count/index.
        if hasattr(MeritOperandType, '_fields'):
            type_names = list(MeritOperandType._fields)
        else:
            type_names = [name for name in dir(MeritOperandType) if not name.startswith('_')]
        logger.info(f"Enumerating {len(type_names)} operand types for catalog")

        operands = []
        skipped = 0
        start_time = time.time()

        for code in type_names:
            try:
                enum_val = getattr(MeritOperandType, code)
                op.ChangeType(enum_val)

                # Read the human-readable type name from OpticStudio
                type_name = ""
                try:
                    type_name = str(op.TypeName) if hasattr(op, 'TypeName') else ""
                except Exception:
                    pass

                parameters = [
                    self._read_operand_cell(op, col_name, col_enum, code)
                    for col_name, col_enum in columns
                ]

                operands.append({
                    "code": code,
                    "type_name": type_name,
                    "parameters": parameters,
                })
            except Exception as e:
                logger.warning(f"Skipping operand {code}: {e}")
                skipped += 1
                # Re-acquire the row reference in case ChangeType corrupted it
                try:
                    op = mfe.GetOperandAt(1)
                except Exception:
                    pass

        elapsed = time.time() - start_time
        logger.info(
            f"Operand catalog complete: {len(operands)} operands in {elapsed:.1f}s "
            f"(skipped {skipped})"
        )

        return {
            "success": True,
            "operands": operands,
            "total_count": len(operands),
        }

    @staticmethod
    def _read_cell_default(cell: Any, data_type: str) -> float | int | str | None:
        """Extract the default value from a COM operand cell.

        Returns None if the value cannot be read or is inf/NaN.
        """
        try:
            if data_type == "Integer" and hasattr(cell, 'IntegerValue'):
                raw = cell.IntegerValue
                if raw is not None and not (isinstance(raw, float) and not math.isfinite(raw)):
                    return int(raw)
            elif data_type == "Double" and hasattr(cell, 'DoubleValue'):
                raw = cell.DoubleValue
                if raw is not None and math.isfinite(raw):
                    return float(raw)
            elif data_type == "String" and hasattr(cell, 'Value'):
                raw = cell.Value
                if raw is not None:
                    return str(raw)
        except Exception:
            pass
        return None

    def _read_operand_cell(
        self, op: Any, col_name: str, col_enum: Any, operand_code: str,
    ) -> dict[str, Any]:
        """Read metadata from a single MFE operand cell.

        Returns a dict with column, header, data_type, is_active, is_read_only,
        and default_value. Falls back to safe defaults if the cell cannot be read.
        """
        try:
            cell = op.GetOperandCell(col_enum)
            data_type = str(cell.DataType).split('.')[-1] if hasattr(cell, 'DataType') else "Unknown"

            default_value = self._read_cell_default(cell, data_type)

            return {
                "column": col_name,
                "header": str(cell.Header) if hasattr(cell, 'Header') else col_name,
                "data_type": data_type,
                "is_active": bool(cell.IsActive) if hasattr(cell, 'IsActive') else False,
                "is_read_only": bool(cell.IsReadOnly) if hasattr(cell, 'IsReadOnly') else False,
                "default_value": default_value,
            }
        except Exception as e:
            logger.debug(f"Failed to read {col_name} for {operand_code}: {e}")
            return {
                "column": col_name,
                "header": col_name,
                "data_type": "Unknown",
                "is_active": False,
                "is_read_only": True,
                "default_value": None,
            }


    # ── MFE row reading helper ─────────────────────────────────────────

    @staticmethod
    def _read_mfe_param_cells(op, param_columns) -> list:
        params = []
        for col in param_columns:
            try:
                cell = op.GetOperandCell(col)
                dt = str(cell.DataType).split('.')[-1] if hasattr(cell, 'DataType') else ''
                if dt == 'String':
                    params.append(None)
                else:
                    raw = float(cell.IntegerValue if dt == 'Integer' else cell.DoubleValue)
                    params.append(None if (math.isinf(raw) or math.isnan(raw)) else raw)
            except Exception:
                params.append(None)
        return params

    def _read_mfe_rows(self, mfe, mfe_cols, param_columns, label: str) -> list[dict[str, Any]]:
        rows = []
        num_operands = mfe.NumberOfOperands
        for i in range(1, num_operands + 1):
            try:
                op = mfe.GetOperandAt(i)
                try:
                    op_code = str(op.Type).split('.')[-1]
                except Exception:
                    op_code = f"UNK_{i}"

                rows.append({
                    "row_index": i - 1,
                    "operand_code": op_code,
                    "params": self._read_mfe_param_cells(op, param_columns),
                    "target": _extract_value(op.Target, 0.0),
                    "weight": _extract_value(op.Weight, 0.0),
                    "value": _extract_value(op.Value, None),
                    "contribution": _extract_value(op.Contribution, None),
                    "comment": _read_comment_cell(op, mfe_cols.Comment),
                })
            except Exception as e:
                logger.warning(f"Error reading {label} MFE row {i}: {e}")
                rows.append({
                    "row_index": i - 1,
                    "operand_code": f"ERR_{i}",
                    "params": [None] * 8,
                    "target": 0.0,
                    "weight": 0.0,
                    "value": None,
                    "contribution": None,
                    "comment": None,
                })
        return rows

    # ── Optimization enum helpers ──────────────────────────────────────

    @staticmethod
    def _resolve_enum(enum_obj, attr_name: str, fallback_name: str, label: str):
        """Resolve an enum member by name, falling back to a default with a warning."""
        resolved = getattr(enum_obj, attr_name, None)
        if resolved is not None:
            return resolved
        logger.warning(
            f"{label} '{attr_name}' not found in enum, falling back to {fallback_name}"
        )
        return getattr(enum_obj, fallback_name, None)

    @staticmethod
    def _resolve_algorithm(zp_module, algorithm: str):
        """Map algorithm string to OptimizationAlgorithm enum value."""
        alg_enum = zp_module.constants.Tools.Optimization.OptimizationAlgorithm
        aliases = {"DLS": "DampedLeastSquares"}
        attr_name = aliases.get(algorithm, algorithm)
        fallback = "DampedLeastSquares"
        return OptimizationMixin._resolve_enum(alg_enum, attr_name, fallback, "Algorithm")

    @staticmethod
    def _resolve_cycles(zp_module, cycles: int | None):
        """Map cycle count to OptimizationCycles enum value.

        The API only supports specific fixed counts (1, 5, 10, 50).
        Unrecognized values fall back to Automatic.
        """
        cycles_enum = zp_module.constants.Tools.Optimization.OptimizationCycles
        fallback = "Automatic"
        if cycles is None:
            return getattr(cycles_enum, fallback, None)
        mapping = {
            1: "Fixed_1_Cycle",
            5: "Fixed_5_Cycles",
            10: "Fixed_10_Cycles",
            50: "Fixed_50_Cycles",
        }
        attr_name = mapping.get(cycles, fallback)
        return OptimizationMixin._resolve_enum(cycles_enum, attr_name, fallback, "Cycles")

    @staticmethod
    def _resolve_save_count(zp_module, num_to_save: int | None):
        """Map save count to OptimizationSaveCount enum value.

        Picks the closest supported value (10, 20, 30, ..., 100).
        """
        save_enum = zp_module.constants.Tools.Optimization.OptimizationSaveCount
        fallback = "Save_10"
        if num_to_save is None:
            return getattr(save_enum, fallback, None)
        # API supports Save_10 through Save_100 in steps of 10
        valid = list(range(10, 101, 10))
        closest = min(valid, key=lambda k: abs(k - num_to_save))
        attr_name = f"Save_{closest}"
        return OptimizationMixin._resolve_enum(save_enum, attr_name, fallback, "SaveCount")

    @staticmethod
    def _read_systems_evaluated(opt_tool) -> int | None:
        """Read systems evaluated count from an optimization tool, or None on failure."""
        try:
            return int(opt_tool.Systems) if hasattr(opt_tool, 'Systems') else None
        except Exception as e:
            logger.debug(f"Could not read systems_evaluated: {e}")
            return None

    def _read_best_solutions(self, opt_tool, mfe, num_to_save: int = 10) -> list[float]:
        """Read the best N merit values from a global optimizer.

        CurrentMeritFunction(j) is 1-indexed and returns the j-th best solution's merit.
        Must be called before opt_tool.Close() — solutions are lost after that.
        """
        # Round up to the actual enum count we set (multiples of 10)
        count = max(10, min(((num_to_save + 9) // 10) * 10, 100))
        solutions: list[float] = []
        try:
            for j in range(1, count + 1):
                mf_val = _extract_value(opt_tool.CurrentMeritFunction(j))
                if mf_val is not None and mf_val > 0:
                    solutions.append(mf_val)
        except Exception as e:
            logger.warning(f"Failed to read CurrentMeritFunction({len(solutions) + 1}): {e}")
        if not solutions:
            try:
                mf_val = _extract_value(mfe.CalculateMeritFunction())
                if mf_val is not None:
                    solutions.append(mf_val)
            except Exception as e:
                logger.warning(f"MFE fallback for best solutions also failed: {e}")
        return solutions

    # ── Main optimization entry point ───────────────────────────────

    def run_optimization(
        self,
        method: str = "local",
        algorithm: str = "DLS",
        cycles: int | None = 5,
        timeout_seconds: float | None = 60,
        num_to_save: int | None = 10,
        operand_rows: list[dict] | None = None,
    ) -> dict[str, Any]:
        """
        Run OpticStudio optimization using Local, Global, or Hammer method.

        Args:
            method: "local" | "global" | "hammer"
            algorithm: "DLS" | "OrthogonalDescent"
            cycles: Cycle count for local optimization (1, 5, 10, 50, or None=auto)
            timeout_seconds: Time limit for global/hammer (they run indefinitely)
            num_to_save: Number of best solutions to retain (global only)
            operand_rows: Explicit MFE operand rows

        Returns:
            Dict with merit_before, merit_after, cycles_requested,
            operand_results, variable_states, and (for global) best_solutions
        """
        zp = self._zp
        mfe = self.oss.MFE

        # Normalize method
        method = (method or "local").lower()

        # Step 1: Populate MFE
        if not operand_rows:
            return {"success": False, "error": "Must provide operand_rows"}

        mfe_result = self.evaluate_merit_function(operand_rows)
        if not mfe_result.get("success"):
            return {
                "success": False,
                "error": f"MFE setup failed: {mfe_result.get('error')}",
            }

        # Step 2: Read initial merit
        try:
            merit_before = _extract_value(mfe.CalculateMeritFunction())
        except Exception as e:
            return {"success": False, "error": f"Initial merit calculation failed: {e}"}

        # Step 3: Run optimization (method-specific)
        best_solutions: list[float] | None = None
        systems_evaluated: int | None = None

        try:
            tools = self.oss.Tools
            resolved_alg = self._resolve_algorithm(zp, algorithm or "DLS")

            if method in ("global", "hammer"):
                # Global and Hammer both use timeout-based execution
                if method == "global":
                    opt_tool = tools.OpenGlobalOptimization()
                    save_count = self._resolve_save_count(zp, num_to_save)
                    if save_count is not None:
                        opt_tool.NumberToSave = save_count
                else:
                    opt_tool = tools.OpenHammerOptimization()

                opt_tool.Algorithm = resolved_alg
                opt_tool.NumberOfCores = 8
                timeout = max(10, min(timeout_seconds or 60, 600))

                try:
                    opt_tool.RunAndWaitWithTimeout(timeout)
                    opt_tool.Cancel()
                    opt_tool.WaitForCompletion()
                finally:
                    if method == "global":
                        best_solutions = self._read_best_solutions(opt_tool, mfe, num_to_save or 10)
                    systems_evaluated = self._read_systems_evaluated(opt_tool)
                    opt_tool.Close()

            else:  # local (default)
                opt_tool = tools.OpenLocalOptimization()
                opt_tool.Algorithm = resolved_alg
                opt_tool.NumberOfCores = 8

                resolved_cycles = self._resolve_cycles(zp, cycles)
                if resolved_cycles is not None and hasattr(opt_tool, 'Cycles'):
                    opt_tool.Cycles = resolved_cycles

                try:
                    opt_tool.RunAndWaitForCompletion()
                finally:
                    opt_tool.Close()

        except Exception as e:
            logger.error(f"Optimization run failed: {e}")
            return {"success": False, "error": f"Optimization failed: {e}"}

        # Step 4: Read final merit
        try:
            merit_after = _extract_value(mfe.CalculateMeritFunction())
        except Exception as e:
            merit_after = merit_before
            logger.warning(f"Post-optimization merit calculation failed: {e}")

        # Step 5: Read operand results from MFE
        operand_results = []
        try:
            mfe_cols = zp.constants.Editors.MFE.MeritColumn
            param_columns = [
                mfe_cols.Param1, mfe_cols.Param2,
                mfe_cols.Param3, mfe_cols.Param4,
                mfe_cols.Param5, mfe_cols.Param6,
                mfe_cols.Param7, mfe_cols.Param8,
            ]
            operand_results = self._read_mfe_rows(mfe, mfe_cols, param_columns, "post-opt")
        except Exception as e:
            logger.warning(f"Error reading post-optimization MFE: {e}")

        # Step 6: Extract variable states from LDE
        variable_states = self._extract_variable_states()

        # Save the modified system so the caller can round-trip via zmxToLlm
        modified_zmx_content = self._try_save_modified_system("optimization")

        result: dict[str, Any] = {
            "success": True,
            "method": method,
            "algorithm": algorithm,
            "merit_before": merit_before,
            "merit_after": merit_after,
            "cycles_requested": cycles if method == "local" else None,
            "operand_results": operand_results,
            "variable_states": variable_states,
            "modified_zmx_content": modified_zmx_content,
        }
        if best_solutions is not None:
            result["best_solutions"] = best_solutions
        if systems_evaluated is not None:
            result["systems_evaluated"] = systems_evaluated
        _log_raw_output("/run-optimization", result)
        return result

    # Maps parameter name -> (cell attribute, value attribute)
    _VARIABLE_PARAMS = {
        "radius": ("RadiusCell", "Radius"),
        "thickness": ("ThicknessCell", "Thickness"),
        "conic": ("ConicCell", "Conic"),
    }

    def _extract_variable_states(self) -> list[dict[str, Any]]:
        """
        Extract current values of all variable parameters from the LDE.

        Iterates all surfaces and checks if radius, thickness, conic, or
        aspheric parameters (Par1-Par12) are marked as variable.

        Returns:
            List of {surface_index, parameter, value, is_variable}
        """
        lde = self.oss.LDE
        variable_states: list[dict[str, Any]] = []

        try:
            num_surfaces = lde.NumberOfSurfaces
            surf_col = self._zp.constants.Editors.LDE.SurfaceColumn
        except Exception as e:
            logger.warning(f"Error accessing LDE for variable extraction: {e}")
            return variable_states

        # Start at surface 1 (skip object surface 0).
        # Return surf_idx as-is (Zemax LDE index) -- the analysis
        # service's surface_patcher uses the same indexing.
        for surf_idx in range(1, num_surfaces):
            try:
                surf = lde.GetSurfaceAt(surf_idx)
            except Exception as e:
                logger.debug(f"Error reading surface {surf_idx} variables: {e}")
                continue

            # Check radius, thickness, conic
            for param_name, (cell_attr, value_attr) in self._VARIABLE_PARAMS.items():
                try:
                    cell = getattr(surf, cell_attr)
                    if not hasattr(cell, 'GetSolveData'):
                        continue
                    solve = cell.GetSolveData()
                    solve_type = str(solve.Type).split('.')[-1] if solve else ""
                    if solve_type == "Variable":
                        # Radius and thickness can be Infinity in OpticStudio
                        # (flat surfaces, afocal systems / infinite conjugates)
                        val = _extract_value(
                            getattr(surf, value_attr), 0.0,
                            allow_inf=(param_name in ("radius", "thickness")),
                        )
                        variable_states.append({
                            "surface_index": surf_idx,
                            "parameter": param_name,
                            "value": val,
                            "is_variable": True,
                        })
                except Exception as e:
                    logger.warning(
                        f"Failed to read variable state for surface {surf_idx}, "
                        f"param '{param_name}': {type(e).__name__}: {e}"
                    )

            # Check aspheric parameters (PARM 1-12 via SurfaceColumn.Par1-Par12).
            # Not all surface types support all 12 params; stop at the first
            # missing column attribute.
            for par_idx in range(1, 13):
                par_attr = f'Par{par_idx}'
                if not hasattr(surf_col, par_attr):
                    break
                try:
                    cell = surf.GetSurfaceCell(getattr(surf_col, par_attr))
                    if cell is None or not hasattr(cell, 'GetSolveData'):
                        continue
                    solve = cell.GetSolveData()
                    solve_type = str(solve.Type).split('.')[-1] if solve else ""
                    if solve_type == "Variable":
                        variable_states.append({
                            "surface_index": surf_idx,
                            "parameter": f"param_{par_idx}",
                            "value": _extract_value(cell.DoubleValue, 0.0),
                            "is_variable": True,
                        })
                except (AttributeError, TypeError):
                    break  # Surface type does not support this parameter
                except Exception as e:
                    logger.debug(f"Error checking Par{par_idx} for surface {surf_idx}: {e}")

        return variable_states

    # ── QuickFocus (native) ──────────────────────────────────────────

    def run_quick_focus(
        self,
        criterion: str = "SpotSizeRadial",
        use_centroid: bool = True,
    ) -> dict[str, Any]:
        """
        Run OpticStudio's native QuickFocus tool.

        Adjusts the last surface thickness (back focal distance) to minimise
        the chosen focus criterion in a single internal call.

        Args:
            criterion: "SpotSizeRadial" or "RMSSpotSizeRadial"
            use_centroid: Whether to use centroid reference (True) or chief ray

        Returns:
            Dict with success, thickness_before, thickness_after, delta_thickness,
            and surface_index (1-based LDE index of the surface that was adjusted).
        """
        zp = self._zp
        lde = self.oss.LDE

        num_surfaces = lde.NumberOfSurfaces
        focus_surf_idx = num_surfaces - 2
        if focus_surf_idx < 1:
            return {"success": False, "error": "System too short for QuickFocus"}

        surf_before = lde.GetSurfaceAt(focus_surf_idx)
        thickness_before = _extract_value(surf_before.Thickness, 0.0, allow_inf=True)

        # Resolve the criterion enum (varies by OpticStudio version)
        criterion_enum = None
        for enum_path in (
            "Tools.General.QuickFocusCriterion",
            "Tools.General.QuickAdjustCriterion",
        ):
            try:
                parts = enum_path.split(".")
                obj = zp.constants
                for part in parts:
                    obj = getattr(obj, part)
                criterion_enum = obj
                break
            except AttributeError:
                continue

        if criterion_enum is None:
            return {"success": False, "error": "QuickFocus criterion enum not found in this OpticStudio version"}

        resolved_criterion = getattr(criterion_enum, criterion, None)
        if resolved_criterion is None:
            resolved_criterion = getattr(criterion_enum, "SpotSizeRadial", None)
            if resolved_criterion is None:
                return {"success": False, "error": f"Criterion '{criterion}' not available"}

        try:
            qf = self.oss.Tools.OpenQuickFocus()
            if qf is None:
                return {"success": False, "error": "Could not open QuickFocus tool (another tool may be active)"}
            qf.Criterion = resolved_criterion
            if hasattr(qf, 'UseCentroid'):
                qf.UseCentroid = use_centroid
            qf.RunAndWaitForCompletion()
            qf.Close()
        except Exception as e:
            return {"success": False, "error": f"QuickFocus failed: {e}"}

        surf_after = lde.GetSurfaceAt(focus_surf_idx)
        thickness_after = _extract_value(surf_after.Thickness, 0.0, allow_inf=True)

        # Save the modified system so the caller can round-trip via zmxToLlm
        modified_zmx_content = self._try_save_modified_system("quick-focus")

        result = {
            "success": True,
            "surface_index": focus_surf_idx,
            "thickness_before": thickness_before,
            "thickness_after": thickness_after,
            "delta_thickness": thickness_after - thickness_before,
            "criterion": criterion,
            "modified_zmx_content": modified_zmx_content,
        }
        _log_raw_output("/quick-focus", result)
        return result

    # ── Scale Lens (native) ─────────────────────────────────────────

    # Unit name → ZOSAPI index mappings
    _UNIT_INDEX = {"mm": 0, "cm": 1, "inches": 2, "meters": 3}
    _UNIT_NAMES = {0: "mm", 1: "cm", 2: "inches", 3: "meters"}
    _ENUM_NAME_TO_INDEX = {"Millimeters": 0, "Centimeters": 1, "Inches": 2, "Meters": 3}

    def run_scale_lens(
        self,
        mode: str = "factor",
        scale_factor: float | None = None,
        target_unit: str | None = None,
    ) -> dict[str, Any]:
        """
        Run OpticStudio's native Scale Lens tool.

        Uniformly scales all dimensions by a factor, or converts between unit systems.

        Args:
            mode: "factor" to scale by numeric factor, "units" to convert unit systems
            scale_factor: Numeric scale factor (required for mode="factor")
            target_unit: Target unit system (required for mode="units")

        Returns:
            Dict with success, mode, scale_factor, efl_before/after, total_track_before/after
        """
        # Validate args
        if mode == "factor" and scale_factor is None:
            return {"success": False, "error": "scale_factor is required for mode='factor'"}
        if mode == "units" and target_unit is None:
            return {"success": False, "error": "target_unit is required for mode='units'"}
        if mode == "units" and target_unit not in self._UNIT_INDEX:
            return {"success": False, "error": f"Invalid target_unit: {target_unit}. Must be one of: mm, cm, inches, meters"}

        # Read pre-scale paraxial data
        efl_before = self._get_efl()
        paraxial_before = self._get_paraxial_from_lde()
        total_track_before = paraxial_before.get("total_track")

        # Read original unit for reporting
        original_unit = None
        try:
            unit_type = self.oss.SystemData.Units.LensUnits
            if isinstance(unit_type, int):
                unit_index = unit_type
            else:
                enum_name = str(unit_type).split(".")[-1]
                unit_index = self._ENUM_NAME_TO_INDEX.get(enum_name)
            original_unit = self._UNIT_NAMES.get(unit_index) if unit_index is not None else None
        except Exception as e:
            logger.warning(f"Could not read original unit: {e}")

        # Run Scale Lens tool
        # IScale interface: ScaleByFactor/ScaleByUnits are mutually exclusive toggles.
        # ScaleToUnit takes ZOSAPI.Tools.General.ScaleToUnits enum.
        try:
            scale_tool = self.oss.Tools.OpenScale()
            if scale_tool is None:
                return {"success": False, "error": "Could not open Scale Lens tool (another tool may be active)"}

            if mode == "factor":
                scale_tool.ScaleByFactor = True
                scale_tool.ScaleFactor = float(scale_factor)
            else:  # mode == "units"
                scale_tool.ScaleByUnits = True
                # Resolve ScaleToUnits enum (ZOSAPI.Tools.General.ScaleToUnits)
                unit_idx = self._UNIT_INDEX[target_unit]
                unit_enum_names = {
                    0: "Millimeters", 1: "Centimeters",
                    2: "Inches", 3: "Meters",
                }
                try:
                    scale_units_enum = self._zp.constants.Tools.General.ScaleToUnits
                    enum_val = getattr(scale_units_enum, unit_enum_names[unit_idx], None)
                    if enum_val is not None:
                        scale_tool.ScaleToUnit = enum_val
                    else:
                        scale_tool.ScaleToUnit = unit_idx
                except Exception:
                    # Fallback: integer index works per example script 11
                    scale_tool.ScaleToUnit = unit_idx

            scale_tool.RunAndWaitForCompletion()
            scale_tool.Close()

        except Exception as e:
            logger.error(f"Scale Lens tool failed: {e}")
            return {"success": False, "error": f"Scale Lens failed: {e}"}

        # Read post-scale paraxial data
        efl_after = self._get_efl()
        paraxial_after = self._get_paraxial_from_lde()
        total_track_after = paraxial_after.get("total_track")

        # Compute effective factor for unit conversion mode
        effective_factor = scale_factor
        if mode == "units" and efl_before and efl_after and abs(efl_before) > 1e-12:
            effective_factor = efl_after / efl_before

        # Save the modified system so the caller can round-trip via zmxToLlm
        modified_zmx_content = self._try_save_modified_system("scale-lens")

        result = {
            "success": True,
            "mode": mode,
            "scale_factor": effective_factor,
            "efl_before": efl_before,
            "efl_after": efl_after,
            "total_track_before": total_track_before,
            "total_track_after": total_track_after,
            "original_unit": original_unit,
            "target_unit": target_unit if mode == "units" else None,
            "modified_zmx_content": modified_zmx_content,
        }
        _log_raw_output("/scale-lens", result)
        return result
