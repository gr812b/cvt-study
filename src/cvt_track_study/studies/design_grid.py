"""Cartesian multi-variable design grids for paired drivetrain studies.

The public configuration accepts either the historical single ``[design_variable]``
table or a list of ``[[design_variables]]`` tables.  Every Cartesian combination
becomes one design candidate.  The helpers in this module intentionally use duck
-typed design points so the extension remains backward compatible with the
existing :class:`DesignPoint` contract.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import product
import json
import math
from typing import Any

import numpy as np

from cvt_track_study.config.uncertainty import UncertainQuantity
from cvt_track_study.config.units import convert_to_si
from cvt_track_study.simulation.service import SimulationError


REFERENCE_INVARIANT_DESIGN_PATHS = {
    "drivetrain.cvt.minimum_reduction_ratio",
    "drivetrain.cvt.maximum_reduction_ratio",
    "drivetrain.final_drive_ratio",
}


@dataclass(frozen=True, slots=True)
class GridDesignPoint:
    """A design point carrying one or more simultaneous numeric overrides."""

    identifier: str
    path: str | None
    display_value: float | str
    value_si: float | None
    choice_value: str | None = None
    level_probability: float | None = None
    level_kind: str = "design_grid"
    nominal: bool = False
    display_values: tuple[tuple[str, float], ...] = ()
    quantity_values_si: tuple[tuple[str, float], ...] = ()
    choice_values: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class _Axis:
    path: str
    values: tuple[float, ...]
    values_si: tuple[float, ...]
    nominal: float


def design_sweep_plan(
    raw: Mapping[str, Any],
    registry: Any,
    replicates_override: int | None,
) -> tuple[tuple[GridDesignPoint, ...], str, int]:
    """Build the Cartesian design grid while preserving the old 1-D contract."""

    sampling = raw.get("sampling", {})
    if not isinstance(sampling, Mapping):
        sampling = {}
    variables = configured_design_variables(raw)
    axes = tuple(_resolve_axis(variable, registry) for variable in variables)

    candidate_count = math.prod(len(axis.values) for axis in axes)
    grid_config = raw.get("design_grid", {})
    if not isinstance(grid_config, Mapping):
        grid_config = {}
    maximum_candidates = int(grid_config.get("maximum_candidates", 500))
    if maximum_candidates < 1:
        raise SimulationError("design_grid.maximum_candidates must be positive.")
    if candidate_count > maximum_candidates:
        raise SimulationError(
            f"Design grid contains {candidate_count} candidates, exceeding "
            f"design_grid.maximum_candidates={maximum_candidates}."
        )

    points: list[GridDesignPoint] = []
    level_ranges = [range(len(axis.values)) for axis in axes]
    for indices in product(*level_ranges):
        display_values = tuple(
            (axis.path, float(axis.values[index]))
            for axis, index in zip(axes, indices, strict=True)
        )
        quantity_values_si = tuple(
            (axis.path, float(axis.values_si[index]))
            for axis, index in zip(axes, indices, strict=True)
        )
        nominal = all(
            np.isclose(axis.values[index], axis.nominal)
            for axis, index in zip(axes, indices, strict=True)
        )
        identifier = " | ".join(
            f"{design_axis_label(path)}={_format_value(value)}"
            for path, value in display_values
        )
        if len(display_values) == 1:
            path, display_value = display_values[0]
            value_si = quantity_values_si[0][1]
        else:
            path = None
            display_value = identifier
            value_si = None
        points.append(
            GridDesignPoint(
                identifier=identifier,
                path=path,
                display_value=display_value,
                value_si=value_si,
                nominal=nominal,
                display_values=display_values,
                quantity_values_si=quantity_values_si,
            )
        )

    _validate_transmission_relationships(points, registry)

    mode = str(sampling.get("mode", "measured_track"))
    replicates = int(
        replicates_override
        if replicates_override is not None
        else sampling.get("replicates", 1)
    )
    if replicates < 1:
        raise SimulationError("replicates must be positive.")
    return tuple(points), mode, replicates


def configured_design_variables(raw: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    """Return one normalized list for old and new configuration syntax."""

    single = raw.get("design_variable")
    multiple = raw.get("design_variables")
    # A legacy [design_variable] table may be present solely so older validators
    # accept the project.  When the modern [[design_variables]] array exists, it
    # is authoritative and the compatibility table is intentionally ignored.
    if multiple is not None:
        if (
            not isinstance(multiple, Sequence)
            or isinstance(multiple, (str, bytes))
            or not multiple
            or not all(isinstance(item, Mapping) for item in multiple)
        ):
            raise SimulationError(
                "[[design_variables]] must contain one or more design-variable tables."
            )
        variables = tuple(multiple)
    elif isinstance(single, Mapping):
        variables = (single,)
    else:
        raise SimulationError(
            "Design sweep requires [design_variable] or one or more [[design_variables]]."
        )

    paths = [str(variable.get("path", "")).strip() for variable in variables]
    if any(not path for path in paths):
        raise SimulationError("Every design variable requires a non-empty path.")
    if len(paths) != len(set(paths)):
        raise SimulationError("Design-variable paths must be unique.")
    return variables


def design_paths(points: Sequence[Any]) -> tuple[str, ...]:
    """Return every distinct path overridden by the supplied design points."""

    ordered: list[str] = []
    for point in points:
        for path in design_display_values(point):
            if path not in ordered:
                ordered.append(path)
    return tuple(ordered)


def design_display_values(point: Any) -> dict[str, float | str]:
    configured = getattr(point, "display_values", ())
    if configured:
        return {str(path): value for path, value in configured}
    path = getattr(point, "path", None)
    if path:
        return {str(path): getattr(point, "display_value", "")}
    return {}


def design_quantity_values_si(point: Any) -> dict[str, float]:
    configured = getattr(point, "quantity_values_si", ())
    if configured:
        return {str(path): float(value) for path, value in configured}
    path = getattr(point, "path", None)
    value = getattr(point, "value_si", None)
    return {str(path): float(value)} if path is not None and value is not None else {}


def design_choice_values(point: Any) -> dict[str, str]:
    configured = getattr(point, "choice_values", ())
    if configured:
        return {str(path): str(value) for path, value in configured}
    path = getattr(point, "path", None)
    value = getattr(point, "choice_value", None)
    return {str(path): str(value)} if path is not None and value is not None else {}


def reference_can_be_shared(paths: Sequence[str]) -> bool:
    """Only transmission-range changes leave the infinite reference invariant."""

    return bool(paths) and set(paths).issubset(REFERENCE_INVARIANT_DESIGN_PATHS)


def design_axis_label(path: str) -> str:
    aliases = {
        "drivetrain.final_drive_ratio": "final_drive",
        "drivetrain.cvt.maximum_reduction_ratio": "cvt_max",
        "drivetrain.cvt.minimum_reduction_ratio": "cvt_min",
    }
    if path in aliases:
        return aliases[path]
    parts = path.split(".")
    return ".".join(parts[-2:]) if len(parts) >= 2 else path


def design_values_json(point: Any) -> str:
    return json.dumps(
        design_display_values(point),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _resolve_axis(variable: Mapping[str, Any], registry: Any) -> _Axis:
    path = str(variable.get("path", "")).strip()
    registered = registry.by_path.get(path)
    if registered is None or not isinstance(registered.value, UncertainQuantity):
        raise SimulationError(
            f"Design variable {path!r} must identify a numeric quantity."
        )
    raw_values = variable.get("values")
    if (
        not isinstance(raw_values, Sequence)
        or isinstance(raw_values, (str, bytes))
        or not raw_values
    ):
        raise SimulationError(
            f"Design variable {path!r} requires a non-empty numeric values array."
        )
    values: list[float] = []
    for raw_value in raw_values:
        if isinstance(raw_value, bool):
            raise SimulationError(
                f"Design value {raw_value!r} for {path!r} is not numeric."
            )
        try:
            value = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise SimulationError(
                f"Design value {raw_value!r} for {path!r} is not numeric."
            ) from exc
        if not math.isfinite(value):
            raise SimulationError(
                f"Design value {raw_value!r} for {path!r} must be finite."
            )
        values.append(value)
    if len(values) != len(set(values)):
        raise SimulationError(f"Design values for {path!r} must be unique.")
    unit = registered.value.unit
    values_si = tuple(float(convert_to_si(value, unit)[0]) for value in values)
    return _Axis(
        path=path,
        values=tuple(values),
        values_si=values_si,
        nominal=float(registered.value.nominal),
    )


def _format_value(value: float) -> str:
    return f"{value:g}"


def _validate_transmission_relationships(
    points: Sequence[GridDesignPoint],
    registry: Any,
) -> None:
    """Reject Cartesian candidates with an impossible CVT ratio ordering."""

    maximum_path = "drivetrain.cvt.maximum_reduction_ratio"
    minimum_path = "drivetrain.cvt.minimum_reduction_ratio"

    def nominal(path: str) -> float | None:
        registered = registry.by_path.get(path)
        value = getattr(registered, "value", None)
        raw = getattr(value, "nominal", None)
        try:
            return float(raw)
        except (TypeError, ValueError):
            return None

    nominal_maximum = nominal(maximum_path)
    nominal_minimum = nominal(minimum_path)
    if nominal_maximum is None or nominal_minimum is None:
        return

    for point in points:
        values = design_display_values(point)
        maximum = float(values.get(maximum_path, nominal_maximum))
        minimum = float(values.get(minimum_path, nominal_minimum))
        if maximum <= minimum:
            raise SimulationError(
                "Every design-grid candidate requires "
                "drivetrain.cvt.maximum_reduction_ratio > "
                "drivetrain.cvt.minimum_reduction_ratio; "
                f"found maximum={maximum:g}, minimum={minimum:g} in "
                f"{point.identifier!r}."
            )
