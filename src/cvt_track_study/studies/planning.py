"""Study-point construction and safe infinite-reference cache policy."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from cvt_track_study.config.uncertainty import UncertainChoice, UncertainQuantity
from cvt_track_study.config.units import convert_to_si
from cvt_track_study.simulation.service import SimulationError
from cvt_track_study.uncertainty import quantity_quantile_si

from .model import DesignPoint

REFERENCE_INVARIANT_DESIGN_PATHS = {
    "drivetrain.cvt.minimum_reduction_ratio",
    "drivetrain.cvt.maximum_reduction_ratio",
    "drivetrain.final_drive_ratio",
}


def study_plan(
    study_type: str,
    raw: Mapping[str, Any],
    registry: Any,
    replicates_override: int | None,
) -> tuple[tuple[DesignPoint, ...], str, int]:
    sampling = raw.get("sampling", {})
    if study_type == "design_sweep":
        variable = raw["design_variable"]
        path = str(variable["path"])
        registered = registry.by_path.get(path)
        if registered is None or not isinstance(registered.value, UncertainQuantity):
            raise SimulationError(f"Design variable {path!r} must identify a numeric quantity.")
        unit = registered.value.unit
        points = tuple(
            DesignPoint(
                identifier=f"{path}={float(value):g}",
                path=path,
                display_value=float(value),
                value_si=float(convert_to_si(float(value), unit)[0]),
                nominal=np.isclose(float(value), registered.value.nominal),
            )
            for value in variable["values"]
        )
        return (
            points,
            str(sampling.get("mode", "measured_track")),
            _replicates(sampling, replicates_override),
        )
    if study_type == "structural_sensitivity":
        sensitivity = raw["sensitivity"]
        probabilities = tuple(
            float(value)
            for value in sensitivity.get("quantiles", [0.05, 0.2, 0.8, 0.95])
        )
        points: list[DesignPoint] = []
        for path in sensitivity["parameters"]:
            registered = registry.by_path.get(str(path))
            if registered is None:
                raise SimulationError(
                    f"Sensitivity parameter {path!r} does not identify a declared input."
                )
            if registered.category != "structural":
                raise SimulationError(
                    f"Sensitivity parameter {path!r} must have uncertainty.role='structural'."
                )
            value = registered.value
            if value.uncertainty.distribution.value == "fixed":
                raise SimulationError(
                    f"Sensitivity parameter {path!r} is explicitly fixed and has no declared range."
                )
            if isinstance(value, UncertainQuantity):
                nominal_si = value.nominal_si()[0]
                points.append(
                    DesignPoint(
                        identifier=f"{path}@nominal",
                        path=str(path),
                        display_value=value.nominal,
                        value_si=nominal_si,
                        level_probability=None,
                        level_kind="nominal",
                        nominal=True,
                    )
                )
                seen_values = {round(float(nominal_si), 14)}
                for probability in probabilities:
                    value_si = quantity_quantile_si(value, probability)
                    key = round(float(value_si), 14)
                    if key in seen_values:
                        continue
                    seen_values.add(key)
                    points.append(
                        DesignPoint(
                            identifier=f"{path}@q{probability:g}",
                            path=str(path),
                            display_value=value_si / convert_to_si(1.0, value.unit)[0],
                            value_si=value_si,
                            level_probability=probability,
                            level_kind="quantile",
                            nominal=False,
                        )
                    )
            elif isinstance(value, UncertainChoice):
                alternatives = value.uncertainty.choices
                if not alternatives:
                    raise SimulationError(
                        f"Sensitivity choice {path!r} has no declared alternatives."
                    )
                ordered = (value.nominal, *(choice for choice in alternatives if choice != value.nominal))
                for index, choice in enumerate(ordered):
                    points.append(
                        DesignPoint(
                            identifier=(
                                f"{path}@nominal" if index == 0 else f"{path}@choice={choice}"
                            ),
                            path=str(path),
                            display_value=choice,
                            value_si=None,
                            choice_value=choice,
                            level_probability=None,
                            level_kind="nominal" if index == 0 else "choice",
                            nominal=index == 0,
                        )
                    )
            else:  # pragma: no cover - registry contract makes this unreachable
                raise SimulationError(
                    f"Sensitivity parameter {path!r} has an unsupported contract type."
                )
        return tuple(points), "nominal", 1
    point = DesignPoint("nominal", None, "nominal", None, nominal=True)
    return (
        (point,),
        str(sampling.get("mode", "measured_track")),
        _replicates(sampling, replicates_override),
    )


def reference_cache_key(
    replicate: int,
    design: DesignPoint,
    *,
    share_across_designs: bool = True,
) -> tuple[int, str]:
    """Return one common reference key for all candidates in a design sweep."""

    return (
        (replicate, "shared")
        if share_across_designs
        else (replicate, design.identifier)
    )


def _replicates(sampling: Mapping[str, Any], override: int | None) -> int:
    value = int(override if override is not None else sampling.get("replicates", 1))
    if value < 1:
        raise SimulationError("replicates must be positive.")
    return value
