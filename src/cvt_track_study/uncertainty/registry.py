"""Discover uncertainty-aware inputs from resolved project and bundle contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from cvt_track_study.bundle import TrackBundle
from cvt_track_study.config.uncertainty import (
    UncertainChoice,
    UncertainQuantity,
    UncertaintyValidationError,
)
from cvt_track_study.contracts.obstacles import obstacle_model_alternatives


GLOBAL_OBSTACLE_SEVERITY_PATH = "track.obstacle_energy_severity"


@dataclass(frozen=True, slots=True)
class RegisteredInput:
    path: str
    category: str
    value: UncertainQuantity | UncertainChoice
    feature_id: str | None = None
    alternative_model_type: str | None = None

    @property
    def correlation_group(self) -> str | None:
        return self.value.correlation_group


@dataclass(frozen=True, slots=True)
class InputRegistry:
    inputs: tuple[RegisteredInput, ...]

    @property
    def by_path(self) -> dict[str, RegisteredInput]:
        return {item.path: item for item in self.inputs}

    def stochastic(self) -> tuple[RegisteredInput, ...]:
        return tuple(
            item
            for item in self.inputs
            if item.value.uncertainty.distribution.value != "fixed"
        )


def build_input_registry(
    *,
    vehicle_raw: Mapping[str, Any],
    base_study_raw: Mapping[str, Any],
    track_raw: Mapping[str, Any],
    bundle: TrackBundle,
) -> InputRegistry:
    rows: list[RegisteredInput] = []
    _collect_physical(vehicle_raw.get("vehicle"), "vehicle", "structural", rows)
    _collect_physical(
        vehicle_raw.get("drivetrain"), "drivetrain", "structural", rows
    )
    _collect_physical(base_study_raw.get("driver"), "driver", "structural", rows)
    _collect_physical(
        base_study_raw.get("initial_conditions"),
        "initial_conditions",
        "initial_condition",
        rows,
    )
    surface = track_raw.get("surface") if isinstance(track_raw, Mapping) else None
    _collect_physical(surface, "track.surface", "structural", rows)

    severity = _global_obstacle_severity_contract(base_study_raw)
    rows.append(
        RegisteredInput(
            path=GLOBAL_OBSTACLE_SEVERITY_PATH,
            category=_resolved_category(severity, "structural"),
            value=severity,
        )
    )

    for feature in bundle.physical_features:
        feature_id = str(feature["id"])
        raw_model = feature.get("obstacle_model")
        if not isinstance(raw_model, Mapping):
            continue
        choice, alternatives = obstacle_model_alternatives(raw_model)
        rows.append(
            RegisteredInput(
                path=f"obstacle.{feature_id}.model_type",
                category=_resolved_category(choice, "structural"),
                value=choice,
                feature_id=feature_id,
            )
        )
        for model_type, parameters in alternatives.items():
            for name, quantity in parameters.items():
                rows.append(
                    RegisteredInput(
                        path=f"obstacle.{feature_id}.{model_type}.{name}",
                        category=_resolved_category(quantity, "structural"),
                        value=quantity,
                        feature_id=feature_id,
                        alternative_model_type=model_type,
                    )
                )
    paths = [item.path for item in rows]
    if len(paths) != len(set(paths)):
        raise UncertaintyValidationError(
            "Uncertainty registry contains duplicate paths."
        )
    return InputRegistry(tuple(rows))


def _global_obstacle_severity_contract(
    base_study_raw: Mapping[str, Any],
) -> UncertainQuantity:
    realization = base_study_raw.get("track_realization", {})
    raw = (
        realization.get("obstacle_energy_severity")
        if isinstance(realization, Mapping)
        else None
    )
    if raw is None:
        raw = {
            "nominal": 1.0,
            "unit": "1",
            "source": {
                "kind": "inherited_default",
                "reference": "global obstacle energy severity screening family v1",
                "notes": (
                    "Low=0.6, nominal=1.0, high=1.4 are structural screening "
                    "levels, not a calibrated probability statement."
                ),
            },
            "uncertainty": {
                "distribution": "triangular",
                "lower": 0.6,
                "mode": 1.0,
                "upper": 1.4,
                "role": "structural",
                "reason": (
                    "Obstacle energy is poorly calibrated; vary one coherent "
                    "course-wide burden while preserving per-feature model and "
                    "parameter uncertainty separately."
                ),
            },
            "notes": (
                "Multiplies dissipative obstacle resistance only. Geometry, "
                "normal-load, grade, traction, model inclusion, and per-obstacle "
                "parameters are not rescaled."
            ),
        }
    if not isinstance(raw, Mapping):
        raise UncertaintyValidationError(
            "track_realization.obstacle_energy_severity must be a quantity table."
        )
    return UncertainQuantity.from_mapping(raw)


def _collect_physical(
    raw: Any,
    prefix: str,
    category: str,
    rows: list[RegisteredInput],
) -> None:
    if not isinstance(raw, Mapping):
        return
    for key, value in raw.items():
        path = f"{prefix}.{key}"
        if not isinstance(value, Mapping):
            continue
        if "nominal" in value:
            parsed: UncertainQuantity | UncertainChoice
            if "unit" in value:
                parsed = UncertainQuantity.from_mapping(value)
            else:
                parsed = UncertainChoice.from_mapping(value)
            rows.append(
                RegisteredInput(
                    path=path,
                    category=_resolved_category(parsed, category),
                    value=parsed,
                )
            )
        else:
            _collect_physical(value, path, category, rows)


def _resolved_category(
    value: UncertainQuantity | UncertainChoice, default: str
) -> str:
    role = value.uncertainty.role
    return role.value if role is not None else default
