"""Shared uncertainty-aware obstacle-model contract."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from cvt_track_study.config.uncertainty import (
    DistributionKind,
    UncertainChoice,
    UncertainQuantity,
    UncertaintyValidationError,
)
from cvt_track_study.config.units import UnitValidationError, require_dimension


OBSTACLE_PARAMETER_DIMENSIONS: dict[str, dict[str, str]] = {
    "none": {},
    "fixed_specific_energy": {
        "specific_energy_loss": "specific_energy",
    },
    "speed_quadratic_energy": {
        "specific_fixed_energy": "specific_energy",
        "impact_coefficient": "mass",
    },
    "distributed_resistance": {
        "resistance_force": "force",
    },
    "roughness_energy_density": {
        "specific_energy_per_distance": "specific_energy_per_distance",
    },
    "smooth_profile": {
        "vertical_amplitude": "length",
        "specific_fixed_energy": "specific_energy",
        "impact_coefficient": "mass",
        "traction_multiplier": "dimensionless",
        "minimum_normal_load_scale": "dimensionless",
        "maximum_normal_load_scale": "dimensionless",
    },
}
OBSTACLE_MODEL_TYPES = frozenset(OBSTACLE_PARAMETER_DIMENSIONS)


def obstacle_model_alternatives(
    raw: Mapping[str, Any],
) -> tuple[UncertainChoice, dict[str, dict[str, UncertainQuantity]]]:
    """Validate one obstacle declaration and return every supported model branch.

    Fixed model choices retain the compact Phase 5 ``parameters`` table. A
    discrete model-form declaration must instead provide one complete parameter
    contract below ``alternatives.<model_type>.parameters`` for every choice.
    This prevents a sampled model switch from silently reusing coefficients with
    incompatible meanings or units.
    """

    status = str(raw.get("status", ""))
    if status != "declared":
        raise UncertaintyValidationError(
            "obstacle_model.status must be 'declared'; choose an explicit none profile when no force model applies."
        )
    model_raw = raw.get("model_type")
    if not isinstance(model_raw, Mapping):
        raise UncertaintyValidationError(
            "obstacle_model.model_type must be an uncertainty-aware categorical choice."
        )
    choice = UncertainChoice.from_mapping(model_raw)
    _require_stochastic_role(choice, "obstacle model choice")
    model_types = (
        choice.uncertainty.choices
        if choice.uncertainty.distribution.value == "discrete"
        else (choice.nominal,)
    )
    unknown = set(model_types) - set(OBSTACLE_MODEL_TYPES)
    if unknown:
        raise UncertaintyValidationError(
            f"Unknown obstacle model(s) {sorted(unknown)}; expected one of {sorted(OBSTACLE_MODEL_TYPES)}."
        )
    alternatives: dict[str, dict[str, UncertainQuantity]] = {}
    if len(model_types) == 1:
        parameters_raw = raw.get("parameters", {})
        alternatives[model_types[0]] = _parse_parameter_contract(
            model_types[0], parameters_raw
        )
        if raw.get("alternatives") not in (None, {}):
            raise UncertaintyValidationError(
                "Fixed obstacle model declarations use parameters, not alternatives."
            )
    else:
        alternatives_raw = raw.get("alternatives")
        if not isinstance(alternatives_raw, Mapping):
            raise UncertaintyValidationError(
                "Discrete obstacle model choices require obstacle_model.alternatives."
            )
        missing = set(model_types) - set(alternatives_raw)
        extra = set(alternatives_raw) - set(model_types)
        if missing or extra:
            raise UncertaintyValidationError(
                "Obstacle model alternatives must match uncertainty.choices exactly; "
                f"missing={sorted(missing)}, unexpected={sorted(extra)}."
            )
        if raw.get("parameters") not in (None, {}):
            raise UncertaintyValidationError(
                "Discrete obstacle model declarations keep parameters inside each alternative."
            )
        for model_type in model_types:
            branch = alternatives_raw[model_type]
            if not isinstance(branch, Mapping):
                raise UncertaintyValidationError(
                    f"Obstacle alternative {model_type!r} must be a table."
                )
            parameters_raw = branch.get("parameters", {})
            alternatives[model_type] = _parse_parameter_contract(
                model_type, parameters_raw
            )
    return choice, alternatives


def validate_obstacle_model_contract(raw: Mapping[str, Any]) -> tuple[str, dict[str, UncertainQuantity]]:
    """Validate one declaration and return the nominal model branch."""

    choice, alternatives = obstacle_model_alternatives(raw)
    return choice.nominal, alternatives[choice.nominal]



def _require_stochastic_role(
    value: UncertainQuantity | UncertainChoice, label: str
) -> None:
    if value.uncertainty.distribution is DistributionKind.FIXED:
        return
    role = value.uncertainty.role
    if role is None or role.value not in {"structural", "measured_track"}:
        raise UncertaintyValidationError(
            f"{label} has non-fixed uncertainty and must declare uncertainty.role as "
            "'structural' or 'measured_track'. Broad engineering priors should use "
            "'structural'; observed lap-to-lap event variability should use "
            "'measured_track'."
        )

def _parse_parameter_contract(
    model_type: str, parameters_raw: Any
) -> dict[str, UncertainQuantity]:
    if not isinstance(parameters_raw, Mapping):
        raise UncertaintyValidationError(
            f"Obstacle model {model_type!r} parameters must be a table."
        )
    expected = OBSTACLE_PARAMETER_DIMENSIONS[model_type]
    actual = set(parameters_raw)
    missing = set(expected) - actual
    extra = actual - set(expected)
    if missing or extra:
        details: list[str] = []
        if missing:
            details.append("missing " + ", ".join(sorted(missing)))
        if extra:
            details.append("unexpected " + ", ".join(sorted(extra)))
        raise UncertaintyValidationError(
            f"Obstacle model {model_type!r} parameter contract failed: {'; '.join(details)}."
        )
    parsed: dict[str, UncertainQuantity] = {}
    for name, dimension in expected.items():
        value = parameters_raw[name]
        if not isinstance(value, Mapping):
            raise UncertaintyValidationError(
                f"Obstacle parameter {name!r} must be an uncertainty-aware quantity table."
            )
        quantity = UncertainQuantity.from_mapping(value)
        try:
            require_dimension(quantity.unit, dimension)
        except UnitValidationError as exc:
            raise UncertaintyValidationError(
                f"Obstacle parameter {name!r}: {exc}"
            ) from exc
        _require_stochastic_role(quantity, f"obstacle parameter {name!r}")
        _validate_parameter_support(model_type, name, quantity)
        parsed[name] = quantity
    _validate_model_relationships(model_type, parsed)
    return parsed


def _validate_parameter_support(
    model_type: str, name: str, quantity: UncertainQuantity
) -> None:
    if model_type == "smooth_profile" and name == "vertical_amplitude":
        return
    lower, _ = _quantity_support(quantity)
    if lower is None:
        raise UncertaintyValidationError(
            f"Obstacle parameter {name!r} must use a distribution with explicit non-negative support; "
            "use truncated_normal rather than an unbounded normal."
        )
    if lower < 0.0:
        raise UncertaintyValidationError(
            f"Obstacle parameter {name!r} uncertainty cannot extend below zero."
        )


def _quantity_support(
    quantity: UncertainQuantity,
) -> tuple[float | None, float | None]:
    spec = quantity.uncertainty
    if spec.distribution is DistributionKind.FIXED:
        return quantity.nominal, quantity.nominal
    if spec.distribution is DistributionKind.NORMAL:
        return None, None
    if spec.distribution in {
        DistributionKind.TRUNCATED_NORMAL,
        DistributionKind.UNIFORM,
        DistributionKind.TRIANGULAR,
    }:
        return spec.lower, spec.upper
    if spec.distribution is DistributionKind.EMPIRICAL:
        return min(spec.samples), max(spec.samples)
    return None, None


def _validate_model_relationships(
    model_type: str, parameters: Mapping[str, UncertainQuantity]
) -> None:
    nominal = {name: quantity.nominal_si()[0] for name, quantity in parameters.items()}
    nonnegative = set(nominal)
    if model_type == "smooth_profile":
        nonnegative.remove("vertical_amplitude")
    for name in nonnegative:
        if nominal[name] < 0.0:
            raise UncertaintyValidationError(
                f"Obstacle parameter {name!r} must be non-negative."
            )
    if model_type == "smooth_profile":
        if nominal["maximum_normal_load_scale"] < nominal["minimum_normal_load_scale"]:
            raise UncertaintyValidationError(
                "maximum_normal_load_scale must be at least minimum_normal_load_scale."
            )
        _, minimum_upper = _quantity_support(parameters["minimum_normal_load_scale"])
        maximum_lower, _ = _quantity_support(parameters["maximum_normal_load_scale"])
        if (
            minimum_upper is None
            or maximum_lower is None
            or maximum_lower < minimum_upper
        ):
            raise UncertaintyValidationError(
                "The uncertainty supports for smooth-profile normal-load scales overlap; "
                "every possible maximum_normal_load_scale must remain at least as large "
                "as every possible minimum_normal_load_scale."
            )

