"""Explicit obstacle equations consumed from the versioned track bundle.

Every model returns a local effect at a track coordinate.  The model contract is
spatial: dissipative energy is represented as a force density whose line integral
matches the declared energy when speed-independent.  No model is inferred from a
GPS speed drop inside this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, isfinite, pi, sin
from typing import Any, Mapping, Protocol

from cvt_track_study.contracts.obstacles import obstacle_model_alternatives

from .models import SimulationInputError


@dataclass(frozen=True, slots=True)
class ObstacleContext:
    local_distance_m: float
    interval_length_m: float
    vehicle_speed_mps: float
    entry_speed_mps: float
    vehicle_mass_kg: float
    gravity_mps2: float


@dataclass(frozen=True, slots=True)
class ObstacleEffect:
    resistance_force_n: float = 0.0
    elevation_offset_m: float = 0.0
    grade_slope_addition: float = 0.0
    normal_load_scale: float = 1.0
    friction_multiplier: float = 1.0


class ObstacleModel(Protocol):
    model_type: str

    def evaluate(self, context: ObstacleContext) -> ObstacleEffect: ...


@dataclass(frozen=True, slots=True)
class NoObstacle:
    model_type: str = "none"

    def evaluate(self, context: ObstacleContext) -> ObstacleEffect:
        return ObstacleEffect()


@dataclass(frozen=True, slots=True)
class FixedSpecificEnergyLoss:
    specific_energy_loss_j_per_kg: float
    model_type: str = "fixed_specific_energy"

    def evaluate(self, context: ObstacleContext) -> ObstacleEffect:
        shape = raised_cosine_density(context.local_distance_m, context.interval_length_m)
        energy_j = self.specific_energy_loss_j_per_kg * context.vehicle_mass_kg
        return ObstacleEffect(resistance_force_n=energy_j * shape)


@dataclass(frozen=True, slots=True)
class SpeedQuadraticEnergyLoss:
    specific_fixed_energy_j_per_kg: float
    impact_coefficient_kg: float
    model_type: str = "speed_quadratic_energy"

    def evaluate(self, context: ObstacleContext) -> ObstacleEffect:
        shape = raised_cosine_density(context.local_distance_m, context.interval_length_m)
        energy_j = (
            self.specific_fixed_energy_j_per_kg * context.vehicle_mass_kg
            + self.impact_coefficient_kg * context.entry_speed_mps**2
        )
        return ObstacleEffect(resistance_force_n=energy_j * shape)


@dataclass(frozen=True, slots=True)
class DistributedResistance:
    resistance_force_n: float
    model_type: str = "distributed_resistance"

    def evaluate(self, context: ObstacleContext) -> ObstacleEffect:
        return ObstacleEffect(resistance_force_n=self.resistance_force_n)


@dataclass(frozen=True, slots=True)
class RoughnessEnergyDensity:
    specific_energy_per_distance_j_per_kg_m: float
    model_type: str = "roughness_energy_density"

    def evaluate(self, context: ObstacleContext) -> ObstacleEffect:
        return ObstacleEffect(
            resistance_force_n=(
                context.vehicle_mass_kg * self.specific_energy_per_distance_j_per_kg_m
            )
        )


@dataclass(frozen=True, slots=True)
class SmoothProfileObstacle:
    """Raised-cosine vertical profile with optional unresolved dissipation.

    z(x) = h/2 [1 - cos(2 pi x/L)]

    The signed slope contributes a conservative grade term.  The vertical
    curvature modifies normal load using the rigid-following estimate
    ``1 + v^2 z'' / g`` and is clipped to the declared bounds.  Unresolved
    suspension/soil work uses the same normalized raised-cosine density as the
    energy models.
    """

    vertical_amplitude_m: float
    specific_fixed_energy_j_per_kg: float
    impact_coefficient_kg: float
    traction_multiplier: float
    minimum_normal_load_scale: float
    maximum_normal_load_scale: float
    model_type: str = "smooth_profile"

    def evaluate(self, context: ObstacleContext) -> ObstacleEffect:
        length = context.interval_length_m
        x = min(max(context.local_distance_m, 0.0), length)
        phase = 2.0 * pi * x / length
        height = self.vertical_amplitude_m
        elevation = 0.5 * height * (1.0 - cos(phase))
        slope = height * pi / length * sin(phase)
        vertical_curvature = 2.0 * pi**2 * height / length**2 * cos(phase)
        raw_scale = 1.0 + context.vehicle_speed_mps**2 * vertical_curvature / context.gravity_mps2
        normal_scale = min(
            self.maximum_normal_load_scale,
            max(self.minimum_normal_load_scale, raw_scale),
        )
        energy_j = (
            self.specific_fixed_energy_j_per_kg * context.vehicle_mass_kg
            + self.impact_coefficient_kg * context.entry_speed_mps**2
        )
        return ObstacleEffect(
            resistance_force_n=energy_j * raised_cosine_density(x, length),
            elevation_offset_m=elevation,
            grade_slope_addition=slope,
            normal_load_scale=normal_scale,
            friction_multiplier=self.traction_multiplier,
        )


def _validate_realized_parameters(
    model_type: str, parameters: Mapping[str, float]
) -> None:
    for name, value in parameters.items():
        if not isfinite(value):
            raise SimulationInputError(
                f"Obstacle realization parameter {name!r} must be finite."
            )
        if not (model_type == "smooth_profile" and name == "vertical_amplitude") and value < 0.0:
            raise SimulationInputError(
                f"Obstacle realization parameter {name!r} cannot be negative."
            )
    if model_type == "smooth_profile" and (
        parameters["maximum_normal_load_scale"]
        < parameters["minimum_normal_load_scale"]
    ):
        raise SimulationInputError(
            "Obstacle realization has maximum_normal_load_scale below minimum_normal_load_scale."
        )


def raised_cosine_density(local_distance_m: float, length_m: float) -> float:
    """Return a smooth non-negative density integrating to one over ``[0, L]``."""

    if not isfinite(length_m) or length_m <= 0.0:
        raise SimulationInputError("Obstacle interval length must be positive and finite.")
    x = min(max(float(local_distance_m), 0.0), length_m)
    phase = 2.0 * pi * x / length_m
    return (1.0 - cos(phase)) / length_m


def obstacle_model_from_contract(
    raw: Mapping[str, Any],
    *,
    model_type_override: str | None = None,
    parameter_overrides_si: Mapping[str, float] | None = None,
) -> ObstacleModel:
    """Build one resolved model from a validated uncertainty declaration.

    Phase 5 calls this without overrides and receives the nominal branch. Phase 6
    may select a discrete model alternative and replace any sampled parameter by
    its SI-valued scenario draw.
    """

    try:
        choice, alternatives = obstacle_model_alternatives(raw)
    except ValueError as exc:
        raise SimulationInputError(str(exc)) from exc
    model_type = model_type_override or choice.nominal
    if model_type not in alternatives:
        raise SimulationInputError(
            f"Obstacle realization selected unavailable model {model_type!r}."
        )
    parsed = alternatives[model_type]
    nominal = {name: quantity.nominal_si()[0] for name, quantity in parsed.items()}
    if parameter_overrides_si:
        unknown = set(parameter_overrides_si) - set(nominal)
        if unknown:
            raise SimulationInputError(
                f"Obstacle realization supplied unknown parameters for {model_type!r}: {sorted(unknown)}."
            )
        nominal.update({name: float(value) for name, value in parameter_overrides_si.items()})
    _validate_realized_parameters(model_type, nominal)
    if model_type == "none":
        return NoObstacle()
    if model_type == "fixed_specific_energy":
        return FixedSpecificEnergyLoss(nominal["specific_energy_loss"])
    if model_type == "speed_quadratic_energy":
        return SpeedQuadraticEnergyLoss(
            nominal["specific_fixed_energy"], nominal["impact_coefficient"]
        )
    if model_type == "distributed_resistance":
        return DistributedResistance(nominal["resistance_force"])
    if model_type == "roughness_energy_density":
        return RoughnessEnergyDensity(nominal["specific_energy_per_distance"])
    if model_type == "smooth_profile":
        return SmoothProfileObstacle(
            vertical_amplitude_m=nominal["vertical_amplitude"],
            specific_fixed_energy_j_per_kg=nominal["specific_fixed_energy"],
            impact_coefficient_kg=nominal["impact_coefficient"],
            traction_multiplier=nominal["traction_multiplier"],
            minimum_normal_load_scale=nominal["minimum_normal_load_scale"],
            maximum_normal_load_scale=nominal["maximum_normal_load_scale"],
        )
    raise AssertionError(f"Unhandled obstacle model {model_type!r}")

