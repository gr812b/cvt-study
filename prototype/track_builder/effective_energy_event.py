from __future__ import annotations

from dataclasses import dataclass
from math import cos, isfinite, pi

from .core import FeatureEffect, TrackEvaluationContext


@dataclass(frozen=True, slots=True)
class EffectiveEnergyEvent:
    """Uncalibrated effective event-loss seed distributed smoothly in space.

    ``specific_energy_loss_j_per_kg`` is multiplied by vehicle mass. This class
    exists to propagate low/nominal/high GPS-derived surrogate scenarios. The
    value must not be described as measured terrain dissipation unless separate
    propulsion, braking, grade, and terrain data support that interpretation.
    """

    name: str
    start_m: float
    length_m: float
    specific_energy_loss_j_per_kg: float
    model_status: str = "uncalibrated_effective_surrogate"

    def __post_init__(self) -> None:
        for field_name, value in (
            ("start_m", self.start_m),
            ("length_m", self.length_m),
            ("specific_energy_loss_j_per_kg", self.specific_energy_loss_j_per_kg),
        ):
            if not isfinite(value):
                raise ValueError(f"{field_name} must be finite")
        if self.start_m < 0.0 or self.length_m <= 0.0:
            raise ValueError("effective event start must be non-negative and length positive")
        if self.specific_energy_loss_j_per_kg < 0.0:
            raise ValueError("specific energy loss must be non-negative")

    @property
    def type_name(self) -> str:
        return "effective_energy_event"

    @property
    def end_m(self) -> float:
        return self.start_m + self.length_m

    def is_active(self, distance_m: float) -> bool:
        return self.start_m <= distance_m < self.end_m

    def evaluate(self, context: TrackEvaluationContext) -> FeatureEffect:
        if not self.is_active(context.distance_m):
            return FeatureEffect()
        local = context.distance_m - self.start_m
        phase = 2.0 * pi * local / self.length_m
        shape_1_per_m = (1.0 - cos(phase)) / self.length_m
        energy_j = self.specific_energy_loss_j_per_kg * context.vehicle_mass_kg
        return FeatureEffect(additional_resistance_force_n=energy_j * shape_1_per_m)

    def to_dict(self) -> dict[str, object]:
        return {
            "type": self.type_name,
            "name": self.name,
            "start_m": self.start_m,
            "length_m": self.length_m,
            "specific_energy_loss_j_per_kg": self.specific_energy_loss_j_per_kg,
            "model_status": self.model_status,
        }
