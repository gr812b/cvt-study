"""Coherent course-wide obstacle energy severity applied inside simulation."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import isfinite
from typing import Any, Mapping

from .models import SimulationInputError, quantity_nominal
from .obstacles import ObstacleContext, ObstacleEffect, ObstacleModel


GLOBAL_OBSTACLE_SEVERITY_PATH = "track.obstacle_energy_severity"


@dataclass(frozen=True, slots=True)
class EnergyScaledObstacle:
    """Scale only dissipative obstacle work, leaving geometry/traction untouched."""

    base: ObstacleModel
    energy_scale: float

    @property
    def model_type(self) -> str:
        return f"{self.base.model_type}:energy_scale={self.energy_scale:g}"

    def evaluate(self, context: ObstacleContext) -> ObstacleEffect:
        effect = self.base.evaluate(context)
        return replace(
            effect,
            resistance_force_n=max(
                0.0, float(effect.resistance_force_n) * self.energy_scale
            ),
        )


def scale_runtime_track(track: Any, energy_scale: float) -> Any:
    scale = float(energy_scale)
    if not isfinite(scale) or scale <= 0.0:
        raise SimulationInputError(
            "track.obstacle_energy_severity must be positive and finite."
        )
    if abs(scale - 1.0) <= 1.0e-15:
        return track
    features = tuple(
        replace(
            feature,
            model=EnergyScaledObstacle(feature.model, scale),
        )
        for feature in track.features
    )
    return replace(track, features=features)


def install_obstacle_severity_patch(service_module: Any) -> None:
    """Wrap the canonical resolver once, before study modules import it."""

    original = service_module.resolve_simulation_cases
    if getattr(original, "_obstacle_severity_patch", False):
        return

    def resolved_with_obstacle_severity(*args: Any, **kwargs: Any) -> Any:
        study_raw = kwargs.get("study_raw")
        if study_raw is None and len(args) >= 3:
            study_raw = args[2]
        study_raw = study_raw if isinstance(study_raw, Mapping) else {}

        scenario = dict(kwargs.get("quantity_values_si") or {})
        design = dict(kwargs.get("design_values_si") or {})
        values = {**scenario, **design}
        scale = values.get(GLOBAL_OBSTACLE_SEVERITY_PATH)
        if scale is None:
            realization = study_raw.get("track_realization", {})
            raw = (
                realization.get("obstacle_energy_severity")
                if isinstance(realization, Mapping)
                else None
            )
            if isinstance(raw, Mapping) and "nominal" in raw:
                scale = quantity_nominal(raw)
            else:
                scale = 1.0

        bounded, reference, settings, track = original(*args, **kwargs)
        return bounded, reference, settings, scale_runtime_track(track, float(scale))

    resolved_with_obstacle_severity._obstacle_severity_patch = True
    resolved_with_obstacle_severity.__name__ = original.__name__
    resolved_with_obstacle_severity.__doc__ = original.__doc__
    service_module.resolve_simulation_cases = resolved_with_obstacle_severity
