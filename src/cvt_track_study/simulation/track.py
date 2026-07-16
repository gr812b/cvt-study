"""Immutable runtime track assembled only from a validated track bundle."""

from __future__ import annotations

from dataclasses import dataclass
from math import atan, degrees, isfinite, sqrt
from typing import Any, Mapping

import numpy as np

from cvt_track_study.bundle import TrackBundle

from .models import SimulationInputError
from .obstacles import ObstacleContext, ObstacleModel, obstacle_model_from_contract


@dataclass(frozen=True, slots=True)
class RuntimeInterval:
    start_s_m: float
    end_s_m: float
    length_m: float
    wraps_start_finish: bool

    def local_distance(self, s_m: float, track_length_m: float) -> float | None:
        s = float(s_m) % track_length_m
        if not self.wraps_start_finish:
            if self.start_s_m <= s < self.end_s_m:
                return s - self.start_s_m
            return None
        if s >= self.start_s_m:
            return s - self.start_s_m
        if s < self.end_s_m:
            return track_length_m - self.start_s_m + s
        return None


@dataclass(frozen=True, slots=True)
class RuntimeFeature:
    identifier: str
    name: str
    response_group_id: str
    interval: RuntimeInterval
    model: ObstacleModel


@dataclass(frozen=True, slots=True)
class RuntimeSpeedGate:
    identifier: str
    response_group_id: str
    name: str
    position_s_m: float
    target_speed_mps: float
    confidence_score: float


@dataclass(frozen=True, slots=True)
class TrackSample:
    s_m: float
    reference_elevation_m: float | None
    modeled_elevation_offset_m: float
    modeled_grade_degrees: float
    curvature_1_per_m: float
    friction_coefficient: float
    normal_load_scale: float
    obstacle_resistance_force_n: float
    feature_resistance_forces_n: tuple[tuple[str, float], ...]
    active_feature_ids: tuple[str, ...]
    active_feature_names: tuple[str, ...]

    @property
    def modeled_grade_radians(self) -> float:
        return np.deg2rad(self.modeled_grade_degrees).item()


@dataclass(frozen=True, slots=True)
class RuntimeTrack:
    name: str
    length_m: float
    closed_course: bool
    centreline_s_m: tuple[float, ...]
    centreline_x_m: tuple[float, ...]
    centreline_y_m: tuple[float, ...]
    centreline_curvature_1_per_m: tuple[float, ...]
    reference_elevation_m: tuple[float | None, ...]
    surface_friction_coefficient: float
    features: tuple[RuntimeFeature, ...]
    speed_gates: tuple[RuntimeSpeedGate, ...]
    gpx_grade_force_enabled: bool

    def __post_init__(self) -> None:
        if not self.closed_course:
            raise SimulationInputError("Phase 5 supports closed-course track bundles only.")
        if not isfinite(self.length_m) or self.length_m <= 0.0:
            raise SimulationInputError("Track length must be positive and finite.")
        lengths = {len(self.centreline_s_m), len(self.centreline_x_m), len(self.centreline_y_m), len(self.centreline_curvature_1_per_m), len(self.reference_elevation_m)}
        if len(lengths) != 1:
            raise SimulationInputError("Centreline distance, geometry, curvature, and elevation arrays must align.")
        if len(self.centreline_s_m) < 2:
            raise SimulationInputError("Track runtime requires at least two centreline samples.")
        if self.surface_friction_coefficient <= 0.0:
            raise SimulationInputError("Surface friction coefficient must be positive.")


    def sample(
        self,
        distance_m: float,
        *,
        vehicle_speed_mps: float,
        vehicle_mass_kg: float,
        gravity_mps2: float,
        feature_entry_speeds_mps: Mapping[str, float] | None = None,
    ) -> TrackSample:
        s = min(max(float(distance_m), 0.0), self.length_m)
        loop_s = 0.0 if s >= self.length_m else s
        reference_elevation = _interpolate_optional(
            loop_s, self.centreline_s_m, self.reference_elevation_m
        )
        curvature = float(np.interp(loop_s, self.centreline_s_m, self.centreline_curvature_1_per_m))
        resistance = 0.0
        elevation_offset = 0.0
        slope = 0.0
        normal_scale = 1.0
        friction = self.surface_friction_coefficient
        active_ids: list[str] = []
        active_names: list[str] = []
        feature_forces: list[tuple[str, float]] = []
        entry_speeds = feature_entry_speeds_mps or {}
        for feature in self.features:
            local = feature.interval.local_distance(loop_s, self.length_m)
            if local is None:
                continue
            effect = feature.model.evaluate(
                ObstacleContext(
                    local_distance_m=local,
                    interval_length_m=feature.interval.length_m,
                    vehicle_speed_mps=max(0.0, vehicle_speed_mps),
                    entry_speed_mps=max(
                        0.0,
                        float(entry_speeds.get(feature.identifier, vehicle_speed_mps)),
                    ),
                    vehicle_mass_kg=vehicle_mass_kg,
                    gravity_mps2=gravity_mps2,
                )
            )
            feature_force = max(0.0, effect.resistance_force_n)
            resistance += feature_force
            feature_forces.append((feature.identifier, feature_force))
            elevation_offset += effect.elevation_offset_m
            slope += effect.grade_slope_addition
            normal_scale *= max(0.0, effect.normal_load_scale)
            friction *= max(0.0, effect.friction_multiplier)
            active_ids.append(feature.identifier)
            active_names.append(feature.name)
        return TrackSample(
            s_m=loop_s,
            reference_elevation_m=reference_elevation,
            modeled_elevation_offset_m=elevation_offset,
            modeled_grade_degrees=degrees(atan(slope)),
            curvature_1_per_m=curvature,
            friction_coefficient=max(0.0, friction),
            normal_load_scale=max(0.0, normal_scale),
            obstacle_resistance_force_n=max(0.0, resistance),
            feature_resistance_forces_n=tuple(feature_forces),
            active_feature_ids=tuple(active_ids),
            active_feature_names=tuple(active_names),
        )

    def safe_speed_ceiling_mps(
        self, distance_m: float, *, braking_deceleration_mps2: float
    ) -> float:
        """Backward-propagate every accepted gate through a finite braking envelope."""

        if braking_deceleration_mps2 <= 0.0:
            raise SimulationInputError("braking_deceleration_mps2 must be positive.")
        s = min(max(float(distance_m), 0.0), self.length_m)
        loop_s = 0.0 if s >= self.length_m else s
        if not self.speed_gates:
            return float("inf")
        result = float("inf")
        for gate in self.speed_gates:
            ahead = (gate.position_s_m - loop_s) % self.length_m
            safe = sqrt(
                max(
                    gate.target_speed_mps**2
                    + 2.0 * braking_deceleration_mps2 * ahead,
                    0.0,
                )
            )
            result = min(result, safe)
        return result



def runtime_track_from_bundle(
    bundle: TrackBundle,
    *,
    surface_friction_coefficient: float,
    gate_speed_statistic: str = "median",
    gate_target_speeds_mps: Mapping[str, float] | None = None,
    obstacle_model_types: Mapping[str, str] | None = None,
    obstacle_parameters_si: Mapping[str, Mapping[str, float]] | None = None,
) -> RuntimeTrack:
    """Resolve one nominal or sampled runtime track from a validated bundle."""

    if gate_speed_statistic not in {"p10", "median", "p90"}:
        raise SimulationInputError("gate_speed_statistic must be p10, median, or p90.")
    simulation = bundle.data["simulation_contract"]
    centreline = simulation["centreline"]["samples"]
    centreline_s = tuple(float(row["s_m"]) for row in centreline)
    centreline_x = tuple(float(row["x_m"]) for row in centreline)
    centreline_y = tuple(float(row["y_m"]) for row in centreline)
    centreline_curvature = _closed_curvature(centreline_s, centreline_x, centreline_y)
    features: list[RuntimeFeature] = []
    for raw in simulation["physical_features"]:
        interval = raw["interval"]
        features.append(
            RuntimeFeature(
                identifier=str(raw["id"]),
                name=str(raw["name"]),
                response_group_id=str(raw["response_group_id"]),
                interval=RuntimeInterval(
                    start_s_m=float(interval["start_s_m"]),
                    end_s_m=float(interval["end_s_m"]),
                    length_m=float(interval["length_m"]),
                    wraps_start_finish=bool(interval["wraps_start_finish"]),
                ),
                model=obstacle_model_from_contract(
                    raw["obstacle_model"],
                    model_type_override=(obstacle_model_types or {}).get(str(raw["id"])),
                    parameter_overrides_si=(obstacle_parameters_si or {}).get(str(raw["id"])),
                ),
            )
        )
    gate_overrides = gate_target_speeds_mps or {}
    speed_key = {
        "p10": "p10_mps",
        "median": "median_mps",
        "p90": "p90_mps",
    }[gate_speed_statistic]
    gates: list[RuntimeSpeedGate] = []
    for raw in simulation["speed_gates"]:
        if not bool(raw["active_by_default"]):
            continue
        summary = raw["target_speed_distribution"]["summary"]
        gates.append(
            RuntimeSpeedGate(
                identifier=str(raw["id"]),
                response_group_id=str(raw["response_group_id"]),
                name=str(raw["name"]),
                position_s_m=float(raw["position_s_m"]),
                target_speed_mps=float(
                    gate_overrides.get(str(raw["id"]), summary[speed_key])
                ),
                confidence_score=float(raw["confidence"]["overall_score"]),
            )
        )
    capabilities = simulation["capabilities"]
    if not bool(capabilities.get("obstacle_models_ready")):
        raise SimulationInputError(
            "Track bundle does not declare complete obstacle models. Rebuild it with Phase 5 event profiles."
        )
    return RuntimeTrack(
        name=str(bundle.data["identity"]["track_name"]),
        length_m=float(simulation["track_length_m"]),
        closed_course=bool(bundle.data["identity"]["closed_course"]),
        centreline_s_m=centreline_s,
        centreline_x_m=centreline_x,
        centreline_y_m=centreline_y,
        centreline_curvature_1_per_m=centreline_curvature,
        reference_elevation_m=tuple(
            None if row.get("reference_elevation_m") is None else float(row["reference_elevation_m"])
            for row in centreline
        ),
        surface_friction_coefficient=surface_friction_coefficient,
        features=tuple(features),
        speed_gates=tuple(sorted(gates, key=lambda gate: gate.position_s_m)),
        gpx_grade_force_enabled=bool(simulation["grade_force_enabled"]),
    )


def _interpolate_optional(
    x: float, sample_x: tuple[float, ...], sample_y: tuple[float | None, ...]
) -> float | None:
    valid = [(sx, sy) for sx, sy in zip(sample_x, sample_y) if sy is not None]
    if len(valid) < 2:
        return None
    xs = np.asarray([item[0] for item in valid], dtype=float)
    ys = np.asarray([item[1] for item in valid], dtype=float)
    return float(np.interp(x, xs, ys))


def _closed_curvature(
    s_m: tuple[float, ...], x_m: tuple[float, ...], y_m: tuple[float, ...]
) -> tuple[float, ...]:
    """Estimate signed planar curvature from the published centreline samples."""

    s = np.asarray(s_m, dtype=float)
    x = np.asarray(x_m, dtype=float)
    y = np.asarray(y_m, dtype=float)
    edge_order = 2 if len(s) >= 3 else 1
    dx = np.gradient(x, s, edge_order=edge_order)
    dy = np.gradient(y, s, edge_order=edge_order)
    ddx = np.gradient(dx, s, edge_order=edge_order)
    ddy = np.gradient(dy, s, edge_order=edge_order)
    denominator = np.maximum((dx * dx + dy * dy) ** 1.5, 1.0e-12)
    curvature = (dx * ddy - dy * ddx) / denominator
    curvature[~np.isfinite(curvature)] = 0.0
    return tuple(float(value) for value in curvature)
