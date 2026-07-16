"""Small simulator-facing views derived only from a validated track bundle.

This module intentionally imports neither GPX nor reconstruction code. Phase 5 can
therefore migrate the vehicle simulator against these immutable structures while
keeping the evidence-producing pipeline behind the bundle boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .model import TrackBundle


@dataclass(frozen=True)
class TrackInterval:
    start_s_m: float
    end_s_m: float
    length_m: float
    wraps_start_finish: bool


@dataclass(frozen=True)
class SimulationFeature:
    identifier: str
    name: str
    sequence: int
    kind: str
    response_group_id: str
    interval: TrackInterval
    obstacle_model_status: str


@dataclass(frozen=True)
class SimulationSpeedGate:
    identifier: str
    response_group_id: str
    name: str
    sequence: int
    position_s_m: float
    empirical_speed_samples_mps: tuple[float, ...]
    p10_speed_mps: float
    median_speed_mps: float
    p90_speed_mps: float
    confidence_score: float


@dataclass(frozen=True)
class SimulationTrackInput:
    name: str
    length_m: float
    closed_course: bool
    centreline_s_m: tuple[float, ...]
    centreline_x_m: tuple[float, ...]
    centreline_y_m: tuple[float, ...]
    reference_elevation_m: tuple[float | None, ...]
    physical_features: tuple[SimulationFeature, ...]
    active_speed_gates: tuple[SimulationSpeedGate, ...]
    grade_force_enabled: bool
    obstacle_models_ready: bool
    uncertainty_roles_ready: bool

    @property
    def ready_for_full_vehicle_simulation(self) -> bool:
        return self.obstacle_models_ready


def simulation_track_from_bundle(bundle: TrackBundle) -> SimulationTrackInput:
    """Create the immutable Phase 5 input without consulting source project files."""

    data = bundle.data
    simulation = data["simulation_contract"]
    centreline = simulation["centreline"]["samples"]
    features = tuple(_feature(row) for row in simulation["physical_features"])
    gates = tuple(
        _gate(row)
        for row in simulation["speed_gates"]
        if bool(row["active_by_default"])
    )
    return SimulationTrackInput(
        name=str(data["identity"]["track_name"]),
        length_m=float(simulation["track_length_m"]),
        closed_course=bool(data["identity"]["closed_course"]),
        centreline_s_m=tuple(float(row["s_m"]) for row in centreline),
        centreline_x_m=tuple(float(row["x_m"]) for row in centreline),
        centreline_y_m=tuple(float(row["y_m"]) for row in centreline),
        reference_elevation_m=tuple(
            _optional_float(row.get("reference_elevation_m")) for row in centreline
        ),
        physical_features=features,
        active_speed_gates=gates,
        grade_force_enabled=bool(simulation["grade_force_enabled"]),
        obstacle_models_ready=bool(simulation["capabilities"]["obstacle_models_ready"]),
        uncertainty_roles_ready=bool(
            simulation["capabilities"]["uncertainty_roles_ready"]
        ),
    )


def _feature(row: Mapping[str, Any]) -> SimulationFeature:
    return SimulationFeature(
        identifier=str(row["id"]),
        name=str(row["name"]),
        sequence=int(row["sequence"]),
        kind=str(row["kind"]),
        response_group_id=str(row["response_group_id"]),
        interval=_interval(row["interval"]),
        obstacle_model_status=str(row["obstacle_model"]["status"]),
    )


def _gate(row: Mapping[str, Any]) -> SimulationSpeedGate:
    distribution = row["target_speed_distribution"]
    summary = distribution["summary"]
    return SimulationSpeedGate(
        identifier=str(row["id"]),
        response_group_id=str(row["response_group_id"]),
        name=str(row["name"]),
        sequence=int(row["sequence"]),
        position_s_m=float(row["position_s_m"]),
        empirical_speed_samples_mps=tuple(
            float(sample["value_mps"]) for sample in distribution["samples"]
        ),
        p10_speed_mps=float(summary["p10_mps"]),
        median_speed_mps=float(summary["median_mps"]),
        p90_speed_mps=float(summary["p90_mps"]),
        confidence_score=float(row["confidence"]["overall_score"]),
    )


def _interval(row: Mapping[str, Any]) -> TrackInterval:
    return TrackInterval(
        start_s_m=float(row["start_s_m"]),
        end_s_m=float(row["end_s_m"]),
        length_m=float(row["length_m"]),
        wraps_start_finish=bool(row["wraps_start_finish"]),
    )


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)
