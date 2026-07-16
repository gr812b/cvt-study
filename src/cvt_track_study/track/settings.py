"""Validated Phase 3 reconstruction settings."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ReconstructionSettings:
    lap_gate_event_id: str
    lap_gate_radius_m: float
    minimum_lap_time_s: float
    maximum_reasonable_speed_mps: float
    maximum_normal_time_step_s: float
    stationary_speed_mps: float
    minimum_speed_coverage_fraction: float
    centreline_spacing_m: float
    profile_spacing_m: float
    maximum_map_error_m: float
    speed_spike_threshold_mps: float
    approach_before_m: float
    approach_gap_m: float
    entry_before_m: float
    entry_gap_m: float
    exit_gap_m: float
    exit_length_m: float
    recovery_limit_m: float
    minimum_valid_passes: int
    target_pass_count: int
    braking_threshold_mps: float
    repeatability_scale_mps: float
    vehicle_agreement_scale_mps: float
    accept_score: float
    review_score: float
    weights: dict[str, float]

    @classmethod
    def from_mapping(cls, track: Mapping[str, Any]) -> "ReconstructionSettings":
        reconstruction = track.get("reconstruction", {})
        gates = track.get("gate_confidence", {})
        windows = track.get("event_windows", {})
        return cls(
            lap_gate_event_id=str(reconstruction.get("lap_gate_event_id", "")),
            lap_gate_radius_m=float(reconstruction.get("lap_gate_radius_m", 15.0)),
            minimum_lap_time_s=float(reconstruction.get("minimum_lap_time_s", 60.0)),
            maximum_reasonable_speed_mps=float(
                reconstruction.get("maximum_reasonable_speed_mps", 25.0)
            ),
            maximum_normal_time_step_s=float(
                reconstruction.get("maximum_normal_time_step_s", 3.0)
            ),
            stationary_speed_mps=float(reconstruction.get("stationary_speed_mps", 0.8)),
            minimum_speed_coverage_fraction=float(
                reconstruction.get("minimum_speed_coverage_fraction", 0.80)
            ),
            centreline_spacing_m=float(reconstruction.get("centreline_spacing_m", 3.0)),
            profile_spacing_m=float(reconstruction.get("profile_spacing_m", 5.0)),
            maximum_map_error_m=float(reconstruction.get("maximum_map_error_m", 20.0)),
            speed_spike_threshold_mps=float(
                reconstruction.get("speed_spike_threshold_mps", 4.0)
            ),
            approach_before_m=float(windows.get("approach_before_m", 35.0)),
            approach_gap_m=float(windows.get("approach_gap_m", 15.0)),
            entry_before_m=float(windows.get("entry_before_m", 8.0)),
            entry_gap_m=float(windows.get("entry_gap_m", 1.5)),
            exit_gap_m=float(windows.get("exit_gap_m", 3.0)),
            exit_length_m=float(windows.get("exit_length_m", 12.0)),
            recovery_limit_m=float(windows.get("recovery_limit_m", 60.0)),
            minimum_valid_passes=int(gates.get("minimum_valid_passes", 5)),
            target_pass_count=int(gates.get("target_pass_count", 10)),
            braking_threshold_mps=float(gates.get("braking_threshold_mps", 0.8)),
            repeatability_scale_mps=float(gates.get("repeatability_scale_mps", 2.0)),
            vehicle_agreement_scale_mps=float(
                gates.get("vehicle_agreement_scale_mps", 2.0)
            ),
            accept_score=float(gates.get("accept_score", 60.0)),
            review_score=float(gates.get("review_score", 40.0)),
            weights={
                "pass_count": float(gates.get("weight_pass_count", 0.15)),
                "speed_repeatability": float(
                    gates.get("weight_speed_repeatability", 0.25)
                ),
                "braking_evidence": float(
                    gates.get("weight_braking_evidence", 0.20)
                ),
                "pace_independence": float(
                    gates.get("weight_pace_independence", 0.15)
                ),
                "coordinate_quality": float(
                    gates.get("weight_coordinate_quality", 0.15)
                ),
                "cross_vehicle_agreement": float(
                    gates.get("weight_cross_vehicle_agreement", 0.10)
                ),
            },
        )
