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

    consensus_minimum_laps: int
    consensus_maximum_iterations: int
    consensus_convergence_tolerance_m: float
    consensus_smoothing_window_nodes: int
    consensus_leave_one_out_p95_limit_m: float
    consensus_sustained_error_threshold_m: float
    consensus_minimum_sustained_outlier_fraction: float
    consensus_strong_sustained_outlier_fraction: float
    consensus_maximum_leave_one_out_shift_m: float
    consensus_robust_mad_multiplier: float
    consensus_maximum_outlier_fraction_per_iteration: float

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
    def from_mapping(
        cls, track: Mapping[str, Any]
    ) -> "ReconstructionSettings":
        reconstruction = _mapping(track.get("reconstruction"))
        consensus = _mapping(track.get("centreline_consensus"))
        gates = _mapping(track.get("gate_confidence"))
        windows = _mapping(track.get("event_windows"))

        maximum_map_error_m = _positive_float(
            reconstruction.get("maximum_map_error_m"), 20.0
        )
        minimum_sustained = _fraction(
            consensus.get("minimum_sustained_outlier_fraction"), 0.05
        )
        strong_sustained = _fraction(
            consensus.get("strong_sustained_outlier_fraction"), 0.15
        )
        if strong_sustained < minimum_sustained:
            raise ValueError(
                "track.centreline_consensus."
                "strong_sustained_outlier_fraction must be at least "
                "minimum_sustained_outlier_fraction"
            )

        smoothing_window = _positive_int(
            consensus.get("smoothing_window_nodes"), 5
        )
        if smoothing_window % 2 == 0:
            raise ValueError(
                "track.centreline_consensus.smoothing_window_nodes "
                "must be odd"
            )

        return cls(
            lap_gate_event_id=str(
                reconstruction.get("lap_gate_event_id", "")
            ),
            lap_gate_radius_m=_positive_float(
                reconstruction.get("lap_gate_radius_m"), 15.0
            ),
            minimum_lap_time_s=_positive_float(
                reconstruction.get("minimum_lap_time_s"), 60.0
            ),
            maximum_reasonable_speed_mps=_positive_float(
                reconstruction.get("maximum_reasonable_speed_mps"), 25.0
            ),
            maximum_normal_time_step_s=_positive_float(
                reconstruction.get("maximum_normal_time_step_s"), 3.0
            ),
            stationary_speed_mps=_nonnegative_float(
                reconstruction.get("stationary_speed_mps"), 0.8
            ),
            minimum_speed_coverage_fraction=_fraction(
                reconstruction.get("minimum_speed_coverage_fraction"), 0.80
            ),
            centreline_spacing_m=_positive_float(
                reconstruction.get("centreline_spacing_m"), 3.0
            ),
            profile_spacing_m=_positive_float(
                reconstruction.get("profile_spacing_m"), 5.0
            ),
            maximum_map_error_m=maximum_map_error_m,
            speed_spike_threshold_mps=_positive_float(
                reconstruction.get("speed_spike_threshold_mps"), 4.0
            ),
            consensus_minimum_laps=_positive_int(
                consensus.get("minimum_laps"), 3
            ),
            consensus_maximum_iterations=_positive_int(
                consensus.get("maximum_iterations"), 6
            ),
            consensus_convergence_tolerance_m=_positive_float(
                consensus.get("convergence_tolerance_m"), 0.10
            ),
            consensus_smoothing_window_nodes=smoothing_window,
            consensus_leave_one_out_p95_limit_m=_positive_float(
                consensus.get("leave_one_out_p95_limit_m"),
                0.75 * maximum_map_error_m,
            ),
            consensus_sustained_error_threshold_m=_positive_float(
                consensus.get("sustained_error_threshold_m"),
                0.75 * maximum_map_error_m,
            ),
            consensus_minimum_sustained_outlier_fraction=minimum_sustained,
            consensus_strong_sustained_outlier_fraction=strong_sustained,
            consensus_maximum_leave_one_out_shift_m=_positive_float(
                consensus.get("maximum_leave_one_out_shift_m"), 5.0
            ),
            consensus_robust_mad_multiplier=_positive_float(
                consensus.get("robust_mad_multiplier"), 3.5
            ),
            consensus_maximum_outlier_fraction_per_iteration=_fraction(
                consensus.get(
                    "maximum_outlier_fraction_per_iteration"
                ),
                0.20,
            ),
            approach_before_m=_positive_float(
                windows.get("approach_before_m"), 35.0
            ),
            approach_gap_m=_nonnegative_float(
                windows.get("approach_gap_m"), 15.0
            ),
            entry_before_m=_positive_float(
                windows.get("entry_before_m"), 8.0
            ),
            entry_gap_m=_nonnegative_float(
                windows.get("entry_gap_m"), 1.5
            ),
            exit_gap_m=_nonnegative_float(
                windows.get("exit_gap_m"), 3.0
            ),
            exit_length_m=_positive_float(
                windows.get("exit_length_m"), 12.0
            ),
            recovery_limit_m=_positive_float(
                windows.get("recovery_limit_m"), 60.0
            ),
            minimum_valid_passes=_positive_int(
                gates.get("minimum_valid_passes"), 5
            ),
            target_pass_count=_positive_int(
                gates.get("target_pass_count"), 10
            ),
            braking_threshold_mps=_nonnegative_float(
                gates.get("braking_threshold_mps"), 0.8
            ),
            repeatability_scale_mps=_positive_float(
                gates.get("repeatability_scale_mps"), 2.0
            ),
            vehicle_agreement_scale_mps=_positive_float(
                gates.get("vehicle_agreement_scale_mps"), 2.0
            ),
            accept_score=float(gates.get("accept_score", 60.0)),
            review_score=float(gates.get("review_score", 40.0)),
            weights={
                "pass_count": float(
                    gates.get("weight_pass_count", 0.15)
                ),
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
                    gates.get(
                        "weight_cross_vehicle_agreement", 0.10
                    )
                ),
            },
        )


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _positive_float(value: Any, default: float) -> float:
    number = default if value is None else float(value)
    if not math_is_finite_positive(number):
        raise ValueError(
            f"Expected a positive finite setting, got {value!r}"
        )
    return number


def _nonnegative_float(value: Any, default: float) -> float:
    number = default if value is None else float(value)
    if not _finite(number) or number < 0:
        raise ValueError(
            f"Expected a nonnegative finite setting, got {value!r}"
        )
    return number


def _positive_int(value: Any, default: int) -> int:
    number = default if value is None else int(value)
    if number <= 0:
        raise ValueError(
            f"Expected a positive integer setting, got {value!r}"
        )
    return number


def _fraction(value: Any, default: float) -> float:
    number = default if value is None else float(value)
    if not _finite(number) or not 0 < number <= 1:
        raise ValueError(
            f"Expected a fraction in (0, 1], got {value!r}"
        )
    return number


def math_is_finite_positive(number: float) -> bool:
    return _finite(number) and number > 0


def _finite(number: float) -> bool:
    return number == number and number not in {
        float("inf"),
        float("-inf"),
    }
