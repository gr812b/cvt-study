"""Empirical and conservative speed-gate contracts for track bundles."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from cvt_track_study.track.model import TrackBuildResult

from .serialization import circular, interval, split_tokens


def speed_gate_contracts(
    result: TrackBuildResult, length: float
) -> list[dict[str, Any]]:
    """Build three explicitly different gate contracts.

    Hard absolute and relative-response gates remain evidence-driven. Every other
    candidate receives only a permissive source-balanced guardrail so a simulated
    vehicle cannot exploit an evidence gap and pass an observed feature at an
    arbitrarily high speed. A guardrail is intentionally *not* represented as an
    accepted fitted feature speed.
    """

    review = result.gate_review.sort_values("sequence")
    eligible = result.event_passes[result.event_passes["eligible"].astype(bool)]
    gates: list[dict[str, Any]] = []

    for _, row in review.iterrows():
        group_id = str(row["event_id"])
        samples_frame = eligible[
            eligible["event_id"].astype(str) == group_id
        ].dropna(subset=["entry_speed_mps"])
        samples = [
            _sample_contract(sample, value_column="entry_speed_mps")
            for _, sample in samples_frame.sort_values(["run_id", "lap_id"]).iterrows()
        ]
        anchor = float(row["anchor_s_m"])
        position = circular(anchor + float(row["feature_start_rel_m"]), length)
        measurement_start = circular(anchor + float(row["entry_start_rel_m"]), length)
        measurement_end = circular(anchor + float(row["entry_end_rel_m"]), length)
        status = str(row["recommendation"])
        hard_absolute = bool(row.get("hard_absolute_gate_qualified", status == "accepted"))

        gates.append(
            {
                "id": f"gate:{group_id}",
                "gate_type": "entry_speed",
                "enforcement_class": "hard_absolute" if hard_absolute else "review_evidence",
                "response_group_id": group_id,
                "name": str(row["event_name"]),
                "sequence": int(row["sequence"]),
                "status": status,
                "active_by_default": hard_absolute,
                "position_s_m": position,
                "position_semantics": "physical_feature_entry_boundary",
                "measurement_window": interval(measurement_start, measurement_end, length),
                "measurement_semantics": (
                    "median vehicle speed over the configured window immediately "
                    "before physical entry"
                ),
                "target_speed_distribution": _empirical_distribution(
                    samples_frame,
                    value_column="entry_speed_mps",
                    samples=samples,
                    pooled_summary={
                        "sample_count": int(row["valid_pass_count"]),
                        "p10_mps": float(row["entry_speed_p10_mps"]),
                        "median_mps": float(row["entry_speed_median_mps"]),
                        "p90_mps": float(row["entry_speed_p90_mps"]),
                        "mean_mps": float(row["entry_speed_mean_mps"]),
                        "standard_deviation_mps": float(
                            row["entry_speed_standard_deviation_mps"]
                        ),
                        "iqr_mps": float(row["entry_speed_iqr_mps"]),
                    },
                ),
                "confidence": {
                    "classification": (
                        "hard_absolute" if hard_absolute else "review_evidence_only"
                    ),
                    "overall_score": float(row["overall_confidence_score"]),
                    "pass_count_score": float(row["pass_count_score"]),
                    "speed_repeatability_score": float(row["speed_repeatability_score"]),
                    "braking_evidence_score": float(row["braking_evidence_score"]),
                    "pace_independence_score": float(row["pace_independence_score"]),
                    "coordinate_quality_score": float(row["coordinate_quality_score"]),
                    "cross_vehicle_agreement_score": float(
                        row["cross_vehicle_agreement_score"]
                    ),
                    "cross_vehicle_status": str(row["cross_vehicle_status"]),
                    "entry_pace_ratio_iqr": _finite_or_none(
                        row.get("entry_pace_ratio_iqr")
                    ),
                    "vehicle_pace_ratio_spread": _finite_or_none(
                        row.get("vehicle_pace_ratio_spread")
                    ),
                    "speed_source_certainty": _certainty_summary(samples),
                    "reasons": split_tokens(row.get("reasons"), separator=";"),
                },
                "review_priority": int(row["review_priority"]),
                "suggested_action": str(row["suggested_action"]),
                "enforcement_contract": {
                    "policy": "one_way_speed_ceiling",
                    "target_vehicle_policy": "use_matching_vehicle_samples_when_available",
                    "slow_vehicle_reset_allowed": False,
                    "braking_envelope": "vehicle_simulation_parameter",
                },
            }
        )

        if bool(row.get("guardrail_active_fallback", False)):
            guardrail_cap = float(row["guardrail_cap_mps"])
            guardrail_samples = [
                _constant_guardrail_sample(sample, guardrail_cap)
                for _, sample in samples_frame.sort_values(["run_id", "lap_id"]).iterrows()
            ]
            # A guardrail is a deterministic safety policy, not a fitted observation.
            # Therefore it remains active even in a future sparse reconstruction with
            # no empirical pass identity; uncertainty simply has nothing to resample
            # for that deterministic ceiling.
            gates.append(
                    {
                        "id": f"gate:{group_id}:guardrail",
                        "gate_type": "entry_speed",
                        "enforcement_class": "conservative_guardrail",
                        "response_group_id": group_id,
                        "name": f"{row['event_name']} conservative guardrail",
                        "sequence": int(row["sequence"]),
                        "status": "accepted",
                        "active_by_default": True,
                        "position_s_m": position,
                        "position_semantics": "physical_feature_entry_boundary",
                        "measurement_window": interval(
                            measurement_start, measurement_end, length
                        ),
                        "measurement_semantics": (
                            "permissive source-balanced upper envelope; not a fitted "
                            "feature-speed estimate"
                        ),
                        "target_speed_distribution": {
                            "distribution": "empirical",
                            "unit": "m/s",
                            "sampling_unit": "eligible_lap_identity_fixed_guardrail",
                            "samples": guardrail_samples,
                            "summary": {
                                "sample_count": len(guardrail_samples),
                                "p10_mps": guardrail_cap,
                                "median_mps": guardrail_cap,
                                "p90_mps": guardrail_cap,
                                "mean_mps": guardrail_cap,
                                "standard_deviation_mps": 0.0,
                                "iqr_mps": 0.0,
                                **_certainty_summary(guardrail_samples),
                            },
                            "measurement_error_contract": {
                                "policy": "none_on_guardrail",
                                "reason": (
                                    "the cap already includes an upward conservatism margin; "
                                    "random perturbation must not make it artificially stricter"
                                ),
                            },
                        },
                        "confidence": {
                            "classification": "conservative_guardrail_not_feature_fit",
                            "overall_score": float(row["overall_confidence_score"]),
                            "guardrail_cap_mps": guardrail_cap,
                            "guardrail_policy": (
                                "maximum vehicle-specific empirical p95 entry speed + 1.5 m/s"
                            ),
                            "reasons": [
                                "hard absolute entry-speed evidence did not meet the acceptance bar",
                                "guardrail intentionally errs high to prevent unconstrained extreme speed",
                            ],
                        },
                        "review_priority": int(row["review_priority"]),
                        "suggested_action": str(row["suggested_action"]),
                        "enforcement_contract": {
                            "policy": "one_way_speed_ceiling",
                            "target_vehicle_policy": "source_upper_envelope_not_vehicle_specific",
                            "slow_vehicle_reset_allowed": False,
                            "braking_envelope": "vehicle_simulation_parameter",
                        },
                    }
            )

        if bool(row.get("sustained_gate_qualified", False)):
            response_frame = eligible[
                eligible["event_id"].astype(str) == group_id
            ].dropna(subset=["event_min_speed_mps", "event_min_rel_m"])
            response_samples = [
                {
                    **_sample_contract(sample, value_column="event_min_speed_mps"),
                    "location_rel_m": float(sample["event_min_rel_m"]),
                    "response_ratio": _response_ratio(sample),
                }
                for _, sample in response_frame.sort_values(
                    ["run_id", "lap_id"]
                ).iterrows()
            ]
            response_position = circular(
                anchor + float(row["median_event_min_rel_m"]), length
            )
            feature_start = circular(
                anchor + float(row["feature_start_rel_m"]), length
            )
            feature_end = circular(
                anchor + float(row["feature_end_rel_m"]), length
            )
            gates.append(
                {
                    "id": f"gate:{group_id}:response_minimum",
                    "gate_type": "sustained_response",
                    "enforcement_class": "relative_response",
                    "response_group_id": group_id,
                    "name": f"{row['event_name']} response minimum",
                    "sequence": int(row["sequence"]),
                    "status": "accepted",
                    "active_by_default": True,
                    "position_s_m": response_position,
                    "position_semantics": (
                        "source-balanced median observed response-minimum location "
                        "within the physical feature"
                    ),
                    "measurement_window": interval(feature_start, feature_end, length),
                    "measurement_semantics": (
                        "minimum vehicle speed within the declared physical-feature interval"
                    ),
                    "target_speed_distribution": _empirical_distribution(
                        response_frame,
                        value_column="event_min_speed_mps",
                        samples=response_samples,
                        pooled_summary={
                            "sample_count": len(response_samples),
                            "p10_mps": float(row["event_min_speed_p10_mps"]),
                            "median_mps": float(row["event_min_speed_median_mps"]),
                            "p90_mps": float(row["event_min_speed_p90_mps"]),
                            "mean_mps": float(row["event_min_speed_mean_mps"]),
                            "standard_deviation_mps": float(
                                row["event_min_speed_standard_deviation_mps"]
                            ),
                            "iqr_mps": float(row["event_min_speed_iqr_mps"]),
                        },
                    ),
                    "confidence": {
                        "classification": "relative_response",
                        "overall_score": float(row["sustained_confidence_score"]),
                        "slowdown_success_fraction": float(
                            row["sustained_slowdown_success_fraction"]
                        ),
                        "worst_source_slowdown_fraction": float(
                            row["sustained_worst_source_slowdown_fraction"]
                        ),
                        "worst_source_slowdown_p_value": float(
                            row["sustained_worst_source_p_value"]
                        ),
                        "response_ratio_median": float(row["response_ratio_median"]),
                        "response_ratio_worst_source_iqr": float(
                            row["response_ratio_worst_source_iqr"]
                        ),
                        "response_ratio_cross_vehicle_spread": float(
                            row["response_ratio_cross_vehicle_spread"]
                        ),
                        "location_iqr_m": float(row["event_min_location_iqr_m"]),
                        "leave_one_out_max_ratio_shift": float(
                            row["sustained_leave_one_out_max_ratio_shift"]
                        ),
                        "leave_one_out_max_location_shift_m": float(
                            row["sustained_leave_one_out_max_location_shift_m"]
                        ),
                        "speed_source_certainty": _certainty_summary(response_samples),
                        "reasons": [str(row["sustained_gate_reason"])],
                    },
                    "review_priority": int(row["review_priority"]),
                    "suggested_action": str(row["suggested_action"]),
                    "enforcement_contract": {
                        "policy": "one_way_speed_ceiling",
                        "evidence_semantics": (
                            "cross-vehicle qualification uses minimum/approach response ratio; "
                            "absolute enforcement uses matching target-vehicle samples"
                        ),
                        "target_vehicle_policy": "use_matching_vehicle_samples_when_available",
                        "slow_vehicle_reset_allowed": False,
                        "braking_envelope": "vehicle_simulation_parameter",
                    },
                }
            )
    return gates


def _empirical_distribution(
    frame: pd.DataFrame,
    *,
    value_column: str,
    samples: list[dict[str, Any]],
    pooled_summary: dict[str, Any],
) -> dict[str, Any]:
    summary = {
        **pooled_summary,
        **_certainty_summary(samples),
    }
    vehicle_summaries: dict[str, dict[str, Any]] = {}
    for vehicle_id, group in frame.groupby("vehicle_id", sort=False):
        values = pd.to_numeric(group[value_column], errors="coerce").dropna()
        if values.empty:
            continue
        vehicle_samples = [
            sample for sample in samples if str(sample.get("vehicle_id")) == str(vehicle_id)
        ]
        vehicle_summaries[str(vehicle_id)] = {
            "sample_count": len(values),
            "p10_mps": float(values.quantile(0.10)),
            "median_mps": float(values.median()),
            "p90_mps": float(values.quantile(0.90)),
            "mean_mps": float(values.mean()),
            "standard_deviation_mps": (
                float(values.std(ddof=1)) if len(values) > 1 else 0.0
            ),
            "iqr_mps": float(values.quantile(0.75) - values.quantile(0.25)),
            **_certainty_summary(vehicle_samples),
        }
    return {
        "distribution": "empirical",
        "unit": "m/s",
        "sampling_unit": "eligible_lap_pass",
        "samples": samples,
        "summary": summary,
        "vehicle_summaries": vehicle_summaries,
        "measurement_error_contract": {
            "native_fit_or_csv": "small error; speed_certainty=native_high",
            "lap_time_reconstructed": (
                "larger error sampled coherently across gates for the selected traversal"
            ),
            "negative_speed_policy": "truncate_at_zero",
        },
    }


def _sample_contract(sample: Any, *, value_column: str) -> dict[str, Any]:
    certainty = str(sample.get("speed_certainty", "unavailable"))
    sigma = sample.get("speed_measurement_standard_deviation_mps", 0.75)
    weight = sample.get("speed_evidence_weight", 0.5)
    return {
        "lap_id": int(sample["lap_id"]),
        "run_id": str(sample["run_id"]),
        "vehicle_id": str(sample["vehicle_id"]),
        "driver_id": str(sample["driver_id"]),
        "value_mps": float(sample[value_column]),
        "speed_certainty": certainty,
        "evidence_weight": float(weight),
        "measurement_standard_deviation_mps": float(sigma),
    }


def _constant_guardrail_sample(sample: Any, cap_mps: float) -> dict[str, Any]:
    return {
        "lap_id": int(sample["lap_id"]),
        "run_id": str(sample["run_id"]),
        "vehicle_id": str(sample["vehicle_id"]),
        "driver_id": str(sample["driver_id"]),
        "value_mps": float(cap_mps),
        "speed_certainty": "conservative_guardrail",
        "evidence_weight": 1.0,
        "measurement_standard_deviation_mps": 0.0,
    }


def _response_ratio(sample: Any) -> float | None:
    approach = float(sample.get("approach_speed_mps", math.nan))
    minimum = float(sample.get("event_min_speed_mps", math.nan))
    if not np.isfinite(approach) or not np.isfinite(minimum) or approach <= 0.5:
        return None
    return max(0.0, minimum / approach)


def _certainty_summary(samples: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    sigmas: list[float] = []
    for sample in samples:
        certainty = str(sample.get("speed_certainty", "unavailable"))
        counts[certainty] = counts.get(certainty, 0) + 1
        sigmas.append(float(sample.get("measurement_standard_deviation_mps", 0.75)))
    return {
        "speed_certainty_counts": counts,
        "mean_measurement_standard_deviation_mps": (
            sum(sigmas) / len(sigmas) if sigmas else 0.0
        ),
        "reconstructed_sample_count": sum(
            count for label, count in counts.items() if label.startswith("reconstructed_")
        ),
    }


def _finite_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None
