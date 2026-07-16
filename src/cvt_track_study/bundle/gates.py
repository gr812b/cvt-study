"""Empirical speed-gate contracts for track bundles."""

from __future__ import annotations

from typing import Any

from cvt_track_study.track.model import TrackBuildResult

from .serialization import circular, interval, split_tokens


def speed_gate_contracts(
    result: TrackBuildResult, length: float
) -> list[dict[str, Any]]:
    review = result.gate_review.sort_values("sequence")
    eligible = result.event_passes[result.event_passes["eligible"].astype(bool)]
    gates: list[dict[str, Any]] = []
    for _, row in review.iterrows():
        group_id = str(row["event_id"])
        samples_frame = eligible[
            eligible["event_id"].astype(str) == group_id
        ].dropna(subset=["entry_speed_mps"])
        samples = [
            {
                "lap_id": int(sample["lap_id"]),
                "run_id": str(sample["run_id"]),
                "vehicle_id": str(sample["vehicle_id"]),
                "driver_id": str(sample["driver_id"]),
                "value_mps": float(sample["entry_speed_mps"]),
            }
            for _, sample in samples_frame.sort_values(["run_id", "lap_id"]).iterrows()
        ]
        anchor = float(row["anchor_s_m"])
        position = circular(anchor + float(row["feature_start_rel_m"]), length)
        measurement_start = circular(anchor + float(row["entry_start_rel_m"]), length)
        measurement_end = circular(anchor + float(row["entry_end_rel_m"]), length)
        status = str(row["recommendation"])
        gates.append(
            {
                "id": f"gate:{group_id}",
                "gate_type": "entry_speed",
                "response_group_id": group_id,
                "name": str(row["event_name"]),
                "sequence": int(row["sequence"]),
                "status": status,
                "active_by_default": status == "accepted",
                "position_s_m": position,
                "position_semantics": "physical_feature_entry_boundary",
                "measurement_window": interval(measurement_start, measurement_end, length),
                "measurement_semantics": (
                    "median vehicle speed over the configured window immediately "
                    "before physical entry"
                ),
                "target_speed_distribution": {
                    "distribution": "empirical",
                    "unit": "m/s",
                    "sampling_unit": "eligible_lap_pass",
                    "samples": samples,
                    "summary": {
                        "sample_count": int(row["valid_pass_count"]),
                        "p10_mps": float(row["entry_speed_p10_mps"]),
                        "median_mps": float(row["entry_speed_median_mps"]),
                        "p90_mps": float(row["entry_speed_p90_mps"]),
                        "mean_mps": float(row["entry_speed_mean_mps"]),
                        "standard_deviation_mps": float(row["entry_speed_standard_deviation_mps"]),
                        "iqr_mps": float(row["entry_speed_iqr_mps"]),
                    },
                },
                "confidence": {
                    "overall_score": float(row["overall_confidence_score"]),
                    "pass_count_score": float(row["pass_count_score"]),
                    "speed_repeatability_score": float(row["speed_repeatability_score"]),
                    "braking_evidence_score": float(row["braking_evidence_score"]),
                    "pace_independence_score": float(row["pace_independence_score"]),
                    "coordinate_quality_score": float(row["coordinate_quality_score"]),
                    "cross_vehicle_agreement_score": float(row["cross_vehicle_agreement_score"]),
                    "cross_vehicle_status": str(row["cross_vehicle_status"]),
                    "reasons": split_tokens(row.get("reasons"), separator=";"),
                },
                "review_priority": int(row["review_priority"]),
                "suggested_action": str(row["suggested_action"]),
                "enforcement_contract": {
                    "policy": "one_way_speed_ceiling",
                    "slow_vehicle_reset_allowed": False,
                    "braking_envelope": "not_parameterized_until_vehicle_simulation_phase",
                },
            }
        )
        if bool(row.get("sustained_gate_qualified", False)):
            response_frame = eligible[
                eligible["event_id"].astype(str) == group_id
            ].dropna(subset=["event_min_speed_mps", "event_min_rel_m"])
            response_samples = [
                {
                    "lap_id": int(sample["lap_id"]),
                    "run_id": str(sample["run_id"]),
                    "vehicle_id": str(sample["vehicle_id"]),
                    "driver_id": str(sample["driver_id"]),
                    "value_mps": float(sample["event_min_speed_mps"]),
                    "location_rel_m": float(sample["event_min_rel_m"]),
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
                    "response_group_id": group_id,
                    "name": f"{row['event_name']} response minimum",
                    "sequence": int(row["sequence"]),
                    "status": "accepted",
                    "active_by_default": True,
                    "position_s_m": response_position,
                    "position_semantics": (
                        "median observed response-minimum location within the physical feature"
                    ),
                    "measurement_window": interval(feature_start, feature_end, length),
                    "measurement_semantics": (
                        "minimum vehicle speed within the declared physical-feature interval"
                    ),
                    "target_speed_distribution": {
                        "distribution": "empirical",
                        "unit": "m/s",
                        "sampling_unit": "eligible_lap_pass",
                        "samples": response_samples,
                        "summary": {
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
                    },
                    "confidence": {
                        "overall_score": float(row["sustained_confidence_score"]),
                        "slowdown_success_fraction": float(
                            row["sustained_slowdown_success_fraction"]
                        ),
                        "slowdown_p_value": float(row["sustained_slowdown_p_value"]),
                        "location_iqr_m": float(row["event_min_location_iqr_m"]),
                        "leave_one_out_max_speed_shift_mps": float(
                            row["sustained_leave_one_out_max_speed_shift_mps"]
                        ),
                        "leave_one_out_max_location_shift_m": float(
                            row["sustained_leave_one_out_max_location_shift_m"]
                        ),
                        "reasons": [str(row["sustained_gate_reason"])],
                    },
                    "review_priority": int(row["review_priority"]),
                    "suggested_action": str(row["suggested_action"]),
                    "enforcement_contract": {
                        "policy": "paired_entry_and_response_ceiling",
                        "paired_entry_gate_id": f"gate:{group_id}",
                        "slow_vehicle_reset_allowed": False,
                        "braking_envelope": "vehicle_simulation_parameter",
                    },
                }
            )
    return gates
