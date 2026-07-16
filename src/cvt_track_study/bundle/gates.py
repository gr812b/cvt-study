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
    return gates
