from __future__ import annotations

from pathlib import Path

import pandas as pd

from cvt_track_study.track.robustness import (
    _DEFAULT_THRESHOLDS,
    _aggregate_gate_stability,
    _write_track_ensemble_manifest,
)


def _nominal_gate_review() -> pd.DataFrame:
    frame = pd.DataFrame(
        [
            {
                "response_group_id": "E01",
                "event_name": "Core gate",
                "gate_candidate": True,
                "recommendation": "accepted",
                "overall_confidence_score": 70.0,
                "entry_speed_median_mps": 5.0,
                "braking_evidence_score": 80.0,
                "median_approach_to_min_slowdown_mps": 2.0,
                "slowdown_lap_fraction": 1.0,
                "pace_correlation": 0.1,
                "reasons": "",
                "suggested_action": "",
            },
            {
                "response_group_id": "E02",
                "event_name": "Conditional high-speed gate",
                "gate_candidate": True,
                "recommendation": "accepted",
                "overall_confidence_score": 60.2,
                "entry_speed_median_mps": 9.0,
                "braking_evidence_score": 20.0,
                "median_approach_to_min_slowdown_mps": 0.1,
                "slowdown_lap_fraction": 0.2,
                "pace_correlation": 0.8,
                "reasons": "weak braking",
                "suggested_action": "review video",
            },
            {
                "response_group_id": "E03",
                "event_name": "Near miss",
                "gate_candidate": True,
                "recommendation": "recommended_review",
                "overall_confidence_score": 58.0,
                "entry_speed_median_mps": 6.0,
                "braking_evidence_score": 55.0,
                "median_approach_to_min_slowdown_mps": 1.0,
                "slowdown_lap_fraction": 0.8,
                "pace_correlation": 0.2,
                "reasons": "",
                "suggested_action": "",
            },
            {
                "response_group_id": "start_finish",
                "event_name": "Start / finish",
                "gate_candidate": False,
                "recommendation": "not_a_candidate",
                "overall_confidence_score": 0.0,
                "entry_speed_median_mps": 10.0,
                "braking_evidence_score": 0.0,
                "median_approach_to_min_slowdown_mps": 0.0,
                "slowdown_lap_fraction": 0.0,
                "pace_correlation": 1.0,
                "reasons": "",
                "suggested_action": "",
            },
        ]
    )
    frame.attrs["accept_score"] = 60.0
    return frame


def _case_rows() -> pd.DataFrame:
    records = []
    outcomes = {
        "E01": [True, True, True, True],
        "E02": [True, False, False, True],
        "E03": [False, True, False, True],
        "start_finish": [False, False, False, False],
    }
    names = {
        "E01": "Core gate",
        "E02": "Conditional high-speed gate",
        "E03": "Near miss",
        "start_finish": "Start / finish",
    }
    nominal = {
        "E01": "accepted",
        "E02": "accepted",
        "E03": "recommended_review",
        "start_finish": "not_a_candidate",
    }
    for case_index, category in enumerate(
        ("gate_policy", "gate_weighting", "event_windows", "gate_policy")
    ):
        for gate_id, values in outcomes.items():
            accepted = values[case_index]
            recommendation = "accepted" if accepted else "recommended_review"
            records.append(
                {
                    "case_id": f"case_{case_index}",
                    "case_category": category,
                    "case_label": f"Case {case_index}",
                    "gate_id": gate_id,
                    "gate_name": names[gate_id],
                    "nominal_recommendation": nominal[gate_id],
                    "case_recommendation": recommendation,
                    "classification_matches_nominal": recommendation == nominal[gate_id],
                    "accepted": accepted,
                    "overall_confidence_score": 60.0 + case_index,
                    "entry_speed_median_mps": {"E01": 5.0, "E02": 9.0, "E03": 6.0, "start_finish": 10.0}[gate_id],
                }
            )
    return pd.DataFrame(records)


def test_gate_frontier_separates_core_conditional_near_miss_and_cap_review() -> None:
    result = _aggregate_gate_stability(
        _case_rows(), _DEFAULT_THRESHOLDS, _nominal_gate_review()
    ).set_index("gate_id")

    assert result.loc["E01", "frontier_classification"] == "core"
    assert result.loc["E02", "frontier_classification"] == "conditional"
    assert bool(result.loc["E02", "high_speed_weak_braking_review"])
    assert result.loc["E03", "frontier_classification"] == "near_miss"
    assert "start_finish" not in result.index


def test_ensemble_excludes_stress_only_and_length_unstable_cases(tmp_path: Path) -> None:
    cases = pd.DataFrame(
        [
            {
                "case_id": "gate_strict",
                "category": "gate_policy",
                "label": "Strict",
                "rationale": "stress",
                "success": True,
                "stress_only": True,
                "centreline_stable": True,
                "track_length_stable": True,
                "event_projection_stable": True,
                "gate_set_stable_for_ensemble": True,
                "gate_set_jaccard": 1.0,
                "nominal_gate_retention_fraction": 1.0,
                "newly_accepted_gate_count": 0,
            },
            {
                "case_id": "coarse",
                "category": "centreline",
                "label": "Coarse",
                "rationale": "spacing",
                "success": True,
                "stress_only": False,
                "centreline_stable": True,
                "track_length_stable": False,
                "event_projection_stable": True,
                "gate_set_stable_for_ensemble": True,
                "gate_set_jaccard": 1.0,
                "nominal_gate_retention_fraction": 1.0,
                "newly_accepted_gate_count": 0,
            },
            {
                "case_id": "cleanup",
                "category": "telemetry_cleanup",
                "label": "Cleanup",
                "rationale": "cleanup",
                "success": True,
                "stress_only": False,
                "centreline_stable": True,
                "track_length_stable": True,
                "event_projection_stable": True,
                "gate_set_stable_for_ensemble": True,
                "gate_set_jaccard": 0.9,
                "nominal_gate_retention_fraction": 0.9,
                "newly_accepted_gate_count": 1,
            },
        ]
    )
    manifest = _write_track_ensemble_manifest(tmp_path, cases, _DEFAULT_THRESHOLDS)

    assert "nominal" in manifest["eligible_cases"]
    assert "cleanup" in manifest["eligible_cases"]
    assert "gate_strict" not in manifest["eligible_cases"]
    assert "coarse" not in manifest["eligible_cases"]
