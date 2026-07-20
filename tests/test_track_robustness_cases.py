from cvt_track_study.track.robustness import build_robustness_cases


def test_track_robustness_covers_all_inference_layers() -> None:
    track = {
        "reconstruction": {
            "maximum_map_error_m": 20.0,
            "centreline_spacing_m": 3.0,
        },
        "centreline_consensus": {
            "smoothing_window_nodes": 5,
            "leave_one_out_p95_limit_m": 15.0,
            "sustained_error_threshold_m": 15.0,
        },
        "event_windows": {
            "approach_before_m": 30.0,
            "entry_before_m": 5.0,
            "exit_length_m": 15.0,
            "recovery_limit_m": 60.0,
        },
        "gate_confidence": {
            "minimum_valid_passes": 5,
            "target_pass_count": 10,
            "accept_score": 60.0,
            "review_score": 40.0,
            "braking_threshold_mps": 0.8,
        },
        "telemetry_cleanup": {
            "enabled": True,
            "maximum_excursion_points": 3,
            "minimum_excursion_leg_m": 35.0,
            "maximum_isolated_map_outlier_points": 3,
        },
    }
    runs = [
        {
            "run_id": "r1",
            "vehicle_id": "v1",
            "driver_id": "d1",
            "use_for_centreline": True,
            "use_for_gate_evidence": True,
        },
        {
            "run_id": "r2",
            "vehicle_id": "v1",
            "driver_id": "d1",
            "use_for_centreline": True,
            "use_for_gate_evidence": True,
        },
    ]
    cases = build_robustness_cases(track, runs, {"robustness": {}})
    categories = {case.category for case in cases}
    assert {
        "data_support",
        "gate_policy",
        "gate_weighting",
        "event_windows",
        "centreline",
        "telemetry_cleanup",
    } <= categories
    assert all(not hasattr(case, "vehicle_id") for case in cases)
