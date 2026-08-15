from __future__ import annotations

from pathlib import Path
import inspect

import numpy as np
import pytest

from cvt_track_study.simulation.traffic import (
    TrafficEvent,
    TrafficRealization,
    TrafficReferenceProfile,
    _parse_stream_times,
    traffic_limit_state,
)


def test_mixed_spreadsheet_timestamps_are_normalized() -> None:
    parsed = _parse_stream_times(
        ["0:18", "29:39:00", "1:01:33", "1:14", "1:14:39"]
    )
    assert parsed == pytest.approx([18.0, 1779.0, 3693.0, 4440.0, 4479.0])


def test_traffic_cap_uses_shared_reference_not_current_slowed_speed() -> None:
    reference = TrafficReferenceProfile((0.0, 100.0), (10.0, 10.0))
    event = TrafficEvent(
        identifier="traffic_01",
        start_offset_s=5.0,
        duration_s=4.0,
        retained_speed_fraction=0.4,
        event_type="slowdown",
        source_stream_id="source",
        source_row=2,
        source_confidence="high",
        baseline_speed_ordinal=3.0,
    )
    world = TrafficRealization(
        scenario_seed=7,
        lap_start_race_time_s=1000.0,
        race_duration_s=14400.0,
        initial_rate_per_hour=60.0,
        field_exponent=5.0,
        calibration_draw_index=0,
        events=(event,),
        model_fingerprint="test",
    )
    ceiling, retained = traffic_limit_state(
        traffic=world,
        reference=reference,
        lap_time_s=6.0,
        distance_m=50.0,
    )
    assert retained == pytest.approx(0.4)
    assert ceiling == pytest.approx(4.0)


def test_overlapping_events_take_the_more_restrictive_cap() -> None:
    reference = TrafficReferenceProfile((0.0, 100.0), (12.0, 12.0))
    common = dict(
        duration_s=10.0,
        event_type="slowdown",
        source_stream_id="source",
        source_row=2,
        source_confidence="medium",
        baseline_speed_ordinal=2.0,
    )
    events = (
        TrafficEvent("traffic_01", 0.0, retained_speed_fraction=0.7, **common),
        TrafficEvent("traffic_02", 2.0, retained_speed_fraction=0.3, **common),
    )
    world = TrafficRealization(1, 0.0, 14400.0, 60.0, 5.0, 0, events, "test")
    ceiling, retained = traffic_limit_state(
        traffic=world, reference=reference, lap_time_s=3.0, distance_m=20.0
    )
    assert retained == pytest.approx(0.3)
    assert ceiling == pytest.approx(3.6)


def test_patched_pipeline_exposes_traffic_contracts() -> None:
    # This is intentionally an integration-shape test: after applying the overlay,
    # these public/private callable boundaries must accept the traffic inputs used
    # by the study runner.
    from cvt_track_study.simulation.dynamics import evaluate_dynamics
    from cvt_track_study.simulation.integrator import run_simulation
    from cvt_track_study.uncertainty import ScenarioDraw

    assert "external_speed_ceiling_mps" in inspect.signature(evaluate_dynamics).parameters
    assert "traffic" in inspect.signature(run_simulation).parameters
    assert "traffic_reference" in inspect.signature(run_simulation).parameters
    assert "traffic_realization" in ScenarioDraw.__dataclass_fields__

    from cvt_track_study.studies.design_replay_v11 import _scenario_from_source

    replayed = _scenario_from_source(
        {
            "replicate": 9,
            "seed": 123,
            "sampling_mode": "all_declared",
            "quantity_values_si": {},
            "choice_values": {},
            "gate_target_speeds_mps": {},
            "independently_sampled_gate_ids": [],
            "traffic_realization": {"scenario_seed": 123, "events": []},
        },
        new_replicate=0,
        design_paths=(),
    )
    assert replayed.traffic_realization == {"scenario_seed": 123, "events": []}

    from cvt_track_study.studies.scenario_reduction import _feature_frame

    assert 'traffic:restriction_burden_s' in inspect.getsource(_feature_frame)


def test_maryland_calibration_regression() -> None:
    from cvt_track_study.simulation.traffic import traffic_model_from_project

    project = Path(__file__).resolve().parents[1] / "projects" / "maryland"
    model = traffic_model_from_project(project)
    assert model is not None
    assert len(model.observations) == 26
    assert len(model.source_observations) == 58
    assert model.initial_rate_per_hour == pytest.approx(32.4130378300, rel=2e-6)
    assert model.source_initial_rate_per_hour == pytest.approx(46.7968000417, rel=2e-6)
    assert model.field_exponent == pytest.approx(2.83504270215, rel=2e-6)
    source_rates = dict(model.source_stream_initial_rates_per_hour)
    assert source_rates["CWRU"] == pytest.approx(86.5582440272, rel=2e-6)
    assert source_rates["ETS"] == pytest.approx(38.8445112446, rel=2e-6)
    assert model.contract()["bootstrap_quantiles"]["field_exponent"][0] < 1.0e-3
    assert model.survival_fraction(3600.0) == pytest.approx(0.8428837888, rel=2e-6)
    assert model.survival_fraction(14400.0) == pytest.approx(0.4493856841, rel=2e-6)
    assert model.target_time_rescaling_ks_pvalue > 0.05
    assert model.time_rescaling_ks_pvalue > 0.05


def test_source_full_stops_and_target_stop_like_marks_are_normalized() -> None:
    from cvt_track_study.simulation.traffic import traffic_model_from_project

    project = Path(__file__).resolve().parents[1] / "projects" / "maryland"
    model = traffic_model_from_project(project)
    assert model is not None
    source_full_stops = [
        event for event in model.source_observations if event.event_type == "full stop"
    ]
    assert len(source_full_stops) == 3
    assert all(event.retained_speed_fraction == 0.0 for event in source_full_stops)
    assert sum(event.retained_speed_fraction == 0.0 for event in model.observations) == 2


def test_traffic_world_is_deterministic_for_scenario_seed() -> None:
    from cvt_track_study.simulation.traffic import traffic_model_from_project

    project = Path(__file__).resolve().parents[1] / "projects" / "maryland"
    model = traffic_model_from_project(project)
    assert model is not None
    first = model.draw_realization(scenario_seed=12345, horizon_s=300.0).serializable()
    second = model.draw_realization(scenario_seed=12345, horizon_s=300.0).serializable()
    assert first == second


def test_full_uncertainty_traffic_report_augmentation_is_idempotent(tmp_path: Path) -> None:
    import json
    import pandas as pd
    from cvt_track_study.reports.traffic import augment_full_uncertainty_traffic_report

    (tmp_path / "run_manifest.json").write_text(
        json.dumps(
            {
                "study_type": "full_uncertainty",
                "traffic_model": {
                    "enabled": True,
                    "target_event_count": 26,
                    "target_observer_exposure_h": 1.0,
                    "source_event_count": 58,
                    "source_observer_exposure_h": 1.5,
                    "field_record_count": 59,
                    "initial_rate_per_hour": 38.25,
                    "field_exponent": 5.2,
                    "average_rate_per_hour_4h": 11.82,
                },
            }
        ),
        encoding="utf-8",
    )
    traffic = {
        "scenario_seed": 1,
        "lap_start_race_time_s": 3600.0,
        "initial_rate_per_hour": 38.0,
        "field_exponent": 5.0,
        "calibration_draw_index": 2,
        "event_count": 1,
        "events": [
            {
                "id": "traffic_01",
                "start_offset_s": 5.0,
                "duration_s": 10.0,
                "retained_speed_fraction": 0.5,
                "event_type": "slowdown",
            }
        ],
    }
    (tmp_path / "scenario_draws.jsonl").write_text(
        json.dumps(
            {
                "replicate": 0,
                "base_draw_id": 0,
                "track_case_id": "route_001",
                "traffic_realization": traffic,
            }
        ) + "\n",
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "reference_traffic_penalty_s": 2.0,
                "bounded_traffic_active_time_s": 10.0,
                "bounded_traffic_event_count": 1.0,
                "bounded_traffic_minimum_retained_fraction": 0.5,
            }
        ]
    ).to_csv(tmp_path / "replicate_results.csv", index=False)
    report = tmp_path / "full_uncertainty_report.html"
    report.write_text("<html><body><h1>Existing report</h1></body></html>", encoding="utf-8")

    assert augment_full_uncertainty_traffic_report(tmp_path) == report
    assert augment_full_uncertainty_traffic_report(tmp_path) == report
    text = report.read_text(encoding="utf-8")
    assert text.count("<!-- TRAFFIC_AUDIT_SECTION -->") == 1
    assert text.count("<!-- /TRAFFIC_AUDIT_SECTION -->") == 1
    assert (tmp_path / "full_uncertainty_traffic_worlds.csv").is_file()
    assert (tmp_path / "full_uncertainty_traffic_events.csv").is_file()
    assert (tmp_path / "full_uncertainty_traffic_impact.csv").is_file()
