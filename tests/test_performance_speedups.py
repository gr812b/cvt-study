"""Regression tests for the performance-only transplant."""

from __future__ import annotations

from dataclasses import dataclass

from cvt_track_study.simulation.integrator import _record_ordered_feature_entry_crossings


@dataclass(frozen=True)
class _Interval:
    start_s_m: float


@dataclass(frozen=True)
class _Feature:
    identifier: str
    interval: _Interval


def test_ordered_feature_crossings_interpolate_and_advance_once() -> None:
    features = (
        _Feature("a", _Interval(2.0)),
        _Feature("b", _Interval(5.0)),
        _Feature("c", _Interval(12.0)),
    )
    recorded: dict[str, float] = {}
    index = _record_ordered_feature_entry_crossings(
        features=features,
        start_index=0,
        recorded=recorded,
        start_distance_m=0.0,
        end_distance_m=6.0,
        start_speed_mps=3.0,
        end_speed_mps=9.0,
    )
    assert index == 2
    assert recorded == {"a": 5.0, "b": 8.0}

    index = _record_ordered_feature_entry_crossings(
        features=features,
        start_index=index,
        recorded=recorded,
        start_distance_m=6.0,
        end_distance_m=13.0,
        start_speed_mps=9.0,
        end_speed_mps=16.0,
    )
    assert index == 3
    assert recorded["c"] == 15.0


def test_parallel_runtime_is_process_based() -> None:
    import cvt_track_study.studies.service_v8 as service
    import cvt_track_study.studies.ensemble_v10 as ensemble

    assert hasattr(service, "_initialize_scenario_process")
    assert hasattr(service, "_execute_scenario_process_initialized")
    assert "interruptible_process_pool" in service.__dict__
    assert "interruptible_process_pool" in ensemble.__dict__
