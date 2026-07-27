from __future__ import annotations

from pathlib import Path

import pytest

from cvt_track_study.gpx.run_config import metadata_from_run_config


def test_nested_run_reconstruction_config_is_optional_and_typed(tmp_path: Path):
    raw = {
        "file": "gpx/ets.gpx",
        "vehicle_id": "ets",
        "run_id": "ets_run",
        "driver_id": "driver",
        "use_for_centreline": True,
        "use_for_gate_evidence": True,
        "lap_time_reconstruction": {
            "enabled": True,
            "lap_times_file": "lap_times/ets.csv",
            "expected_point_period_s": 1.0,
            "alignment_mode": "cadence_dynamic_programming",
            "minimum_csv_lap_time_s": 180.0,
            "maximum_csv_lap_time_s": 420.0,
            "minimum_matched_laps": 20,
        },
    }
    metadata = metadata_from_run_config(raw, runs_directory=tmp_path)
    config = metadata.lap_time_reconstruction
    assert config is not None
    assert config.lap_times_file == (tmp_path / "lap_times/ets.csv").resolve()
    assert config.expected_point_period_s == 1.0
    assert config.alignment_mode == "cadence_dynamic_programming"
    assert config.maximum_csv_lap_time_s == 420.0
    assert config.minimum_matched_laps == 20


def test_invalid_alignment_mode_is_rejected(tmp_path: Path):
    raw = {
        "file": "gpx/ets.gpx",
        "vehicle_id": "ets",
        "run_id": "ets_run",
        "driver_id": "driver",
        "use_for_centreline": True,
        "use_for_gate_evidence": True,
        "lap_time_reconstruction": {
            "lap_times_file": "lap_times/ets.csv",
            "alignment_mode": "guess",
        },
    }
    with pytest.raises(ValueError, match="alignment_mode"):
        metadata_from_run_config(raw, runs_directory=tmp_path)
