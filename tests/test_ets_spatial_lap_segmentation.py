from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from cvt_track_study.gpx.lap_time_reconstruction import (
    _align_lap_times,
    _point_order_gate_crossings,
)


def test_segment_crossing_detects_gate_between_sparse_points():
    # About 22 m west and east of the gate. Neither point enters an 8 m circle,
    # but the connecting GPX segment crosses the gate exactly.
    points = pd.DataFrame(
        {
            "latitude_deg": [0.0, 0.0, 0.0, 0.0],
            "longitude_deg": [-0.0002, 0.0002, 0.01, -0.0002],
        }
    )
    crossings = _point_order_gate_crossings(
        points,
        gate_latitude_deg=0.0,
        gate_longitude_deg=0.0,
        gate_radius_m=8.0,
        minimum_points_between_crossings=1,
    )
    assert crossings[0] in {0, 1}


def test_strict_index_preserves_excluded_rows_in_sequence():
    detected = pd.DataFrame(
        {
            "detected_lap_index": [1, 2, 3, 4],
            "start_source_point_index": [0, 100, 200, 300],
            "end_source_point_index": [100, 200, 300, 400],
            "point_count": [101, 101, 101, 101],
            "point_interval_count": [100, 100, 100, 100],
        }
    )
    declared = pd.DataFrame(
        {
            "csv_row_index": [1, 2, 3],
            "csv_lap_index": [1, 2, 3],
            "csv_lap_value": ["1", "2", "3"],
            "csv_timestamp": ["", "", ""],
            "lap_time_s": [100.0, 900.0, 100.0],
            "csv_included": [True, False, True],
            "csv_exclusion_reason": ["", "pit", ""],
            "csv_metadata_json": ["{}", "{}", "{}"],
        }
    )
    config = SimpleNamespace(
        alignment_mode="strict_index",
        expected_point_period_s=1.0,
        maximum_alignment_error_fraction=0.65,
        skip_csv_lap_penalty=1.5,
        skip_detected_lap_penalty=1.0,
    )
    aligned = _align_lap_times(detected, declared, config)
    matched = aligned[aligned["alignment_status"] == "matched"]
    assert list(matched["csv_row_index"]) == [1, 2, 3]
    assert list(matched["detected_lap_index"]) == [1, 2, 3]
    assert bool(matched.iloc[1]["csv_included"]) is False
    assert (
        aligned["alignment_status"] == "unmatched_detected_lap"
    ).sum() == 1
