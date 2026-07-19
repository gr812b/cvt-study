from __future__ import annotations

from pathlib import Path

import pandas as pd

from cvt_track_study.track.review import (
    _split_interval_for_plot,
    build_event_interval_audit,
    create_event_group_timeline,
    write_review_html,
)


def _features() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "sequence": 1,
                "response_group_id": "normal",
                "name": "Normal interval",
                "source_event_ids": "normal",
                "source_event_names": "Normal interval",
                "analysis_feature_type": "individual",
                "anchor_s_m": 100.0,
                "feature_start_rel_m": 0.0,
                "feature_end_rel_m": 30.0,
                "review_flags": "",
            },
            {
                "sequence": 2,
                "response_group_id": "wrap",
                "name": "Start/finish interval",
                "source_event_ids": "wrap",
                "source_event_names": "Start/finish interval",
                "analysis_feature_type": "individual",
                "anchor_s_m": 980.0,
                "feature_start_rel_m": 0.0,
                "feature_end_rel_m": 40.0,
                "review_flags": "",
            },
            {
                "sequence": 3,
                "response_group_id": "bad",
                "name": "Nearly full-lap interval",
                "source_event_ids": "bad",
                "source_event_names": "Nearly full-lap interval",
                "analysis_feature_type": "individual",
                "anchor_s_m": 300.0,
                "feature_start_rel_m": 0.0,
                "feature_end_rel_m": 990.0,
                "review_flags": "response_group_extent_very_long",
            },
        ]
    )


def _gate_review() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "sequence": 1,
                "event_name": "Normal interval",
                "overall_confidence_score": 75.0,
                "recommendation": "accepted",
                "valid_pass_count": 10,
                "entry_speed_median_mps": 5.0,
                "entry_speed_p10_mps": 4.0,
                "entry_speed_p90_mps": 6.0,
                "coordinate_effective_error_m": 3.0,
                "slowdown_signature": "present",
                "cross_vehicle_status": "one_vehicle_only",
                "reasons": "",
                "suggested_action": "",
            }
        ]
    )


def _laps() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "lap_id": 1,
                "run_id": "run",
                "vehicle_id": "vehicle",
                "duration_s": 100.0,
                "analysis_valid": True,
                "speed_coverage_fraction": 1.0,
                "quality_flags": "",
                "p95_map_error_m": 2.0,
                "time_gap_count": 0,
            }
        ]
    )


def test_interval_audit_exposes_length_and_wrap() -> None:
    audit = build_event_interval_audit(_features(), 1000.0)
    normal = audit.set_index("response_group_id").loc["normal"]
    wrap = audit.set_index("response_group_id").loc["wrap"]
    bad = audit.set_index("response_group_id").loc["bad"]

    assert normal["feature_length_m"] == 30.0
    assert not bool(normal["wraps_start_finish"])

    assert bool(wrap["wraps_start_finish"])
    assert wrap["feature_start_s_m"] == 980.0
    assert wrap["feature_end_s_m"] == 20.0

    assert bad["feature_length_m"] == 990.0
    assert "covers_more_than_half_of_track" in bad["interval_audit_flags"]


def test_wrapping_interval_is_split_at_start_finish() -> None:
    assert _split_interval_for_plot(980.0, 1020.0, 1000.0) == [
        (980.0, 20.0),
        (0.0, 20.0),
    ]


def test_timeline_and_html_are_written(tmp_path: Path) -> None:
    audit = build_event_interval_audit(_features(), 1000.0)
    timeline = tmp_path / "timeline.png"
    create_event_group_timeline(timeline, audit, 1000.0)
    assert timeline.exists()
    assert timeline.stat().st_size > 0

    # The HTML writer only requires valid image bytes for embedding.
    map_path = tmp_path / "map.png"
    elevation_path = tmp_path / "elevation.png"
    map_path.write_bytes(timeline.read_bytes())
    elevation_path.write_bytes(timeline.read_bytes())

    report = tmp_path / "track_review.html"
    write_review_html(
        report,
        map_path=map_path,
        timeline_path=timeline,
        elevation_path=elevation_path,
        interval_audit=audit,
        gate_review=_gate_review(),
        laps=_laps(),
    )
    text = report.read_text(encoding="utf-8")
    assert "Along-track event-group timeline" in text
    assert "Event interval audit" in text
    assert "Nearly full-lap interval" in text
    assert "covers_more_than_half_of_track" in text
