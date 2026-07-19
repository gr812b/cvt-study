from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cvt_track_study.config.diagnostics import DiagnosticBag
from cvt_track_study.gpx.cleanup import (
    apply_telemetry_cleanup,
    remove_isolated_map_outliers,
)
from cvt_track_study.gpx.model import (
    CANONICAL_POINT_COLUMNS,
    GPXIngestionResult,
    GPXRunMetadata,
)


def _result(points: list[tuple[float, float]]) -> GPXIngestionResult:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rows = []
    for index, (lat, lon) in enumerate(points):
        rows.append(
            {
                "run_id": "run_1",
                "vehicle_id": "vehicle_A",
                "driver_id": "driver_1",
                "source_file": "run.gpx",
                "source_sha256": "abc",
                "source_format": "gpx",
                "track_index": 0,
                "segment_index": 0,
                "point_index": index,
                "timestamp_utc": start + timedelta(seconds=index),
                "latitude_deg": lat,
                "longitude_deg": lon,
                "elevation_m": 100.0,
                "elevation_source": "gpx_elevation",
                "device_distance_m": np.nan,
                "device_speed_mps": np.nan,
                "reported_speed_mps": np.nan,
                "derived_speed_mps": np.nan,
                "analysis_speed_mps": np.nan,
                "analysis_speed_source": "unavailable",
                "speed_certainty": "unavailable",
                "course_deg": np.nan,
                "fix_type": "3d",
                "satellites": 8,
                "horizontal_accuracy_m": np.nan,
                "hdop": np.nan,
                "vdop": np.nan,
                "pdop": np.nan,
                "step_distance_m": np.nan,
                "time_step_s": np.nan,
                "extension_json": "{}",
            }
        )
    frame = pd.DataFrame(rows).reindex(columns=CANONICAL_POINT_COLUMNS)
    metadata = GPXRunMetadata(
        run_id="run_1",
        vehicle_id="vehicle_A",
        driver_id="driver_1",
        source_file=Path("run.gpx"),
        use_for_centreline=True,
        use_for_gate_evidence=True,
    )
    return GPXIngestionResult(
        metadata=metadata,
        points=frame,
        segments=pd.DataFrame(),
        summary={
            "source_sha256": "abc",
            "source_format": "gpx",
            "invalid_coordinate_count": 0,
            "valid_point_count": len(frame),
        },
        diagnostics=(),
    )


def _config() -> dict:
    return {
        "reconstruction": {
            "maximum_reasonable_speed_mps": 25.0,
            "maximum_map_error_m": 20.0,
        },
        "telemetry_cleanup": {
            "enabled": True,
            "maximum_excursion_points": 3,
            "minimum_excursion_leg_m": 35.0,
            "impossible_speed_multiplier": 1.5,
            "maximum_bridge_speed_multiplier": 1.0,
            "maximum_bridge_gap_s": 8.0,
            "maximum_auto_removed_fraction": 0.20,
            "maximum_auto_removed_points": 25,
            "isolated_map_error_m": 40.0,
            "maximum_isolated_map_outlier_points": 3,
        },
    }


def test_isolated_out_and_back_coordinate_is_removed() -> None:
    result = _result(
        [
            (43.00000, -79.00000),
            (43.00003, -79.00000),
            (43.01000, -79.01000),  # isolated GPS excursion
            (43.00006, -79.00000),
            (43.00009, -79.00000),
        ]
    )
    cleaned = apply_telemetry_cleanup(result, _config())
    assert len(cleaned.points) == 4
    assert cleaned.rejected_points["point_index"].tolist() == [2]
    assert cleaned.summary["isolated_excursion_point_count"] == 1
    assert cleaned.points["step_distance_m"].fillna(0).max() < 10.0


def test_sustained_excursion_is_not_silently_repaired() -> None:
    result = _result(
        [
            (43.00000, -79.00000),
            (43.00003, -79.00000),
            (43.01000, -79.01000),
            (43.01003, -79.01000),
            (43.01006, -79.01000),
            (43.01009, -79.01000),
            (43.00006, -79.00000),
            (43.00009, -79.00000),
        ]
    )
    cleaned = apply_telemetry_cleanup(result, _config())
    assert len(cleaned.points) == len(result.points)
    assert cleaned.rejected_points.empty


def test_automatic_removal_cap_prevents_mass_deletion() -> None:
    result = _result(
        [
            (43.00000, -79.00000),
            (43.00003, -79.00000),
            (43.01000, -79.01000),
            (43.00006, -79.00000),
            (43.01000, -79.01000),
            (43.00009, -79.00000),
        ]
    )
    config = _config()
    config["telemetry_cleanup"]["maximum_auto_removed_points"] = 1
    cleaned = apply_telemetry_cleanup(result, config)
    assert len(cleaned.points) == len(result.points)
    assert cleaned.rejected_points.empty
    assert any(
        item.code == "TELEMETRY_CLEANUP_LIMIT_EXCEEDED"
        for item in cleaned.diagnostics
    )


def test_isolated_map_outlier_is_removed_and_quality_recomputed() -> None:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    matched = pd.DataFrame(
        {
            "lap_id": [1] * 6,
            "run_id": ["run_1"] * 6,
            "track_index": [0] * 6,
            "segment_index": [0] * 6,
            "point_index": list(range(6)),
            "timestamp_utc": [
                start + timedelta(seconds=i) for i in range(6)
            ],
            "latitude_deg": [43.0 + i * 0.00001 for i in range(6)],
            "longitude_deg": [-79.0] * 6,
            "map_error_m": [2.0, 3.0, 75.0, 3.0, 2.0, 2.0],
            "s_m": [0.0, 5.0, 10.0, 15.0, 20.0, 25.0],
        }
    )
    laps = pd.DataFrame(
        {
            "lap_id": [1],
            "analysis_valid": [False],
            "quality_flags": ["p95_map_error_exceeds_limit"],
            "median_map_error_m": [3.0],
            "p95_map_error_m": [60.0],
            "maximum_map_error_m": [75.0],
            "large_backward_match_count": [0],
        }
    )
    bag = DiagnosticBag()
    cleaned, updated, rejected = remove_isolated_map_outliers(
        matched, laps, _config(), bag
    )
    assert len(cleaned) == 5
    assert len(rejected) == 1
    assert bool(updated.loc[0, "analysis_valid"])
    assert updated.loc[0, "quality_flags"] == ""
