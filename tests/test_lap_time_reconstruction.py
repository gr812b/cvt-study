from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cvt_track_study.gpx.lap_time_reconstruction import (
    _parse_lap_duration,
    _read_lap_times,
    apply_optional_lap_time_reconstruction,
)
from cvt_track_study.gpx.model import (
    CANONICAL_POINT_COLUMNS,
    GPXIngestionResult,
    GPXRunMetadata,
    LapTimeReconstructionConfig,
)


def test_lap_duration_formats():
    assert _parse_lap_duration(271.5) == pytest.approx(271.5)
    assert _parse_lap_duration("04:31.5") == pytest.approx(271.5)
    assert _parse_lap_duration("1:04:31.5") == pytest.approx(3871.5)


def test_ets_preamble_and_long_laps_remain_auditable(tmp_path: Path):
    path = tmp_path / "ets.csv"
    path.write_text(
        "Car Number,78\nTeam Name,Baja ÉTS\ntimestamp,lap_time_s\n"
        "12:00:00,250.0\n12:05:00,827.246\n",
        encoding="utf-8",
    )
    config = LapTimeReconstructionConfig(
        lap_times_file=path,
        lap_time_column="lap_time_s",
        minimum_csv_lap_time_s=180.0,
        maximum_csv_lap_time_s=420.0,
    )
    frame = _read_lap_times(path, config)
    assert frame["csv_included"].tolist() == [True, False]
    assert frame.loc[1, "csv_exclusion_reason"] == "above_maximum_csv_lap_time"
    assert "Baja ÉTS" in frame.loc[0, "csv_metadata_json"]


def test_untimed_gpx_reconstructs_two_laps_and_exports_gpx(tmp_path: Path):
    csv_path = tmp_path / "laps.csv"
    csv_path.write_text(
        "lap,lap_time,include\n1,10,true\n2,12,true\n",
        encoding="utf-8",
    )
    metadata = GPXRunMetadata(
        run_id="ets",
        vehicle_id="ets",
        driver_id="driver",
        source_file=tmp_path / "ets.gpx",
        use_for_centreline=True,
        use_for_gate_evidence=True,
        lap_time_reconstruction=LapTimeReconstructionConfig(
            lap_times_file=csv_path,
            minimum_points_per_lap=5,
            minimum_matched_laps=2,
            expected_point_period_s=2.0,
            maximum_point_period_error_fraction=0.6,
            alignment_mode="strict_index",
            synthetic_start_time_utc=datetime(2000, 1, 1, tzinfo=timezone.utc),
        ),
    )
    points = _two_lap_points(metadata)
    result = GPXIngestionResult(
        metadata=metadata,
        points=points,
        segments=pd.DataFrame(),
        summary={
            "source_sha256": "raw",
            "source_format": "gpx",
            "isolated_excursion_point_count": 0,
            "clean_positioned_point_count": len(points),
        },
        diagnostics=(),
    )
    reconstructed = apply_optional_lap_time_reconstruction(
        result,
        track_config={
            "reconstruction": {
                "lap_gate_event_id": "start_finish",
                "lap_gate_radius_m": 5.0,
            }
        },
        events=(_lap_gate(),),
    )

    matched = reconstructed.reconstruction_laps.query("alignment_status == 'matched'")
    assert len(matched) == 2
    assert matched["output_segment_index"].tolist() == [0, 1]
    assert set(reconstructed.points["analysis_speed_source"]) == {
        "lap_time_reconstructed"
    }
    durations = []
    for _, segment in reconstructed.points.groupby("segment_index"):
        times = pd.to_datetime(segment["timestamp_utc"], utc=True)
        durations.append((times.iloc[-1] - times.iloc[0]).total_seconds())
    assert durations == pytest.approx([10.0, 12.0])
    assert reconstructed.augmented_gpx_text is not None
    assert reconstructed.augmented_gpx_text.count("<trkseg>") == 2
    assert "declared_lap_time_s" in reconstructed.augmented_gpx_text


def test_mismatched_csv_lap_count_fails_closed(tmp_path: Path):
    csv_path = tmp_path / "laps.csv"
    csv_path.write_text("lap,lap_time\n1,10\n", encoding="utf-8")
    metadata = GPXRunMetadata(
        run_id="ets",
        vehicle_id="ets",
        driver_id="driver",
        source_file=tmp_path / "ets.gpx",
        use_for_centreline=True,
        use_for_gate_evidence=True,
        lap_time_reconstruction=LapTimeReconstructionConfig(
            lap_times_file=csv_path,
            minimum_points_per_lap=5,
            minimum_matched_laps=1,
            alignment_mode="strict_index",
        ),
    )
    result = GPXIngestionResult(
        metadata=metadata,
        points=_two_lap_points(metadata),
        segments=pd.DataFrame(),
        summary={
            "source_sha256": "raw",
            "source_format": "gpx",
            "isolated_excursion_point_count": 0,
            "clean_positioned_point_count": 13,
        },
        diagnostics=(),
    )
    with pytest.raises(ValueError, match="one-to-one alignment"):
        apply_optional_lap_time_reconstruction(
            result,
            track_config={
                "reconstruction": {
                    "lap_gate_event_id": "start_finish",
                    "lap_gate_radius_m": 5.0,
                }
            },
            events=(_lap_gate(),),
        )


def _lap_gate() -> dict[str, object]:
    return {
        "id": "start_finish",
        "analysis_role": "lap_gate",
        "anchor": {"latitude_deg": 0.0, "longitude_deg": 0.0},
    }


def _two_lap_points(metadata: GPXRunMetadata) -> pd.DataFrame:
    coordinates = [
        (0.0, 0.0),
        (0.0, 0.0010),
        (0.0010, 0.0010),
        (0.0010, 0.0),
        (0.0005, -0.0010),
        (0.0, -0.0010),
        (0.0, 0.0),
        (0.0, 0.0010),
        (0.0010, 0.0010),
        (0.0010, 0.0),
        (0.0005, -0.0010),
        (0.0, -0.0010),
        (0.0, 0.0),
    ]
    rows = []
    for index, (lat, lon) in enumerate(coordinates):
        row = {column: np.nan for column in CANONICAL_POINT_COLUMNS}
        row.update(
            {
                "run_id": metadata.run_id,
                "vehicle_id": metadata.vehicle_id,
                "driver_id": metadata.driver_id,
                "source_file": str(metadata.source_file),
                "source_sha256": "raw",
                "source_format": "gpx",
                "track_index": 0,
                "segment_index": 0,
                "point_index": index,
                "timestamp_utc": pd.NaT,
                "latitude_deg": lat,
                "longitude_deg": lon,
                "elevation_m": 0.0,
                "elevation_source": "gpx_elevation",
                "analysis_speed_source": "unavailable",
                "speed_certainty": "unavailable",
                "extension_json": "{}",
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).reindex(columns=CANONICAL_POINT_COLUMNS)
