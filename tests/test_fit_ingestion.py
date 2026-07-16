from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from garmin_fit_sdk import Encoder, Profile

from cvt_track_study.gpx import GPXRunMetadata, ingest_fit_run, ingest_telemetry_run


def _metadata(path: Path) -> GPXRunMetadata:
    return GPXRunMetadata(
        run_id="fit_run",
        vehicle_id="vehicle_A",
        driver_id="driver_1",
        source_file=path,
        use_for_centreline=True,
        use_for_gate_evidence=True,
    )


def _semicircles(degrees: float) -> int:
    return round(degrees * (2**31) / 180.0)


def _write_fit(path: Path) -> None:
    encoder = Encoder()
    start = datetime(2026, 7, 15, 12, 0, tzinfo=timezone.utc)
    records = [
        {
            "mesg_num": Profile["mesg_num"]["RECORD"],
            "timestamp": start,
            "position_lat": _semicircles(43.0),
            "position_long": _semicircles(-79.0),
            "enhanced_speed": 6.25,
            "distance": 100.0,
            "enhanced_altitude": 212.4,
        },
        {
            "mesg_num": Profile["mesg_num"]["RECORD"],
            "timestamp": start + timedelta(seconds=1),
            "position_lat": _semicircles(43.00001),
            "position_long": _semicircles(-79.0),
            "enhanced_speed": 6.5,
            "distance": 106.4,
            "enhanced_altitude": 212.8,
        },
    ]
    for record in records:
        encoder.write_mesg(record)
    path.write_bytes(encoder.close())


def test_fit_preserves_native_speed_distance_and_altitude(tmp_path: Path) -> None:
    path = tmp_path / "run.fit"
    _write_fit(path)

    result = ingest_fit_run(_metadata(path))

    assert result.summary["source_format"] == "fit"
    assert result.summary["device_speed_count"] == 2
    assert result.points["device_speed_mps"].tolist() == pytest.approx([6.25, 6.5])
    assert result.points["analysis_speed_mps"].tolist() == pytest.approx([6.25, 6.5])
    assert result.points["analysis_speed_source"].tolist() == ["fit_device", "fit_device"]
    assert result.points["speed_certainty"].tolist() == ["native_high", "native_high"]
    assert result.points["device_distance_m"].tolist() == pytest.approx([100.0, 106.4])
    assert result.points["step_distance_m"].iloc[1] == pytest.approx(6.4)
    assert result.points["elevation_source"].tolist() == [
        "fit_enhanced_altitude",
        "fit_enhanced_altitude",
    ]


def test_format_dispatch_accepts_fit(tmp_path: Path) -> None:
    path = tmp_path / "run.fit"
    _write_fit(path)
    assert ingest_telemetry_run(_metadata(path)).summary["source_format"] == "fit"
