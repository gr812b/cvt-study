from pathlib import Path

import pytest

from cvt_track_study.gpx.csv_parser import ingest_csv_run
from cvt_track_study.gpx.model import GPXRunMetadata


def test_cornell_csv_is_native_track_and_gate_telemetry():
    source = Path("projects/arizona/track/csv/CORNELL.csv").resolve()
    metadata = GPXRunMetadata(
        run_id="arizona_cornell_endurance",
        vehicle_id="cornell",
        driver_id="cornell_driver_unknown",
        source_file=source,
        use_for_centreline=True,
        use_for_gate_evidence=True,
    )
    result = ingest_csv_run(metadata)
    assert result.summary["source_format"] == "csv"
    assert result.summary["valid_point_count"] == 12328
    assert result.summary["device_speed_count"] == 12328
    assert set(result.points["analysis_speed_source"].dropna().unique()) == {"csv_native"}
    assert set(result.points["speed_certainty"].dropna().unique()) == {"native_high"}
    assert result.points["analysis_speed_mps"].max() == pytest.approx(49.2588 / 3.6)


def test_cornell_is_not_traffic_evidence():
    traffic = Path("projects/arizona/track/traffic.toml").read_text(encoding="utf-8")
    assert "CORNELL" not in traffic.upper()
    assert not Path("projects/arizona/track/traffic/cornell_az_2025.csv").exists()
