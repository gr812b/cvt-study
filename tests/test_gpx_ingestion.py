from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from cvt_track_study.gpx import GPXParseError, GPXRunMetadata, ingest_gpx_run


def _metadata(path: Path) -> GPXRunMetadata:
    return GPXRunMetadata(
        run_id="run_1",
        vehicle_id="vehicle_A",
        driver_id="driver_1",
        source_file=path,
        use_for_centreline=True,
        use_for_gate_evidence=True,
    )


def test_gpx_11_preserves_elevation_segments_and_extensions(tmp_path: Path) -> None:
    path = tmp_path / "run.gpx"
    path.write_text(
        '''<?xml version="1.0"?>
<gpx version="1.1" xmlns="http://www.topografix.com/GPX/1/1"
 xmlns:gpxtpx="http://www.garmin.com/xmlschemas/TrackPointExtension/v1">
 <wpt lat="43" lon="-79"/>
 <rte><rtept lat="43" lon="-79"/></rte>
 <trk><trkseg>
  <trkpt lat="43.0" lon="-79.0"><ele>100</ele><time>2026-07-15T12:00:00Z</time>
   <extensions><gpxtpx:TrackPointExtension><gpxtpx:speed>3.5</gpxtpx:speed></gpxtpx:TrackPointExtension></extensions>
  </trkpt>
  <trkpt lat="43.00001" lon="-79.0"><ele>101</ele><time>2026-07-15T12:00:01Z</time></trkpt>
 </trkseg><trkseg>
  <trkpt lat="43.00002" lon="-79.0"><ele>102</ele><time>2026-07-15T12:00:03Z</time></trkpt>
 </trkseg></trk>
</gpx>''',
        encoding="utf-8",
    )
    result = ingest_gpx_run(_metadata(path))
    assert len(result.points) == 3
    assert len(result.segments) == 2
    assert result.points["elevation_m"].tolist() == [100.0, 101.0, 102.0]
    assert result.points.loc[0, "reported_speed_mps"] == 3.5
    assert "gpxtpx:speed" in result.points.loc[0, "extension_json"]
    assert any(item.code == "NON_TRACK_GPX_CONTENT_IGNORED" for item in result.diagnostics)


def test_segment_duration_is_missing_when_timestamps_regress(tmp_path: Path) -> None:
    path = tmp_path / "regress.gpx"
    path.write_text(
        '''<gpx version="1.1"><trk><trkseg>
<trkpt lat="43" lon="-79"><time>2026-07-15T12:00:02Z</time></trkpt>
<trkpt lat="43.00001" lon="-79"><time>2026-07-15T12:00:01Z</time></trkpt>
</trkseg></trk></gpx>''',
        encoding="utf-8",
    )
    result = ingest_gpx_run(_metadata(path))
    assert not bool(result.segments.loc[0, "timestamps_monotonic"])
    assert pd.isna(result.segments.loc[0, "duration_s"])
    assert any(item.code == "GPX_TIMESTAMP_REGRESSION" for item in result.diagnostics)


def test_entity_expansion_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "unsafe.gpx"
    path.write_text(
        '''<?xml version="1.0"?>
<!DOCTYPE gpx [<!ENTITY xxe SYSTEM "file:///etc/passwd">]>
<gpx version="1.1"><trk><name>&xxe;</name><trkseg>
<trkpt lat="43" lon="-79"/></trkseg></trk></gpx>''',
        encoding="utf-8",
    )
    with pytest.raises(GPXParseError):
        ingest_gpx_run(_metadata(path))


def test_route_only_gpx_is_not_silently_treated_as_track(tmp_path: Path) -> None:
    path = tmp_path / "route.gpx"
    path.write_text(
        '<gpx version="1.1"><rte><rtept lat="43" lon="-79"/></rte></gpx>',
        encoding="utf-8",
    )
    with pytest.raises(GPXParseError, match="no <trk>"):
        ingest_gpx_run(_metadata(path))


def test_invalid_timestamp_is_distinguished_from_missing_timestamp(tmp_path: Path) -> None:
    path = tmp_path / "invalid-time.gpx"
    path.write_text(
        '''<gpx version="1.1"><trk><trkseg>
<trkpt lat="43" lon="-79"><time>not-a-time</time></trkpt>
<trkpt lat="43.00001" lon="-79"/>
</trkseg></trk></gpx>''',
        encoding="utf-8",
    )
    result = ingest_gpx_run(_metadata(path))
    assert result.summary["missing_timestamp_count"] == 1
    assert result.summary["invalid_timestamp_count"] == 1
    assert result.summary["unusable_timestamp_count"] == 2
    assert any(item.code == "GPX_TIMESTAMPS_INVALID" for item in result.diagnostics)


def test_invalid_coordinate_is_excluded_with_warning(tmp_path: Path) -> None:
    path = tmp_path / "invalid-coordinate.gpx"
    path.write_text(
        '''<gpx version="1.1"><trk><trkseg>
<trkpt lat="95" lon="-79"><time>2026-07-15T12:00:00Z</time></trkpt>
<trkpt lat="43" lon="-79"><time>2026-07-15T12:00:01Z</time></trkpt>
</trkseg></trk></gpx>''',
        encoding="utf-8",
    )
    result = ingest_gpx_run(_metadata(path))
    diagnostic = next(item for item in result.diagnostics if item.code == "INVALID_GPX_COORDINATE")
    assert diagnostic.severity.value == "warning"
    assert result.error_count == 0
    assert len(result.points) == 1
