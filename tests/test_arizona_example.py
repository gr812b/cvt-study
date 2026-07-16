from __future__ import annotations

from pathlib import Path

from cvt_track_study.gpx import ingest_project
from cvt_track_study.track import build_project_track


ARIZONA = Path(__file__).resolve().parents[1] / "examples" / "arizona_endurance_project"


def test_arizona_gpx_is_canonical_real_data_example(tmp_path: Path) -> None:
    resolution, runs, _ = ingest_project(
        ARIZONA, output_directory=tmp_path / "ingestion"
    )
    assert resolution.error_count == 0
    assert len(runs) == 1
    run = runs[0]
    assert len(run.points) == 6822
    assert run.points["elevation_m"].notna().all()
    assert run.points["reported_speed_mps"].isna().all()
    assert run.points["analysis_speed_mps"].notna().sum() >= 6820
    assert any(item.code == "DUPLICATE_GPX_TIMESTAMPS" for item in run.diagnostics)


def test_arizona_example_reconstructs_real_track(tmp_path: Path) -> None:
    result = build_project_track(ARIZONA, output_directory=tmp_path / "build")
    assert 1765.0 < result.centreline.length_m < 1785.0
    assert len(result.laps) == 13
    assert int(result.laps["analysis_valid"].sum()) == 11
    assert len(result.event_projection) == 40
    assert len(result.response_features) == 37
    assert result.track_profile["median_elevation_m"].notna().all()
    assert (tmp_path / "build" / "review" / "track_map.png").exists()
