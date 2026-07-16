from __future__ import annotations

from pathlib import Path


from cvt_track_study.track import build_project_track


REFERENCE = Path(__file__).resolve().parents[1] / "examples" / "reference_project"


def test_reference_project_builds_reviewable_track(tmp_path: Path) -> None:
    result = build_project_track(REFERENCE, output_directory=tmp_path / "build")
    assert 590 < result.centreline.length_m < 610
    assert len(result.laps) >= 8
    assert result.laps["analysis_valid"].all()
    assert (result.gate_review["recommendation"] == "accepted").sum() == 3
    assert (tmp_path / "build" / "review" / "track_map.png").exists()
    assert (tmp_path / "build" / "review" / "elevation_profile.png").exists()
    assert (tmp_path / "build" / "review" / "track_review.html").exists()
    assert (tmp_path / "build" / "track" / "response_features.csv").exists()
    assert {
        "feature_start_projection_error_m",
        "feature_start_horizontal_uncertainty_m",
        "feature_start_effective_error_m",
        "feature_end_projection_error_m",
        "feature_end_horizontal_uncertainty_m",
        "feature_end_effective_error_m",
    }.issubset(result.gate_review.columns)


def test_lap_gate_is_not_automatically_a_speed_gate(tmp_path: Path) -> None:
    result = build_project_track(REFERENCE, output_directory=tmp_path / "build")
    row = result.gate_evidence.set_index("event_id").loc["start_finish"]
    assert row["recommendation"] == "not_a_candidate"


def test_coordinate_uncertainty_reduces_coordinate_quality(tmp_path: Path) -> None:
    project = tmp_path / "project"
    import shutil

    shutil.copytree(REFERENCE, project)
    events = project / "track" / "events.toml"
    text = events.read_text(encoding="utf-8")
    block_start = text.index('id = "logs_west"')
    block_end = text.find("[[events]]", block_start)
    if block_end < 0:
        block_end = len(text)
    block = text[block_start:block_end].replace(
        "horizontal_uncertainty_m = 2.0",
        "horizontal_uncertainty_m = 18.0",
        1,
    )
    events.write_text(text[:block_start] + block + text[block_end:], encoding="utf-8")
    result = build_project_track(project, output_directory=tmp_path / "build")
    row = result.gate_evidence.set_index("event_id").loc["logs_west"]
    assert row["coordinate_effective_error_m"] > 17
    assert row["coordinate_quality_score"] < 15


def test_response_group_collapses_multiple_physical_features(tmp_path: Path) -> None:
    project = tmp_path / "project"
    import shutil

    shutil.copytree(REFERENCE, project)
    events = project / "track" / "events.toml"
    text = events.read_text(encoding="utf-8")
    text = text.replace('response_group_id = "ruts_south"', 'response_group_id = "logs_west"')
    events.write_text(text, encoding="utf-8")
    result = build_project_track(project, output_directory=tmp_path / "build")
    compound = result.response_features.set_index("id").loc["logs_west"]
    assert compound["analysis_feature_type"] == "response_group"
    assert "logs_west" in compound["source_event_ids"]
    assert "ruts_south" in compound["source_event_ids"]
    assert (result.gate_evidence["event_id"] == "logs_west").sum() == 1


def test_track_build_rejects_fatal_ingestion_errors(tmp_path: Path) -> None:
    import shutil
    import pytest

    from cvt_track_study.config import ProjectError

    project = tmp_path / "project"
    shutil.copytree(REFERENCE, project)
    gpx = project / "track" / "gpx" / "reference_vehicle_A.gpx"
    text = gpx.read_text(encoding="utf-8")
    first = "2026-07-15T12:00:01Z"
    replacement = "2026-07-15T11:59:59Z"
    assert first in text
    gpx.write_text(text.replace(first, replacement, 1), encoding="utf-8")

    with pytest.raises(ProjectError, match="fatal timing/data errors"):
        build_project_track(project, output_directory=tmp_path / "build")


def test_track_build_handles_completely_missing_elevation(tmp_path: Path) -> None:
    import re
    import shutil
    import warnings

    project = tmp_path / "project"
    shutil.copytree(REFERENCE, project)
    gpx = project / "track" / "gpx" / "reference_vehicle_A.gpx"
    text = re.sub(r"<ele>[^<]*</ele>", "", gpx.read_text(encoding="utf-8"))
    gpx.write_text(text, encoding="utf-8")

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        result = build_project_track(project, output_directory=tmp_path / "build")
    assert result.track_profile["median_elevation_m"].isna().all()
    assert (tmp_path / "build" / "review" / "elevation_profile.png").exists()


def test_feature_overlap_is_checked_across_start_finish_wrap(tmp_path: Path) -> None:
    import shutil

    project = tmp_path / "project"
    shutil.copytree(REFERENCE, project)
    events = project / "track" / "events.toml"
    text = events.read_text(encoding="utf-8")
    block_start = text.index('id = "ruts_south"')
    block = text[block_start:].replace("after_anchor_m = 7.0", "after_anchor_m = 200.0", 1)
    events.write_text(text[:block_start] + block, encoding="utf-8")

    result = build_project_track(project, output_directory=tmp_path / "build")
    projected = result.event_projection.set_index("id")
    assert "overlaps_adjacent_feature" in projected.loc["ruts_south", "review_flags"]
    assert "overlaps_adjacent_feature" in projected.loc["start_finish", "review_flags"]


def test_extent_uncertainty_reduces_gate_coordinate_quality(tmp_path: Path) -> None:
    import shutil

    project = tmp_path / "project"
    shutil.copytree(REFERENCE, project)
    events = project / "track" / "events.toml"
    text = events.read_text(encoding="utf-8")
    block_start = text.index('id = "logs_west"')
    block_end = text.find("[[events]]", block_start)
    if block_end < 0:
        block_end = len(text)
    block = text[block_start:block_end].replace(
        "before_anchor_uncertainty_m = 2.0",
        "before_anchor_uncertainty_m = 18.0",
        1,
    )
    events.write_text(text[:block_start] + block + text[block_end:], encoding="utf-8")

    result = build_project_track(project, output_directory=tmp_path / "build")
    row = result.gate_evidence.set_index("event_id").loc["logs_west"]
    assert row["coordinate_effective_error_m"] > 17
    assert row["coordinate_quality_score"] < 15


def test_track_build_accepts_explicit_endpoints_without_extent(tmp_path: Path) -> None:
    import shutil

    from cvt_track_study.config.toml_io import dump_toml, load_toml

    project = tmp_path / "project"
    shutil.copytree(REFERENCE, project)
    events_path = project / "track" / "events.toml"
    raw = load_toml(events_path)
    event = raw["events"][1]
    anchor = event["anchor"]
    event["start"] = {
        "latitude_deg": anchor["latitude_deg"],
        "longitude_deg": anchor["longitude_deg"] - 0.00005,
        "horizontal_uncertainty_m": 2.0,
        "source": "synthetic explicit start",
    }
    event["end"] = {
        "latitude_deg": anchor["latitude_deg"],
        "longitude_deg": anchor["longitude_deg"] + 0.00005,
        "horizontal_uncertainty_m": 2.0,
        "source": "synthetic explicit end",
    }
    del event["extent"]
    dump_toml(raw, events_path)

    result = build_project_track(project, output_directory=tmp_path / "build")
    row = result.event_projection.set_index("id").loc["turn_north"]
    assert row["feature_start_source"] == "explicit_coordinate"
    assert row["feature_end_source"] == "explicit_coordinate"


def test_non_candidate_with_bad_geometry_is_still_must_fix(tmp_path: Path) -> None:
    import shutil

    project = tmp_path / "project"
    shutil.copytree(REFERENCE, project)
    events = project / "track" / "events.toml"
    text = events.read_text(encoding="utf-8")
    block_start = text.index('id = "start_finish"')
    block_end = text.find("[[events]]", block_start)
    block = text[block_start:block_end].replace(
        "horizontal_uncertainty_m = 2.0",
        "horizontal_uncertainty_m = 30.0",
        1,
    )
    events.write_text(text[:block_start] + block + text[block_end:], encoding="utf-8")

    result = build_project_track(project, output_directory=tmp_path / "build")
    row = result.gate_review.set_index("event_id").loc["start_finish"]
    assert row["recommendation"] == "must_fix"


def test_multiple_vehicles_contribute_measured_agreement(tmp_path: Path) -> None:
    import shutil

    from cvt_track_study.config.toml_io import dump_toml, load_toml

    project = tmp_path / "project"
    shutil.copytree(REFERENCE, project)
    vehicle_b = project / "vehicles" / "vehicle_B"
    shutil.copytree(project / "vehicles" / "vehicle_A", vehicle_b)
    vehicle_path = vehicle_b / "vehicle.toml"
    vehicle = load_toml(vehicle_path)
    vehicle["vehicle"]["id"] = "vehicle_B"
    dump_toml(vehicle, vehicle_path)

    source_gpx = project / "track" / "gpx" / "reference_vehicle_A.gpx"
    second_gpx = project / "track" / "gpx" / "reference_vehicle_B.gpx"
    shutil.copy2(source_gpx, second_gpx)
    runs_path = project / "track" / "runs.toml"
    runs = load_toml(runs_path)
    runs["runs"].append(
        {
            "file": "gpx/reference_vehicle_B.gpx",
            "vehicle_id": "vehicle_B",
            "run_id": "B01",
            "driver_id": "driver_2",
            "use_for_centreline": False,
            "use_for_gate_evidence": True,
        }
    )
    dump_toml(runs, runs_path)

    result = build_project_track(project, output_directory=tmp_path / "build")
    accepted = result.gate_evidence[result.gate_evidence["gate_candidate"]]
    assert (accepted["vehicle_count"] == 2).all()
    assert (accepted["cross_vehicle_status"] == "measured").all()
