from cvt_track_study.config import ProjectLoader


def test_arizona_project_accepts_cornell_csv_external_telemetry():
    result = ProjectLoader().resolve("projects/arizona")
    errors = [item.format() for item in result.diagnostics if item.severity.value == "error"]
    assert errors == []
    run = next(
        item for item in result.data["runs"]
        if item["run_id"] == "arizona_cornell_endurance"
    )
    assert run["use_for_centreline"] is True
    assert run["use_for_gate_evidence"] is True
    assert run["external_vehicle"] is True
