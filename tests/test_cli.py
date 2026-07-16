from pathlib import Path

from cvt_track_study.cli import main


def test_init_and_validate_commands(tmp_path: Path, capsys) -> None:
    project = tmp_path / "workspace"
    assert main(["init", str(project), "--name", "cli project"]) == 0
    assert main(["validate", str(project), "--no-export"]) == 0
    output = capsys.readouterr().out
    assert "Created project" in output
    assert "Validation summary" in output


def test_strict_validation_fails_on_template_warnings(tmp_path: Path) -> None:
    project = tmp_path / "workspace"
    assert main(["init", str(project)]) == 0
    assert main(["validate", str(project), "--no-export", "--strict"]) == 1


def test_ingest_and_build_track_commands(tmp_path: Path, capsys) -> None:
    import shutil

    reference = Path(__file__).resolve().parents[1] / "examples" / "reference_project"
    project = tmp_path / "reference_project"
    shutil.copytree(reference, project)
    ingest_output = tmp_path / "ingestion"
    build_output = tmp_path / "track_build"

    assert main(["ingest", str(project), "--output", str(ingest_output)]) == 0
    assert (ingest_output / "canonical_points.csv").exists()
    assert (ingest_output / "configuration" / "resolved_inputs.toml").exists()

    assert main(["build-track", str(project), "--output", str(build_output)]) == 0
    assert (build_output / "track" / "gate_review.csv").exists()
    assert (build_output / "review" / "track_review.html").exists()
    assert (build_output / "configuration" / "provenance.json").exists()
    output = capsys.readouterr().out
    assert "Track build:" in output
    assert "Review package:" in output


def test_validate_bundle_command(tmp_path: Path, capsys) -> None:
    import shutil

    reference = Path(__file__).resolve().parents[1] / "examples" / "reference_project"
    project = tmp_path / "reference_project"
    shutil.copytree(reference, project)
    output = tmp_path / "track_build"
    assert main(["build-track", str(project), "--output", str(output)]) == 0
    capsys.readouterr()
    assert main(["validate-bundle", str(output / "track_bundle.json")]) == 0
    captured = capsys.readouterr().out
    assert "Valid track bundle 1.2.0" in captured
    assert "active speed gate" in captured
