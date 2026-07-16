from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from cvt_track_study.config import ProjectLoader, initialize_project
from cvt_track_study.config.toml_io import dump_toml, load_toml


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "templates" / "project"


def copy_project(tmp_path: Path) -> Path:
    destination = tmp_path / "project"
    shutil.copytree(TEMPLATE, destination)
    return destination


def codes(result) -> set[str]:
    return {item.code for item in result.diagnostics}


def test_template_resolves_profiles_and_local_overrides(tmp_path: Path) -> None:
    project = copy_project(tmp_path)
    result = ProjectLoader().resolve(project, study="final_drive_sweep")
    assert result.is_valid
    vehicle = result.data["vehicles"]["vehicle_A"]
    assert vehicle["vehicle"]["mass"]["nominal"] == 245.0
    assert vehicle["vehicle"]["aero"]["drag_area"]["nominal"] == 0.67
    assert vehicle["drivetrain"]["efficiency"]["nominal"] == 0.80
    assert result.data["active_study"] == "final_drive_sweep"
    assert "INHERITED_DEFAULTS_ACTIVE" in codes(result)


def test_complete_quantity_override_replaces_old_distribution_fields(tmp_path: Path) -> None:
    project = copy_project(tmp_path)
    result = ProjectLoader().resolve(project)
    uncertainty = result.data["vehicles"]["vehicle_A"]["vehicle"]["mass"]["uncertainty"]
    assert uncertainty["distribution"] == "normal"
    assert "mode" not in uncertainty
    assert "lower" not in uncertainty
    assert "upper" not in uncertainty


def test_provenance_contains_profile_and_project_layers(tmp_path: Path) -> None:
    project = copy_project(tmp_path)
    result = ProjectLoader().resolve(project)
    inherited_path = "vehicles.vehicle_A.vehicle.aero.drag_area.nominal"
    local_path = "vehicles.vehicle_A.vehicle.mass.nominal"
    assert result.provenance[inherited_path][-1].layer == "builtin_profile"
    assert result.provenance[local_path][-1].layer == "project_vehicle"


def test_resolution_export_is_readable_toml(tmp_path: Path) -> None:
    project = copy_project(tmp_path)
    result = ProjectLoader().resolve(project, study="baseline")
    output = tmp_path / "resolved"
    result.export(output)
    resolved = load_toml(output / "resolved_inputs.toml")
    assert resolved["vehicles"]["vehicle_A"]["drivetrain"]["final_drive_ratio"]["nominal"] == 7.556
    assert (output / "provenance.json").exists()
    assert (output / "validation_report.json").exists()
    assert (output / "resolution_manifest.json").exists()


def test_initialize_project_uses_bundled_template(tmp_path: Path) -> None:
    destination = initialize_project(tmp_path / "new_project", name="my track")
    assert (destination / "project.toml").exists()
    assert load_toml(destination / "project.toml")["project"]["name"] == "my track"
    result = ProjectLoader().resolve(destination)
    assert result.is_valid


def test_initialize_refuses_nonempty_destination(tmp_path: Path) -> None:
    destination = tmp_path / "existing"
    destination.mkdir()
    (destination / "keep.txt").write_text("do not overwrite")
    with pytest.raises(Exception, match="not empty"):
        initialize_project(destination)


def test_cli_override_changes_existing_leaf_and_records_warning(tmp_path: Path) -> None:
    project = copy_project(tmp_path)
    result = ProjectLoader().resolve(
        project,
        cli_overrides=(("vehicles.vehicle_A.vehicle.mass.nominal", 246.0),),
    )
    assert result.data["vehicles"]["vehicle_A"]["vehicle"]["mass"]["nominal"] == 246.0
    assert "CLI_NOMINAL_OVERRIDE_REUSES_UNCERTAINTY" in codes(result)
    assert result.provenance["vehicles.vehicle_A.vehicle.mass.nominal"][-1].layer == "command_line"


def test_unknown_cli_override_path_is_error(tmp_path: Path) -> None:
    project = copy_project(tmp_path)
    result = ProjectLoader().resolve(
        project,
        cli_overrides=(("vehicles.vehicle_A.not_real", 1),),
    )
    assert not result.is_valid
    assert "UNKNOWN_OVERRIDE_PATH" in codes(result)


def test_study_design_path_is_validated(tmp_path: Path) -> None:
    project = copy_project(tmp_path)
    path = project / "studies" / "gearing_sweep.toml"
    study = load_toml(path)
    study["design_variable"]["path"] = "drivetrain.missing_ratio"
    dump_toml(study, path)
    result = ProjectLoader().resolve(project, study="final_drive_sweep")
    assert "DESIGN_PATH_NOT_FOUND" in codes(result)


def test_user_profile_can_extend_builtin_profile(tmp_path: Path) -> None:
    project = copy_project(tmp_path)
    profile_path = project / "profiles" / "vehicles" / "alternate.toml"
    profile_path.write_text(
        '''
[profile]
id = "team.alternate_v1"
scope = "vehicle"
version = 1
extends = ["builtin.vehicle.baja_generic_v1"]

[config.vehicle.aero.drag_area]
nominal = 0.8
unit = "m^2"
source = { kind = "engineering_estimate", reference = "CAD estimate" }
uncertainty = { distribution = "uniform", lower = 0.7, upper = 0.9 }
'''.strip()
        + "\n",
        encoding="utf-8",
    )
    vehicle_path = project / "vehicles" / "vehicle_A" / "vehicle.toml"
    vehicle = load_toml(vehicle_path)
    vehicle["vehicle"]["profiles"] = ["team.alternate_v1"]
    dump_toml(vehicle, vehicle_path)
    result = ProjectLoader().resolve(project)
    assert result.is_valid
    assert result.data["vehicles"]["vehicle_A"]["vehicle"]["aero"]["drag_area"]["nominal"] == 0.8


def test_missing_profile_is_actionable_error(tmp_path: Path) -> None:
    project = copy_project(tmp_path)
    vehicle_path = project / "vehicles" / "vehicle_A" / "vehicle.toml"
    vehicle = load_toml(vehicle_path)
    vehicle["vehicle"]["profiles"] = ["does.not.exist"]
    dump_toml(vehicle, vehicle_path)
    result = ProjectLoader().resolve(project)
    assert "PROFILE_NOT_FOUND" in codes(result)


def test_valid_study_override_replaces_complete_quantity(tmp_path: Path) -> None:
    project = copy_project(tmp_path)
    study_path = project / "studies" / "baseline.toml"
    raw = load_toml(study_path)
    raw["config_overrides"] = {
        "vehicles": {
            "vehicle_A": {
                "drivetrain": {
                    "efficiency": {
                        "nominal": 0.78,
                        "unit": "1",
                        "source": {
                            "kind": "calibrated",
                            "reference": "specific dyno condition",
                        },
                        "uncertainty": {
                            "distribution": "normal",
                            "standard_deviation": 0.02,
                        },
                    }
                }
            }
        }
    }
    dump_toml(raw, study_path)
    result = ProjectLoader().resolve(project, study="baseline")
    assert result.is_valid
    efficiency = result.data["vehicles"]["vehicle_A"]["drivetrain"]["efficiency"]
    assert efficiency["nominal"] == 0.78
    assert efficiency["uncertainty"] == {
        "distribution": "normal",
        "standard_deviation": 0.02,
    }
    path = "vehicles.vehicle_A.drivetrain.efficiency.nominal"
    assert result.provenance[path][-1].layer == "study_override"


def test_unknown_study_override_path_is_error(tmp_path: Path) -> None:
    project = copy_project(tmp_path)
    study_path = project / "studies" / "baseline.toml"
    raw = load_toml(study_path)
    raw["config_overrides"] = {
        "vehicles": {"vehicle_A": {"drivetrain": {"typo": 1.0}}}
    }
    dump_toml(raw, study_path)
    result = ProjectLoader().resolve(project, study="baseline")
    assert "UNKNOWN_STUDY_OVERRIDE_PATH" in codes(result)
    assert "typo" not in result.data["vehicles"]["vehicle_A"]["drivetrain"]


def test_shared_profile_root_outside_project_is_supported(tmp_path: Path) -> None:
    project = copy_project(tmp_path)
    shared = tmp_path / "shared_profiles"
    shared.mkdir()
    (shared / "shared.toml").write_text(
        '''
[profile]
id = "team.shared_vehicle_v1"
scope = "vehicle"
version = 1
extends = ["builtin.vehicle.baja_generic_v1"]

[config.drivetrain.efficiency]
nominal = 0.79
unit = "1"
source = { kind = "engineering_estimate", reference = "shared team estimate" }
uncertainty = { distribution = "uniform", lower = 0.75, upper = 0.83 }
'''.strip()
        + "\n",
        encoding="utf-8",
    )
    project_path = project / "project.toml"
    project_config = load_toml(project_path)
    project_config["profiles"]["roots"] = [str(shared)]
    dump_toml(project_config, project_path)
    vehicle_path = project / "vehicles" / "vehicle_A" / "vehicle.toml"
    vehicle = load_toml(vehicle_path)
    vehicle["vehicle"]["profiles"] = ["team.shared_vehicle_v1"]
    dump_toml(vehicle, vehicle_path)
    result = ProjectLoader().resolve(project)
    assert result.is_valid
    assert result.data["vehicles"]["vehicle_A"]["drivetrain"]["efficiency"]["nominal"] == 0.79


def test_existing_gpx_run_and_vehicle_reference_validate(tmp_path: Path) -> None:
    project = copy_project(tmp_path)
    gpx = project / "track" / "gpx" / "run_A01.gpx"
    gpx.write_text("<gpx version=\"1.1\"></gpx>\n", encoding="utf-8")
    dump_toml(
        {
            "runs": [
                {
                    "file": "gpx/run_A01.gpx",
                    "vehicle_id": "vehicle_A",
                    "run_id": "A01",
                    "driver_id": "driver_1",
                    "use_for_centreline": True,
                    "use_for_gate_evidence": True,
                }
            ]
        },
        project / "track" / "runs.toml",
    )
    result = ProjectLoader().resolve(project)
    assert result.is_valid
    assert "NO_GPX_RUNS" not in codes(result)
