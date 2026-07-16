from __future__ import annotations

import shutil
from pathlib import Path

from cvt_track_study.config import ProjectLoader
from cvt_track_study.config.toml_io import dump_toml, load_toml


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "templates" / "project"


def copy_project(tmp_path: Path) -> Path:
    destination = tmp_path / "project"
    shutil.copytree(TEMPLATE, destination)
    return destination


def codes(result) -> set[str]:
    return {item.code for item in result.diagnostics}


def test_missing_uncertainty_is_error(tmp_path: Path) -> None:
    project = copy_project(tmp_path)
    path = project / "vehicles" / "vehicle_A" / "drivetrain.toml"
    raw = load_toml(path)
    del raw["drivetrain"]["final_drive_ratio"]["uncertainty"]
    dump_toml(raw, path)
    result = ProjectLoader().resolve(project)
    assert "INVALID_UNCERTAIN_INPUT" in codes(result)


def test_bare_physical_number_is_error(tmp_path: Path) -> None:
    project = copy_project(tmp_path)
    path = project / "vehicles" / "vehicle_A" / "vehicle.toml"
    raw = load_toml(path)
    raw["vehicle"]["wheelbase"] = 1.5
    dump_toml(raw, path)
    result = ProjectLoader().resolve(project)
    assert "BARE_PHYSICAL_NUMBER" in codes(result)


def test_wrong_unit_dimension_is_error(tmp_path: Path) -> None:
    project = copy_project(tmp_path)
    path = project / "vehicles" / "vehicle_A" / "vehicle.toml"
    raw = load_toml(path)
    raw["vehicle"]["mass"]["unit"] = "m"
    dump_toml(raw, path)
    result = ProjectLoader().resolve(project)
    assert "INVALID_UNCERTAIN_INPUT" in codes(result)


def test_fixed_without_reason_is_error(tmp_path: Path) -> None:
    project = copy_project(tmp_path)
    path = project / "vehicles" / "vehicle_A" / "drivetrain.toml"
    raw = load_toml(path)
    del raw["drivetrain"]["final_drive_ratio"]["uncertainty"]["reason"]
    dump_toml(raw, path)
    result = ProjectLoader().resolve(project)
    assert "INVALID_UNCERTAIN_INPUT" in codes(result)


def test_unsupported_telemetry_run_is_error(tmp_path: Path) -> None:
    project = copy_project(tmp_path)
    runs_path = project / "track" / "runs.toml"
    dump_toml(
        {
            "runs": [
                {
                    "file": "gpx/run.csv",
                    "vehicle_id": "vehicle_A",
                    "run_id": "A01",
                    "driver_id": "driver",
                    "use_for_centreline": True,
                    "use_for_gate_evidence": True,
                }
            ]
        },
        runs_path,
    )
    result = ProjectLoader().resolve(project)
    assert "RUN_FORMAT_UNSUPPORTED" in codes(result)


def test_grade_force_cannot_be_enabled_yet(tmp_path: Path) -> None:
    project = copy_project(tmp_path)
    track_path = project / "track" / "track.toml"
    raw = load_toml(track_path)
    raw["track"]["elevation"]["use_for_grade_force"] = True
    dump_toml(raw, track_path)
    result = ProjectLoader().resolve(project)
    assert "GRADE_FORCE_NOT_IMPLEMENTED" in codes(result)


def test_duplicate_profile_id_is_error(tmp_path: Path) -> None:
    project = copy_project(tmp_path)
    original = project / "profiles" / "vehicles" / "team_baja_defaults_v1.toml"
    duplicate = project / "profiles" / "vehicles" / "duplicate.toml"
    duplicate.write_text(original.read_text(encoding="utf-8"), encoding="utf-8")
    result = ProjectLoader().resolve(project)
    assert "DUPLICATE_PROFILE_ID" in codes(result)


def test_profile_inheritance_cycle_is_error(tmp_path: Path) -> None:
    project = copy_project(tmp_path)
    root = project / "profiles" / "vehicles"
    (root / "cycle_a.toml").write_text(
        '''
[profile]
id = "cycle.a"
scope = "vehicle"
version = 1
extends = ["cycle.b"]
[config]
'''.strip()
        + "\n",
        encoding="utf-8",
    )
    (root / "cycle_b.toml").write_text(
        '''
[profile]
id = "cycle.b"
scope = "vehicle"
version = 1
extends = ["cycle.a"]
[config]
'''.strip()
        + "\n",
        encoding="utf-8",
    )
    vehicle_path = project / "vehicles" / "vehicle_A" / "vehicle.toml"
    vehicle = load_toml(vehicle_path)
    vehicle["vehicle"]["profiles"] = ["cycle.a"]
    dump_toml(vehicle, vehicle_path)
    result = ProjectLoader().resolve(project)
    assert "PROFILE_INHERITANCE_CYCLE" in codes(result)


def test_event_coordinate_uncertainty_is_required(tmp_path: Path) -> None:
    project = copy_project(tmp_path)
    events = project / "track" / "events.toml"
    text = events.read_text(encoding="utf-8").replace(
        "horizontal_uncertainty_m = 10.0\n", ""
    )
    events.write_text(text, encoding="utf-8")
    result = ProjectLoader().resolve(project)
    assert any(
        item.code == "EVENT_COORDINATE_UNCERTAINTY_MISSING"
        for item in result.diagnostics
    )


def test_zero_event_coordinate_uncertainty_requires_reason(tmp_path: Path) -> None:
    project = copy_project(tmp_path)
    events = project / "track" / "events.toml"
    text = events.read_text(encoding="utf-8").replace(
        "horizontal_uncertainty_m = 10.0",
        "horizontal_uncertainty_m = 0.0",
    )
    events.write_text(text, encoding="utf-8")
    result = ProjectLoader().resolve(project)
    assert any(
        item.code == "FIXED_EVENT_COORDINATE_REASON_MISSING"
        for item in result.diagnostics
    )


def test_configured_lap_gate_must_exist(tmp_path: Path) -> None:
    project = copy_project(tmp_path)
    track = project / "track" / "track.toml"
    track.write_text(
        track.read_text(encoding="utf-8").replace(
            'lap_gate_event_id = "start_finish"',
            'lap_gate_event_id = "missing_gate"',
        ),
        encoding="utf-8",
    )
    result = ProjectLoader().resolve(project)
    assert any(item.code == "LAP_GATE_EVENT_NOT_FOUND" for item in result.diagnostics)


def test_event_extent_uncertainty_is_required(tmp_path: Path) -> None:
    project = copy_project(tmp_path)
    events = project / "track" / "events.toml"
    text = events.read_text(encoding="utf-8").replace(
        "before_anchor_uncertainty_m = 2.0\n", "", 1
    )
    events.write_text(text, encoding="utf-8")
    result = ProjectLoader().resolve(project)
    assert "EVENT_EXTENT_UNCERTAINTY_MISSING" in codes(result)


def test_explicit_event_endpoint_requires_uncertainty_and_source(tmp_path: Path) -> None:
    project = copy_project(tmp_path)
    events = project / "track" / "events.toml"
    with events.open("a", encoding="utf-8") as handle:
        handle.write(
            '''

[events.start]
latitude_deg = 0.00001
longitude_deg = 0.00001
'''
        )
    result = ProjectLoader().resolve(project)
    assert "EVENT_ENDPOINT_UNCERTAINTY_MISSING" in codes(result)
    assert "EVENT_ENDPOINT_SOURCE_MISSING" in codes(result)


def test_explicit_start_and_end_do_not_require_extent(tmp_path: Path) -> None:
    project = copy_project(tmp_path)
    events_path = project / "track" / "events.toml"
    raw = load_toml(events_path)
    event = raw["events"][0]
    event["start"] = {
        "latitude_deg": 0.00001,
        "longitude_deg": 0.00001,
        "horizontal_uncertainty_m": 1.0,
        "source": "synthetic surveyed start",
    }
    event["end"] = {
        "latitude_deg": -0.00001,
        "longitude_deg": -0.00001,
        "horizontal_uncertainty_m": 1.0,
        "source": "synthetic surveyed end",
    }
    del event["extent"]
    dump_toml(raw, events_path)

    result = ProjectLoader().resolve(project)
    assert "EVENT_EXTENT_MISSING" not in codes(result)
    assert result.error_count == 0


def test_speed_coverage_threshold_must_be_fraction(tmp_path: Path) -> None:
    project = copy_project(tmp_path)
    track_path = project / "track" / "track.toml"
    raw = load_toml(track_path)
    raw["track"]["reconstruction"]["minimum_speed_coverage_fraction"] = 1.2
    dump_toml(raw, track_path)
    result = ProjectLoader().resolve(project)
    assert "INVALID_SPEED_COVERAGE_THRESHOLD" in codes(result)


def test_unbounded_normal_with_material_impossible_tail_is_invalid(tmp_path: Path) -> None:
    project = copy_project(tmp_path)
    path = project / "vehicles" / "vehicle_A" / "vehicle.toml"
    raw = load_toml(path)
    raw["vehicle"]["mass"]["uncertainty"] = {
        "distribution": "normal",
        "standard_deviation": 200.0,
    }
    dump_toml(raw, path)
    result = ProjectLoader().resolve(project)
    assert "UNBOUNDED_NORMAL_PHYSICAL_SUPPORT" in codes(result)
    assert result.error_count > 0


def test_structural_sensitivity_can_target_track_surface_input(tmp_path: Path) -> None:
    project = copy_project(tmp_path)
    path = project / "studies" / "structural_sensitivity.toml"
    raw = load_toml(path)
    raw["sensitivity"]["parameters"] = ["track.surface.friction_coefficient"]
    dump_toml(raw, path)
    result = ProjectLoader().resolve(project)
    assert "SENSITIVITY_PATH_NOT_FOUND" not in codes(result)
    assert "SENSITIVITY_PARAMETER_FIXED" not in codes(result)


def test_structural_sensitivity_rejects_explicitly_fixed_parameter(tmp_path: Path) -> None:
    project = copy_project(tmp_path)
    path = project / "studies" / "structural_sensitivity.toml"
    raw = load_toml(path)
    raw["sensitivity"]["parameters"] = ["drivetrain.final_drive_ratio"]
    dump_toml(raw, path)
    result = ProjectLoader().resolve(project)
    assert "SENSITIVITY_PARAMETER_FIXED" in codes(result)


def test_positive_quantity_bounded_uncertainty_cannot_cross_zero(tmp_path: Path) -> None:
    project = copy_project(tmp_path)
    path = project / "vehicles" / "vehicle_A" / "vehicle.toml"
    raw = load_toml(path)
    raw["vehicle"]["mass"]["uncertainty"] = {
        "distribution": "uniform",
        "lower": -1.0,
        "upper": 300.0,
    }
    dump_toml(raw, path)
    result = ProjectLoader().resolve(project)
    assert "INVALID_UNCERTAIN_INPUT" in codes(result)


def test_cvt_ratio_uncertainty_supports_cannot_cross(tmp_path: Path) -> None:
    project = copy_project(tmp_path)
    path = project / "vehicles" / "vehicle_A" / "drivetrain.toml"
    raw = load_toml(path)
    raw["drivetrain"]["cvt"]["maximum_reduction_ratio"]["uncertainty"] = {
        "distribution": "uniform",
        "lower": 0.8,
        "upper": 3.6,
    }
    dump_toml(raw, path)
    result = ProjectLoader().resolve(project)
    assert "CVT_RATIO_UNCERTAINTY_OVERLAP" in codes(result)


def test_selected_structural_mode_requires_explicit_paths(tmp_path: Path) -> None:
    project = copy_project(tmp_path)
    path = project / "studies" / "full_uncertainty.toml"
    raw = load_toml(path)
    raw["sampling"]["mode"] = "selected_structural"
    raw["sampling"].pop("paths", None)
    dump_toml(raw, path)
    result = ProjectLoader().resolve(project)
    assert "SELECTED_STRUCTURAL_PATHS_MISSING" in codes(result)


def test_non_psd_study_correlation_matrix_is_rejected_during_validation(tmp_path: Path) -> None:
    project = copy_project(tmp_path)
    path = project / "studies" / "full_uncertainty.toml"
    raw = load_toml(path)
    raw["correlations"] = [
        {
            "id": "bad",
            "members": ["vehicle.mass", "vehicle.aero.drag_area"],
            "matrix": [[1.0, 2.0], [2.0, 1.0]],
        }
    ]
    dump_toml(raw, path)
    result = ProjectLoader().resolve(project)
    assert "NON_PSD_CORRELATION_MATRIX" in codes(result)


def test_selected_structural_rejects_non_structural_role(tmp_path: Path) -> None:
    project = copy_project(tmp_path)
    path = project / "studies" / "full_uncertainty.toml"
    raw = load_toml(path)
    raw["sampling"]["mode"] = "selected_structural"
    raw["sampling"]["paths"] = ["initial_conditions.vehicle_speed"]
    dump_toml(raw, path)
    result = ProjectLoader().resolve(project)
    assert "SELECTED_STRUCTURAL_PATH_FIXED" in codes(result)
    assert "SELECTED_STRUCTURAL_ROLE_MISMATCH" in codes(result)


def test_design_sweep_rejects_physically_invalid_values(tmp_path: Path) -> None:
    project = copy_project(tmp_path)
    path = project / "studies" / "gearing_sweep.toml"
    raw = load_toml(path)
    raw["design_variable"]["path"] = "drivetrain.final_drive_ratio"
    raw["design_variable"]["values"] = [7.556, 0.0]
    dump_toml(raw, path)
    result = ProjectLoader().resolve(project)
    assert "DESIGN_VALUE_OUT_OF_DOMAIN" in codes(result)


def test_cvt_ratio_sweep_rejects_crossed_bounds(tmp_path: Path) -> None:
    project = copy_project(tmp_path)
    path = project / "studies" / "gearing_sweep.toml"
    raw = load_toml(path)
    raw["design_variable"]["path"] = "drivetrain.cvt.minimum_reduction_ratio"
    raw["design_variable"]["values"] = [0.9, 4.0]
    dump_toml(raw, path)
    result = ProjectLoader().resolve(project)
    assert "DESIGN_VALUE_OUT_OF_DOMAIN" in codes(result)


def test_correlation_members_must_exist_and_be_sampled_by_study_mode(tmp_path: Path) -> None:
    project = copy_project(tmp_path)
    path = project / "studies" / "track_robustness.toml"
    raw = load_toml(path)
    raw["correlations"] = [
        {
            "id": "bad_members",
            "members": ["vehicle.mass", "vehicle.not_real"],
            "matrix": [[1.0, 0.5], [0.5, 1.0]],
        }
    ]
    dump_toml(raw, path)
    result = ProjectLoader().resolve(project)
    assert "CORRELATION_MEMBER_NOT_FOUND" in codes(result)
    assert "CORRELATION_MEMBER_NOT_SAMPLED" in codes(result)


def test_design_variable_cannot_remain_in_sampling_correlation_group(tmp_path: Path) -> None:
    project = copy_project(tmp_path)
    path = project / "studies" / "gearing_sweep.toml"
    raw = load_toml(path)
    raw["sampling"]["mode"] = "all_declared"
    raw["correlations"] = [
        {
            "id": "drivetrain_pair",
            "members": ["drivetrain.final_drive_ratio", "drivetrain.efficiency"],
            "matrix": [[1.0, 0.2], [0.2, 1.0]],
        }
    ]
    dump_toml(raw, path)
    result = ProjectLoader().resolve(project)
    assert "CORRELATION_MEMBER_NOT_SAMPLED" in codes(result)


def test_study_random_seed_must_be_nonnegative_integer(tmp_path: Path) -> None:
    project = copy_project(tmp_path)
    path = project / "studies" / "track_robustness.toml"
    raw = load_toml(path)
    raw["study"]["random_seed"] = -1
    dump_toml(raw, path)
    result = ProjectLoader().resolve(project)
    assert "INVALID_RANDOM_SEED" in codes(result)
