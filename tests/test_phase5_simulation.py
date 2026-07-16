from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from cvt_track_study.bundle import TrackBundle, build_track_bundle
from cvt_track_study.config import ProjectLoader
from cvt_track_study.config.uncertainty import UncertaintyValidationError
from cvt_track_study.contracts.obstacles import validate_obstacle_model_contract
from cvt_track_study.simulation.dynamics import driver_command
from cvt_track_study.simulation.integrator import (
    _record_feature_entry_crossings,
    implicit_tire_slip_step,
)
from cvt_track_study.simulation.models import (
    CVTModel,
    DriverModel,
    EngineModel,
    SimulationSettings,
)
from cvt_track_study.simulation.obstacles import (
    FixedSpecificEnergyLoss,
    NoObstacle,
    ObstacleContext,
    SmoothProfileObstacle,
    SpeedQuadraticEnergyLoss,
    raised_cosine_density,
)
from cvt_track_study.simulation.powertrain import evaluate_powertrain
from cvt_track_study.simulation.service import (
    resolve_simulation_cases,
    run_baseline_project,
)
from cvt_track_study.simulation.track import (
    RuntimeFeature,
    RuntimeInterval,
    RuntimeSpeedGate,
    RuntimeTrack,
)
from cvt_track_study.track import build_project_track

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "examples" / "reference_project"


def fixed_quantity(nominal: float, unit: str) -> dict[str, object]:
    return {
        "nominal": nominal,
        "unit": unit,
        "source": {"kind": "engineering_estimate", "reference": "unit test"},
        "uncertainty": {"distribution": "fixed", "reason": "unit-test constant"},
    }


def fixed_choice(nominal: str) -> dict[str, object]:
    return {
        "nominal": nominal,
        "source": {"kind": "engineering_estimate", "reference": "unit test"},
        "uncertainty": {"distribution": "fixed", "reason": "unit-test choice"},
    }


def simple_track(*, gate_speed_mps: float | None = None) -> RuntimeTrack:
    gates: tuple[RuntimeSpeedGate, ...] = ()
    if gate_speed_mps is not None:
        gates = (
            RuntimeSpeedGate(
                identifier="gate:test",
                response_group_id="test",
                name="test",
                position_s_m=50.0,
                target_speed_mps=gate_speed_mps,
                confidence_score=90.0,
            ),
        )
    return RuntimeTrack(
        name="test",
        length_m=100.0,
        closed_course=True,
        centreline_s_m=(0.0, 100.0),
        centreline_x_m=(0.0, 0.0),
        centreline_y_m=(0.0, 0.0),
        centreline_curvature_1_per_m=(0.0, 0.0),
        reference_elevation_m=(None, None),
        surface_friction_coefficient=0.8,
        features=(),
        speed_gates=gates,
        gpx_grade_force_enabled=False,
    )


def test_obstacle_contract_requires_explicit_none() -> None:
    with pytest.raises(UncertaintyValidationError, match="status must be 'declared'"):
        validate_obstacle_model_contract(
            {"status": "undeclared", "model_type": None, "parameters": {}}
        )
    model_type, parameters = validate_obstacle_model_contract(
        {"status": "declared", "model_type": fixed_choice("none"), "parameters": {}}
    )
    assert model_type == "none"
    assert parameters == {}


def test_obstacle_contract_rejects_wrong_units_and_extra_parameters() -> None:
    raw = {
        "status": "declared",
        "model_type": fixed_choice("distributed_resistance"),
        "parameters": {"resistance_force": fixed_quantity(100.0, "m")},
    }
    with pytest.raises(UncertaintyValidationError, match="expected .force."):
        validate_obstacle_model_contract(raw)
    raw["parameters"] = {
        "resistance_force": fixed_quantity(100.0, "N"),
        "mystery": fixed_quantity(1.0, "1"),
    }
    with pytest.raises(UncertaintyValidationError, match="unexpected mystery"):
        validate_obstacle_model_contract(raw)


def test_raised_cosine_energy_model_integrates_to_declared_energy() -> None:
    length = 8.0
    x = np.linspace(0.0, length, 20001)
    density = np.asarray([raised_cosine_density(value, length) for value in x])
    assert np.trapezoid(density, x) == pytest.approx(1.0, rel=1e-8)

    model = FixedSpecificEnergyLoss(specific_energy_loss_j_per_kg=3.0)
    context = ObstacleContext(
        local_distance_m=0.0,
        interval_length_m=length,
        vehicle_speed_mps=5.0,
        entry_speed_mps=5.0,
        vehicle_mass_kg=250.0,
        gravity_mps2=9.80665,
    )
    force = np.asarray(
        [model.evaluate(replace(context, local_distance_m=value)).resistance_force_n for value in x]
    )
    assert np.trapezoid(force, x) == pytest.approx(750.0, rel=1e-8)


def test_smooth_profile_is_conservative_when_dissipation_is_zero() -> None:
    model = SmoothProfileObstacle(
        vertical_amplitude_m=0.2,
        specific_fixed_energy_j_per_kg=0.0,
        impact_coefficient_kg=0.0,
        traction_multiplier=1.0,
        minimum_normal_load_scale=0.0,
        maximum_normal_load_scale=4.0,
    )
    base = ObstacleContext(
        local_distance_m=0.0,
        interval_length_m=10.0,
        vehicle_speed_mps=4.0,
        entry_speed_mps=4.0,
        vehicle_mass_kg=250.0,
        gravity_mps2=9.80665,
    )
    start = model.evaluate(base)
    crest = model.evaluate(replace(base, local_distance_m=5.0))
    end = model.evaluate(replace(base, local_distance_m=10.0))
    assert start.elevation_offset_m == pytest.approx(0.0)
    assert crest.elevation_offset_m == pytest.approx(0.2)
    assert end.elevation_offset_m == pytest.approx(0.0, abs=1e-12)
    assert start.resistance_force_n == 0.0
    assert crest.resistance_force_n == 0.0


def test_speed_quadratic_obstacle_uses_entry_speed_not_local_slowdown() -> None:
    model = SpeedQuadraticEnergyLoss(
        specific_fixed_energy_j_per_kg=0.0, impact_coefficient_kg=5.0
    )
    base = ObstacleContext(
        local_distance_m=4.0,
        interval_length_m=8.0,
        vehicle_speed_mps=2.0,
        entry_speed_mps=6.0,
        vehicle_mass_kg=250.0,
        gravity_mps2=9.80665,
    )
    force = model.evaluate(base).resistance_force_n
    expected_energy = 5.0 * 6.0**2
    assert force == pytest.approx(
        expected_energy * raised_cosine_density(4.0, 8.0)
    )
    assert model.evaluate(replace(base, vehicle_speed_mps=0.5)).resistance_force_n == pytest.approx(force)


def test_feature_entry_speed_is_interpolated_at_simulated_boundary_crossing() -> None:
    feature = RuntimeFeature(
        identifier="rocks",
        name="Rocks",
        response_group_id="rocks",
        interval=RuntimeInterval(10.0, 20.0, 10.0, False),
        model=NoObstacle(),
    )
    recorded: dict[str, float] = {}
    _record_feature_entry_crossings(
        features=(feature,),
        recorded=recorded,
        start_distance_m=9.0,
        end_distance_m=11.0,
        start_speed_mps=8.0,
        end_speed_mps=6.0,
    )
    assert recorded["rocks"] == pytest.approx(7.0)


def test_gate_is_a_one_way_ceiling_not_a_speed_reset() -> None:
    track = simple_track(gate_speed_mps=5.0)
    driver = DriverModel(
        maximum_braking_deceleration_mps2=4.0,
        maximum_brake_force_n=1000.0,
        braking_trigger_margin_mps=0.1,
    )
    slow = driver_command(
        track=track,
        distance_m=49.0,
        vehicle_speed_mps=2.0,
        driver=driver,
        braking_deceleration_mps2=4.0,
    )
    fast = driver_command(
        track=track,
        distance_m=49.0,
        vehicle_speed_mps=6.0,
        driver=driver,
        braking_deceleration_mps2=4.0,
    )
    assert slow.throttle == 1.0
    assert slow.brake_force_n == 0.0
    assert fast.throttle == 0.0
    assert fast.brake_force_n == 1000.0


def test_bounded_and_infinite_powertrain_modes() -> None:
    engine = EngineModel.baja_br10_reference(target_rpm=3000.0, power_scale=1.0)
    cvt = CVTModel(
        minimum_reduction_ratio=0.9,
        maximum_reduction_ratio=3.5,
        final_drive_ratio=7.556,
        efficiency=0.8,
    )
    settings = SimulationSettings(
        maximum_time_s=10.0,
        integration_step_s=0.001,
        report_step_s=0.02,
        initial_vehicle_speed_mps=0.0,
        initial_wheel_speed_rad_s=0.0,
    )
    launch = evaluate_powertrain(
        wheel_speed_rad_s=0.0,
        throttle=1.0,
        engine=engine,
        cvt=cvt,
        infinite_cvt=False,
    )
    high_speed = evaluate_powertrain(
        wheel_speed_rad_s=100.0,
        throttle=1.0,
        engine=engine,
        cvt=cvt,
        infinite_cvt=False,
    )
    reference_launch = evaluate_powertrain(
        wheel_speed_rad_s=0.0,
        throttle=1.0,
        engine=engine,
        cvt=cvt,
        infinite_cvt=True,
    )
    reference = evaluate_powertrain(
        wheel_speed_rad_s=20.0,
        throttle=1.0,
        engine=engine,
        cvt=cvt,
        infinite_cvt=True,
    )
    assert launch.mode == "maximum_ratio_clutch"
    assert launch.clutch_loss_power_w > 0.0
    assert high_speed.mode == "minimum_ratio_synchronous"
    assert reference_launch.mode == "infinite_launch_clutch"
    assert reference_launch.clutch_loss_power_w > 0.0
    assert reference.mode == "infinite_target_rpm"
    assert reference.engine_speed_rpm == pytest.approx(3000.0)


def test_design_candidates_resolve_to_one_scenario_level_infinite_reference(
    tmp_path: Path,
) -> None:
    resolution = ProjectLoader().resolve(REFERENCE, study="final_drive_sweep")
    assert resolution.is_valid
    build = build_project_track(
        REFERENCE, output_directory=tmp_path / "track_build"
    )
    bundle = TrackBundle(build_track_bundle(build))
    vehicle_raw = resolution.data["vehicles"]["vehicle_A"]
    base_study_raw = resolution.data["studies"]["baseline"]
    common = {
        "vehicle_id": "vehicle_A",
        "vehicle_raw": vehicle_raw,
        "study_raw": base_study_raw,
        "track_raw": resolution.data["track"],
        "bundle": bundle,
        "shared_reference": True,
    }
    bounded_low, reference_low, _, _ = resolve_simulation_cases(
        **common,
        design_values_si={"drivetrain.final_drive_ratio": 6.0},
    )
    bounded_high, reference_high, _, _ = resolve_simulation_cases(
        **common,
        design_values_si={"drivetrain.final_drive_ratio": 9.0},
    )

    assert bounded_low.cvt.final_drive_ratio == pytest.approx(6.0)
    assert bounded_high.cvt.final_drive_ratio == pytest.approx(9.0)
    assert reference_low.cvt == reference_high.cvt
    assert reference_low.cvt.infinite_launch_wheel_torque_cap_nm is not None


def test_implicit_slip_root_satisfies_backward_euler_residual() -> None:
    slip, force = implicit_tire_slip_step(
        previous_slip_speed_mps=-0.4,
        step_s=0.001,
        free_slip_acceleration_mps2=500.0,
        tire_force_coefficient=0.12,
        tire_limit_n=1000.0,
        tire_stiffness_n_per_mps=2500.0,
    )
    residual = slip - (-0.4) - 0.001 * (500.0 - 0.12 * force)
    assert abs(residual) < 1e-7
    assert abs(force) <= 1000.0


def test_reference_project_baseline_closes_energy_and_meets_gates(tmp_path: Path) -> None:
    project = tmp_path / "project"
    shutil.copytree(REFERENCE, project)
    # A 2 ms test step exercises the same mechanism while keeping the regression
    # suite fast.  The packaged example retains the 1 ms engineering default.
    study = project / "studies" / "baseline.toml"
    study.write_text(
        study.read_text(encoding="utf-8").replace(
            "integration_step_s = 0.001", "integration_step_s = 0.002"
        ),
        encoding="utf-8",
    )
    output = run_baseline_project(project, output_directory=tmp_path / "baseline")
    comparison = json.loads((output / "comparison_summary.json").read_text())
    manifest = json.loads((output / "run_manifest.json").read_text())
    assert manifest["track_bundle"] == "track_bundle.json"
    assert (output / "track_bundle.json").is_file()
    assert (output / "track_bundle.sha256").is_file()
    assert comparison["reference_dominance_pass"] is True
    assert comparison["reference_shared_launch_loss_energy_kj"] > 0.0
    assert comparison["finite_ratio_opportunity_loss_energy_kj"] == pytest.approx(
        comparison["bounded_total_opportunity_loss_energy_kj"]
        - comparison["reference_shared_launch_loss_energy_kj"]
    )
    assert abs(comparison["bounded_energy_balance_relative_error"]) < 0.01
    assert abs(comparison["reference_energy_balance_relative_error"]) < 0.01
    rows = list(__import__("csv").DictReader((output / "gate_compliance.csv").open()))
    assert rows
    assert all(row["bounded_compliant_0p5_kmh"] == "True" for row in rows)
    assert all(row["reference_compliant_0p5_kmh"] == "True" for row in rows)
    bounded_summary = json.loads((output / "bounded_summary.json").read_text())
    reference_summary = json.loads(
        (output / "infinite_reference_summary.json").read_text()
    )
    assert abs(bounded_summary["powertrain_energy_balance_relative_error"]) < 0.01
    assert abs(reference_summary["powertrain_energy_balance_relative_error"]) < 0.01
    assert sum(bounded_summary["obstacle_energy_by_feature_kj"].values()) == pytest.approx(
        bounded_summary["obstacle_loss_energy_kj"], rel=1e-10
    )
    feature_rows = list(
        __import__("csv").DictReader(
            (output / "obstacle_energy_by_feature.csv").open()
        )
    )
    assert feature_rows
    assert all(row["bounded_entry_speed_mps"] != "" for row in feature_rows)
