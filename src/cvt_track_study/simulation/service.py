"""Project-level Phase 5 baseline simulation service."""

from __future__ import annotations

from datetime import datetime, timezone
import shutil
from uuid import uuid4
from pathlib import Path
from typing import Any, Mapping

from cvt_track_study.bundle import TrackBundle, load_track_bundle
from cvt_track_study.config import ProjectError, ProjectLoader
from cvt_track_study.runtime.provenance import (
    build_provenance,
    canonical_fingerprint,
    write_provenance,
)
from cvt_track_study.runtime.results import write_results_index
from cvt_track_study.track import build_project_track

from .integrator import SimulationTrace, run_simulation
from .metrics import compare_summaries, summarize_trace
from .models import (
    CVTModel,
    DriverModel,
    EngineModel,
    RAD_PER_SECOND_TO_RPM,
    SimulationInputError,
    SimulationSettings,
    StudyCase,
    TireModel,
    VehicleModel,
    choice_nominal,
    quantity_nominal,
    quantity_si,
)
from .reporting import write_baseline_outputs
from .reporting_v8 import write_baseline_hierarchy
from .track import RuntimeTrack, runtime_track_from_bundle


class SimulationError(RuntimeError):
    """Raised when a project cannot be simulated under the Phase 5 contract."""


def run_baseline_project(
    project: str | Path,
    *,
    study: str = "baseline",
    bundle_path: Path | None = None,
    output_directory: Path | None = None,
    command: tuple[str, ...] = (),
) -> Path:
    """Run one bounded design and its otherwise-identical infinite reference.

    When ``bundle_path`` is omitted, the track is rebuilt first.  This default
    avoids silently simulating stale evidence.  Passing a validated bundle is the
    explicit cache/reproducibility path.
    """

    resolution = ProjectLoader().resolve(project, study=study)
    if resolution.error_count:
        details = "\n".join(item.format() for item in resolution.diagnostics)
        raise SimulationError(f"Project validation failed:\n{details}")
    study_raw = resolution.data["studies"].get(study)
    if not isinstance(study_raw, Mapping):
        raise SimulationError(f"Study {study!r} was not found.")
    if str(study_raw.get("study", {}).get("type", "")) != "baseline":
        raise SimulationError(f"Study {study!r} is not a baseline study.")

    if bundle_path is None:
        build = build_project_track(project)
        if build.error_count:
            raise SimulationError("Track build failed before simulation.")
        bundle_path = build.output_directory / "track_bundle.json"
    bundle = load_track_bundle(bundle_path)

    vehicle_id = str(study_raw["study"]["vehicle_id"])
    vehicle_raw = resolution.data["vehicles"].get(vehicle_id)
    if not isinstance(vehicle_raw, Mapping):
        raise SimulationError(f"Vehicle {vehicle_id!r} is not resolved.")
    bounded_case, reference_case, settings, runtime_track = _resolve_nominal_cases(
        vehicle_id=vehicle_id,
        vehicle_raw=vehicle_raw,
        study_raw=study_raw,
        track_raw=resolution.data["track"],
        bundle=bundle,
    )

    output = output_directory or (
        resolution.paths.results_directory / "baseline" / _timestamp()
    )
    output = output.resolve()
    if output.exists():
        raise SimulationError(f"Output directory already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(f".{output.name}.tmp-{uuid4().hex}")
    staging.mkdir(parents=False, exist_ok=False)
    try:
        resolution.export(staging / "resolved_inputs")
        bounded = run_simulation(
            case=bounded_case, track=runtime_track, settings=settings
        )
        reference = run_simulation(
            case=reference_case, track=runtime_track, settings=settings
        )
        bounded_summary = summarize_trace(
            bounded,
            target_engine_rpm=bounded_case.engine.target_rpm,
            target_power_w=bounded_case.engine.target_power_w,
        )
        reference_summary = summarize_trace(
            reference,
            target_engine_rpm=reference_case.engine.target_rpm,
            target_power_w=reference_case.engine.target_power_w,
        )
        comparison = compare_summaries(bounded_summary, reference_summary)
        write_baseline_outputs(
            output=staging,
            bounded=bounded,
            reference=reference,
            bounded_summary=bounded_summary,
            reference_summary=reference_summary,
            comparison=comparison,
            bounded_case=bounded_case,
            settings=settings,
            track=runtime_track,
            bundle=bundle,
            bundle_path=bundle_path,
            study_name=study,
            vehicle_id=vehicle_id,
        )
        manifest_path = staging / "run_manifest.json"
        manifest = __import__("json").loads(manifest_path.read_text(encoding="utf-8"))
        write_baseline_hierarchy(
            output=staging,
            bounded=bounded_summary,
            reference=reference_summary,
            comparison=comparison,
            manifest=manifest,
        )
        study_fingerprint = canonical_fingerprint(
            {
                "schema": "framework-baseline-v0.8",
                "study_name": study,
                "study": study_raw,
                "vehicle": vehicle_raw,
                "track": resolution.data["track"],
                "bundle_content_fingerprint": bundle.data.get(
                    "content_fingerprint_sha256"
                ),
            }
        )
        write_provenance(
            staging,
            build_provenance(
                command=command
                or ("drivetrain-study", "run", "baseline", str(resolution.paths.root)),
                project=resolution.paths.root,
                bundle_path=staging / "track_bundle.json",
                study_name=study,
                study_fingerprint=study_fingerprint,
                resolved_configuration_fingerprint=canonical_fingerprint(resolution.data),
            ),
        )
        staging.replace(output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    write_results_index(resolution.paths.results_directory)
    return output


def resolve_simulation_cases(
    *,
    vehicle_id: str,
    vehicle_raw: Mapping[str, Any],
    study_raw: Mapping[str, Any],
    track_raw: Mapping[str, Any],
    bundle: TrackBundle,
    quantity_values_si: Mapping[str, float] | None = None,
    choice_values: Mapping[str, str] | None = None,
    gate_target_speeds_mps: Mapping[str, float] | None = None,
    design_values_si: Mapping[str, float] | None = None,
) -> tuple[StudyCase, StudyCase, SimulationSettings, RuntimeTrack]:
    quantities = dict(quantity_values_si or {})
    quantities.update(design_values_si or {})
    choices = dict(choice_values or {})

    def qsi(path: str, raw: Mapping[str, Any], dimension: str) -> float:
        return float(quantities.get(path, quantity_si(raw, dimension)))

    def qnom(path: str, raw: Mapping[str, Any], dimension: str = "dimensionless") -> float:
        return float(quantities.get(path, quantity_nominal(raw, dimension)))

    def cval(path: str, raw: Mapping[str, Any]) -> str:
        return str(choices.get(path, choice_nominal(raw)))

    try:
        v = _mapping(vehicle_raw, "vehicle")
        drivetrain = _mapping(vehicle_raw, "drivetrain")
        cvt_raw = _mapping(drivetrain, "cvt")
        engine_raw = _mapping(drivetrain, "engine")
        tire_raw = _mapping(v, "tire")
        aero_raw = _mapping(v, "aero")
        mass = qsi("vehicle.mass", _mapping(v, "mass"), "mass")
        diameter = qsi("vehicle.tire_diameter", _mapping(v, "tire_diameter"), "length")
        vehicle = VehicleModel(
            mass_kg=mass,
            wheel_radius_m=0.5 * diameter,
            wheel_rotational_inertia_kg_m2=qsi(
                "vehicle.wheel_rotational_inertia",
                _mapping(v, "wheel_rotational_inertia"),
                "rotational_inertia",
            ),
            driven_normal_load_fraction=qnom(
                "vehicle.driven_normal_load_fraction",
                _mapping(v, "driven_normal_load_fraction"),
            ),
            drag_area_m2=qsi("vehicle.aero.drag_area", _mapping(aero_raw, "drag_area"), "area"),
            air_density_kg_m3=qsi("vehicle.aero.air_density", _mapping(aero_raw, "air_density"), "density"),
            rolling_resistance_coefficient=qnom(
                "vehicle.rolling_resistance_coefficient",
                _mapping(v, "rolling_resistance_coefficient"),
            ),
            gravity_mps2=qsi("vehicle.gravity", _mapping(v, "gravity"), "acceleration"),
        )
        tire = TireModel(
            peak_traction_scale=qnom("vehicle.tire.peak_traction_scale", _mapping(tire_raw, "peak_traction_scale")),
            slip_stiffness_n_per_mps=qsi(
                "vehicle.tire.slip_stiffness",
                _mapping(tire_raw, "slip_stiffness"),
                "slip_stiffness",
            ),
        )
        engine_model = cval("drivetrain.engine.model", _mapping(engine_raw, "model"))
        target_rpm = qsi(
            "drivetrain.engine.target_speed",
            _mapping(engine_raw, "target_speed"),
            "angular_speed",
        ) * RAD_PER_SECOND_TO_RPM
        power_scale = qnom("drivetrain.engine.power_scale", _mapping(engine_raw, "power_scale"))
        if engine_model != "baja_br10_reference_v1":
            raise SimulationInputError(f"Unsupported engine model {engine_model!r}.")
        engine = EngineModel.baja_br10_reference(
            target_rpm=target_rpm, power_scale=power_scale
        )
        launch_model = cval("drivetrain.cvt.launch_clutch_model", _mapping(cvt_raw, "launch_clutch_model"))
        if launch_model not in {"ideal_slip", "disabled"}:
            raise SimulationInputError(f"Unsupported launch clutch model {launch_model!r}.")
        cvt = CVTModel(
            minimum_reduction_ratio=qnom(
                "drivetrain.cvt.minimum_reduction_ratio",
                _mapping(cvt_raw, "minimum_reduction_ratio"),
            ),
            maximum_reduction_ratio=qnom(
                "drivetrain.cvt.maximum_reduction_ratio",
                _mapping(cvt_raw, "maximum_reduction_ratio"),
            ),
            final_drive_ratio=qnom("drivetrain.final_drive_ratio", _mapping(drivetrain, "final_drive_ratio")),
            efficiency=qnom("drivetrain.efficiency", _mapping(drivetrain, "efficiency")),
            ideal_launch_clutch=launch_model == "ideal_slip",
        )
        driver_raw = _mapping(study_raw, "driver")
        driver = DriverModel(
            maximum_braking_deceleration_mps2=qsi(
                "driver.maximum_braking_deceleration",
                _mapping(driver_raw, "maximum_braking_deceleration"),
                "acceleration",
            ),
            maximum_brake_force_n=qsi(
                "driver.maximum_brake_force",
                _mapping(driver_raw, "maximum_brake_force"),
                "force",
            ),
            braking_trigger_margin_mps=qsi(
                "driver.braking_trigger_margin",
                _mapping(driver_raw, "braking_trigger_margin"),
                "speed",
            ),
        )
        numerical = _mapping(study_raw, "simulation")
        initial = _mapping(study_raw, "initial_conditions")
        settings = SimulationSettings(
            maximum_time_s=float(numerical["maximum_time_s"]),
            integration_step_s=float(numerical["integration_step_s"]),
            report_step_s=float(numerical["report_step_s"]),
            initial_vehicle_speed_mps=qsi(
                "initial_conditions.vehicle_speed",
                _mapping(initial, "vehicle_speed"),
                "speed",
            ),
            initial_wheel_speed_rad_s=qsi(
                "initial_conditions.wheel_speed",
                _mapping(initial, "wheel_speed"),
                "angular_speed",
            ),
        )
        surface = _mapping(_mapping(track_raw, "surface"), "friction_coefficient")
        realization = _mapping(study_raw, "track_realization")
        runtime_track = runtime_track_from_bundle(
            bundle,
            surface_friction_coefficient=qnom(
                "track.surface.friction_coefficient", surface
            ),
            gate_speed_statistic=str(realization.get("gate_speed_statistic", "median")),
            gate_target_speeds_mps=gate_target_speeds_mps,
            obstacle_model_types=_obstacle_model_types(choices),
            obstacle_parameters_si=_obstacle_parameters(quantities, choices, bundle),
        )
    except (KeyError, TypeError, ValueError, SimulationInputError) as exc:
        raise SimulationError(f"Could not resolve nominal simulation case: {exc}") from exc
    bounded = StudyCase(
        name=f"{vehicle_id}_bounded", engine=engine, vehicle=vehicle, tire=tire,
        cvt=cvt, driver=driver, infinite_cvt=False,
    )
    reference = StudyCase(
        name=f"{vehicle_id}_infinite_reference", engine=engine, vehicle=vehicle,
        tire=tire, cvt=cvt, driver=driver, infinite_cvt=True,
    )
    return bounded, reference, settings, runtime_track


def _resolve_nominal_cases(
    *,
    vehicle_id: str,
    vehicle_raw: Mapping[str, Any],
    study_raw: Mapping[str, Any],
    track_raw: Mapping[str, Any],
    bundle: TrackBundle,
) -> tuple[StudyCase, StudyCase, SimulationSettings, RuntimeTrack]:
    return resolve_simulation_cases(
        vehicle_id=vehicle_id,
        vehicle_raw=vehicle_raw,
        study_raw=study_raw,
        track_raw=track_raw,
        bundle=bundle,
    )


def _obstacle_model_types(choices: Mapping[str, str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for path, value in choices.items():
        parts = path.split(".")
        if len(parts) == 3 and parts[0] == "obstacle" and parts[2] == "model_type":
            result[parts[1]] = value
    return result


def _obstacle_parameters(
    quantities: Mapping[str, float],
    choices: Mapping[str, str],
    bundle: TrackBundle,
) -> dict[str, dict[str, float]]:
    selected_types = _obstacle_model_types(choices)
    nominal_types = {
        str(feature["id"]): str(feature["obstacle_model"]["model_type"]["nominal"])
        for feature in bundle.physical_features
    }
    result: dict[str, dict[str, float]] = {}
    for path, value in quantities.items():
        parts = path.split(".")
        if len(parts) != 4 or parts[0] != "obstacle":
            continue
        _, feature_id, model_type, parameter = parts
        selected = selected_types.get(feature_id, nominal_types.get(feature_id))
        if model_type == selected:
            result.setdefault(feature_id, {})[parameter] = float(value)
    return result

def _mapping(raw: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = raw.get(key)
    if not isinstance(value, Mapping):
        raise SimulationInputError(f"Required configuration table {key!r} is missing.")
    return value


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
