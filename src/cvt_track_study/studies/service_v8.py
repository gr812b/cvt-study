"""Resumable, cached, parallel study orchestration for the Phase 8 framework."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping

from cvt_track_study.bundle import TrackBundle, load_track_bundle
from cvt_track_study.config import ProjectLoader
from cvt_track_study.runtime import (
    ProgressReporter,
    ResultWorkspace,
    SimulationCache,
    assess_evidence,
)
from cvt_track_study.runtime.provenance import (
    build_provenance,
    canonical_fingerprint,
    write_provenance,
)
from cvt_track_study.runtime.results import write_results_index
from cvt_track_study.simulation.integrator import run_simulation
from cvt_track_study.simulation.metrics import (
    compare_summaries,
    gate_compliance_rows,
    summarize_trace,
)
from cvt_track_study.simulation.service import SimulationError, resolve_simulation_cases
from cvt_track_study.track import build_project_track
from cvt_track_study.uncertainty import (
    SamplingPlan,
    ScenarioDraw,
    ScenarioSampler,
    build_input_registry,
    correlation_groups_from_study,
)

from . import service as phase6_service
from .analysis import (
    METRICS,
    convergence_summary,
    input_contracts,
    quality_summary,
    summarize_study,
)
from .model import DesignPoint, StudyExecution
from .planning import reference_cache_key, study_plan
from .reporting_v8 import write_study_outputs

_SUPPORTED_TYPES = {
    "design_sweep",
    "track_robustness",
    "structural_sensitivity",
    "full_uncertainty",
}


def run_study_project(
    project: str | Path,
    *,
    study: str,
    bundle_path: Path | None = None,
    output_directory: Path | None = None,
    replicates_override: int | None = None,
    workers: int = 1,
    resume: bool = False,
    restart: bool = False,
    use_cache: bool = True,
    progress: bool = True,
    run_name: str | None = None,
    command: tuple[str, ...] = (),
) -> Path:
    if workers < 1:
        raise SimulationError("workers must be at least one.")
    resolution = ProjectLoader().resolve(project, study=study)
    if resolution.error_count:
        details = "\n".join(item.format() for item in resolution.diagnostics)
        raise SimulationError(f"Project validation failed:\n{details}")
    study_raw = resolution.data["studies"].get(study)
    if not isinstance(study_raw, Mapping):
        raise SimulationError(f"Study {study!r} was not found.")
    study_type = str(study_raw.get("study", {}).get("type", ""))
    if study_type not in _SUPPORTED_TYPES:
        raise SimulationError(
            f"Study {study!r} has type {study_type!r}; expected one of {sorted(_SUPPORTED_TYPES)}."
        )
    base_name = str(study_raw.get("base_case", {}).get("study", "baseline"))
    base_study = resolution.data["studies"].get(base_name)
    if not isinstance(base_study, Mapping) or str(
        base_study.get("study", {}).get("type")
    ) != "baseline":
        raise SimulationError(
            f"Study {study!r} requires base_case.study to reference a baseline study."
        )
    vehicle_id = str(study_raw["study"]["vehicle_id"])
    vehicle_raw = resolution.data["vehicles"].get(vehicle_id)
    if not isinstance(vehicle_raw, Mapping):
        raise SimulationError(f"Vehicle {vehicle_id!r} is not resolved.")

    if bundle_path is None:
        build = build_project_track(project)
        if build.error_count:
            raise SimulationError("Track build failed before study execution.")
        bundle_path = build.output_directory / "track_bundle.json"
    bundle_path = bundle_path.resolve()
    bundle = load_track_bundle(bundle_path)
    registry = build_input_registry(
        vehicle_raw=vehicle_raw,
        base_study_raw=base_study,
        track_raw=resolution.data["track"],
        bundle=bundle,
    )
    fingerprint = canonical_fingerprint(
        {
            "schema": "framework-study-v0.8",
            "study_name": study,
            "study": study_raw,
            "base_study": base_study,
            "vehicle": vehicle_raw,
            "track": resolution.data["track"],
            "bundle_content_fingerprint": bundle.data.get("content_fingerprint_sha256"),
            "replicates_override": replicates_override,
        }
    )
    output = (
        output_directory.resolve()
        if output_directory is not None
        else (
            resolution.paths.results_directory
            / study_type
            / f"{_safe_name(run_name or study)}--{fingerprint[:10]}"
        ).resolve()
    )
    try:
        workspace = ResultWorkspace(
            output,
            fingerprint=fingerprint,
            resume=resume,
            restart=restart,
        )
    except RuntimeError as exc:
        raise SimulationError(str(exc)) from exc
    cache = SimulationCache(
        resolution.paths.root / ".drivetrain-study-cache" / "simulations",
        enabled=use_cache,
    )

    execution = _execute(
        study_name=study,
        study_type=study_type,
        study_raw=study_raw,
        base_study=base_study,
        vehicle_id=vehicle_id,
        vehicle_raw=vehicle_raw,
        track_raw=resolution.data["track"],
        bundle=bundle,
        registry=registry,
        replicates_override=replicates_override,
        workspace=workspace,
        cache=cache,
        workers=workers,
        progress=progress,
        study_fingerprint=fingerprint,
        evidence_assessment=assess_evidence(
            diagnostics=resolution.diagnostics,
            bundle=bundle,
        ),
    )
    resolution.export(workspace.path / "resolved_inputs")
    phase6_service._write_bundle_snapshot(workspace.path, bundle, bundle_path)
    write_study_outputs(
        output=workspace.path,
        rows=execution.rows,
        scenario_draws=execution.scenario_draws,
        summary=execution.summary,
        convergence=execution.convergence,
        manifest=execution.manifest,
        input_contracts=execution.input_contracts,
        study_type=study_type,
    )
    provenance = build_provenance(
        command=command
        or ("drivetrain-study", "run", _command_for_type(study_type), str(resolution.paths.root)),
        project=resolution.paths.root,
        bundle_path=workspace.path / "track_bundle.json",
        study_name=study,
        study_fingerprint=fingerprint,
        resolved_configuration_fingerprint=canonical_fingerprint(resolution.data),
    )
    write_provenance(workspace.path, provenance)
    committed = workspace.commit()
    write_results_index(resolution.paths.results_directory)
    return committed


def _execute(
    *,
    study_name: str,
    study_type: str,
    study_raw: Mapping[str, Any],
    base_study: Mapping[str, Any],
    vehicle_id: str,
    vehicle_raw: Mapping[str, Any],
    track_raw: Mapping[str, Any],
    bundle: TrackBundle,
    registry: Any,
    replicates_override: int | None,
    workspace: ResultWorkspace,
    cache: SimulationCache,
    workers: int,
    progress: bool,
    study_fingerprint: str,
    evidence_assessment: Mapping[str, Any],
) -> StudyExecution:
    design_points, sampling_mode, replicates = study_plan(
        study_type, study_raw, registry, replicates_override
    )
    design_path = design_points[0].path if study_type == "design_sweep" else None
    sampling = study_raw.get("sampling", {})
    seed = int(study_raw.get("study", {}).get("random_seed", 20260715))
    plan = SamplingPlan(
        mode=sampling_mode,
        replicates=replicates,
        random_seed=seed,
        selected_paths=tuple(str(path) for path in sampling.get("paths", ())),
        excluded_paths=(design_path,) if design_path else (),
        correlation_groups=correlation_groups_from_study(study_raw),
        gate_sampling=str(sampling.get("gate_sampling", "paired_lap")),
    )
    sampler = ScenarioSampler(registry=registry, bundle=bundle, plan=plan)
    scenarios = (
        sampler.draw_all()
        if study_type != "structural_sensitivity"
        else (ScenarioDraw(0, seed, "nominal"),)
    )
    reporter = ProgressReporter(total=len(scenarios), label="scenarios", enabled=progress)

    def execute_or_resume(scenario: ScenarioDraw) -> dict[str, Any]:
        checkpoint = workspace.load_checkpoint(scenario.replicate)
        if checkpoint is not None:
            result = dict(checkpoint["result"])
            result["resumed"] = True
            return result
        result = _execute_scenario(
            scenario=scenario,
            design_points=design_points,
            study_type=study_type,
            vehicle_id=vehicle_id,
            vehicle_raw=vehicle_raw,
            base_study=base_study,
            track_raw=track_raw,
            bundle=bundle,
            cache=cache,
        )
        workspace.write_checkpoint(scenario.replicate, {"result": result})
        return result

    scenario_results: list[dict[str, Any]] = []
    if workers == 1 or len(scenarios) == 1:
        for scenario in scenarios:
            scenario_results.append(execute_or_resume(scenario))
            reporter.advance(f"replicate {scenario.replicate}")
    else:
        with ThreadPoolExecutor(max_workers=min(workers, len(scenarios))) as executor:
            futures = {
                executor.submit(execute_or_resume, scenario): scenario
                for scenario in scenarios
            }
            for future in as_completed(futures):
                scenario = futures[future]
                scenario_results.append(future.result())
                reporter.advance(f"replicate {scenario.replicate}")

    order = {point.identifier: index for index, point in enumerate(design_points)}
    rows = [row for result in scenario_results for row in result["rows"]]
    rows.sort(key=lambda row: (int(row["replicate"]), order[str(row["design_id"])]))
    quality = quality_summary(rows, study_raw)
    summary = {
        **summarize_study(study_type, rows, study_raw, seed),
        "numerical_quality": quality,
    }
    convergence = convergence_summary(study_type, rows)
    fallback_gate_ids = sorted(
        {
            gate_id
            for scenario in scenarios
            for gate_id in scenario.independently_sampled_gate_ids
        }
    )
    sampled_gate_ids = sorted(
        {
            gate_id
            for scenario in scenarios
            for gate_id in scenario.gate_target_speeds_mps
        }
    )
    stochastic_by_role: dict[str, list[str]] = {}
    for registered in registry.stochastic():
        stochastic_by_role.setdefault(registered.category, []).append(registered.path)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "framework_contract": "measured-track-drivetrain-framework-v0.8",
        "study_fingerprint_sha256": study_fingerprint,
        "study_name": study_name,
        "study_type": study_type,
        "vehicle_id": vehicle_id,
        "sampling_mode": sampling_mode,
        "random_seed": seed,
        "scenario_count": len(scenarios),
        "resumed_scenario_count": sum(bool(item.get("resumed")) for item in scenario_results),
        "design_point_count": len(design_points),
        "bounded_case_count": sum(int(item["bounded_case_count"]) for item in scenario_results),
        "reference_case_count": sum(int(item["reference_case_count"]) for item in scenario_results),
        "bounded_simulation_count": sum(int(item["bounded_simulation_count"]) for item in scenario_results),
        "reference_simulation_count": sum(int(item["reference_simulation_count"]) for item in scenario_results),
        "reference_cache_hits": sum(int(item["reference_cache_hits"]) for item in scenario_results),
        "simulation_cache_hits": sum(int(item["simulation_cache_hits"]) for item in scenario_results),
        "simulation_cache_enabled": cache.enabled,
        "simulation_cache_status": cache.status(),
        "parallel_workers": workers,
        "reference_cache_policy": (
            "one scenario-level infinite reference shared by every design candidate"
        ),
        "paired_scenarios": True,
        "sampled_input_paths": list(sampler.sampled_paths),
        "sampled_input_count": len(sampler.sampled_paths),
        "declared_stochastic_input_paths_by_role": {
            role: sorted(paths) for role, paths in sorted(stochastic_by_role.items())
        },
        "gate_sampling_policy": plan.gate_sampling,
        "sampled_gate_ids": sampled_gate_ids,
        "sampled_gate_count": len(sampled_gate_ids),
        "paired_gate_identity_count": sampler.paired_gate_identity_count,
        "independent_gate_fallback_ids": fallback_gate_ids,
        "track_bundle_content_fingerprint": bundle.data.get("content_fingerprint_sha256"),
        "track_bundle_sha256": bundle.sha256,
        "bootstrap_resamples": int(
            study_raw.get("reporting", {}).get("bootstrap_resamples", 1000)
        ),
        "numerical_quality": quality,
        "evidence_assessment": dict(evidence_assessment),
        "uncertainty_not_propagated": [
            "physical feature geometry uncertainty (resolved bundle geometry is used)",
            "telemetry elevation uncertainty (grade force remains disabled pending the materiality screen)",
        ],
    }
    return StudyExecution(
        rows=tuple(rows),
        scenario_draws=tuple(scenario.serializable() for scenario in scenarios),
        summary=summary,
        convergence=convergence,
        manifest=manifest,
        input_contracts=input_contracts(registry),
    )


def _execute_scenario(
    *,
    scenario: ScenarioDraw,
    design_points: tuple[DesignPoint, ...],
    study_type: str,
    vehicle_id: str,
    vehicle_raw: Mapping[str, Any],
    base_study: Mapping[str, Any],
    track_raw: Mapping[str, Any],
    bundle: TrackBundle,
    cache: SimulationCache,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    references: dict[tuple[int, str], tuple[dict[str, Any], str]] = {}
    bounded_runs = reference_runs = reference_reuses = persistent_hits = 0
    for design in design_points:
        design_values = (
            {design.path: float(design.value_si)}
            if design.path is not None and design.value_si is not None
            else {}
        )
        choices = dict(scenario.choice_values)
        if design.path is not None and design.choice_value is not None:
            choices[design.path] = design.choice_value
        try:
            bounded_case, reference_case, settings, runtime_track = resolve_simulation_cases(
                vehicle_id=vehicle_id,
                vehicle_raw=vehicle_raw,
                study_raw=base_study,
                track_raw=track_raw,
                bundle=bundle,
                quantity_values_si=scenario.quantity_values_si,
                choice_values=choices,
                gate_target_speeds_mps=scenario.gate_target_speeds_mps,
                design_values_si=design_values,
                shared_reference=study_type == "design_sweep",
            )
        except Exception as exc:
            raise SimulationError(
                f"Scenario {scenario.replicate}, design {design.identifier!r} could not form a valid physical case: {exc}"
            ) from exc
        bounded_record, cached = _run_case_summary_cached(
            bounded_case, settings, runtime_track, cache
        )
        bounded_runs += int(not cached)
        persistent_hits += int(cached)
        key = reference_cache_key(
            scenario.replicate,
            design,
            share_across_designs=study_type == "design_sweep",
        )
        if key in references:
            reference_record, reference_fingerprint = references[key]
            reference_reuses += 1
        else:
            reference_record, cached = _run_case_summary_cached(
                reference_case, settings, runtime_track, cache
            )
            reference_runs += int(not cached)
            persistent_hits += int(cached)
            reference_fingerprint = phase6_service._reference_fingerprint(
                scenario, design, reference_case, runtime_track
            )
            references[key] = (reference_record, reference_fingerprint)
        bounded_summary = bounded_record["summary"]
        reference_summary = reference_record["summary"]
        comparison = compare_summaries(bounded_summary, reference_summary)
        row: dict[str, Any] = {
            "replicate": scenario.replicate,
            "scenario_seed": scenario.seed,
            "design_id": design.identifier,
            "design_path": design.path or "nominal",
            "design_value": design.display_value,
            "design_value_si": design.value_si,
            "design_choice_value": design.choice_value,
            "level_probability": design.level_probability,
            "level_kind": design.level_kind,
            "parameter_path": design.path if study_type == "structural_sensitivity" else None,
            "reference_fingerprint": reference_fingerprint,
            "bounded_completed": bool(bounded_summary["completed"]),
            "reference_completed": bool(reference_summary["completed"]),
            "reference_dominance_pass": bool(comparison["reference_dominance_pass"]),
            "bounded_energy_balance_relative_error": float(comparison["bounded_energy_balance_relative_error"]),
            "reference_energy_balance_relative_error": float(comparison["reference_energy_balance_relative_error"]),
            "bounded_powertrain_energy_balance_relative_error": float(
                bounded_summary["powertrain_energy_balance_relative_error"]
            ),
            "reference_powertrain_energy_balance_relative_error": float(
                reference_summary["powertrain_energy_balance_relative_error"]
            ),
            "bounded_max_gate_excess_kmh": bounded_record["maximum_gate_excess_kmh"],
            "reference_max_gate_excess_kmh": reference_record["maximum_gate_excess_kmh"],
            "bounded_gates_compliant_0p5_kmh": bounded_record["gates_compliant_0p5_kmh"],
            "reference_gates_compliant_0p5_kmh": reference_record["gates_compliant_0p5_kmh"],
        }
        row.update({metric: float(comparison[metric]) for metric in METRICS})
        _add_summary_fields(row, "bounded", bounded_summary)
        _add_summary_fields(row, "reference", reference_summary)
        rows.append(row)
    return {
        "rows": rows,
        "bounded_case_count": len(design_points),
        "reference_case_count": (
            1 if study_type == "design_sweep" and design_points else len(design_points)
        ),
        "bounded_simulation_count": bounded_runs,
        "reference_simulation_count": reference_runs,
        "reference_cache_hits": reference_reuses,
        "simulation_cache_hits": persistent_hits,
        "resumed": False,
    }


_SUMMARY_FIELDS = (
    "engine_energy_kj",
    "transmitted_energy_kj",
    "drivetrain_loss_energy_kj",
    "clutch_loss_energy_kj",
    "engine_operating_shortfall_energy_kj",
    "finite_ratio_opportunity_loss_energy_kj",
    "tire_slip_loss_energy_kj",
    "brake_loss_energy_kj",
    "rolling_loss_energy_kj",
    "aerodynamic_loss_energy_kj",
    "obstacle_loss_energy_kj",
    "net_grade_work_kj",
    "initial_total_kinetic_energy_kj",
    "final_total_kinetic_energy_kj",
    "energy_balance_residual_kj",
    "powertrain_energy_balance_residual_kj",
    "time_maximum_ratio_s",
    "time_variable_ratio_s",
    "time_minimum_ratio_s",
)


def _add_summary_fields(row: dict[str, Any], prefix: str, summary: Mapping[str, Any]) -> None:
    for field in _SUMMARY_FIELDS:
        row[f"{prefix}_{field}"] = float(summary[field])
    row[f"{prefix}_obstacle_energy_by_feature_kj_json"] = json.dumps(
        summary.get("obstacle_energy_by_feature_kj", {}),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _run_case_summary_cached(
    case: Any, settings: Any, track: Any, cache: SimulationCache
) -> tuple[dict[str, Any], bool]:
    key = SimulationCache.key(
        {
            "schema": "simulation-summary-v1",
            "case": asdict(case),
            "settings": asdict(settings),
            "track": _track_cache_contract(track),
        }
    )
    cached = cache.get(key)
    if cached is not None:
        return cached, True
    trace = run_simulation(case=case, track=track, settings=settings)
    summary = summarize_trace(
        trace,
        target_engine_rpm=case.engine.target_rpm,
        target_power_w=case.engine.target_power_w,
    )
    gate_rows = gate_compliance_rows(trace, track)
    record = {
        "summary": summary,
        "maximum_gate_excess_kmh": max(
            (float(item["excess_over_ceiling_kmh"]) for item in gate_rows), default=0.0
        ),
        "gates_compliant_0p5_kmh": all(
            bool(item["compliant_within_0p5_kmh"]) for item in gate_rows
        ),
    }
    cache.put(key, record)
    return record, False


def _track_cache_contract(track: Any) -> dict[str, Any]:
    return {
        "name": track.name,
        "length_m": track.length_m,
        "closed_course": track.closed_course,
        "centreline_fingerprint": canonical_fingerprint(
            {
                "s": track.centreline_s_m,
                "x": track.centreline_x_m,
                "y": track.centreline_y_m,
                "curvature": track.centreline_curvature_1_per_m,
            }
        ),
        "surface_friction_coefficient": track.surface_friction_coefficient,
        "features": [
            {
                "id": feature.identifier,
                "interval": asdict(feature.interval),
                "model": repr(feature.model),
            }
            for feature in track.features
        ],
        "gates": [asdict(gate) for gate in track.speed_gates],
        "gpx_grade_force_enabled": track.gpx_grade_force_enabled,
    }


def _safe_name(value: str) -> str:
    name = "-".join(
        part
        for part in "".join(
            char.lower() if char.isalnum() else "-" for char in value
        ).split("-")
        if part
    )
    return name or "study"


def _command_for_type(study_type: str) -> str:
    return {
        "design_sweep": "sweep",
        "track_robustness": "track-robustness",
        "structural_sensitivity": "structural-sensitivity",
        "full_uncertainty": "uncertainty",
    }[study_type]
