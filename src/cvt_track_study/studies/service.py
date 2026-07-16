"""Phase 6 study orchestration and paired uncertainty propagation."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Mapping
from uuid import uuid4


from cvt_track_study.bundle import TrackBundle, load_track_bundle
from cvt_track_study.config import ProjectLoader
from cvt_track_study.simulation.integrator import run_simulation
from cvt_track_study.simulation.metrics import compare_summaries, gate_compliance_rows, summarize_trace
from cvt_track_study.simulation.service import SimulationError, resolve_simulation_cases
from cvt_track_study.track import build_project_track
from cvt_track_study.uncertainty import (
    SamplingPlan,
    ScenarioDraw,
    ScenarioSampler,
    build_input_registry,
    correlation_groups_from_study,
)

from .analysis import (
    METRICS,
    convergence_summary,
    input_contracts,
    quality_summary,
    summarize_study,
)
from .model import DesignPoint, StudyExecution
from .planning import reference_cache_key, study_plan
from .reporting import write_study_outputs

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
) -> Path:
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
    if not isinstance(base_study, Mapping) or str(base_study.get("study", {}).get("type")) != "baseline":
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
    bundle = load_track_bundle(bundle_path)

    registry = build_input_registry(
        vehicle_raw=vehicle_raw,
        base_study_raw=base_study,
        track_raw=resolution.data["track"],
        bundle=bundle,
    )
    output = output_directory or (
        resolution.paths.results_directory / study_type / _timestamp()
    )
    output = output.resolve()
    if output.exists():
        raise SimulationError(f"Output directory already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(f".{output.name}.tmp-{uuid4().hex}")
    staging.mkdir(parents=False, exist_ok=False)
    try:
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
        )
        resolution.export(staging / "resolved_inputs")
        _write_bundle_snapshot(staging, bundle, bundle_path)
        write_study_outputs(
            output=staging,
            rows=execution.rows,
            scenario_draws=execution.scenario_draws,
            summary=execution.summary,
            convergence=execution.convergence,
            manifest=execution.manifest,
            input_contracts=execution.input_contracts,
            study_type=study_type,
        )
        staging.replace(output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output


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
    scenarios = sampler.draw_all() if study_type != "structural_sensitivity" else (
        ScenarioDraw(0, seed, "nominal"),
    )

    rows: list[dict[str, Any]] = []
    reference_cache: dict[
        tuple[int, str], tuple[dict[str, Any], str, float, bool]
    ] = {}
    bounded_count = 0
    reference_count = 0
    cache_hits = 0
    for scenario in scenarios:
        for design in design_points:
            design_values = (
                {design.path: float(design.value_si)}
                if design.path is not None and design.value_si is not None
                else {}
            )
            resolved_choices = dict(scenario.choice_values)
            if design.path is not None and design.choice_value is not None:
                resolved_choices[design.path] = design.choice_value
            try:
                bounded_case, reference_case, settings, runtime_track = resolve_simulation_cases(
                    vehicle_id=vehicle_id,
                    vehicle_raw=vehicle_raw,
                    study_raw=base_study,
                    track_raw=track_raw,
                    bundle=bundle,
                    quantity_values_si=scenario.quantity_values_si,
                    choice_values=resolved_choices,
                    gate_target_speeds_mps=scenario.gate_target_speeds_mps,
                    design_values_si=design_values,
                    shared_reference=study_type == "design_sweep",
                )
            except Exception as exc:
                raise SimulationError(
                    f"Scenario {scenario.replicate}, design {design.identifier!r} could not form a valid physical case: {exc}"
                ) from exc
            bounded = run_simulation(case=bounded_case, track=runtime_track, settings=settings)
            bounded_count += 1
            bounded_summary = summarize_trace(
                bounded,
                target_engine_rpm=bounded_case.engine.target_rpm,
                target_power_w=bounded_case.engine.target_power_w,
            )
            cache_key = reference_cache_key(
                scenario.replicate,
                design,
                share_across_designs=study_type == "design_sweep",
            )
            if cache_key in reference_cache:
                (
                    reference_summary,
                    reference_fingerprint,
                    reference_max_gate_excess_kmh,
                    reference_gates_compliant,
                ) = reference_cache[cache_key]
                cache_hits += 1
            else:
                reference = run_simulation(case=reference_case, track=runtime_track, settings=settings)
                reference_count += 1
                reference_summary = summarize_trace(
                    reference,
                    target_engine_rpm=reference_case.engine.target_rpm,
                    target_power_w=reference_case.engine.target_power_w,
                )
                reference_fingerprint = _reference_fingerprint(
                    scenario, design, reference_case, runtime_track
                )
                reference_gate_rows = gate_compliance_rows(reference, runtime_track)
                reference_max_gate_excess_kmh = max(
                    (float(item["excess_over_ceiling_kmh"]) for item in reference_gate_rows),
                    default=0.0,
                )
                reference_gates_compliant = all(
                    bool(item["compliant_within_0p5_kmh"]) for item in reference_gate_rows
                )
                reference_cache[cache_key] = (
                    reference_summary,
                    reference_fingerprint,
                    reference_max_gate_excess_kmh,
                    reference_gates_compliant,
                )
            comparison = compare_summaries(bounded_summary, reference_summary)
            bounded_gate_rows = gate_compliance_rows(bounded, runtime_track)
            bounded_max_gate_excess_kmh = max(
                (float(item["excess_over_ceiling_kmh"]) for item in bounded_gate_rows),
                default=0.0,
            )
            bounded_gates_compliant = all(
                bool(item["compliant_within_0p5_kmh"]) for item in bounded_gate_rows
            )
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
                "bounded_max_gate_excess_kmh": bounded_max_gate_excess_kmh,
                "reference_max_gate_excess_kmh": reference_max_gate_excess_kmh,
                "bounded_gates_compliant_0p5_kmh": bounded_gates_compliant,
                "reference_gates_compliant_0p5_kmh": reference_gates_compliant,
            }
            row.update({metric: float(comparison[metric]) for metric in METRICS})
            rows.append(row)

    quality = quality_summary(rows, study_raw)
    summary = summarize_study(study_type, rows, study_raw, seed)
    summary = {**summary, "numerical_quality": quality}
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
    stochastic_by_role = {
        role: sorted(paths) for role, paths in sorted(stochastic_by_role.items())
    }
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "study_name": study_name,
        "study_type": study_type,
        "vehicle_id": vehicle_id,
        "sampling_mode": sampling_mode,
        "random_seed": seed,
        "scenario_count": len(scenarios),
        "design_point_count": len(design_points),
        "bounded_simulation_count": bounded_count,
        "reference_simulation_count": reference_count,
        "reference_cache_hits": cache_hits,
        "reference_cache_policy": (
            "one scenario-level infinite reference shared by every design candidate"
        ),
        "paired_scenarios": True,
        "sampled_input_paths": list(sampler.sampled_paths),
        "sampled_input_count": len(sampler.sampled_paths),
        "declared_stochastic_input_paths_by_role": stochastic_by_role,
        "gate_sampling_policy": plan.gate_sampling,
        "sampled_gate_ids": sampled_gate_ids,
        "sampled_gate_count": len(sampled_gate_ids),
        "paired_gate_identity_count": sampler.paired_gate_identity_count,
        "independent_gate_fallback_ids": fallback_gate_ids,
        "track_bundle_content_fingerprint": bundle.data.get("content_fingerprint_sha256"),
        "track_bundle_sha256": bundle.sha256,
        "bootstrap_resamples": int(study_raw.get("reporting", {}).get("bootstrap_resamples", 1000)),
        "numerical_quality": quality,
        "uncertainty_not_propagated": [
            "physical feature geometry uncertainty (the simulation uses the resolved bundle geometry)",
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


def _reference_fingerprint(
    scenario: ScenarioDraw,
    design: DesignPoint,
    reference_case: Any,
    runtime_track: Any,
) -> str:
    payload = {
        "scenario": scenario.serializable(),
        "reference_contract": "scenario_level_common_infinite_v1",
        "engine": asdict(reference_case.engine),
        "vehicle": asdict(reference_case.vehicle),
        "tire": asdict(reference_case.tire),
        "cvt": asdict(reference_case.cvt),
        "driver": asdict(reference_case.driver),
        "gates": [(gate.identifier, gate.target_speed_mps) for gate in runtime_track.speed_gates],
        "features": [
            (feature.identifier, feature.model.model_type, repr(feature.model))
            for feature in runtime_track.features
        ],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _write_bundle_snapshot(output: Path, bundle: TrackBundle, bundle_path: Path) -> None:
    source = bundle.path or bundle_path.resolve()
    target = output / "track_bundle.json"
    target.write_bytes(source.read_bytes())
    target.with_name("track_bundle.sha256").write_text(
        f"{bundle.sha256}  {target.name}\n", encoding="utf-8"
    )


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
