"""Uncertainty-informed paired design screening from saved source worlds."""

from __future__ import annotations

from collections.abc import Mapping
from concurrent.futures import as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from cvt_track_study.bundle import load_track_bundle
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
from cvt_track_study.runtime.process_pool import interruptible_process_pool
from cvt_track_study.simulation.service import SimulationError
from cvt_track_study.uncertainty import (
    GateSampleIdentity,
    ScenarioDraw,
    build_input_registry,
)

from . import service as phase6_service
from . import service_v8
from .analysis import (
    convergence_summary,
    input_contracts,
    quality_summary,
    summarize_study,
)
from .ensemble_v10 import TrackVariant, _write_track_ensemble_snapshot
from .design_grid import (
    design_paths as configured_design_paths,
    design_sweep_plan,
)
from .design_grid_execution import execute_design_grid_scenario
from .reporting_v8 import write_study_outputs
from .scenario_reduction import (
    ReductionSettings,
    ScenarioReductionError,
    load_uncertainty_source,
    reduce_uncertainty_source,
    resolve_source_result,
    write_reduction_artifacts,
)


_DESIGN_PROCESS_CONTEXT: dict[str, Any] | None = None


def _initialize_design_process(
    design_points: tuple[Any, ...],
    vehicle_id: str,
    vehicle_raw: Mapping[str, Any],
    base_study: Mapping[str, Any],
    track_raw: Mapping[str, Any],
    bundles: Mapping[str, Any],
    cache_root: Path,
    cache_enabled: bool,
) -> None:
    """Initialize immutable design-grid context once per worker process."""

    global _DESIGN_PROCESS_CONTEXT
    _DESIGN_PROCESS_CONTEXT = {
        "design_points": design_points,
        "vehicle_id": vehicle_id,
        "vehicle_raw": vehicle_raw,
        "base_study": base_study,
        "track_raw": track_raw,
        "bundles": dict(bundles),
        "cache": SimulationCache(Path(cache_root), enabled=cache_enabled),
    }


def _execute_design_process_initialized(
    scenario: ScenarioDraw, bundle_key: str
) -> dict[str, Any]:
    """Execute one complete design grid in one paired uncertainty world."""

    context = _DESIGN_PROCESS_CONTEXT
    if context is None:
        raise RuntimeError("Design process context was not initialized.")
    cache: SimulationCache = context["cache"]
    before = (cache.hits, cache.misses, cache.writes)
    result = execute_design_grid_scenario(
        scenario=scenario,
        design_points=context["design_points"],
        study_type="design_sweep",
        vehicle_id=context["vehicle_id"],
        vehicle_raw=context["vehicle_raw"],
        base_study=context["base_study"],
        track_raw=context["track_raw"],
        bundle=context["bundles"][bundle_key],
        cache=cache,
    )
    result["_process_cache_counts"] = {
        "hits": cache.hits - before[0],
        "misses": cache.misses - before[1],
        "writes": cache.writes - before[2],
    }
    return result


def run_uncertainty_informed_design_project(
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
    """Replay representative worlds from a completed uncertainty result.

    ``--replicates N`` overrides the number of selected source base draws.  It does
    not generate new Monte-Carlo draws in this mode.
    """

    if bundle_path is not None:
        raise SimulationError(
            "An uncertainty-informed design study uses the track bundles saved in "
            "its source uncertainty result and cannot also use --bundle."
        )
    if workers < 1:
        raise SimulationError("workers must be at least one.")

    resolution = ProjectLoader().resolve(project, study=study)
    if resolution.error_count:
        details = "\n".join(item.format() for item in resolution.diagnostics)
        raise SimulationError(f"Project validation failed:\n{details}")
    raw = resolution.data.get("studies", {}).get(study)
    if not isinstance(raw, Mapping):
        raise SimulationError(f"Study {study!r} was not found.")
    if str(raw.get("study", {}).get("type", "")) != "design_sweep":
        raise SimulationError(
            "Uncertainty-informed replay is available only for design_sweep studies."
        )

    base_name = str(raw.get("base_case", {}).get("study", "baseline"))
    base_study = resolution.data.get("studies", {}).get(base_name)
    if not isinstance(base_study, Mapping) or str(
        base_study.get("study", {}).get("type", "")
    ) != "baseline":
        raise SimulationError(
            f"Study {study!r} requires base_case.study to reference a baseline study."
        )
    vehicle_id = str(raw.get("study", {}).get("vehicle_id", ""))
    vehicle_raw = resolution.data.get("vehicles", {}).get(vehicle_id)
    if not isinstance(vehicle_raw, Mapping):
        raise SimulationError(f"Vehicle {vehicle_id!r} is not resolved.")

    config = raw.get("uncertainty_informed_design", {})
    try:
        settings = ReductionSettings.from_mapping(
            config if isinstance(config, Mapping) else {},
            base_draw_override=replicates_override,
        )
        source_path = resolve_source_result(
            project_root=resolution.paths.root,
            results_directory=resolution.paths.results_directory,
            source_value=settings.source_result,
        )
        source = load_uncertainty_source(source_path)
    except ScenarioReductionError as exc:
        raise SimulationError(str(exc)) from exc

    source_vehicle = str(source.manifest.get("vehicle_id", ""))
    if source_vehicle and source_vehicle != vehicle_id:
        raise SimulationError(
            f"Source uncertainty result belongs to vehicle {source_vehicle!r}, not {vehicle_id!r}."
        )

    source_case_by_id = {case.case_id: case for case in source.track_cases}
    nominal_case = source_case_by_id.get("nominal", source.track_cases[0])
    nominal_path = nominal_case.bundle_path
    nominal_bundle = load_track_bundle(nominal_path)
    registry = build_input_registry(
        vehicle_raw=vehicle_raw,
        base_study_raw=base_study,
        track_raw=resolution.data["track"],
        bundle=nominal_bundle,
    )
    design_points, _, _ = design_sweep_plan(raw, registry, None)
    if not design_points:
        raise SimulationError("The design sweep produced no design candidates.")
    design_paths = configured_design_paths(design_points)
    if not design_paths:
        raise SimulationError("The design sweep produced no design-variable paths.")

    try:
        reduction = reduce_uncertainty_source(
            source,
            settings=settings,
            design_paths=design_paths,
        )
    except ScenarioReductionError as exc:
        raise SimulationError(str(exc)) from exc

    variants = tuple(
        TrackVariant(
            case_id=case_id,
            category=source_case_by_id[case_id].category,
            label=source_case_by_id[case_id].label,
            bundle_path=source_case_by_id[case_id].bundle_path,
            bundle=load_track_bundle(source_case_by_id[case_id].bundle_path),
        )
        for case_id in reduction.selected_track_case_ids
    )
    variant_by_id = {variant.case_id: variant for variant in variants}

    scheduled: list[tuple[TrackVariant, ScenarioDraw, int, int]] = []
    for new_replicate, record in enumerate(reduction.selected_scenarios):
        case_id = str(record["track_case_id"])
        scenario = _scenario_from_source(
            record,
            new_replicate=new_replicate,
            design_paths=design_paths,
        )
        scheduled.append(
            (
                variant_by_id[case_id],
                scenario,
                int(record["base_draw_id"]),
                int(record["replicate"]),
            )
        )

    seed = int(raw.get("study", {}).get("random_seed", 20260715))
    fingerprint = canonical_fingerprint(
        {
            "schema": "framework-uncertainty-informed-design-v1",
            "study_name": study,
            "study": raw,
            "base_study": base_study,
            "vehicle": vehicle_raw,
            "track": resolution.data["track"],
            "source_study_fingerprint": source.manifest.get(
                "study_fingerprint_sha256"
            ),
            "selected_base_draw_ids": reduction.selected_base_draw_ids,
            "selected_track_case_ids": reduction.selected_track_case_ids,
            "design_paths": design_paths,
            "replicates_override": replicates_override,
        }
    )
    default_output = (
        resolution.paths.results_directory
        / "design_sweep"
        / f"{service_v8._safe_name(run_name or study)}--{fingerprint[:10]}"
    )
    output = output_directory.resolve() if output_directory else default_output.resolve()
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
    reporter = ProgressReporter(
        total=len(scheduled), label="source worlds", enabled=progress
    )
    reporter.begin(
        f"{len(scheduled)} uncertainty-informed worlds from "
        f"{len(reduction.selected_base_draw_ids)} source draw(s) and "
        f"{len(variants)} track case(s); {len(design_points)} design point(s), "
        f"{workers} worker(s)"
    )

    source_metadata_by_replicate: dict[int, dict[str, Any]] = {}

    def register_source_metadata(
        item: tuple[TrackVariant, ScenarioDraw, int, int]
    ) -> None:
        variant, scenario, base_draw_id, source_replicate = item
        source_metadata_by_replicate[scenario.replicate] = {
            "base_draw_id": base_draw_id,
            "source_replicate": source_replicate,
            "track_case_id": variant.case_id,
        }

    def annotate_result(
        item: tuple[TrackVariant, ScenarioDraw, int, int],
        result: dict[str, Any],
    ) -> None:
        variant, _scenario, base_draw_id, source_replicate = item
        for row in result["rows"]:
            row["base_draw_id"] = base_draw_id
            row["track_pair_id"] = f"source-draw-{base_draw_id:06d}"
            row["track_case_id"] = variant.case_id
            row["track_case_category"] = variant.category
            row["track_bundle_fingerprint"] = variant.bundle.data.get(
                "content_fingerprint_sha256"
            )
            row["source_uncertainty_replicate"] = source_replicate
            row["source_uncertainty_result"] = str(source.result_directory)
        result["track_case_id"] = variant.case_id
        result["base_draw_id"] = base_draw_id
        result["source_uncertainty_replicate"] = source_replicate

    def execute_or_resume(
        item: tuple[TrackVariant, ScenarioDraw, int, int]
    ) -> dict[str, Any]:
        variant, scenario, _base_draw_id, _source_replicate = item
        register_source_metadata(item)
        checkpoint = workspace.load_checkpoint(scenario.replicate)
        if checkpoint is not None:
            result = dict(checkpoint["result"])
            result["resumed"] = True
            return result
        result = execute_design_grid_scenario(
            scenario=scenario,
            design_points=design_points,
            study_type="design_sweep",
            vehicle_id=vehicle_id,
            vehicle_raw=vehicle_raw,
            base_study=base_study,
            track_raw=resolution.data["track"],
            bundle=variant.bundle,
            cache=cache,
        )
        annotate_result(item, result)
        workspace.write_checkpoint(scenario.replicate, {"result": result})
        return result

    results: list[dict[str, Any]] = []
    parallel_backend = "serial"
    if workers == 1 or len(scheduled) == 1:
        for item in scheduled:
            result = execute_or_resume(item)
            results.append(result)
            reporter.advance(
                f"source replicate {item[3]}; draw={item[2]}; track={item[0].case_id}"
            )
    else:
        pending: list[tuple[TrackVariant, ScenarioDraw, int, int]] = []
        for item in scheduled:
            register_source_metadata(item)
            scenario = item[1]
            checkpoint = workspace.load_checkpoint(scenario.replicate)
            if checkpoint is None:
                pending.append(item)
                continue
            result = dict(checkpoint["result"])
            result["resumed"] = True
            results.append(result)
            reporter.advance(
                f"source replicate {item[3]}; draw={item[2]}; "
                f"track={item[0].case_id} (resumed)"
            )

        if pending:
            parallel_backend = "process"
            bundles_by_case = {item[0].case_id: item[0].bundle for item in scheduled}
            with interruptible_process_pool(
                max_workers=min(workers, len(pending)),
                initializer=_initialize_design_process,
                initargs=(
                    tuple(design_points),
                    vehicle_id,
                    vehicle_raw,
                    base_study,
                    resolution.data["track"],
                    bundles_by_case,
                    cache.root,
                    cache.enabled,
                ),
            ) as executor:
                futures = {
                    executor.submit(
                        _execute_design_process_initialized,
                        item[1],
                        item[0].case_id,
                    ): item
                    for item in pending
                }
                for future in as_completed(futures):
                    item = futures[future]
                    result = future.result()
                    service_v8._merge_process_cache_counts(cache, result)
                    annotate_result(item, result)
                    workspace.write_checkpoint(
                        item[1].replicate, {"result": result}
                    )
                    results.append(result)
                    reporter.advance(
                        f"source replicate {item[3]}; draw={item[2]}; "
                        f"track={item[0].case_id}"
                    )

    order = {point.identifier: index for index, point in enumerate(design_points)}
    rows = [row for result in results for row in result["rows"]]
    rows.sort(key=lambda row: (int(row["replicate"]), order[str(row["design_id"])]))
    quality = quality_summary(rows, raw)
    summary = {
        **summarize_study("design_sweep", rows, raw, seed),
        "numerical_quality": quality,
    }
    convergence = convergence_summary("design_sweep", rows)

    scenario_payloads: list[dict[str, Any]] = []
    for variant, scenario, base_draw_id, source_replicate in scheduled:
        payload = dict(scenario.serializable())
        payload["base_draw_id"] = base_draw_id
        payload["track_pair_id"] = f"source-draw-{base_draw_id:06d}"
        payload["track_case_id"] = variant.case_id
        payload["track_case_category"] = variant.category
        payload["track_bundle_fingerprint"] = variant.bundle.data.get(
            "content_fingerprint_sha256"
        )
        payload["source_uncertainty_replicate"] = source_replicate
        payload["source_uncertainty_result"] = str(source.result_directory)
        scenario_payloads.append(payload)

    sampled_paths = sorted(
        {
            path
            for _, scenario, _, _ in scheduled
            for path in (
                *scenario.quantity_values_si.keys(),
                *scenario.choice_values.keys(),
            )
        }
    )
    sampled_gate_ids = sorted(
        {
            gate_id
            for _, scenario, _, _ in scheduled
            for gate_id in scenario.gate_target_speeds_mps
        }
    )
    fallback_gate_ids = sorted(
        {
            gate_id
            for _, scenario, _, _ in scheduled
            for gate_id in scenario.independently_sampled_gate_ids
        }
    )
    stochastic_by_role: dict[str, list[str]] = {}
    for registered in registry.stochastic():
        stochastic_by_role.setdefault(registered.category, []).append(registered.path)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "framework_contract": "measured-track-drivetrain-framework-v1.0-six-report",
        "study_fingerprint_sha256": fingerprint,
        "study_name": study,
        "study_type": "design_sweep",
        "vehicle_id": vehicle_id,
        "sampling_mode": "uncertainty_result_representative_replay",
        "random_seed": seed,
        "scenario_count": len(scheduled),
        "resumed_scenario_count": sum(bool(item.get("resumed")) for item in results),
        "design_point_count": len(design_points),
        "bounded_case_count": sum(int(item["bounded_case_count"]) for item in results),
        "reference_case_count": sum(int(item["reference_case_count"]) for item in results),
        "bounded_simulation_count": sum(
            int(item["bounded_simulation_count"]) for item in results
        ),
        "reference_simulation_count": sum(
            int(item["reference_simulation_count"]) for item in results
        ),
        "reference_cache_hits": sum(int(item["reference_cache_hits"]) for item in results),
        "simulation_cache_hits": sum(int(item["simulation_cache_hits"]) for item in results),
        "simulation_cache_enabled": cache.enabled,
        "simulation_cache_status": cache.status(),
        "parallel_workers": workers,
        "parallel_backend": parallel_backend,
        "paired_scenarios": True,
        "traffic_model": source.manifest.get("traffic_model", {"enabled": False}),
        "traffic_replay_policy": "exact_source_traffic_world_replay",
        "reference_cache_policy": (
            "one scenario-level infinite reference shared by every design candidate"
        ),
        "sampled_input_paths": sampled_paths,
        "sampled_input_count": len(sampled_paths),
        "declared_stochastic_input_paths_by_role": {
            role: sorted(paths) for role, paths in sorted(stochastic_by_role.items())
        },
        "gate_sampling_policy": "exact_values_replayed_from_source_uncertainty_result",
        "sampled_gate_ids": sampled_gate_ids,
        "sampled_gate_count": len(sampled_gate_ids),
        "independent_gate_fallback_ids": fallback_gate_ids,
        "track_bundle_content_fingerprint": nominal_bundle.data.get(
            "content_fingerprint_sha256"
        ),
        "track_bundle_sha256": nominal_bundle.sha256,
        "track_ensemble_enabled": len(variants) > 1,
        "track_ensemble_case_count": len(variants),
        "track_ensemble_case_ids": [variant.case_id for variant in variants],
        "track_ensemble_source": str(source.result_directory),
        "track_ensemble_policy": (
            "materially_distinct_cases_selected_from_completed_full_uncertainty_result"
        ),
        "track_case_assignment": "fully_crossed_selected_source_draws",
        "sampling_layout": "uncertainty_result_reduced_replay",
        "base_draw_count": len(reduction.selected_base_draw_ids),
        "scenarios_per_track_case": len(reduction.selected_base_draw_ids),
        "track_case_pairing_complete": True,
        "sampling_replicates_interpretation": (
            "selected source uncertainty base draws; no new uncertainty sampling"
        ),
        "bootstrap_resamples": int(raw.get("reporting", {}).get("bootstrap_resamples", 1000)),
        "numerical_quality": quality,
        "evidence_assessment": dict(
            assess_evidence(diagnostics=resolution.diagnostics, bundle=nominal_bundle)
        ),
        "design_variable_paths": list(design_paths),
        "uncertainty_source": dict(reduction.metadata),
        "uncertainty_not_propagated": list(
            source.manifest.get("uncertainty_not_propagated", [])
        ),
    }

    resolution.export(workspace.path / "resolved_inputs")
    phase6_service._write_bundle_snapshot(workspace.path, nominal_bundle, nominal_path)
    _write_track_ensemble_snapshot(workspace.path, variants, source.result_directory)
    write_study_outputs(
        output=workspace.path,
        rows=rows,
        scenario_draws=scenario_payloads,
        summary=summary,
        convergence=convergence,
        manifest=manifest,
        input_contracts=input_contracts(registry),
        study_type="design_sweep",
    )
    write_reduction_artifacts(workspace.path, reduction)
    write_provenance(
        workspace.path,
        build_provenance(
            command=command
            or (
                "drivetrain-study",
                "run",
                "design-comparison",
                str(resolution.paths.root),
            ),
            project=resolution.paths.root,
            bundle_path=workspace.path / "track_bundle.json",
            study_name=study,
            study_fingerprint=fingerprint,
            resolved_configuration_fingerprint=canonical_fingerprint(resolution.data),
        ),
    )
    reporter.finish(
        f"{len(rows)} design-scenario rows from {len(scheduled)} source worlds"
    )
    committed = workspace.commit()
    write_results_index(resolution.paths.results_directory)
    return committed


def _scenario_from_source(
    record: Mapping[str, Any],
    *,
    new_replicate: int,
    design_paths: tuple[str, ...],
) -> ScenarioDraw:
    identity_raw = record.get("gate_sample_identity")
    identity = None
    if isinstance(identity_raw, Mapping):
        try:
            identity = GateSampleIdentity(
                run_id=str(identity_raw["run_id"]),
                lap_id=int(identity_raw["lap_id"]),
                vehicle_id=str(identity_raw["vehicle_id"]),
                driver_id=str(identity_raw["driver_id"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise SimulationError(
                f"Malformed gate_sample_identity in source replicate {record.get('replicate')}."
            ) from exc

    removed = set(design_paths)
    quantities = {
        str(path): float(value)
        for path, value in dict(record.get("quantity_values_si", {})).items()
        if str(path) not in removed
    }
    choices = {
        str(path): str(value)
        for path, value in dict(record.get("choice_values", {})).items()
        if str(path) not in removed
    }
    gate_values = {
        str(path): float(value)
        for path, value in dict(record.get("gate_target_speeds_mps", {})).items()
    }
    traffic_raw = record.get("traffic_realization")
    traffic_payload = dict(traffic_raw) if isinstance(traffic_raw, Mapping) else None
    return ScenarioDraw(
        replicate=new_replicate,
        seed=int(record.get("seed", 0)),
        sampling_mode=str(record.get("sampling_mode", "all_declared")),
        quantity_values_si=quantities,
        choice_values=choices,
        gate_target_speeds_mps=gate_values,
        gate_sample_identity=identity,
        independently_sampled_gate_ids=tuple(
            str(value)
            for value in record.get("independently_sampled_gate_ids", ())
        ),
        sampling_design="replayed_from_completed_full_uncertainty_result",
        traffic_realization=traffic_payload,
    )
