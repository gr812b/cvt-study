"""All-declared structural runner with progress, ETA, and full reporting."""

from __future__ import annotations

from concurrent.futures import (
    ThreadPoolExecutor,
    as_completed,
)
from datetime import datetime, timezone
from pathlib import Path
from collections.abc import Mapping
from typing import Any
import json

from cvt_track_study.bundle import (
    load_track_bundle,
)
from cvt_track_study.config import (
    ProjectLoader,
)
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
from cvt_track_study.runtime.results import (
    write_results_index,
)
from cvt_track_study.simulation.metrics import (
    compare_summaries,
)
from cvt_track_study.simulation.service import (
    SimulationError,
    resolve_simulation_cases,
)
from cvt_track_study.track import (
    build_project_track,
)
from cvt_track_study.uncertainty import (
    ScenarioDraw,
    build_input_registry,
)

from . import service as phase6_service
from . import service_v8
from .analysis import (
    METRICS,
    convergence_summary,
    input_contracts,
    quality_summary,
)
from .planning import (
    selected_structural_paths,
    study_plan,
)
from .reporting_v8 import (
    write_study_outputs,
)
from .structural_analysis import (
    summarize_structural_screening,
)
from .structural_reporting import (
    write_structural_outputs,
)


_ABSOLUTE_SUMMARY_FIELDS = (
    "lap_time_s",
    "distance_m",
    "average_speed_kmh",
    "maximum_speed_kmh",
    "minimum_engine_rpm",
    "maximum_engine_rpm",
    "positive_demand_time_maximum_ratio_s",
    "positive_demand_time_variable_ratio_s",
    "positive_demand_time_minimum_ratio_s",
    "time_braking_s",
    "time_traction_limited_s",
    "maximum_abs_tire_slip_speed_mps",
    "target_engine_rpm",
    "target_power_w",
)


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
    """Route only structural sensitivity through the upgraded runner."""

    resolution = ProjectLoader().resolve(
        project, study=study
    )
    study_raw = resolution.data.get(
        "studies", {}
    ).get(study)
    study_type = (
        str(
            study_raw.get(
                "study", {}
            ).get("type", "")
        )
        if isinstance(
            study_raw, Mapping
        )
        else ""
    )
    if study_type != "structural_sensitivity":
        return service_v8.run_study_project(
            project,
            study=study,
            bundle_path=bundle_path,
            output_directory=output_directory,
            replicates_override=replicates_override,
            workers=workers,
            resume=resume,
            restart=restart,
            use_cache=use_cache,
            progress=progress,
            run_name=run_name,
            command=command,
        )
    return _run_structural_project(
        project,
        study=study,
        resolution=resolution,
        study_raw=study_raw,
        bundle_path=bundle_path,
        output_directory=output_directory,
        workers=workers,
        resume=resume,
        restart=restart,
        use_cache=use_cache,
        progress=progress,
        run_name=run_name,
        command=command,
    )


def _run_structural_project(
    project: str | Path,
    *,
    study: str,
    resolution: Any,
    study_raw: Mapping[str, Any],
    bundle_path: Path | None,
    output_directory: Path | None,
    workers: int,
    resume: bool,
    restart: bool,
    use_cache: bool,
    progress: bool,
    run_name: str | None,
    command: tuple[str, ...],
) -> Path:
    if workers < 1:
        raise SimulationError(
            "workers must be at least one."
        )
    if resolution.error_count:
        details = "\n".join(
            item.format()
            for item in resolution.diagnostics
        )
        raise SimulationError(
            f"Project validation failed:\n{details}"
        )

    base_name = str(
        study_raw.get(
            "base_case", {}
        ).get("study", "baseline")
    )
    base_study = resolution.data[
        "studies"
    ].get(base_name)
    if (
        not isinstance(
            base_study, Mapping
        )
        or str(
            base_study.get(
                "study", {}
            ).get("type")
        )
        != "baseline"
    ):
        raise SimulationError(
            f"Study {study!r} requires base_case.study "
            "to reference a baseline study."
        )

    vehicle_id = str(
        study_raw["study"]["vehicle_id"]
    )
    vehicle_raw = resolution.data[
        "vehicles"
    ].get(vehicle_id)
    if not isinstance(
        vehicle_raw, Mapping
    ):
        raise SimulationError(
            f"Vehicle {vehicle_id!r} is not resolved."
        )

    if bundle_path is None:
        build = build_project_track(project)
        if build.error_count:
            raise SimulationError(
                "Track build failed before study execution."
            )
        bundle_path = (
            build.output_directory
            / "track_bundle.json"
        )
    bundle_path = bundle_path.resolve()
    bundle = load_track_bundle(
        bundle_path
    )

    registry = build_input_registry(
        vehicle_raw=vehicle_raw,
        base_study_raw=base_study,
        track_raw=resolution.data[
            "track"
        ],
        bundle=bundle,
    )
    design_points, _, _ = study_plan(
        "structural_sensitivity",
        study_raw,
        registry,
        None,
    )
    selected_paths = (
        selected_structural_paths(
            study_raw, registry
        )
    )
    seed = int(
        study_raw.get(
            "study", {}
        ).get(
            "random_seed", 20260715
        )
    )

    fingerprint = canonical_fingerprint(
        {
            "schema": (
                "framework-structural-study-v0.9"
            ),
            "study_name": study,
            "study": study_raw,
            "base_study": base_study,
            "vehicle": vehicle_raw,
            "track": resolution.data[
                "track"
            ],
            "bundle_content_fingerprint": (
                bundle.data.get(
                    "content_fingerprint_sha256"
                )
            ),
            "selected_structural_paths": (
                selected_paths
            ),
        }
    )
    output = (
        output_directory.resolve()
        if output_directory is not None
        else (
            resolution.paths.results_directory
            / "structural_sensitivity"
            / (
                f"{service_v8._safe_name(run_name or study)}"
                f"--{fingerprint[:10]}"
            )
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
        raise SimulationError(
            str(exc)
        ) from exc

    cache = SimulationCache(
        resolution.paths.root
        / ".drivetrain-study-cache"
        / "simulations",
        enabled=use_cache,
    )
    scenario = ScenarioDraw(
        0, seed, "nominal"
    )
    reporter = ProgressReporter(
        total=len(design_points),
        label="parameter levels",
        enabled=progress,
    )
    reporter.begin(
        f"{len(selected_paths)} structural parameters, "
        f"up to {2 * len(design_points)} paired simulations, "
        f"{workers} worker(s)"
    )

    def execute_index(
        index: int,
    ) -> dict[str, Any]:
        checkpoint = (
            workspace.load_checkpoint(index)
        )
        if checkpoint is not None:
            result = dict(
                checkpoint["result"]
            )
            result["resumed"] = True
            return result

        result = _execute_design_point(
            index=index,
            design=design_points[index],
            scenario=scenario,
            vehicle_id=vehicle_id,
            vehicle_raw=vehicle_raw,
            base_study=base_study,
            track_raw=resolution.data[
                "track"
            ],
            bundle=bundle,
            cache=cache,
        )
        workspace.write_checkpoint(
            index, {"result": result}
        )
        return result

    results: list[dict[str, Any]] = []
    if (
        workers == 1
        or len(design_points) == 1
    ):
        for index in range(
            len(design_points)
        ):
            result = execute_index(index)
            results.append(result)
            reporter.advance(
                _progress_message(result)
            )
    else:
        with ThreadPoolExecutor(
            max_workers=min(
                workers,
                len(design_points),
            )
        ) as executor:
            futures = {
                executor.submit(
                    execute_index, index
                ): index
                for index in range(
                    len(design_points)
                )
            }
            for future in as_completed(
                futures
            ):
                result = future.result()
                results.append(result)
                reporter.advance(
                    _progress_message(result)
                )

    results.sort(
        key=lambda item: int(
            item["index"]
        )
    )
    rows = [
        result["row"]
        for result in results
    ]
    quality = quality_summary(
        rows, study_raw
    )
    summary = {
        **summarize_structural_screening(
            rows
        ),
        "numerical_quality": quality,
    }
    convergence = convergence_summary(
        "structural_sensitivity", rows
    )

    stochastic_by_role: dict[
        str, list[str]
    ] = {}
    for registered in registry.stochastic():
        stochastic_by_role.setdefault(
            registered.category, []
        ).append(registered.path)

    manifest = {
        "created_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "framework_contract": (
            "measured-track-drivetrain-framework-v0.9"
        ),
        "study_fingerprint_sha256": (
            fingerprint
        ),
        "study_name": study,
        "study_type": (
            "structural_sensitivity"
        ),
        "vehicle_id": vehicle_id,
        "sampling_mode": "nominal",
        "random_seed": seed,
        "scenario_count": 1,
        "design_point_count": len(
            design_points
        ),
        "structural_parameter_count": len(
            selected_paths
        ),
        "structural_parameter_paths": list(
            selected_paths
        ),
        "structural_selection_mode": (
            "all_declared_structural"
            if str(
                study_raw.get(
                    "sensitivity", {}
                ).get("selection", "")
            ).lower()
            in {
                "all_declared",
                "all_declared_structural",
                "all",
                "*",
            }
            or study_raw.get(
                "sensitivity", {}
            ).get("parameters")
            == ["*"]
            else "explicit"
        ),
        "structural_level_count": len(
            design_points
        ),
        "requested_simulation_count": (
            2 * len(design_points)
        ),
        "bounded_case_count": len(
            design_points
        ),
        "reference_case_count": len(
            design_points
        ),
        "bounded_simulation_count": sum(
            int(
                result[
                    "bounded_simulation_count"
                ]
            )
            for result in results
        ),
        "reference_simulation_count": sum(
            int(
                result[
                    "reference_simulation_count"
                ]
            )
            for result in results
        ),
        "reference_cache_hits": 0,
        "simulation_cache_hits": sum(
            int(
                result[
                    "simulation_cache_hits"
                ]
            )
            for result in results
        ),
        "resumed_scenario_count": sum(
            bool(
                result.get("resumed")
            )
            for result in results
        ),
        "simulation_cache_enabled": (
            cache.enabled
        ),
        "simulation_cache_status": (
            cache.status()
        ),
        "parallel_workers": workers,
        "progress_unit": (
            "parameter levels"
        ),
        "paired_scenarios": True,
        "reference_cache_policy": (
            "each structural level receives its physically matched infinite reference; "
            "identical cases may reuse the content-addressed simulation cache"
        ),
        "sampled_input_paths": [],
        "sampled_input_count": 0,
        "declared_stochastic_input_paths_by_role": {
            role: sorted(paths)
            for role, paths in sorted(
                stochastic_by_role.items()
            )
        },
        "track_bundle_content_fingerprint": (
            bundle.data.get(
                "content_fingerprint_sha256"
            )
        ),
        "track_bundle_sha256": (
            bundle.sha256
        ),
        "bootstrap_resamples": int(
            study_raw.get(
                "reporting", {}
            ).get(
                "bootstrap_resamples",
                1000,
            )
        ),
        "structural_report_top_parameter_count": int(
            study_raw.get(
                "reporting", {}
            ).get(
                "top_parameter_count", 15
            )
        ),
        "numerical_quality": quality,
        "evidence_assessment": dict(
            assess_evidence(
                diagnostics=resolution.diagnostics,
                bundle=bundle,
            )
        ),
        "uncertainty_not_propagated": [
            (
                "interactions between structural inputs "
                "(one-at-a-time screen)"
            ),
            (
                "physical feature geometry uncertainty "
                "(resolved bundle geometry is used)"
            ),
            (
                "telemetry elevation uncertainty "
                "(grade force remains disabled)"
            ),
        ],
    }

    resolution.export(
        workspace.path
        / "resolved_inputs"
    )
    phase6_service._write_bundle_snapshot(
        workspace.path,
        bundle,
        bundle_path,
    )
    contracts = input_contracts(
        registry
    )
    write_study_outputs(
        output=workspace.path,
        rows=tuple(rows),
        scenario_draws=(
            scenario.serializable(),
        ),
        summary=summary,
        convergence=convergence,
        manifest=manifest,
        input_contracts=contracts,
        study_type=(
            "structural_sensitivity"
        ),
    )
    write_structural_outputs(
        output=workspace.path,
        rows=tuple(rows),
        summary=summary,
        manifest=manifest,
        input_contracts=contracts,
    )

    provenance = build_provenance(
        command=command
        or (
            "drivetrain-study",
            "run",
            "structural-sensitivity",
            str(
                resolution.paths.root
            ),
        ),
        project=resolution.paths.root,
        bundle_path=(
            workspace.path
            / "track_bundle.json"
        ),
        study_name=study,
        study_fingerprint=fingerprint,
        resolved_configuration_fingerprint=(
            canonical_fingerprint(
                resolution.data
            )
        ),
    )
    write_provenance(
        workspace.path, provenance
    )
    reporter.finish(
        f"{len(selected_paths)} parameters screened"
    )
    committed = workspace.commit()
    write_results_index(
        resolution.paths.results_directory
    )
    return committed


def _execute_design_point(
    *,
    index: int,
    design: Any,
    scenario: ScenarioDraw,
    vehicle_id: str,
    vehicle_raw: Mapping[str, Any],
    base_study: Mapping[str, Any],
    track_raw: Mapping[str, Any],
    bundle: Any,
    cache: SimulationCache,
) -> dict[str, Any]:
    design_values = (
        {
            design.path: float(
                design.value_si
            )
        }
        if design.path is not None
        and design.value_si is not None
        else {}
    )
    choices = dict(
        scenario.choice_values
    )
    if (
        design.path is not None
        and design.choice_value
        is not None
    ):
        choices[design.path] = (
            design.choice_value
        )

    try:
        (
            bounded_case,
            reference_case,
            settings,
            runtime_track,
        ) = resolve_simulation_cases(
            vehicle_id=vehicle_id,
            vehicle_raw=vehicle_raw,
            study_raw=base_study,
            track_raw=track_raw,
            bundle=bundle,
            quantity_values_si=(
                scenario.quantity_values_si
            ),
            choice_values=choices,
            gate_target_speeds_mps=(
                scenario.gate_target_speeds_mps
            ),
            design_values_si=design_values,
            shared_reference=False,
        )
    except Exception as exc:
        raise SimulationError(
            f"Structural level {design.identifier!r} could not form a valid physical case: {exc}"
        ) from exc

    bounded_record, bounded_cached = (
        service_v8._run_case_summary_cached(
            bounded_case,
            settings,
            runtime_track,
            cache,
        )
    )
    reference_record, reference_cached = (
        service_v8._run_case_summary_cached(
            reference_case,
            settings,
            runtime_track,
            cache,
        )
    )
    bounded_summary = bounded_record[
        "summary"
    ]
    reference_summary = reference_record[
        "summary"
    ]
    comparison = compare_summaries(
        bounded_summary,
        reference_summary,
    )
    reference_fingerprint = (
        phase6_service._reference_fingerprint(
            scenario,
            design,
            reference_case,
            runtime_track,
        )
    )

    row: dict[str, Any] = {
        "replicate": 0,
        "scenario_seed": scenario.seed,
        "design_id": design.identifier,
        "design_path": design.path,
        "design_value": design.display_value,
        "design_value_si": design.value_si,
        "design_choice_value": (
            design.choice_value
        ),
        "level_probability": (
            design.level_probability
        ),
        "level_kind": design.level_kind,
        "parameter_path": design.path,
        "reference_fingerprint": (
            reference_fingerprint
        ),
        "bounded_completed": bool(
            bounded_summary["completed"]
        ),
        "reference_completed": bool(
            reference_summary["completed"]
        ),
        "bounded_termination_reason": str(
            bounded_summary.get(
                "termination_reason", ""
            )
        ),
        "reference_termination_reason": str(
            reference_summary.get(
                "termination_reason", ""
            )
        ),
        "reference_dominance_pass": bool(
            comparison[
                "reference_dominance_pass"
            ]
        ),
        "bounded_energy_balance_relative_error": float(
            comparison[
                "bounded_energy_balance_relative_error"
            ]
        ),
        "reference_energy_balance_relative_error": float(
            comparison[
                "reference_energy_balance_relative_error"
            ]
        ),
        "bounded_powertrain_energy_balance_relative_error": float(
            bounded_summary[
                "powertrain_energy_balance_relative_error"
            ]
        ),
        "reference_powertrain_energy_balance_relative_error": float(
            reference_summary[
                "powertrain_energy_balance_relative_error"
            ]
        ),
        "bounded_max_gate_excess_kmh": (
            bounded_record[
                "maximum_gate_excess_kmh"
            ]
        ),
        "reference_max_gate_excess_kmh": (
            reference_record[
                "maximum_gate_excess_kmh"
            ]
        ),
        "bounded_gates_compliant_0p5_kmh": (
            bounded_record[
                "gates_compliant_0p5_kmh"
            ]
        ),
        "reference_gates_compliant_0p5_kmh": (
            reference_record[
                "gates_compliant_0p5_kmh"
            ]
        ),
    }
    row.update(
        {
            metric: float(
                comparison[metric]
            )
            for metric in METRICS
        }
    )
    service_v8._add_summary_fields(
        row,
        "bounded",
        bounded_summary,
    )
    service_v8._add_summary_fields(
        row,
        "reference",
        reference_summary,
    )
    _add_absolute_fields(
        row,
        "bounded",
        bounded_summary,
    )
    _add_absolute_fields(
        row,
        "reference",
        reference_summary,
    )

    return {
        "index": index,
        "row": row,
        "identifier": design.identifier,
        "parameter_path": design.path,
        "level_kind": design.level_kind,
        "bounded_simulation_count": int(
            not bounded_cached
        ),
        "reference_simulation_count": int(
            not reference_cached
        ),
        "simulation_cache_hits": int(
            bounded_cached
        )
        + int(reference_cached),
        "resumed": False,
    }


def _add_absolute_fields(
    row: dict[str, Any],
    prefix: str,
    summary: Mapping[str, Any],
) -> None:
    for field in _ABSOLUTE_SUMMARY_FIELDS:
        value = summary.get(field)
        if value is None:
            continue
        row[
            f"{prefix}_{field}"
        ] = float(value)


def _progress_message(
    result: Mapping[str, Any],
) -> str:
    cache_hits = int(
        result.get(
            "simulation_cache_hits", 0
        )
    )
    resumed = bool(
        result.get("resumed")
    )
    status = (
        "resumed"
        if resumed
        else (
            f"{cache_hits}/2 cache hits"
        )
    )
    return (
        f"{result.get('parameter_path')} "
        f"[{result.get('level_kind')}] — "
        f"{status}"
    )
