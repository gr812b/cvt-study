"""Joint answer-uncertainty and paired design studies over track ensembles.

The existing uncertainty engine samples measured traversal and declared physical
inputs.  This v10 orchestration adds an outer, explicitly unweighted ensemble of
track reconstructions produced by the data-only track-robustness report.  Each
scenario retains its ``track_case_id`` so epistemic track policies are never
mistaken for calibrated probabilities.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import shutil
from typing import Any

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
from cvt_track_study.simulation.service import SimulationError
from cvt_track_study.track import build_project_track
from cvt_track_study.track.robustness import run_track_robustness_project
from cvt_track_study.uncertainty import (
    GateSampleIdentity,
    SamplingPlan,
    ScenarioDraw,
    ScenarioSampler,
    build_input_registry,
    correlation_groups_from_study,
)

from . import service as phase6_service
from . import service_v8
from .analysis import (
    convergence_summary,
    input_contracts,
    quality_summary,
    summarize_study,
)
from .planning import study_plan
from .reporting_v8 import write_study_outputs


@dataclass(frozen=True)
class TrackVariant:
    case_id: str
    category: str
    label: str
    bundle_path: Path
    bundle: TrackBundle



@dataclass(frozen=True)
class ScheduledScenario:
    variant: TrackVariant
    scenario: ScenarioDraw
    base_draw_id: int


def run_joint_ensemble_project(
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
    """Run full uncertainty or design comparison across track variants."""

    if workers < 1:
        raise SimulationError("workers must be at least one.")
    resolution = ProjectLoader().resolve(project, study=study)
    if resolution.error_count:
        details = "\n".join(item.format() for item in resolution.diagnostics)
        raise SimulationError(f"Project validation failed:\n{details}")
    raw = resolution.data.get("studies", {}).get(study)
    if not isinstance(raw, Mapping):
        raise SimulationError(f"Study {study!r} was not found.")
    study_type = str(raw.get("study", {}).get("type", ""))
    if study_type not in {"full_uncertainty", "design_sweep"}:
        raise SimulationError(
            "Track-ensemble orchestration supports full_uncertainty and design_sweep."
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

    nominal_path = _nominal_bundle_path(project, resolution, bundle_path)
    nominal_bundle = load_track_bundle(nominal_path)
    variants, ensemble_source = _track_variants(
        project=project,
        resolution=resolution,
        raw=raw,
        nominal_path=nominal_path,
        nominal_bundle=nominal_bundle,
        progress=progress,
    )

    registry = build_input_registry(
        vehicle_raw=vehicle_raw,
        base_study_raw=base_study,
        track_raw=resolution.data["track"],
        bundle=nominal_bundle,
    )
    design_points, sampling_mode, replicates = study_plan(
        study_type, raw, registry, replicates_override
    )
    design_path = design_points[0].path if study_type == "design_sweep" else None
    sampling = raw.get("sampling", {})
    seed = int(raw.get("study", {}).get("random_seed", 20260715))
    plan = SamplingPlan(
        mode=sampling_mode,
        replicates=replicates,
        random_seed=seed,
        selected_paths=tuple(str(path) for path in sampling.get("paths", ())),
        excluded_paths=(design_path,) if design_path else (),
        correlation_groups=correlation_groups_from_study(raw),
        gate_sampling=str(sampling.get("gate_sampling", "paired_lap")),
    )

    samplers = {
        variant.case_id: ScenarioSampler(registry=registry, bundle=variant.bundle, plan=plan)
        for variant in variants
    }
    sampling_layout = _sampling_layout(sampling)
    selected, schedule_metadata = _schedule_scenarios(
        variants=variants,
        samplers=samplers,
        plan=plan,
        replicates=replicates,
        sampling_layout=sampling_layout,
    )

    fingerprint = canonical_fingerprint(
        {
            "schema": "framework-joint-track-ensemble-v1",
            "study_name": study,
            "study": raw,
            "base_study": base_study,
            "vehicle": vehicle_raw,
            "track": resolution.data["track"],
            "nominal_bundle": nominal_bundle.data.get("content_fingerprint_sha256"),
            "track_variants": [
                {
                    "case_id": variant.case_id,
                    "fingerprint": variant.bundle.data.get("content_fingerprint_sha256"),
                }
                for variant in variants
            ],
            "replicates_override": replicates_override,
            "sampling_layout": sampling_layout,
            "base_draw_count": schedule_metadata["base_draw_count"],
        }
    )
    default_output = (
        resolution.paths.results_directory
        / study_type
        / f"{service_v8._safe_name(run_name or study)}--{fingerprint[:10]}"
    )
    output = (output_directory.resolve() if output_directory else default_output.resolve())
    try:
        workspace = ResultWorkspace(
            output, fingerprint=fingerprint, resume=resume, restart=restart
        )
    except RuntimeError as exc:
        raise SimulationError(str(exc)) from exc
    cache = SimulationCache(
        resolution.paths.root / ".drivetrain-study-cache" / "simulations",
        enabled=use_cache,
    )
    reporter = ProgressReporter(total=len(selected), label="scenarios", enabled=progress)
    reporter.begin(
        f"{len(selected)} paired scenarios across {len(variants)} track case(s), "
        f"{schedule_metadata['base_draw_count']} common draw(s), "
        f"{len(design_points)} design point(s), {workers} worker(s)"
    )

    def execute_or_resume(item: ScheduledScenario) -> dict[str, Any]:
        variant = item.variant
        scenario = item.scenario
        checkpoint = workspace.load_checkpoint(scenario.replicate)
        if checkpoint is not None:
            result = dict(checkpoint["result"])
            result["resumed"] = True
            return result
        result = service_v8._execute_scenario(
            scenario=scenario,
            design_points=design_points,
            study_type=study_type,
            vehicle_id=vehicle_id,
            vehicle_raw=vehicle_raw,
            base_study=base_study,
            track_raw=resolution.data["track"],
            bundle=variant.bundle,
            cache=cache,
        )
        for row in result["rows"]:
            row["base_draw_id"] = item.base_draw_id
            row["track_pair_id"] = f"draw-{item.base_draw_id:06d}"
            row["track_case_id"] = variant.case_id
            row["track_case_category"] = variant.category
            row["track_bundle_fingerprint"] = variant.bundle.data.get(
                "content_fingerprint_sha256"
            )
        result["track_case_id"] = variant.case_id
        result["base_draw_id"] = item.base_draw_id
        workspace.write_checkpoint(scenario.replicate, {"result": result})
        return result

    results: list[dict[str, Any]] = []
    if workers == 1 or len(selected) == 1:
        for item in selected:
            result = execute_or_resume(item)
            results.append(result)
            reporter.advance(
                f"scenario {item.scenario.replicate}; draw={item.base_draw_id}; track={result['track_case_id']}"
            )
    else:
        with ThreadPoolExecutor(max_workers=min(workers, len(selected))) as executor:
            futures = {executor.submit(execute_or_resume, item): item for item in selected}
            for future in as_completed(futures):
                item = futures[future]
                result = future.result()
                results.append(result)
                reporter.advance(
                    f"scenario {item.scenario.replicate}; draw={item.base_draw_id}; track={result['track_case_id']}"
                )

    order = {point.identifier: index for index, point in enumerate(design_points)}
    rows = [row for result in results for row in result["rows"]]
    rows.sort(key=lambda row: (int(row["replicate"]), order[str(row["design_id"])]))
    quality = quality_summary(rows, raw)
    summary = {
        **summarize_study(study_type, rows, raw, seed),
        "numerical_quality": quality,
    }
    convergence = convergence_summary(study_type, rows)

    scenario_payloads: list[dict[str, Any]] = []
    for item in selected:
        variant = item.variant
        scenario = item.scenario
        payload = dict(scenario.serializable())
        payload["base_draw_id"] = item.base_draw_id
        payload["track_pair_id"] = f"draw-{item.base_draw_id:06d}"
        payload["track_case_id"] = variant.case_id
        payload["track_case_category"] = variant.category
        payload["track_bundle_fingerprint"] = variant.bundle.data.get(
            "content_fingerprint_sha256"
        )
        scenario_payloads.append(payload)

    sampled_gate_ids = sorted(
        {
            gate_id
            for item in selected
            for gate_id in item.scenario.gate_target_speeds_mps
        }
    )
    fallback_gate_ids = sorted(
        {
            gate_id
            for item in selected
            for gate_id in item.scenario.independently_sampled_gate_ids
        }
    )
    stochastic_by_role: dict[str, list[str]] = {}
    for registered in registry.stochastic():
        stochastic_by_role.setdefault(registered.category, []).append(registered.path)
    requested = int(schedule_metadata["requested_sampler_draw_count"])
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "framework_contract": "measured-track-drivetrain-framework-v1.0-six-report",
        "study_fingerprint_sha256": fingerprint,
        "study_name": study,
        "study_type": study_type,
        "vehicle_id": vehicle_id,
        "sampling_mode": sampling_mode,
        "random_seed": seed,
        "scenario_count": len(selected),
        "resumed_scenario_count": sum(bool(item.get("resumed")) for item in results),
        "design_point_count": len(design_points),
        "bounded_case_count": sum(int(item["bounded_case_count"]) for item in results),
        "reference_case_count": sum(int(item["reference_case_count"]) for item in results),
        "bounded_simulation_count": sum(int(item["bounded_simulation_count"]) for item in results),
        "reference_simulation_count": sum(int(item["reference_simulation_count"]) for item in results),
        "reference_cache_hits": sum(int(item["reference_cache_hits"]) for item in results),
        "simulation_cache_hits": sum(int(item["simulation_cache_hits"]) for item in results),
        "simulation_cache_enabled": cache.enabled,
        "simulation_cache_status": cache.status(),
        "parallel_workers": workers,
        "paired_scenarios": True,
        "reference_cache_policy": (
            "one scenario-level infinite reference shared by every design candidate"
        ),
        "sampled_input_paths": list(next(iter(samplers.values())).sampled_paths),
        "sampled_input_count": len(next(iter(samplers.values())).sampled_paths),
        "declared_stochastic_input_paths_by_role": {
            role: sorted(paths) for role, paths in sorted(stochastic_by_role.items())
        },
        "gate_sampling_policy": plan.gate_sampling,
        "sampled_gate_ids": sampled_gate_ids,
        "sampled_gate_count": len(sampled_gate_ids),
        "paired_gate_identity_count": int(
            schedule_metadata["common_gate_identity_count"]
        ),
        "independent_gate_fallback_ids": fallback_gate_ids,
        "track_bundle_content_fingerprint": nominal_bundle.data.get(
            "content_fingerprint_sha256"
        ),
        "track_bundle_sha256": nominal_bundle.sha256,
        "track_ensemble_enabled": len(variants) > 1,
        "track_ensemble_case_count": len(variants),
        "track_ensemble_case_ids": [variant.case_id for variant in variants],
        "track_ensemble_source": str(ensemble_source) if ensemble_source else None,
        "track_ensemble_policy": (
            "unweighted_epistemic_scenarios_not_calibrated_probabilities"
        ),
        "track_case_assignment": str(schedule_metadata["track_case_assignment"]),
        "sampling_layout": sampling_layout,
        "base_draw_count": int(schedule_metadata["base_draw_count"]),
        "scenarios_per_track_case": int(schedule_metadata["scenarios_per_track_case"]),
        "track_case_pairing_complete": bool(schedule_metadata["track_case_pairing_complete"]),
        "common_gate_identity_count_across_track_ensemble": int(
            schedule_metadata["common_gate_identity_count"]
        ),
        "sampling_replicates_interpretation": str(
            schedule_metadata["replicates_interpretation"]
        ),
        "unused_sampler_draw_count": max(0, requested - len(selected)),
        "bootstrap_resamples": int(raw.get("reporting", {}).get("bootstrap_resamples", 1000)),
        "numerical_quality": quality,
        "evidence_assessment": dict(
            assess_evidence(diagnostics=resolution.diagnostics, bundle=nominal_bundle)
        ),
        "uncertainty_not_propagated": [
            "telemetry elevation uncertainty (grade force remains disabled pending the materiality screen)"
        ],
    }

    nominal_reference_rows: list[dict[str, Any]] = []
    if study_type == "full_uncertainty":
        nominal_scenario = ScenarioDraw(
            replicate=-1,
            seed=seed,
            sampling_mode="nominal",
        )
        nominal_result = service_v8._execute_scenario(
            scenario=nominal_scenario,
            design_points=design_points,
            study_type=study_type,
            vehicle_id=vehicle_id,
            vehicle_raw=vehicle_raw,
            base_study=base_study,
            track_raw=resolution.data["track"],
            bundle=nominal_bundle,
            cache=cache,
        )
        nominal_reference_rows = list(nominal_result["rows"])
        for row in nominal_reference_rows:
            row["base_draw_id"] = -1
            row["track_pair_id"] = "nominal-reference"
            row["track_case_id"] = "nominal"
            row["track_case_category"] = "nominal"
            row["track_bundle_fingerprint"] = nominal_bundle.data.get(
                "content_fingerprint_sha256"
            )
        manifest["nominal_reference_available"] = bool(nominal_reference_rows)
        manifest["nominal_reference_bounded_simulation_count"] = int(
            nominal_result["bounded_simulation_count"]
        )
        manifest["nominal_reference_reference_simulation_count"] = int(
            nominal_result["reference_simulation_count"]
        )

    resolution.export(workspace.path / "resolved_inputs")
    phase6_service._write_bundle_snapshot(workspace.path, nominal_bundle, nominal_path)
    _write_track_ensemble_snapshot(workspace.path, variants, ensemble_source)
    write_study_outputs(
        output=workspace.path,
        rows=rows,
        scenario_draws=scenario_payloads,
        summary=summary,
        convergence=convergence,
        manifest=manifest,
        input_contracts=input_contracts(registry),
        study_type=study_type,
    )
    if nominal_reference_rows:
        (workspace.path / "nominal_reference.json").write_text(
            json.dumps(nominal_reference_rows, indent=2, sort_keys=True, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
    write_provenance(
        workspace.path,
        build_provenance(
            command=command
            or (
                "drivetrain-study",
                "run",
                "full-uncertainty" if study_type == "full_uncertainty" else "design-comparison",
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
        f"{len(rows)} design-scenario rows; {len(variants)} track case(s)"
    )
    committed = workspace.commit()
    write_results_index(resolution.paths.results_directory)
    return committed



def _sampling_layout(sampling: Mapping[str, Any]) -> str:
    raw = str(sampling.get("layout", "round_robin_track_cases")).strip().lower()
    aliases = {
        "round_robin": "round_robin_track_cases",
        "legacy": "round_robin_track_cases",
        "crossed": "cross_track_cases",
        "fully_crossed": "cross_track_cases",
        "paired_track_cases": "cross_track_cases",
    }
    layout = aliases.get(raw, raw)
    if layout not in {"round_robin_track_cases", "cross_track_cases"}:
        raise SimulationError(
            "sampling.layout must be 'round_robin_track_cases' or "
            "'cross_track_cases'."
        )
    return layout


def _schedule_scenarios(
    *,
    variants: Sequence[TrackVariant],
    samplers: Mapping[str, ScenarioSampler],
    plan: SamplingPlan,
    replicates: int,
    sampling_layout: str,
) -> tuple[tuple[ScheduledScenario, ...], dict[str, Any]]:
    if sampling_layout == "round_robin_track_cases":
        draws = {case_id: sampler.draw_all() for case_id, sampler in samplers.items()}
        scheduled = tuple(
            ScheduledScenario(
                variant=variants[replicate % len(variants)],
                scenario=draws[variants[replicate % len(variants)].case_id][replicate],
                base_draw_id=replicate,
            )
            for replicate in range(replicates)
        )
        counts = [
            sum(item.variant.case_id == variant.case_id for item in scheduled)
            for variant in variants
        ]
        common_count = min(
            (sampler.paired_gate_identity_count for sampler in samplers.values()),
            default=0,
        )
        return scheduled, {
            "base_draw_count": replicates,
            "scenarios_per_track_case": min(counts, default=0),
            "track_case_pairing_complete": len(variants) <= 1,
            "common_gate_identity_count": common_count,
            "requested_sampler_draw_count": sum(len(points) for points in draws.values()),
            "track_case_assignment": "deterministic_round_robin_across_replicates",
            "replicates_interpretation": "total joint scenarios",
        }

    if plan.gate_sampling != "paired_lap":
        raise SimulationError(
            "sampling.layout='cross_track_cases' requires "
            "sampling.gate_sampling='paired_lap' so one measured traversal "
            "identity can be replayed on every track interpretation."
        )

    nominal_sampler = samplers[variants[0].case_id]
    base_draws = nominal_sampler.draw_all()
    common_identities = _common_gate_identities(variants)
    has_active_gates = any(variant.bundle.active_speed_gates for variant in variants)
    if has_active_gates and not common_identities:
        raise SimulationError(
            "No measured lap identity is available at every active gate across "
            "the selected track ensemble. Reduce the track-case set or add "
            "evidence before using cross_track_cases."
        )

    scheduled: list[ScheduledScenario] = []
    scenario_index = 0
    for base_draw_id, base in enumerate(base_draws):
        identity = None
        if common_identities:
            chooser = random.Random(int(base.seed) ^ 0x4356545F54524143)
            identity = common_identities[chooser.randrange(len(common_identities))]
        for variant in variants:
            gate_values = (
                _gate_values_for_identity(variant.bundle, identity)
                if identity is not None
                else {}
            )
            scenario = ScenarioDraw(
                replicate=scenario_index,
                seed=base.seed,
                sampling_mode=base.sampling_mode,
                quantity_values_si=base.quantity_values_si,
                choice_values=base.choice_values,
                gate_target_speeds_mps=gate_values,
                gate_sample_identity=identity,
                independently_sampled_gate_ids=(),
            )
            scheduled.append(
                ScheduledScenario(
                    variant=variant,
                    scenario=scenario,
                    base_draw_id=base_draw_id,
                )
            )
            scenario_index += 1

    return tuple(scheduled), {
        "base_draw_count": len(base_draws),
        "scenarios_per_track_case": len(base_draws),
        "track_case_pairing_complete": True,
        "common_gate_identity_count": len(common_identities),
        "requested_sampler_draw_count": len(base_draws),
        "track_case_assignment": "fully_crossed_common_draws",
        "replicates_interpretation": "common structural/traversal draws per track case",
    }


def _gate_samples_for_bundle(
    bundle: TrackBundle,
) -> dict[str, dict[tuple[str, int, str, str], float]]:
    rows: dict[str, dict[tuple[str, int, str, str], float]] = {}
    for gate in bundle.active_speed_gates:
        parsed: dict[tuple[str, int, str, str], float] = {}
        distribution = gate.get("target_speed_distribution", {})
        for sample in distribution.get("samples", []):
            key = (
                str(sample["run_id"]),
                int(sample["lap_id"]),
                str(sample["vehicle_id"]),
                str(sample["driver_id"]),
            )
            parsed[key] = float(sample["value_mps"])
        if parsed:
            rows[str(gate["id"])] = parsed
    return rows


def _common_gate_identities(
    variants: Sequence[TrackVariant],
) -> tuple[GateSampleIdentity, ...]:
    identity_sets: list[set[tuple[str, int, str, str]]] = []
    for variant in variants:
        for samples in _gate_samples_for_bundle(variant.bundle).values():
            identity_sets.append(set(samples))
    if not identity_sets:
        return ()
    common = set.intersection(*identity_sets)
    return tuple(GateSampleIdentity(*key) for key in sorted(common))


def _gate_values_for_identity(
    bundle: TrackBundle,
    identity: GateSampleIdentity,
) -> dict[str, float]:
    values: dict[str, float] = {}
    missing: list[str] = []
    for gate_id, samples in _gate_samples_for_bundle(bundle).items():
        if identity.key not in samples:
            missing.append(gate_id)
            continue
        values[gate_id] = samples[identity.key]
    if missing:
        raise SimulationError(
            f"Measured traversal {identity.key!r} lacks gate evidence for "
            + ", ".join(sorted(missing))
        )
    return values


def _nominal_bundle_path(
    project: str | Path,
    resolution: Any,
    bundle_path: Path | None,
) -> Path:
    if bundle_path is not None:
        return bundle_path.resolve()
    build = build_project_track(project)
    if build.error_count:
        raise SimulationError("Track build failed before study execution.")
    return (build.output_directory / "track_bundle.json").resolve()


def _track_variants(
    *,
    project: str | Path,
    resolution: Any,
    raw: Mapping[str, Any],
    nominal_path: Path,
    nominal_bundle: TrackBundle,
    progress: bool,
) -> tuple[tuple[TrackVariant, ...], Path | None]:
    config = raw.get("track_ensemble", {})
    if not isinstance(config, Mapping):
        config = {}
    enabled = bool(config.get("enabled", True))
    nominal = TrackVariant(
        "nominal", "nominal", "Nominal reconstructed track", nominal_path, nominal_bundle
    )
    if not enabled:
        return (nominal,), None

    source_value = config.get("result_directory", "latest")
    source: Path | None = None
    if source_value not in (None, "", "latest"):
        source = Path(str(source_value))
        if not source.is_absolute():
            source = (resolution.paths.root / source).resolve()
    elif source_value == "latest":
        source = _latest_track_robustness_result(resolution.paths.results_directory)

    if source is None and bool(config.get("auto_build", True)):
        robustness_study = str(config.get("study", "track_robustness"))
        source = run_track_robustness_project(
            project,
            study=robustness_study,
            resume=True,
            progress=progress,
            run_name=robustness_study,
            command=("drivetrain-study", "run", "track-robustness", str(project)),
        )
    if source is None:
        return (nominal,), None

    manifest_path = source / "track_ensemble_manifest.json"
    if not manifest_path.is_file():
        raise SimulationError(
            f"Track robustness result lacks track_ensemble_manifest.json: {source}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    allowed_categories = {
        str(value)
        for value in config.get(
            "include_categories",
            [
                "nominal",
                "data_support",
                "telemetry_cleanup",
                "centreline",
                "event_windows",
                "gate_policy",
                "gate_weighting",
            ],
        )
    }
    maximum = int(config.get("maximum_cases", 20))
    records = []
    for record in manifest.get("cases", []):
        if not isinstance(record, Mapping):
            continue
        if not bool(record.get("eligible_for_answer_uncertainty", False)):
            continue
        if str(record.get("category", "")) not in allowed_categories:
            continue
        if str(record.get("case_id")) == "nominal":
            continue
        path = (source / str(record.get("bundle", ""))).resolve()
        if not path.is_file():
            continue
        records.append(
            TrackVariant(
                case_id=str(record["case_id"]),
                category=str(record.get("category", "unknown")),
                label=str(record.get("label", record["case_id"])),
                bundle_path=path,
                bundle=load_track_bundle(path),
            )
        )
    records = records[: max(0, maximum - 1)]
    return tuple([nominal, *records]), source


def _latest_track_robustness_result(results_directory: Path) -> Path | None:
    root = results_directory / "track_robustness"
    if not root.is_dir():
        return None
    candidates = [
        path
        for path in root.iterdir()
        if path.is_dir() and (path / "track_ensemble_manifest.json").is_file()
    ]
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


def _write_track_ensemble_snapshot(
    output: Path,
    variants: Sequence[TrackVariant],
    source: Path | None,
) -> None:
    directory = output / "track_ensemble"
    directory.mkdir(parents=True, exist_ok=True)
    records = []
    for variant in variants:
        target = directory / f"{service_v8._safe_name(variant.case_id)}.json"
        shutil.copy2(variant.bundle_path, target)
        records.append(
            {
                "case_id": variant.case_id,
                "category": variant.category,
                "label": variant.label,
                "file": str(target.relative_to(output)),
                "fingerprint": variant.bundle.data.get("content_fingerprint_sha256"),
            }
        )
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "interpretation": (
                    "Unweighted epistemic reconstruction scenarios; not probability weights."
                ),
                "source_track_robustness_result": str(source) if source else None,
                "cases": records,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
