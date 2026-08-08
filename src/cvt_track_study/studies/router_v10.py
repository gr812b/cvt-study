"""Route the six-report study types to their correct engines and reports."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from cvt_track_study.config import ProjectLoader
from cvt_track_study.reports import (
    write_design_comparison_report,
    write_full_uncertainty_report,
    write_structural_report_manifest,
)
from cvt_track_study.reports.design_grid_report import (
    write_multivariable_design_comparison_report,
)
from cvt_track_study.track.robustness import run_track_robustness_project
from cvt_track_study.track.robustness_family import (
    route_family_robustness_enabled,
    run_route_family_track_robustness_project,
)
from cvt_track_study.runtime.results import write_results_index

from .service_v9 import run_study_project as _run_legacy_study
from .ensemble_v10 import run_joint_ensemble_project
from .design_replay_v11 import run_uncertainty_informed_design_project
from .scenario_reduction import uncertainty_informed_design_enabled
from .route_family import family_balanced_track_ensemble_runtime


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
    resolution = ProjectLoader().resolve(project)
    raw = resolution.data.get("studies", {}).get(study)
    if not isinstance(raw, Mapping):
        raise ValueError(f"Study {study!r} was not found.")
    study_type = str(raw.get("study", {}).get("type", ""))

    if study_type == "track_robustness":
        if bundle_path is not None:
            raise ValueError(
                "Track robustness reconstructs the track from raw telemetry and cannot use --bundle."
            )
        if replicates_override is not None:
            raise ValueError(
                "Track robustness uses explicit reconstruction cases, not Monte Carlo replicates."
            )
        runner = (
            run_route_family_track_robustness_project
            if route_family_robustness_enabled(resolution)
            else run_track_robustness_project
        )
        output = runner(
            project,
            study=study,
            output_directory=output_directory,
            workers=workers,
            resume=resume,
            restart=restart,
            progress=progress,
            run_name=run_name,
            command=command,
        )
        write_results_index(resolution.paths.results_directory)
        return output

    if study_type == "design_sweep" and uncertainty_informed_design_enabled(raw):
        runner = run_uncertainty_informed_design_project
    elif study_type in {"full_uncertainty", "design_sweep"}:
        runner = run_joint_ensemble_project
    else:
        runner = _run_legacy_study

    # Joint uncertainty/design studies already understand a flat track ensemble.
    # When the latest robustness result is a route family, temporarily adapt the
    # loader so its configured maximum_cases is applied symmetrically across the
    # long/short course traversals instead of truncating one route first.
    if study_type in {"full_uncertainty", "design_sweep"} and runner is run_joint_ensemble_project:
        with family_balanced_track_ensemble_runtime():
            output = runner(
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
    else:
        output = runner(
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
    if study_type == "structural_sensitivity":
        write_structural_report_manifest(output)
    elif study_type == "full_uncertainty":
        write_full_uncertainty_report(output)
    elif study_type == "design_sweep":
        variables = raw.get("design_variables")
        if isinstance(variables, list) and len(variables) >= 2:
            write_multivariable_design_comparison_report(output)
        else:
            write_design_comparison_report(output)
    write_results_index(resolution.paths.results_directory)
    return output
