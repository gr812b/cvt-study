"""Route the six-report study types to their correct engines and reports."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from cvt_track_study.config import ProjectLoader
from cvt_track_study.reports import (
    write_design_comparison_report,
    write_full_uncertainty_report,
    write_structural_report_manifest,
)
from cvt_track_study.track.robustness import run_track_robustness_project
from cvt_track_study.runtime.results import write_results_index

from .service_v9 import run_study_project as _run_legacy_study
from .ensemble_v10 import run_joint_ensemble_project


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
        output = run_track_robustness_project(
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

    runner = (
        run_joint_ensemble_project
        if study_type in {"full_uncertainty", "design_sweep"}
        else _run_legacy_study
    )
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
        write_design_comparison_report(output)
    write_results_index(resolution.paths.results_directory)
    return output
