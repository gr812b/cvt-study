"""Track-build facade that registers the canonical evidence report."""

from __future__ import annotations

from pathlib import Path

from .service import build_project_track as _build_project_track
from .model import TrackBuildResult
from cvt_track_study.reports import write_track_evidence_report
from cvt_track_study.runtime.results import write_results_index


def build_project_track(
    project: str | Path,
    *,
    output_directory: Path | None = None,
) -> TrackBuildResult:
    result = _build_project_track(project, output_directory=output_directory)
    if result.output_directory is not None:
        write_track_evidence_report(result.output_directory)
        write_results_index(result.resolution.paths.results_directory)
    return result
