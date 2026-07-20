"""Nominal simulation facade with the canonical in-depth HTML report."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .service import run_baseline_project as _run_baseline_project
from cvt_track_study.reports import write_nominal_simulation_report
from cvt_track_study.config import ProjectLoader
from cvt_track_study.runtime.results import write_results_index


def run_baseline_project(*args: Any, **kwargs: Any) -> Path:
    output = _run_baseline_project(*args, **kwargs)
    write_nominal_simulation_report(output)
    project = args[0] if args else kwargs.get("project")
    if project is not None:
        resolution = ProjectLoader().resolve(project)
        write_results_index(resolution.paths.results_directory)
    return output
