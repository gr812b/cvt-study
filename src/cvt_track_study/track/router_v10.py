"""Track-build facade that registers the canonical evidence report."""

from __future__ import annotations

from pathlib import Path

from .service import build_project_track as _build_project_track
from .model import TrackBuildResult
from .family_review import augment_track_review_with_route_family
from cvt_track_study.reports import write_track_evidence_report
from cvt_track_study.runtime.results import write_results_index


def build_project_track(
    project: str | Path,
    *,
    output_directory: Path | None = None,
) -> TrackBuildResult:
    result = _build_project_track(project, output_directory=output_directory)
    if result.output_directory is not None:
        # The canonical report is generated after the lower-level track exporter.
        # Route-family augmentation therefore belongs here, after report generation,
        # otherwise the framework postprocessor overwrites it.
        canonical_report = write_track_evidence_report(result.output_directory)
        review = result.output_directory / "review"
        family_map = review / "route_family_map.png"
        augment_track_review_with_route_family(
            canonical_report,
            family_map,
            result,
        )

        # Preserve the historical alias with the same branch-aware content.
        legacy_report = review / "track_review.html"
        if legacy_report.resolve() != canonical_report.resolve():
            augment_track_review_with_route_family(
                legacy_report,
                family_map,
                result,
            )
        write_results_index(result.resolution.paths.results_directory)
    return result
