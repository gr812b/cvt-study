"""Track reconstruction result structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from cvt_track_study.config.diagnostics import Diagnostic
from cvt_track_study.config.project import ResolutionResult
from cvt_track_study.gpx.model import GPXIngestionResult

from .geo import Centreline


@dataclass(frozen=True)
class TrackBuildResult:
    resolution: ResolutionResult
    ingestion_results: tuple[GPXIngestionResult, ...]
    centreline: Centreline
    laps: pd.DataFrame
    matched_points: pd.DataFrame
    track_profile: pd.DataFrame
    event_projection: pd.DataFrame
    response_features: pd.DataFrame
    event_passes: pd.DataFrame
    gate_evidence: pd.DataFrame
    gate_review: pd.DataFrame
    diagnostics: tuple[Diagnostic, ...]
    metadata: dict[str, Any]
    rejected_map_points: pd.DataFrame = field(default_factory=pd.DataFrame)
    route_variant_summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    route_variant_pairwise: pd.DataFrame = field(default_factory=pd.DataFrame)
    route_variant_id: str = "route_001"
    nominal_route_variant_id: str = "route_001"
    shared_gate_evidence: pd.DataFrame = field(default_factory=pd.DataFrame)
    event_route_applicability: pd.DataFrame = field(default_factory=pd.DataFrame)
    course_cases: pd.DataFrame = field(default_factory=pd.DataFrame)
    branch_summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    event_route_sections: pd.DataFrame = field(default_factory=pd.DataFrame)
    route_family_members: dict[str, "TrackBuildResult"] = field(
        default_factory=dict, repr=False, compare=False
    )
    output_directory: Path | None = None

    @property
    def error_count(self) -> int:
        return sum(item.severity.value == "error" for item in self.diagnostics)

    @property
    def warning_count(self) -> int:
        return sum(item.severity.value == "warning" for item in self.diagnostics)

    @property
    def is_route_family(self) -> bool:
        return len(self.route_family_members) > 1
