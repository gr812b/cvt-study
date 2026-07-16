"""Track reconstruction result structures."""

from __future__ import annotations

from dataclasses import dataclass
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
    output_directory: Path | None = None

    @property
    def error_count(self) -> int:
        return sum(item.severity.value == "error" for item in self.diagnostics)

    @property
    def warning_count(self) -> int:
        return sum(item.severity.value == "warning" for item in self.diagnostics)
