"""Canonical GPX ingestion data structures."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from cvt_track_study.config.diagnostics import Diagnostic


CANONICAL_POINT_COLUMNS = (
    "run_id",
    "vehicle_id",
    "driver_id",
    "source_file",
    "source_sha256",
    "track_index",
    "segment_index",
    "point_index",
    "timestamp_utc",
    "latitude_deg",
    "longitude_deg",
    "elevation_m",
    "reported_speed_mps",
    "derived_speed_mps",
    "analysis_speed_mps",
    "course_deg",
    "fix_type",
    "satellites",
    "hdop",
    "vdop",
    "pdop",
    "step_distance_m",
    "time_step_s",
    "extension_json",
)


@dataclass(frozen=True)
class GPXRunMetadata:
    run_id: str
    vehicle_id: str
    driver_id: str
    source_file: Path
    use_for_centreline: bool
    use_for_gate_evidence: bool


@dataclass(frozen=True)
class GPXIngestionResult:
    metadata: GPXRunMetadata
    points: pd.DataFrame
    segments: pd.DataFrame
    summary: dict[str, Any]
    diagnostics: tuple[Diagnostic, ...]

    @property
    def error_count(self) -> int:
        return sum(item.severity.value == "error" for item in self.diagnostics)

    @property
    def warning_count(self) -> int:
        return sum(item.severity.value == "warning" for item in self.diagnostics)
