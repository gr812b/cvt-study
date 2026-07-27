"""Canonical telemetry ingestion data structures.

The historical GPX names remain public aliases so existing callers do not need
to change when FIT recordings or optional lap-time reconstruction are used.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
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
    "source_format",
    "track_index",
    "segment_index",
    "point_index",
    "timestamp_utc",
    "latitude_deg",
    "longitude_deg",
    "elevation_m",
    "elevation_source",
    "device_distance_m",
    "device_speed_mps",
    "reported_speed_mps",
    "derived_speed_mps",
    "analysis_speed_mps",
    "analysis_speed_source",
    "speed_certainty",
    "course_deg",
    "fix_type",
    "satellites",
    "horizontal_accuracy_m",
    "hdop",
    "vdop",
    "pdop",
    "step_distance_m",
    "time_step_s",
    "extension_json",
)


@dataclass(frozen=True)
class LapTimeReconstructionConfig:
    """Optional contract for assigning timing to an untimed GPX.

    The GPX point order is retained. Each detected lap receives the duration
    declared in the CSV and a constant point-to-point time interval within
    that lap. Local speed then follows from measured point spacing divided by
    that reconstructed interval.
    """

    lap_times_file: Path
    lap_index_column: str = "lap"
    lap_time_column: str = "lap_time"
    include_column: str | None = "include"
    gate_radius_m: float | None = None
    minimum_points_per_lap: int = 30
    expected_point_period_s: float = 1.0
    maximum_point_period_error_fraction: float = 0.35
    alignment_mode: str = "cadence_dynamic_programming"
    minimum_csv_lap_time_s: float | None = None
    maximum_csv_lap_time_s: float | None = None
    maximum_alignment_error_fraction: float = 0.65
    skip_detected_lap_penalty: float = 1.0
    skip_csv_lap_penalty: float = 1.5
    minimum_matched_laps: int = 3
    allow_unmatched_laps: bool = False
    export_augmented_gpx: bool = True
    synthetic_start_time_utc: datetime | None = None


@dataclass(frozen=True)
class GPXRunMetadata:
    run_id: str
    vehicle_id: str
    driver_id: str
    source_file: Path
    use_for_centreline: bool
    use_for_gate_evidence: bool
    lap_time_reconstruction: LapTimeReconstructionConfig | None = None


@dataclass(frozen=True)
class GPXIngestionResult:
    metadata: GPXRunMetadata
    points: pd.DataFrame
    segments: pd.DataFrame
    summary: dict[str, Any]
    diagnostics: tuple[Diagnostic, ...]
    rejected_points: pd.DataFrame = field(default_factory=pd.DataFrame)
    reconstruction_laps: pd.DataFrame = field(default_factory=pd.DataFrame)
    augmented_gpx_text: str | None = None

    @property
    def error_count(self) -> int:
        return sum(item.severity.value == "error" for item in self.diagnostics)

    @property
    def warning_count(self) -> int:
        return sum(item.severity.value == "warning" for item in self.diagnostics)


# Format-neutral names for new code; legacy names are intentionally retained.
TelemetryRunMetadata = GPXRunMetadata
TelemetryIngestionResult = GPXIngestionResult
