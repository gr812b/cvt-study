"""Resolve one [[runs]] declaration into typed ingestion metadata."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .model import GPXRunMetadata, LapTimeReconstructionConfig


def metadata_from_run_config(
    raw: Mapping[str, Any],
    *,
    runs_directory: Path,
) -> GPXRunMetadata:
    reconstruction = _lap_time_reconstruction(
        raw.get("lap_time_reconstruction"),
        runs_directory=runs_directory,
    )
    return GPXRunMetadata(
        run_id=str(raw["run_id"]),
        vehicle_id=str(raw["vehicle_id"]),
        driver_id=str(raw["driver_id"]),
        source_file=(runs_directory / str(raw["file"])).resolve(),
        use_for_centreline=bool(raw["use_for_centreline"]),
        use_for_gate_evidence=bool(raw["use_for_gate_evidence"]),
        lap_time_reconstruction=reconstruction,
    )


def _lap_time_reconstruction(
    raw: Any,
    *,
    runs_directory: Path,
) -> LapTimeReconstructionConfig | None:
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise ValueError("runs.lap_time_reconstruction must be a TOML table.")
    enabled = raw.get("enabled", True)
    if not isinstance(enabled, bool):
        raise ValueError("runs.lap_time_reconstruction.enabled must be true or false.")
    if not enabled:
        return None

    file_text = str(raw.get("lap_times_file", "")).strip()
    if not file_text:
        raise ValueError(
            "Enabled lap-time reconstruction requires lap_times_file."
        )
    lap_times_file = (runs_directory / file_text).resolve()

    gate_radius = raw.get("gate_radius_m")
    gate_radius_m = None if gate_radius is None else _positive_float(
        gate_radius, "gate_radius_m"
    )

    synthetic_start = raw.get("synthetic_start_time_utc")
    synthetic_start_time_utc = (
        _timestamp(synthetic_start)
        if synthetic_start is not None
        else datetime(2000, 1, 1, tzinfo=timezone.utc)
    )

    include_column = raw.get("include_column", "include")
    if include_column is not None:
        include_column = _nonempty_string(include_column, "include_column")

    return LapTimeReconstructionConfig(
        lap_times_file=lap_times_file,
        lap_index_column=_nonempty_string(
            raw.get("lap_index_column", "lap"), "lap_index_column"
        ),
        lap_time_column=_nonempty_string(
            raw.get("lap_time_column", "lap_time"), "lap_time_column"
        ),
        include_column=include_column,
        gate_radius_m=gate_radius_m,
        minimum_points_per_lap=_positive_int(
            raw.get("minimum_points_per_lap", 30),
            "minimum_points_per_lap",
        ),
        expected_point_period_s=_positive_float(
            raw.get("expected_point_period_s", 1.0),
            "expected_point_period_s",
        ),
        maximum_point_period_error_fraction=_positive_float(
            raw.get("maximum_point_period_error_fraction", 0.35),
            "maximum_point_period_error_fraction",
        ),
        alignment_mode=_alignment_mode(
            raw.get("alignment_mode", "cadence_dynamic_programming")
        ),
        minimum_csv_lap_time_s=_optional_positive_float(
            raw.get("minimum_csv_lap_time_s"), "minimum_csv_lap_time_s"
        ),
        maximum_csv_lap_time_s=_optional_positive_float(
            raw.get("maximum_csv_lap_time_s"), "maximum_csv_lap_time_s"
        ),
        maximum_alignment_error_fraction=_positive_float(
            raw.get("maximum_alignment_error_fraction", 0.65),
            "maximum_alignment_error_fraction",
        ),
        skip_detected_lap_penalty=_positive_float(
            raw.get("skip_detected_lap_penalty", 1.0),
            "skip_detected_lap_penalty",
        ),
        skip_csv_lap_penalty=_positive_float(
            raw.get("skip_csv_lap_penalty", 1.5),
            "skip_csv_lap_penalty",
        ),
        minimum_matched_laps=_positive_int(
            raw.get("minimum_matched_laps", 3), "minimum_matched_laps"
        ),
        allow_unmatched_laps=_boolean(
            raw.get("allow_unmatched_laps", False),
            "allow_unmatched_laps",
        ),
        export_augmented_gpx=_boolean(
            raw.get("export_augmented_gpx", True),
            "export_augmented_gpx",
        ),
        synthetic_start_time_utc=synthetic_start_time_utc,
    )


def _nonempty_string(value: Any, name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} must be a non-empty string.")
    return text


def _positive_float(value: Any, name: str) -> float:
    number = float(value)
    if number <= 0.0:
        raise ValueError(f"{name} must be positive.")
    return number


def _positive_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be true or false.")
    return value


def _timestamp(value: Any) -> datetime:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError(
            "synthetic_start_time_utc must include an explicit timezone."
        )
    return parsed.astimezone(timezone.utc)


def _optional_positive_float(value: Any, name: str) -> float | None:
    return None if value is None else _positive_float(value, name)


def _alignment_mode(value: Any) -> str:
    mode = str(value).strip().lower()
    allowed = {"strict_index", "cadence_dynamic_programming"}
    if mode not in allowed:
        raise ValueError(f"alignment_mode must be one of: {', '.join(sorted(allowed))}.")
    return mode
