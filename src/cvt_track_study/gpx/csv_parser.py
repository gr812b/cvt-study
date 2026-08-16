"""Native CSV telemetry parser normalized to the canonical telemetry schema."""

from __future__ import annotations

import json
import math
from typing import Any, Iterable

import numpy as np
import pandas as pd

from cvt_track_study.config.diagnostics import DiagnosticBag

from .model import CANONICAL_POINT_COLUMNS, GPXIngestionResult, GPXRunMetadata
from .parser import (
    TelemetryParseError,
    _derive_kinematics,
    _float_or_none,
    _normalize_numeric_columns,
    _parse_timestamp,
    _sha256,
    _summarize_segments,
)


class CSVParseError(TelemetryParseError):
    """Fatal CSV parsing or contract failure."""


_TIMESTAMP_ALIASES = ("timestamp_utc", "timestamp", "time", "datetime")
_LATITUDE_ALIASES = ("latitude_deg", "latitude", "lat")
_LONGITUDE_ALIASES = ("longitude_deg", "longitude", "lon", "lng")
_ELEVATION_ALIASES = ("elevation_m", "elevation", "altitude_m", "altitude")
_DISTANCE_ALIASES = ("distance_m", "device_distance_m", "distance")
_COURSE_ALIASES = ("course_deg", "course", "heading_deg", "heading")
_SPEED_ALIASES = (
    ("speed_mps", 1.0),
    ("device_speed_mps", 1.0),
    ("speed_kmh", 1.0 / 3.6),
    ("speed_kph", 1.0 / 3.6),
    ("speed_mph", 0.44704),
)


def ingest_csv_run(metadata: GPXRunMetadata) -> GPXIngestionResult:
    """Ingest timestamp/position/speed CSV as ordinary track telemetry.

    Native CSV speed is high-certainty measured gate evidence. Nothing in this
    parser interprets slow speeds, pauses, or outliers as traffic events.
    """
    diagnostics = DiagnosticBag()
    path = metadata.source_file
    try:
        frame = pd.read_csv(path)
    except (OSError, pd.errors.ParserError, UnicodeError) as exc:
        raise CSVParseError(f"Unable to parse CSV file {path}: {exc}") from exc
    if frame.empty:
        raise CSVParseError(f"CSV file {path} contains no rows.")

    timestamp_col = _find_column(frame, _TIMESTAMP_ALIASES)
    latitude_col = _find_column(frame, _LATITUDE_ALIASES)
    longitude_col = _find_column(frame, _LONGITUDE_ALIASES)
    speed_col, speed_scale = _find_speed_column(frame)
    elevation_col = _find_column(frame, _ELEVATION_ALIASES, required=False)
    distance_col = _find_column(frame, _DISTANCE_ALIASES, required=False)
    course_col = _find_column(frame, _COURSE_ALIASES, required=False)

    sha256 = _sha256(path)
    rows: list[dict[str, Any]] = []
    invalid_coordinates = 0
    invalid_time = 0
    naive_time = 0
    missing_elevation = 0

    for source_index, source_row in frame.iterrows():
        latitude = _float_or_none(source_row.get(latitude_col))
        longitude = _float_or_none(source_row.get(longitude_col))
        if (
            latitude is None
            or longitude is None
            or not -90.0 <= latitude <= 90.0
            or not -180.0 <= longitude <= 180.0
        ):
            invalid_coordinates += 1
            diagnostics.warning(
                "INVALID_CSV_COORDINATE",
                "CSV row has a missing or out-of-range position and was excluded.",
                path=f"row[{source_index}]",
                source=str(path),
            )
            continue

        raw_timestamp = source_row.get(timestamp_col)
        timestamp, was_naive = _parse_timestamp(
            None if pd.isna(raw_timestamp) else str(raw_timestamp)
        )
        if timestamp is None:
            invalid_time += 1
        elif was_naive:
            naive_time += 1

        speed = _float_or_none(source_row.get(speed_col))
        speed_mps = None if speed is None or speed < 0.0 else float(speed) * speed_scale
        elevation = (
            _float_or_none(source_row.get(elevation_col))
            if elevation_col is not None else None
        )
        if elevation is None:
            missing_elevation += 1
        distance = (
            _float_or_none(source_row.get(distance_col))
            if distance_col is not None else None
        )
        course = (
            _float_or_none(source_row.get(course_col))
            if course_col is not None else None
        )

        rows.append(
            {
                "run_id": metadata.run_id,
                "vehicle_id": metadata.vehicle_id,
                "driver_id": metadata.driver_id,
                "source_file": str(path),
                "source_sha256": sha256,
                "source_format": "csv",
                "track_index": 0,
                "segment_index": 0,
                "point_index": int(source_index),
                "timestamp_utc": timestamp,
                "latitude_deg": latitude,
                "longitude_deg": longitude,
                "elevation_m": elevation,
                "elevation_source": (
                    f"csv:{elevation_col}" if elevation_col is not None else "unavailable"
                ),
                "device_distance_m": distance,
                "device_speed_mps": speed_mps,
                "reported_speed_mps": math.nan,
                "derived_speed_mps": math.nan,
                "analysis_speed_mps": math.nan,
                "analysis_speed_source": "unavailable",
                "speed_certainty": "unavailable",
                "course_deg": course,
                "fix_type": None,
                "satellites": math.nan,
                "horizontal_accuracy_m": math.nan,
                "hdop": math.nan,
                "vdop": math.nan,
                "pdop": math.nan,
                "step_distance_m": math.nan,
                "time_step_s": math.nan,
                "extension_json": json.dumps(
                    _json_safe_row(source_row), sort_keys=True, default=str
                ),
            }
        )

    if not rows:
        raise CSVParseError(f"CSV file {path} contains no valid positioned rows.")

    points = _normalize_numeric_columns(pd.DataFrame(rows))
    points = _derive_kinematics(points, diagnostics)
    native_speed = points["device_speed_mps"].notna()
    points.loc[native_speed, "analysis_speed_source"] = "csv_native"
    points.loc[native_speed, "speed_certainty"] = "native_high"
    points = points.reindex(columns=CANONICAL_POINT_COLUMNS)

    segments = _summarize_segments(
        points,
        [{
            "run_id": metadata.run_id,
            "track_index": 0,
            "segment_index": 0,
            "start_row": 0,
            "point_count": len(points),
        }],
    )

    if invalid_time:
        diagnostics.warning(
            "CSV_TIMESTAMPS_INVALID",
            f"{invalid_time} positioned CSV row(s) have no usable timestamp.",
            source=str(path),
        )
    if naive_time:
        diagnostics.warning(
            "CSV_TIMEZONE_ASSUMED_UTC",
            f"{naive_time} timestamp(s) omitted a timezone and were interpreted as UTC.",
            source=str(path),
            hint=(
                "Track reconstruction uses elapsed timing; add an explicit timezone "
                "only if absolute race-clock provenance is required."
            ),
        )
    if missing_elevation:
        diagnostics.warning(
            "CSV_ELEVATION_INCOMPLETE",
            f"Elevation is missing for {missing_elevation} of {len(points)} valid point(s).",
            source=str(path),
            hint="Geometry and native speed evidence remain usable without elevation.",
        )

    summary = {
        "run_id": metadata.run_id,
        "vehicle_id": metadata.vehicle_id,
        "driver_id": metadata.driver_id,
        "source_file": str(path),
        "source_sha256": sha256,
        "source_format": "csv",
        "track_count": 1,
        "segment_count": 1,
        "csv_row_count": int(len(frame)),
        "valid_point_count": len(points),
        "invalid_coordinate_count": invalid_coordinates,
        "missing_timestamp_count": 0,
        "invalid_timestamp_count": invalid_time,
        "unusable_timestamp_count": int(points["timestamp_utc"].isna().sum()),
        "missing_elevation_count": int(points["elevation_m"].isna().sum()),
        "reported_speed_count": 0,
        "device_speed_count": int(points["device_speed_mps"].notna().sum()),
        "derived_speed_count": int(points["derived_speed_mps"].notna().sum()),
        "device_distance_count": int(points["device_distance_m"].notna().sum()),
        "total_path_distance_m": float(points["step_distance_m"].fillna(0.0).sum()),
        "native_speed_column": speed_col,
        "native_speed_unit_scale_to_mps": speed_scale,
    }
    return GPXIngestionResult(
        metadata=metadata,
        points=points,
        segments=segments,
        summary=summary,
        diagnostics=diagnostics.items,
    )


def _find_column(
    frame: pd.DataFrame, aliases: Iterable[str], *, required: bool = True
) -> str | None:
    lookup = {str(column).strip().lower(): str(column) for column in frame.columns}
    for alias in aliases:
        if alias.lower() in lookup:
            return lookup[alias.lower()]
    if required:
        raise CSVParseError(
            "CSV telemetry is missing a required column. Expected one of: "
            + ", ".join(aliases)
        )
    return None


def _find_speed_column(frame: pd.DataFrame) -> tuple[str, float]:
    lookup = {str(column).strip().lower(): str(column) for column in frame.columns}
    for alias, scale in _SPEED_ALIASES:
        if alias in lookup:
            return lookup[alias], scale
    raise CSVParseError(
        "CSV telemetry requires a native speed column in m/s, km/h, kph, or mph."
    )


def _json_safe_row(row: pd.Series) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in row.items():
        if pd.isna(value):
            output[str(key)] = None
        elif isinstance(value, np.generic):
            output[str(key)] = value.item()
        else:
            output[str(key)] = value
    return output
